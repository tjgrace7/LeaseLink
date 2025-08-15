from datetime import datetime
import base64
import tiktoken
from anthropic import Anthropic
import os
import json
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
import tempfile
import time
import re

def encode_pdf_to_base64(file_path):
    """Encode PDF file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# --- JSON extraction helpers ---
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.DOTALL)

def _first_balanced_json_object(s: str) -> str | None:
    """Return the first balanced {...} substring, ignoring braces inside strings."""
    start = s.find("{")
    if start == -1:
        return None

    in_str = False
    esc = False
    depth = 0
    i = start
    while i < len(s):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        i += 1
    return None  # no balanced object found

def extract_json_from_text(text: str) -> dict:
    cleaned = JSON_FENCE_RE.sub("", text.strip())
    # First try: entire string is JSON
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Second try: find the first balanced {...} block
    candidate = _first_balanced_json_object(cleaned)
    if candidate:
        return json.loads(candidate)

    # Last resort: maybe caller passed raw JSON after trimming fences incorrectly
    return json.loads(text)

# --- Value filters ---
def is_real_value(val):
    if val is None:
        return False
    if isinstance(val, (int, float)) and val == 0:
        return True
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in {'', 'n/a', 'none specified', 'not specified'}:
            return False
        return True
    if isinstance(val, (list, dict)):
        return len(val) > 0
    return True

# --- Date normalization ---
STRICT_DATE_KEYS = {
    'lease_execution_date',
    'lease_commencement_date',
    'lease_expiration_date',
    'delivery_possession_date',
    'cam_start_date',
    'rent_abatement_end',
    'rent_commencement_date',
}
DATE_YMD_SLASH = re.compile(r"^\d{4}/\d{2}/\d{2}$")

def try_parse_to_yyyy_mm_dd(s: str) -> str | None:
    s = s.strip()
    if DATE_YMD_SLASH.match(s):
        return s
    s2 = re.sub(r"[-\.]", "/", s)
    candidates = [
        "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d",
        "%m-%d-%Y", "%m-%d-%y", "%B %d, %Y", "%b %d, %Y",
        "%d %B %Y", "%d %b %Y"
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(s2, fmt)
            return f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}"
        except Exception:
            pass
    return None

def normalize_strict_dates_in_obj(obj: dict) -> dict:
    out = {}
    for k, v in obj.items():
        if k.lower() in STRICT_DATE_KEYS:
            if isinstance(v, str):
                norm = try_parse_to_yyyy_mm_dd(v)
                if norm is None:
                    continue
                out[k] = norm
            else:
                continue
        else:
            out[k] = v
    return out

def merge_extraction_results(results_list):
    merged = {}
    for result in results_list:
        if not result:
            continue
        result = normalize_strict_dates_in_obj(result)
        for key, value in result.items():
            if key in merged:
                existing_value = merged[key]
                if isinstance(existing_value, list) and isinstance(value, list):
                    merged[key] = existing_value + [v for v in value if v not in existing_value]
                elif isinstance(existing_value, dict) and isinstance(value, dict):
                    merged[key] = {**value, **existing_value,
                                   **{k: v for k, v in value.items() if k not in existing_value}}
                else:
                    old_s = str(existing_value).strip()
                    new_s = str(value).strip()
                    if old_s.lower() != new_s.lower():
                        merged[key] = f"{old_s}; {new_s}"
            else:
                merged[key] = value
    return merged

def get_lease_column_names(supabase_client):
    try:
        response = supabase_client.rpc("get_lease_column_names").execute()
        if not response.data:
            raise Exception("RPC returned no data.")
        return {row['column_name'] for row in response.data}
    except Exception as e:
        raise Exception(f"Supabase RPC failed: {e}")

def make_api_call_with_retry(claude_client, claude_model, conversation_messages, verbose=False, max_retries=3):
    base_delay = 60
    now = datetime.now()
    system_message = f"""You are a leasing document analyzer. Respond only with a JSON object containing all the requested fields. If information is not available, omit that field. Use exact keys from the prompt. Format all dates as yyyy/mm/dd if the key name contains the word date and it’s a single date (not a range). Do not add anything that is not yyyy/mm/dd. Use the current date {now} to determine relevant rent or cost values."""
    for attempt in range(max_retries):
        try:
            return claude_client.messages.create(
                model=claude_model,
                max_tokens=8000,
                messages=conversation_messages,
                system=system_message
            )
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    if verbose:
                        print(f"⚠️ Rate limit hit (attempt {attempt + 1}), waiting {delay}s...")
                    time.sleep(delay)
                else:
                    raise
            elif "maximum of 100 pdf pages" in error_str:
                raise Exception("PDF chunk size too large - exceeded 100 page limit")
            else:
                raise

def claude_extraction(pdf, claude_client, supabase_client, claude_model, verbose=False):
    try:
        start_time = datetime.now()
        ALLOWED_KEYS = get_lease_column_names(supabase_client)

        reader = PdfReader(BytesIO(pdf))
        total_pages = len(reader.pages)
        chunk_size = 95
        all_results = []
        total_cost_sum = 0.0

        extraction_prompt = getExtractionPrompt()

        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            writer = PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            writer.write(temp_file)
            temp_file.close()
            temp_path = temp_file.name

            try:
                encoded = encode_pdf_to_base64(temp_path)
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": encoded
                            }
                        },
                        { "type": "text", "text": extraction_prompt }
                    ]
                }]

                if verbose:
                    print(f"📄 Sending pages {start+1}-{end} to Claude")

                response = make_api_call_with_retry(
                    claude_client, claude_model, messages, verbose
                )

                token_count = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                if token_count > 200000:
                    input_cost = 0.000006
                    output_cost = 0.0000225
                else:
                    input_cost = 0.000003
                    output_cost = 0.000015
                total_cost_sum += token_count * input_cost + output_tokens * output_cost

                text_blocks = [block.text for block in response.content if block.type == "text"]
                raw_text = "\n".join(text_blocks).strip()

                try:
                    parsed = extract_json_from_text(raw_text)
                    cleaned = {
                        key: parsed[key]
                        for key in parsed.keys()
                        if key in ALLOWED_KEYS and is_real_value(parsed[key])
                    }
                    all_results.append(cleaned)
                    if verbose:
                        print(f"✅ Parsed chunk {start//chunk_size + 1} successfully")
                except Exception as e:
                    print(f"❌ JSON parsing error in chunk {start+1}-{end}: {e}")
                    print(f"Raw response (start): {raw_text[:300]}...")
                    continue
            finally:
                os.remove(temp_path)

            if end < total_pages:
                time.sleep(15)

        final_result = merge_extraction_results(all_results)
        if verbose:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"✅ All chunks processed successfully")
            print(f"📄 Total pages: {total_pages}")
            print(f"🧩 Final extracted fields: {len(final_result)}")
            print(f"💵 Estimated cost: ${total_cost_sum:,.4f}")
            print(f"⏱️ Duration: {duration:.2f} seconds")

        return final_result, total_cost_sum, total_pages

    except Exception as e:
        if verbose:
            print(f"❌ Error during extraction: {str(e)}")
        return None, 0, 0


def getExtractionPrompt():
    extraction_prompt = """Now that you have seen this lease document chunk, please extract the following information. Do calculations as necessary. Respond only with a valid JSON object using these keys. Omit fields that aren't found:

-lease_execution_date (The Day the document is signed (Often handwritten))
-base_rent_monthly (Price of rent per month for the building before expenses. )
-rent_escalation (Give details on the rent schedule during the initial lease. List by date and amount.) 
-security_deposit_amount (The amount the rent has to put as a "down payment" to hold there space. Is paid back at the end of the lease if the property is left in good condition.) 
-base_rent_psf (The Per Square Foot price for Base Rent (Calculate: Annual Rent / SF)) 
-base_rent_annually (Calculate: Base rent amount paid across 12 months) 
-operating_expenses_CAM_psf (CAM + operating expenses per square foot. That includes taxes and Insurance (Calculate: monthly/ SF)
-operating_expenses_CAM_monthly (Monthly estimated amount that tenants are pay in all expenses they are responsible for via the lease. That includes taxes and insurance monthly)  
-property_taxes (who pays taxes) 
-CAM_Summary (Make note of any operating expenses not allowed to be charged back to the tenant.) 
-tenant_reimbursements (A summary of the system in which the landlord is able to bill the tenant for expenses they initially paid for or the rights in which the tenants have to recoup the money in which they overpaid for building expenses.) 
-insurance_requirements (Insurance requirements for the renters of the space. (General liability or Property Insurance) 
-property_insurance (Who carries the Property Insurance)
-lease_commencement_date (The Day the Lease takes effect, and terms start. yyyy/mm/dd force into format) 
-lease_expiration_date (Day that the lease ends before any options to renew. Use the formula lease commencement date + term if required. Use yyyy/mm/dd force into format) 
-delivery_possession_date (The day the tenants may access the space, yyyy/mm/dd) 
-CAM_start_date (The date in which the tenant is responsible for paying estimated CAM amounts, yyyy/mm/dd force into format) 
-rent_abatement_end (the date where the tenants rent abatement runs out. Format yyyy/mm/dd) 
-rent_commencement_date (Rent Commencement Date. The Day Rent Starts (Or commences) yyyy/mm/dd force into format)
-Property_Address (The listed address of the property) 
-suite_identifier (The number or letter of the suite without the address if applicable) 
-lease_term (Length of lease term in months)
-option_exercise_deadlines (When does the tenant notify the landlord of their intent to renew) 
-renewal_options (The amount of options the tenant has and the terms that change upon the commencement of these options.)
-termination_rights (Any terms that allow either party to terminate the lease early.
-expansion_contraction_rights (The provisions that allow the tenant to grow into more space or shrink out of other space.) 
-co_tenancy_clauses (Obligations that must be met by the landlord in accordance to other tenants and if not met the consequences.) 
-purchase_option (Options the tenant has to purchase the building in within the terms of the lease.) 
-rentable_square_footage (Useable SF + share of common areas (hallways, restrooms, etc.)) 
-usable_square_footage (The amount of square footage in the lease) 
-premises_description (Gives a more general and knowledgeable description of the rentable area) 
-parking_allocation (How much parking the tenant gets.) 
-storage_additional_space (If any storage is allotted or additional space is allotted to the tenant) 
-tenant_maintenance_responsibilities (What is the Lessee's/Tenant responsibility to maintain and repair the unit) 
-landlord_maintenance_responsibilities (What is the Landlord's/Property Managers responsibility to maintain and repair the building/unit) 
-hvac_responsibilities (The HVAC responsibilities in detail) 
-utility_responsibilities (Utility Responsibility in detail) 
-default_and_remedies (The actions and ability to take actions of either part in the event of default by the other.) 
-assignment_and_subletting (Is subletting allowed in the space? If so, under what terms and conditions?) 
-indemnity_clauses (The landlords protection from being held legally liable for anything. (Tenant can't sue landlord)) 
-force_majeure (excuses one or both parties from performing their obligations when extraordinary events occur that are outside their control.) 
-estoppel_certificate_required (The requirement that tenants answer certain questions in certain occasions. Normally when selling or refinancing.) 
-signage_rights (What signage rights the tenant has) 
-permitted_use (What type of business is permitted to use the unit?) 
-exclusive_use_clause (Gives the tenant permission to be the sole operator allowed to do something.) 
-guarantor_information (the details about any person or entity that guarantees the tenant's obligations under the lease.) 
-tenant_improvement_allowance (The amount the Landlord gives to the tenant to improve the property for the tenants use.) 
-holdover_terms (Terms that apply when the tenant overstays their lease without a renewal.) 
-landlord_work (Work that is the responsibility of the landlord, normally before the tenant moves in.) 
-Tenant_work (The work or improvements that the tenant is held responsible if applicable) 
-security_deposit_term (The terms that define security deposit rules.) 
-ROFR_ROFO_clauses (Right of First Refusal clauses or Right of First Offer clauses) 
-security_access_rights (The rights of security and the limits to the landlords access.) 
-exclusivity_rights (Blocks the landlord from allowing any competing business.)

Please respond with a valid JSON object only."""
    return extraction_prompt