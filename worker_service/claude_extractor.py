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

def encode_pdf_to_base64(file_path):
    """Encode PDF file to base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def is_real_value(val):
    if not val:
        return False
    cleaned = str(val).strip().lower()
    return cleaned not in {'n/a', 'none specified', 'not specified'}

def get_lease_column_names(supabase_client):
    try:
        response = supabase_client.rpc("get_lease_column_names").execute()
        if not response.data:
            raise Exception("RPC returned no data.")
        return {row['column_name'] for row in response.data}
    except Exception as e:
        raise Exception(f"Supabase RPC failed: {e}")

def make_api_call_with_retry(claude_client, claude_model, conversation_messages, verbose=False, max_retries=3):
    """
    Make API call with exponential backoff retry logic
    """
    base_delay = 60
    now = datetime.now()
    system_message = f"""You are a leasing document analyzer. Respond only with a JSON object containing all the requested fields. If information is not available, omit that field. Use exact keys from the prompt. Format all dates as yyyy/mm/dd. Use the current date {now} to determine relevant rent or cost values."""
    for attempt in range(max_retries):
        try:
            response = claude_client.messages.create(
                model=claude_model,
                max_tokens=1024,
                messages=conversation_messages,
                system=system_message
            )
            return response
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle different types of errors
            if "rate_limit" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    if verbose:
                        print(f"⚠️  Rate limit hit (attempt {attempt + 1}), waiting {delay}s...")
                    time.sleep(delay)
                else:
                    if verbose:
                        print(f"❌ Max retries exceeded for rate limiting")
                    raise
                    
            elif "maximum of 100 pdf pages" in error_str:
                if verbose:
                    print(f"❌ PDF page limit exceeded - this should not happen with current chunk size")
                raise Exception("PDF chunk size too large - exceeded 100 page limit")
                
            else:
                # For other errors, don't retry
                if verbose:
                    print(f"❌ Non-retryable error: {e}")
                raise


def merge_extraction_results(results_list):
    """
    Merge multiple JSON extraction results, appending values for duplicate keys
    """
    merged = {}
    
    for result in results_list:
        if not result:
            continue
            
        for key, value in result.items():
            if key in merged:
                # Key exists - append values
                existing_value = str(merged[key]).strip()
                new_value = str(value).strip()
                
                # Avoid duplicating identical values
                if existing_value.lower() != new_value.lower():
                    merged[key] = f"{existing_value}; {new_value}"
            else:
                # New key - add it
                merged[key] = value
    
    return merged

def claude_extraction(pdf, claude_client, supabase_client, claude_model, verbose=False):

    try:
        start_time = datetime.now()
        ALLOWED_KEYS = get_lease_column_names(supabase_client)

        reader = PdfReader(BytesIO(pdf))
        total_pages = len(reader.pages)
        chunk_size = 95
        all_results = []

        extraction_prompt = """Now that you have seen this lease document chunk, please extract the following information. Do calculations as necessary. Respond only with a valid JSON object using these keys. Omit fields that aren't found:

- lease_execution_date
- base_rent_monthly
- rent_escalation
- security_deposit_amount
- base_rent_psf
- base_rent_annually
- operating_expenses_CAM_psf
- operating_expenses_CAM_monthly
- property_taxes
- insurance_costs
- CAM_Summary
- tenant_reimbursements
- insurance_requirements
- lease_commencement_date
- lease_expiration_date
- delivery_possession_date
- CAM_start_date
- rent_abatement_end
- rent_commencement_date
- Property_Address
- suite_identifier
- lease_term
- renewal_notice_deadline
- option_exercise_deadlines
- renewal_options
- termination_rights
- expansion_contraction_rights
- co_tenancy_clauses
- purchase_option
- rentable_square_footage
- usable_square_footage
- premises_description
- parking_allocation
- storage_additional_space
- tenant_maintenance_responsibilities
- landlord_maintenance_responsibilities
- hvac_responsibilities
- utility_responsibilities
- default_and_remedies
- assignment_and_subletting
- indemnity_clauses
- force_majeure
- estoppel_certificate_required
- signage_rights
- permitted_use
- exclusive_use_clause
- guarantor_information
- tenant_improvement_allowance
- holdover_terms
- landlord_work
- Tenant_work
- security_deposit_term
- ROFR_ROFO_clauses
- security_access_rights
- exclusivity_rights

Please respond with a valid JSON object only."""

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
                    'role': 'user',
                    'content': [
                        { "type": "text", "text": extraction_prompt },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": encoded
                            }
                        }
                    ]
                }]

                if verbose:
                    print(f"📄 Sending pages {start+1}-{end} to Claude")

                response = make_api_call_with_retry(
                    claude_client, claude_model, messages, verbose
                )

                raw_text = response.content[0].text
                try:
                    parsed = json.loads(raw_text)
                    cleaned = {
                        key: value
                        for key, value in parsed.items()
                        if key in ALLOWED_KEYS and is_real_value(value)
                    }
                    all_results.append(cleaned)
                    if verbose:
                        print(f"✅ Parsed chunk {start//chunk_size + 1} successfully")
                except Exception as e:
                    print(f"❌ JSON parsing error in chunk {start+1}-{end}: {e}")
                    print(f"Raw response: {raw_text[:300]}...")
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
            print(f"⏱️ Duration: {duration:.2f} seconds")

        return final_result, 0, total_pages  # cost not calculated here

    except Exception as e:
        if verbose:
            print(f"❌ Error during extraction: {str(e)}")
        return None, 0, 0
