
import os
from openai import OpenAI
from anthropic import Anthropic
from qdrant_client import QdrantClient
from qdrant_client.http.models import  Filter, FieldCondition, MatchValue, MatchAny
import common.Supabase_api as Supabase_api
import json
import re
import ast
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Optional, Union
import unicodedata
import tiktoken
import uuid
from worker_service import extraction_prompts
import traceback

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPEN_AI_PROJECT_KEY")
CLAUDE_API_KEY = os.environ.get("Claude_API_KEY")

CRITICAL_FIELDS = [
    'base_rent_amount_current',
    'lease_expiration_date', 
    'lease_signed_date',
    'lease_commencement_date',
    'lease_term_months'
]

# Clients (parent process only)
OpenAIClient = OpenAI(api_key=OPENAI_API_KEY)
claude= Anthropic(api_key=CLAUDE_API_KEY)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase = Supabase_api.supabase_client_setup()

import re

MONEY_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+\.\d{2}\b")
DATEISH_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", re.I)

RENT_KWS = [
    "base rent", "minimum rent", "fixed rent", "monthly installments",
    "rent schedule", "annual rent", "per annum", "per year", "per month"
]
PERIOD_KWS = ["commencing", "beginning", "from and after", "through", "during", "for the period", "year", "month"]

def rent_chunk_score(text: str) -> int:
    t = text.lower()
    score = 0
    if any(k in t for k in ["base rent", "minimum rent", "fixed rent"]): score += 3
    if any(k in t for k in ["rent schedule", "exhibit", "schedule", "table"]): score += 2
    if MONEY_RE.search(text): score += 2
    if any(k in t for k in PERIOD_KWS) or DATEISH_RE.search(text): score += 2
    if any(k in t for k in ["amendment", "modification", "restated", "supersedes", "notwithstanding"]): score += 1
    return score

def pass_a_is_valid(chunks):
    scores = [rent_chunk_score(c.payload['text']) for c in chunks]
    return (max(scores, default=0) >= 6) or (sum(sorted(scores, reverse=True)[:3]) >= 14)



def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def parse_model_payload(text: str) -> Optional[List[Dict]]:
    """
    Handles:
      1) Strict JSON list: [{"needs_correction": true, ...}]
      2) Strict JSON object: {"needs_correction": true, ...}
      3) Python-ish list: [{'needs_correction': True, ...}]
      4) Python-ish object: {'needs_correction': True, ...}
    Returns: list[dict] or None
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # 1) If there's a fenced block, prefer its content (common with Claude)
    fence_match = re.search(r"```(?:json|python)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        # Otherwise, remove any stray fence markers
        cleaned = re.sub(r"```(?:json|python)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

    # 2) Remove optional leading label like "json:"
    cleaned = re.sub(r"^\s*json\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # 3) Extract the first top-level JSON-ish block: either [...] or {...}
    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
    candidate = m.group(1).strip() if m else cleaned

    def normalize_return(obj: Union[list, dict]) -> Optional[List[Dict]]:
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            return obj
        return None

    # 4) Try strict JSON
    try:
        return normalize_return(json.loads(candidate))
    except Exception:
        pass

    # 5) Fallback: Python literal (single quotes, True/False)
    try:
        return normalize_return(ast.literal_eval(candidate))
    except Exception:
        return None

def lease_check(tenant_id, unit_id,collection_name, leases, default = 'Present'):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_embedding_tokens = 0
    sorted_leases = []

    for lease in leases:
        lease_id = lease['lease_id']
        lease_description = "lease term start date, commencement date, effective date, expiration date, end date, term period"
        lease_results, embedding_tokens = query_description(tenant_id, unit_id, lease_description, collection_name, 10, lease['lease_file_path'])
        response, prompt_tokens, completion_tokens = active_lease(lease_results, lease_id, default)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_embedding_tokens += embedding_tokens
        parsed = parse_model_payload(response)
        print("Parsed Response", parsed)
        if parsed and len(parsed) > 0:
            for item in parsed:

                effective = item.get('effective_date')
                status = item.get('status')
                document_type = item.get('document_type')

                try:
                    if effective and effective not in [None, "", 'null']:
                        effective_dt = datetime.strptime(effective, '%Y-%m-%d')
                    else:
                        effective_dt = datetime(1900,1,1)
                except (ValueError, TypeError):
                    effective_dt = datetime(1900,1,1)
                
                sorted_leases.append({ 
                    'lease': lease,
                    'effective_date': effective,
                    'effective_dt': effective_dt,
                    'status': status,
                    'document_type': document_type
                })

                
    sorted_leases = sorted(
        sorted_leases,
        key = lambda x: x['effective_dt'], 
        reverse=True
    )


    return sorted_leases, total_prompt_tokens, total_completion_tokens, total_embedding_tokens
def active_lease(lease_results, lease_id, default):
    now = datetime.now()
    lease_context = "\n\n".join([
        f"source_doc = {r.payload.get('source_doc', 'unknown')}, pageNumber = {r.payload.get('pageNumber', 'N/A')})\n, text = {normalize_text(r.payload['text'])}"
        for r in lease_results if "text" in r.payload
    ])


    system_prompt = f"""
You extract key dates from a SINGLE lease document to determine its temporal status.

TODAY'S DATE: {now} (America/Chicago)

IMPORTANT: You are analyzing ONE lease document at a time. Do not worry about comparing to other leases.

DATE EXTRACTION RULES:

1. effective_date (when THIS lease's term starts):
   - Search for: "Commencement Date", "Effective Date", "Start Date", "Term Commences"
   - Priority: Commencement Date > Effective Date > Start Date
   - If the document is an amendment, extract the ORIGINAL effective date if mentioned, OR the amendment's new effective date if it restates the term
   - Make your best estimate from context -  Only return None if there is absolutely no way to estimate effective_date. 
   - Most of the time there should be an effectitve_date. It should be rare to have None
   - Format: yyyy-mm-dd


2. expiration_date (when THIS lease's term ends):
   - Search for: "Expiration Date", "Term End Date", "Lease Expires", "End Date"
   - If only term length given: calculate effective_date + term_length
   - If this is an amendment that extends the term, use the NEW expiration date
   - Format: yyyy-mm-dd
   - Only Return None if there is absolutely no way to estimate expiration_date

3. term_length_months:
   - Extract term duration in months (convert years to months if needed)
   - Return null if cannot determine

4. status (MUST BE CONSISTENT WITH THE DATES YOU OUTPUT):
   - Define TODAY_ISO = "{now.strftime('%Y-%m-%d')}" (yyyy-mm-dd)
   - If expiration_date is not null AND expiration_date < TODAY_ISO → status MUST be "Past"
   - Else if effective_date is not null AND effective_date > TODAY_ISO → status MUST be "Future"
   - Else if effective_date is not null AND expiration_date is not null AND effective_date <= TODAY_ISO <= expiration_date → status MUST be "Present"
   - Else → status MUST be "{default}"

   VALIDATION (REQUIRED):
   - After choosing status, verify it matches the above rules.
   - If it does not match, you MUST change status to the correct value.
   - NEVER output "Present" when expiration_date < TODAY_ISO.

5. document_type (what kind of document is this):
   - "Original" = base lease agreement
   - "Amendment" = modifies an existing lease
   - "Renewal" = extends/renews an existing lease
   - Look for keywords: "Amendment", "First Amendment", "Renewal", "Extension", "Addendum"

EXAMPLES:
- Original lease: Effective 2021-08-01, Term 5 years → Expiration 2026-07-31, Status "Present", Type "Original"
- Amendment: "Lease term extended to 2027-12-31" → Expiration 2027-12-31, Status "Present", Type "Amendment"
- Old lease: Effective 2018-01-01, Term 3 years → Expiration 2021-01-01, Status "Past", Type "Original"

Lease ID: {lease_id}

CONTEXT:
{lease_context}

Return STRICT JSON ONLY (no markdown, no extra text):
{{
  "lease_id": "{lease_id}",
  "effective_date": "yyyy-mm-dd" or null,
  "expiration_date": "yyyy-mm-dd" or null,
  "term_length_months": integer or null,
  "status": "Past" or "Present" or "Future or if unclear Default to {default}",
  "document_type": "Original" or "Amendment" or "Renewal" or "Unknown"
}}
"""

    chat_response = OpenAIClient.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "Determine if the given lease document has an active term"
            }
        ],
        max_tokens=300,
        temperature=0,   # IMPORTANT for boolean/date logic
    )

    response = chat_response.choices[0].message.content
    prompt_tokens = chat_response.usage.prompt_tokens
    completion_tokens = chat_response.usage.completion_tokens
    return response, prompt_tokens, completion_tokens

def extract_tenant_data(tenant_id: str, unit_id: str, company_id: str, claude_model="claude-sonnet-4-20250514", collection_name = 'Lease_Link', time_update = False) -> float:
    """
    Run final check on lease data for a given tenant_id.
    Returns the cost of the final check operation.
    """
    # Fetch lease data from Supabase
    extraction_id = str(uuid.uuid4())
    try:
        
        tenant_response = supabase.table('Lease_Extractions').select('*').eq('tenant_id', tenant_id).eq('unit_id', unit_id).eq('Is_Current', True).execute()
        is_created = False
        rows = tenant_response.data or []
        if len(rows) == 0:
            insert_payload = {
                'tenant_id': tenant_id,
                'unit_id': unit_id,
                'id': extraction_id,
                'Is_Current': True
            }
            created = supabase.table("Lease_Extractions").insert(insert_payload).execute()
            tenant = created.data[0]
            is_created = True
        elif len(rows) ==1:
            tenant = rows[0]
        else:
            raise RuntimeError("Multiple Current Lease_Extraction Rows Found")
        total_prompt_tokens_claude = 0
        total_completion_tokens_claude = 0
        total_prompt_tokens_GPT = 0
        total_completion_tokens_GPT = 0
        total_embedding_tokens = 0
        all_columns = {}
        lease_response = supabase.table('lease_documents').select('*').eq('tenant_id', tenant_id).eq('unit_id', unit_id).execute()
        leases = lease_response.data
        



        sorted_leases, p, c, e = lease_check(tenant_id, unit_id, collection_name, leases)
        total_prompt_tokens_GPT += p
        total_completion_tokens_GPT += c
        total_embedding_tokens += e
        
        GPT_prompt_cost =total_prompt_tokens_GPT /1000 * .0020
        GPT_completion_cost = total_completion_tokens_GPT / 1000 * .0080
        lease_files = sorted_leases
        future_leases = []
        current_leases = []
        past_leases = []
        future_effective_date = None
        future_context = ""
        current_context = ""
        past_context = ""
        lease_commencement_date = None
        if tenant['lease_commencement_date'] != None:
            if tenant['lease_commencement_date']['is_manual_change']:
                lease_commencement_date = tenant['lease_commencement_date']['value']

        if not tenant:
            print(f"No tenant found with tenant_id: {tenant_id}")
            return 0.0
        for column in tenant:
            resp = supabase.rpc('get_column_description', {'p_table': 'Lease_Extractions', 'p_column': column}).execute()
            desc = resp.data

            future_top_k = 10
            past_top_k = 10
            present_top_k = 10
            if column in ['unit_id', 'Is_Current', 'tenant_id', 'created_at', 'Tenant_Name', 'DBA', 'property_management_id', 'photo_file_path', 'Active', 'Available', 'Modified_By', 'archived', 'cost_per_upload', 'id', 'company_id']:
                continue
            for lease_row in lease_files:
                status = lease_row['status']
                if status == "Future":
                    future_leases.append(lease_row)
                if status== 'Present':
                    current_leases.append(lease_row)
                if status == "Past":
                    past_leases.append(lease_row)

            required_status = extraction_prompts.prompts[column]['required_document(s)']
            original_context = ""
            if required_status == 'Current_Lease':
                future_top_k = 5
                past_top_k = 5
                present_top_k = 15
            if required_status == 'Original_Lease':
                future_top_k = 5
                past_top_k = 15
                present_top_k = 5
                lease_file_paths = []
                for lease in past_leases:

                    if lease['document_type'] == 'Original':
                        lease_file_paths.append(lease['lease']['lease_file_path'])
                original_context = context_get(query_description(tenant_id, unit_id, desc, collection_name, past_top_k, lease_file_paths))

            if len(future_leases) > 0:
                lease_file_paths = []
                for lease in future_leases:
                    lease_file_paths.append(lease['lease']['lease_file_path'])
                    if len(future_leases) ==1:
                        future_effective_date = lease['effective_date']
                future_context = context_get(query_description(tenant_id, unit_id, desc, collection_name, future_top_k, lease_file_paths))
            if len(current_leases) > 0:
                lease_file_paths = []
                for lease in current_leases:
                    lease_file_paths.append(lease['lease']['lease_file_path'])
                query = query_description(tenant_id, unit_id, desc, collection_name, present_top_k, lease_file_paths)
                result = context_get(query)
                current_context = result
            if len(past_leases) > 0 and required_status != 'Original_Lease':
                lease_file_paths = []
                for lease in past_leases:
                    lease_file_paths.append(lease['lease']['lease_file_path'])
                past_context = context_get(query_description(tenant_id, unit_id, desc, collection_name, past_top_k, lease_file_paths))
                
            prompt_tokens, completion_tokens, value = review_extraction_clases(column,  tenant, future_context, future_effective_date, current_context, past_context, original_context, claude_model, lease_commencement_date, time_update)
            if column == 'lease_commencement_date' and lease_commencement_date == None:
                try:
                    data = supabase.table('Lease_Extractions').select('lease_commencement_date').eq('id', extraction_id).single().execute()
                    print("Data", data)
                    date_json= data.data.get('lease_commencement_date', {})
                    if date_json not in [None, "null", ""]:
                        lease_commencement_date = date_json.get('value', None)
                except Exception as e:
                    print("Lease Commencement:", lease_commencement_date)
                    print("Error fetching lease commencement date", e)
            total_prompt_tokens_claude += prompt_tokens
            total_completion_tokens_claude += completion_tokens
            all_columns[column] = value
        
        prompt_cost = (total_prompt_tokens_claude / 1000 * 0.003)
        embedding_cost = total_embedding_tokens / 1000 * 0.00013
        completion_cost = total_completion_tokens_claude / 1000 * 0.015
        tenant_cost = completion_cost + prompt_cost + embedding_cost + GPT_prompt_cost + GPT_completion_cost
        supabase.table('Lease_Extractions').upsert({
            'id': extraction_id,
            'cost_per_upload': tenant_cost,
            'unit_id': unit_id,
            'tenant_id': tenant_id,
            'company_id': company_id,
            'Is_Current': True,
            'lease_commencement_date': all_columns['lease_commencement_date'],
            'lease_signed_date': all_columns['lease_signed_date'],
            'latest_lease_modification_signed_date': all_columns['latest_lease_modification_signed_date'],
            'base_rent_amount_current': all_columns['base_rent_amount_current'],
            'base_rent_frequency': all_columns['base_rent_frequency'],
            'base_rent_payment_timing': all_columns['base_rent_payment_timing'],
            'base_rent_due_day': all_columns['base_rent_due_day'],
            'base_rent_effective_date': all_columns['base_rent_effective_date'],
            'base_rent_schedule': all_columns['base_rent_schedule'],
            'security_type': all_columns['security_type'],
            'security_deposit_amount': all_columns['security_deposit_amount'],
            'additional_rent_components': all_columns['additional_rent_components'],
            'additional_rent_billing_method': all_columns['additional_rent_billing_method'],
            'additional_rent_commencement_date': all_columns['additional_rent_commencement_date'],
            'additional_rent_limitations': all_columns['additional_rent_limitations'],
            'possession_date': all_columns['possession_date'],
            'rent_commencement_date': all_columns['rent_commencement_date'],
            'rent_abatement_end_date': all_columns['rent_abatement_end_date'],
            'lease_expiration_date': all_columns['lease_expiration_date'],
            'lease_term_months': all_columns['lease_term_months'],
            'rights_index': all_columns['rights_index'],
            'renewal_options_summary': all_columns['renewal_options_summary'],
            'renewal_notice_requirements_summary': all_columns['renewal_notice_requirements_summary'],
            'premises_description': all_columns['premises_description'],
            'parking_allocation': all_columns['parking_allocation'],
            'tenant_maintenance_responsibilities': all_columns['tenant_maintenance_responsibilities'],
            'landlord_maintenance_responsibilities': all_columns['landlord_maintenance_responsibilities'],
            'hvac_responsibilities': all_columns['hvac_responsibilities'],
            'utility_responsibilities': all_columns['utility_responsibilities'],
            'permitted_use': all_columns['permitted_use']
            },
            on_conflict='id').execute()
        print(f"Final Cost - ${tenant_cost} (Prompt: ${prompt_cost + GPT_prompt_cost}, Completion: ${completion_cost + GPT_completion_cost}, Embedding: ${embedding_cost})")
        #Make Old Extraction Column Inactive
        if not is_created:
            supabase.table('Lease_Extractions').update({'Is_Current': False}).eq('id', tenant.get('id')).execute()
    except Exception as e:
        traceback.print_exc()
        print("Error Extracting Lease", e)
        supabase.table('Lease_Extractions').update({'Is_Current': False}).eq('id', extraction_id).execute()


def query_description(tenant_id, unit_id, description, collection_name, top_k, leases):
    
    message_vector = OpenAIClient.embeddings.create(
            input="Use most recent document, Amendment or Renewal if available. Use primary lease when no info available for description in amendment" + description,
            model="text-embedding-3-large"
     ).data[0].embedding
    if isinstance(leases, str):
        leases = [leases]
    resp = qdrant.query_points(
        collection_name=collection_name,
        query=message_vector,
        using='dense_vector',
        limit=top_k,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenantid",
                    match=MatchValue(value=tenant_id)),
                FieldCondition(
                    key="unitid",
                    match=MatchValue(value=unit_id)),
                FieldCondition(
                    key='source_doc',
                    match=MatchAny(any=leases)
                )
            ]
        ),
    )
    points = resp.points
    results = points
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    embedding_token_count = len(encoding.encode(description))
    print("Pre Screen Count of Results", len(results))
    if points:
        best = points[0].score
        cutoff = best * .6
        results = [p for p in points if p.score >= cutoff]
    print("Post Screen Count of Results", len(results))
    return results, embedding_token_count
def qdrant_points_to_json(points):
    return [
        {
            "id": p.id,
            "score": p.score,
            "payload": p.payload,   # already JSON-serializable
        }
        for p in points
    ]
def context_get(results):
    # If results is (points, something), pull out points
    if isinstance(results, tuple) and len(results) >= 1:
        results = results[0]

    # If someone passed a single point, wrap it
    if results is None:
        return ""
    if not isinstance(results, (list, tuple)):
        results = [results]

    context = "\n\n".join(
        f"source_doc = {r.payload.get('source_doc','unknown')}, "
        f"pageNumber = {r.payload.get('pageNumber','N/A')}, "
        f"score = {getattr(r, 'score', 'N/A')}\n"
        f"{r.payload.get('text','')}"
        for r in results
        if hasattr(r, "payload")
        and isinstance(r.payload, dict)
        and r.payload.get("text")
    )
    return context

def review_extraction_clases(column, tenant, future_context, future_effective_date, current_context, past_context, original_context, claude_model, lease_commencement_date, time_update):
    """
    Review extraction for a specific column.
    """
    print("Column", column)
    now = datetime.now()
    prompts = extraction_prompts.prompts
    unique_prompt = prompts[column]['prompt']
    required_document = prompts[column]['required_document(s)']
    minimum_required_confidence = prompts.get(column, {}).get('minimum_required_confidence', .8)
    cell = tenant.get(column) or {}
    lease_commence = tenant.get('lease_commencement_date') or {}
    is_manual_lease_commencement = False
    if lease_commence != {}:
        is_manual_lease_commencement = lease_commence['is_manual_change'] 


    current_value = cell.get('value')
    is_manual_change = cell.get('is_manual_change')

    system_prompt = f"""You extract specific lease terms from documents with high accuracy.

═══════════════════════════════════════════════════════════════════
REFERENCE DATE
═══════════════════════════════════════════════════════════════════
TODAY'S DATE: {now.strftime('%Y-%m-%d')} ({now.strftime('%B %d, %Y')})
Use this to determine which lease periods are active, past, or future.

═══════════════════════════════════════════════════════════════════
FIELD TO EXTRACT
═══════════════════════════════════════════════════════════════════
Column: {column}



{unique_prompt}



═══════════════════════════════════════════════════════════════════
DOCUMENT CONTEXT
═══════════════════════════════════════════════════════════════════

CURRENT/ACTIVE LEASE CONTEXT:
{current_context if current_context else "[No current lease context]"}

PAST/EXPIRED LEASE CONTEXT:
{past_context if past_context else "[No past lease context]"}

ORIGINAL LEASE CONTEXT:
{original_context if original_context else "[No original lease context - only use when specifically required]"}

FUTURE LEASE CONTEXT:
{future_context if future_context else "[No future lease context]"}
{f"Future Effective Date: {future_effective_date}" if future_effective_date else ""}

═══════════════════════════════════════════════════════════════════
EXTRACTION INSTRUCTIONS
═══════════════════════════════════════════════════════════════════

1. PRIMARY SOURCE PRIORITY:
   {f"⚠️ REQUIRED: Extract primarily from {required_document}" if required_document else "Use current context first, then past context if needed"}

2. CONTEXT USAGE RULES:
   - CURRENT context: Use for active lease terms (primary source)
   - PAST context: Use only when current lacks information OR to support current OR if the extraction primarily requests ANY Document to be extracted
   - ORIGINAL context: Use ONLY if unique_prompt explicitly requires it
   - FUTURE context: Use ONLY for future_value field, NEVER for value field

3. DATE-BASED EXTRACTION (for rent schedules, escalations, etc.):
   When you see a schedule with multiple time periods:
   
   Step 1: Identify which period contains TODAY ({now.strftime('%Y-%m-%d')})
   Step 2: Extract that period's value as "value"
   Step 3: Extract all periods AFTER today as "future_value"
   Step 4: If a Date isn't given use this as original lease commencement date to calculate dates: {lease_commencement_date}
        If there is an amendment date or renewal date, those trump commencement date
   
   Example table:
   "10/1/2025 - 9/30/2026 | $1,831.25
    10/1/2026 - 9/30/2027 | $1,892.29
    10/1/2027 - 12/31/2027 | $1,953.33"
   
   If today is {now.strftime('%Y-%m-%d')}:
   - Find which row's date range contains today
   - That row's amount = "value"
   - All subsequent rows = "future_value"
   
   ⚠️ DO NOT extract the last/highest/most recent value blindly
   ⚠️ Position in table is irrelevant - only dates matter

4. NEEDS_CORRECTION LOGIC:
   needs_correction = true IF:
   - No current database value exists, OR
   - Current database value differs from extracted value in meaning
   - Here is the current database value: {current_value}
   
   needs_correction = false IF:
   - Extracted value matches current database value (after normalizing whitespace)

5. CONFIDENCE SCORING (BE CONSERVATIVE):
   Your scores historically run 15% too high. When uncertain, choose lower score.
   
   HIGH (0.85-1.0):
   - Explicitly labeled field with clear language
   - Multiple confirming references
   - High search relevance score (>0.7) AND unambiguous text
   
   MEDIUM (0.65-0.84):
   - Clear but requires some inference
   - Single clear reference
   - Moderate search relevance (0.5-0.7)
   
   LOW (0.0-0.64):
   - Ambiguous or conflicting information
   - Low search relevance (<0.5)
   - Significant inference required
   - Expected field is missing (e.g., lease_signed_date should always exist)
6. MANUAL REVIEW:
   - If the confidence score is < {minimum_required_confidence}. We need to trigger a manual review of the term
   - In the JSON output mark manual_review as true is it meets the above criteria

6. FIELD-SPECIFIC RULES:
   - Dates: Always format as YYYY-MM-DD
   - Currency: Preserve dollar signs and decimals (e.g., "$1,831.25")
   - Null values: Use string "Null" (not JSON null)
   - Expired: Set true ONLY if the lease term containing this value has ended

7. REASONING REQUIREMENT:
   In the "reason" field, explain:
   - Where you found the value (which context, which page)
   - Why you chose this value over alternatives
   - What affected your confidence score
   - If time-based: which period you identified as current
8. Manual Adjustment: This column was manually adjusted: {is_manual_change} (If True do the following, if false ignore)
    - If it was manually changed, only change if confidence is very high that change is wrong. Greater than .95
    - Lease Commencement Date, Do not change if manually adjusted
9. LEASE COMMENCEMENT DATE CALCULATION:
   Lease commencement was manually adjusted: {is_manual_lease_commencement}
   {f"Manually set commencement date: {lease_commencement_date}" if is_manual_lease_commencement and lease_commencement_date else ""}
   
   {"⚠️ CALCULATE FROM COMMENCEMENT: If the rent schedule uses relative periods (e.g., 'Month 1-12', 'Year 1-2') instead of specific dates, calculate actual dates using the lease commencement date above." if is_manual_lease_commencement and lease_commencement_date else ""}
   {"base_rent_effective_date, and rent_commencement_date may be calculated if explicitely stated" if is_manual_lease_commencement and lease_commencement_date else ""}
   

   Example with manual commencement date of 2024-03-15:
   - Document says: "Month 1-12: $2,000; Month 13-24: $2,200"
   - Calculate: "2024-03-15 to 2025-03-14: $2,000; 2025-03-15 to 2026-03-14: $2,200"
   - Needs_Correction Should be true in this case!
═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════

Return ONLY valid JSON (no markdown, no code blocks, no explanation):

{{
  "needs_correction": true,
  "page": 5,
  "source_doc": "Company/UploadSessionId/lease_amendment_2024.pdf", (RETURN FULL SOURCE DOC PATH)
  "expired": false,
  "value": "$1,831.25",
  "confidence_score": 0.88,
  "future_value": "2026-10-01: $1,892.29; 2027-10-01: $1,953.33",
  "future_effective_date": "2026-10-01",
  "reason": "Found in current context page 5, rent schedule for period 10/1/2025-9/30/2026 which contains today's date. High confidence due to explicit table format.",
  "manual_review": True or False,
}}

CRITICAL JSON RULES:
- Use double quotes for all keys and string values
- Boolean values: true/false (lowercase, no quotes)
- Numeric values: no quotes (confidence_score is a float)
- Use string "Null" for missing values (not null without quotes)
- All dates in YYYY-MM-DD format
"""





    user_message = f"""Extract the field: {column}

Provide ONLY the JSON response with no additional text."""
    section_prompt_tokens = 0
    section_completion_tokens = 0
    if time_update:
       if column not in ['base_rent_amount_current', 'base_rent_schedule', 'base_rent_effective_date', 'rent_commencement_date']:
            if current_value not in [None, '', "Null", 'null']:
                value ={
                                'page': cell['page'],
                                'source_doc': cell['source_doc'],
                                'value': current_value,
                                'confidence_score': cell['confidence_score'],
                                'future_value': cell['future_value'],
                                'future_effective_date': cell['future_effective_date'],
                                'reason': cell['reason'],
                                'manual_review': cell['manual_review'],
                                'is_manual_change': cell['is_manual_change']
                                
                            }
            else: 
                value = None
            return section_prompt_tokens, section_completion_tokens, value
    
    response, prompt_tokens, completion_tokens = claude_message(claude_model, system_prompt, user_message, max_tokens=500)
    section_prompt_tokens += prompt_tokens
    section_completion_tokens += completion_tokens
    parsed = parse_model_payload(response)
    print("Parsed Response", parsed)
    value = None
    if parsed and len(parsed) > 0:
        for item in parsed:
            if item['needs_correction']:
                extracted_value = item.get('value', '')
                confidence = item.get('confidence_score', 0.0)
                future_value = item.get('future_value', "null")
                future_effect_date = item.get('future_effective_date', 'null')
                reason = item.get('reason')
                review = item.get('manual_review')

                    
                if extracted_value not in [None, '', "Null", 'null']:
                    value ={
                        'page': item.get('page', 0),
                        'source_doc': item.get('source_doc', ''),
                        'value': extracted_value,
                        'confidence_score': confidence,
                        'future_value': future_value,
                        'future_effective_date': future_effect_date,
                        'reason': reason,
                        'manual_review': review,
                        'is_manual_change': False
                    }
            elif not item['needs_correction'] or (column == 'lease_commencement_date' and is_manual_change):
                if current_value not in [None, '', "Null", 'null']:
                    value ={
                            'page': cell['page'],
                            'source_doc': cell['source_doc'],
                            'value': current_value,
                            'confidence_score': cell['confidence_score'],
                            'future_value': cell['future_value'],
                            'future_effective_date': cell['future_effective_date'],
                            'reason': cell['reason'],
                            'manual_review': cell['manual_review'],
                            'is_manual_change': cell['is_manual_change']
                        }
                    


    return section_prompt_tokens, section_completion_tokens, value


def claude_message(claude_model, system_prompt, user_message, max_tokens = 300):
    chat_response = claude.messages.create(
        model=claude_model,
        system=(system_prompt),
        messages=[
            {"role": "user", 'content': [
                {
                    'type': 'text',
                    'text': user_message
                }
            ]}
        ],
        temperature=0.0,
        max_tokens=max_tokens
    )
    response = chat_response.content[0].text
    prompt_tokens = chat_response.usage.input_tokens
    completion_tokens = chat_response.usage.output_tokens
    return response, prompt_tokens, completion_tokens

def point_to_dict(p):
    if hasattr(p, "model_dump"):          # pydantic v2
        return p.model_dump()
    if hasattr(p, "dict"):               # pydantic v1
        return p.dict()
    return {
        "id": getattr(p, "id", None),
        "score": getattr(p, "score", None),
        "payload": getattr(p, "payload", None),
        "vector": getattr(p, "vector", None),
    }


