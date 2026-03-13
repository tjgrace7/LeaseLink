"""
Property-level chat — answers questions across all tenants in a property in parallel.

This module powers the "property" entity_type branch of the entity_questions endpoint.
Rather than querying a single tenant's lease, it:

  1. Fetches all tenants for the property from Property_Tenant and validates the
     company_id matches.
  2. Rephrases the user's question with Claude (rephrase_question), which also
     determines whether a property-wide summary is needed (needs_overview flag).
  3. Embeds the rephrased question with text-embedding-3-large.
  4. Runs per-tenant AI queries in parallel (ThreadPoolExecutor, max 5 workers) via
     run_ai_for_tenant → tenant_ai_response.
  5. If needs_overview is True, generates a final property-wide summary by feeding all
     per-tenant short answers back through Claude (summary_response).
  6. Returns the list of per-tenant answer dicts (plus optional summary) together with
     aggregated token/cost statistics.

Helper functions sort_json and normalize_sources handle parsing and deduplicating the
JSON citation blocks that Claude returns inside fenced code blocks.
"""

from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from qdrant_client.http import models as rest
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import common.Supabase_api as Supabase_api
from web_api import Qdrant_ChatGPT
from worker_service.final_check import lease_check
import re
from qdrant_client import QdrantClient
from openai import OpenAI
from anthropic import Anthropic
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPEN_AI_PROJECT_KEY")
CLAUDE_API_KEY = os.getenv("Claude_API_KEY")

OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)
claude = Anthropic(api_key=CLAUDE_API_KEY)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase = Supabase_api.supabase_client_setup()

MAX_WORKERS = 5
from datetime import datetime


def get_propertyTenants(property_id, company_id):
    """Return all tenant rows for a property, or None if the company_id does not match.

    Queries Property_Tenant to get tenant IDs, then fetches the full tenant rows and
    validates that all tenants belong to the requesting company.
    """
    print("Get Property Tenants")
    pt = supabase.table("Property_Tenant").select("*").eq("property_id", property_id).execute()
    tenant_ids = [row["tenant_id"] for row in (pt.data or []) if row.get("tenant_id")]
    if not tenant_ids:
        print("Tenants: []")
    

    tenants_resp = supabase.table('tenant').select('*').in_('tenant_id', tenant_ids).execute()

    tenants = tenants_resp.data or []
    if tenants[0].get('property_management_id') != company_id:
        print("Company Id does not match id for tenants")
        return None
    return tenants

def normalize_tenant(tenant):
    """Ensure tenant is returned as a plain dict regardless of whether it came as a tuple."""
    if isinstance(tenant, tuple):
        if len(tenant) >= 2 and isinstance(tenant[1], dict):
            tenant = tenant[1]
        elif len(tenant) >= 1 and isinstance(tenant[0], dict):
            tenant = tenant[0]
    if not isinstance(tenant, dict):
        raise TypeError(f"Expected tenant dict, got {type(tenant)}: {tenant}")
    return tenant

def build_tenant_job_payload(tenant):
    """Build the job dict passed to run_ai_for_tenant, including summed unit square footage."""
    tenant_id = tenant.get("tenant_id")
    if not tenant_id:
        return None

    tenant_name = tenant.get("Tenant_Name")

    # Do Supabase reads in the main thread (safer), and compute sqft here.
    unit_res = supabase.table("Units").select("*").eq("tenant_id", tenant_id).execute()
    units = unit_res.data or []
    tenant_square_footage = sum((u.get("square_footage") or 0) for u in units)

    return {
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "property_management_id": tenant.get("property_management_id"),
        "tenant_square_footage": tenant_square_footage,
    }

def run_ai_for_tenant(job, collection_name, message_vector, ai_message, claude_model):
    """
    Runs in a worker thread. Keep this purely AI/network work if possible.
    Return a tuple so the main thread can aggregate safely.
    """
    tenant_id = job["tenant_id"]
    pm_id = job["property_management_id"]
    lease_ids = []
    total_completion_tokens = 0

    total_prompt_tokens =0

    if not pm_id:
        # If this can be missing, skip cleanly
        return None

    result = tenant_ai_response(
        tenant_id,
        pm_id,
        collection_name,
        message_vector,
        ai_message,
        claude_model,
        lease_ids, 
    )
    if not result:
        return None

    message_data, long_answer, json_data, prompt_tokens, completion_tokens = result
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens
    row = { 
        "short_answer": f"{job['tenant_name']} \n {message_data}",
        "tenant_id": tenant_id,
        "lease_file_path": None,
        "tenant_name": job["tenant_name"],
        "source_docs": json_data,
        'long_answer': long_answer,
        "square_footage": job["tenant_square_footage"],
    }


    return row, (total_prompt_tokens or 0), (total_completion_tokens or 0)
def get_supabase_data(tenants, claude_model, ai_message, collection_name, message_vector):
    """Fan out per-tenant AI queries in parallel and aggregate the results.

    Builds a job payload for each tenant, submits them to a ThreadPoolExecutor, and
    collects the per-tenant answer rows together with aggregated token counts.
    Returns (data, total_prompt_tokens, total_completion_tokens).
    """
    print("Get Supabase Data")



    data = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_square_footage = 0
    jobs = []

    for tenant in tenants:
        tenant = normalize_tenant(tenant)
    
            
        job = build_tenant_job_payload(tenant)
        total_square_footage += job["tenant_square_footage"]
        jobs.append(job)



    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_tenant = {
            executor.submit(
                run_ai_for_tenant,
                job,
                collection_name,
                message_vector,
                ai_message,
                claude_model,
            ): job
            for job in jobs
            if job is not None
        }

        for future in as_completed(future_to_tenant):
            result = future.result()
            if result is None:
                continue

            row, prompt_tokens, completion_tokens = result
            data.append(row)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
    return data, total_prompt_tokens, total_completion_tokens

def tenant_query(lease_ids, collection_name, message_vector, tenant_id, company_id, top_k=50):
    """Search Qdrant for lease chunks relevant to the query vector for a single tenant.

    Filters by tenant_id and company_id; optionally restricts to a specific set of
    lease_ids.  Applies a score-cutoff filter (60% of the best score) to drop low-
    relevance results.  Returns a formatted context string or None if no results.
    """
    if(lease_ids != []):
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
                        key="managementcompany_id",
                        match=MatchValue(value=company_id)),
                    FieldCondition(
                        key="lease_id",
                        match=MatchAny(any=lease_ids))
                ]
            ),)
    else:
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
                        key="managementcompany_id",
                        match=MatchValue(value=company_id))
                ]
            ),)
    points = resp.points
    results = points

    if points:
        best = points[0].score
        cutoff = best * .6
        results = [p for p in points if p.score >= cutoff]


    if not results:
        print("No Results found for Tenant_ID and Company_Id", tenant_id, company_id)
        return


    context = "\n\n".join([
     f"source_doc = {r.payload.get('source_doc', 'unknown')}, pageNumber = {r.payload.get('pageNumber', 'N/A')})\n{r.payload['text']}"
        for r in results if "text" in r.payload
        ])
    return context


def normalize_sources(raw_sources):
    """
    Returns a list[dict] shaped like:
    { tenant_name, source_doc, pageNumber, highlight_text }
    Ignores strings like 'Signed URL: ...'
    """
    if not raw_sources:
        print("Nothing Returned")
        return []

    # If it's a dict, assume it's one source object
    if isinstance(raw_sources, dict):
        raw_sources = [raw_sources]

    # If it's a string, it's not a source object
    if isinstance(raw_sources, str):
        return []

    normalized = []
    for s in raw_sources:
        if not isinstance(s, dict):
            # Skip strings, numbers, etc.
            continue

        # Require at least doc + page to be useful
        source_doc = s.get("source_doc") or s.get("sourceDoc")
        page = s.get("pageNumber") or s.get("page") or s.get("page_number")

        try:
            page = int(page)
        except: 
            page = "N/A"
        normalized.append({
            "tenant_name": s.get("tenant_name") or s.get("tenantName"),
            "source_doc": source_doc,
            "pageNumber": page,
            "highlight_text": s.get("highlight_text") or s.get("highlightText") or "",
        })

    return normalized
def tenant_ai_response(tenant_id, company_id, collection_name, message_vector, ai_message, claude_model, lease_ids):
    """Call Claude with retrieved lease context for a single tenant and return the structured answer.

    Retrieves context via tenant_query, builds a detailed system prompt, calls Claude,
    strips JSON fences from the text response, and parses the JSON citation block.
    Returns (short_answer, long_answer, source_docs, prompt_tokens, completion_tokens).
    """
    context = tenant_query(lease_ids, collection_name, message_vector, tenant_id, company_id)
    now = datetime.now()
    system_prompt = f"""You are a helpful assistant answering questions about lease documents.

CONTEXT:
- Current date: {now.strftime("%B %d, %Y")}
- Your goal is to provide the most accurate answer possible for the tenant. This system is answering questions for each tenant seperately. Use only the provided context for the specific tenant.
- DO NOT USE Tenant Name in short or long_answer

TASK:
Use the provided lease document excerpts to answer the user's question concisely and accurately.

RULES:
1. If excerpts don't contain relevant information, respond with: "No Data Available"
2. Be concise - avoid unnecessary elaboration
3. Extract tenant_id from context when answering
4. Prioritize more recent documents when information conflicts

CALCULATION & ANALYSIS CAPABILITIES:
You may need to:
- **Calculate rent amounts**: Apply escalation clauses, percentage rent, or prorated amounts based on dates
- **Determine time periods**: Calculate lease terms, notice periods, or option exercise deadlines
- **Interpret co-tenancy clauses**: Define triggers, remedy periods, and rent reduction formulas
- **Apply conditional logic**: Determine if conditions are met (e.g., sales thresholds, occupancy requirements)

When performing calculations:
- Show your work clearly but concisely
- State the formula or clause used
- Include relevant dates and amounts
- Note any assumptions made

RELEVANT LEASE DOCUMENT EXCERPTS:
{context}

---

ANSWER FORMAT EXAMPLES:

**For rent calculations:**
"Monthly base rent: $5,000. With 3% annual escalation from Jan 1, 2024, current rent (as of {now.strftime("%B %Y")}) is $5,150. (tenant_id: ABC-001)"

**For co-tenancy clauses:**
"Co-tenancy triggered if Anchor Tenant (defined as grocery store ≥40,000 SF) goes dark for >90 days. Remedy: Tenant may pay alternative rent of lesser of (a) $2/SF/month or (b) 8% of gross sales. (tenant_id: ABC-001)"

**For date-based questions:**
"Lease expires June 30, 2026. Tenant has two 5-year renewal options, exercisable with 180 days notice. Next option deadline: December 31, 2025. (tenant_id: ABC-001)"

**For conditional clauses:**
"Percentage rent triggers at $500,000 annual sales threshold. Current lease year: Jan 1 - Dec 31, 2026. If threshold met, tenant pays 6% of gross sales exceeding $500,000. (tenant_id: ABC-001)"

---

CITATION REQUIREMENTS:
At the end of your answer, return ONLY the context chunks you actually used in JSON format.

IMPORTANT:
- Include ONLY chunks that directly supported your answer or calculations
- If multiple chunks from the same document page were used, COMBINE them into a single JSON object
- The `highlight_text` should contain the exact text from the chunk(s) used
- DO NOT return duplicate entries with the same `source_doc` and `pageNumber`

Required JSON format:

The json wants a short and long answer. This is where you will answer the questions. Short answer less than 750 characters. Short Answer Must contain the Tenants Name

 Long answer to limit of max tokens

```json
[    
    Curly Bracket
    'short_answer': Enter_Short_Response Here (Do not add Tenant Name to message)
    'long_answer': Enter_Long_Answer Here (Do not add Tenant Name to Message)
(1 short and long answer per response. Many sources potential)
  (If no sources: Curly Bracket  
  "tenant_name": name of the tenant,
  )'sources': Curly Bracket 
    "tenant_name": "The name of the tenant",
    "source_doc": "leaselink/dairy_queen/",
    "pageNumber": 12,
    "highlight_text": "abc-123",
    Curly Bracket close
  Curly bracket close
]

"""
    chat_response = claude.messages.create(
        model=claude_model,
        system=(system_prompt),
        messages=[
            {"role": "user", 'content': [
                {
                    'type': 'text',
                    'text': ai_message
                }
            ]}
        ],
        temperature=0.0,
        max_tokens=4000
    )
    token_usage = chat_response.usage
    prompt_tokens = token_usage.input_tokens
    completion_tokens = token_usage.output_tokens
    chat_message = chat_response.content[0].text
    response_message = re.sub(r"```(?:json|emailjson)\s*.*?```", "", chat_message, flags=re.DOTALL|re.IGNORECASE).strip()

    


    json_data = Qdrant_ChatGPT._extract_after_fence(chat_message, "json")

    json_data, short_answer, long_answer = sort_json(json_data)
    
    return short_answer, long_answer, json_data or [], prompt_tokens, completion_tokens


def sort_json(json_data):
    """Extract short_answer, long_answer, and a deduplicated/enriched source list from the AI response JSON.

    Pulls the first object's short_answer and long_answer, then iterates all source
    objects, generates signed PDF URLs via Supabase, and merges duplicate (source_doc,
    pageNumber) pairs by concatenating their highlight_text.
    Returns (source_list, short_answer, long_answer).
    """
    print("json_data type:", type(json_data))
    try:
        print(json.dumps(json_data, indent=2))
    except Exception:
        print("json_data (non-serializable):", str(json_data)[:500])

    if json_data is None:
        return [], None, None

    # Normalize to a list of dicts
    if isinstance(json_data, dict):
        items = [json_data]
    elif isinstance(json_data, list):
        items = [x for x in json_data if isinstance(x, dict)]
        if not items:
            return [], None, None
    else:
        return [], None, None

    # Pull answers from the first object
    first = items[0]
    short_answer = first.get("short_answer")
    long_answer = first.get("long_answer")

    merged = {}

    for data in items:
        sources = normalize_sources(data.get("sources"))

        for s in sources:
            if not isinstance(s, dict):
                continue

            tenant_name = s.get("tenant_name")
            source_doc = s.get("source_doc")
            page = s.get("pageNumber")
            highlight = s.get("highlight_text", "") or ""

            # Case 1: has doc + page -> build viewer_url + dedupe by (doc, page)
            if source_doc is not None and page is not None:
                key = (source_doc, page)

                signed_url = Supabase_api.get_signed_url(
                    supabase, "lease-docs", source_doc
                )

                # page is not None here, so always use #page=
                viewer_url = f"{signed_url}#page={page}&highlight_text={highlight}"

                if key not in merged:
                    merged[key] = {
                        "tenant_name": tenant_name,
                        "source_doc": source_doc,
                        "pageNumber": page,
                        "highlight_text": highlight,
                        "viewer_url": viewer_url,
                    }
                else:
                    # append highlight text (avoid duplicates)
                    if highlight and highlight not in merged[key]["highlight_text"]:
                        merged[key]["highlight_text"] = (
                            merged[key]["highlight_text"] + " | " + highlight
                        ).strip(" |")

            # Case 2: missing doc or page -> fallback to tenant_name grouping
            else:
                key = tenant_name or "unknown_tenant"
                if key not in merged:
                    merged[key] = {"tenant_name": tenant_name}

    out = list(merged.values())
    return out, short_answer, long_answer


def rephrase_question(question: str, claude_model: str) -> str:
        """Rewrite the property-level question as a Qdrant-optimised search query.

        Claude also determines whether the question requires a property-wide overview
        (needs_overview flag returned in a JSON fence block).  The rephrased text is
        embedded with text-embedding-3-large to produce the search vector.
        Returns (ai_message, message_vector, prompt_tokens, completion_tokens,
                 embedding_token_count, needs_overview).
        """
        print("Rephrase Question")
        now = datetime.now()
        message_summary = claude.messages.create(
            model=claude_model,
            system=(f"""
You are preparing a search query for a vector database (Qdrant) containing lease documents.

Your task: Rewrite the user's question to maximize semantic search relevance.

DOCUMENT CONTEXT:
- Documents sharing a tenant_id belong to the same lease
- Types: main_lease, amendment, renewal, addendum
- Newer documents override older ones when they conflict
- Current date: {now}

QUERY OPTIMIZATION GUIDELINES:

For property size questions:
→ Include: square footage, acreage, land area, site dimensions, parcel size

For financial questions:
→ Include: rent amounts, payment terms, commencement dates, dollar values, financial obligations

For date-sensitive questions:
→ Include: effective dates, expiration dates, renewal dates, term length

RULES:
1. Return ONE semantically precise query
2. Remove conversational fluff
3. Preserve key details (tenant names, addresses, specific dates/amounts if mentioned)
4. Don't add information not in the original question
5. Determine if this question requires a full property overview. 

```json Open Curly Bracket
    needs_overview: True/False

    Clost Curly Bracket`
            """),
            messages=[
                {"role": "user", "content": [{
                    'type': 'text',
                    'text': question}]}
            ],
            temperature=0.0,
            max_tokens=10000
        )
        token_usage = message_summary.usage
        prompt_tokens = token_usage.input_tokens
        completion_tokens = token_usage.output_tokens
        ai_message = message_summary.content[0].text
        response_message = re.sub(r"```(?:json|emailjson)\s*.*?```", "", ai_message, flags=re.DOTALL|re.IGNORECASE).strip()
        json_data = Qdrant_ChatGPT._extract_after_fence(ai_message, "json")

        needs_review = json_data['needs_overview']
    


        
        message_vector = OpenAIclient.embeddings.create(
            input=response_message,
            model="text-embedding-3-large"
        ).data[0].embedding

        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        embedding_token_count = len(encoding.encode(ai_message))
        print("Embedding Token Count:", embedding_token_count)


                

        return ai_message, message_vector, prompt_tokens, completion_tokens, embedding_token_count, needs_review or False

def summary_response(short_messages, claude_model, question):
    """Generate a single consolidated summary answer across all tenants for a property-overview question.

    Takes the per-tenant short answers as input context, calls Claude to produce a
    unified short + long answer with merged source citations, and returns
    (short_answer, long_answer, json_data, completion_tokens, prompt_tokens).
    """
    now = datetime.now()
    system_prompt = f"""Your job is to summarize all the tenantdata from {short_messages}. It contains the source documents that were used to find the data. Combine each individual tenant Answer into 1 conscise summary.
                TASK:
Use the provided lease document excerpts to answer the user's question concisely and accurately.

RULES:
1. If excerpts don't contain relevant information, respond with: "No Data Available"
2. Be concise - avoid unnecessary elaboration
3. Extract tenant_id from context when answering
4. Prioritize more recent documents when information conflicts

CALCULATION & ANALYSIS CAPABILITIES:
You may need to:
- **Calculate rent amounts**: Apply escalation clauses, percentage rent, or prorated amounts based on dates
- **Determine time periods**: Calculate lease terms, notice periods, or option exercise deadlines
- **Interpret co-tenancy clauses**: Define triggers, remedy periods, and rent reduction formulas
- **Apply conditional logic**: Determine if conditions are met (e.g., sales thresholds, occupancy requirements)

When performing calculations:
- Show your work clearly but concisely
- State the formula or clause used
- Include relevant dates and amounts
- Note any assumptions made



---

ANSWER FORMAT EXAMPLES:

**For rent calculations:**
"Monthly base rent: $5,000. With 3% annual escalation from Jan 1, 2024, current rent (as of {now.strftime("%B %Y")}) is $5,150. (tenant_id: ABC-001)"

**For co-tenancy clauses:**
"Co-tenancy triggered if Anchor Tenant (defined as grocery store ≥40,000 SF) goes dark for >90 days. Remedy: Tenant may pay alternative rent of lesser of (a) $2/SF/month or (b) 8% of gross sales. (tenant_id: ABC-001)"

**For date-based questions:**
"Lease expires June 30, 2026. Tenant has two 5-year renewal options, exercisable with 180 days notice. Next option deadline: December 31, 2025. (tenant_id: ABC-001)"

**For conditional clauses:**
"Percentage rent triggers at $500,000 annual sales threshold. Current lease year: Jan 1 - Dec 31, 2026. If threshold met, tenant pays 6% of gross sales exceeding $500,000. (tenant_id: ABC-001)"

---

CITATION REQUIREMENTS:
At the end of your answer, return ONLY the context chunks you actually used in JSON format.

IMPORTANT:
- Include ONLY chunks that directly supported your answer or calculations
- If multiple chunks from the same document page were used, COMBINE them into a single JSON object
- The `highlight_text` should contain the exact text from the chunk(s) used
- DO NOT return duplicate entries with the same `source_doc` and `pageNumber`

Required JSON format:

The json wants a short and long answer. This is where you will answer the questions. Short answer less than 750 characters. Short Answer Must contain the Tenants Name

 Long answer to limit of max tokens

```json
[    
    Curly Bracket
    'short_answer': Enter_Short_Response Here
    'long_answer': Enter_Long_Answer Here
(1 short and long answer per response. Many sources potential)
  (If no sources: omit)'sources': Curly Bracket 
    "tenant_name": "All Tenants",
    "source_doc": "leaselink/dairy_queen/",
    "pageNumber": 12,
    "highlight_text": "abc-123",
    Curly Bracket close
  Curly bracket close
]
             """
    message_summary = claude.messages.create(
            model=claude_model,
            system=system_prompt,
            messages=[
                    {"role": "user", "content": [{
                        'type': 'text',
                        'text': question}]}
                ],
            temperature=0.0,
            max_tokens=10000)
    token_usage = message_summary.usage
    prompt_tokens = token_usage.input_tokens
    completion_tokens = token_usage.output_tokens
    ai_message = message_summary.content[0].text
    json_data = Qdrant_ChatGPT._extract_after_fence(ai_message, 'json')

    json_data, short_answer, long_answer = sort_json(json_data)

    return short_answer, long_answer, json_data, completion_tokens, prompt_tokens
            

def property_chat_request(collection_name, property_id, message, oldData, claude_model, company_id):
    """Entry point for a property-level chat request.

    Orchestrates the full flow: fetch tenants, rephrase question, run parallel
    per-tenant queries, optionally generate a property-wide summary, then compute
    and return costs.  Returns (tenant_data, prompt_tokens, prompt_cost,
    completion_tokens, completion_cost).
    """
    print("Company Id", company_id)

    all_prompt_tokens = 0
    all_completion_tokens = 0
    all_embedding_token_count = 0
    default_response = (
        "Sorry, there was an error processing your question. Please try again later.",
    )
    try:
        default_response = (
        "Sorry, there was an error processing your question. Please try again later.",
        )
        tenants = get_propertyTenants(property_id, company_id)
        
        if tenants == None:
            return default_response, all_prompt_tokens, 0.0, all_completion_tokens, 0.0


        ai_message, message_vector, prompt_tokens, completion_tokens, embedding_token_count, needs_overview = rephrase_question(message, claude_model)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
        all_embedding_token_count += embedding_token_count


        tenantdata, prompt_tokens, completion_tokens= get_supabase_data(tenants,claude_model, ai_message,  collection_name, message_vector)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
        print("Needs Overview", needs_overview)
        if needs_overview:
           ai_answers = [
                (answer['short_answer'], answer['source_docs'], answer['square_footage'])
                for answer in tenantdata
            ]
           short_answer, long_answer, json_data,prompt_tokens, completion_tokens = summary_response(ai_answers, claude_model, message)
           all_prompt_tokens += prompt_tokens
           all_completion_tokens += completion_tokens
           summary = {
            "short_answer": f"All Tenants \n {short_answer}",
            "tenant_id": "All Tenants",
            "lease_file_path": None,
            "tenant_name": "All Tenants",
            "source_docs": json_data,
            'long_answer': long_answer,
            "square_footage": sum([sf['square_footage'] for sf in tenantdata]),
           }
           tenantdata.append(summary)

        prompt_cost = (all_prompt_tokens / 1000 * 0.003) + (all_embedding_token_count / 1000 * 0.00013)
        completion_cost = all_completion_tokens / 1000 * 0.015
        
        print("Total Cost: $", prompt_cost+completion_cost, "Prompt Cost:", prompt_cost, "Completion Cost:", completion_cost)
        return tenantdata, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost
    except Exception as e:
        print("Error in property chat request:", e)
        traceback.print_exc()
        prompt_cost = (all_prompt_tokens / 1000 * 0.01) + (all_embedding_token_count / 1000 * 0.00013)
        completion_cost = all_completion_tokens / 1000 * 0.03
        
        return default_response, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost
    
