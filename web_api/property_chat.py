from qdrant_client.http.models import  Filter, FieldCondition, MatchValue, MatchAny
from qdrant_client.http import models as rest
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import common.Supabase_api as Supabase_api
from web_api import Qdrant_ChatGPT
from worker_service.final_check import lease_check
import  re
from qdrant_client import QdrantClient
from openai import OpenAI
from anthropic import Anthropic
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPEN_AI_PROJECT_KEY")
CLAUDE_API_KEY = os.getenv("Claude_API_KEY")

OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)
claude = Anthropic(api_key=CLAUDE_API_KEY)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase = Supabase_api.supabase_client_setup()

MAX_WORKERS = 5
from datetime import datetime


def get_propertyTenants(property_id):
    print("Get Property Tenants")
    pt = supabase.table("Property_Tenant").select("*").eq("property_id", property_id).execute()
    tenant_ids = [row["tenant_id"] for row in (pt.data or []) if row.get("tenant_id")]
    if not tenant_ids:
        print("Tenants: []")
    

    tenants_resp = supabase.table('tenant').select('*').in_('tenant_id', tenant_ids).execute()

    tenants = tenants_resp.data or []
    print("Tenants:", tenants)
    return tenants

def normalize_tenant(tenant):
    if isinstance(tenant, tuple):
        if len(tenant) >= 2 and isinstance(tenant[1], dict):
            tenant = tenant[1]
        elif len(tenant) >= 1 and isinstance(tenant[0], dict):
            tenant = tenant[0]
    if not isinstance(tenant, dict):
        raise TypeError(f"Expected tenant dict, got {type(tenant)}: {tenant}")
    return tenant

def build_tenant_job_payload(tenant):
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

    message_data, json_data, prompt_tokens, completion_tokens = result
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens
    print("json_data:", json.dumps(json_data, indent=2))
    row = { 
        "ai_response": message_data,
        "tenant_id": tenant_id,
        "lease_file_path": None,
        "tenant_name": job["tenant_name"],
        "source_docs": json_data,
        "square_footage": job["tenant_square_footage"],
    }

    return row, (total_prompt_tokens or 0), (total_completion_tokens or 0)
def get_supabase_data(tenants,  claude_model, ai_message, collection_name, message_vector):
    print("Get Supabase Data")



    data = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_embedding_tokens = 0
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
    return data, total_prompt_tokens, total_completion_tokens, total_square_footage, total_embedding_tokens

def tenant_query(lease_ids, collection_name, message_vector, tenant_id, company_id, top_k=70):
    if(lease_ids != []):
        print("Filtering by Lease Ids:", lease_ids)
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
    print("Count of Results Pre Cutoff:", len(results))
    print("Scores of Results Pre Cutoff:", [p.score for p in points])
    if points:
        best = points[0].score
        cutoff = best * .6
        results = [p for p in points if p.score >= cutoff]

    print("Count of Results Post Cutoff:", len(results))
    if not results:
        print("No Results found for Tenant_ID and Company_Id", tenant_id, company_id)
        return


    context = "\n\n".join([
     f"source_doc = {r.payload.get('source_doc', 'unknown')}, pageNumber = {r.payload.get('pageNumber', 'N/A')})\n{r.payload['text']}"
        for r in results if "text" in r.payload
        ])
    return context

def tenant_ai_response(tenant_id, company_id, collection_name, message_vector, ai_message, claude_model, lease_ids):
    context = tenant_query(lease_ids, collection_name, message_vector, tenant_id, company_id)
    now = datetime.now()
    system_prompt = f"""You are a helpful assistant answering questions about lease documents.

CONTEXT:
- Current date: {now.strftime("%B %d, %Y")}
- You are step 2 in a multi-step process that may repeat for each tenant at a property
- Your answer will be fed to another AI system, NOT directly to the user

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

```json
[
  Curly Bracket
    "source_doc": "leaselink/dairy_queen/",
    "pageNumber": 12,
    "highlight_text": "abc-123"
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
        max_tokens=800
    )
    token_usage = chat_response.usage
    prompt_tokens = token_usage.input_tokens
    completion_tokens = token_usage.output_tokens
    chat_message = chat_response.content[0].text
    response_message = re.sub(r"```(?:json|emailjson)\s*.*?```", "", chat_message, flags=re.DOTALL|re.IGNORECASE).strip()

    


    json_data = Qdrant_ChatGPT._extract_after_fence(chat_message, "json")

    if json_data:
        merged = {}
        for d in json_data:


            key = (d.get('source_doc'), d.get('pageNumber'))
            if key not in merged:
                merged[key] = {
                    "source_doc": d.get("source_doc"),
                    "pageNumber": d.get('pageNumber'),
                    "highlight_text": d.get('highlight_text', ""),
                    
                }

            else:
                ht = d.get("highlight_text", "")
                if ht and ht not in merged[key]['highlight_text']:
                    merged[key]['highlight_text'] = (merged[key]['highlight_text'] + " | " + ht).strip(" |")
            json_data = list(merged.values())

    return response_message, json_data or [], prompt_tokens, completion_tokens

def rephrase_question(question: str, claude_model: str) -> str:
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

    


        
        message_vector = OpenAIclient.embeddings.create(
            input=response_message,
            model="text-embedding-3-large"
        ).data[0].embedding

        encoding = tiktoken.encoding_for_model("text-embedding-3-large")
        embedding_token_count = len(encoding.encode(ai_message))
        print("Embedding Token Count:", embedding_token_count)


                

        return ai_message, message_vector, prompt_tokens, completion_tokens, embedding_token_count

def final_property_chat(rephrased_message, tenant_data, oldmessages, claude_model, collection_name, total_square_footage):
    print("Final Property Chat")
    now = datetime.now().strftime("%B %d, %Y")
    total_prompt_tokens = 0
    total_completion_tokens = 0
    system_prompt = f"""You are a helpful assistant answering questions about ai commercial leases for property management companies. Because of the large amount of data you have to work with we have already filtered it down to relevant information.
    columns that end with psf mean per square foot.
    If they asked about total square footage of the property use this number: {total_square_footage}. If they didn't ask about it. It will be zero.
    Be very careful giving averages for expected rents financial obligations. We don't want to have any semblance of price fixing.
    Here is each tenants relevant data: {tenant_data}. It is sorted by Tenant_id and within each tenant it is sorted by the most recent lease information.
    Here are previous messages related to this tenant:
Each message includes a role:
- "user" means it was written by the property manager
- "assistant" means it was your previous response

{oldmessages}
If a user asks a time-based question (e.g., about rent, terms, insurance), use the following as the current date:
** {now}**

Many Time Based Questions will reference documents that say term between September 2021 - August 2025

If it is a Day in July 2025. That falls within that period. If the month and Year are outside that date and time. It does not fall within that period.
---
If the provided tenant data says something like ai_message: "I don't have any context for this tenant" Do not ask for more information about that tenant. It has already been queried and found nothing.
""" + """
``` json
[
{ 
    {
    "tenant_id": "the tenant uuid that you need more information about"
    "vector_query": "a concise vector search query that will help find more relavant information about this tenant",
    },
    {... loop for each tenant}}]

---
If you have enough information to answer the question, provide a concise and accurate answer based on the tenant data provided.
At the end of your answer, if you used any specific tenant data chunks, return them in (Anything sorted by tenant_id that you used in your answer. Source_doc and lease_file_path are the same thing if no page number included use 0):
``` json
[
{
    "tenant_name": "The name of the tenant",
    "source_doc": "leaselink/dairy_queen/",
    "pageNumber": 12,
    "highlight_text": "abc-123"
},
{... loop for each chunk used in your answer}}]
"""
    chat_response = claude.messages.create(
        model=claude_model,
        system=(system_prompt),
        messages=[
            {"role": "user", 'content': [
                {
                    'type': 'text',
                    'text': rephrased_message
                }
            ]}
        ],
        temperature=0.0,
        max_tokens=8000
    )
    chat_message = chat_response.content[0].text
    tokens = chat_response.usage
    total_prompt_tokens += tokens.input_tokens
    total_completion_tokens += tokens.output_tokens
    final_embedding_token_count = 0



    json_data = Qdrant_ChatGPT._extract_after_fence(chat_message, "json")
    print("JSON Data Extracted:", json_data)
    if json_data:
        merged = {}
        for d in json_data:
                key = (d.get('source_doc'), d.get('pageNumber'))
                if key not in merged:
                    if d.get('source_doc') is None:
                        continue
                    signed_url = Supabase_api.get_signed_url(supabase, "lease-docs", d.get('source_doc'))
                    viewer_url = ""
                    if d.get('pageNumber') is None:
                        viewer_url = f"{signed_url}&highlight_text={d.get('highlight_text')}"
                    else:
                        viewer_url = f"{signed_url}#page={d.get('pageNumber')}&highlight_text={d.get('highlight_text')}"
                    key = (d.get('source_doc'), d.get('pageNumber'))
                    
                    merged[key] = {
                        'tenant_name': d.get('tenant_name'),
                        "source_doc": d.get("source_doc"),
                        "pageNumber": d.get('pageNumber'),
                        "highlight_text": d.get('highlight_text', ""),
                        "viewer_url": viewer_url
                    }

                else:
                    ht = d.get("highlight_text", "")
                    if ht and ht not in merged[key]['highlight_text']:
                        merged[key]['highlight_text'] = (merged[key]['highlight_text'] + " | " + ht).strip(" |")
        json_data = list(merged.values())

    response_message = re.sub(r"```(?:json)\s*.*?```", "", chat_message, flags=re.DOTALL|re.IGNORECASE).strip()

    return response_message, json_data, total_prompt_tokens, total_completion_tokens,final_embedding_token_count or 0.0


def final_query(query, chunks, tenant_id, collection_name): 
    print("Final Query for Property:", tenant_id)

    message_vector = OpenAIclient.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        ).data[0].embedding

    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    embedding_token_count = len(encoding.encode(query))

    response = qdrant.query_points(
        collection_name=collection_name,
        query=message_vector,
        using='dense_vector',
        limit=chunks,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenantid",
                    match=MatchValue(value=tenant_id)),
                
                
            ]
        ),
    )

    if not response:
        print("No Results found for Final Query for Tenant_ID", tenant_id)
        return []
    return response, embedding_token_count

        

def property_chat_request(collection_name, property_id,message, oldData, claude_model):
    all_prompt_tokens = 0
    all_completion_tokens = 0
    all_embedding_token_count = 0
    default_response = (
        "Sorry, there was an error processing your question. Please try again later.",
    )
    try:
        
        tenants = get_propertyTenants(property_id)


        ai_message, message_vector, prompt_tokens, completion_tokens, embedding_token_count = rephrase_question(message, claude_model)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
        all_embedding_token_count += embedding_token_count


        tenantdata, prompt_tokens, completion_tokens, total_square_footage, embedding_tokens = get_supabase_data(tenants,claude_model, ai_message,  collection_name, message_vector)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
        all_embedding_token_count += embedding_tokens

        final_response, json_data, prompt_tokens, completion_tokens, embedding_token_count_2 = final_property_chat(message, tenantdata, oldData, claude_model, collection_name, total_square_footage)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
        all_embedding_token_count += embedding_token_count_2
        prompt_cost = (all_prompt_tokens / 1000 * 0.003) + (all_embedding_token_count / 1000 * 0.00013)
        completion_cost = all_completion_tokens / 1000 * 0.015
        
        print("Final Response", final_response)
        print("Total Cost: $", prompt_cost+completion_cost, "Prompt Cost:", prompt_cost, "Completion Cost:", completion_cost)
        return final_response, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost, json_data
    except Exception as e:
        print("Error in property chat request:", e)
        prompt_cost = (all_prompt_tokens / 1000 * 0.01) + (all_embedding_token_count / 1000 * 0.00013)
        completion_cost = all_completion_tokens / 1000 * 0.03
        
        return default_response, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost, []
    
