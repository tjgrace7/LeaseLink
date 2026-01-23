from qdrant_client.http.models import  Filter, FieldCondition, MatchValue, SearchParams
from qdrant_client.http import models as rest
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import common.Supabase_api as Supabase_api
from web_api import Qdrant_ChatGPT
from memory_profiler import profile
import posixpath, re
from urllib.parse import quote
from qdrant_client import QdrantClient
from openai import OpenAI
from anthropic import Anthropic
import os
import math
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPEN_AI_PROJECT_KEY")
CLAUDE_API_KEY = os.getenv("Claude_API_KEY")

OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)
claude = Anthropic(api_key=CLAUDE_API_KEY)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase = Supabase_api.supabase_client_setup()

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

def get_supabase_data(tenants,  claude_model, ai_message, collection_name, message_vector, property_id):
    print("Get Supabase Data")



    data = []

    total_prompt_tokens = 0
    total_completion_tokens = 0

    for tenant in tenants:
        if isinstance(tenant, tuple):
            if len(tenant) >= 2 and isinstance(tenant[1], dict):
                tenant = tenant[1]
            elif len(tenant) >=1 and isinstance(tenant[0], dict):
                tenant = tenant[0]
        if not isinstance(tenant, dict):
            raise TypeError(f"Expected tenant dict, got {type(tenant)}: {tenant}")
        
        tenant_id = tenant.get('tenant_id')
        tenant_name = tenant.get("Tenant_Name")
        total_square_footage = 0
        unit_res = supabase.table('Units').select('*').eq('tenant_id', tenant_id).execute()
        units = unit_res.data
        tenant_square_footage = 0
        for unit in units:
            tenant_square_footage += unit['square_footage']
            total_square_footage += unit['square_footage']

            

        

        if not tenant_id: 
            print("Skipping tenant with missing tenant_id:", tenant)
            continue

        result = tenant_ai_response(tenant_id, tenant['property_management_id'],  collection_name, message_vector, ai_message, claude_model) 
        if not result:
            continue
        message_data, json_data, prompt_tokens, completion_tokens = result
        total_prompt_tokens += (prompt_tokens or 0)
        total_completion_tokens += (completion_tokens or 0)
        data.append({'ai_response': message_data, 'tenant_id': tenant_id, 'lease_file_path': None, "tenant_name": tenant_name, "source_docs": json_data, 'square_footage': tenant_square_footage})
    print(data)
    return data, total_prompt_tokens, total_completion_tokens, total_square_footage
        


def tenant_ai_response(tenant_id, company_id, collection_name, message_vector, ai_message, claude_model, top_k=7):
    print("Tenant AI Response")
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=('dense_vector', message_vector),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenantid",
                    match=MatchValue(value=tenant_id)),
                FieldCondition(
                    key="managementcompany_id",
                    match=MatchValue(value=company_id))
            ]
        ),
    )
    if not results:
        print("No Results found for Tenant_ID and Company_Id", tenant_id, company_id)
        return


    context = "\n\n".join([
     f"source_doc = {r.payload.get('source_doc', 'unknown')}, pageNumber = {r.payload.get('pageNumber', 'N/A')})\n{r.payload['text']}"
        for r in results if "text" in r.payload
        ])
    now = datetime.now()
    system_prompt = f"""You are a helpful assistant answering questions about lease documents.
    The current date is {now.strftime("%B %d, %Y")}. Use the provided lease document excerpts to answer the user's question.
    If the excerpts do not contain relevant information, respond with "I don't know based on the provided documents."
    Be concise and accurate in your responses. You are the 2nd step in a multi-step process to answer the user's question. This process could be repeated many times based on the number of tenants at the property.
    So Provide Consise AnswersYour answer will be fed to another ai not the user directly.
    Here are the relevant lease document excerpts: 
    {context}


    Answer the question clearly. If you don't know. Say - 'No Data Available'

At the end of your answer, if you used any specific context chunks, return them in the following JSON format. The `highlight_text` should be the exact text from the chunk you used in your answer. Use tenantid from context to answer

Do NOT include any chunks that were not used in your answer.

If two chunks share the same `source_doc` and `pageNumber`, and both were used in your answer, combine their `highlight_text` fields into one string, and return a single JSON object for that page! DO NOT RETURN TWO JSON STRINGS WITH THE SAME source_doc and pageNumber

Use this format exactly:

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
        max_tokens=400
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
You are preparing a search query for a vector database (Qdrant) to help retrieve the most relevant lease documents for answering a property management question.

Your goal is to rewrite or summarize the user's question in a way that improves semantic search relevance.

Important considerations:

- Documents with the same tenant_id belong to the same lease context and may include amendments, renewals, or overrides.
- These documents can conflict. In such cases, **more recent documents should take precedence**. The current date is: {now}
- Document types include: main_lease, amendment, renewal, addendum, etc.
- Amendments or renewals may override clauses in the original lease — always prioritize newer documents for accuracy.
- Only use what's needed from the question to guide the search (avoid restating unrelated fluff).

If the user asks about square footage, land size, or area, generate a query that includes terms like:
- land size
- square footage
- site area
- parcel size
- property area

Land size is often expressed as square feet or acres. The answer may come from county property reports or appraisals.

Return a **single, semantically precise** version of the user's question that will help match the most relevant document chunks in the vector database.

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
        message_vector = OpenAIclient.embeddings.create(
            input=ai_message,
            model="text-embedding-3-large"
        ).data[0].embedding
        print("Embedding Created")
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
If you require more information about a specific tenant to answer the question answer exactly this:""" + """
``` json
[
{ 
    {
    "more_info" : true,
    "tenant_id": "the tenant uuid that you need more information about"
    "vector_query": "a concise vector search query that will help find more relavant information about this tenant",
    },
    {... loop for each tenant}}]

---
If you have enough information to answer the question, provide a concise and accurate answer based on the tenant data provided.
At the end of your answer, if you used any specific tenant data chunks, return them in (Anything sorted by tenant_id that you used in your answer. Source_doc and lease_file_path are the same thing if no page number included use 0):
``` json
[
    {more_info": False },
{
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
        max_tokens=4000
    )
    chat_message = chat_response.content[0].text
    print("Chat Message:", chat_message)
    tokens = chat_response.usage
    total_prompt_tokens += tokens.input_tokens
    total_completion_tokens += tokens.output_tokens
    final_embedding_token_count = 0



    json_data = Qdrant_ChatGPT._extract_after_fence(chat_message, "json")
    print("JSON Data Extracted:", json_data)
    next_message = False
    query_results = []
    if json_data:
        merged = {}
        for d in json_data:
            more_info = d.get("more_info", False)
            if more_info:
                key = f"{d.get('tenant_id'), d.get('vector_query')}"
                if key not in merged:
                    next_message = True
                    final_query_result, embedding_token_count = final_query(d.get('vector_query'), d.get('chunks', 5), tenant_id=d.get('tenant_id'), collection_name=collection_name)
                    final_embedding_token_count += embedding_token_count  
                    print("Final Query Result for Tenant:", d.get('tenant_id'), final_query_result)
                    query_results.append({
                        "tenant_id": d.get('tenant_id'),
                        "vector_query": d.get('vector_query'),
                        "results": final_query_result})
                    continue
            else:
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

    if next_message:
        print("Getting Next Message with more info")
        now = datetime.now().strftime("%B %d, %Y")
        system_prompt_2 = f"""You are a helpful assistant answering questions about ai commercial leases for property management companies. Because of the large amount of data you have to work with we have already filtered it down to relevant information.
        Here is the relevant data from tenants {tenant_data}. It is sorted by Tenant_id and within each tenant it is sorted by the most recent lease information. We have also retrieved more relevant information based on your last request.
         If they asked about total square footage of the property use this number: {total_square_footage}. If they didn't ask about it. It will be zero.
        Here is the new relevant data we found for you:{query_results}.
        Here are previous messages related to this tenant:
Each message includes a role:
- "user" means it was written by the property manager
- "assistant" means it was your previous response{oldmessages}If a user asks a time-based question (e.g., about rent, terms, insurance), use the following as the current date:
**{now}**

        Many Time Based Questions will reference documents that say term between September 2021 - August 2025
        If it is a Day in July 2025. That falls within that period. If the month and Year are outside that date and time. It does not fall within that period.
        ---
        Answer the question concisely and accurately based on the tenant data provided.
        At the end of your answer, if you used any specific tenant data chunks, return them in (Anything sorted by tenant_id that you used in your answer. Source_doc and lease_file_path are the same thing if no page number included use 0)::""" + """
        ``` json
        [
        {
            "source_doc": "leaselink/dairy_queen/",
            "pageNumber": 12,
            "highlight_text": "abc-123"  }]```"""
        chat_response = claude.messages.create(
            model=claude_model,
            system=(system_prompt_2),
            messages=[
                {"role": "user", 'content': [
                    {
                        'type': 'text',
                        'text': rephrased_message
                    }
                ]}
            ],
            temperature=0.0,
            max_tokens=8000)
        chat_message = chat_response.content[0].text
        tokens = chat_response.usage
        total_prompt_tokens += tokens.input_tokens
        total_completion_tokens += tokens.output_tokens
        




        json_data = Qdrant_ChatGPT._extract_after_fence(chat_message, "json")

        
        if json_data:
            merged = {}
            for d in json_data:
                signed_url = Supabase_api.get_signed_url(supabase, "lease-docs", d.get('source_doc'))
                viewer_url = ""
                if d.get('pageNumber') is None:
                    viewer_url = f"{signed_url}&highlight_text={d.get('highlight_text')}"
                else:
                    viewer_url = f"{signed_url}#page={d.get('pageNumber')}&highlight_text={d.get('highlight_text')}"
                key = (d.get('source_doc'), d.get('pageNumber'))
                
                if key not in merged:
                    merged[key] = {
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
    print("Final JSON Data:", json.dumps(json_data, indent=2))
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

def combine_by_tenant(records):
    grouped = defaultdict(list)
    final_group = []
    for rec in records:
        tenant_id = rec.get('tenant_id')
        if not tenant_id:
            continue
        grouped[tenant_id].append(rec)
    

    final_group = compress_grouped_tenants(grouped)
    return final_group
    
def compress_grouped_tenants(grouped: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for tenant_id, items in grouped.items():
        tenant_name = None
        for it in items:
            tn = it.get("tenant_name")
            if isinstance(tn, str) and tn.strip():
                tenant_name = tn
                break
        
        compressed_items: List[Dict[str, Any]] = []

        for it in items:
            new_it = {k:v for k, v in it.items() if k not in ('tenant_id', 'tenant_name')}
            compressed_items.append(new_it)
        
        out[tenant_id] = {
            "tenant_id": tenant_id,
            "tenant_name": tenant_name,
            "items": compressed_items
        }
    return out
        

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

        tenantdata, prompt_tokens, completion_tokens, total_square_footage = get_supabase_data(tenants,claude_model, ai_message,  collection_name, message_vector, property_id)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens

        prettytenantdata = combine_by_tenant(tenantdata)
        
        final_response, json_data, prompt_tokens, completion_tokens, embedding_token_count_2 = final_property_chat(message, prettytenantdata, oldData, claude_model, collection_name, total_square_footage)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
        all_embedding_token_count += embedding_token_count_2
        prompt_cost = (all_prompt_tokens / 1000 * 0.003) + (all_embedding_token_count / 1000 * 0.00013)
        completion_cost = all_completion_tokens / 1000 * 0.015
        print("Prompt Cost:", prompt_cost, "Completion Cost:", completion_cost)
        print("Final Response", final_response)

        return final_response, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost, json_data
    except Exception as e:
        print("Error in property chat request:", e)
        prompt_cost = (all_prompt_tokens / 1000 * 0.01) + (all_embedding_token_count / 1000 * 0.00013)
        completion_cost = all_completion_tokens / 1000 * 0.03
        
        return default_response, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost, []
    
