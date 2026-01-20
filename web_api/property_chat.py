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

def get_recent_field(fieldname, data):
    print("Get Recent Field")
    print("Field Name:", fieldname)
    print("Data:", data)
    if not data:
        return None

    def get_sort_date(lease):
        date_str = lease.get("lease_commencement_date") or lease.get("lease_execution_date")
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            return None

    value = None
    file_path = None
    if len(data) > 1:
        # filter leases that have a date AND the field
        valid_leases = [
            lease for lease in data
            if get_sort_date(lease) and lease.get(fieldname) is not None
        ]
        print(valid_leases)

        if not valid_leases:
            return None

        # sort newest → oldest
        sorted_leases = sorted(
            valid_leases,
            key=get_sort_date,
            reverse=True
        )

        value = sorted_leases[0].get(fieldname)
        print("Value:", value)

        file_path = sorted_leases[0].get("lease_file_path")
        print(file_path)
    else:
        value = data[0].get(fieldname)
        file_path = data[0].get("lease_file_path")

    # replicate JS "{a,b,c}" → newline list
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        value = "\n".join(
            item.strip()
            for item in value[1:-1].split(",")
        )

    return value, file_path

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

    need_square_footage = False



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

        if need_square_footage:

            unit_resp = supabase.table('Units').select('*').eq('property_id', property_id).execute()
            unit_data = unit_resp.data or []
            for unit in unit_data:
                print("Unit", unit)
                sf = unit.get('square_footage')
                if sf and isinstance(sf, (int, float)):
                    total_square_footage = (total_square_footage or 0) + sf 

            

        

        if not tenant_id: 
            print("Skipping tenant with missing tenant_id:", tenant)
            continue

        result = tenant_ai_response(tenant_id, tenant['property_management_id'],  collection_name, message_vector, ai_message, claude_model) 
        if not result:
            continue
        message_data, json_data, prompt_tokens, completion_tokens = result
        total_prompt_tokens += (prompt_tokens or 0)
        total_completion_tokens += (completion_tokens or 0)
        data.append({'ai_response': message_data, 'tenant_id': tenant_id, 'lease_file_path': None, "tenant_name": tenant_name, "source_docs": json_data})
    print(data)
    return data, total_prompt_tokens, total_completion_tokens, total_square_footage
        


def tenant_ai_response(tenant_id, company_id, collection_name, message_vector, ai_message, claude_model, top_k=7):
    print("Tenant AI Response")
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=('dense-vector', message_vector),
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
    So only provide the bare minimum information needed to answer the user's question based on the provided excerpts.Your answer will be fed to another ai not the user directly.
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
def get_supabase_column(message, claude_model):
    print("Get Supabase column")
    message_summary = claude.messages.create(
            model=claude_model,
            system=("""
                    You are trying to determine which column(s) from a Supabase table would best help answer the user's question about commercial lease documents.
                    Here is the question: """ +message + """ If the column name has date in the title it is of type date. The rest are text types
                    Here are the available columns in the Supabase table (If it is not in this list. it does not exist. Only choose from this list):
                    [
  {
    "column_name": "lease_execution_date",
    "description": "The day the lease was signed",
  },
  {
    "column_name": "lease_commencement_date",
    "description": "The day that the lease take effect",
  },
  {
    "column_name": "Property_Address",
    "description": "Address of Property",
  },
  {
    "column_name": "suite_identifier",
    "description": "Number or Letter of the suite at a specific address",
  },
  {
    "column_name": "lease_type",
    "description": "There are three options (NNN, Gross, Percentage) They can also be hybrids between these, but normally they are more one one than the others.",
  },
  {
    "column_name": "lease_expiration_date",
    "description": "Day that the lease ends before any options to renew.",,
  },
  {
    "column_name": "lease_term",
    "description": "Length of lease in years/ months",
  },
  {
    "column_name": "base_rent_monthly",
    "description": "Amount of rent for the building before expenses",
  },
  {
    "column_name": "rent_escalation",
    "description": "The rent increase within the current term of the lease",
  },
  {
    "column_name": "security_deposit_amount",
    "description": "The amount the rent has to put as a \"down payment\" to hold there space. Is paid back at the end of the lease if the property is left in good condition.",
  },
  {
    "column_name": "base_rent_psf",
    "description": "The Per Square Foot Base Rent (Annual Rent / SF)",
  },
  {
    "column_name": "base_rent_annually",
    "description": "Base rent amount paid across 12 months",
  },
  {
    "column_name": "operating_expenses_CAM_psf",
    "description": "the PSF estimated amount the tenant is pay for their responsibility towards building expenses.",
  },
  {
    "column_name": "operating_expenses_CAM_monthly",
    "description": "Monthly estimated amount that tenants are pay in all expenses they are responsible for via the lease",
  },
  {
    "column_name": "CAM_Summary",
    "description": "A summary of the Common Area Maintenance and who is responsible for expenses.",
  },
  {
    "column_name": "property_taxes",
    "description": "A summary of who has responsibility to pay the property taxes for the building.",
  },
  {
    "column_name": "insurance_costs",
    "description": "A summary of insurance expectations for both the tenant and the Landlord.",
  },
  {
    "column_name": "tenant_reimbursements",
    "description": "A summary of the system in which the landlord is able to bill the tenant for expenses they initially paid for or the rights in which the tenants have to recoup the money in which they overpaid for building expenses.",
  },
  {
    "column_name": "rent_abatement_end",
    "description": "the date where the tenants rent abatement runs out.",
  },
  {
    "column_name": "rent_commencement_date",
    "description": "Date that rent starts",
  },
  {
    "column_name": "renewal_notice_deadline",
    "description": "The amount of time before the lease expires that the tenant has to let the landlord know they are interested in renewing",
  },
  {
    "column_name": "CAM_start_date",
    "description": "The date in which the tenant is responsible for paying estimated CAM amounts",
  },
  {
    "column_name": "option_exercise_deadlines",
    "description": "The time in which the tenant must have accepted the option to renew",
  },
  {
    "column_name": "delivery_possession_date",
    "description": "The day in which the tenant gets access to the space. (Can sometimes be before the commencement date or after the commencement date if the landlord must do work) I would make this a date but the wording on this is often: \"90 days after the landlord completes their work.\"",
  },
  {
    "column_name": "renewal_options",
    "description": "The amount of options the tenant has and the terms that change upon the commencement of these options.",
  },
  {
    "column_name": "termination_rights",
    "description": "Any terms that allow either party to terminate the lease early.",
  },
  {
    "column_name": "expansion_contraction_rights",
    "description": "The provisions that allow the tenant to grow into more space or shrink out of other space.",
  },
  {
    "column_name": "ROFR_ROFO_clauses",
    "description": "Right of First Refusal clauses or Right of First Offer clauses",
  },
  {
    "column_name": "exclusivity_rights",
    "description": "Blocks the landlord from allowing any competing business.",
  },
  {
    "column_name": "co_tenancy_clauses",
    "description": "Obligations that must be met by the landlord in accordance to other tenants and if not met the consequences.",
  },
  {
    "column_name": "purchase_option",
    "description": "Options the tenant has to purchase the building in within the terms of the lease.",
  },
  {
    "column_name": "rentable_square_footage",
    "description": "Useable SF + share of common areas (hallways, restrooms, etc.)",
  },
  {
    "column_name": "usable_square_footage",
    "description": "The amount of square footage the tenant occupies",
  },
  {
    "column_name": "premises_description",
    "description": "Gives a more general and knowledgeable description of the rentable area",
  },
  {
    "column_name": "parking_allocation",
    "description": "How much parking the tenant gets.",
  },
  {
    "column_name": "storage_additional_space",
    "description": "If any storage is allotted or additional space is allotted to the tenant",
  },
  {
    "column_name": "tenant_maintenance_responsibilities",
    "description": "Tenant's maintenance responsibilities.",
  },
  {
    "column_name": "landlord_maintenance_responsibilities",
    "description": "Landlord's Maintenance Responsibilities",
  },
  {
    "column_name": "hvac_responsibilities",
    "description": "The HVAC responsibilities in detail",
  },
  {
    "column_name": "utility_responsibilities",
    "description": "Utility Responsibility in detail",
  },
  {
    "column_name": "default_and_remedies",
    "description": "The actions and ability to take actions of either part in the event of default by the other.",
  },
  {
    "column_name": "assignment_and_subletting",
    "description": "What is permissible by the tenant if they desire to assign the lease or sublet the space.",
  },
  {
    "column_name": "insurance_requirements",
    "description": "Insurance requirements for the renters of the space. (General or more probably liability)",
  },
  {
    "column_name": "indemnity_clauses",
    "description": "The landlords protection from being held legally liable for anything. (Tenant can't sue landlord)",
  },
  {
    "column_name": "force_majeure",
    "description": "excuses one or both parties from performing their obligations when extraordinary events occur that are outside their control.",
  },
  {
    "column_name": "estoppel_certificate_required",
    "description": "The requirement that tenants answer certain questions in certain occasions. Normally when selling  or refinancing.",
  },
  {
    "column_name": "signage_rights",
    "description": "What signage rights the tenant has",
  },
  {
    "column_name": "permitted_use",
    "description": "What the tenant is allowed to use the space for.",
  },
  {
    "column_name": "exclusive_use_clause",
    "description": "Gives the tenant permission to be the sole operator allowed to do something.",
  },
  {
    "column_name": "guarantor_information",
    "description": "the details about any person or entity that guarantees the tenant’s obligations under the lease.",
  },
  {
    "column_name": "tenant_improvement_allowance",
    "description": "The amount the Landlord gives to the tenant to improve the property for the tenants use.",
  },
  {
    "column_name": "holdover_terms",
    "description": "Terms that apply when the tenant overstays their lease without a renewal.",
  },
  {
    "column_name": "security_access_rights",
    "description": "The rights of security and the limits to the landlords access.",
  },
  {
    "column_name": "landlord_work",
    "description": "Work that is the responsibility of the landlord, normally before the tenant moves in.",
  },
  {
    "column_name": "Tenant_work",
    "description": "The work or improvements that the tenant is held responsible for upon receiving access to the space",
  },
  {
    "column_name": "security_deposit_term",
    "description": "The terms that define security deposit rules.",
  },
  {
    "column_name": "page_count",
    "description": "The number of pages in a document",
  },
  {
    "column_name": "general_liability",
    "description": "General Liability Insurance Requirements",
  },
  {
    "column_name": "property_insurance",
    "description": "Who maintains insurance on the property",

  }
  {
    "column_name": "total_square_footage",
    "description": "Total square footage of the property",}
]
Frame your answer as a comma seperated list of column names that are most relevant to answer the user's question. 
Do not create columns not listed. Do not respond with anything other than the json list of column names.
            """),
            messages=[
                {"role": "user", "content": [{
                    'type': 'text',
                    'text': message}]}
            ],
            temperature=0.0,
            max_tokens=4000
            )
    token_usage = message_summary.usage
    column_response = message_summary.content[0].text
    prompt_tokens = token_usage.input_tokens
    completion_tokens = token_usage.output_tokens
    print("Columns Response:", column_response)
    return column_response, prompt_tokens, completion_tokens
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
    print("Final Query for Tenant:", tenant_id)

    message_vector = OpenAIclient.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        ).data[0].embedding

    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    embedding_token_count = len(encoding.encode(query))

    response = qdrant.search(
        collection_name=collection_name,
        query_vector=('dense-vector', message_vector),
        limit=chunks,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenantid",
                    match=MatchValue(value=tenant_id))
            ]
        ),
    )

    if not response:
        print("No Results found for Final Query for Tenant_ID", tenant_id)
        return []
    print("Final Query Results:", response)
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

        return final_response, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost, json_data
    except Exception as e:
        print("Error in property chat request:", e)
        prompt_cost = (all_prompt_tokens / 1000 * 0.01) + (all_embedding_token_count / 1000 * 0.00013)
        completion_cost = all_completion_tokens / 1000 * 0.03
        
        return default_response, all_prompt_tokens, prompt_cost, all_completion_tokens, completion_cost, []
