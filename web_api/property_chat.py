from qdrant_client.http.models import  Filter, FieldCondition, MatchValue, SearchParams
from qdrant_client.http import models as rest
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import common.Supabase_api as Supabase_api
from memory_profiler import profile
import posixpath, re
from urllib.parse import quote
from qdrant_client import QdrantClient
from openai import OpenAI
from anthropic import Anthropic
import os


OPENAI_API_KEY = os.getenv("OPEN_AI_PROJECT_KEY")
CLAUDE_API_KEY = os.getenv("Claude_API_KEY")

OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)
claude = Anthropic(api_key=CLAUDE_API_KEY)
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase = Supabase_api.supabase_client_setup()

from datetime import datetime

def get_recent_field(fieldname, data):
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

    if len(data) > 1:
        # filter leases that have a date AND the field
        valid_leases = [
            lease for lease in data
            if get_sort_date(lease) and lease.get(fieldname) is not None
        ]

        if not valid_leases:
            return None

        # sort newest → oldest
        sorted_leases = sorted(
            valid_leases,
            key=get_sort_date,
            reverse=True
        )

        value = sorted_leases[0].get(fieldname)

    else:
        value = data[0].get(fieldname)

    # replicate JS "{a,b,c}" → newline list
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        value = "\n".join(
            item.strip()
            for item in value[1:-1].split(",")
        )

    return value

def get_propertyTenants(property_id, company_id):
    response = supabase.table("Property_Tenant").select("*").eq("property_id", property_id).eq("company_id", company_id).execute()
    tenants = []
    for id in response.data:
        tenants += supabase.table('tenant').select('*').eq('tenant_id', id['tenant_id']).execute()
    print("Tenants:", tenants)
    return tenants

def get_supabase_data(tenants, column_names):
    data = []
    for tenant in tenants:
        tenant_id = tenant['tenant_id']
        response = supabase.table("lease_documents").select(column_names).eq("tenant_id", tenant_id).execute()
        

def tenant_ai_response(tenant_id, company_id, collection_name, message_vector, ai_message, emailCollection, top_k=5):
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=('dense-vector', message_vector),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id)),
                FieldCondition(
                    key="managementcompany_id",
                    match=MatchValue(value=company_id))
            ]
        ),
    )
    emailresults = qdrant.search(
                collection_name=emailCollection,
                query_vector= message_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
                        FieldCondition(key="company_id", match=MatchValue(value=company_id))
                    ]
                )
            )
    now = datetime.now()
    system_prompt = f"""You are a helpful assistant answering questions about lease documents."""


def get_supabase_column(message, claude_model):
    message_summary = claude.messages.create(
            model=claude_model,
            system=(f"""
                    You are trying to determine which column(s) from a Supabase table would best help answer the user's question about commercial lease documents.
                    Here is the question: "{message}" If the column name has date in the title it is of type date. The rest are text types
                    Here are the available columns in the Supabase table:
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
            max_tokens=4000
        )
        token_usage = message_summary.usage
        prompt_tokens = token_usage.input_tokens
        completion_tokens = token_usage.output_tokens
        input = message_summary.content[0].text
        message_vector = OpenAIclient.embeddings.create(
            input=input,
            model="text-embedding-3-large"
        ).data[0].embedding

        return input, message_vector, prompt_tokens, completion_tokens



async def property_chat_request(collection_name, property_id, company_id, message, oldData, claude_model, emailCollection):
    try:
        all_prompt_tokens = 0
        all_completion_tokens = 0
        tenants = get_propertyTenants(property_id, company_id)
        columns, prompt_tokens, completion_tokens = await get_supabase_column(message, claude_model)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens

        await get_supabase_data(tenants, columns)

        ai_message, message_vector, prompt_tokens, completion_tokens = await rephrase_question(message, claude_model)
        all_prompt_tokens += prompt_tokens
        all_completion_tokens += completion_tokens
    except Exception as e:
        print("Error in property chat request:", e)
        return {"error": "Failed to process property chat request."}
