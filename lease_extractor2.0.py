from qdrant_client.http.models import  Filter, FieldCondition, MatchValue
import json
from dotenv import load_dotenv
import tiktoken
from qdrant_client import QdrantClient
from openai import OpenAI
from datetime import datetime

from memory_profiler import profile

session = "e8591474-29d1-4eec-a5f0-a2f8d400b974"
qdrant =  QdrantClient(
    url = "https://3ecddab7-3429-41fc-9acd-7f555d763f3e.us-west-2-0.aws.cloud.qdrant.io:6333",
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.iFAxDJ2i34RExukQycaIB_ytJySH7JC4aZ21hOxf8Rc"
)
collection = "Test-Leases"
OpenAIclient = OpenAI(api_key="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")

def trim_chunks(chunks, max_tokens=3000):
    selected = []
    total_tokens = 0
    for chunk in chunks:
        tokens = len(chunk.split())  # rough estimate
        if total_tokens + tokens > max_tokens:
            break
        selected.append(chunk)
        total_tokens += tokens
    print(total_tokens)
    return selected

def costCalculator(token_usage, search_query):
        encoding = tiktoken.encoding_for_model('text-embedding-3-large')
        embedding_token_count = len(encoding.encode(search_query))
        embeddingcost = embedding_token_count*.00000013
        prompt_tokens = token_usage.prompt_tokens
        prompt_cost = (prompt_tokens/1000*.01)

        completion_tokens = token_usage.completion_tokens
        completion_cost = completion_tokens/1000*.03

        total_cost = prompt_cost+completion_cost+embeddingcost
        return total_cost


def lease_extractor(query, chatGPT, q_client, collection_Name, session_id, mainFilter, sideFilter1, sideFilter2, top_k=30):
    try:
        
        prompt_embed = chatGPT.embeddings.create(
            input=query,
            model='text-embedding-3-large'
        )
        filter = Filter(
            must = [
                FieldCondition(key='session_id', match=MatchValue(value=session_id)),
                FieldCondition(key='embedding_class', match=MatchValue(value=mainFilter))
            ]
        )
        query_vector = prompt_embed.data[0].embedding
        results = q_client.query_points(
            collection_name = collection_Name,
            query_vector = query_vector,
            limit = top_k,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[FieldCondition(key='session_id', match=MatchValue(value=session_id))]
            )
        )
        
        if not results:
            raise ValueError(f"No Chunks found for session_id", {session_id})
        top_chunks = [point.payload['text'] for point in results]
        context = trim_chunks(top_chunks)
        
        finalprompt = f"{context} \n\n {query}"
        with open("Chunks.txt", 'w', encoding='utf-8') as f:
            f.write(finalprompt)
        now = datetime.now()
        chat_response = chatGPT.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': f"""You are a leasing document analyzer. Respond only with a JSON object. Do not add null values. Omit missing fields. Do not include any text outside the JSON object. **Do not add fields that may apply. Only send keys that are listed above. Errors will occur if extra fields send** Dates must be formatted as yyyy/mm/dd. Dates not in this format will fail (Omit if not complete)
Items in () describe the item being searched for. Don't include anything inside the () in the json key.
If the json key description closely matches with a lease item, but not quite use it. Do not add keys or columns
DO NOT CHANGE THE TITLE OF ANY FIELD
                 When asked about current Rent or Dates compare dates to current date: {now}
                 """},

                {'role': 'user', 'content': finalprompt}
            ],
            temperature = 0.1
        )
        json_start=chat_response.choices[0].message.content.find('{')
        json_string = chat_response.choices[0].message.content[json_start:]
        total_cost = costCalculator(chat_response.usage, query)
        return json.loads(json_string), total_cost
    except Exception as e:
        print("Error Extracting Lease", e)
        raise e
def get_relevant_chunks_from_lease(collection_Name, q_client, chatGPT, session_id) -> dict:
    
    query2 = """Extract key details from lease like:
    Do calculations as necessary
    -base_rent_monthly (Price of rent per month for the building before expenses)
    -rent_escalation (Give details on the rent schedule during the initial lease. List by date and amount.) 
    -security_deposit_amount (The amount the rent has to put as a "down payment" to hold there space. Is paid back at the end of the lease if the property is left in good condition.) 
    -base_rent_psf (The Per Square Foot price for Base Rent (Calculate: Annual Rent / SF)) 
    -base_rent_annually (Calculate: Base rent amount paid across 12 months) 
    -operating_expenses_CAM_psf (CAM + operating expenses per square foot. That includes taxes and Insurance (Calculate: monthly/ SF)
    -operating_expenses_CAM_monthly (Monthly estimated amount that tenants are pay in all expenses they are responsible for via the lease. That includes taxes and insurance monthly)  
    -property_taxes (A summary of who has responsibility to pay the property taxes for the building.) 
    -insurance_costs (A summary of insurance expectations for both the tenant and the Landlord.) 
    -CAM_Summary (Make note of any operating expenses not allowed to be charged back to the tenant.) 
    -tenant_reimbursements (A summary of the system in which the landlord is able to bill the tenant for expenses they initially paid for or the rights in which the tenants have to recoup the money in which they overpaid for building expenses.) 
    -insurance_requirements (Insurance requirements for the renters of the space. (General or more probably liability)) 
    -lease_commencement_date (The Day the Lease takes effect, yyyy/mm/dd force into format) 
    -lease_expiration_date (Day that the lease ends before any options to renew. Use the formula lease commencement date + term if required. Use yyyy/mm/dd force into format) 
    -delivery_possession_date (The day the tenants may access the space, yyyy/mm/dd) 
    -CAM_start_date (The date in which the tenant is responsible for paying estimated CAM amounts, yyyy/mm/dd force into format) 
    -rent_abatement_end (the date where the tenants rent abatement runs out. Format yyyy/mm/dd) 
    -rent_commencement_date (Date that rent starts, yyyy/mm/dd force into format)
    """
    financial, financial_cost = lease_extractor(query2, chatGPT, q_client, collection_Name, session_id, 'rent', 'CAM', 'taxes')
    query3 = """Extract Key Details from lease like:
    Do calculations as necessary
    -Property_Address (The listed address of the property) 
    -suite_identifier (The number or letter of the suite without the address if applicable) 
    -lease_term (Length of lease term in months)
    -renewal_notice_deadline (The amount of time before the lease expires that the tenant has to let the landlord know they are interested in renewing) 
    -option_exercise_deadlines (The time in which the tenant must have accepted the option to renew) 
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
    -guarantor_information (the details about any person or entity that guarantees the tenant’s obligations under the lease.) 
    -tenant_improvement_allowance (The amount the Landlord gives to the tenant to improve the property for the tenants use.) 
    -holdover_terms (Terms that apply when the tenant overstays their lease without a renewal.) 
    -landlord_work (Work that is the responsibility of the landlord, normally before the tenant moves in.) 
    -Tenant_work (The work or improvements that the tenant is held responsible if applicable) 
    -security_deposit_term (The terms that define security deposit rules.) 
    -ROFR_ROFO_clauses (Right of First Refusal clauses or Right of First Offer clauses) 
    -security_access_rights (The rights of security and the limits to the landlords access.) 
    -exclusivity_rights (Blocks the landlord from allowing any competing business.)
    """
    term, term_cost = lease_extractor(query3, chatGPT, q_client, collection_Name, session_id,'','','')
    combined = { **financial, **term}
    total_cost = term_cost + financial_cost 
    print (total_cost)
    with open("Upload Data.txt", 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)

get_relevant_chunks_from_lease(collection, qdrant, OpenAIclient, session)
