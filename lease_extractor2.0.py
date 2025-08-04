from qdrant_client.http.models import  Filter, FieldCondition, MatchValue
import json
from dotenv import load_dotenv
import tiktoken
from qdrant_client import QdrantClient
from openai import OpenAI

from memory_profiler import profile

session = "bac5ad08-5300-4605-aa86-957b8da41096"
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
        results = q_client.search(
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
        chat_response = chatGPT.chat.completions.create(
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': """You are a leasing document analyzer. Respond only with a JSON object. Do not add null values. Omit missing fields. Do not include any text outside the JSON object. **Do not add fields that may apply. Only send keys that are listed above. Errors will occur if extra fields send** Dates must be formatted as yyyy/mm/dd. Dates not in this format will fail (Omit if not complete)
Items in () describe the item being searched for. Don't include anything inside the () in the json key.
If the json key description closely matches with a lease item, but not quite use it. Do not add keys or columns
DO NOT CHANGE THE TITLE OF ANY FIELDS"""},

                {'role': 'user', 'content': finalprompt}
            ],
            temperature = 0.1
        )
        print(chat_response)
        json_start=chat_response.choices[0].message.content.find('{')
        json_string = chat_response.choices[0].message.content[json_start:]
        total_cost = costCalculator(chat_response.usage, query)
        return json.loads(json_string), total_cost
    except Exception as e:
        print("Error Extracting Lease", e)
        raise e
def get_relevant_chunks_from_lease(collection_Name, q_client, chatGPT, session_id) -> dict:
    
    query1 = """Extract key details from lease like:
    -lease_execution_date (the Day the lease was signe, yyyy/mm/dd force into format) 
    -lease_commencement_date (The Day the Lease takes effect, yyyy/mm/dd force into format) 
    -lease_expiration_date (Day that the lease ends before any options to renew. Use the formula lease commencement date + term if required. Use yyyy/mm/dd force into format) 
    -delivery_possession_date (The day the tenants may access the space, yyyy/mm/dd) 
    -CAM_start_date (The date in which the tenant is responsible for paying estimated CAM amounts, yyyy/mm/dd force into format) 
    -rent_abatement_end (the date where the tenants rent abatement runs out. Format yyyy/mm/dd) 
    -rent_commencement_date (Date that rent starts, yyyy/mm/dd force into format)
    """
    #dates, date_cost = lease_extractor(query1, chatGPT, q_client, collection_Name, session_id)
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
    """
    costs, costs_cost = lease_extractor(query2, chatGPT, q_client, collection_Name, session_id, 'rent', 'CAM', 'taxes')
    print(costs)
    print (costs_cost)

get_relevant_chunks_from_lease(collection, qdrant, OpenAIclient, session)
