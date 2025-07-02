from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, ScrollRequest, PayloadSchemaType
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import Supabase_api

#For Updating Qdrant, update function to change filter fields in qdrant as desired
def update_filter_fields():
    load_dotenv()
    print("Update_filter_fields")
    q_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=False
    )
    q_client.create_payload_index(
        collection_name="Test-Leases",
        field_name="managementcompany_id",
        field_schema="keyword"
        
    )
#testing function
def Check_Vectors(q_client, collection_Name):
    response = q_client.scroll(
        collection_name=collection_Name,
        limit = 5,
        with_payload=True,
        with_vectors=True
    )
    for point in response[0]:
        print("ID: ", point.id)
        print("Vector Length:", len(point.vector))
        print("Payload keys:", point.payload.keys())
        print("First 3 vector values:", point.vector[:3])

#Clears entire qdrant collection **FOR TESTING ONLY**
def clear_collection(q_client, collection_Name):
    q_client.delete(
        collection_name=collection_Name,
        points_selector=Filter(must=[])
        
    )
    print("Qdrant cleared")

def extract_json_from_response(response_text: str):
    #Find everything between ```json and ```
    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if not match:
        return None
    
    json_str = match.group(1)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Json parsing error:", e)
        return None
 #Gets data from vector db that was just uploaded for ChatGPT
def get_relevant_chunks(collection_Name, q_client, filtertype1, filterid1, company_id, message, openAI, oldData, supabase_client):
    print("get_relevant chunks")
    now = datetime.now()
    print("GPT rephrase question")
    message_summary = openAI.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"""
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

            """},
            {"role": "user", "content": message}
        ],
        temperature=0.3
    )
    token_usage = message_summary.usage
    prompt_tokens = token_usage.prompt_tokens
    completion_tokens = token_usage.completion_tokens
    input = message_summary.choices[0].message.content
    print("Embed Question")
    message_vector = openAI.embeddings.create(
        input=input,
        model="text-embedding-3-large"
    ).data[0].embedding
    print("Encode question for pricing")
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    embedding_token_count = len(encoding.encode(input))
    print("Qdrant Search")
    try:
        results = q_client.search(
            collection_name = collection_Name,
            query_vector = message_vector,
            limit=10,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[FieldCondition(
                    key=filtertype1,
                    match=MatchValue(value=filterid1)
                ),
                FieldCondition(
                    key="managementcompany_id",
                    match=MatchValue(value=company_id)
                )]
            )   
        )    

        if not results:
            raise ValueError(f"No Results found for tenant_id/company_id together")
        print("Combining Search Results")
        context = "\n\n".join([
        f"source_doc = {r.payload.get('source_doc', 'unknown')}, pageNumber = {r.payload.get('pageNumber', 'N/A')}\n{r.payload['text']}"
        for r in results if "text" in r.payload
        ])
        oldmessages = "\n\n".join([f"message: {data['message']}, role: {data['role']}" for data in oldData if data["message"]])     
        systemprompt = f"""You are a helpful assistant answering questions about lease documents.

{context}

The context above includes a list of content chunks, each labeled with:
- Document Name (source_doc)
- pageNumber
- highlight_id

If two documents provide conflicting information, use the most recent one.

If a user asks a time-based question (e.g., about rent, terms, insurance), use the following as the current date:
**{now}**

---

Here are previous messages related to this tenant:
Each message includes a role:
- "user" means it was written by the property manager
- "assistant" means it was your previous response

{oldmessages}
If no messages are shown, this is the first interaction.

Use these previous messages as additional context to help answer the current question.

The user is a property manager asking about a tenant, property, unit, or other property management issue. The lease information is provided in the context above.

---

Answer the question clearly.

At the end of your answer, if you used any specific context chunks, return them in the following JSON format. The `highlight_text` should be the exact text from the chunk you used in your answer.

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
        print("Messaging ChatGPT")
        chat_response = openAI.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": message}
            ],
            temperature=0.2
        )
        token_usage = chat_response.usage
        prompt_tokens += token_usage.prompt_tokens
        prompt_cost = (prompt_tokens/1000*.01) + (embedding_token_count/1000*.00013)
        completion_tokens += token_usage.completion_tokens
        completion_cost = completion_tokens/1000*.03
        print("total_cost", completion_cost+prompt_cost)
        chat_message = chat_response.choices[0].message.content
        parts = chat_message.split("```json")
        json_data = extract_json_from_response(chat_message)
        if json_data:
            for data in json_data:
                file_path = data["source_doc"]
                print(file_path)
                signed_url = Supabase_api.get_signed_url(supabase_client, "lease-docs", file_path)
                viewer_url = f"{signed_url}#page={data['pageNumber']}&highlight_text={data['highlight_text']}"
                print(viewer_url)
                data["viewer_url"] = viewer_url
            
        final_message = parts[0].strip()
        print(final_message)
        return final_message, prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data
    except Exception as e:
        print("Error:", e)



def get_relevant_chunks_from_lease(collection_Name, q_client, chatGPT, session_id, top_k=30) -> dict:
        #ChatGPT analysis lease to determine lease type, effective, and execution dates prompt below
    query = "Classify this lease and extract key details like term, rent, maintenance, taxes, rent increases, maintenance terms, insurance, CAMS, square-footage, state-of-registration, mailing address, effective date, and execution date"
    print("Get_relevant_chunk_from_lease_inner_function")
    prompt_embed = chatGPT.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    )
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    embedding_token_count = len(encoding.encode(query))
    embeddingcost = embedding_token_count*.00000013
    query_vector = prompt_embed.data[0].embedding
    results = q_client.search(
        collection_name = collection_Name,
        query_vector = query_vector,
        limit = top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(
            must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        )
    )
    if not results:
        raise ValueError(f"No Chunks found for session_id: {session_id}")

    context = "\n\n".join([r.payload.get("text", "") for r in results])
    prompt = f"""
Here is the lease text:
{context}

---

**Task 1:** Classify the document into one of the following types:
- "main_lease"
- "amendment"
- "exhibit"
- "guaranties"
- "assignments"
- "addendum"
- "sublease_agreements"
- "notice_documents"
- "letters_of_work"
- "other"

**Task 2:** Extract the following fields if available:
- effective_date
- execution_date
- term
- current_rent
- rent_increase
- maintenance_terms
- taxes
- insurance
- CAMS
- square_footage
- state_of_registration
- mailing_address
- details

Respond only with a JSON object. Do not add null values. Omit missing fields. Do not include any text outside the JSON object. **Do not add fields that may apply. Only send keys that are listed above. Errors will occur if extra fields send** Dates must be formatted as yyyy/mm/dd
"""
    print("Sending Message")
    chat_response = chatGPT.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a leasing document analyzer. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.2
    )
    try:
        json_start=chat_response.choices[0].message.content.find("{")
        print("json_start:", json_start)
        json_string=chat_response.choices[0].message.content[json_start:]
        print("json_string:", json_string)
        return json.loads(json_string), embeddingcost
    except Exception as e:
        print("Failed to parse JSON:", chat_response.choices[0].message.content)
        raise e