from qdrant_client.http.models import  Filter, FieldCondition, MatchValue
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import Supabase_api
from memory_profiler import profile
from upload_lease_manager import Clear_Uploads


#Clears entire qdrant collection **FOR TESTING ONLY**
def clear_collection(q_client, collection_Name):
    q_client.delete(
        collection_name=collection_Name,
        points_selector=Filter(must=[])
        
    )
    print("Qdrant cleared")

@profile
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
    print("get_relevant_chunks")
    now = datetime.now()

    prompt_tokens = 0
    prompt_cost = 0 
    completion_tokens = 0 
    completion_cost = 0
    # Default return values in case of failure
    default_response = (
        "Sorry, there was an error processing your question. Please try again later.",
    )

    try:
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
        results = q_client.search(
            collection_name=collection_Name,
            query_vector=message_vector,
            limit=10,
            with_payload=True,
            with_vectors=False,
            query_filter=Filter(
                must=[
                    FieldCondition(key=filtertype1, match=MatchValue(value=filterid1)),
                    FieldCondition(key="managementcompany_id", match=MatchValue(value=company_id))
                ]
            )
        )

        if not results:
            raise ValueError("No results found for tenant_id/company_id.")

        print("Combining Search Results")
        context = "\n\n".join([
            f"source_doc = {r.payload.get('source_doc', 'unknown')}, pageNumber = {r.payload.get('pageNumber', 'N/A')}\n{r.payload['text']}"
            for r in results if "text" in r.payload
        ])


    except Exception as e:
        print("Error during preprocessing/Qdrant:", e)
        context = "null"


    oldmessages = "\n\n".join([
        f"message: {data['message']}, role: {data['role']}"
        for data in oldData if data.get("message")
        ])
    try:
        systemprompt = f"""You are a helpful assistant answering questions about lease documents.

{context}

The context above includes a list of content chunks, each labeled with:
- Document Name (source_doc)
- pageNumber
- highlight_id

If two documents provide conflicting information, use the most recent one.
If context is null. Tell the user there was an error retriving lease context. And factor that into your response about the question

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
        prompt_cost = (prompt_tokens / 1000 * 0.01) + (embedding_token_count / 1000 * 0.00013)
        completion_tokens += token_usage.completion_tokens
        completion_cost = completion_tokens / 1000 * 0.03

        print("total_cost", completion_cost + prompt_cost)
        chat_message = chat_response.choices[0].message.content

        parts = chat_message.split("```json")
        final_message = parts[0].strip()

        json_data = extract_json_from_response(chat_message)
        if json_data:
            for data in json_data:
                file_path = data["source_doc"]
                print(file_path)
                signed_url = Supabase_api.get_signed_url(supabase_client, "lease-docs", file_path)
                viewer_url = f"{signed_url}#page={data['pageNumber']}&highlight_text={data['highlight_text']}"
                data["viewer_url"] = viewer_url

        print(final_message)
        return final_message or default_response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data or []

    except Exception as e:
        print("Error in final GPT step:", e)
        return default_response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, []














def get_relevant_chunks_from_lease(collection_Name, q_client, chatGPT, session_id, top_k=30) -> dict:
    try:
        #ChatGPT analysis lease to determine lease type, effective, and execution dates prompt below
        query = "Classify this lease and extract key details like term, rent, maintenance, taxes, rent increases, maintenance terms, insurance, CAMS, square-footage, state-of-registration, mailing address, effective date, and execution date"
        print("Get_relevant_chunk_from_lease_inner_function")
        query_vector = ''
        try:

            prompt_embed = chatGPT.embeddings.create(
                input=query,
                model="text-embedding-3-large"
            )
            encoding = tiktoken.encoding_for_model("text-embedding-3-large")
            embedding_token_count = len(encoding.encode(query))
            embeddingcost = embedding_token_count*.00000013
            query_vector = prompt_embed.data[0].embedding
        except Exception as e:
            print("Error Embedding file", e)
        results = []
        try:
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
        except Exception as e:
            print("Error Getting Response from Qdrant")
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
- lease_execution_date (the Day the lease was signed)
- lease_commencement_date (The Day the Lease takes effect)
- Property_Address
- suite_identifier (The number or letter of the suite without the address)
- lease_type (NNN, Gross, Percentage)
- lease_expiration_date
- lease_term
- base_rent_monthly
- rent_escalation
- security_deposit_amount
- base_rent_psf (The Per Square Foot Base Rent (Annual Rent / SF))
- base_rent_annually
- operating_expenses_CAM_psf (CAM expenses per square foot)
- operating_expenses_CAM_monthly
- CAM_Summary
- property_taxes
- insurance_costs
- tenant_reimbursements
- rent_abatement_end (the date where the tenants rent abatement runs out.)
- rent_commencement_date
- renewal_notice_deadline
- CAM_start_date
- option_exercise_deadlines (The time in which the tenant must have accepted the option to renew)
- delivery_possession_date (The day the tenants may access the space)
- renewal_options
- termination_rights
- expansion_contraction_rights (The provisions that allow the tenant to grow into more space or shrink out of other space.)
- co_tenancy_clauses
- purchase_option
- rentable_square_footage (Useable SF + share of common areas (hallways, restrooms, etc.))
- usable_square_footage (The amount of square footage the tenant occupies)
- premises_description
- parking_allocation
- storage_additional_space
- tenant_maintenance_responsibilities
- landlord_maintenance_responsibilities
- hvac_responsibilities
- utility_responsibilities
- default_and_remedies (The actions and ability to take actions of either part in the event of default by the other.)
- assignment_and_subletting (What is permissible by the tenant if they desire to assign the lease or sublet the space.)
- insurance_requirements
- indemnity_clauses
- compliance_with_laws
- force_majeure
- estoppel_certificate_required (Details)
- signage_rights
- permitted_use
- exclusive_use_clause
- guarantor_information
- tenant_improvement_allowance
- holdover_terms (Terms that apply when the tenant overstays their lease without a renewal.)
- landlord_work
- Tenant_work
- security_deposit_term
- ROFR_ROFO_clauses (Right of First Refusal clauses or Right of First Offer clauses)
- security_access_rights
- exclusivity_rights



Respond only with a JSON object. Do not add null values. Omit missing fields. Do not include any text outside the JSON object. **Do not add fields that may apply. Only send keys that are listed above. Errors will occur if extra fields send** Dates must be formatted as yyyy/mm/dd. Dates not in this format will fail (Omit if not complete)
Items in () describe the item being searched for do not include in json response,

DO NOT CHANGE THE TITLE OF ANY FIELDS

If json key is N/A just exclude
"""
        print("prompt")
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
            token_usage = chat_response.usage
            prompt_tokens = token_usage.prompt_tokens
            prompt_cost = (prompt_tokens/1000*.01) + (embedding_token_count/1000*.00013)
            completion_tokens = token_usage.completion_tokens
            completion_cost = completion_tokens/1000*.03
            print("json_start:", json_start)
            total_cost = prompt_cost + completion_cost + embeddingcost
            json_string=chat_response.choices[0].message.content[json_start:]
            print("json_string:", json_string)
            del prompt_tokens, prompt_cost, completion_tokens, completion_cost, prompt, context, results, query_vector,  
            return json.loads(json_string), total_cost
        except Exception as e:
            print("Failed to parse JSON:", chat_response.choices[0].message.content)
            raise e
    except Exception as e:
        print("Error Parsing Lease", e)
        raise e
