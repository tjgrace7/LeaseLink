from qdrant_client.http.models import  Filter, FieldCondition, MatchValue
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import common.Supabase_api as Supabase_api
from memory_profiler import profile





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
def get_relevant_chunks(collection_Name, q_client, filtertype1, filterid1, company_id, message, openAI, claude, oldData, supabase_client, claude_model):
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
                    'text': message}]}
            ],
            temperature=0.0,
            max_tokens=4000
        )
        token_usage = message_summary.usage
        prompt_tokens = token_usage.input_tokens
        completion_tokens = token_usage.output_tokens
        input = message_summary.content[0].text

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
            limit=20,
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
        chat_response = claude.messages.create(
            model=claude_model,
            system=(systemprompt),
            messages=[
                {"role": "user", "content": [
                    {
                        'type': 'text',
                        'text': message
                    }]}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        token_usage = chat_response.usage
        prompt_tokens += token_usage.input_tokens
        prompt_cost = (prompt_tokens / 1000 * 0.01) + (embedding_token_count / 1000 * 0.00013)
        completion_tokens += token_usage.output_tokens
        completion_cost = completion_tokens / 1000 * 0.03

        print("total_cost", completion_cost + prompt_cost)
        chat_message = chat_response.content[0].text

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