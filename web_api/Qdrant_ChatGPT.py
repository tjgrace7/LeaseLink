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


def underscorize(s: str) -> str:
    return re.sub(r'\s+', '_', s.strip())


    
@profile
def _extract_braced_json(text: str, start_idx: int):
    """
    Starting at start_idx, find the first { or [ and extract the first COMPLETE
    JSON object/array by counting braces/brackets. Handles strings/escapes.
    Returns (parsed_obj, end_pos) or (None, None) if not found/parseable.
    """
    n = len(text)
    # find first opening brace/bracket
    while start_idx < n and text[start_idx] not in '{[':
        start_idx += 1
    if start_idx >= n:
        return None, None

    open_char = text[start_idx]
    close_char = '}' if open_char == '{' else ']'
    depth = 0
    i = start_idx
    in_string = False
    escape = False

    while i < n:
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    # inclusive slice [start_idx:i]
                    raw = text[start_idx:i+1]
                    try:
                        return json.loads(raw), i+1
                    except json.JSONDecodeError:
                        # If parse fails, still return substring for debugging
                        return None, i+1
        i += 1

    return None, None

def _extract_after_fence(response_text: str, fence_name: str):
    """
    Find ```<fence_name>, then extract the first COMPLETE JSON value ({} or [])
    following it by brace balancing. Ignores any prose after.
    """
    print("Extracting after fence:", fence_name)
    if not isinstance(response_text, str):
        response_text = str(response_text)

    # Locate the fence start
    fence_pat = re.compile(rf"```{re.escape(fence_name)}\b", re.IGNORECASE)
    m = fence_pat.search(response_text)
    if not m:
        return None

    # Move to first char after the fence line
    # Allow optional whitespace/newline after fence
    i = m.end()
    # skip any whitespace/newlines
    while i < len(response_text) and response_text[i] in ' \t\r\n':
        i += 1

    # Extract a complete JSON value
    obj, endpos = _extract_braced_json(response_text, i)
    if obj is not None:
        return obj

    # Fallback: try to capture fenced block with closing ```
    # and json.loads() the largest plausible block inside
    block_match = re.search(rf"```{re.escape(fence_name)}\s*(.*?)```", response_text, re.IGNORECASE | re.DOTALL)
    if block_match:
        body = block_match.group(1).strip()
        # Try a direct load first (AI might already return valid JSON)
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # Try to cut at the first balanced structure within the body
            obj2, _ = _extract_braced_json(body, 0)
            return obj2

    return None

 #Gets data from vector db that was just uploaded for ChatGPT
def get_relevant_chunks(collection_Name, q_client,  filterid1, company_id, message, openAI, claude, oldData, supabase_client, claude_model, emailCollection, unit_id = ""):
    print("get_relevant_chunks")
    now = datetime.now()
    prompt_tokens = 0
    prompt_cost = 0 
    completion_tokens = 0 
    completion_cost = 0
    res = supabase_client.table('Property_Management_Companies').select("*").eq("company_id", company_id).limit(1).execute()
    print("Company Info:", res)
    company = res.data[0]
    # Default return values in case of failure
    default_response = (
        "Sorry, there was an error processing your question. Please try again later.",
    )
    if company["Base_Function"] != True:
        return 
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
        print("Unit Id", unit_id)
        print("Qdrant Search")
        results = []
        print("Company ID", company_id)
        if unit_id== None:
            print("Unit Id Empty")
            print("Tenant Id:", filterid1)

            results = q_client.search(
                collection_name=collection_Name,
                query_vector=("dense_vector", message_vector),
                limit=30,
                with_payload=True,
                with_vectors=False,
                query_filter=Filter(
                    must=[
                        FieldCondition(key='tenantid', match=MatchValue(value=filterid1)),
                        FieldCondition(key="managementcompany_id", match=MatchValue(value=company_id))
                    ]
                )
            )
        else:
            print("Tenant Id:", filterid1)

            results = q_client.search(
                collection_name=collection_Name,
                query_vector=("dense_vector", message_vector),
                limit=30,
                with_payload=True,
                with_vectors=False,
                query_filter=Filter(
                    must=[
                        FieldCondition(key='tenantid', match=MatchValue(value=filterid1)),
                        FieldCondition(key="managementcompany_id", match=MatchValue(value=company_id)),
                        FieldCondition(key='unitid', match=MatchValue(value=unit_id))
                    ]
                )
            )
        print(len(results))
        if not results:
            print("No results found for tenant_id/company_id.")

        print("Combining Search Results")
        if results:
            context = "\n\n".join([
                f"source_doc = {r.payload.get('source_doc', 'unknown')}, pageNumber = {r.payload.get('pageNumber', 'N/A')})\n{r.payload['text']}"
                for r in results if "text" in r.payload
            ])
        else: 
            context = "null"
        emailscript = ""
        if company["Email_Function"]:

            emailresults = q_client.search(
                collection_name=emailCollection,
                query_vector= message_vector,
                limit=5,
                with_payload=True,
                with_vectors=False,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="tenant_id", match=MatchValue(value=filterid1)),
                        FieldCondition(key="company_id", match=MatchValue(value=company_id))
                    ]
                )
            )


            emailcontext = "\n\n".join([
                f"email_body = {e.payload.get('body', 'unknown')}, sender = {e.payload.get('Sender_Name', 'unknown')}, subject = {e.payload.get('subject')}"
                for e in emailresults

            ])
            emailscript =f"""
            ---

            Here are emails with contacts of the tenant. 

            If the referenced email context applies use that in your reference. Tell us who the email is from and the subject

            {emailcontext}
            ---

            If there are emails that you reference in your response, use this as your email bracket source
            ```emailjson
            [
            Curly Bracket
                "Sender_Name": "Jayton Taylor",
                "subject": "App issues",
                "body": "email integration issues..."
            Curly Bracket Close
            ]
            """
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

Many Time Based Questions will reference documents that say term between September 2021 - August 2025

If it is a Day in July 2025. That falls within that period. If the month and Year are outside that date and time. It does not fall within that period.
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

Here are emails with contacts of the tenant. 

If the referenced email context applies use that in your reference. Tell us who the email is from and the subject

---

Answer the question clearly.

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

{emailscript}
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
            temperature=0.0,
            max_tokens=4000
        )
        token_usage = chat_response.usage
        prompt_tokens += token_usage.input_tokens
        prompt_cost = (prompt_tokens / 1000 * 0.003) + (embedding_token_count / 1000 * 0.00013)
        completion_tokens += token_usage.output_tokens
        completion_cost = completion_tokens / 1000 * 0.015

        print("total_cost", completion_cost + prompt_cost)
        chat_message = chat_response.content[0].text
        final_message = re.sub(r"```(?:json|emailjson)\s*.*?```", "", chat_message, flags=re.DOTALL|re.IGNORECASE).strip()




        json_data = _extract_after_fence(chat_message, "json")
        email_data = []
        if company["Email_Function"]:
            email_data = _extract_after_fence(chat_message, "emailjson")
        
        if json_data:
            merged = {}
            for d in json_data:
                key = (d.get('source_doc'), d.get('pageNumber'))
                signed_url = Supabase_api.get_signed_url(supabase_client, "lease-docs", d.get('source_doc'))
                if key not in merged:
                    merged[key] = {
                        "source_doc": d.get("source_doc"),
                        "pageNumber": d.get('pageNumber'),
                        "highlight_text": d.get('highlight_text', ""),
                        "viewer_url": f"{signed_url}#page={d.get('pageNumber')}&highlight_text={d.get('highlight_text')}"
                    }

                else:
                    ht = d.get("highlight_text", "")
                    if ht and ht not in merged[key]['highlight_text']:
                        merged[key]['highlight_text'] = (merged[key]['highlight_text'] + " | " + ht).strip(" |")
                json_data = list(merged.values())

        return final_message or default_response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data or [], email_data or []

    except Exception as e:
        print("Error in final GPT step:", e)
        return default_response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, []
