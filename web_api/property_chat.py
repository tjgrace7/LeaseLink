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

def get_propertyTenants(property_id, company_id):
    response = supabase.table("Property_Tenant").select("*").eq("property_id", property_id).eq("company_id", company_id).execute()
    tenants = []
    for id in response.data:
        tenants += supabase.table('tenant').select('*').eq('tenant_id', id['tenant_id']).execute()
    print("Tenants:", tenants)
    return tenants

def tenant_ai_response(tenant_id, company_id, collection_name, message_vector, ai_message, top_k=5):
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

        return input, message_vector prompt_tokens, completion_tokens



def property_chat_request(collection_name, property_id, company_id, message, oldData, claude_model, emailCollection):
    try:
        tenants = get_propertyTenants(property_id, company_id)
        ai_message, message_vector, prompt_tokens, completion_tokens = rephrase_question(message, claude_model)
        

    except Exception as e:
        print("Error in property chat request:", e)
        return {"error": "Failed to process property chat request."}
