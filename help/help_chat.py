"""
RAG-based help chat for the LeaseLink documentation.

This module answers user questions about how to use the LeaseLink application by
performing a two-step pipeline:

  1. rephrase_question: Uses Claude to rewrite the user's question into a
     semantically precise Qdrant search query.  Claude can also request more than
     the default 3 chunks if the question is broad (signalled via a JSON fence block).

  2. search_qdrant: Embeds the rephrased query with text-embedding-3-small and
     searches the "Source-Docs" Qdrant collection for the most relevant documentation
     chunks.

  3. generate_answer: Feeds the retrieved chunks and conversation history to Claude,
     which produces a plain-text answer and extracts any relevant documentation URLs
     from a JSON fence block in the response.

The public entry point is help_chat(), which chains these steps and returns the
answer, source links, and token cost breakdowns.
"""

import os
from qdrant_client import QdrantClient
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import re
from web_api.Qdrant_ChatGPT import _extract_after_fence

load_dotenv()

CLAUDE_API_KEY = os.getenv("Claude_API_KEY")

claude = Anthropic(api_key=CLAUDE_API_KEY)

OPENAI_API_KEY = os.getenv("OPEN_AI_PROJECT_KEY")

OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)

Qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

# Qdrant client configured with gRPC for lower latency and keepalive settings to
# maintain the connection under long idle periods.
client = QdrantClient(
    url=Qdrant_url,
    api_key=qdrant_key,
    prefer_grpc=True,
    timeout=120.0,
    grpc_options={
        "grpc.keepalive_time_ms": 20000,
        "grpc.keepalive_timeout_ms": 10000,
        "grpc.keepalive_permit_without_calls": True,
    },
)


def rephrase_question(message, claude_model):
        """Rewrite the user's question as a search query optimised for Qdrant semantic search.

        Claude also outputs an optional JSON block requesting a higher top_k when the
        question is broad.  Returns (rephrased_query, prompt_tokens, completion_tokens, top_k).
        """
        message_summary = claude.messages.create(
        model=claude_model,
        system=(f"""
You are a helpful assistant designed to rephrase a users question to search a qdrant vector DB for relevant information about how to use the app Lease Link. The user will asks questions on how to use the app, you will rephrase the question to search a qdrant vector database for the written source documentation.
                
                Some questions may require additional context from qdrant. The default setting for Qdrant is to return 3 relevant source chunks, but if the question is very broad, you can request additional chunks but doing this:

                json
                ---
            """ + """
            {
                total_chunks: numChunks
            }
            ---
            """),

            messages=[
                {"role": "user", "content": [{
                    'type': 'text',
                    'text': message}]}
            ],
            max_tokens=1000
        )
            
        token_usage = message_summary.usage
        prompt_tokens = token_usage.input_tokens
        completion_tokens = token_usage.output_tokens
        input = message_summary.content[0].text

        answer = re.sub(r"```(?:json|emailjson)\s*.*?```", "", input, flags=re.DOTALL|re.IGNORECASE).strip()

        json_data = _extract_after_fence(message_summary.content[0].text, "json")
        top_k = json_data.get("total_chunks", 3) if json_data else 3

        print("Top K for Qdrant Search:", top_k)


        return answer, prompt_tokens, completion_tokens, top_k

def search_qdrant(query, top_k=3):
    """Embed query with text-embedding-3-small and retrieve top_k chunks from Source-Docs.

    Returns (points, embedding_token_count).
    """
    message_vector = OpenAIclient.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding
    
    print("Encode question for pricing")
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    embedding_token_count = len(encoding.encode(query))
    
    search_result = client.query_points(
        collection_name="Source-Docs",
        query=message_vector,
        limit=top_k,
        with_payload=True
    )
    return search_result.points, embedding_token_count

def generate_answer(search_results, user_message, claude_model, old_messages):
    """Generate a plain-text answer using retrieved documentation chunks and conversation history.

    Constructs a system prompt containing the source-doc context, calls Claude, strips
    the JSON fence block from the response, and extracts any source URLs from it.
    Returns (answer, links, prompt_tokens, completion_tokens).
    """
    print ("Search Results", search_results)
    context = "\n\n".join([f"Source Doc: {result.payload.get('data', '')}, url: {result.payload.get('url', '')}"
        for result in search_results])
    prompt = f"""You are a helpful assistant designed to answer questions about how to use the app Lease Link. Use the following source documentation to answer the question. If you don't know the answer, say you don't know. You may take slight inferences from the source documentaion, but don't make up random stuff. EX: if the docs say email integration. Subscription but also say subscription is testing, you can assume they have access to a feature. 
Context: {context}
There may be previous messages with the user, here are those messages and your previous answers: {old_messages} use these to inform context but don't solely rely on them for truth as they may be missing context as well.
Provide the urls to the source documentation in your answer if it is relevant to the question. In Json
""" + """
json
---

links: {
pageName: 'Chat Page',
url:
 'https://leaselink-docs.onrender.com/docs/Lease-Link-Pages/{insert page}
 },
{
pageName: 'Dashboard',
url: 'https://leaselink-docs.onrender.com/docs/Lease-Link-Pages/Dashboard'
}

---

"""
    message_summary = claude.messages.create(
        model=claude_model,
        system=prompt,
            messages=[{"role": "user", "content": [{
                    'type': 'text',
                    'text': user_message}]}
            ],
            max_tokens=1000
        )
    token_usage = message_summary.usage
    prompt_tokens = token_usage.input_tokens
    completion_tokens = token_usage.output_tokens
    answer = message_summary.content[0].text

    answer = re.sub(r"```(?:json|emailjson)\s*.*?```", "", answer, flags=re.DOTALL|re.IGNORECASE).strip()

    json_data = _extract_after_fence(message_summary.content[0].text, "json")

    links = json_data.get("links", []) if json_data else []
    

    return answer, links, prompt_tokens, completion_tokens

def help_chat(user_message, old_messages, claude_model = "claude-sonnet-4-20250514"):
    """Entry point for the help chat pipeline.

    Rephrases the question, searches Qdrant for relevant docs, generates an answer
    with Claude, then aggregates token counts and calculates prompt/completion costs.
    Returns (answer, links, prompt_cost, completion_cost).
    """
    rephrased_query, prompt_tokens, completion_tokens, top_k = rephrase_question(user_message, claude_model)
    search_results, embedding_token_count = search_qdrant(rephrased_query)

    answer, links, answer_prompt_tokens, answer_completion_tokens = generate_answer(search_results, user_message, claude_model, old_messages)
    total_prompt_tokens = prompt_tokens + answer_prompt_tokens
    total_completion_tokens = completion_tokens + answer_completion_tokens

    print("Prompt Tokens:", total_prompt_tokens)
    print("Completion Tokens:", total_completion_tokens)
    print("Embedding Tokens:", embedding_token_count)
    print("Answer", answer)
    print ("Links", links)
    prompt_cost = (total_prompt_tokens / 1000 * 0.01) + (embedding_token_count / 1000 * 0.00013)
    completion_cost = total_completion_tokens / 1000 * 0.03
    return answer, links, prompt_cost, completion_cost

    

