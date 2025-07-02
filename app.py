from fastapi import FastAPI, Request, Header, HTTPException
from typing import Optional
import os
import uuid
import upload_lease_manager
import Qdrant_ChatGPT
import Supabase_api
from dotenv import load_dotenv
import threading
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from qdrant_client.models import PayloadSchemaType
from supabase import create_client
from supabase.lib.client_options import ClientOptions
import jwt
import Supabase_api

app = FastAPI()
job_status = {}
load_dotenv()
EDGE_SECRET = os.getenv("PYTHON_EDGE_SECRET")
collectionName = "Test-Leases"
supabase_url = os.getenv("SUPABASE_URL")
JWKS_URL = f"{supabase_url}/auth/v1/keys"
SUPABASE_JWT = os.getenv("SUPABASE_JWT")

#Connects OpenAI api key
OpenAIclient = OpenAI(api_key=os.getenv("OPEN_AI_PROJECT_KEY"))
#Sets up qdrant_client for easy access
qdrant_client = QdrantClient(
    url = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY")
)
supabase_client = Supabase_api.supabase_client_setup()
def verify_supabase_jwt(token: str):

    payload = jwt.decode(
        token,
        key=SUPABASE_JWT,
        algorithms=["HS256"],
        audience="authenticated",
        options={"verify_aud": True}
    )
    return payload
def export_lease(job_id, lease_request):
    try:

        job_status[job_id]["status"] = "in_progess"
        #Continues embedding and uploading on seperate thread
        print("Start LeaseLink")
        upload_lease_manager.load_pdf(
            lease_request.get("user_id"),         # 👈 Use .get() instead of dot
            lease_request.get("property_id"),
            lease_request.get("unit_id"),
            lease_request.get("tenant_id"),
            lease_request.get("file_path"),
            lease_request.get("lease_document_id"),
            lease_request.get("bucket"),
            lease_request.get("company_id"),
            collectionName,
            OpenAIclient,
            qdrant_client,
            supabase_client
        )

        job_status[job_id] = "done"
    except Exception as e:

        lease_id = lease_request.get("lease_document_id")
        bucket = lease_request.get("bucket")
        file_path = lease_request.get("file_path")
        upload_lease_manager.Clear_Uploads(lease_id, bucket, file_path, e)
        #Add Database update with error status
        job_status[job_id] = f"error: {str(e)}"
        print(f"Error processing job {job_id}: {e}")

@app.post("/process-lease")
async def process_file(request: Request, authorization: Optional[str]=Header(default=None)):
    job_id = str(uuid.uuid4())
    job_status[job_id] = {"status": "pending", "error": None, "result": None}
    if authorization != f"Bearer {EDGE_SECRET}":
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    lease_request = await request.json()
    file_path = lease_request.get("file_path")
    if not file_path:    
        raise HTTPException(status_code=400, datail="Missing file_path")

    thread = threading.Thread(target=export_lease, args=(job_id, lease_request))
    thread.start()
    print(thread)
    return {
        "status": "processing_started",
        "job_id": job_id
    }


@app.post("/tenant_questions")
async def tenant_send_message(request: Request, authorization: Optional[str]=Header(default=None), content_type: Optional[str] = Header(default=None)):
    body = await request.body()
    print("raw body: ", body)
    message_request = await request.json()

    token = authorization.replace("Bearer", "").strip() if authorization else None
    if not token:
        return JSONResponse(status_code=401, content={"error": "Missing or invalid token"})

    auth = verify_supabase_jwt(token)

    if not auth:
        raise HTTPException(status_code=403, detail="Unauthorized")
    user_question_id = message_request.get("user_question_id")
    response_question_id = message_request.get("response_question_id")
    tenant_id = message_request.get("tenant_id")

    company_id = message_request.get("company_id")

    message = message_request.get("message")

    session_id = message_request.get("session_id")

    if not tenant_id or not company_id or not message or not session_id or not user_question_id or not response_question_id:
        raise HTTPException(status_code=400, detail="Bad Request")
    try:
        oldmessages = Supabase_api.message_get_request(supabase_client, session_id, "tenant_questions")
        final_message,  prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data = Qdrant_ChatGPT.get_relevant_chunks(collectionName, qdrant_client, "tenantid", tenant_id, company_id, message, OpenAIclient, oldmessages, supabase_client)
        if final_message != None:
            
            supabase_client.table("tenant_questions").update([
                {
                    "message_cost": prompt_cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0   
                }
            ]).eq("tenant_question_id", user_question_id).execute()
            supabase_client.table("tenant_questions").update([
                {

                    "message": final_message,
                    "message_cost": completion_cost,
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "sources": json_data
                }
            ]).eq("tenant_question_id", response_question_id).execute()


                
            return {
                "response": final_message,
                "session_id": session_id,
                "pdf_reference(s)": json_data
                
            }
    except Exception as e:
        print("chatGPT message failure:", e)