from fastapi import FastAPI, Request, Header, HTTPException
from typing import Optional
import os
import uuid
from worker_service import upload_lease_manager
from web_api import Qdrant_ChatGPT
import common.Supabase_api as Supabase_api
from dotenv import load_dotenv
import threading
from openai import OpenAI
from qdrant_client import QdrantClient
import jwt
import common.Supabase_api as Supabase_api
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # ✅ Add this
import sys
import traceback
import signal
from queue import Queue
import threading


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.leaselink.ai", 'http://localhost:5173'],  # 👈 Add your local dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
job_status = {}
load_dotenv()
EDGE_SECRET = os.getenv("PYTHON_EDGE_SECRET")
collectionName = "Test-Leases"
supabase_url = os.getenv("SUPABASE_URL")
JWKS_URL = f"{supabase_url}/auth/v1/keys"
SUPABASE_JWT = os.getenv("SUPABASE_JWT")


job_queue = Queue()
MAX_WORKERS = 2

def job_worker():
    while True:
        job_id, lease_request = job_queue.get()
        try:
            print(f"[{job_id}] Starting Job")
            job_status[job_id]["status"] = 'in_progress'
            export_lease(job_id, lease_request)
            job_status[job_id]["status"] = "done"
        except Exception as e:
            print(f"[{job_id}] Job failed: {e}")
            job_status[job_id]["status"] = "error"
            job_status[job_id]["error"] = str(e)
            bucket = lease_request.get("bucket")
            file_path = lease_request.get("file_path")
            
            upload_lease_manager.Clear_Uploads(job_id, bucket, file_path, job_status[job_id])
        finally:
            job_queue.task_done()
            
            supabase_client.table('Upload_Job_Status').update({"job_info": job_status[job_id]}).eq('job_id', job_id).execute()

for _ in range(MAX_WORKERS):
    t = threading.Thread(target = job_worker, daemon=True)
    t.start()
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        return
    print("Unhandled Exception:", "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = handle_exception

#Do not remove frame. required for signal handler
def signal_handler(sig, frame):
    print(f"Received Signal: {sig}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


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
            supabase_client,
            job_id,
            job_status[job_id]
        )

    except Exception as e:

        bucket = lease_request.get("bucket")
        file_path = lease_request.get("file_path")
        upload_lease_manager.Clear_Uploads(job_id, bucket, file_path, job_status[job_id])
        print(f"Error processing job {job_id}: {e}")
@app.get("/")
def root():
    return {"message": "API is running"}
@app.head("/")
@app.get('/job-status/{job_id}')
def get_job_status(job_id: str):
    status = job_status.get(job_id)
    if not status:
        return {"Status": "unknown"}
    return status
@app.post("/process-lease")
async def process_file(request: Request, authorization: Optional[str] = Header(default=None)):
    job_id = str(uuid.uuid4())
    job_status[job_id] = {"status": "pending", "error": None, "result": None}
    
    if authorization != f"Bearer {EDGE_SECRET}":
        raise HTTPException(status_code=403, detail="Unauthorized")

    lease_request = await request.json()
    print(f"[{job_id}] lease_request: {lease_request}")

    file_path = lease_request.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="Missing file_path")

    print(f"[{job_id}] Creating thread")
    try:
        job_queue.put_nowait((job_id, lease_request))
    except Exception as e:
        print(f"[{job_id}] Failed to queue job: {e}")
        job_status[job_id] = {"status": "error", "error": str(e), "result": None}
        supabase_client.table('Upload_Job_Status').update({"job_info": job_status[job_id]}).eq('job_id', job_id).execute()
        raise HTTPException(status_code=500, detail=f"Queue failed: {e}")

    return {
        "status": "queued",
        "job_id": job_id
    }

@app.post("/entity_questions")
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
    auth_id = message_request.get("auth_id")
    entity_type = message_request.get("entity_type")
    entity_id = message_request.get("entity_id")

    company_id = message_request.get("company_id")

    message = message_request.get("message")

    session_id = message_request.get("session_id")
    print(f"Auth ID: {auth_id}")
    print(f"entity Id: {entity_id}")
    print(f"entity type: {entity_type}")
    print(f"Company Id: {company_id}")
    print(f"Message: {message}")
    if not company_id or not message or not session_id or not auth_id or not entity_type:
        raise HTTPException(status_code=400, detail="Bad Request")
    if auth["sub"] != auth_id:
        raise HTTPException(status_code=403, detail="auth_id does not match token")
    try:
        filtertype = ""
        if entity_type == "tenant":
            filtertype = "tenantid"
        elif entity_type == "property":
            filtertype = "propertyid"
        elif entity_type == "unit":
            filtertype = "unitid"
        elif entity_type == "company":
            filtertype = "managementcompany_id"
        oldmessages = Supabase_api.message_get_request(supabase_client, session_id, "entity_questions")
        print(entity_type)
        print(filtertype)
        final_message,  prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data = Qdrant_ChatGPT.get_relevant_chunks(collectionName, qdrant_client, filtertype, entity_id, company_id, message, OpenAIclient, oldmessages, supabase_client)
        if final_message != None:
            supabase_client.table("entity_questions").insert([
                {
                    "entity_id": entity_id,
                    "company_id": company_id,
                    "message": message,
                    "role": 'user',
                    "session_id": session_id,
                    "auth_id": auth_id,
                    "message_cost": prompt_cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0,
                    "entity": entity_type
                }]).execute()
            supabase_client.table('entity_questions').insert([{
                    "entity_id": entity_id,
                    "company_id": company_id,
                    "message": final_message,
                    "role": 'assistant',
                    "session_id": session_id,
                    "auth_id": auth_id,
                    "message_cost": completion_cost,
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens, 
                    "entity": entity_type,
                    "sources": json_data
                }
            ]).execute()


            print("Success")
            return {
                "response": final_message,
                "session_id": session_id,
                "pdf_reference(s)": json_data
                
            }
        else:
            raise HTTPException(status_code=500, detail="GPT Response Failure")
    except Exception as e:
        print("chatGPT message failure:", e)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
