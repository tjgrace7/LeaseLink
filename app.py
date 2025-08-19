# ---------- put these caps at the VERY TOP (before heavy imports) ----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# --------------------------------------------------------------------------

from fastapi import FastAPI, Request, Header, HTTPException
from typing import Optional
import uuid
from worker_service import upload_lease_manager
from web_api import Qdrant_ChatGPT
import common.Supabase_api as Supabase_api
from dotenv import load_dotenv
import threading
from openai import OpenAI
from qdrant_client import QdrantClient
import jwt
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import traceback
import signal
from queue import Queue
from anthropic import Anthropic
from datetime import datetime, timedelta, timezone
import logging, traceback

log = logging.getLogger("cron")
log.setLevel(logging.INFO)

app = FastAPI()
claude_model = "claude-sonnet-4-20250514"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.leaselink.ai", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_status = {}
load_dotenv()
EDGE_SECRET = os.getenv("PYTHON_EDGE_SECRET")
collectionName = "Lease_Link"
supabase_url = os.getenv("SUPABASE_URL")
JWKS_URL = f"{supabase_url}/auth/v1/keys"
SUPABASE_JWT = os.getenv("SUPABASE_JWT")

# API keys
OPENAI_API_KEY = os.getenv("OPEN_AI_PROJECT_KEY")
CLAUDE_API_KEY = os.getenv("Claude_API_KEY")
CRON_SECRET = os.getenv('CRON_SECRET', "")

# Clients (used only in the parent process / web process)
OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)
claude_client = Anthropic(api_key=CLAUDE_API_KEY)
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase_client = Supabase_api.supabase_client_setup()

# Bounded parallel jobs at the "lease job" level (keep this modest)
MAX_WORKERS = int(os.getenv("LEASELINK_MAX_JOB_WORKERS", "2"))
job_queue = Queue()

def job_worker():
    while True:
        item = job_queue.get()  # item is a dict
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}: {item!r}")

            job_id = item.get("job_id")
            lease_request = item.get("lease_request") or {}

            if not job_id:
                raise ValueError("Missing job_id in queue item")
            if not lease_request:
                raise ValueError("Missing lease_request in queue item")
            print(job_id)
            print(lease_request)
            print(f"[{job_id}] Starting Job")
            job_status[job_id]["status"] = "in_progress"  # ✅ fixed typo
            export_lease(job_id, lease_request)
        except Exception as e:
            print(f"[{job_id}] Job failed: {e}")
            job_status[job_id]["status"] = "error"
            job_status[job_id]["error"] = str(e)
            bucket = lease_request.get("bucket")
            file_path = lease_request.get("file_path")
            upload_lease_manager.Clear_Uploads(job_id, bucket, file_path, job_status[job_id])
        finally:
            job_queue.task_done()
            supabase_client.table("Upload_Job_Status").update({"job_info": job_status[job_id]}).eq("job_id", job_id).execute()

for _ in range(MAX_WORKERS):
    t = threading.Thread(target=job_worker, daemon=True)
    t.start()

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        return
    print("Unhandled Exception:", "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = handle_exception

def signal_handler(sig, frame):
    print(f"Received Signal: {sig}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def verify_supabase_jwt(token: str):
    payload = jwt.decode(
        token, key=SUPABASE_JWT, algorithms=["HS256"], audience="authenticated", options={"verify_aud": True}
    )
    return payload

def export_lease(job_id, lease_request):
    """
    This stays a thin wrapper that calls your upload_lease_manager.load_pdf().
    The *internal* page work is now process-parallel (see file #2).
    """
    try:
        job_status[job_id]["status"] = "in_progress"  # ✅ keep consistent
        print("Start LeaseLink")

        upload_lease_manager.load_pdf(
            lease_request.get("user_id"),
            lease_request.get("property_id"),
            lease_request.get("unit_id"),
            lease_request.get("tenant_id"),
            lease_request.get("file_path"),
            lease_request.get("lease_document_id"),
            lease_request.get("bucket"),
            lease_request.get("company_id"),
            collectionName,
            # pass only primitives; page workers will re-init their own clients from keys
            OPENAI_API_KEY,
            qdrant_client,
            supabase_client,  # safe to use only in parent process
            job_id,
            job_status[job_id],
            claude_client,
            claude_model,
        )

    except Exception as e:
        bucket = lease_request.get("bucket")
        file_path = lease_request.get("file_path")
        upload_lease_manager.Clear_Uploads(job_id, bucket, file_path, job_status[job_id])
        print(f"Error processing job {job_id}: {e}")

def handle_entity_question(message_request, supabase_client, qdrant_client, OpenAIclient, collectionName):
    try:
        auth_id = message_request.get("auth_id")
        entity_type = message_request.get("entity_type")
        entity_id = message_request.get("entity_id")
        company_id = message_request.get("company_id")
        message = message_request.get("message")
        session_id = message_request.get("session_id")

        if not company_id or not message or not session_id or not auth_id or not entity_type:
            print("Missing required fields")
            return

        filtertype = {
            "tenant": "tenantid",
            "property": "propertyid",
            "unit": "unitid",
            "company": "managementcompany_id",
        }.get(entity_type)

        if not filtertype:
            print(f"Invalid entity_type: {entity_type}")
            return

        oldmessages = Supabase_api.message_get_request(supabase_client, session_id, "entity_questions")

        final_message, prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data = Qdrant_ChatGPT.get_relevant_chunks(
            collectionName, qdrant_client, filtertype, entity_id, company_id, message, OpenAIclient, claude_client, oldmessages, supabase_client, claude_model
        )

        if final_message:
            supabase_client.table("entity_questions").insert(
                [
                    {
                        "entity_id": entity_id,
                        "company_id": company_id,
                        "message": message,
                        "role": "user",
                        "session_id": session_id,
                        "auth_id": auth_id,
                        "message_cost": prompt_cost,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 0,
                        "entity": entity_type,
                    },
                    {
                        "entity_id": entity_id,
                        "company_id": company_id,
                        "message": final_message,
                        "role": "assistant",
                        "session_id": session_id,
                        "auth_id": auth_id,
                        "message_cost": completion_cost,
                        "prompt_tokens": 0,
                        "completion_tokens": completion_tokens,
                        "entity": entity_type,
                        "sources": json_data,
                    },
                ]
            ).execute()
            print("Message successfully processed.")
        else:
            print("GPT returned None.")
    except Exception as e:
        print("Error in threaded message handler:", e)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.head("/")
@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    status = job_status.get(job_id)
    if not status:
        return {"Status": "unknown"}
    return status


@app.post("/internal/cron/tick")
def cron_tick(x_cron_secret: str = Header(default="")):
    try:
        # 1) Auth
        if x_cron_secret != CRON_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # 2) Throttle if a job is already processing in the last 5 min
        five_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        busy_resp = (
            supabase_client
            .table("Upload_Job_Status")
            .select("job_id, job_info, updated_at", count="exact")
            .filter("job_info->>status", "eq", "processing")
            .gte("updated_at", five_min_ago)
            .execute()
        )
        print(five_min_ago)
        busy_count = getattr(busy_resp, "count", None) or (len(busy_resp.data or []) if getattr(busy_resp, "data", None) else 0)
        if busy_count > 0:
            return {"ok": True, "skipped": "processing in progress", "busy_count": busy_count}

        # 3) Claim the next job
        claim = supabase_client.rpc("claim_next_upload_job").execute()
        print(claim)
        job = None
        if claim and getattr(claim, "data", None):
            job = claim.data[0] if isinstance(claim.data, list) else claim.data

        if not job:
            return {"ok": True, "no_pending": True}

        job_id = job.get("job_id")
        print(job_id)
        lease_id = job.get("lease_id")
        if not job_id or not lease_id:
            raise HTTPException(status_code=400, detail=f"RPC payload missing keys: {job}")

        # 4) Fetch the lease row (use .single() to avoid list indexing issues)
        lease_resp = (
            supabase_client
            .table("lease_documents")
            .select("*")
            .eq("lease_id", lease_id)
            .single()
            .execute()
        )
        lease_row = lease_resp.data  # .single() returns an object, not a list
        if not lease_row:
            raise HTTPException(status_code=404, detail=f"Lease not found for lease_id={lease_id}")

        file_path = lease_row.get("lease_file_path")
        if not file_path:
            raise HTTPException(status_code=400, detail="Missing file_path on lease")

        # 5) Enqueue minimal payload (ensure job_queue exists and is thread-safe)
        payload = {
            'job_id': job_id,
            'lease_request':
                {
                    "lease_document_id": lease_id,
                "tenant_id": lease_row.get("tenant_id"),
                "file_path": file_path,
                'user_id': lease_row.get('created_by'),
                'property_id': lease_row.get("property_id"),
                'unit_id':lease_row.get("unit_id"),
                'tenant_id': lease_row.get("tenant_id"),
                'bucket': 'lease-docs',
                'company_id' : lease_row.get("company_id")      }    
        }
        job_status[job_id] = {"status": 'preparing', 'error': 'null', 'results': 'null'}
        
        job_queue.put_nowait(payload)

        # 6) Example update (ensure table/column names are correct in your schema)
        try:
            (
                supabase_client.table("tenant")  # change to "tenants" if that's your actual table
                .update({"Available": False})    # change casing if your column is "available"
                .eq("tenant_id", lease_row.get("tenant_id"))
                .execute()

            )
        except Exception as e:
            # Non-fatal: log and continue
            log.warning(f"Tenant availability update failed: {e}")
        try:
            (
                supabase_client.table('Upload_Job_Status').update({'job_status': job_status[job_id]}).eq('job_id', job_id).execute()
            )
        except Exception as e:
            log.warning(f"Job_Status not uploaded")
        # 7) Record status
        print("Complete")
        return {"ok": True, "job_id": job_id, "status": job_status[job_id]}

    except HTTPException:
        # Let FastAPI return the intended HTTP code, but also log detail
        log.error("HTTPException in cron_tick:\n%s", traceback.format_exc())
        raise
    except Exception as e:
        # Log full traceback for debugging
        log.error("Unhandled exception in cron_tick: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))



    
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
        (
            supabase_client.table("tenant")
            .update({"Available": False})
            .eq("tenant_id", lease_request.get("tenant_id"))
            .execute()  # ✅ actually run it
        )
    except Exception as e:
        print(f"[{job_id}] Failed to queue job: {e}")
        job_status[job_id] = {"status": "error", "error": str(e), "result": None}
        supabase_client.table("Upload_Job_Status").update({"job_info": job_status[job_id]}).eq("job_id", job_id).execute()
        raise HTTPException(status_code=500, detail=f"Queue failed: {e}")

    return {"status": job_status[job_id], "job_id": job_id}

@app.post("/entity_questions")
async def tenant_send_message(request: Request, authorization: Optional[str] = Header(default=None), content_type: Optional[str] = Header(default=None)):
    body = await request.body()
    print("raw body: ", body)
    message_request = await request.json()

    token = authorization.replace("Bearer", "").strip() if authorization else None
    if not token:
        return JSONResponse(status_code=401, content={"error": "Missing or invalid token"})

    auth = verify_supabase_jwt(token)
    if not auth:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if auth["sub"] != message_request.get("auth_id"):
        raise HTTPException(status_code=403, detail="auth_id does not match token")

    threading.Thread(
        target=handle_entity_question,
        args=(message_request, supabase_client, qdrant_client, OpenAIclient, collectionName),
        daemon=True,
    ).start()

    return {"status": "Message is being processed"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
