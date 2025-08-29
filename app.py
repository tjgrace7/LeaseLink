# ---------- put these caps at the VERY TOP (before heavy imports) ----------
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# --------------------------------------------------------------------------

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
from queue import Queue
import threading
import uuid
import jwt
import logging
import traceback
import sys
import signal
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from qdrant_client import QdrantClient

import common.Supabase_api as Supabase_api
from worker_service import upload_lease_manager
from web_api import Qdrant_ChatGPT

# --------------------------- Logging ---------------------------------
log = logging.getLogger("leaselink-app")
log.setLevel(logging.INFO)

# --------------------------- App Setup --------------------------------
app = FastAPI()
claude_model = "claude-sonnet-4-20250514"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.leaselink.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Globals ----------------------------------
job_status = {}  # { job_id: {status, error, result} }

load_dotenv()
EDGE_SECRET = os.getenv("PYTHON_EDGE_SECRET")
collectionName = os.getenv("QDRANT_COLLECTION", "Lease_Link")

supabase_url = os.getenv("SUPABASE_URL")
JWKS_URL = f"{supabase_url}/auth/v1/keys" if supabase_url else None
SUPABASE_JWT = os.getenv("SUPABASE_JWT")

# API keys
OPENAI_API_KEY = os.getenv("OPEN_AI_PROJECT_KEY")
CLAUDE_API_KEY = os.getenv("Claude_API_KEY")
CRON_SECRET = os.getenv("CRON_SECRET", "")

# Clients (parent process only)
OpenAIclient = OpenAI(api_key=OPENAI_API_KEY)
claude_client = Anthropic(api_key=CLAUDE_API_KEY)
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase_client = Supabase_api.supabase_client_setup()

# Queue / Workers
MAX_WORKERS = int(os.getenv("LEASELINK_MAX_JOB_WORKERS", "4"))         # threads consuming the queue
BACKLOG_TARGET = int(os.getenv("LEASELINK_QUEUE_BACKLOG", "1"))        # keep queue warm up to this size
job_queue = Queue()

# --------------------------- Helpers ----------------------------------
def verify_supabase_jwt(token: str):
    payload = jwt.decode(
        token,
        key=SUPABASE_JWT,
        algorithms=["HS256"],
        audience="authenticated",
        options={"verify_aud": True},
    )
    return payload

def export_lease(job_id, lease_request):
    """
    Thin wrapper that calls upload_lease_manager.load_pdf().
    Internal page work remains parallelized within your worker_service.
    """
    try:
        job_status[job_id]["status"] = "in_progress"
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
            OpenAIclient,        # OpenAI client (parent)
            qdrant_client,       # Qdrant client (parent)
            supabase_client,     # Supabase client (parent)
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
        raise

def enqueue_next_pending_job(limit=4) -> int:
    """
    Claims up to `limit` jobs via Supabase RPC and enqueues them.
    Returns the number of jobs enqueued.
    """
    try:
        # Call your updated RPC that supports a "job_limit" argument
        claim = supabase_client.rpc("claim_next_upload_job", {"job_limit": limit}).execute()
        jobs = claim.data or []
        if not jobs:
            return 0

        count = 0
        for job in jobs:
            job_id = job.get("job_id")
            lease_id = job.get("lease_id")
            if not job_id or not lease_id:
                log.warning(f"Skipping invalid job payload: {job}")
                continue

            # Fetch the lease row
            lease_resp = (
                supabase_client
                .table("lease_documents")
                .select("*")
                .eq("lease_id", lease_id)
                .single()
                .execute()
            )
            lease_row = lease_resp.data
            if not lease_row:
                log.error(f"Lease not found for lease_id={lease_id}")
                continue

            file_path = lease_row.get("lease_file_path")
            if not file_path:
                log.error("Missing file_path on lease")
                continue

            payload = {
                "job_id": job_id,
                "lease_request": {
                    "lease_document_id": lease_id,
                    "tenant_id": lease_row.get("tenant_id"),
                    "file_path": file_path,
                    "user_id": lease_row.get("created_by"),
                    "property_id": lease_row.get("property_id"),
                    "unit_id": lease_row.get("unit_id"),
                    "bucket": "lease-docs",
                    "company_id": lease_row.get("company_id"),
                }
            }

            job_status[job_id] = {"status": "queued", "error": None, "result": None}
            job_queue.put_nowait(payload)
            count += 1

            # reflect "queued" in Upload_Job_Status
            try:
                (
                    supabase_client
                    .table("Upload_Job_Status")
                    .update({"job_info": job_status[job_id]})
                    .eq("job_id", job_id)
                    .execute()
                )
            except Exception as e:
                log.warning(f"Failed to write queued status for {job_id}: {e}")

        return count

    except Exception as e:
        log.error(f"enqueue_next_pending_job error: {e}\n{traceback.format_exc()}")
        return 0

def job_worker():
    while True:
        item = job_queue.get()
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}: {item!r}")

            job_id = item.get("job_id")
            lease_request = item.get("lease_request") or {}

            if not job_id:
                raise ValueError("Missing job_id in queue item")
            if not lease_request:
                raise ValueError("Missing lease_request in queue item")

            print(f"[{job_id}] Starting Job")
            job_status[job_id]["status"] = "in_progress"
            export_lease(job_id, lease_request)

            # If load_pdf didn't set "success", mark it here
            if job_status[job_id].get("status") not in ("success", "error", "extracted"):
                job_status[job_id]["status"] = "success"

        except Exception as e:
            jid = item.get("job_id")
            print(f"[{jid}] Job failed: {e}")
            job_status.setdefault(jid, {})
            job_status[jid]["status"] = "error"
            job_status[jid]["error"] = str(e)

            # best-effort cleanup
            try:
                bucket = lease_request.get("bucket")
                file_path = lease_request.get("file_path")
                upload_lease_manager.Clear_Uploads(jid, bucket, file_path, job_status[jid])
            except Exception:
                pass

        finally:
            # reflect status for this job
            try:
                supabase_client.table("Upload_Job_Status").update(
                    {"job_info": job_status[item.get('job_id')]}
                ).eq("job_id", item.get("job_id")).execute()
            except Exception as e:
                log.warning(f"Failed to write job status in finally: {e}")

            job_queue.task_done()

            # 🔁 Auto-refill the queue from Supabase if we have room
            try:
                while job_queue.qsize() < BACKLOG_TARGET:
                    added = enqueue_next_pending_job()
                    if not added:
                        break  # nothing pending
            except Exception as e:
                log.warning(f"Auto-refill failed: {e}")

# start workers
for _ in range(MAX_WORKERS):
    t = threading.Thread(target=job_worker, daemon=True)
    t.start()

# ---------------------- Global exception/signal hooks -----------------
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

# ------------------------------ Routes --------------------------------
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

        # 2) Throttle if a job is already processing recently
        fifteen_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        busy_resp = (
            supabase_client
            .table("Upload_Job_Status")
            .select("job_id, job_info, updated_at")
            .in_("job_info->>status", ["processing", "in_progress", "extracted"])
            .gte("updated_at", fifteen_min_ago)
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        busy_count = getattr(busy_resp, "count", None) or (len(busy_resp.data or []) if getattr(busy_resp, "data", None) else 0)
        if busy_count > 4:
            return {"ok": True, "skipped": "processing in progress", "busy_count": busy_count}

        # 3) Try to enqueue ONE pending job
        enqueued = enqueue_next_pending_job()
        if not enqueued:
            return {"ok": True, "no_pending": True}

        return JSONResponse({"ok": True, "enqueued": True}, status_code=200)

    except HTTPException:
        log.error("HTTPException in cron_tick:\n%s", traceback.format_exc())
        raise
    except Exception as e:
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
        # enqueue as dict (NOT tuple) to match worker expectations
        job_queue.put_nowait({
            "job_id": job_id,
            "lease_request": lease_request
        })

        # reflect queued status in db
        (
            supabase_client.table("Upload_Job_Status")
            .update({"job_info": job_status[job_id]})
            .eq("job_id", job_id)
            .execute()
        )

        # optional: mark tenant unavailable
        try:
            (
                supabase_client.table("tenant")
                .update({"Available": False})
                .eq("tenant_id", lease_request.get("tenant_id"))
                .execute()
            )
        except Exception as e:
            log.warning(f"Tenant availability update failed: {e}")

    except Exception as e:
        print(f"[{job_id}] Failed to queue job: {e}")
        job_status[job_id] = {"status": "error", "error": str(e), "result": None}
        supabase_client.table("Upload_Job_Status").update({"job_info": job_status[job_id]}).eq("job_id", job_id).execute()
        raise HTTPException(status_code=500, detail=f"Queue failed: {e}")

    return {"status": job_status[job_id], "job_id": job_id}

@app.post("/entity_questions")
async def tenant_send_message(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    content_type: Optional[str] = Header(default=None)
):
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

# ------------------------------ Main ----------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
