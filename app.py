"""
LeaseLink FastAPI server — entry point for all API routes.

This module bootstraps the application and wires together all subsystems:

  - CORS middleware configured for the production domain and local development.
  - In-memory job_status dict and a thread-safe Queue for lease upload jobs.
  - A pool of MAX_WORKERS daemon threads (job_worker) that consume from the queue,
    call upload_lease_manager.load_pdf, and auto-refill the queue up to BACKLOG_TARGET
    jobs from Supabase when they finish.
  - JWT authentication against Supabase using the HS256 SUPABASE_JWT secret.
  - HMAC-SHA256 signature verification for internal Edge Function callbacks.

Routes:
  GET  /                              Health check (plain text "ok").
  GET  /health                        Health check (JSON).
  GET  /job-status/{job_id}           Returns the in-memory job_status dict for a job.
  POST /internal/cron/tick            Cron endpoint — claims pending jobs from Supabase
                                      and fills the worker queue.
  POST /firstLease                    Synchronous first-lease upload (bypasses queue).
  POST /refresh_tenant                Re-runs field extraction for an existing tenant.
  POST /entity_questions              Tenant or property chat question (async, threaded).
  POST /help                          Help documentation chat (async, threaded).
  GET  /api/integrations/email/start  Starts OAuth flow for Gmail or Outlook.
  GET  /api/gmail/oauth/callback      OAuth callback for Google.
  GET  /api/outlook/oauth/callback    OAuth callback for Microsoft.
  POST /api/email/resync              Re-syncs email for an existing integration.
  POST /api/email/new_contact         Syncs emails for a newly added tenant contact.
  POST /api/integrations/email/disconnect  Removes an email integration.
"""

# ---------- put these caps at the VERY TOP (before heavy imports) ----------
print("BOOT 1: module import Start", flush=True)
import os, urllib.parse, secrets, httpx, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# --------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

from email_integration import PKCE
from typing import Optional
from queue import Queue
import threading

import jwt
import logging
import traceback
import sys
import signal
import time
from starlette.concurrency import run_in_threadpool
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from anthropic import Anthropic
from qdrant_client import QdrantClient
from fastapi.responses import PlainTextResponse

import common.Supabase_api as Supabase_api
from worker_service import upload_lease_manager, final_check
from web_api import Qdrant_ChatGPT
from web_api import property_chat
from email_integration import email_integration
import hmac
import hashlib
from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks, Response
import asyncio

from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from help.help_chat import help_chat
print("Boot 2: imports complete", flush=True)
# --------------------------- Logging ---------------------------------
log = logging.getLogger("leaselink-app")
log.setLevel(logging.INFO)

# --------------------------- App Setup --------------------------------
app = FastAPI()
print("Boot 3: FastAPI app created", flush=True)
claude_model = "claude-opus-4-7"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.leaselink.ai", "http://localhost:5173"],  # add localhost if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Globals ----------------------------------
job_status = {}  # { job_id: {status, error, result, ...} }

EDGE_SECRET = os.getenv("PYTHON_EDGE_SECRET")
collectionName = os.getenv("QDRANT_COLLECTION", "Lease_Link")

emailCollection = 'email_chunks_v1'

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
MAX_WORKERS = int(os.getenv("LEASELINK_MAX_JOB_WORKERS", "10"))   # number of threads
BACKLOG_TARGET = int(os.getenv("LEASELINK_QUEUE_BACKLOG", "10"))  # try to keep queue this full
JOB_CLAIM_BATCH = int(os.getenv("LEASELINK_JOB_CLAIM_BATCH", "10"))  # how many to claim per RPC
#Google/Microsoft Ids
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://www.leaselink.ai")


job_queue = Queue()


# --------------------------- Helpers ----------------------------------
def authorization_check(company_id: str = "Not Required", tenant_id: str = "Not Required", auth_id: str = "Not Required", unit_id: str = "Not Required", property_id: str = "Not Required"):
    """Verify the Authorization header contains a valid Supabase JWT with the expected auth_id.

    Raises HTTPException(401) if the token is missing/invalid, or 403 if the auth_id doesn't match.
    """
    user_data = (
        supabase_client
        .table("User_Data")
        .select("role_id, company_id, user_id")
        .eq("auth_id", auth_id)
        .single()
        .execute()
    )

    if not user_data.data:
        raise HTTPException(status_code=403, detail="user not found")

    print("User Data:", user_data)

    user_role = (
        supabase_client
        .table("Roles")
        .select("Is_LeaseLink_Admin, View_All_Tenants, View_All_Properties")
        .eq("id", user_data.data["role_id"])
        .single()
        .execute()
    )

    if not user_role.data:
        raise HTTPException(status_code=403, detail="role not found")

    print("User Role:", user_role)

    role = user_role.data
    user_company_id = user_data.data["company_id"]
    user_id = user_data.data["user_id"]

    # Company check
    if not role.get("Is_LeaseLink_Admin"):
        if company_id != user_company_id and company_id != "Not Required":
            raise HTTPException(
                status_code=403,
                detail="company id does not match user company id"
            )
    company = supabase_client.table("Property_Management_Companies").select('Base_Function, propertyChat').eq('company_id', company_id).single().execute()

    if tenant_id != "Not Required":

        authorize_tenant_access(supabase_client, role, user_id, company_id, tenant_id)

    if property_id != "Not Required":
        authorize_property_access(supabase_client, role, user_id, company_id, property_id)
def verify_supabase_jwt(token: str):
    """Decode and verify a Supabase-issued JWT using the shared HS256 secret.

    Raises jwt.exceptions.* on invalid/expired tokens.  Returns the decoded payload
    dict (which includes 'sub' = auth_id) on success.
    """
    payload = jwt.decode(
        token,
        key=SUPABASE_JWT,
        algorithms=["HS256"],
        audience="authenticated",
        options={"verify_aud": True},
    )
    return payload

def verify_signature(body: bytes, signature: str, timestamp: str) -> bool:
    """Verify an HMAC-SHA256 signature produced by the Edge Function shared secret.

    The message is constructed as: timestamp_bytes + b"." + body_bytes.
    Uses constant-time comparison (hmac.compare_digest) to prevent timing attacks.
    """
    # optional: reject very old timestamps to prevent replay
    # (e.g., if abs(now - timestamp) > 5 minutes: return False)

    msg = timestamp.encode("utf-8") + b"." + body
    expected = hmac.new(EDGE_SECRET, msg, hashlib.sha256).hexdigest()
    # constant-time comparison
    return hmac.compare_digest(expected, signature)

def enqueue_next_pending_job(limit: int = JOB_CLAIM_BATCH) -> int:
    """
    Claims up to `limit` jobs via Supabase RPC and enqueues them.
    Returns the number of jobs enqueued.
    """
    try:
        claim = supabase_client.rpc("claim_next_upload_job", {"job_limit": limit}).execute()
        raw = getattr(claim, "data", None)
        if not raw:
            return 0

        # The RPC returns JSON array. SDK usually deserializes to a list of dicts.
        jobs = raw if isinstance(raw, list) else [raw]
        if not jobs:
            return 0

        count = 0
        for job in jobs:
            job_id = job.get("job_id")
            lease_id = job.get("lease_id")
            group_id = job.get("group_id")
            if not job_id or not lease_id:
                log.warning(f"Skipping invalid job payload: {job}")
                continue

            # Fetch lease row to get file path & join metadata
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
                log.error(f"Missing file_path on lease {lease_id}")
                continue

            payload = {
                "job_id": job_id,
                "group_id": group_id,
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

            job_status[job_id] = {
                "status": "processing",
                "error": None,
                "result": None,
                "queued_at": datetime.now(timezone.utc).isoformat(),
            }
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


def export_lease(job_id, lease_request, group_id, first_lease=False):
    """Thin wrapper around upload_lease_manager.load_pdf() that manages job_status bookkeeping.

    Sets the job status to 'in_progress', delegates to load_pdf with all required
    parameters, and calls Clear_Uploads on any exception before re-raising.
    When first_lease=True returns a success dict (used by the /firstLease route).
    """
    #Thin wrapper that calls upload_lease_manager.load_pdf().
    #Internal page work remains parallelized within your worker_service.
    
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        job_status[job_id]["status"] = "in_progress"
        job_status[job_id]["started_at"] = now_iso
        print(f"[{job_id}] Start LeaseLink")

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
            claude_model,
            group_id
        )
        if first_lease:
            return {"message": "First Lease Upload Completed"}

    except Exception as e:
        file_path = lease_request.get("file_path")
        # best-effort cleanup if the job crashed after S3/Storage writes
        try:
            upload_lease_manager.Clear_Uploads(job_id, file_path, job_status[job_id], group_id)
        except Exception:
            pass
        print(f"[{job_id}] Error processing job: {e}")
        raise


def job_worker():
    """Long-running worker thread that consumes jobs from job_queue and processes them.

    Each iteration:
      1. Blocks on job_queue.get() until a job payload is available.
      2. Calls export_lease to run the full upload pipeline.
      3. On success, ensures job_status is set to 'success' if not already terminal.
      4. On failure, sets job_status to 'error' and calls Clear_Uploads for cleanup.
      5. In the finally block, stamps elapsed_seconds and finished_at, persists the
         final status to Upload_Job_Status, marks the task done, and auto-refills
         the queue up to BACKLOG_TARGET pending jobs from Supabase.
    """
    while True:
        item = job_queue.get()
        try:
            if not isinstance(item, dict):
                raise TypeError(f"Expected dict, got {type(item)}: {item!r}")

            job_id = item.get("job_id")
            lease_request = item.get("lease_request") or {}
            group_id = item.get('group_id')
            if not job_id:
                raise ValueError("Missing job_id in queue item")
            if not lease_request:
                raise ValueError("Missing lease_request in queue item")

            print(f"[{job_id}] Starting Job")
            job_status[job_id]["status"] = "in_progress"
            export_lease(job_id, lease_request, group_id)

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
                file_path = lease_request.get("file_path")
                upload_lease_manager.Clear_Uploads(jid, file_path, job_status[jid], group_id)
            except Exception:
                pass

        finally:
            # stamp elapsed_seconds / finished_at
            try:
                st = job_status[item.get("job_id")].get("started_at")
                if st:
                    started = datetime.fromisoformat(st.replace("Z", "+00:00"))
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                    job_status[item.get("job_id")]["elapsed_seconds"] = elapsed
                job_status[item.get("job_id")]["finished_at"] = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass

            # reflect status for this job
            try:
                supabase_client.table("Upload_Job_Status").update(
                    {"job_info": job_status[item.get('job_id')]}
                ).eq("job_id", item.get("job_id")).execute()
            except Exception as e:
                log.warning(f"Failed to write job status in finally: {e}")

            job_queue.task_done()

            # 🔁 Auto-refill the queue up to a small warm target
            try:
                target = min(BACKLOG_TARGET, MAX_WORKERS)
                while job_queue.qsize() < target:
                    added = enqueue_next_pending_job(limit=target - job_queue.qsize())
                    if not added:
                        break
            except Exception as e:
                log.warning(f"Auto-refill failed: {e}")


# start workers
for _ in range(MAX_WORKERS):
    t = threading.Thread(target=job_worker, daemon=True)
    t.start()


# ---------------------- Global exception/signal hooks -----------------
def handle_exception(exc_type, exc_value, exc_traceback):
    """Custom sys.excepthook that logs unhandled exceptions without swallowing KeyboardInterrupt."""
    if issubclass(exc_type, KeyboardInterrupt):
        return
    print("Unhandled Exception:", "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
sys.excepthook = handle_exception

def signal_handler(sig, frame):
    """Gracefully handle SIGINT and SIGTERM by exiting cleanly."""
    print(f"Received Signal: {sig}")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ------------------------------ Routes --------------------------------

@app.get("/", include_in_schema=False)
def root():
    return PlainTextResponse("ok")

@app.head("/")
@app.get("/job-status/{job_id}")
def get_job_status(job_id: str):
    status = job_status.get(job_id)
    if not status:
        return {"Status": "unknown"}
    return status


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

@app.post("/internal/cron/tick")
def cron_tick(x_cron_secret: str = Header(default="")):
    try:
        # 1) Auth
        if x_cron_secret != CRON_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # 2) Throttle if system already busy (rough count)
        fifteen_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        busy_resp = (
            supabase_client
            .table("Upload_Job_Status")
            .select("job_id, job_info, updated_at")
            .in_("job_info->>status", ["processing", "in_progress", "extracted"])
            .gte("updated_at", fifteen_min_ago)
            .limit(100)
            .execute()
        )
        busy_count = len(busy_resp.data or [])
        if busy_count >= MAX_WORKERS:
            return {"ok": True, "skipped": "processing in progress", "busy_count": busy_count}

        # 3) Fill queue up to BACKLOG_TARGET (don’t exceed workers)
        enqueued_total = 0
        target = min(BACKLOG_TARGET, MAX_WORKERS)
        while job_queue.qsize() < target:
            added = enqueue_next_pending_job(limit=min(JOB_CLAIM_BATCH, target - job_queue.qsize()))
            if not added:
                break
            enqueued_total += added

        if enqueued_total == 0:
            return {"ok": True, "no_pending": True}

        return JSONResponse({"ok": True, "enqueued": enqueued_total}, status_code=200)

    except HTTPException:
        log.error("HTTPException in cron_tick:\n%s", traceback.format_exc())
        raise
    except Exception as e:
        log.error("Unhandled exception in cron_tick: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/refresh_tenant')
async def refresh_tenant(request: Request, background_tasks: BackgroundTasks, authorization: Optional[str] = Header(default=None)):
    try:
        body = await request.body()
        print("raw body: ", body)
        update_request = await request.json()
        auth_id = update_request.get("auth_id")
        tenant_id = update_request.get('tenant_id')
        unit_id = update_request.get('unit_id')
        company_id = update_request.get('company_id')
        

        # Validate required fields
        if not auth_id or not tenant_id or not unit_id or not company_id:
            raise HTTPException(status_code=400, detail="Missing required fields: auth_id, tenant_id, unit_id or company_id")

        token = authorization.replace("Bearer", "").strip() if authorization else None
        if not token:
            raise HTTPException(status_code=401, detail="Missing or invalid token")

        auth = verify_supabase_jwt(token)
        if not auth:
            print("Auth failed")
            raise HTTPException(status_code=403, detail="Unauthorized")

        if auth["sub"] != auth_id:
            print("Auth ID mismatch")
            raise HTTPException(status_code=403, detail="auth_id does not match token")
        authorization_check(company_id=company_id, tenant_id=tenant_id, auth_id=auth_id, unit_id=unit_id)

        background_tasks.add_task(
        final_check.extract_tenant_data,
        tenant_id,
        unit_id,
        company_id,
        time_update=True,  # time_update=True
    )

    # ✅ return immediately
        return JSONResponse(status_code=202, content={"message": "Tenant refresh started"})
    except Exception as e:
        raise e
        
@app.post('/firstLease')
async def first_lease(request: Request, authorization: Optional[str] = Header(default=None)):
    try:
        body = await request.body()
        print("raw body: ", body)
        lease_request = await request.json()
        auth_id = lease_request.get("auth_id")
        job_id = lease_request.get('job_id')
        group_id = lease_request.get('group_id')
        lease_data = lease_request.get('lease_data')
        

        # Validate required fields
        if not auth_id or not job_id or not lease_data:
            raise HTTPException(status_code=400, detail="Missing required fields: auth_id, job_id, or lease_data")

        token = authorization.replace("Bearer", "").strip() if authorization else None
        if not token:
            raise HTTPException(status_code=401, detail="Missing or invalid token")

        auth = verify_supabase_jwt(token)
        if not auth:
            print("Auth failed")
            raise HTTPException(status_code=403, detail="Unauthorized")

        if auth["sub"] != auth_id:
            print("Auth ID mismatch")
            raise HTTPException(status_code=403, detail="auth_id does not match token")
        
        # Check user's First_Value status
        user_data_resp = supabase_client.table("User_Data").select("First_Value").eq('auth_id', auth_id).single().execute()

        # For supabase-py v2: user_data_resp.data holds the row (or None)
        user_row = getattr(user_data_resp, "data", None)

        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")

        # Now safely read First_Value
        first_value = user_row.get("First_Value")

        #if first_value is True:
            #raise HTTPException(status_code=403, detail='User has already received First Value Upload')
        
        # Initialize job status
        job_status[job_id] = {
            "status": "in_progress",
            "error": None,
            "result": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        
        supabase_client.table("Upload_Job_Status").update(
            {"job_info": job_status[job_id]}
        ).eq("job_id", job_id).execute()
        
        # Execute the job synchronously (bypass queue)
        # Run in thread to avoid blocking the event loop

        try:
            t0 = time.time()
            result = await run_in_threadpool(export_lease, job_id, lease_data, group_id, True)
            print("End export_lease, seconds", time.time() - t0)
            job_status[job_id]["status"] = "success"
            job_status[job_id]["result"] = result
            job_status[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

            supabase_client.table("Upload_Job_Status").update({"job_info": job_status[job_id]}).eq("job_id", job_id).execute()


            return JSONResponse(
            status_code=200,
            content={"message": "Lease upload started successfully", "job_id": job_id, "result": result}
        )
        except Exception as e:
            log.error(f"Export lease failed in thread: {e}")
            job_status[job_id]["status"] = "error"
            job_status[job_id]["error"] = str(e)
            supabase_client.table("Upload_Job_Status").update(
                {"job_info": job_status[job_id]}
            ).eq("job_id", job_id).execute()
        
        
        
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        log.error(f"Unexpected error in /firstLease: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/entity_questions")
async def tenant_send_message(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    content_type: Optional[str] = Header(default=None)
):
    body = await request.body()
    print("raw body: ", body)
    message_request = await request.json()
    auth_id = message_request.get("auth_id")
    entity_type = message_request.get("entity_type")
    entity_id = message_request.get("entity_id")
    company_id = message_request.get("company_id")

    token = authorization.replace("Bearer", "").strip() if authorization else None
    if not token:
        return JSONResponse(status_code=401, content={"error": "Missing or invalid token"})

    auth = verify_supabase_jwt(token)
    if not auth:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if auth["sub"] != message_request.get("auth_id"):
        raise HTTPException(status_code=403, detail="auth_id does not match token")
    
    user_data = (
        supabase_client
        .table("User_Data")
        .select("role_id, company_id, user_id")
        .eq("auth_id", auth_id)
        .single()
        .execute()
    )

    if not user_data.data:
        raise HTTPException(status_code=403, detail="user not found")

    print("User Data:", user_data)

    user_role = (
        supabase_client
        .table("Roles")
        .select("Is_LeaseLink_Admin, View_All_Tenants, View_All_Properties")
        .eq("id", user_data.data["role_id"])
        .single()
        .execute()
    )

    if not user_role.data:
        raise HTTPException(status_code=403, detail="role not found")

    print("User Role:", user_role)

    role = user_role.data
    user_company_id = user_data.data["company_id"]
    user_id = user_data.data["user_id"]

    # Company check
    if not role.get("Is_LeaseLink_Admin"):
        if company_id != user_company_id:
            raise HTTPException(
                status_code=403,
                detail="company id does not match user company id"
            )
    company = supabase_client.table("Property_Management_Companies").select('Base_Function, propertyChat').eq('company_id', company_id).single().execute()

    if entity_type == "tenant":
        if not company.data['Base_Function']:
                raise HTTPException(status_code=403, detail="company does not have access to tenant chat")
        authorize_tenant_access(supabase_client, role, user_id, company_id, entity_id)

    elif entity_type == "property":
        if not company.data['propertyChat']:
                raise HTTPException(status_code=403, detail="company does not have access to property chat")
        authorize_property_access(supabase_client, role, user_id, company_id, entity_id)


    else:
        raise HTTPException(status_code=400, detail="invalid entity_type")

    threading.Thread(
        target=handle_entity_question,
        args=(message_request, supabase_client, qdrant_client, OpenAIclient, collectionName),
        daemon=True,
    ).start()

    return {"status": "Message is being processed"}

@app.post("/help")
async def help_send_message(
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    body = await request.body()
    print("raw body: ", body)
    message_request = await request.json()
    print(message_request)

    token = authorization.replace("Bearer", "").strip() if authorization else None
    if not token:
        return JSONResponse(status_code=401, content={"error": "Missing or invalid token"})

    auth = verify_supabase_jwt(token)
    if not auth:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if auth["sub"] != message_request.get("auth_id"):
        raise HTTPException(status_code=403, detail="auth_id does not match token")
    authorization_check(company_id=message_request.get("company_id"), auth_id=message_request.get("auth_id"))

    threading.Thread(
        target=handle_help_chat,
        args=(message_request,),
        daemon=True,
    ).start()

    return {"status": "Message is being processed"}
@app.get('/api/integrations/email/start')
async def start_email_integration(request: Request, provider: str):

    """
    Step 2:
    Redirect the user to the correct provider's Oauth Consent Screen"""
    app_user_id = request.query_params.get('uid')
    if not app_user_id or app_user_id in ("undefined", "null"):
        #redirect to sign in
        return RedirectResponse(f"{FRONTEND_URL}/auth?error=missing_uid")

    state_payload = {
        "uid": app_user_id,
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        "nonce": secrets.token_urlsafe(8)
    }


    state = jwt.encode(state_payload, EDGE_SECRET, algorithm="HS256")

    if provider == 'microsoft':
        base = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
        query = {
            'client_id': MS_CLIENT_ID,
            'response_type': "code",
            'redirect_uri': MS_REDIRECT_URI,
            'response_mode': 'query',
            'scope': 'openid offline_access Mail.Read',
            'state': state,
        }
        url = f"{base}?{urllib.parse.urlencode(query)}"
        return RedirectResponse(url)
    elif provider == 'google':
        base = 'https://accounts.google.com/o/oauth2/v2/auth'
        query = {
            "client_id": GOOGLE_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "access_type": "offline",
            "prompt": "consent",
            "scope": "openid email profile https://www.googleapis.com/auth/gmail.readonly",
            "state": state,
        }
        url = f"{base}?{urllib.parse.urlencode(query)}"
        return RedirectResponse(url)
    else:
        referer = request.headers.get("referer")
        redirect_url = referer or "https://www.leaselink.ai/dashboard"
        return RedirectResponse(redirect_url)
    
@app.get('/api/gmail/oauth/callback')
async def gmail_callback(request: Request):
   return await email_integration.integration_callback( request, "google")


@app.get('/api/outlook/adminconsent/callback')
async def ms_admin_consent_callback(request: Request):
    qp = request.query_params
     # approval case
    admin_consent = (qp.get("admin_consent") or "").lower()
    tenant = qp.get("tenant") or ""
    state = qp.get("state") or ""

    if admin_consent == "true":
        # store tenant approved if you want (tenant is the GUID)
        # save_tenant_admin_consent(tenant_id=tenant, ...)
        return RedirectResponse(
            f"{FRONTEND_URL}/settings/integrations?provider=microsoft&admin_approved=1&tenant={tenant}"
        )

    return RedirectResponse(
        f"{FRONTEND_URL}/settings/integrations?provider=microsoft&error=admin_consent_unknown"
    )
@app.get('/api/outlook/oauth/callback')
async def outlook_callback(request: Request):
   return await email_integration.integration_callback(request, "microsoft")  


@app.post('/api/email/resync')
async def email_sync(request: Request, authorization: Optional[str] = Header(default=None)):
    body = await request.body()
    print("raw body: ", body)
    sync_request = await request.json()
    auth_id = sync_request.get("auth_id")
    

    token = authorization.replace("Bearer", "").strip() if authorization else None
    if not token:
        return JSONResponse(status_code=401, content={"error": "Missing or invalid token"})

    auth = verify_supabase_jwt(token)
    if not auth:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if auth["sub"] != sync_request.get("auth_id"):
        raise HTTPException(status_code=403, detail="auth_id does not match token")
    res = supabase_client.table("Access_Tokens").select("provider").eq('user_auth_id', auth_id).limit(1).execute()
    if len(res.data) <= 0:
        return
    else:
        provider = res.data[0].get('provider')
        await email_integration.SyncMail(auth_id, provider)

@app.post("/api/email/new_contact")
async def new_contact_email_sync(
    request: Request,
    x_edge_token: str = Header(..., alias="X-Edge-Token"),  # required header
):
    # 1. Authenticate the request using the shared secret in the header
    if x_edge_token != EDGE_SECRET:
        # Don't leak details
        raise HTTPException(status_code=403, detail="Unauthorized")

    # 2. Parse JSON body (no secrets here)
    try:
        sync_request = await request.json()
        print(sync_request)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    auth_id = sync_request.get("auth_id")
    #Sent a contact from supabase
    contact = sync_request.get("contacts")
    print(contact)
    if not auth_id:
        raise HTTPException(status_code=400, detail="Missing auth_id")
    authorization_check(auth_id=auth_id, company_id=contact['company_id'])
    # 3. Look up provider for this auth_id
    res = (
        supabase_client.table("Access_Tokens")
        .select("provider")
        .eq("user_auth_id", auth_id)
        .execute()
    )

    if not res.data:
        # no token for this auth_id; nothing to sync
        return Response(status_code=204)

    for email in res.data:
        
        provider = email.get("provider")
        if not provider:
            return Response(status_code=204)
        
        contacts = [contact]
        print("Sync Mail")
        # 4. Trigger sync (fire-and-forget style)
        return await email_integration.SyncMail(auth_id, provider, True, contacts)
@app.post('/api/integrations/email/disconnect')
async def delete_email_integration(request: Request, authorization: Optional[str] = Header(default=None)):
    body = await request.body()
    print("raw body: ", body)
    delete_request = await request.json()
    auth_id = delete_request.get("auth_id")
    delete_qdrant = delete_request.get("delete_qdrant", False)
    provider = delete_request.get("provider")
    

    token = authorization.replace("Bearer", "").strip() if authorization else None
    if not token:
        return JSONResponse(status_code=401, content={"error": "Missing or invalid token"})

    auth = verify_supabase_jwt(token)
    if not auth:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if auth["sub"] != auth_id:
        raise HTTPException(status_code=403, detail="auth_id does not match token")
    
    supabase_client.table("Access_Tokens").update({"Active": False}).eq("user_auth_id", auth_id).eq("provider", provider).execute()
    return {"status": "Integration removed"}


def handle_entity_question(message_request, supabase_client, qdrant_client, OpenAIclient, collectionName):
    """Thread target for processing a tenant or property chat question.

    Dispatches to Qdrant_ChatGPT.get_relevant_chunks for tenant questions or
    property_chat.property_chat_request for property questions, then persists the
    user message and assistant response to the entity_questions table.
    """
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



        oldmessages = Supabase_api.message_get_request(supabase_client, session_id, "entity_questions")
        final_message = ""
        email_data = ""
        prompt_cost = 0.0
        prompt_tokens = 0.0
        json_data = {}

        user_data = supabase_client.table("User_Data").select('role_id', 'company_id', 'user_id').eq('auth_id', auth_id).single().execute()
        print('User Data: ', user_data)
        user_role = supabase_client.table('Roles').select('Is_LeaseLink_Admin', 'View_All_Tenants', 'View_All_Properties').eq('id', user_data.data.get('role_id')).single().execute()
        print('User Role: ', user_role)

        role = user_role.data
        if not role.get('Is_LeaseLink_Admin'):
            if company_id != user_data.data.get('company_id'):
                print("Company ID mismatch for non-admin user")
                return
        if not role.get('View_All_Tenants') and entity_type == 'tenant':
            tenants = supabase_client.table('User_Tenant').select('tenant_id').eq('user_id', user_data.data.get('user_id')).execute()
            tenant_ids = [t['tenant_id'] for t in tenants.data]
            if entity_id not in tenant_ids:
                print("Tenant access violation for user")
                return
        if not role.get('View_All_Properties') and entity_type == 'property':
            properties = supabase_client.table('User_Property').select('property_id').eq('user_id', user_data.data.get('user_id')).execute()
            property_ids = [p['property_id'] for p in properties.data]
            if entity_id not in property_ids:
                print("Property access violation for user")
                return
        if entity_type == "tenant":
            unit_id = message_request.get('unit_id')

            final_message, prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data, email_data = Qdrant_ChatGPT.get_relevant_chunks(
                collectionName, qdrant_client, entity_id, company_id, message,
                OpenAIclient, claude_client, oldmessages, supabase_client, claude_model, emailCollection, unit_id)
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
                        }]).execute()
                time.sleep(2)
                supabase_client.table("entity_questions").insert(
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
                            "email_sources": email_data
                        },
                    
                ).execute()
            else:
                print("LLM returned None.")
        elif entity_type =='property':
    
            tenant_data, prompt_tokens, prompt_cost, completion_tokens, completion_cost = property_chat.property_chat_request(collectionName, entity_id, message, oldmessages, claude_model, company_id)
            if tenant_data:
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
                        }]).execute()
                time.sleep(2)
                if isinstance(tenant_data, list) and all(isinstance(t, dict) for t in tenant_data):
                    for tenant in tenant_data:
                        print("Tenant")
                        supabase_client.table('entity_questions').insert(
                            [
                                {
                                    "entity_id": entity_id,
                                    "company_id": company_id,
                                    "message": tenant['short_answer'],
                                    "role": 'assistant',
                                    "session_id": session_id,
                                    'auth_id': auth_id,
                                    'message_cost': completion_cost/len(tenant_data),
                                    'prompt_tokens': 0,
                                    'completion_tokens': completion_tokens/len(tenant_data),
                                    'entity': entity_type,
                                    'sources': tenant['source_docs'],
                                    'longAnswer': tenant['long_answer']
                                }
                            ]
                        ).execute()
                else:
                        supabase_client.table('help_chat').insert(
                            [
                                {
                                    "entity_id": entity_id,
                                    "company_id": company_id,
                                    "message": tenant_data,
                                    "role": 'assistant',
                                    "session_id": session_id,
                                    'auth_id': auth_id,
                                    'message_cost': completion_cost,
                                    'prompt_tokens': 0,
                                    'completion_tokens': completion_tokens,
                                    'entity': entity_type,
                                    'sources': "",
                                    'longAnswer': ""
                                }
                            ]
                        ).execute()
            else:
                print("LLM returned None.")
        print(email_data)
        

    except Exception as e:
        print("Error in threaded message handler:", e)

def authorize_tenant_access(supabase_client, role, user_id, company_id, tenant_id):

    if role.get("View_All_Tenants"):
        result = (
            supabase_client
            .table("tenant")
            .select("tenant_id")
            .eq("tenant_id", tenant_id)
            .eq("property_management_id", company_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=403, detail="tenant id does not match company tenant ids")
    else:
        result = (
            supabase_client
            .table("User_Tenant")
            .select("tenant_id")
            .eq("user_id", user_id)
            .eq("tenant_id", tenant_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=403, detail="tenant id does not match user tenant ids")



def authorize_property_access(supabase_client, role, user_id, company_id, property_id):
    if role.get("View_All_Properties"):
        result = (
            supabase_client
            .table("properties")
            .select("prop_id")
            .eq("prop_id", property_id)
            .eq("pm_company", company_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=403, detail="property id does not match company property ids")
    else:
        result = (
            supabase_client
            .table("User_Property")
            .select("property_id")
            .eq("user_id", user_id)
            .eq("property_id", property_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=403, detail="property id does not match user property ids")

def handle_help_chat(message_request):
    """Thread target for processing a help documentation chat question.

    Fetches the session's previous messages, calls help_chat.help_chat for the RAG
    answer, then persists both the user message and the assistant response (with source
    links) to the Help_Chats table.
    """
    try:
        auth_id = message_request.get("auth_id")
        message = message_request.get("message")
        session_id = message_request.get("session_id")
        company_id = message_request.get("company_id")
        
        if not company_id or not message or not session_id or not auth_id:
            print("Missing required fields")
            return

        oldmessages = Supabase_api.message_get_request(supabase_client, session_id, "Help_Chats")
        final_message, links, prompt_cost, completion_cost = help_chat(
            message,
            oldmessages,
            claude_model,
        )
        print("Links", links)
        if final_message:
            supabase_client.table("Help_Chats").insert(
                [
                    {
                        "company_id": company_id,
                        "message": message,
                        "role": "user",
                        "session_id": session_id,
                        "auth_id": auth_id,
                        "message_cost": prompt_cost,
                    }]).execute()
            time.sleep(2)
            supabase_client.table("Help_Chats").insert(
                    {
                        "company_id": company_id,
                        "message": final_message,
                        "role": "assistant",
                        "session_id": session_id,
                        "auth_id": auth_id,
                        "message_cost": completion_cost,
                        "links": links
                    },
                
            ).execute()
        else:
            print("LLM returned None.")
    except Exception as e:
        print("Error in threaded help chat handler:", e)
# ------------------------------ Main ----------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
