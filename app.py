# ---------- put these caps at the VERY TOP (before heavy imports) ----------
import os, urllib.parse, secrets, httpx, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# --------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks, Response
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from email_integration import PKCE
from typing import Optional
from queue import Queue
import threading
import uuid
import jwt
import logging
import traceback
import sys
import signal
import time
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from anthropic import Anthropic
from qdrant_client import QdrantClient

import common.Supabase_api as Supabase_api
from worker_service import upload_lease_manager
from web_api import Qdrant_ChatGPT
from email_integration import email_integration
import hmac
import hashlib

# --------------------------- Logging ---------------------------------
log = logging.getLogger("leaselink-app")
log.setLevel(logging.INFO)

# --------------------------- App Setup --------------------------------
app = FastAPI()
claude_model = "claude-sonnet-4-20250514"

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
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")


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

def verify_signature(body: bytes, signature: str, timestamp: str) -> bool:
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


def export_lease(job_id, lease_request, group_id):
    
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
            claude_client,
            claude_model,
            group_id
        )

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

@app.post('/firstLease')
async def first_lease(request: Request, authorization: Optional[str] = Header(default=None)):
    body = await request.body()
    print("raw body: ", body)
    lease_request = await request.json()
    auth_id = lease_request.get("auth_id")
    job_id = lease_request.get('job_id')
    group_id = lease_request.get('group_id')
    lease_data = lease_request.get('lease_data')

    token = authorization.replace("Bearer", "").strip() if authorization else None
    if not token:
        return JSONResponse(status_code=401, content={"error": "Missing or invalid token"})

    auth = verify_supabase_jwt(token)
    if not auth:
        raise HTTPException(status_code=403, detail="Unauthorized")

    if auth["sub"] != lease_request.get("auth_id"):
        raise HTTPException(status_code=403, detail="auth_id does not match token")
    user_data = supabase_client.table("User_Data").select("*").eq('auth_id', auth_id).single().execute()
    if user_data.First_Value:
        raise HTTPException(status_code=403, detail='User has already received First Value Upload')
    
    res = await export_lease(job_id=job_id, lease_request=lease_data, group_id=group_id)

    return JSONResponse(
    status_code=200,
    content={"message": "Lease uploaded successfully", "job_id": job_id}
)


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

    # 3. Look up provider for this auth_id
    res = (
        supabase_client.table("Access_Tokens")
        .select("provider")
        .eq("user_auth_id", auth_id)
        .limit(1)
        .execute()
    )

    if not res.data:
        # no token for this auth_id; nothing to sync
        return Response(status_code=204)

    provider = res.data[0].get("provider")
    if not provider:
        return Response(status_code=204)
    contacts = [contact]
    print("Sync Mail")
    # 4. Trigger sync (fire-and-forget style)
    return await email_integration.SyncMail(auth_id, provider, True, contacts)



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

        final_message, prompt_tokens, prompt_cost, completion_tokens, completion_cost, json_data, email_data = Qdrant_ChatGPT.get_relevant_chunks(
            collectionName, qdrant_client, filtertype, entity_id, company_id, message,
            OpenAIclient, claude_client, oldmessages, supabase_client, claude_model, emailCollection
        )
        print(email_data)
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
            print("Message successfully processed.")
        else:
            print("LLM returned None.")
    except Exception as e:
        print("Error in threaded message handler:", e)

# ------------------------------ Main ----------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)