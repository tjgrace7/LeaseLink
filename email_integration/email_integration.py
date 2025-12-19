import time, os, httpx
from fastapi import HTTPException, Request, BackgroundTasks
import asyncio
import supabase
from common.Encrypt import encrypt_token, decrypt_token
from datetime import datetime, timedelta, timezone
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import common.Supabase_api as Supabase_api
import requests
from openai import OpenAI
from typing import Iterable, List, Dict, Any, Optional, Tuple
import tiktoken
from uuid import uuid4
from worker_service.Textract import ensure_collection_exists, _to_vec_list
from bs4 import BeautifulSoup
from fastapi.responses import RedirectResponse
import jwt
import base64
from qdrant_client.models import Filter, MatchValue, FieldCondition, models
import resend
import traceback

TENANT = os.getenv("MS_TENANT", "common")

MICROSOFT_TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
MICROSOFT_CLIENT_ID = os.environ["MS_CLIENT_ID"]
MICROSOFT_CLIENT_SECRET = os.environ["Microsoft_Value"]
MICROSOFT_REDIRECT_URI = os.environ.get("MS_REDIRECT_URI", "http://localhost:8000/api/outlook/oauth/callback")
MICROSOFT_DEFAULT_SCOPES = "offline_access Mail.Read"

GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GOOGLE_DEFAULT_SCOPES = "openid email profile https://www.googleapis.com/auth/gmail.readonly offline_access"
GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"

EXP_SKEW = timedelta(seconds=60)
EDGE_SECRET = os.getenv("PYTHON_EDGE_SECRET")
qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
supabase = Supabase_api.supabase_client_setup()
Graph = 'https://graph.microsoft.com/v1.0'
PAGE_SIZE=50
OPENAI_API_KEY = os.getenv("OPEN_AI_PROJECT_KEY")
ChatGPT = OpenAI(api_key=OPENAI_API_KEY)
Consistency_Header= {"ConsistencyLevel": 'eventual'}
COLL = "email_chunks_v1"
Resend_key = os.getenv("RESEND_SECRET_KEY")


FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://leaselink.ai")

async def get_internal_user_id(user_id):
    print("Get Internal User ID")
    res = supabase.table("User_Data").select("user_id").eq("auth_id", user_id).limit(1).execute()

    internal_user_id = None
    if res and getattr(res, "data", None):
        if len(res.data) > 0:
            internal_user_id = res.data[0].get("user_id")
            print("Internal User ID:", internal_user_id)

    if not internal_user_id:
        print(f"[supabase_sync] No matching internal user_id for auth_id={user_id}")
        return
    return internal_user_id
async def supabase_sync(user_id, sync_status, provider):
    internal_user_id = await get_internal_user_id(user_id)
    now = datetime.now(timezone.utc).isoformat()

    # 1) Try update
    res = supabase.table("Email_Sync_Logs") \
        .update({"last_sync": now, "sync_status": sync_status}) \
        .eq("user_id", internal_user_id) \
        .eq("provider", provider) \
        .execute()

    # supabase-py returns updated rows in res.data for update (if you use .select())
    # Safer: request the updated row back
    if not getattr(res, "data", None):
        supabase.table("Email_Sync_Logs").insert({
            "user_id": internal_user_id,
            "provider": provider,
            "last_sync": now,
            "sync_status": sync_status,
        }).execute()


async def previous_subabase_sync(user_id):
    print("Previous Supabase Sync")
    internal_user_id = await get_internal_user_id(user_id)


    sync_res = supabase.table("Email_Sync_Logs").select("*").eq("user_id", internal_user_id).limit(1).execute()
    if len(sync_res.data) >0:
        return sync_res.data[0]
    else: 
        return None
    

def save_ms_tokens_for_user(*, app_user_id: str, provider_account_id: str, access_token: str, refresh_token: str, expires_in: str, provider: str):
    print("Save MS Tokens for User")
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    expires_at_iso = expires_at.isoformat()
    
    encrypted_access_token = encrypt_token(access_token)
    encrypted_refresh_token = encrypt_token(refresh_token)
    response = supabase.table("Access_Tokens").upsert({
        'access_token': encrypted_access_token,
        'refresh_token': encrypted_refresh_token,
        'expires_at': expires_at_iso,
        'user_auth_id': app_user_id,
        'provider_account_id': provider_account_id,
        'provider': provider
    },
        on_conflict='provider_account_id'
    ).execute()
    print("Supabase Insert Response:", response)


async def exchange_code_for_tokens(code: str, provider: str):
    """
    Exchange Authorization Code for Tokens
    """
    print("Exchange Code for Tokens")
    if provider == 'microsoft':
        token_url = MICROSOFT_TOKEN_URL
        payload = {
            "client_id": MICROSOFT_CLIENT_ID,
            "grant_type": "authorization_code",
            "code": code,
            'redirect_uri': MICROSOFT_REDIRECT_URI,
            'client_secret': MICROSOFT_CLIENT_SECRET
        }
    elif provider == "google":
        token_url = GOOGLE_TOKEN_URL
        payload = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": GOOGLE_REDIRECT_URI
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported Provider")

    
    headers = {"Content-Type": 'application/x-www-form-urlencoded'}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(token_url, data=payload, headers=headers)
        data = resp.json()
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail={"token_error": data})
        
        return data
    
async def fetchMessages(user_id, provider, contact, folder: Optional[str] = None, previous_sync: Optional[datetime] = None):
    print("Start Fetch Messages")
    email = contact["email"] if isinstance(contact, dict) else contact.email
    company_id = contact["company_id"] if isinstance(contact, dict) else contact.company_id
    content_type = ""
    content_html = ""
    if provider == "microsoft":
        async for message in fetch_messages_for_sender_microsoft(user_id=user_id, sender_email=email, folder=folder, received_after_utc=previous_sync):
            graph_id = message.get("id")
            message_id = message.get("internetMessageId")
            if not graph_id:
                continue
            if qdrant_email_exists(message_id, company_id):
                continue
            content_type, content_html, full_msg = await fetch_message_body_html_microsoft(user_id, graph_id)
            merged = {**message, **full_msg}
            await handle_message_upload(contact, merged, content_type, content_html, message_id, provider, user_id)
    if provider == "google":
        async for message in fetch_messages_for_sender_google(user_id=user_id, sender_email=email, folder=folder, received_after_utc=previous_sync):
            graph_id = message.get('id')
            message_id = message.get("message_id")
            if not graph_id:
                continue
            if qdrant_email_exists(message_id, company_id):
                continue
            ct, html, _env = await fetch_message_body_html_google(user_id, graph_id)
            
            await handle_message_upload(contact, _env, ct, html, message_id, provider, user_id)
    print("Finish Fetch Messages")
    return True



async def SyncMail(user_id, provider, new_contact: bool = False, contacts: list = []):
    print("Start Sync Mail")
    print(contacts)
    try:
        previous_sync = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        if not new_contact:
            sync = await previous_subabase_sync(user_id)
        

            if sync:
                raw_last_sync = sync.get("last_sync")
                if raw_last_sync:
                    if isinstance(raw_last_sync, datetime):
                        # Ensure it's timezone-aware; assume UTC if naive
                        if raw_last_sync.tzinfo is None:
                            previous_sync = raw_last_sync.replace(tzinfo=timezone.utc)
                        else:
                            previous_sync = raw_last_sync
                    elif isinstance(raw_last_sync, str):
                        try:
                            if raw_last_sync.endswith("Z"):
                                raw_last_sync = raw_last_sync.replace("Z", "+00:00")
                            dt = datetime.fromisoformat(raw_last_sync)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            previous_sync = dt
                        except Exception as e:
                            print(f"[SyncMail] Failed to parse last_sync '{raw_last_sync}': {e}")
                    # keep default epoch

        if(len(contacts) <= 0):
            contacts = await getContacts(user_id)
        await supabase_sync(user_id, "in_progress", provider)
        for contact in contacts:
            await fetchMessages(user_id=user_id, provider=provider, contact=contact, previous_sync=previous_sync)
        await supabase_sync(user_id, 'complete', provider)
        sync_notification(user_id, contacts)
        print("Messages Fetched")

        return True
    except Exception as e:
        print(traceback.format_exc)
        await supabase_sync(user_id, "error", provider)
        print("Failed to Sync Mail", e)

def sync_notification(user_id, contacts):
    print("Send Sync Notification")
    admin = supabase.auth.admin

    resp = admin.get_user_by_id(user_id)
    user_data = resp.user
    print("User Data",user_data)
    email = user_data.email
    contact_count = len(contacts)
    user_name = user_data.user_metadata.get("name", "") if user_data.user_metadata else ""
    bullet_list_html = "<ul style='padding-left:20px;margin:0;'>"
    for c in contacts:
        bullet_list_html += f"<li>{c['contact_name']}</li>"
    params: resend.Emails.SendParams = {
        'from': "Lease Link <no-reply@leaselink.ai>",
        'to': email,
        "subject": "Email Successfully Synced",
        'html': f"""
<!DOCTYPE html>
<html lang="en" style="margin:0;padding:0;background:#f7f7f7;">
  <body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f7f7f7;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#f7f7f7;padding:20px 0;">
      <tr>
        <td align="center">
          <table width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.06);">
            
            <!-- Header -->
            <tr>
              <td style="background:#3B82F6;padding:20px 30px;text-align:center;">
                <h1 style="margin:0;font-size:24px;color:#ffffff;font-weight:700;">
                  Email Sync Successful 🎉
                </h1>
              </td>
            </tr>

            <!-- Body -->
            <tr>
              <td style="padding:30px;color:#333333;font-size:16px;line-height:1.6;">
                <p>Hi <strong>{user_name}</strong>,</p>

                <p>Your email account has been successfully synced with <strong>Lease Link</strong>.</p>

                <p>
                  We identified <strong>{contact_count}</strong> tenant contacts in your synced emails.
                  These contacts are now linked to your tenant communication inside Lease Link.
                </p>

                <p>Here are the contacts we found:</p>

                {bullet_list_html}

                <p style="margin-top:20px;">
                  Emails associated with these contacts will now appear directly inside your tenant chat 
                  for streamlined communication and better record-keeping.
                </p>

                <p style="margin-top:20px;">
                  If you didn’t request this or notice anything unusual, please contact Lease Link support.
                </p>
              </td>
            </tr>

            <!-- Button -->
            <tr>
              <td align="center" style="padding:10px 30px 30px 30px;">
                <a 
                  href="https://www.leaselink.ai"
                  style="display:inline-block;padding:12px 22px;background:#3B82F6;color:#ffffff;text-decoration:none;font-size:16px;font-weight:600;border-radius:8px;"
                >
                  Open Lease Link
                </a>
              </td>
            </tr>

            <!-- Footer -->
            <tr>
              <td style="padding:20px 30px 30px 30px;color:#888888;font-size:13px;text-align:center;line-height:1.5;">
                <p style="margin:0;">Lease Link — Smarter Leasing Starts Here</p>
                <p style="margin:5px 0 0;">You're receiving this because your mailbox was added to Lease Link.</p>
              </td>
            </tr>

          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    }
    send = resend.Emails.send(params)
    print(send)


async def handle_message_upload(contact, message, content_type, content_html, message_id, provider, user_id):
    print("Handle Message Upload")
    clean_text = html_to_text_microsoft(content_type, content_html)
    print("Contact:", contact)
    contact_id = contact["contact_id"] if isinstance(contact, dict) else contact.contact_id
    company_id = contact["company_id"] if isinstance(contact, dict) else contact.company_id
    print("Contact Id:", contact_id)
    res = supabase.table("Tenant_Contact").select("tenant_id").eq("contact_id", contact_id).limit(1).execute()
    tenant_id = (res.data[0]["tenant_id"] if res and getattr(res, "data", None) else None)
    print(tenant_id)
    from_field = (message.get("sender") or message.get("from") or {})
    email_addr = ((from_field.get("emailAddress") or {}).get("address")) or ""
    name_val = ((from_field.get('emailAddress') or {}).get("name")) or ""
    if not email_addr:
        email_addr = email_addr or ((message.get("from") or {}).get("emailAddress") or {}).get("address", "")
    if not name_val:
        name_val = name_val or ((message.get('from') or {}).get("emailAddress") or {}).get("name", "") or \
        contact["contact_name"] if isinstance(contact, dict) else contact.contact_name

    to_emails = message.get("to_emails")
    if to_emails is None:
        to_emails = [
            ((r.get("emailAddress") or {}).get("address", ""))
            for r in (message.get("toRecipients") or [])
            if (r.get("emailAddress") or {}).get("address")
        ]
    cc_emails = message.get("cc_emails")
    if cc_emails is None:
        cc_emails = [
            ((r.get("emailAddress") or {}).get("address", ""))
            for r in (message.get("ccRecipients") or [])
            if (r.get("emailAddress") or {}).get("address")
        ]
    label_ids = message.get("label_ids") or message.get("labelIds") or []
    folder = message.get("folder") or message.get("parentFolderId")

    if provider == "google":
        is_read = message.get("is_read")
        if is_read is None:
            is_read = "UNREAD" not in label_ids
    elif provider == "microsoft":
        is_read = message.get("isRead", True)
    else:
        is_read = True

    has_attachments = message.get("hasAttachments") or message.get("has_attachments") or False

    received_datetime = message.get("received_datetime")
    if received_datetime is None:
        if "internalDate" in message:
            received_datetime = int(int(message["internalDate"]))/1000
        else:
            rd = message.get("receivedDateTime")
            if rd:
                dt = datetime.fromisoformat(rd.replace("Z", "+00:00"))
                received_datetime = int(dt.timestamp())
            else:
                received_datetime = int(time.time())
    received_at = message.get("received_at") or int(time.time())

    conversation_id = (
        message.get("conversationId")
        or message.get("thread_id")
        or message.get("threadId")
        or message.get("conversation_id")
        or message.get("id")
    )

    subject = message.get("subject") or ""
    thread_id = message.get("thread_id") or message.get("threadId")

    attachment_names = message.get("attachment_names") or []


    try:
        embeddingcost = await UploadMail(
            tenant_id=tenant_id, 
            company_id=company_id, 
            contact_id=contact_id, 
            from_email=email_addr,
            provider=provider,
            label_ids=label_ids,
            has_attachments = has_attachments,
            to_emails = to_emails,
             body=clean_text, 
            received_datetime=received_datetime,
            conversation_id=conversation_id,
            received_at=received_at,
            subject=subject,
            is_read=is_read,
            message_id=message_id,
            cc_emails=cc_emails,
            attachment_names=attachment_names,
            thread_id=thread_id,
            folder=folder,
            sender_name=name_val,
            user_id=user_id,
        )
        return True
    except Exception as e:
        print("Failed to UploadMail:", e)



async def UploadMail    (
    tenant_id: Optional[str],
    company_id: Optional[str],
    contact_id: Optional[str],
    provider: str,
    label_ids: Optional[list] = None,
    has_attachments: bool = False,
    to_emails: Optional[list] = None,
    body: str = "",
    received_datetime: Optional[int] = None,
    from_email: str = "",
    conversation_id: Optional[str] = None,
    received_at: Optional[int] = None,
    subject: str = "",
    is_read: bool = True,
    message_id: Optional[str] = None,
    cc_emails: Optional[list] = None,
    attachment_names: Optional[list] = None,
    thread_id: Optional[str] = None,
    folder: Optional[str] = None,
    sender_name: str = "",
    user_id = None,
    collection: str = COLL,
    ) -> float:

    print("Upload Mail to Qdrant")
    text = (body or "").strip()
    if not text:
        return None, 0.0
    resp = ChatGPT.embeddings.create(input=text, model='text-embedding-3-large')
    vector=resp.data[0].embedding

    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    token_count = len(encoding.encode(text)) or 0
    embedding_cost = token_count*0.00000013
    print("Tenant_id", tenant_id)

    point = PointStruct(
        id=str(uuid4()),
        vector=vector,
        payload={
        "tenant_id": tenant_id,
        "company_id": company_id,
        "contact_id": contact_id,
        "provider": provider,
        "label_ids": label_ids or [],
        "has_attachments": bool(has_attachments),
        "to_emails": to_emails or [],
        "body": text,
        "received_datetime": int(received_datetime),
        "from_email": from_email,
        "conversation_id": conversation_id,
        "received_at": int(received_at),
        "subject": subject,
        "is_read": bool(is_read),
        "message_id": message_id,
        "cc_emails": cc_emails or [],
        "attachment_names": attachment_names or [],
        "thread_id": thread_id,
        "folder": folder,
        "Sender_Name": sender_name,
        "auth_id": user_id
        },
    )
    if isinstance(point, PointStruct):
        vec_len = len(_to_vec_list(point)) 
        ensure_collection_exists(collection, vec_len, qdrant_client)
        try: 
            qdrant_client.upsert(collection, points=[point])
        except Exception as e:
            print("Error uploading to qdrant:", e)
    return embedding_cost









async def getContacts(user_id):
    print("Get Contacts for User")
    resp = supabase.rpc('get_user_contacts', {'p_auth_id': user_id}).execute()
    if getattr(resp, 'error', None):
        raise RuntimeError(resp.error)
    print(resp.data)
    return resp.data


async def integration_callback(request: Request, provider: str):
    print("Integration Callback Triggered")
    qp=request.query_params
    if 'error' in qp:
        #TODO Make sure this link goes somewhere the user knows code is missing
        return RedirectResponse(f"{FRONTEND_URL}/settings/integrations?provider={provider}&error={qp.get('error_description','consent_failed')}")
    
    code = qp.get('code')
    state = qp.get('state')
    if not code:
        return RedirectResponse(f"{FRONTEND_URL}/settings/integrations?provider={provider}&error=missing_code")
    
    try:
        st = jwt.decode(state, EDGE_SECRET, algorithms=["HS256"])
        app_user_id = st.get('uid')
        if not app_user_id:
            return RedirectResponse(f"{FRONTEND_URL}/login?error=not_signed_in")
    except Exception:
        return RedirectResponse(f"{FRONTEND_URL}/settings/integrations?error=bad_state")

    token_data = await exchange_code_for_tokens(code, provider)
    access_token = token_data["access_token"]
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get('expires_in', 3600)

    #Indentify MailBox Ownerhttps://chatgpt.com/g/g-p-684f5df2a764819187c79375b41612c5-leaselink-app/project
    async with httpx.AsyncClient(timeout=30) as client:
        me_url=""
        provider_account_id = ''
        if provider == "microsoft":
            me_url = "https://graph.microsoft.com/v1.0/me"
        else:
            me_url = 'https://gmail.googleapis.com/gmail/v1/users/me/profile'
        me_resp = await client.get(
            me_url,
            headers={"Authorization": f"Bearer {access_token}"}
            )
        print(me_resp)
    if me_resp.status_code != 200:
        #If we can't read /me, still bounce back with error
        return RedirectResponse(
            f"{FRONTEND_URL}/settings/integrations?provider={provider}&error=me_fetch_failed"
        )
    me = me_resp.json()   
    print(me)
    if provider == "microsoft":
        provider_account_id = me.get("id") or ""

    else:
        async with httpx.AsyncClient(timeout=30) as client:
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
        if userinfo_resp.status_code ==200:
            userinfo = userinfo_resp.json()
            provider_account_id = userinfo.get("sub", "")
            print(userinfo)
            

    save_ms_tokens_for_user(
        app_user_id=app_user_id,
        provider_account_id=provider_account_id,
        access_token=access_token,
        refresh_token=refresh_token or "",
        expires_in=expires_in,
        provider=provider
    )
    asyncio.create_task(SyncMail(app_user_id, provider))
    return RedirectResponse(
        f"{FRONTEND_URL}/settings/integrations?provider={provider}&connected=1"
    )




async def refresh_access_token(user_id: str) -> dict:
    print("refresh access token")
    """
    Load user's tokens, refresh if expired (with skew), save rotated refresh_token,
    and return a dict with access_token, expires_at (UTC ISO), scope, provider_account_id.
    """
    # 1) Load current tokens row
    # Supabase-py v2 pattern:
    row = (
        supabase.table("Access_Tokens")
        .select("*")
        .eq("user_auth_id", user_id)
        .single()
        .execute()
    )
    if getattr(row, "error", None):
        raise RuntimeError(f"Supabase select error: {row.error}")

    data = row.data or {}
    if not data:
        raise RuntimeError("No token record found for user.")

    # 2) Parse times in UTC
    # Make sure your DB stores UTC; if it's a string like "2025-10-24T15:00:00+00:00"
    expires_at_db = data["expires_at"]
    if isinstance(expires_at_db, str):
        # 2025-10-24T15:00:00+00:00 or without tz
        dt = datetime.fromisoformat(expires_at_db.replace("Z", "+00:00"))
    elif isinstance(expires_at_db, datetime):
        dt = expires_at_db
    else:
        raise RuntimeError("Unexpected expires_at type from DB.")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # assume UTC if naive

    now_utc = datetime.now(timezone.utc)

    # 3) If still valid (with skew), return decrypted access token
    if now_utc + EXP_SKEW < dt:
        access_token = decrypt_token(data["access_token"])
        return {
            "access_token": access_token,
            "expires_at": dt.isoformat(),
            "scope": data.get("scope"),
            "provider_account_id": data.get("provider_account_id"),
            "refreshed": False,
        }

    # 4) Otherwise, refresh using httpx (async, non-blocking)
    refresh_token_plain = decrypt_token(data["refresh_token"])
    token_url = ""
    form = {}
    URL = ""
    provider = data["provider"]
    if provider == "microsoft":
        token_url = MICROSOFT_TOKEN_URL
        URL = "https://graph.microsoft.com/v1.0/me"
        form = {
            "client_id": MICROSOFT_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token_plain,
            # optional on refresh; if you include, it must be equal/subset of original:
            "scope": MICROSOFT_DEFAULT_SCOPES,
            "redirect_uri": MICROSOFT_REDIRECT_URI,
        }
        # Include secret for confidential (server) apps
        if MICROSOFT_CLIENT_SECRET:
            form["client_secret"] = MICROSOFT_CLIENT_SECRET

    elif provider == "google":
        token_url = GOOGLE_TOKEN_URL
        URL = f"{GMAIL_BASE}/users/me/profile"
        form = {
            "client_id": GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'grant_type': "refresh_token",
            'refresh_token': refresh_token_plain
        }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(token_url, data=form)
        if resp.status_code != 200:
            # Typical errors: invalid_grant (revoked/expired RT), invalid_client (wrong secret)
            raise RuntimeError(f"Refresh failed: {resp.status_code} {resp.text}")
        payload = resp.json()

        new_access_token = payload["access_token"]
        new_refresh_token = payload.get("refresh_token", refresh_token_plain)  # rotate if provided
        expires_in = int(payload.get("expires_in", 3600))
        new_expires_at = now_utc + timedelta(seconds=expires_in)

        # Optional: verify token & capture account id
        me_resp = await client.get(
            URL,
            headers={"Authorization": f"Bearer {new_access_token}"},
        )
        if me_resp.status_code != 200:
            raise RuntimeError(f"/me failed: {me_resp.status_code} {me_resp.text}")
        me = me_resp.json()
        provider_account_id = me.get("id") or data.get("provider_account_id") or ""

    # 5) Persist: SAVE THE ROTATED REFRESH TOKEN
    save_ms_tokens_for_user(
        app_user_id=user_id,
        provider_account_id=provider_account_id,
        access_token=new_access_token,           # let your save fn do encryption
        refresh_token=new_refresh_token,         # let your save fn do encryption
        expires_in=expires_in,                   # or change save fn to accept absolute expires_at
        provider=provider,
    )

    return {
        "access_token": new_access_token,
        "expires_at": new_expires_at.isoformat(),
        "scope": payload.get("scope", data.get("scope")),
        "provider_account_id": provider_account_id,
        "refreshed": True,
    }

#Google Specific Functions
def gmail_search_query(sender_email: str, received_after_utc: Optional[datetime]) -> str:
    print("Gmail Search Query")
    q = [f"from:{sender_email}"]
    print(received_after_utc)

    if received_after_utc:
        if received_after_utc.tzinfo is None:
            received_after_utc = received_after_utc.replace(tzinfo=timezone.utc)
        
        epoch = datetime(1970, 1, 2, tzinfo=timezone.utc)
        if received_after_utc > epoch:
            ts = int(received_after_utc.astimezone(timezone.utc).timestamp())
            q.append(f"after:{ts}")
    print(" ".join(q))
    return " ".join(q)

async def gmail_client(user_id: str) -> httpx.AsyncClient:
    print("Gmail Client")
    tokens = await refresh_access_token(user_id)
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    return httpx.AsyncClient(base_url=GMAIL_BASE, headers=headers, timeout=30)

def gmail_get_header(headers: List[Dict[str, str]], name: str) -> str:
    print("Gmail Get Header")
    for h in headers or []:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""

def b64url_decode(data_b64url: Optional[str]) -> str:
    print("Base64 URL Decode")
    if not data_b64url:
        return ""
    padding = "=" * (-len(data_b64url)%4)
    raw = base64.urlsafe_b64decode(data_b64url + padding)
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")
    
def gmail_find_html_or_text(payload: Dict[str, Any]) -> Tuple[str, str]:
    print("Gmail Find HTML or Text")
    if not payload:
        return ("html", "")
    mt = payload.get("mimeType", "")
    data = payload.get("body", {}).get("data")
    
    if data and (mt == "text/html" or mt == "text/plain"):
        return ("html" if mt == "text/html" else "plain", b64url_decode(data))
    
    for prefer in ("text/html", "text/plain"):
        stack = list(payload.get("parts", []) or [])
        while stack:
            p=stack.pop(0)
            p_mt = p.get("mimeType", "")
            p_data = p.get("body", {}).get("data")
            if p_data and p_mt == prefer:
                return ("html") if prefer == "text/html" else "plain", b64url_decode(p_data)
            if p.get("parts"):
                stack.extend(p["parts"])
    if data:
        return ("plain", b64url_decode(data))
    return ("html", "")

def gmail_normalize_list_item(msg_full: Dict[str, Any], folder: Optional[str] = None) -> Dict[str, Any]:
    print("Gmail Normalize List Item")
    payload = msg_full.get("payload", {}) or {}
    headers = payload.get("headers", []) or []

    from_raw = gmail_get_header(headers, "from")
    to_raw = gmail_get_header(headers, "To")
    cc_raw = gmail_get_header(headers, "Cc")
    subject = gmail_get_header(headers, "Subject")
    message_id_raw = gmail_get_header(headers, "Message-Id") or ""

    normalized_msg_id = (
        message_id_raw.strip().strip("<>").strip().lower()
        if message_id_raw else None
    )
    def parse_list(raw: str) -> list[str]:
        if not raw:
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]
    to_emails = parse_list(to_raw)
    cc_emails = parse_list(cc_raw)

    internal_ms = int(msg_full.get("internalDate", "0"))
    received_ts = internal_ms // 1000

    label_ids = msg_full.get("labelIds", []) or []
    
    #How do 
    #Gmail Labels: UNREAD means not read
    is_read = "UNREAD" not in label_ids



    def wrap(addr: str) -> Dict[str, Any]:
        return {"emailAddress": {"address": addr, "name": ""}}
    return {
        "id": msg_full.get("id"),
        "thread_id": msg_full.get("threadId"),
        "label_ids": label_ids,
        "subject": subject,
        "from": wrap(from_raw),
        "sender": wrap(from_raw),
        "toRecipients": [wrap(to_raw)] if to_raw else [],
        "to_emails": to_emails,
        "cc_emails": cc_emails,
        "receivedDateTime": datetime.fromtimestamp(
            int(msg_full.get("internalDate", "9"))/1000.0,
            tz=timezone.utc
        ).isoformat(),
        "received_datetime": received_ts,
        "received_at": int(time.time()),
        "hasAttachments": any(
            (p.get("filename") or "") for p in (payload.get("parts") or [])
        ),
        "bodyPreview": msg_full.get("snippet", ""),
        "parentFolderId": folder or None,
        "folder": folder or None,
        "is_read": is_read,
        "message_id": normalized_msg_id
    }
#Gmail Fetchers
async def fetch_messages_for_sender_google(user_id: str, sender_email: str, received_after_utc: Optional[datetime] = None, top: int = PAGE_SIZE, folder: Optional[str] = None,) -> Iterable[Dict[str, Any]]:
    print("Fetch Messages for Sender Google")
    q = gmail_search_query(sender_email, received_after_utc)

    async with await gmail_client(user_id) as client:
        url = "/users/me/messages"
        params: Dict[str, Any] = {"q": q, "maxResults": min(top, 100)}
        if folder:
            params['labelIds'] = [folder]
        
        while True:
            r = await client.get(url, params=params)
            if r.status_code in (429, 403) and r.headers.get("Retry-After"):
                retry = int(r.headers.get("Retry-After", "2"))
                await asyncio.sleep(retry)
                continue
            r.raise_for_status()
            data = r.json()
            ids = [m["id"] for m in data.get("messages", [])]
            if not ids:
                return
            
            rows: List[Dict[str, Any]] = []
            for mid in ids:
                gr = await client.get(
                    f"/users/me/messages/{mid}",
                    params={"format": "metadata", "metadataHeaders": ["From", "To", "Subject", "Date", "Message-Id"]}
                )
                if gr.status_code in (429, 403) and gr.headers.get("Retry-After"):
                    retry = int(gr.headers.get("Retry-After", "2"))
                    await asyncio.sleep(retry)
                    gr = await client.get(
                        f"/users/me/messages/{mid}",
                        params={"format": "metadata", "metadataHeaders": ["From", "To", "Subject", "Date"]}
                    )
                gr.raise_for_status()
                rows.append(gmail_normalize_list_item(gr.json()))
            
            rows.sort(key=lambda x: x.get("receivedDateTime", ""), reverse = True)
            for item in rows:
                yield item
            
            page_token = data.get("nextPageToken")
            if not page_token:
                return
            params["pageToken"] = page_token

async def fetch_message_body_html_google(
        user_id: str,
        message_id: str,
) -> Tuple[str, str, Dict[str, Any]]:
    print("Fetch Message Body HTML Google")
    async with await gmail_client(user_id) as client:
        r = await client.get(f"/users/me/messages/{message_id}", params={"format": "full"})
        if r.status_code in (429, 403) and r.headers.get("Retry-After"):
            retry = int(r.headers.get("Retry-After", "2"))
            await asyncio.sleep(retry)
            r = await client.get(f"/users/me/messages/{message_id}", params={"format": "full"})
        r.raise_for_status()
        msg = r.json()
        content_type, content = gmail_find_html_or_text(msg.get("payload", {}))
        env = gmail_normalize_list_item(msg)
        return content_type, content, env

#Microsoft Specific Functions

async def _graph_client_microsoft(user_id: str) -> httpx.AsyncClient:
    print("Graph Client Microsoft")
    tokens = await refresh_access_token(user_id)
    headers = {"Authorization": f"bearer {tokens['access_token']}"}
    return httpx.AsyncClient(base_url=Graph, headers=headers, timeout=30)
async def fetch_messages_for_sender_microsoft(
    user_id: str,
    sender_email: str,
    received_after_utc: Optional[datetime] = None,
    top: int = PAGE_SIZE,
    folder: str = "Inbox",
) -> Iterable[Dict[str, Any]]:
    print("Fetch Messages for Sender Microsoft")
    fields = "id,subject,from,sender,toRecipients,receivedDateTime,hasAttachments,bodyPreview,parentFolderId,internetMessageId"
    base_headers = {"ConsistencyLevel": "eventual"}  # needed for search & filter+orderby combos

    filter_clause = f"(from/emailAddress/address eq '{sender_email}' or sender/emailAddress/address eq '{sender_email}')"

    if received_after_utc:
        if received_after_utc.tzinfo is None:
            received_after_utc = received_after_utc.replace(tzinfo=timezone.utc)
        
        epoch = datetime(1970,1,2, tzinfo=timezone.utc)

        if received_after_utc > epoch:
            filter_clause += (
                f" and receivedDateTime ge "
                f"{received_after_utc.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )
    # 1) Try strict OData filter first (works with $orderby)
    params = {
        "$select": fields,
        "$orderby": "receivedDateTime desc",
        "$top": str(top),
        "$filter": filter_clause
    }

    async with await _graph_client_microsoft(user_id) as client:
        print("Inside Fetch Messages for Sender Microsoft")
        url = f"/me/messages"
        try:
            while True:
                r = await client.get(url, params=params, headers=base_headers)
                if r.status_code == 429:
                    retry = int(r.headers.get("Retry-After", "2")); await asyncio.sleep(retry); continue
                r.raise_for_status()
                data = r.json()
                for item in data.get("value", []):
                    yield item
                next_link = data.get("@odata.nextLink")
                if not next_link: return
                url, params = next_link, None  # Graph nextLink already encodes params
        except httpx.HTTPStatusError as e:
            # 2) Fallback to $search (NO $orderby allowed)
            search_params = {
                "$search": f"\"participants:{sender_email}\"",  # do not manually escape quotes
                "$select": fields,
                "$top": str(top),
                # no $orderby with $search
            }
            url = "/me/messages"  # broader scope tends to be more stable for $search

            while True:
                rs = await client.get(url, params=search_params, headers=base_headers)
                if rs.status_code == 429:
                    retry = int(rs.headers.get("Retry-After", "2")); await asyncio.sleep(retry); continue
                rs.raise_for_status()
                data = rs.json()
                rows = data.get("value", [])

                # Optional: tighten to exact sender match and/or received_after filter
                out = []
                for m in rows:
                    addr_from   = ((m.get("from") or {}).get("emailAddress") or {}).get("address", "")
                    addr_sender = ((m.get("sender") or {}).get("emailAddress") or {}).get("address", "")
                    if sender_email.lower() in {addr_from.lower(), addr_sender.lower()}:
                        if received_after_utc:
                            dt = datetime.fromisoformat(m["receivedDateTime"].replace("Z","+00:00"))
                            if dt < received_after_utc.astimezone(timezone.utc):
                                continue
                        out.append(m)

                # Since $orderby is not allowed with $search, sort locally
                out.sort(key=lambda x: x.get("receivedDateTime",""), reverse=True)
                for m in out:
                    yield m

                next_link = data.get("@odata.nextLink")
                if not next_link: return
                url, search_params = next_link, None


async def fetch_message_body_html_microsoft(user_id: str, message_id: str) -> Tuple[str, str]:
    print("Fetch Message Body HTML Microsoft")
    async with await _graph_client_microsoft(user_id) as client:
        r = await client.get(
            f"/me/messages/{message_id}",
            params={"$select": "id,body,conversationId,conversationIndex,internetMessageId,from,sender,toRecipients,ccRecipients,receivedDateTime,hasAttachments,parentFolderId,isRead"},

        )
        r.raise_for_status()
        m = r.json()
        body = m.get("body", {})
        content_type = body.get("contentType", "html")
        content = body.get("content", "") or ""
        return content_type, content, m

async def remove_integration_tokens(user_id: str, provider: str, delete_qdrant):
    print("Remove Integration Tokens")
    resp = supabase.table("Access_Tokens").delete().eq("user_auth_id", user_id).eq("provider", provider).execute()
    
    if getattr(resp, "error", None):
        print("Error deleting tokens:", resp.error)
    uid = await get_internal_user_id(user_id)
    await supabase_sync(user_id, "disconnected", provider)
    if delete_qdrant:
        try:
            count = qdrant_client.count(
                collection_name="email_chunks_v1",
                count_filter=qdrant_filter,
                exact=True
            )
            print("Matching points:", count.count)

            # Build a payload filter: auth_id == user_id AND provider == provider
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="auth_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    models.FieldCondition(
                        key="provider",
                        match=models.MatchValue(value=provider),
                    ),
                ]
            )

            qdrant_client.delete(
                collection_name="email_chunks_v1",
                points_selector=models.FilterSelector(filter=qdrant_filter),
                wait=True,  # wait for completion
            )

            print(f"Deleted Qdrant points for auth_id={user_id}, provider={provider}")
        except Exception as e:
            print("Error deleting Qdrant collection:", e)    

def html_to_text_microsoft(content_type: str, body: str) -> str:
    print("HTML to Text Microsoft")
    if content_type.lower() == "html":
        # Basic sanitize: strip tags, collapse whitespace
        soup = BeautifulSoup(body or "", "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())
    print(body)
    return (body or "").strip()


#Qdrant Search
def qdrant_email_exists(unique_message_id, company_id) -> bool:
    print("Qdrant Email Exists Check")
    res = qdrant_client.scroll(
        collection_name="email_chunks_v1",
        limit=1,
        with_vectors=False,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="message_id",
                    match=MatchValue(value=unique_message_id)
                ),
                FieldCondition(
                    key="company_id",
                    match=MatchValue(value=company_id)
                )
            ]
        )
    )
    return len(res[0]) >0