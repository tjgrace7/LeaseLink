from supabase import create_client
import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import  Filter, FieldCondition, MatchValue
from datetime import datetime, timezone
import resend
import html
import os.path
from typing import List, Dict, Any, Optional

supabaseurl = os.getenv("SUPABASE_URL")
service_key = os.getenv("SUPABASE_SERVICE_API_KEY")
supabase = create_client(supabaseurl, service_key)
resend.api_key = os.getenv('RESEND_SECRET_KEY')

def Clear_Uploads(job_id, file_path, job_status, group_id):

        print(job_status)        
        qdrant_client = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )

        try:
            group = supabase.table('upload_groups').select("error_jobs").eq('id', group_id).single().execute()
            errors = group.get('error_jobs')

            supabase.table("Upload_Job_Status").update({"job_info": job_status}).eq("job_id", job_id).execute()
            supabase.table("upload_groups").update({'error_jobs': errors + 1}).eq('group_id', group_id)
            CheckGroupComplete(group_id)
            qdrant_client.delete(
            collection_name="Lease_Link",
            points_selector=Filter(
                must=[
                    FieldCondition(key="source_doc", match=MatchValue(value=file_path))
                ]
            )
        )
        except Exception as e:
             print("Error updating Upload:" )


        print("Cleared Qdrant and Uploaded File_Status to Error")


# ---------- Supabase helpers ----------
def sb_ok(resp):
    """Raise on error; return resp.data (dict/list/None)."""
    if getattr(resp, "error", None):
        err = resp.error
        msg = getattr(err, "message", str(err))
        raise RuntimeError(msg)
    return getattr(resp, "data", None)

def sb_single(table_call):
    """Convenience for .single().execute() -> data (dict or None)."""
    return sb_ok(table_call.single().execute())

def sb_exec(table_call):
    """Convenience for .execute() -> data (list/dict/None)."""
    return sb_ok(table_call.execute())

def get_auth_email(supabase, auth_id: str) -> str | None:
    r = supabase.auth.admin.get_user_by_id(auth_id)
    # supabase-py shapes vary; handle both
    if hasattr(r, "user") and r.user:
        return r.user.get("email") if isinstance(r.user, dict) else getattr(r.user, "email", None)
    if hasattr(r, "data") and isinstance(r.data, dict):
        u = r.data.get("user")
        if isinstance(u, dict): return u.get("email")
        return getattr(u, "email", None)
    return None

# ---------- Your function ----------
def CheckGroupComplete(group_id: str):
    group = sb_single(
        supabase.table("upload_groups")
        .select("id, company_id, tenant_id, total_jobs, done_jobs, error_jobs, completed_at")
        .eq("id", group_id)
    )
    
    if not group:
        raise RuntimeError(f"upload_group {group_id} not found")

    total = group.get("total_jobs", 0) or 0
    done  = group.get("done_jobs", 0) or 0
    err   = group.get("error_jobs", 0) or 0
    if total != done + err:
        return None  # not complete yet

    # Tenant (best effort)
    tenant_name = "(Unknown Tenant)"
    if group.get("tenant_id"):
        t = sb_single(
            supabase.table("tenant").select("Tenant_Name").eq("tenant_id", group["tenant_id"])
        ) or {}
        tenant_name = t.get("Tenant_Name") or tenant_name

    # Find recipients: users in company whose Role has Create_Lease_Documents = true
    users = sb_exec(
        supabase.table("User_Data").select("auth_id, role_id").eq("company_id", group["company_id"])
    ) or []

    sendto = []
    for u in users:
        role_id = u.get("role_id")
        if not role_id:
            continue
        role = sb_single(
            supabase.table("Roles").select("Create_Lease_Documents").eq("id", role_id)
        ) or {}
        if not role.get("Create_Lease_Documents", False):
            continue

        email = get_auth_email(supabase, u.get("auth_id"))
        if email:
            sendto.append(email)

    # Update completed_at regardless of recipients
    now_iso = datetime.now(timezone.utc).isoformat()
    sb_exec(supabase.table("upload_groups").update({"completed_at": now_iso}).eq("id", group_id))

    # Build job/document table
    jobs = sb_exec(
        supabase.table("Upload_Job_Status").select("lease_id, job_info").eq("group_id", group_id)
    ) or []

    documents = []
    for j in jobs:
        lease_id = j.get("lease_id")
        info = j.get("job_info") or {}
        status = info.get("status", "unknown")
        error  = info.get("error")

        filename = "(unknown)"
        if lease_id:
            ld = sb_single(
                supabase.table("lease_documents").select("lease_file_path").eq("lease_id", lease_id)
            ) or {}
            path = ld.get("lease_file_path")
            if path:
                filename = os.path.basename(path)

        documents.append({"file": filename, "status": status, "error": error})

    # Nothing to email? stop here
    if not sendto:
        return {
            "group_id": group_id,
            "tenant": tenant_name,
            "recipients": [],
            "documents": documents,
            "message": "Group completed; no eligible recipients."
        }

    # Render HTML email
    def esc(x): return html.escape("" if x is None else str(x))
    rows = "".join(
        f"<tr>"
        f"<td style='padding:6px 10px;border:1px solid #e5e7eb;'>{esc(d['file'])}</td>"
        f"<td style='padding:6px 10px;border:1px solid #e5e7eb;'>{esc(d['status'])}</td>"
        f"<td style='padding:6px 10px;border:1px solid #e5e7eb;color:#b91c1c;'>{esc(d.get('error',''))}</td>"
        f"</tr>"
        for d in documents
    )
    html_body = f"""
    <div style="font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;">
      <h2 style="margin:0 0 10px 0;">Upload Complete</h2>
      <p style="margin:0 0 12px 0;">Tenant: <strong>{esc(tenant_name)}</strong></p>
      <table cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid #e5e7eb;">
        <thead>
          <tr>
            <th style="text-align:left;padding:8px 10px;border:1px solid #e5e7eb;">Document</th>
            <th style="text-align:left;padding:8px 10px;border:1px solid #e5e7eb;">Status</th>
            <th style="text-align:left;padding:8px 10px;border:1px solid #e5e7eb;">Error</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="margin-top:12px;color:#6b7280;">Timestamp: {esc(now_iso)}</p>
    </div>
    """.strip()

    from_addr = os.getenv("RESEND_FROM", "Lease Link <no-reply@leaselink.ai>")
    subject = f"Upload Complete for {tenant_name}"

    resend_result = resend.Emails.send({
        "from": from_addr,
        "to": sendto,
        "subject": subject,
        "html": html_body
    })

    return {
        "group_id": group_id,
        "tenant": tenant_name,
        "recipients": sendto,
        "documents": documents,
        "resend_result": str(resend_result),
    }
