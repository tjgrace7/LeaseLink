"""
Error cleanup and batch-completion utilities for the LeaseLink upload pipeline.

This module provides two main responsibilities:

1. Cleanup on failure (Clear_Uploads):
   When a lease upload job fails, this module deletes any Qdrant vector points that
   were already written for that file and updates the Upload_Job_Status table to
   reflect the error so the UI can surface it to the user.

2. Batch-completion notification (CheckGroupComplete / NotifyComplete):
   After each job finishes, CheckGroupComplete determines whether all jobs in the
   upload group are done (succeeded + errored == total).  When the group is complete,
   NotifyComplete marks the group with a completed_at timestamp and sends an HTML
   summary email via Resend to every user in the company whose role has
   Create_Lease_Documents permission.
"""

from supabase import create_client
import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from datetime import datetime, timezone
import resend
import html
import os.path
from typing import List, Dict, Any, Optional
from common.Supabase_api import supabase_client_setup


# Module-level Supabase client and Resend initialisation.
supabase = supabase_client_setup()
resend.api_key = os.getenv('RESEND_SECRET_KEY')


def Clear_Uploads(job_id, file_path, job_status, group_id):
        """Remove failed-job artifacts and update the job status record.

        On a job failure this function:
        1. Persists the error job_status dict to Upload_Job_Status.
        2. Increments the error_jobs counter on the upload_groups row.
        3. Checks whether the group is now fully complete (all jobs done or errored).
        4. Deletes all Qdrant points whose source_doc matches file_path so the vector
            index is not left with partial/corrupt data for this lease.
        """

        print(job_status)
        # Build a fresh Qdrant client for cleanup; module-level client may not be available.
        qdrant_client = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )

        try:
            # Retrieve the current error count for this upload group before incrementing.
            group = supabase.table('upload_groups').select("error_jobs").eq('id', group_id).single().execute()
            errors = group.get('error_jobs')

            # Persist the error status for this specific job.
            supabase.table("Upload_Job_Status").update({"job_info": job_status}).eq("job_id", job_id).execute()
            # Increment the error counter on the group so completion checks are accurate.
            supabase.table("upload_groups").update({'error_jobs': errors + 1}).eq('group_id', group_id)
            CheckGroupComplete(group_id)
            # Delete all Qdrant vectors for this file to avoid stale/partial data.
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


# --- Supabase helpers ---

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
    """Look up the email address for a Supabase auth user by their UUID.

    Handles both dict-style and object-style user responses from supabase-py to
    accommodate different SDK versions.  Returns None if the user is not found.
    """
    r = supabase.auth.admin.get_user_by_id(auth_id)
    # supabase-py shapes vary; handle both
    if hasattr(r, "user") and r.user:
        return r.user.get("email") if isinstance(r.user, dict) else getattr(r.user, "email", None)
    if hasattr(r, "data") and isinstance(r.data, dict):
        u = r.data.get("user")
        if isinstance(u, dict): return u.get("email")
        return getattr(u, "email", None)
    return None


# --- Group completion checks ---

def CheckGroupComplete(group_id: str):
    """Return whether all jobs in an upload group have finished (succeeded or errored).

    Queries the upload_groups table to compare total_jobs against the sum of
    done_jobs and error_jobs.  Returns {'is_done': True} when the counts match,
    {'is_done': False} otherwise.  Raises RuntimeError if the group is not found.
    """
    group = sb_single(
        supabase.table("upload_groups")
        .select("id, company_id, tenantId, total_jobs, done_jobs, error_jobs, completed_at")
        .eq("id", group_id)
    )
    print(group)
    if not group:
        raise RuntimeError(f"upload_group {group_id} not found")

    total = group.get("total_jobs", 0) or 0
    done  = group.get("done_jobs", 0) or 0
    err   = group.get("error_jobs", 0) or 0
    if total != done + err:
        return {'is_done': False}  # not complete yet
    else:
        return {'is_done': True}
def NotifyComplete(group_id: str):
    """Send a completion email to eligible company users when an upload group finishes.

    Steps:
      1. Fetches the upload_group row and resolves the tenant name (best-effort).
      2. Identifies all users in the company whose role has Create_Lease_Documents=True
         and looks up their email addresses via the Supabase auth admin API.
      3. Stamps the group's completed_at timestamp.
      4. Builds a per-document status table from Upload_Job_Status rows.
      5. Renders an HTML email and sends it via Resend.
      6. Returns a summary dict with recipients, documents, and the Resend result.

    If there are no eligible recipients the function returns early without sending.
    """
    group = sb_single(
        supabase.table("upload_groups")
        .select("id, company_id, tenantId, total_jobs, done_jobs, error_jobs, completed_at")
        .eq("id", group_id)
    )
    # Tenant (best effort)
    tenant_name = "(Unknown Tenant)"
    if group.get("tenantId"):
        t = sb_single(
            supabase.table("tenant").select("Tenant_Name").eq("tenant_id", group["tenantId"])
        ) or {}
        print(t.get("Tenant_Name"))
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

    from_addr = "Lease Link <no-reply@leaselink.ai>"
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
        'is_done': True
    }
