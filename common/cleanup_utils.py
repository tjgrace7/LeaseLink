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




def CheckGroupComplete(supabase, resend, group_id: str) -> Optional[Dict[str, Any]]:
    """
    - Verifies group is complete (done_jobs + error_jobs == total_jobs)
    - Finds users in the same company who have Roles.Create_Lease_Documents = true
    - Gathers each job's lease filename + status (+ error if present)
    - Updates upload_groups.completed_at
    - Sends an email via Resend to all qualified recipients

    Returns a summary dict (or None if not complete or no recipients).
    """
    # 1) Load the group
    try:
        grp_resp = (
            supabase.table("upload_groups")
            .select("id, company_id, tenant_id, total_jobs, done_jobs, error_jobs, completed_at")
            .eq("id", group_id)
            .single()
            .execute()
        )
        group = getattr(grp_resp, "data", None)
        if not group:
            raise RuntimeError(f"upload_group {group_id} not found")

        total_jobs = (group or {}).get("total_jobs", 0) or 0
        done_jobs  = (group or {}).get("done_jobs", 0) or 0
        error_jobs = (group or {}).get("error_jobs", 0) or 0
        print(total_jobs)
        print(done_jobs)
        print(error_jobs)
        # Not complete? bail early.
        if total_jobs != (done_jobs + error_jobs):
            return None

    # 2) Load tenant name (best-effort)
        tenant_name = "(Unknown Tenant)"
        if group.get("tenant_id"):
            t_resp = (
                supabase.table("tenant")
                .select("Tenant_Name")
                .eq("tenant_id", group["tenant_id"])
                .single()
                .execute()
            )
            t_data = getattr(t_resp, "data", None) or {}
            tenant_name = t_data.get("Tenant_Name") or tenant_name
            print(tenant_name)

        # 3) Find company users whose Role allows Create_Lease_Documents
        sendto: List[str] = []
        u_resp = (
            supabase.table("User_Data")
            .select("auth_id, role_id")
            .eq("company_id", group["company_id"])
            .execute()
        )
        users = getattr(u_resp, "data", None) or []
        print(users)

        for u in users:
            role_id = u.get("role_id")
            if not role_id:
                continue

            r_resp = (
                supabase.table("Roles")
                .select("Create_Lease_Documents")
                .eq("id", role_id)
                .single()
                .execute()
            )
            r_data = getattr(r_resp, "data", None) or {}
            if not r_data.get("Create_Lease_Documents", False):
                continue  # skip users who shouldn't get upload emails

        # Get the user's auth email via Admin API
            auth_id = u.get("auth_id")
            if not auth_id:
                continue

        # supabase-py returns a dict-like; handle both shapes gracefully
            admin_resp = supabase.auth.admin.get_user_by_id(auth_id)
        # Try common shapes:
            user_obj = None
            if isinstance(admin_resp, dict):
                user_obj = admin_resp.get("user")
            elif hasattr(admin_resp, "user"):
                user_obj = admin_resp.user
            elif hasattr(admin_resp, "data"):
                user_obj = getattr(admin_resp, "data", {}).get("user")

            email = (user_obj or {}).get("email")
            if email:
                sendto.append(email)

    # If no one to email, still mark complete and exit
        now_iso = datetime.now(timezone.utc).isoformat()
        (
            supabase.table("upload_groups")
            .update({"completed_at": now_iso})
            .eq("id", group_id)
            .execute()
        )

        if not sendto:
            return {
                "group_id": group_id,
                "tenant": tenant_name,
                "recipients": [],
                "message": "Group completed; no eligible recipients found."
            }

    # 4) Build per-document status list
        jobs_resp = (
            supabase.table("Upload_Job_Status")
            .select("lease_id, job_info")
            .eq("group_id", group_id)
            .execute()
        )
        jobs = getattr(jobs_resp, "data", None) or []

        documents: List[Dict[str, Any]] = []
        for job in jobs:
            lease_id = job.get("lease_id")
            info = job.get("job_info") or {}
            status = info.get("status", "unknown")
            err = info.get("error")

            lease_name = "(unknown)"
            if lease_id:
                lease_resp = (
                    supabase.table("lease_documents")
                    .select("lease_file_path")
                    .eq("lease_id", lease_id)
                    .single()
                    .execute()
                )
                lease = getattr(lease_resp, "data", None) or {}
                path = lease.get("lease_file_path")
                if path:
                    lease_name = os.path.basename(path)

            documents.append({
                "file": lease_name,
                "status": status,
                "error": err
            })

        # 5) Render a simple HTML email
        def esc(s: Any) -> str:
            return html.escape("" if s is None else str(s))

        rows = "".join(
            f"<tr>"
            f"<td style='padding:6px 10px;border:1px solid #e5e7eb;'>{esc(d['file'])}</td>"
            f"<td style='padding:6px 10px;border:1px solid #e5e7eb;'>{esc(d['status'])}</td>"
            f"<td style='padding:6px 10px;border:1px solid #e5e7eb;color:#b91c1c;'>{esc(d.get('error',''))}</td>"
            f"</tr>"
            for d in documents
        )
        body = f"""
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
        <tbody>
          {rows}
        </tbody>
      </table>
      <p style="margin-top:12px;color:#6b7280;">Timestamp: {esc(now_iso)}</p>
    </div>
    """.strip()

    # 6) Send the email (Resend Python SDK: resend.Emails.send({...}))
    # Make sure you have RESEND_FROM set like "Lease Link <no-reply@leaselink.ai>"
        from_addr = os.getenv("RESEND_FROM", "Lease Link <no-reply@leaselink.ai>")
        subject = f"Upload Complete for {tenant_name}"

        resend_res = resend.Emails.send({
            "from": from_addr,
            "to": sendto,
            "subject": subject,
            "html": body
        })

        return {
            "group_id": group_id,
            "tenant": tenant_name,
            "recipients": sendto,
            "documents": documents,
            "resend_result": getattr(resend_res, "__dict__", str(resend_res))
        }
    except Exception as e:
        print("Error Sending Email", e)
