from dotenv import load_dotenv
import json
import uuid
from datetime import datetime
import common.Supabase_api as Supabase_api
import traceback
from common.cleanup_utils import Clear_Uploads, CheckGroupComplete, NotifyComplete
import traceback


from . import Textract, final_check

def load_pdf(
    auth_id: str,
    propertyid: str,
    unit_id: str,
    tenant_id: str,
    get_pdf: str,                 # storage path (e.g. "company/tenant/file.pdf")
    lease_id: str,
    bucket_name: str,
    company_id: str,
    collectionName: str,
    OpenAI,          # ✅ pass API key (string), not a client
    qdrant_client,                # parent-only client (safe)
    supabase_client,              # parent-only client (safe)
    job_id: str,
    job_status: dict,               # ok to use in parent for metadata extraction
    claude_model: str,
    group_id: str
):
    """
    1) Download PDF from Storage
    2) Extract structured fields with Claude → upsert lease_documents
    3) Fan-out per-page OCR/text + embeddings via ProcessPoolExecutor in lease_chunker
    4) Batch upsert to Qdrant (done inside lease_chunker parent)
    5) Track cost back in lease_documents
    """
    load_dotenv()
    start = datetime.now()
    upload_session_id = str(uuid.uuid4())
    print(f"Upload_Session_id: {upload_session_id}")

    # 1) Download the PDF
    pdf_file = Supabase_api.download_file(supabase_client, bucket_name, get_pdf)

    try:


 

        
        print("Starting PDF text extraction + embedding")



        total_embedding_cost, total_pages = Textract.runTextract(
             pdf=pdf_file,
             file_path=get_pdf,
             tenantid=tenant_id,
             propertymanagerid=auth_id,
             propertyid=propertyid,
             unit_id=unit_id,
             upload_session_id=upload_session_id,
             company_id=company_id, 
             embedding_client=OpenAI,
            qdrant_client=qdrant_client,
            jobid=job_id,
            collectionName=collectionName,
            group_id=group_id,
            lease_id=lease_id
             )

        textract_cost = total_pages * .015 + total_embedding_cost
        # Status bookkeeping
        job_status["status"] = "success"
        (
            supabase_client
            .table("Upload_Job_Status")
            .update({"job_info": job_status})
            .eq("job_id", job_id)
            .execute()                                           # ✅ ensure update runs
        )

        # 4) Persist costs back to lease_documents (same row; using your helper)
        cost_upload = {
            "cost_per_upload": (textract_cost),
            "lease_id": lease_id,
            "upload_session_id": upload_session_id,
        }
        Supabase_api.supabase_post_request(supabase_client, [cost_upload], "lease_documents")
        
        print("Run Group")
        group = supabase_client.table('upload_groups').select('done_jobs').eq('id', group_id).single().execute()
        print(group)
        if not group.data:
         raise RuntimeError(f"upload_group {group_id} not found: {group}")
        row = group.data
        done_jobs = int(row.get('done_jobs') or 0) + 1
        res = supabase_client.table('upload_groups').update({'done_jobs': done_jobs}).eq('id', group_id).execute()
        print(res)
        print("Check Complete Group")
        res = CheckGroupComplete(group_id)

        is_done = res['is_done']
        if is_done:
            final_check.extract_tenant_data(tenant_id, unit_id,company_id, claude_model, collectionName)
            NotifyComplete(group_id)

        end = datetime.now()
        duration = (end-start).total_seconds() 
        print("Duration:", duration)
        print("Success")
    except Exception as e:
        print("Upload Error", e)
        print(traceback.format_exc)
        uploadError(e, job_status, supabase_client, job_id, get_pdf, group_id)



def uploadError(e, job_status, supabase_client, job_id, get_pdf, group_id):
        traceback.print_exc()
        print(f"GPT extraction or supabase insert failed: {e}")
        job_status["status"] = "error"
        # reflect error to job status table
        (
            supabase_client
            .table("Upload_Job_Status")
            .update({"job_info": job_status})
            .eq("job_id", job_id)
            .execute()
        )
        # cleanup uploaded artifacts
        Clear_Uploads(job_id, get_pdf, job_status, group_id)
