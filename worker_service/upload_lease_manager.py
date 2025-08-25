from dotenv import load_dotenv
from . import claude_extractor
import json
import uuid
import os
from . import lease_chunker
import common.Supabase_api as Supabase_api
from common.cleanup_utils import Clear_Uploads

def load_pdf(
    auth_id: str,
    propertyid: str,
    unit_id: str,
    tenantid: str,
    get_pdf: str,                 # storage path (e.g. "company/tenant/file.pdf")
    lease_id: str,
    bucket_name: str,
    company_id: str,
    collectionName: str,
    openai_api_key: str,          # ✅ pass API key (string), not a client
    qdrant_client,                # parent-only client (safe)
    supabase_client,              # parent-only client (safe)
    job_id: str,
    job_status: dict,
    claude_client,                # ok to use in parent for metadata extraction
    claude_model: str,
):
    """
    1) Download PDF from Storage
    2) Extract structured fields with Claude → upsert lease_documents
    3) Fan-out per-page OCR/text + embeddings via ProcessPoolExecutor in lease_chunker
    4) Batch upsert to Qdrant (done inside lease_chunker parent)
    5) Track cost back in lease_documents
    """
    load_dotenv()

    upload_session_id = str(uuid.uuid4())
    print(f"Upload_Session_id: {upload_session_id}")

    # 1) Download the PDF
    pdf_file = Supabase_api.download_file(supabase_client, bucket_name, get_pdf)

    try:
        # 2) Claude extraction (metadata/fields)
        extracted_lease_data, claude_total_cost, total_pages = claude_extractor.claude_extraction(
            pdf_file, claude_client, supabase_client, claude_model, verbose=True
        )
        if extracted_lease_data is None:
            uploadError("No Extraction Data")
        # Normalize to dict
        lease_data = (
            json.loads(extracted_lease_data)
            if isinstance(extracted_lease_data, str)
            else extracted_lease_data
        ) or {}

        # Attach our additional fields
        lease_data["upload_session_id"] = upload_session_id
        lease_data["lease_file_path"] = get_pdf
        lease_data["tenant_id"] = tenantid
        lease_data["created_by"] = auth_id
        lease_data["lease_id"] = lease_id
        lease_data["page_count"] = total_pages

        print(lease_data)
        # Upsert into lease_documents
        Supabase_api.supabase_post_request(supabase_client, [lease_data], "lease_documents")

        # Status bookkeeping
        job_status["status"] = "extracted"
        (
            supabase_client
            .table("Upload_Job_Status")
            .update({"job_info": job_status})
            .eq("job_id", job_id)
            .execute()                                           # ✅ ensure update runs
        )
        (
            supabase_client
            .table("tenant")
            .update({"Available": True})
            .eq("tenant_id", tenantid)
            .execute()                                           # ✅ ensure update runs
        )

        print("Success")
        print("Starting PDF text extraction + embedding")

        # 3) Per-page extraction + embeddings (process pool inside lease_chunker)
        #    NOTE: we pass openai_api_key (string). The child process creates its own client.
        source_doc_name = os.path.basename(get_pdf)

        total_embedding_cost = lease_chunker.extract_text_from_pdf(
            pdf=pdf_file,
            openai_api_key=openai_api_key,          # ✅ string, not client
            tenantid=tenantid,
            propertymanagerid=auth_id,              # you previously passed auth_id here; keeping same mapping
            propertyid=propertyid,
            unit_id=unit_id,
            upload_session_id=upload_session_id,
            source_doc_name=source_doc_name,
            company_id=company_id,
            job_id=job_id,
            bucket=bucket_name,
            file_path=get_pdf,                      # full storage path
            qdrant_client=qdrant_client,            # parent client; upserts happen in parent
            job_status=job_status,
            collectionName=collectionName,
            total_pages=total_pages,
        )

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
            "cost_per_upload": (total_embedding_cost or 0.0) + (claude_total_cost or 0.0),
            "lease_id": lease_id,
            "upload_session_id": upload_session_id,
        }
        Supabase_api.supabase_post_request(supabase_client, [cost_upload], "lease_documents")

    except Exception as e:
        uploadError(e)

def uploadError(e):
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
        Clear_Uploads(job_id, bucket_name, get_pdf, job_status)
