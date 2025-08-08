from dotenv import load_dotenv  
from . import claude_extractor
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import uuid
import json
from . import lease_chunker
import common.Supabase_api as Supabase_api
from common.cleanup_utils import Clear_Uploads
import time
import tempfile
import os



def load_pdf(auth_id, propertyid, unit_id, tenantid, get_pdf, lease_id, bucket_name, company_id, collectionName, OpenAIclient, qdrant_client, supabase_client, job_id, job_status, claude_client):
    load_dotenv()
    upload_session_id = str(uuid.uuid4())
    print(f"Upload_Session_id: {upload_session_id}")
    extracted_lease_data = {}

    pdf_file = Supabase_api.download_file(supabase_client, bucket_name, get_pdf)

    total_cost = 0.0
    try:


        extracted_lease_data, total_cost = claude_extractor.claude_extraction(pdf_file, claude_client, verbose=True)


        if isinstance(extracted_lease_data, str):
            #if returned as string converts to json
            lease_data = json.loads(extracted_lease_data)
        else:
            #otherwise just sends json
            lease_data=extracted_lease_data
        #Adds upload_session_id into lease_data
        lease_data["upload_session_id"] = upload_session_id
        #adds file_path from bubble into lease_data
        lease_data["lease_file_path"] = get_pdf
        #adds tenant_id into lease_data
        lease_data["tenant_id"] = tenantid
        #Adds auth_id into lease_data
        lease_data["created_by"] = auth_id
        lease_data["lease_id"] = lease_id
        lease_data['page_count'] = total_pages
        #upserts lease_data into lease_documents table in supabase
        Supabase_api.supabase_post_request(supabase_client, [lease_data], "lease_documents")
        job_status['status'] = 'extracted'
        supabase_client.table('Upload_Job_Status').update({'job_info': job_status}).eq('job_id', job_id)
        supabase_client.table('tenant').update({'Available': True}).eq('tenant_id', tenantid)
        
        print("Success")
        print("Starting PDF text extraction + embedding")
        total_embedding_cost = lease_chunker.extract_text_from_pdf(pdf_file, OpenAIclient, tenantid, auth_id, propertyid, unit_id,upload_session_id, get_pdf, company_id, job_id, bucket_name, get_pdf, qdrant_client, job_status, collectionName, total_pages)
        job_status['status'] = 'success'
        supabase_client.table('Upload_Job_Status').update({'job_info': job_status}).eq('job_id', job_id)
    
        cost_upload = {}

        cost_upload['cost_per_upload'] = total_embedding_cost + total_cost
        cost_upload['lease_id'] = lease_id
        cost_upload['upload_session_id'] = upload_session_id
        Supabase_api.supabase_post_request(supabase_client, [cost_upload], 'lease_documents')

    except Exception as e:
        print(f"GPT extraction or supabase insert failed: {e}")
        job_status['status'] = "error"
        Clear_Uploads(job_id, bucket_name, get_pdf, job_status)
    #Converts pdf to image, then to text, chunks text, embeds text, and converts to json payload for qdrant




