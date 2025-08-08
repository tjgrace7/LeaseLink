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

def is_real_value(val):
    return val and str(val).strip().lower() != 'n/a'

def load_pdf(auth_id, propertyid, unit_id, tenantid, get_pdf, lease_id, bucket_name, company_id, collectionName, OpenAIclient, qdrant_client, supabase_client, job_id, job_status, claude_client):
    load_dotenv()
    upload_session_id = str(uuid.uuid4())
    print(f"Upload_Session_id: {upload_session_id}")
    extracted_lease_data = {}

    pdf_file = Supabase_api.download_file(supabase_client, bucket_name, get_pdf)
    reader = ''
    total_cost = 0.0
    try:
        reader = PdfReader(BytesIO(pdf_file))
        total_pages = len(reader.pages)
        print("starting extraction")

        chunk_size=5
        combined_extracted_data = {}
        for start in range(0, total_pages, chunk_size):
            writer = PdfWriter()
            end = min(start+chunk_size, total_pages)
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')

            writer.write(temp_file)
            temp_file.close()
            temp_file_path = temp_file.name

            extracted_lease, cost = claude_extractor.claude_extraction(temp_file_path, claude_client, verbose=True)
            print("Lease Extracted", extracted_lease)
            for key, value in extracted_lease.items():
                existing = combined_extracted_data.get(key)

                if key not in combined_extracted_data:
                    combined_extracted_data[key] = value
                elif not is_real_value(existing) and is_real_value(value):
                    combined_extracted_data[key] = value
                elif is_real_value(existing) and is_real_value(value):
                    if not isinstance(existing, list):
                        combined_extracted_data[key] = [existing]
                    if value not in combined_extracted_data[key]:
                        combined_extracted_data[key].append(value)
            total_cost += cost
            if(total_pages > chunk_size):
                time.sleep(30)
            os.remove(temp_file_path)
        extracted_lease_data = combined_extracted_data
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




