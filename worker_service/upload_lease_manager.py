from dotenv import load_dotenv  
from . import lease_extractor
import uuid
import json
from . import lease_chunker
import common.Supabase_api as Supabase_api
from common.cleanup_utils import Clear_Uploads



def load_pdf(auth_id, propertyid, unit_id, tenantid, get_pdf, lease_id, bucket_name, company_id, collectionName, OpenAIclient, qdrant_client, supabase_client, job_id, job_status):
    load_dotenv()
    upload_session_id = str(uuid.uuid4())
    print(f"Upload_Session_id: {upload_session_id}")
    extracted_lease_data = {}

    pdf_file = Supabase_api.download_file(supabase_client, bucket_name, get_pdf)
    #Converts pdf to image, then to text, chunks text, embeds text, and converts to json payload for qdrant
    print("Starting PDF text extraction + embedding")
    total_pages = lease_chunker.extract_text_from_pdf(pdf_file, OpenAIclient, tenantid, auth_id, propertyid, unit_id,upload_session_id, get_pdf, company_id, job_id, bucket_name, get_pdf, qdrant_client, job_status)
    print("Finished Embedding Total Page: ", total_pages)


    try:
        print("starting extraction")
        #Sends message to ChatGPT to extract needed data from lease
        extracted_lease_data, total_cost = lease_extractor.get_relevant_chunks_from_lease(collectionName, qdrant_client, OpenAIclient, upload_session_id)

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
        lease_data["cost_per_upload"] = total_cost
        lease_data['page_count'] = total_pages
        #upserts lease_data into lease_documents table in supabase
        Supabase_api.supabase_post_request(supabase_client, [lease_data], "lease_documents")
        job_status[job_id]['status'] = 'done'
        supabase_client.table('Upload_Job_Status').update({'job_info': job_status}).eq('job_id', job_id)

        
        print("Success")

    except Exception as e:
        print(f"GPT extraction or supabase insert failed: {e}")
        Clear_Uploads(job_id, bucket_name, get_pdf, job_status)
    finally:
        del supabase_client
        del extracted_lease_data
        del qdrant_client
        del upload_session_id
        del pdf_file
        del total_pages

