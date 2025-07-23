from dotenv import load_dotenv  
import os
import lease_chunker
import uuid
import Qdrant_ChatGPT
import json
from supabase import create_client
import Supabase_api
import requests

def load_pdf(auth_id, propertyid, unit_id, tenantid, get_pdf, lease_id, bucket_name, company_id, collectionName, OpenAIclient, qdrant_client, supabase_client):
    load_dotenv()
    upload_session_id = str(uuid.uuid4())
    print(f"Upload_Session_id: {upload_session_id}")
    extracted_lease_data = {}

    pdf_file = Supabase_api.download_file(supabase_client, bucket_name, get_pdf)
    #Converts pdf to image, then to text, chunks text, embeds text, and converts to json payload for qdrant
    vectors, total_pages = lease_chunker.extract_text_from_pdf(pdf_file, OpenAIclient, tenantid, auth_id, propertyid, unit_id,upload_session_id, get_pdf, company_id)

    batch_size = 50
    #Takes embeded leases and upserts in qdrant 
    for i in range(0, len(vectors), batch_size):
        #Limits qdrant point load to 50, so it doesn't overload qdrant upload limits
        batch = vectors[i:i + batch_size]
        try:
            qdrant_client.upsert(
            collection_name=collectionName,
            wait=True,
            points = batch
        )
        except Exception as e:
            print(f"Failed on batch {i // batch_size + 1}: {e}")


    try:
        print("starting extraction")
        #Sends message to ChatGPT to extract needed data from lease
        extracted_lease_data, total_cost = Qdrant_ChatGPT.get_relevant_chunks_from_lease(collectionName, qdrant_client, OpenAIclient, upload_session_id)

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
        lease_data["status"] = "Complete"
        lease_data["cost_per_upload"] = total_cost
        lease_data['page_count'] = total_pages
        #upserts lease_data into lease_documents table in supabase
        Supabase_api.supabase_post_request(supabase_client, [lease_data], "lease_documents")
    except Exception as e:
        print(f"GPT extraction or supabase insert failed: {e}")
        Clear_Uploads(lease_id, bucket_name, get_pdf, e)

def Clear_Uploads(lease_id, bucket, file_path, error):
        supabaseurl = os.getenv("SUPABASE_URL")
        service_key = os.getenv("SUPABASE_SERVICE_API_KEY")
        supabase = create_client(supabaseurl, service_key)

        url = f"{supabaseurl}/storage/v1/object/{bucket}/{file_path}"

        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}"
        }
        supabase.table("lease_documents").update({"lease_file_path": f"File Ebed Failed: {error}", "status": "Error"}).eq("lease_id", lease_id).execute()
        response = requests.delete(url, headers=headers)
        if response.status_code ==200:
            print("File Deleted")
        else:
            print("Error Deleting File")