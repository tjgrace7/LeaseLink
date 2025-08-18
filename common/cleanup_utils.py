from supabase import create_client
import os
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import  Filter, FieldCondition, MatchValue

def Clear_Uploads(job_id, bucket, file_path, job_status):
        supabaseurl = os.getenv("SUPABASE_URL")
        service_key = os.getenv("SUPABASE_SERVICE_API_KEY")
        supabase = create_client(supabaseurl, service_key)


        try:
            url = f"{supabaseurl}/storage/v1/object/{bucket}/{file_path}"


            headers = {
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}"
            }
            supabase.table("Upload_Job_Status").update({"job_info": job_status}).eq("job_id", job_id).execute()
        except Exception as e:
             print("Error updating Upload:" )
        response = requests.delete(url, headers=headers)
        qdrant_client = QdrantClient(
            url = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )
        qdrant_client.delete(
            collection_name="Lease_Link",
            points_selector=Filter(
                must=[
                    FieldCondition(key="source_doc", match=MatchValue(value=file_path))
                ]
        )
)

        if response.status_code ==200:
            print("File Deleted")
        else:
            print("Error Deleting File")