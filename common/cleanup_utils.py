from supabase import create_client
import os
import requests
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