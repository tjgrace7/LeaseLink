from supabase import create_client
from supabase.lib.client_options import ClientOptions
import os
import os.path
from dotenv import load_dotenv

# ✅ Create Supabase client using service role key
def supabase_client_setup():
    supabase_service_key = os.getenv("SUPABASE_SERVICE_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")

    supabase = create_client(supabase_url, supabase_service_key)
    return supabase

#Get Messages from session_id
def message_get_request(supabase_client, session_id, table_name):
    try:
        response = supabase_client.table(table_name).select("*").eq("session_id", session_id).order("created_at", desc=True).limit(20).execute()
        messages = sorted(response.data, key=lambda m: m["created_at"])

        return messages
    except Exception as e:
        print("Error:", e)
        return ""

# ✅ Insert Lease Data into Supabase table
def supabase_post_request(supabase_client, data: dict, table: str):
    for item in data:
        if "lease_id" in item:
            
            lease_id = item.pop("lease_id")
            response = supabase_client.table(table).update(item).eq("lease_id", lease_id).execute()
            print(response)
        else:
            print("lease_id not found")

# ✅ Download PDF file from Supabase Storage
def download_file(supabase_client, bucket_name: str, file_path: str):
    print("Downloading_file")
    #Gets file_basename
    local_filename = os.path.basename(file_path)
    storage = supabase_client.storage.from_(bucket_name)
    #Gets file bytes from stored location in supabase
    file_bytes = storage.download(file_path)
    if file_bytes:
        print("file_bytest returned!")
        #returns file_bytes
        return file_bytes
        
    else:
        print("❌ Download failed")
        return None


def get_signed_url(supabase_client, bucket, file_path):
    print("get signed url")
    if file_path != None:
        response = supabase_client.storage.from_(bucket).create_signed_url(file_path, expires_in=3600)
        signed_url = response["signedURL"]

        print("Signed URL:", signed_url)
        
    else:
        signed_url = ""
    return signed_url