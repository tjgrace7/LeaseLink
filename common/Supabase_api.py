"""
Supabase client setup and common database/storage helper functions.

This module is the single place responsible for constructing the Supabase service-role
client and provides reusable helpers for:
  - Fetching chat message history for a session
  - Upserting lease document data by lease_id
  - Downloading raw PDF bytes from Supabase Storage
  - Generating short-lived signed URLs for lease documents
"""

from supabase import create_client
import os
from dotenv import load_dotenv
load_dotenv()


# --- Client Setup ---

# ✅ Create Supabase client using service role key
def supabase_client_setup():
    """Initialise and return a Supabase client authenticated with the service role key.

    The service role key bypasses Row Level Security (RLS), so this client should
    only be used in trusted server-side code paths.
    """
    supabase_service_key = os.getenv("SUPABASE_SERVICE_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL")

    supabase = create_client(supabase_url, supabase_service_key)
    return supabase


# --- Message Helpers ---

#Get Messages from session_id
def message_get_request(supabase_client, session_id, table_name):
    """Fetch the last 20 messages for a chat session, returned in chronological order.

    Queries the given table for all rows matching session_id, sorts them oldest-first
    so they can be fed directly into an LLM conversation history, and returns the list.
    Returns an empty string on error.
    """
    try:
        response = supabase_client.table(table_name).select("*").eq("session_id", session_id).order("created_at", desc=True).limit(20).execute()
        # Re-sort ascending so the LLM receives messages in chronological order.
        messages = sorted(response.data, key=lambda m: m["created_at"])

        return messages
    except Exception as e:
        print("Error:", e)
        return ""


# --- Lease Data Helpers ---

# ✅ Insert Lease Data into Supabase table
def supabase_post_request(supabase_client, data: dict, table: str):
    """Upsert a list of lease data dicts into the specified table, keyed by lease_id.

    Each dict in data must contain a 'lease_id' key which is popped and used as the
    WHERE clause for an UPDATE.  Any dict missing lease_id is skipped with a warning.
    """
    for item in data:
        if "lease_id" in item:
            # Pop lease_id so it is used only as the filter, not as an updated column.
            lease_id = item.pop("lease_id")
            response = supabase_client.table(table).update(item).eq("lease_id", lease_id).execute()
            print(response)
        else:
            print("lease_id not found")


# --- Storage Helpers ---

# ✅ Download PDF file from Supabase Storage
def download_file(supabase_client, bucket_name: str, file_path: str):
    """Download and return raw bytes for a file stored in Supabase Storage.

    Args:
        bucket_name: The storage bucket (e.g. "lease-docs").
        file_path:   The object path within the bucket.

    Returns:
        bytes on success, or None if the download fails.
    """
    try:
        print("Downloading_file")
        #Gets file_basename
        storage = supabase_client.storage.from_(bucket_name)
        #Gets file bytes from stored location in supabase

        file_bytes = storage.download(file_path)

        print("file_bytest returned!")
        #returns file_bytes
        return file_bytes

    except Exception as e:
        print("❌ Download failed", e)
        return None


def get_signed_url(supabase_client, bucket, file_path):
    """Generate a 1-hour signed URL for a private Storage object.

    Returns an empty string if file_path is None or if URL generation fails.
    The signed URL is used to give the frontend temporary read access to lease PDFs.
    """
    print("get signed url")
    if file_path != None:
        try:
            # Create a signed URL valid for 3600 seconds (1 hour).
            response = supabase_client.storage.from_(bucket).create_signed_url(file_path, expires_in=3600)
            signed_url = response["signedURL"]

            print("Signed URL:", signed_url)
        except Exception as e:
            print("Error getting signed URL", e)
            signed_url = ""
    else:
        signed_url = ""
    return signed_url