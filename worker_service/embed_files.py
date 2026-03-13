"""
OpenAI embedding helper for individual lease text chunks.

EmbedFiles takes a single text chunk and all the metadata needed to identify it
(tenant, property, unit, page, source document, etc.) and returns a Qdrant
PointStruct ready for upsert along with the USD embedding cost.

The model used is text-embedding-3-large (3072 dimensions, $0.13 / 1M tokens).
Cost is calculated from the tiktoken token count of the input text.
"""

from datetime import datetime, timezone
from uuid import uuid4
from qdrant_client.models import PointStruct
import tiktoken


def EmbedFiles(
    client,
    chunk,
    tenantid,
    propertymanagerid,
    propertyid,
    unitid,
    upload_session_id,
    pagenumber,
    sourcedocname,
    chunkindex,
    company_id,
    lease_id
):
    """Embed a single text chunk and return a populated Qdrant PointStruct plus the embedding cost.

    Args:
        client:            OpenAI client used to call the embeddings API.
        chunk:             The raw text string to embed.
        tenantid:          Tenant UUID for the Qdrant payload filter.
        propertymanagerid: Auth UUID of the property manager who uploaded the lease.
        propertyid:        Property UUID.
        unitid:            Unit UUID.
        upload_session_id: Session UUID shared across all chunks in one upload.
        pagenumber:        1-based page number within the source PDF.
        sourcedocname:     Storage path of the source PDF (used as source_doc in payload).
        chunkindex:        Zero-based index of this chunk within the page.
        company_id:        Company UUID for multi-tenant filtering.
        lease_id:          Lease document UUID.

    Returns:
        (PointStruct, embedding_cost_usd) on success, or (None, 0.0) on error or empty input.
    """
    tenantid = tenantid or ""

    try:
        text = (chunk or "").strip()
        if not text:
            # Return a no-op point and zero cost if empty
            return None, 0.0

        # Embed
        resp = client.embeddings.create(input=text, model="text-embedding-3-large")
        vector = resp.data[0].embedding  # <-- this is list[float]

        # (Optional) sanity check against your Qdrant vector size (e.g., 3072)
        # assert len(vector) == 3072, f"Unexpected embedding dim: {len(vector)}"

        # Cost calc (text-embedding-3-large is $0.13 / 1M tokens = 1.3e-7 per token)
        try:
            encoding = tiktoken.encoding_for_model("text-embedding-3-large")
            token_count = len(encoding.encode(text))
        except Exception:
            token_count = 0
        embedding_cost = token_count * 0.00000013
        
        # Build Qdrant point — vector MUST be a flat list, not {"dense": ...}
        point = PointStruct(
            id=str(uuid4()),
            vector={"dense_vector": vector},  # <-- flat list OK for single-vector collections
            payload={
                "tenantid": tenantid,
                "propertymanagerid": propertymanagerid,
                "propertyid": propertyid,
                "unitid": unitid,
                "pageNumber": pagenumber,
                "source_doc": sourcedocname,
                "upload_date": datetime.now(timezone.utc).isoformat(),
                "text": text,
                "session_id": upload_session_id,
                "source_id": f"{upload_session_id}_{tenantid}_{sourcedocname}_{pagenumber}_{chunkindex}",
                "managementcompany_id": company_id,
                "lease_id": lease_id,
                "highlight_id": str(uuid4()),
            },
        )
        return point, float(embedding_cost)

    except Exception as e:
        print("Error Embedding Files:", e)
        return None, 0.0
