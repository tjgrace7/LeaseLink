from datetime import datetime
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
    chunk_class
):
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
            vector={'dense-vector':vector},  # <-- flat list OK for single-vector collections
            payload={
                "tenantid": tenantid,
                "propertymanagerid": propertymanagerid,
                "propertyid": propertyid,
                "unitid": unitid,
                "pageNumber": pagenumber,
                "source_doc": sourcedocname,
                "upload_date": datetime.now.isoformat(),
                "text": text,
                "session_id": upload_session_id,
                "source_id": f"{upload_session_id}_{tenantid}_{sourcedocname}_{pagenumber}_{chunkindex}",
                "managementcompany_id": company_id,
                "highlight_id": str(uuid4()),
                "embedding_class": chunk_class,
            },
        )
        return point, float(embedding_cost)

    except Exception as e:
        print("Error Embedding Files:", e)
        return None, 0.0
