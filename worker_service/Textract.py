import os, time, json, re, uuid, boto3, botocore
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from PyPDF2 import PdfReader
from io import BytesIO
from qdrant_client.models import Distance, VectorParams, PointStruct
from common.cleanup_utils import Clear_Uploads

# your embedding module
from . import embed_files

load_dotenv(find_dotenv())

# ----------------------------- GLOBALS --------------------------------

QDRANT_COLLECTION = "Test"

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")

# Used only for Textract async (S3-backed) jobs
S3_BUCKET = os.getenv("TEXTRACT_S3_BUCKET", "leaselinkleases")

# How large a chunk can be (characters) before we split it
MAX_CHARS_PER_CHUNK = 4000  # set None to disable

# Upsert to Qdrant in batches (each chunk is still an individual point/payload)
UPSERT_BATCH_SIZE = 64

# ----------------------------- CHUNKING -------------------------------
section_regex = re.compile(
    r"^\s*(article\s+[ivx]+|section\s+\d+(\.\d+)*|\d+\.\d+|exhibit\s+[a-z])\b",
    re.IGNORECASE
)

def chunk_text_by_sections(text: str) -> List[str]:
    lines = text.splitlines()
    chunks, current = [], []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if section_regex.match(stripped):
            if current:
                chunks.append("\n".join(current))
                current = [stripped]
            else:
                current.append(stripped)
        else:
            current.append(stripped)
    if current:
        chunks.append("\n".join(current))
    return chunks

def split_long_chunk(c: str, limit: int) -> List[str]:
    if not limit or len(c) <= limit:
        return [c]
    parts, start = [], 0
    while start < len(c):
        end = min(start + limit, len(c))
        nl = c.rfind("\n", start, end)
        sp = c.rfind(" ", start, end)
        cut = max(nl, sp)
        if cut > start + int(limit * 0.6):
            end = cut
        parts.append(c[start:end].strip())
        start = end
    return [p for p in parts if p]

# ----------------------------- CLIENTS --------------------------------
def s3_client() -> Any:
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def textract_client() -> Any:
    return boto3.client(
        "textract",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

# ----------------------------- TEXTRACT -------------------------------
def start_text_job(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    If the doc is small (≤ 5 pages), run synchronous Bytes-based OCR.
    If large (> 5 pages), upload to S3 and start an async job.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    tx = textract_client()

    if total_pages <= 5:
        resp = tx.detect_document_text(Document={"Bytes": pdf_bytes})
        blocks = resp.get("Blocks", [])
        return {"mode": "sync", "blocks": blocks, "pages": total_pages}

    # async path
    key = filename
    if not key.lower().endswith(".pdf"):
        key = f"{key}.pdf"

    s3 = s3_client()
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )

    resp = tx.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": S3_BUCKET, "Name": key}}
    )
    return {"mode": "async", "job_id": resp["JobId"], "s3_key": key, "pages": total_pages}

def wait_for_text_job(job_id: str, poll_seconds: float = 2.0, max_wait_seconds: int = 1800) -> str:
    tx = textract_client()
    waited, backoff = 0, poll_seconds
    while True:
        jr = tx.get_document_text_detection(JobId=job_id, MaxResults=1000)
        status = jr["JobStatus"]
        if status in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
            return status
        time.sleep(backoff)
        waited += backoff
        backoff = min(backoff * 1.25, 10)
        if waited >= max_wait_seconds:
            raise TimeoutError(f"Textract job {job_id} did not finish within {max_wait_seconds}s")

def fetch_all_text_blocks(job_id: str) -> List[Dict[str,Any]]:
    tx = textract_client()
    blocks, next_token = [], None
    while True:
        kwargs = {"JobId": job_id, "MaxResults": 1000}
        if next_token:
            kwargs["NextToken"] = next_token
        jr = tx.get_document_text_detection(**kwargs)
        blocks.extend(jr.get("Blocks", []))
        next_token = jr.get("NextToken")
        if not next_token:
            break
    return blocks

def build_pages_text_from_lines(blocks: List[Dict[str,Any]]) -> List[str]:
    pages: Dict[int, List[str]] = {}
    for b in blocks:
        if b.get("BlockType") == "LINE":
            p = b.get("Page", 1)
            txt = b.get("Text", "")
            if txt:
                pages.setdefault(p, []).append(txt)
    return ["\n".join(pages[p]) for p in sorted(pages.keys())]

# ----------------------------- OCR -> CHUNKS --------------------------
def run_ocr_and_chunk(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Returns chunk metadata only. (No embedding or upsert here.)
    """
    t0 = datetime.now()

    start = start_text_job(pdf_bytes, filename)
    if start["mode"] == "sync":
        status = "SUCCEEDED"
        blocks = start["blocks"]
        total_pages = start["pages"]
    else:
        status = wait_for_text_job(start["job_id"])
        blocks = fetch_all_text_blocks(start["job_id"])
        total_pages = start["pages"]

    pages_text = build_pages_text_from_lines(blocks)

    chunk_records: List[Dict[str, Any]] = []
    total_chunks = 0

    for page_num, page_text in enumerate(pages_text, start=1):
        # 1) Split by section-ish headings
        sections = chunk_text_by_sections(page_text)
        # 2) Split long sections by char limit
        final_chunks: List[str] = []
        for c in sections:
            final_chunks.extend(split_long_chunk(c, MAX_CHARS_PER_CHUNK))

        for chunk_index, c in enumerate(final_chunks):
            chunk_records.append({
                "page": page_num,
                "chunk_index": chunk_index,
                "text": c
            })
        total_chunks += len(final_chunks)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"OCR+Chunking: {total_chunks} chunks across {total_pages} pages in {elapsed:.2f}s")

    return {
        "job_status": status,
        "pages_count": total_pages,
        "pages_text_preview": [p[:300] for p in pages_text[:3]],
        "chunks": chunk_records,
        "chunks_count": total_chunks,
        "elapsed_seconds": elapsed,
    }

# ----------------------------- QDRANT HELPERS -------------------------
def ensure_collection_exists(collection: str, vector_size: int, qdrant_client) -> None:
    try:
        qdrant_client.get_collection(collection_name=collection)
    except Exception:
        # Create if missing
        qdrant_client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        # Add payload indexes here if desired.

def upsert_points(collection: str, points: List[PointStruct], qdrant_client) -> None:
    if points:
        qdrant_client.upsert(collection_name=collection, points=points)

# ----------------------------- EMBED + UPSERT (BATCHED) ---------------
def embed_chunks_and_upsert(
    chunks: List[Dict[str, Any]],
    *,
    embedding_client,
    tenantid: str,
    propertymanagerid: str,
    propertyid: str,
    unit_id: str,
    upload_session_id: str,
    company_id: str,
    source_doc_name: str,
    qdrant_client,
    collection: str = QDRANT_COLLECTION,
    upsert_batch_size: int = UPSERT_BATCH_SIZE
) -> Tuple[int, float]:
    """
    Embeds chunks and upserts them to Qdrant in batches. Each chunk is still
    a separate point with its own payload.

    Returns:
        (inserted_points_count, total_embedding_cost)
    """
    total_cost = 0.0
    inserted = 0
    collection_ready = False

    batch_points: List[PointStruct] = []

    def _as_vector_list(v) -> List[float]:
        if isinstance(v, PointStruct):
            return list(v.vector or [])
        return list(v or [])

    for ch in chunks:
        page_number = int(ch["page"])
        chunk_index = int(ch["chunk_index"])
        chunk_text  = ch["text"]

        # Preferred signature: returns (PointStruct, cost)
        try:
            vector_data, embeddingcost = embed_files.EmbedFiles(
                embedding_client,
                chunk_text,
                tenantid,
                propertymanagerid,
                propertyid,
                unit_id,
                upload_session_id,
                page_number,              # 1-based page
                source_doc_name,
                chunk_index,
                company_id,
            )
            if isinstance(vector_data, PointStruct):
                point = vector_data
                # ensure it has an id (defensive)
                if not getattr(point, "id", None):
                    point.id = str(uuid.uuid4())
            else:
                # Legacy: returns a raw vector
                vec_list = _as_vector_list(vector_data)
                if not vec_list:
                    raise ValueError("Embedding returned empty vector (legacy path).")
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec_list,
                    payload={
                        "tenantid": tenantid,
                        "propertymanagerid": propertymanagerid,
                        "propertyid": propertyid,
                        "unitid": unit_id,
                        "upload_session_id": upload_session_id,
                        "company_id": company_id,
                        "pageNumber": page_number,
                        "chunk_index": chunk_index,
                        "source_doc": source_doc_name,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    },
                )
        except ValueError:
            # If your EmbedFiles uses ValueError to signal legacy tuple, fall back cleanly
            raw_vec, embeddingcost = embed_files.EmbedFiles(
                embedding_client,
                chunk_text,
                tenantid,
                propertymanagerid,
                propertyid,
                unit_id,
                upload_session_id,
                page_number,
                source_doc_name,
                chunk_index,
                company_id,
            )
            vec_list = _as_vector_list(raw_vec)
            if not vec_list:
                raise ValueError("Embedding returned empty vector (legacy path).")
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vec_list,
                payload={
                    "tenantid": tenantid,
                    "propertymanagerid": propertymanagerid,
                    "propertyid": propertyid,
                    "unitid": unit_id,
                    "upload_session_id": upload_session_id,
                    "company_id": company_id,
                    "pageNumber": page_number,
                    "chunk_index": chunk_index,
                    "source_doc": source_doc_name,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                },
            )

        total_cost += float(embeddingcost or 0.0)

        # Ensure collection exists once we know vector length
        if not collection_ready:
            vec_len = len(_as_vector_list(point))
            if vec_len <= 0:
                raise ValueError("Embedding returned empty vector.")
            ensure_collection_exists(collection, vec_len, qdrant_client)
            collection_ready = True

        # Add to batch (each point is a separate chunk/payload)
        batch_points.append(point)

        # Flush if batch full
        if len(batch_points) >= upsert_batch_size:
            upsert_points(collection, batch_points, qdrant_client)
            inserted += len(batch_points)
            batch_points.clear()

    # Flush remainder
    if batch_points:
        upsert_points(collection, batch_points, qdrant_client)
        inserted += len(batch_points)
        batch_points.clear()

    return inserted, total_cost

# ----------------------------- MAIN -----------------------------------
def runTextract(
    pdf: bytes | str,
    file_path: str,
    *,
    # ids passed through to EmbedFiles (and mirrored in payload)
    tenantid: str,
    propertymanagerid: str,
    propertyid: str,
    unit_id: str,
    upload_session_id: str,
    company_id: str,
    # clients
    embedding_client,
    qdrant_client,
    bucket,
    jobid
):
    """
    pdf: either bytes or path
    file_path: original path/name of the PDF; used as source_doc_name and for S3 async path
    """
    try:
        if isinstance(pdf, (bytes, bytearray)):
            pdf_bytes = bytes(pdf)
            filename = file_path
        else:
            filename = file_path
            with open(pdf, "rb") as f:
                pdf_bytes = f.read()

        # 1) OCR + chunk (no embedding here)
        ocr = run_ocr_and_chunk(pdf_bytes, filename)
        chunks = ocr["chunks"]

        # 2) Embed + batched upsert (each chunk = separate point/payload)
        inserted, emb_cost = embed_chunks_and_upsert(
            chunks,
            embedding_client=embedding_client,
            tenantid=tenantid,
            propertymanagerid=propertymanagerid,
            propertyid=propertyid,
            unit_id=unit_id,
            upload_session_id=upload_session_id,
            company_id=company_id,
            source_doc_name=file_path,
            qdrant_client=qdrant_client,
            collection=QDRANT_COLLECTION,
            upsert_batch_size=UPSERT_BATCH_SIZE,
        )

        result = {
            "ocr_status": ocr["job_status"],
            "pages_count": ocr["pages_count"],
            "chunks_count": ocr["chunks_count"],
            "embedding_cost": emb_cost,
            "upserted_points": inserted,
            "collection": QDRANT_COLLECTION,
        }
        print(json.dumps(result, indent=2))
        return result

    except botocore.exceptions.ClientError as e:
        print("AWS ClientError:", e.response.get("Error", {}))
        Clear_Uploads(bucket=bucket, job_id=jobid, job_status='error')
        raise
    except Exception as e:
        print("Failed:", e)
        # include file_path when available
        try:
            Clear_Uploads(bucket=bucket, job_id=jobid, file_path=file_path, job_status='error')
        except Exception:
            pass
        raise
