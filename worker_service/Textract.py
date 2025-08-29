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


AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")

S3_BUCKET = os.getenv("TEXTRACT_S3_BUCKET", "leaselinkleases")

# Optional: max characters per chunk (post-section split)
MAX_CHARS_PER_CHUNK = 4000  # set None to disable
TEXTRACT_DETECT_PRICE_PER_PAGE   = 0.015
TEXTRACT_STARTJOB_PRICE_PER_PAGE = 0.015

def estimate_textract_cost(mode: str, pages: int) -> float:
    """
    Rough per-page cost estimator so you can track OCR costs per document.
    Configure via env vars above; AWS does not return per-call cost in responses.
    """
    if pages <= 0:
        return 0.0
    if mode == "sync":
        return pages * TEXTRACT_DETECT_PRICE_PER_PAGE
    # "async" or anything else falls back to async price
    return pages * TEXTRACT_STARTJOB_PRICE_PER_PAGE

# ----------------------------- CHUNKING -------------------------------
section_regex = re.compile(
    r"^\s*(article\s+[ivx]+|section\s+\d+(\.\d+)*|\d+\.\d+|exhibit\s+[a-z])\b",
    re.IGNORECASE
)

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
    return [c for c in chunks if c.strip()]

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
    key = filename if filename.lower().endswith(".pdf") else f"{filename}.pdf"

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

# ----------------------------- QDRANT HELPERS -------------------------
def ensure_collection_exists(collection: str, vector_size: int, qdrant_client) -> None:
    try:
        qdrant_client.get_collection(collection_name=collection)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        # Add payload indexes here if desired.

def upsert_point(collection: str, point: PointStruct, qdrant_client) -> None:
    if point:
        qdrant_client.upsert(collection_name=collection, points=[point])

# ----------------------------- EMBED + UPSERT (ONE-AT-A-TIME) ---------
def _to_vec_list(v) -> List[float]:
    if isinstance(v, PointStruct):
        return list(v.vector or [])
    return list(v or [])

def embed_one_chunk_and_upsert(
    *,
    embedding_client,
    chunk_text: str,
    page_number: int,
    chunk_index: int,
    tenantid: str,
    propertymanagerid: str,
    propertyid: str,
    unit_id: str,
    upload_session_id: str,
    company_id: str,
    source_doc_name: str,
    qdrant_client,
    collection: str,
    ensure_collection_once: Dict[str, Any],
) -> float:
    """
    Embeds a single chunk and immediately upserts it as its own point/payload.
    Returns the embedding cost for this chunk.
    """

    # Preferred signature: returns (PointStruct, cost). Legacy: (vector_list, cost)
    vec_or_point, embeddingcost = embed_files.EmbedFiles(
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

    if isinstance(vec_or_point, PointStruct):
        point = vec_or_point
        vec_len = len(_to_vec_list(point))
        # Ensure collection exists once
        if not ensure_collection_once.get("done"):
            ensure_collection_exists(collection, vec_len, qdrant_client)
            ensure_collection_once["done"] = True
        # Immediate upsert: ONE chunk
       
        return float(embeddingcost or 0.0), point

# ----------------------------- OCR -> CHUNKS -> PER-CHUNK UPSERT ------
def runTextract(
    pdf: bytes | str,
    file_path: str,
    *,
    tenantid: str,
    propertymanagerid: str,
    propertyid: str,
    unit_id: str,
    upload_session_id: str,
    company_id: str,
    embedding_client,
    qdrant_client,
    bucket,
    jobid,
    collectionName
):
    """
    pdf: either bytes or a local path.
    file_path: used as source_doc_name and for S3 async naming.
    """
    try:
        # --- Load bytes
        if isinstance(pdf, (bytes, bytearray)):
            pdf_bytes = bytes(pdf)
            filename = file_path
        else:
            filename = file_path
            with open(pdf, "rb") as f:
                pdf_bytes = f.read()

        # --- OCR
        start = start_text_job(pdf_bytes, filename)
        if start["mode"] == "sync":
            status = "SUCCEEDED"
            blocks = start["blocks"]
            total_pages = start["pages"]
            ocr_mode = "sync"
        else:
            status = wait_for_text_job(start["job_id"])
            blocks = fetch_all_text_blocks(start["job_id"])
            total_pages = start["pages"]
            ocr_mode = "async"


        # --- Build page-wise text
        pages_text = build_pages_text_from_lines(blocks)

        # --- Chunk per page, and IMMEDIATELY embed + upsert ONE AT A TIME
        ensure_once = {"done": False}
        total_chunks = 0
        total_cost = 0.0
        points = []
        for page_num, page_text in enumerate(pages_text, start=1):
            sections = chunk_text_by_sections(page_text)
            final_chunks: List[str] = []
            for c in sections if sections else [page_text]:
                final_chunks.extend(split_long_chunk(c, MAX_CHARS_PER_CHUNK))

            for chunk_index, c in enumerate(final_chunks):
                cost, point = embed_one_chunk_and_upsert(
                    embedding_client=embedding_client,
                    chunk_text=c,
                    page_number=page_num,
                    chunk_index=chunk_index,
                    tenantid=tenantid,
                    propertymanagerid=propertymanagerid,
                    propertyid=propertyid,
                    unit_id=unit_id,
                    upload_session_id=upload_session_id,
                    company_id=company_id,
                    source_doc_name=file_path,
                    qdrant_client=qdrant_client,
                    collection=collectionName,
                    ensure_collection_once=ensure_once,

                )
                total_cost += cost
                total_chunks += 1
                points.append(point)
                if total_chunks >=20:
                    total_chunks = 0
                    for point in points:
                        upsert_point(collectionName, point, qdrant_client)
        if total_chunks > 0:
            for point in points:
                upsert_point(collectionName, point, qdrant_client)
        print("Success")
        ocr_cost = estimate_textract_cost(ocr_mode, total_pages)
        total_cost += ocr_cost
        return total_cost

    except botocore.exceptions.ClientError as e:
        print("AWS ClientError:", e.response.get("Error", {}))
        Clear_Uploads(bucket=bucket, job_id=jobid, job_status='error')
        raise
    except Exception as e:
        print("Failed:", e)
        try:
            Clear_Uploads(bucket=bucket, job_id=jobid, file_path=file_path, job_status='error')
        except Exception:
            pass
        raise
