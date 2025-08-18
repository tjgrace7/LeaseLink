import gc
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
import re
from . import embed_files
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import time
from common.cleanup_utils import Clear_Uploads
from datetime import datetime
import os
from openai import OpenAI  # used inside child
import threading

# --------------------- corrections & helpers (unchanged) ---------------------
corrections = {
    "Shail": "Shall",
    "/f": "If",
    "Ali": "All",
    "Aritrate": "Arbitrate",
    "Rightof": "Right of",
    "/n": "In",
    "Settie": "Settle",
    '(FF &E)': '("FF&E")',
    "Governmental!": "Governmental",
    "equipment-landiord": "equipment-landlord",
    "1°!": "1st",
    "7,": "7.",
    "Fhis": "This",
    "Titie": "Title",
}

def apply_corrections(text):
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

def is_gibberish(line):
    line = line.strip()
    if re.match(r'^\d{1,2}(\.\d+)?\s+[A-Z].+', line): return False
    if re.match(r'^Exhibit\s+"?[A-Z]"?', line, re.IGNORECASE): return False
    if re.match(r'^[A-Z][A-Z\s\-&,]+\d{1,3}', line, re.IGNORECASE): return False
    if re.search(r'(.)\1{5,}', line): return True
    if len(set(line)) < 5 and len(line) > 40: return True
    word_count = len(re.findall(r'\b\w+\b', line))
    if len(line) > 100 and word_count < 3: return True
    symbol_ratio = len(re.findall(r'\W', line)) / (len(line) + 1)
    if word_count < 2 and symbol_ratio > .6 and len(line) > 30: return True
    return False

EMBEDDING_CLASSES = {
    "financial": [
        "base rent monthly",
        "rent escalation",
        "security deposit",
        "rent",
        "operating expenses",
        "cam",
        "tenant",
        "insurance",
        "start_date",
        "common area maintenance",
        "tenant_reimbursements",
        "delivery possession date",
        "rent abatement",
        "rent commencement",
    ],
    "term": [
        "address",
        "suite",
        "term",
        "length",
        "month",
        "renewal",
        "renewal notice",
        "termination_rights",
        "expansion rights",
        "shrinkage rights",
        "contraction rights",
        "co tenancy",
        "purchase options",
        "square footage",
        "rentable square footage",
        "premises",
        "parking",
        "storage",
        "maintenance",
        "hvac",
        "utility",
        "default",
        "assignment",
        "subletting",
        "indemnity",
        "force majeure",
        "estoppel",
        "signage",
        "permitted use",
        "exclusive_use",
        "guarantor",
        "tenant improvement",
        "holdover terms",
        "landlord work",
        "tenant work",
        "security deposit term",
        "right of first refusal",
        "rofr",
        "right of first offer",
        "rofo",
        "security access",
        "exclusivity",
    ],
}

def classify_chunk(text):
    text_lower = text.lower()
    for label, keywords in EMBEDDING_CLASSES.items():
        if any(k in text_lower for k in keywords):
            return label
    return "general"

section_regex = re.compile(r'^\d+(\.\d+)?\s+[A-Z \-]+:?')

def chunk(text):
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    start_time = datetime.now()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if section_regex.match(stripped):
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [stripped]
            else:
                current_chunk.append(stripped)
        else:
            current_chunk.append(stripped)
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    end_time = datetime.now()
    print(f"Chunking time: {(end_time - start_time).total_seconds():.3f} seconds")
    return chunks

# --------------------- child-process worker ---------------------

def _make_openai_client(api_key: str):
    return OpenAI(api_key=api_key)

def process_page(
    pdf: bytes,
    page_number: int,
    openai_api_key: str,
    tenantid: str,
    propertymanagerid: str,
    propertyid: str,
    unit_id: str,
    upload_session_id: str,
    source_doc_name: str,
    company_id: str,
):
    """
    Runs in a child process:
      - Extract page
      - Try text-layer extraction first; fall back to OCR
      - Chunk + build serializable vector payloads using embed_files
    Returns: (embedding_cost: float, points: list[dict])
             where each point is {"id": str, "vector": list[float], "payload": dict}
    """
    try:
        tname = threading.current_thread().name  # mostly "MainThread" in subprocess
        print(f"[pid={os.getpid()}] start page {page_number+1}")

        # Extract a single page PDF as bytes
        reader = PdfReader(BytesIO(pdf))
        page = reader.pages[page_number]
        text_layer = page.extract_text() or ""
        del reader

        if text_layer.strip():
            # Use the text layer—much faster than OCR
            page_text = text_layer
        else:
            # Fall back to OCR
            writer = PdfWriter()
            writer.add_page(page)
            single_page_pdf = BytesIO()
            writer.write(single_page_pdf)
            del writer, page

            # Slightly reduced DPI to balance speed and quality
            image = convert_from_bytes(single_page_pdf.getvalue(), dpi=250)[0]
            del single_page_pdf

            page_text = image_to_string(image, config="--psm 6")
            image.close()
            del image

        chunks = chunk(page_text)
        del page_text

        client = _make_openai_client(openai_api_key)

        points = []
        embedding_cost = 0.0

        for chunk_index, chunk_text in enumerate(chunks):
            if not (isinstance(chunk_text, str) and chunk_text.strip()):
                continue
            chunk_class = classify_chunk(chunk_text)

            # Let embed_files create something like a PointStruct; convert to dict for pickling
            vector_data, embeddingcost = embed_files.EmbedFiles(
                client,
                chunk_text,
                tenantid,
                propertymanagerid,
                propertyid,
                unit_id,
                upload_session_id,
                page_number + 1,
                source_doc_name,
                chunk_index,
                company_id,
                chunk_class,
            )
            embedding_cost += float(embeddingcost or 0.0)

            # Convert PointStruct -> serializable dict (id, vector, payload)
            # Handle both dict-like and object-like returns
            if isinstance(vector_data, dict):
                pt = {
                    "id": vector_data.get("id"),
                    "vector": vector_data.get("vector"),
                    "payload": vector_data.get("payload"),
                }
            else:
                # object with attrs
                pt = {
                    "id": getattr(vector_data, "id"),
                    "vector": getattr(vector_data, "vector"),
                    "payload": getattr(vector_data, "payload"),
                }
            points.append(pt)

            # Free per-chunk memory
            del vector_data

        gc.collect()
        print(f"[pid={os.getpid()}] done page {page_number+1} (points={len(points)})")
        return embedding_cost, points

    except Exception as e:
        print(f"Error processing page {page_number + 1}: {e}")
        return 0.0, []

# --------------------- parent-side orchestrator ---------------------

def extract_text_from_pdf(
    pdf: bytes,
    openai_api_key: str,
    tenantid: str,
    propertymanagerid: str,
    propertyid: str,
    unit_id: str,
    upload_session_id: str,
    source_doc_name: str,
    company_id: str,
    job_id: str,
    bucket: str,
    file_path: str,
    qdrant_client,  # parent-only client
    job_status: dict,
    collectionName: str,
    total_pages: int,
):
    """
    Parent process:
      - spawns a process pool (spawn)
      - collects (cost, points) from children
      - batches Qdrant upserts
    """
    total_embedding_cost = 0.0
    batched_points = []
    batch_size = int(os.getenv("QDRANT_UPSERT_BATCH", "200"))
    workers = int(os.getenv("OCR_WORKERS", max(1, (os.cpu_count() or 2) - 1)))

    print(f"Launching OCR with {workers} worker processes over {total_pages} pages")

    try:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        futures = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for page_number in range(total_pages):
                f = pool.submit(
                    process_page,
                    pdf,
                    page_number,
                    openai_api_key,
                    tenantid,
                    propertymanagerid,
                    propertyid,
                    unit_id,
                    upload_session_id,
                    source_doc_name,
                    company_id,
                )
                futures.append(f)

            for i, f in enumerate(as_completed(futures), 1):
                try:
                    cost, points = f.result()
                    total_embedding_cost += float(cost or 0.0)

                    if points:
                        batched_points.extend(points)
                        print(batched_points)
                        if len(batched_points) >= batch_size:
                            qdrant_client.upsert(collection_name=collectionName, points=batched_points)
                            batched_points.clear()
                            gc.collect()

                    print(f"[{i}/{total_pages}] page done; running total cost: {total_embedding_cost:.4f}")

                except Exception as e:
                    print("Page task failed:", e)

        # flush any remaining points
        if batched_points:
            print(batched_points)
            qdrant_client.upsert(collection_name=collectionName, points=batched_points)
            batched_points.clear()

        print("Image to Text success")
        return total_embedding_cost

    except Exception as e:
        print("Error Getting Vector. Deleting Files from supabase", e)
        job_status['status'] = 'error'
        job_status['error'] = e
        Clear_Uploads(job_id, bucket, file_path, job_status)
        return 0.0
