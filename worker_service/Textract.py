import os, time, json, re, uuid, boto3, botocore
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from PyPDF2 import PdfReader
from io import BytesIO
from qdrant_client.models import Distance, VectorParams, PointStruct
from common.cleanup_utils import Clear_Uploads
from collections import Counter
from qdrant_client.http.models import  Filter, FieldCondition, MatchValue, MatchAny
from common import Supabase_api




supabase = Supabase_api.supabase_client_setup()
bucket = 'lease-docs'

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
HEADER_RE = re.compile(
    r"^(ARTICLE|SECTION)\s+[IVXLC\d]+\.?\s+.*$|^ARTICLE\s+[IVXLC]+\.?$|^SECTION\s+\d+\.?$",
    re.IGNORECASE
)

money_label_re = re.compile(r".*:\s*\$\s*$")  # e.g. "CAM_PSF: $" or "CAM_PSF:    $

def is_incomplete_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    # Ends with colon but not colon-dollar
    if line.endswith(":") and not line.endswith(": $"):
        return True
    # Ends with lone dollar sign
    if line.endswith("$") and len(line) <= 2:
        return True
    return False


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Strong signal words
    if HEADER_RE.match(s):
        return True

    # Generic uppercase-ish heading heuristic
    letters = [ch for ch in s if ch.isalpha()]
    if len(letters) < 4:
        return False

    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    short = len(s) <= 70
    no_period_end = not s.endswith(".")  # headings often no period, but ARTICLE IV. RENT has one — so don't rely solely
    looks_like_heading = upper_ratio >= 0.85 and short

    # If it's all-caps and short, treat as heading even if it ends with a period
    return looks_like_heading

def glue_headings(lines: List[str]) -> List[str]:
    """
    If we find a heading line, append the next non-empty line to it (or even the next 2 lines).
    This prevents orphan headings like 'ARTICLE IV. RENT'.
    """
    out = []
    i = 0
    while i < len(lines):
        cur = (lines[i] or "").strip()
        if not cur:
            i += 1
            continue

        if is_heading(cur):
            # find next non-empty
            j = i + 1
            while j < len(lines) and not (lines[j] or "").strip():
                j += 1

            # If there's content after heading, glue it
            if j < len(lines):
                nxt = (lines[j] or "").strip()

                # Optionally glue 2nd line too if it's still "header-y" (e.g., subheading) or very short
                k = j + 1
                while k < len(lines) and not (lines[k] or "").strip():
                    k += 1
                glued = f"{cur}\n{nxt}"

                if k < len(lines):
                    nxt2 = (lines[k] or "").strip()
                    # If nxt is tiny or nxt2 continues the thought, glue nxt2 as well
                    if len(nxt) < 40 and len(nxt2) < 200:
                        glued = f"{glued}\n{nxt2}"
                        i = k
                    else:
                        i = j
                else:
                    i = j

                out.append(glued)
                i += 1
                continue

        out.append(cur)
        i += 1

    return out

def chunk_text_by_sections(text: str) -> List[str]:
    lines = text.splitlines()
    chunks, current = [], []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            print("Not Stripped")
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
    output = [p for p in parts if p]
    return output

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

def safe_page_count(pdf_bytes: bytes) -> int | None:
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        return len(reader.pages)
    except Exception as e:
        print(f"[Warn] Could not count pages with PyPDF2: {e}")
        return 0
# ----------------------------- TEXTRACT -------------------------------
def start_analysis_job(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    tx = textract_client()

    total_pages = safe_page_count(pdf_bytes)

    key = filename if filename.lower().endswith(".pdf") else f"{filename}.pdf"

    print("Uploading to s3")
    s3 = s3_client()
    s3.put_object(

        Bucket=S3_BUCKET,
        Key=key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )

    # ✅ TABLES + FORMS come from "DocumentAnalysis", not "TextDetection"
    resp = tx.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": S3_BUCKET, "Name": key}},
        FeatureTypes=["TABLES", "FORMS"],
    )

    return {"mode": "async", "job_id": resp["JobId"], "s3_key": key, "pages": total_pages}

def wait_for_analysis_job(job_id: str, poll_seconds: float = 2.0, max_wait_seconds: int = 1800) -> str:
    tx = textract_client()
    waited, backoff = 0.0, poll_seconds

    while True:
        jr = tx.get_document_analysis(JobId=job_id, MaxResults=1000)
        status = jr["JobStatus"]

        if status in ("SUCCEEDED", "FAILED", "PARTIAL_SUCCESS"):
            return status

        time.sleep(backoff)
        waited += backoff
        backoff = min(backoff * 1.25, 10)

        if waited >= max_wait_seconds:
            raise TimeoutError(f"Textract job {job_id} did not finish within {max_wait_seconds}s")

def fetch_all_analysis_blocks(job_id: str) -> List[Dict[str, Any]]:
    tx = textract_client()
    blocks: List[Dict[str, Any]] = []
    next_token = None

    while True:
        kwargs = {"JobId": job_id, "MaxResults": 1000}
        if next_token:
            kwargs["NextToken"] = next_token

        jr = tx.get_document_analysis(**kwargs)
        blocks.extend(jr.get("Blocks", []))

        next_token = jr.get("NextToken")
        if not next_token:
            break

    return blocks
def is_incomplete_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    # Ends with colon but not colon-dollar
    if line.endswith(":") and not line.endswith(": $"):
        return True
    # Ends with lone dollar sign
    if line.endswith("$") and len(line) <= 2:
        return True
    return False

def sort_lines(lines):
    return sorted(
        lines,
        key=lambda x: (round(float(x['bb'].get("Top", 0.0)), 3), float(x['bb'].get("Left", 0.0)))
    )

def overlap(a,b) -> bool:
    ax1, ay1 = float(a.get("Left", 0)), float(a.get("Top", 0))
    ax2, ay2 = ax1 + float(a.get("Width", 0)), ay1 + float(a.get("Height", 0))
    bx1, by1 = float(b.get("Left", 0)), float(b.get("Top", 0))
    bx2, by2 = bx1 + float(b.get("Width", 0)), by1 + float(b.get("Height", 0))

    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def filter_lines_outside_table(line_objs, table_blocks):
    table_bbs = []
    for t in table_blocks:
        bb=(t.get("Geometry", {}) or {}).get("BoundingBox", {}) or {}
        if bb: table_bbs.append(bb)
    if not table_bbs:
        return line_objs
    
    kept = []
    for lo in line_objs:
        lbb = lo["bb"] or {}
        if not lbb:
            kept.append(lo)
            continue
        if any(overlap(lbb, tbb) for tbb in table_bbs):
            continue
        kept.append(lo)
    return kept
def build_pages_structured(blocks: List[Dict[str,Any]]) -> List[Dict[str, Any]]:
    pages: Dict[int, Dict[str, Any]] = {}
    for b in blocks:
        page = b.get("Page", 1)
        pages.setdefault(page, {"lines": [], "tables": []})
        if b.get("BlockType") == "TABLE":

            pages[page]['tables'].append(b)
        elif b.get("BlockType") == "LINE":
            bb = (b.get("Geometry", {}) or {}).get("BoundingBox", {}) or {}
            pages[page]["lines"].append({
                "text": b.get("Text", "").strip(),
                "bb": bb})
    output = []
    for page_num in sorted(pages.keys()):
        raw_lines = pages[page_num]["lines"]


        line_objs = sort_lines(raw_lines)
        line_objs = filter_lines_outside_table(line_objs, pages[page_num]['tables'])
        raw_lines = [x['text'] for x in line_objs if x['text']]


        cleaned_lines = []
        i = 0
        label_re = re.compile(r".*:\s*$")
        while i < len(raw_lines):
            line = raw_lines[i]
            s = (line or "").strip()
            if not s:
                i +=1
                continue
            
            is_label = bool(label_re.match(s))
            is_money_label = bool(money_label_re.match(s))
            if is_label:
                j = i + 1

                while j < len(raw_lines) and not (raw_lines[j] or "").strip():
                    j +=1
                
                if j < len (raw_lines) and (raw_lines[j] or "").strip() == "$":
                    s = f"{s} $"
                    j += 1
                    while j < len(raw_lines) and not (raw_lines[j] or "").strip():
                        j +=1
                if j < len(raw_lines):
                    nxt = (raw_lines[j] or "").strip()
                    line = f"{s} {nxt}"
                    i = j
                else:
                    line = s

            cleaned_lines.append(line)
            i+=1
        cleaned_lines = glue_headings(cleaned_lines)

        if cleaned_lines or pages[page_num]['tables']:
            output.append({
                "page": page_num,
                "text": "\n".join(cleaned_lines),
                "tables": pages[page_num]['tables']
            })
    return output
    
def extract_table_text(table_block: dict, blocks_by_id: Dict[str, dict]) -> str:
    """
    Converts a Textract TABLE block into a readable text table.
    Output is row strings joined with " | ".
    """

    # 1) Get CELL blocks for this table
    cells = []
    for rel in table_block.get("Relationships", []) or []:
        if rel.get("Type") == "CHILD":
            for cid in rel.get("Ids", []) or []:
                cell = blocks_by_id.get(cid)
                if cell and cell.get("BlockType") == "CELL":
                    cells.append(cell)

    if not cells:
        return ""

    # 2) Build grid
    grid: Dict[tuple[int, int], str] = {}
    max_row, max_col = 0, 0

    for cell in cells:
        row = int(cell.get("RowIndex", 1))
        col = int(cell.get("ColumnIndex", 1))
        row_span = int(cell.get("RowSpan", 1) or 1)
        col_span = int(cell.get("ColumnSpan", 1) or 1)

        max_row = max(max_row, row + row_span - 1)
        max_col = max(max_col, col + col_span - 1)

        # Extract text from CHILD blocks (WORD + selection)
        parts = []
        for rel in cell.get("Relationships", []) or []:
            if rel.get("Type") == "CHILD":
                for wid in rel.get("Ids", []) or []:
                    w = blocks_by_id.get(wid)
                    if not w:
                        continue
                    bt = w.get("BlockType")
                    if bt == "WORD":
                        t = w.get("Text")
                        if t:
                            parts.append(t)
                    elif bt == "SELECTION_ELEMENT":
                        if w.get("SelectionStatus") == "SELECTED":
                            parts.append("[X]")

        text = " ".join(parts).strip()

        # Fill spanned cells (don’t overwrite existing text)
        for r in range(row, row + row_span):
            for c in range(col, col + col_span):
                if (r, c) not in grid or not grid[(r, c)]:
                    grid[(r, c)] = text

    # 3) Render rows
    rows = []
    for r in range(1, max_row + 1):
        row_cells = [grid.get((r, c), "") for c in range(1, max_col + 1)]
        # Optional: trim trailing empty columns for cleanliness
        while row_cells and row_cells[-1] == "":
            row_cells.pop()
        row_str = " | ".join(row_cells).strip()
        if row_str:
            rows.append(f"R{r}: {row_str}")

    # Optional: trim trailing empty rows
    out = "\n".join(rows).strip()
    if not out:
        return ""

    return "\n".join(rows)

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

def upsert_point(collection: str, point: PointStruct, qdrant_client, page_num) -> None:
    if point:
        response = qdrant_client.upsert(collection_name=collection, points=[point])
        
# ----------------------------- EMBED + UPSERT (ONE-AT-A-TIME) ---------
def _to_vec_list(v) -> List[float]:
    if isinstance(v, PointStruct):
        return list(v.vector or [])
    return list(v or [])

def embed_one_chunk(
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
    lease_id: str,
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
        lease_id
    )
    
    if isinstance(vec_or_point, PointStruct):
        point = vec_or_point
        vec_len = len(_to_vec_list(point))
        # Ensure collection exists once
        if not ensure_collection_once.get("done"):
            ensure_collection_exists(collection, vec_len, qdrant_client)
            ensure_collection_once["done"] = True

       
        return float(embeddingcost or 0.0), point
def textract_exists(object_path) -> bool:
    try:
        raw = supabase.storage.from_(bucket).download(object_path)
        if hasattr(raw, "decode"):
            raw_bytes = raw
        else: 
            raw_bytes = raw

        data = json.loads(raw_bytes.decode('utf-8'))

        if isinstance(data, dict) and "blocks" in data:
            data = data['blocks']
        if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
            raise ValueError(f"Unexpected Textract JSON Shape: {type(data)}")
        return True, data
    except Exception:
        return False, None
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
    jobid,
    collectionName,
    group_id,
    lease_id,
    resetSupabase = False
):
    """
    pdf: either bytes or a local path.
    file_path: used as source_doc_name and for S3 async naming.
    """

    supabase = Supabase_api.supabase_client_setup()
    try:
        folder_path = os.path.dirname(file_path)
        # --- Load bytes
        if isinstance(pdf, (bytes, bytearray)):
            pdf_bytes = bytes(pdf)
            filename = file_path
        else:
            filename = file_path
            with open(pdf, "rb") as f:
                pdf_bytes = f.read()
        textract, data = textract_exists(f"{folder_path}/{lease_id}_textract.json")
        # --- OCR
        if not textract:
            print("Starting OCR")
            start = start_analysis_job(pdf_bytes, filename)
            if start["mode"] == "sync":
                status = "SUCCEEDED"
                blocks = start["blocks"]
                total_pages = start["pages"]
                ocr_mode = "sync"
            else:
                status = wait_for_analysis_job(start["job_id"])
                print("Fetching All Blocks")
                blocks = fetch_all_analysis_blocks(start["job_id"])

                ocr_mode = "async"

            supabase.storage.from_(bucket).upload(
                path=f"{folder_path}/{lease_id}_textract.json",
                file=json.dumps(blocks).encode("utf-8"),
                file_options={
                    "content-type": "application/json",
                    "upsert": "true"
                }
            )
            print("Splitting Pages")
        else:
            blocks = data
        section_pages = build_pages_structured(blocks)
        total_pages = len(section_pages)
        
        # --- Chunk per page, and IMMEDIATELY embed + upsert ONE AT A TIME
        ensure_once = {"done": False}
        total_chunks = 0
        total_cost = 0.0

        blocks_by_id = {b["Id"]: b for b in blocks}
        all_table_text = []


        for page in section_pages:

            page_text = page['text']
            page_num = page['page']
            sections = chunk_text_by_sections(page_text)
            final_chunks: List[str] = []

            for c in sections if sections else [page_text]:

                final_chunks.extend(split_long_chunk(c, MAX_CHARS_PER_CHUNK))
            print(f"Final Chunks {len(final_chunks)}")
            
            for chunk_index, c in enumerate(final_chunks):
                
                if c and c.strip():
                    cost, point = embed_one_chunk(
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
                        lease_id=lease_id

                    )

                    total_cost += cost
                    
                    upsert_point(collectionName, point, qdrant_client, page_num)
                    total_chunks += 1
            for t_index, table_block in enumerate(page.get('tables', [])):
                table_text = extract_table_text(table_block, blocks_by_id)
                all_table_text.append([table_text, page])
                if table_text and table_text.strip():
                    cost, table_point = embed_one_chunk(
                        embedding_client=embedding_client,
                        chunk_text=table_text,
                        page_number=page_num,
                        chunk_index=t_index,
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
                        lease_id=lease_id
                    )
                    
                    total_cost += cost
                    total_chunks += 1
                    upsert_point(collectionName, table_point, qdrant_client, page_num)

        print("Success")
        print("Total Chunks", total_chunks)
        if not textract:
            ocr_cost = estimate_textract_cost(ocr_mode, total_pages)
        else: 
            ocr_cost = 0
        total_cost += ocr_cost
        return total_cost, total_pages

    except botocore.exceptions.ClientError as e:
        print("AWS ClientError:", e.response.get("Error", {}))
        #Clear_Uploads(job_id=jobid, file_path=file_path, job_status='error', group_id=group_id)
        raise
    except Exception as e:
        print("Failed:", e)
        try:
            print("Succeeded")
            #Clear_Uploads(job_id=jobid, file_path=file_path, job_status='error', group_id=group_id)
        except Exception:
            pass
        raise

