"""
AWS Textract OCR pipeline — converts lease PDFs to embedded Qdrant vector chunks.

This module is the core document ingestion engine.  runTextract() is the main entry
point and performs the full pipeline:

  1. Cache check: looks for a previously stored Textract JSON output in Supabase
     Storage (bucket: lease-docs, path: <folder>/<lease_id>_textract.json).
     If found, the OCR step is skipped and the cached blocks are used directly.

  2. OCR: uploads the PDF to S3, starts an async Textract DocumentAnalysis job
     (TABLES + FORMS feature types), polls for completion, paginates through all
     result blocks, and caches the raw block JSON back to Supabase Storage.

  3. Page reconstruction (build_pages_structured):
     - Groups LINE and TABLE blocks by page number.
     - Sorts lines by their bounding-box top/left coordinates.
     - Filters out LINE blocks that overlap table regions (to avoid duplicate text).
     - Glues orphan headings to the next content line (glue_headings).
     - Joins label lines that end with ":" to their immediately following value.

  4. Chunking: each page's reconstructed text is split into sections using a regex
     that matches lease section headings (ARTICLE, SECTION, numbered clauses, exhibits).
     Sections longer than MAX_CHARS_PER_CHUNK are further split at word/newline
     boundaries.  Table text is embedded as its own separate chunk per page.

  5. Embedding and upsert (embed_one_chunk → embed_files.EmbedFiles):
     Each chunk is embedded immediately and upserted to Qdrant as an individual
     PointStruct with full metadata payload (tenant, property, unit, page, source_doc,
     company, lease_id, highlight_id, etc.).

  6. Cost tracking: returns (total_embedding_cost, total_pages) so the caller can
     persist costs to lease_documents.
"""

import os, time, json, re, uuid, boto3, botocore
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
from PyPDF2 import PdfReader
from io import BytesIO
from qdrant_client.models import Distance, VectorParams, PointStruct
from common.cleanup_utils import Clear_Uploads
from collections import Counter
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from common import Supabase_api


supabase = Supabase_api.supabase_client_setup()
bucket = 'lease-docs'

# Embedding module — EmbedFiles creates the OpenAI vector and returns a PointStruct.
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
    """Return True if the line looks like a label/heading that has no value yet.

    Detects lines ending with a bare colon (label pattern) or a lone "$" sign
    so they can be glued to the following value line during page reconstruction.
    """
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
import time
# Matches: "1.", "1.2", "1.2.3", "1.2.3.4" optionally followed by text
NUMBERED_SECTION_RE = re.compile(
    r'^(\d+\.)+(\d+)?'          # e.g. "1." or "1.2" or "1.2.3"
    r'(\s+[A-Z][A-Za-z0-9 \';,&/\-–—()]*)?$'  # optional title
)

# Matches: "ARTICLE I.", "ARTICLE XIV. TITLE", "SECTION 3."
ARTICLE_SECTION_RE = re.compile(
    r'^(ARTICLE|SECTION)\s+[IVXLCDM\d]+\.?\s*.*$',
    re.IGNORECASE
)

# Matches labeled exhibits: "Exhibit A", "EXHIBIT B-1"
EXHIBIT_RE = re.compile(
    r'^EXHIBIT\s+[A-Z0-9][-\w]*\.?',
    re.IGNORECASE
)

# Short all-caps line (no lowercase letters, short, typically a label)
# Excludes lines that look like body text (contain common body-text punctuation patterns)
ALLCAPS_LABEL_RE = re.compile(
    r'^[A-Z][A-Z0-9 &;:\'\-–/(),.]+$'
)

# Lines that are clearly NOT headings even if all-caps:
# - End with a comma (continuation)
# - Are very long (>90 chars)
# - Contain lowercase (mixed case body text)

def is_heading(line: str) -> bool:
    """
    Return True if the line is likely a document section heading.

    Prioritizes structural/syntactic signals over fragile heuristics.
    Designed for commercial lease documents but generalizes reasonably well.
    """
    s = line.strip()
    if not s:
        return False

    # --- Disqualifiers: things that look like headings but aren't ---

    # Too long to be a heading (body sentences run long)
    if len(s) > 120:
        return False

    # Ends with a comma — it's a continuation line, not a heading
    if s.endswith(','):
        return False

    # --- Strong structural signals ---

    if ARTICLE_SECTION_RE.match(s):
        return True

    if NUMBERED_SECTION_RE.match(s):
        return True

    if EXHIBIT_RE.match(s):
        return True

    # --- All-caps label heuristic (applied conservatively) ---
    # Only fires if the line is short, all-caps, and doesn't end with a period
    # followed by more text (which would indicate a sentence, not a label)
    if ALLCAPS_LABEL_RE.match(s):
        # Exclude if it ends with a period AND is longer than ~40 chars
        # (short period-ending labels like "WITNESSETH:" are fine)
        if s.endswith('.') and len(s) > 40:
            return False
        # Exclude if it looks like an enumeration item (single letter + period)
        if re.match(r'^[A-Z]\.$', s):
            return False
        # Require minimum length to avoid matching stray abbreviations
        if len(s) >= 4:
            return True

    return False

def glue_headings(lines: Dict[str, int], currentHeading: str = "") -> List[str]:
    """
    If we find a heading line, append the next non-empty line to it (or even the next 2 lines).
    This prevents orphan headings like 'ARTICLE IV. RENT'.
    """
    out = []
    i = 0
    
    while i < len(lines):
        chunks = ""
        cur = (lines[i]['text'] or "").strip()
        if not cur:
            i += 1
            continue
        if is_heading(cur):
            currentHeading = cur
        else:
            chunks += currentHeading + '\n'
            # find next non-empty
        j = i + 1
        chunks += cur + " "
        while j < len(lines) and lines[j]['text'] and not is_heading(lines[j]['text']):
            nxt = lines[j]['text'].strip()
            if nxt:
                chunks += nxt + " "
            j += 1

        out.append({'text': chunks.strip(), 'page_num': lines[i]['page_num']})
        i += 1
            

    return out, currentHeading

def chunk_text_by_sections(text: str) -> List[str]:
    """Split a page's text into logical chunks at lease section boundaries.

    Uses section_regex to detect Article/Section/Exhibit headings as split points.
    Returns a list of non-empty chunk strings.
    """
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

def split_long_chunk(c: str, limit: int, currentHeading: str = "") -> List[str]:
    """Split a single chunk into sub-chunks no longer than limit characters.

    Prefers to split at newline or space boundaries that fall in the last 40% of
    each sub-chunk to avoid cutting mid-word.  Returns [c] unchanged if c fits
    within the limit or limit is falsy.
    """
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
        parts.append(currentHeading + ": " + c[start:end].strip())
        start = end
    output = [p for p in parts if p]
    return output

# ----------------------------- CLIENTS --------------------------------
def s3_client() -> Any:
    """Return a boto3 S3 client authenticated with the configured AWS credentials."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def textract_client() -> Any:
    """Return a boto3 Textract client authenticated with the configured AWS credentials."""
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
    """Upload a PDF to S3 and start an async Textract DocumentAnalysis job.

    The job is configured with TABLES and FORMS feature types so that table cell
    text is extracted separately from line text.  Returns a dict containing the
    Textract job_id, the S3 key, the estimated page count, and mode='async'.
    """
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
    """Poll the Textract job until it finishes, using exponential back-off up to 10 seconds.

    Returns the final status string ('SUCCEEDED', 'FAILED', or 'PARTIAL_SUCCESS').
    Raises TimeoutError if max_wait_seconds is exceeded.
    """
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
    """Paginate through all Textract result blocks for a completed DocumentAnalysis job.

    Textract returns results in pages of up to 1000 blocks; this function collects
    them all into a flat list before returning.
    """
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
    """Sort LINE block objects by their bounding-box position: top-to-bottom, then left-to-right."""
    return sorted(
        lines,
        key=lambda x: (round(float(x['bb'].get("Top", 0.0)), 3), float(x['bb'].get("Left", 0.0)))
    )

def overlap(a,b) -> bool:
    """Return True if two Textract bounding boxes overlap (used to detect lines inside tables)."""
    ax1, ay1 = float(a.get("Left", 0)), float(a.get("Top", 0))
    ax2, ay2 = ax1 + float(a.get("Width", 0)), ay1 + float(a.get("Height", 0))
    bx1, by1 = float(b.get("Left", 0)), float(b.get("Top", 0))
    bx2, by2 = bx1 + float(b.get("Width", 0)), by1 + float(b.get("Height", 0))

    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

def filter_lines_outside_table(line_objs, table_blocks):
    """Remove LINE blocks that spatially overlap any TABLE block on the same page.

    Prevents duplicate text: table cell content is captured by extract_table_text
    and should not also appear in the main LINE-based text stream.
    """
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
    """Reconstruct page text from raw Textract blocks, cleaning up label/value fragmentation.

    Groups LINE and TABLE blocks by page, sorts lines spatially, removes lines that
    sit inside table regions, glues colon-terminated labels to their values, and
    then calls glue_headings to attach orphan heading lines to following content.
    Returns a list of page dicts: {'page': int, 'text': str, 'tables': list}.
    """
    pages: Dict[int, Dict[str, Any]] = {}
    total_pages = 0
    for b in blocks:
        page = b.get("Page", 1)
        pages.setdefault(page, {"lines": [], "tables": []})
        total_pages = max(total_pages, page)
        if b.get("BlockType") == "TABLE":

            pages[page]['tables'].append(b)
        elif b.get("BlockType") == "LINE":
            bb = (b.get("Geometry", {}) or {}).get("BoundingBox", {}) or {}
            pages[page]["lines"].append({
                "text": b.get("Text", "").strip(),
                "bb": bb})

    tables = []
    cleaned_lines = []
    for page_num in sorted(pages.keys()):
        raw_lines = pages[page_num]["lines"]


        line_objs = sort_lines(raw_lines)
        line_objs = filter_lines_outside_table(line_objs, pages[page_num]['tables'])
        raw_lines = [x['text'] for x in line_objs if x['text']]
        if(pages[page_num]['tables']):
            tables.append({"table": pages[page_num]['tables'], "page_num": page_num})


        
        i = 0
        label_re = re.compile(r".*:\s*$")
        currentHeading = ""
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

            cleaned_lines.append({"text": line, "page_num": page_num})
            i+=1
    cleaned_lines, currentHeading = glue_headings(cleaned_lines, currentHeading=currentHeading)


    return cleaned_lines, tables, total_pages, currentHeading
    
def extract_table_text(table_block: dict, blocks_by_id: Dict[str, dict]) -> str:
    """
    Converts a Textract TABLE block into a readable text table.
    Output is row strings joined with " | ".
    """

    # 1) Get CELL blocks for this table
    cells = []
    print("Extracting table text for table block:", table_block)
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
    """Upsert a single PointStruct to the specified Qdrant collection (no-op if point is None)."""
    if point:
        response = qdrant_client.upsert(collection_name=collection, points=[point])
        
# ----------------------------- EMBED + UPSERT (ONE-AT-A-TIME) ---------
def _to_vec_list(v) -> List[float]:
    """Coerce a PointStruct or raw vector value to a plain Python list of floats."""
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
    """Check whether a cached Textract JSON output exists in Supabase Storage.

    Downloads the object at object_path from the lease-docs bucket and validates
    that it contains a list of Textract block dicts.
    Returns (True, blocks_list) if found and valid, (False, None) otherwise.
    """
    try:
        print("Object Path", object_path)
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
    collectionName,
    lease_id,
):
    """
    pdf: either bytes or a local path.
    file_path: used as source_doc_name and for S3 async naming.
    """

    supabase = Supabase_api.supabase_client_setup()
    try:
        total_pages = 0
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
        cleaned_lines, tables, total_pages, currentHeading = build_pages_structured(blocks)
        
        # --- Chunk per page, and IMMEDIATELY embed + upsert ONE AT A TIME
        ensure_once = {"done": False}
        total_chunks = 0
        total_cost = 0.0

        blocks_by_id = {b["Id"]: b for b in blocks}
        all_table_text = []

        final_index = 1
        for line in cleaned_lines:
            page_text = line.get("text", "")
            page_num = line.get("page_num", 1)


            final_chunks: List[str] = []



            final_chunks.extend(split_long_chunk(page_text, MAX_CHARS_PER_CHUNK, currentHeading))
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
                    final_index = chunk_index + 1
        for table in tables:
            table_blocks = table.get("table", [])
            page_num = table.get("page_num", 1)

            for table_block in table_blocks:
                table_text = extract_table_text(table_block, blocks_by_id)


                all_table_text.append([table_text, table.get("page_num", 1)])
                if table_text and table_text.strip():
                    cost, table_point = embed_one_chunk(
                        embedding_client=embedding_client,
                        chunk_text=table_text,
                        page_number=page_num,
                        chunk_index=final_index,
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
                    final_index += 1
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


