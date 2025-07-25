import gc
from pdf2image import convert_from_bytes 
from pytesseract import image_to_string
import re
import embed_files
from concurrent.futures import ThreadPoolExecutor
import psutil, os
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import time
from memory_profiler import profile
from cleanup_utils import Clear_Uploads

corrections = {
    "Shail":"Shall",
    "/f": "If",
    "Ali": "All",
    "Aritrate": "Arbitrate",
    "Rightof": "Right of",
    "/n" : "In",
    "Settie": "Settle",
    "(FF &E)": '("FF&E")',
    "Governmental!": "Governmental",
    "equipment-landiord": "equipment-landlord",
    "1°!" : "1st",
    "7," : "7.",
    "Fhis" : "This",
    "Titie" : "Title"
}
#Takes list of corrections and changes errors into what they should be
def apply_corrections(text):
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    return text

#cleans tesseract text to get rid of useless characters and extra line spaces
def is_gibberish(line):
    line = line.strip()
    
    #preserve Table of Contents entries (even without dot leaders)
    if re.match(r'^\d{1,2}(\.\d+)?\s+[A-Z].+', line):
        return False
    #If the entry contains the word "Exhibit" followed by 1 letter, it is not gibberish
    if re.match(r'^Exhibit\s+"?[A-Z]"?', line, re.IGNORECASE):
        return False
    #If the line has an uppercase character followed by uppercase letter, space, -, &, or ,. It is not gibberish
    if re.match(r'^[A-Z][A-Z\s\-&,]+\d{1,3}', line, re.IGNORECASE):
        return False
    #if the line has 5 or more periods in a row it is gibberish
    if re.search(r'(.)\1{5,}', line):
        return True
    #if the number of unique characters is less than 5 while the number of total characters is greater than 40 it is probably gibberish
    if len(set(line)) < 5 and len(line) > 40:
        return True
    #Gets the number of words in 1 line
    word_count = len(re.findall(r'\b\w+\b', line))
    #if the number of characters or lines is > 100 and the number of words is less than 3. It is gibberish
    if len(line) > 100 and word_count < 3:
            return True
    #Gets the percentage of symbols compared to the number of characters + 1
    symbol_ratio = len(re.findall(r'\W', line)) / (len(line) + 1)
    #if the number of words is less than 2 and the symbol ration is greater than .6 and the number of characters is greater than 30. It is gibberish
    if word_count <2 and symbol_ratio >.6 and len(line) >30:
        return True
    return False

#Takes the tesseract generated text and cleans it
def clean_ocr_text(text):
    #Applies corrections from above function
    cleaned_text = apply_corrections(text)
    #Splits text based on new lines
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    #Checks each line to see if it is gibberish and makes no sense (From Tesseract conversion)
    for line in lines:
        if not is_gibberish(line):
            cleaned_lines.append(line)
    #Joins Lines together
    del cleaned_text
    del lines
    return '\n'.join(cleaned_lines)

def chunk(text):
    #Splits each text paragraph into a chunk if it has two \n new line texts
    chunks = [p.strip() for p in text.split("\n\n")]
    return chunks

# (Keep corrections, apply_corrections, is_gibberish, clean_ocr_text, and chunk as-is)

@profile
def process_page(pdf, page_number, client, tenantid, propertymanagerid, propertyid, unit_id, upload_session_id, source_doc_name, company_id, qdrant_client, lease_id, bucket, file_path, dry_run=False):
    try:
        batch_size = 5
        # Extract only this page from the PDF
        reader = PdfReader(BytesIO(pdf))
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number])
        single_page_pdf = BytesIO()
        writer.write(single_page_pdf)

        # Convert just this page to an image
        image = convert_from_bytes(single_page_pdf.getvalue(), dpi=300)[0]  # ✅ Lower DPI for speed + memory

        # Convert to grayscale and binarize
       # gray = image.convert("L")
        #binary = gray.point(lambda x: 0 if x < 180 else 255, '1')

        # OCR
        text = image_to_string(image)
        clean_text = clean_ocr_text(text)
        chunks = chunk(clean_text)

        print(f"Processed page {page_number + 1}")
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        print('Memory (MB):', mem_mb)

        if mem_mb > 350:
            print("High Memory detected. Slowing Down")
            time.sleep(2)
        if mem_mb > 450:
            print("High Memory Detected > 450mb. Slowing Down Further")
            time.sleep(5)
            
        if mem_mb > 1000:
            Clear_Uploads(lease_id, bucket, file_path, 'Memory over 1 gigabyte')


        vectors = []
        for chunk_index, chunk_text in enumerate(chunks):
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                continue
            if dry_run:
                print(f"[Dry Run] Page {page_number+1} - Chunk {chunk_index}: {chunk_text[:80]}...\n")
                continue
            vector_data = embed_files.EmbedFiles(
                client,
                chunk_text,
                tenantid,
                propertymanagerid,
                propertyid,
                unit_id,
                upload_session_id,
                page_number + 1,  # Display page as 1-based
                source_doc_name,
                chunk_index,
                company_id
            )
            vectors.append(vector_data)
            #Uploades Vectors into qdrant once 10 or more are active
            if len(vectors) >= batch_size:
                print("Uploading to Qdrant")
                qdrant_client.upsert(collection_name="Test-Leases", points=vectors)
                vectors.clear()
                gc.collect()
            del vector_data
        image.close()
        del image#, gray, binary
        gc.collect()

        return vectors

    except Exception as e:
        print(f"Error processing page {page_number + 1}: {e}")
        return []

def extract_text_from_pdf(pdf, client, tenantid, propertymanagerid, propertyid, unit_id, upload_session_id, source_doc_name, company_id, lease_id, bucket, file_path, qdrant_client):
    reader = ''
    try:
        reader = PdfReader(BytesIO(pdf))
        total_pages = len(reader.pages)
    except Exception as e:
        print("Could not read pages", e)
    print("Processing each page with threading (one at a time in memory)")

    all_vectors = []
    #Runs each images converted from bytes on seperate thread for efficiency and speed
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    process_page,
                    pdf,  # Full byte stream
                    page_number,
                    client,
                    tenantid,
                    propertymanagerid,
                    propertyid,
                    unit_id,
                    upload_session_id,
                    source_doc_name,
                    company_id,
                    qdrant_client,
                    lease_id,
                    bucket,
                    file_path
                )
                for page_number in range(total_pages)
            ]
        for future in futures:
            result = future.result()
            if result:
                all_vectors.extend(result)

        print("Image to Text success")
        return all_vectors, total_pages
    except Exception as e:
        print("Error Getting Vector. Deleting Files from supabase", e)
        Clear_Uploads(lease_id, bucket, file_path, e)



