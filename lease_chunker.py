from pdf2image import convert_from_bytes 
from pytesseract import image_to_string
import re
import embed_files
from concurrent.futures import ThreadPoolExecutor
import psutil, os


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
    return '\n'.join(cleaned_lines)

def chunk(text):
    #Splits each text paragraph into a chunk if it has two \n new line texts
    chunks = [p.strip() for p in text.split("\n\n")]
    return chunks

# (Keep corrections, apply_corrections, is_gibberish, clean_ocr_text, and chunk as-is)

def process_page(img, page_number, client, tenantid, propertymanagerid, propertyid, unit_id, upload_session_id, source_doc_name, company_id):
    try:
        # Convert to grayscale and binarize
        gray = img.convert("L")
        binary = gray.point(lambda x: 0 if x < 180 else 255, '1')

        # OCR
        text = image_to_string(binary)
        clean_text = clean_ocr_text(text)
        chunks = chunk(clean_text)
        print('Memory (MB):', psutil.Process(os.getpid()).memory_info().rss/1024**2)
        vectors = []
        #Breaks down every chunk on given page and turns it into vector with a payload
        for chunk_index, chunk_text in enumerate(chunks):
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                print(f"Skipping invalid chunk at page {page_number}, index {chunk_index}")
                continue
                #Calls embed_files.EmbedFiles to use text-embedding-large (by OpenAI) to create vectors for each page
            vector_data = embed_files.EmbedFiles(
                client,
                chunk_text,
                tenantid,
                propertymanagerid,
                propertyid,
                unit_id,
                upload_session_id,
                page_number,
                source_doc_name,
                chunk_index,
                company_id
            )
            #add vector into vectors list
            vectors.append(vector_data)

        return vectors

    except Exception as e:
        print(f"Error processing page {page_number}: {e}")
        return []

def extract_text_from_pdf(pdf, client, tenantid, propertymanagerid, propertyid, unit_id, upload_session_id, source_doc_name, company_id):
    print("Converting Pdf to Images")

    #converts all bytes downloaded from supabase into images for tesseract to convert to text and embed
    images = convert_from_bytes(pdf, dpi=300)
    total_pages = len(images)

    print("Processing Images/Cleaning/Embedding with threading")

    all_vectors = []
    #Runs each images converted from bytes on seperate thread for efficiency and speed
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            #takes each page and submits it to process_page on seperate thread
            executor.submit(
                process_page,
                img,
                page_number,
                client,
                tenantid,
                propertymanagerid,
                propertyid,
                unit_id,
                upload_session_id,
                source_doc_name,
                company_id
            )
            #for loop runs each page simultaniously
            for page_number, img in enumerate(images, start=1)
        ]
        #Adds each future page into extended all_vectors as they are processed
        for future in futures:
            result = future.result()
            if result:
                all_vectors.extend(result)

    print("Image to Text success")
    return all_vectors, total_pages


