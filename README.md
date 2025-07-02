# LeaseLink Backend 🏢🤖

This is the backend for **LeaseLink**, an AI-powered lease assistant designed to help property managers query, summarize, and extract insights from commercial lease documents.

Built with:
- 🧠 FastAPI
- 📄 OpenAI (GPT-4 + Embeddings)
- 📦 Supabase (for auth, storage, and message history)
- 🔍 Qdrant (for semantic search)
- 🧾 PDF + OCR (pdf2image, Tesseract)

---

## 🚀 Features

- Upload and process lease PDFs with OCR
- Embed lease chunks into Qdrant for semantic search
- Ask natural-language questions about a lease
- Returns responses with file references, page numbers, and optional highlights
- Supabase message history and session tracking
- Signed URLs for Bubble or React-based document preview

---

## 🛠️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/leaselink-backend.git
cd leaselink-backend

#Create a virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

#Install Dependencies
pip install -r requirements.txt

#Set up .env. DO NOT COMMIT .env to version control it's already in .gitignore
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_PUBLIC_API_KEY=...
SUPABASE_JWT=...

OPEN_AI_PROJECT_KEY=...


QDRANT_URL=...
QDRANT_API_KEY=...

PYTHON_EDGE_SECRET=...

#Run App Locally
uvicorn app:app --reload

#project structure
.
├── app.py                  # Main FastAPI app entrypoint
├── Supabase_api.py         # Supabase read/write helper functions
├── lease_chunker.py        # Chunking and metadata tagging
├── embed_files.py          # Embedding + vector creation for Qdrant
├── Qdrant_ChatGPT.py       # GPT + Qdrant chat flow logic
├── upload_lease_manager.py # PDF processing + routing
├── .env                    # 🔒 Environment variables (not committed)
├── requirements.txt        # Dependencies
├── .gitignore              # Git exclusions

#🧪 TODOs / Coming Soon
# Frontend switch to React

 #PDF highlight integration with pdf.js

#User roles and authentication (Bubble & Supabase sync)

 #Full-text search fallback

# Usage-based billing system

#🧠 Credits
#Maintained by @TylerGrace and team.
#Built to help property managers stop digging through massive leases.