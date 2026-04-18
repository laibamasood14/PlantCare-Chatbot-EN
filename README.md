# Plant Care RAG Chatbot

A lightweight, memory-efficient Retrieval-Augmented Generation chatbot for plant care questions using PDF knowledge bases.

## Stack

- FastAPI backend
- ChromaDB persistent vector store
- sentence-transformers (`all-MiniLM-L6-v2`) embeddings
- Groq API (`llama-3.1-8b-instant`) for generation
- PyMuPDF for PDF parsing
- Single-file HTML frontend

## Project Structure

```text
PlantCare Chatbot (EN)/
├── pdfs/
├── chroma_db/
├── main.py
├── ingest.py
├── templates/
│   └── index.html
├── requirements.txt
├── .env
└── render.yaml
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Add your key in `.env`:
   - `GROQ_API_KEY=your_actual_key`
4. Place PDF files inside `pdfs/`.

## Run Ingestion

Build the vector database once (or whenever your PDFs change):

```bash
python ingest.py
```

This script:
- extracts text from all PDFs in `pdfs/`
- chunks text into 400 chars with 50 overlap
- embeds chunks with `all-MiniLM-L6-v2`
- stores data in `chroma_db/`

## Run the App

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Deployment Steps (Render)

1. Run `python ingest.py` locally to build `chroma_db/` from your PDFs.
2. Commit `chroma_db/` to your repo (or use Render Disk as configured).
3. Push to GitHub.
4. Connect repo to Render; it auto-deploys via `render.yaml`.
5. Add `GROQ_API_KEY` in Render environment variables dashboard.

## Memory Notes

- Uses CPU-only torch from `https://download.pytorch.org/whl/cpu`.
- Embedding model is loaded once at startup.
- Chroma telemetry is disabled.
- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` is set in app and ingestion scripts.
- Retrieval is limited to top 4 chunks.
