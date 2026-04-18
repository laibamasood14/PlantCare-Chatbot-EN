import os
from pathlib import Path
from typing import List

import chromadb
import fitz
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


PDF_DIR = Path("pdfs")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "plantcare_docs"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(cleaned), step):
        end = start + chunk_size
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
    return chunks


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def main() -> None:
    os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")
    embedder = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in pdfs/. Nothing to ingest.")
        return

    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[dict] = []
    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(text)
        print(f"  Extracted {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            all_ids.append(f"{pdf_path.stem}_{idx}")
            all_docs.append(chunk)
            all_metas.append({"source": pdf_path.name, "chunk_index": idx})

        total_chunks += len(chunks)

    if all_docs:
        print("Generating embeddings...")
        embeddings = embedder.encode(all_docs, show_progress_bar=True, convert_to_numpy=True).tolist()

        # Clear old data to avoid duplicate entries when re-running ingestion.
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

        print("Writing to ChromaDB...")
        collection.add(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            embeddings=embeddings,
        )

    print(f"Ingestion complete. Total chunks stored: {total_chunks}")


if __name__ == "__main__":
    main()
