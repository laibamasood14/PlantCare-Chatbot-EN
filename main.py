import os
from typing import List

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
COLLECTION_NAME = "plantcare_docs"
TOP_K = 4

app = FastAPI(title="Plant Care RAG Chatbot")

embedder = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: List[str]


def build_prompt(user_message: str, context_chunks: List[str]) -> str:
    context_text = "\n\n".join(context_chunks)
    return (
        "You are a friendly and knowledgeable plant care expert.\n"
        "Always try to answer the user's plant care question helpfully.\n"
        "First prioritize the provided context if it's relevant.\n"
        "If the context doesn't cover the question, fall back on your own\n"
        "plant care knowledge to give a useful answer.\n"
        "Only say you don't know if the question is completely outside\n"
        "the domain of plant care.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User: {user_message}"
    )


@app.get("/")
async def read_root() -> FileResponse:
    try:
        return FileResponse("templates/index.html")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load UI.") from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        question = payload.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Message must not be empty.")

        query_embedding = embedder.encode([question], convert_to_numpy=True).tolist()[0]
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )

        documents = result.get("documents", [[]])
        chunks = documents[0] if documents else []
        if chunks is None:
            chunks = []

        prompt = build_prompt(question, chunks)

        try:
            llm_response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a friendly and knowledgeable plant care expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            answer = llm_response.choices[0].message.content or ""
        except Exception:
            answer = "I'm having trouble thinking right now, please try again!"

        source_previews = [chunk[:180] + ("..." if len(chunk) > 180 else "") for chunk in chunks]
        return ChatResponse(response=answer, sources=source_previews)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to process chat request.") from exc
