import os
import hashlib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-20250514"
PORT = int(os.environ.get("PORT", 8000))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_collection = None

def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        from chromadb import EmbeddingFunction, Embeddings

        class LocalEF(EmbeddingFunction):
            def name(self): return "local_ngram_embedding"
            def __call__(self, input: List[str]) -> Embeddings:
                return [self._embed(t) for t in input]
            def _embed(self, text, dim=384):
                text = text.lower()
                vec = np.zeros(dim)
                for n in [3, 4, 5]:
                    for i in range(len(text) - n + 1):
                        h = int(hashlib.md5(text[i:i+n].encode()).hexdigest(), 16)
                        vec[h % dim] += 1
                for w in text.split():
                    h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                    vec[h % dim] += 2
                norm = np.linalg.norm(vec)
                if norm > 0: vec = vec / norm
                return vec.tolist()

        db_path = os.environ.get("DB_PATH", "./astro_knowledge_db")
        client = chromadb.PersistentClient(path=db_path)
        _collection = client.get_collection(
            "astrology_knowledge",
            embedding_function=LocalEF()
        )
        print(f"✓ DB loaded: {_collection.count()} chunks")
    except Exception as e:
        print(f"⚠ DB not loaded: {e}")
    return _collection

SYSTEM = """You are AstroOracle — an expert astrologer and numerologist trained on the complete 12-volume Classical Astrology series by Sergei Vronsky and Matthew Goodwin's Numerology Complete Guide. You use Swiss Ephemeris precision for all calculations. Give detailed, authoritative readings drawing from classical Vronsky methodology."""

class Msg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[Msg]
    api_key: Optional[str] = None

class ChatRes(BaseModel):
    reply: str
    chunks_used: int

@app.get("/")
def root():
    col = get_collection()
    db_info = f"{col.count()} chunks" if col else "not loaded"
    return {"status": "AstroOracle running", "database": db_info}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatRes)
async def chat(req: ChatReq):
    api_key = req.api_key or ANTHROPIC_API_KEY
    user_msgs = [m for m in req.messages if m.role == "user"]
    latest = user_msgs[-1].content if user_msgs else ""

    knowledge = ""
    chunks_used = 0
    col = get_collection()
    if col and latest:
        try:
            results = col.query(
                query_texts=[latest],
                n_results=5,
                include=["documents", "metadatas"]
            )
            passages = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                src = meta.get("source", "").replace("_", " ").title()
                passages.append(f"[{src} | pg.{meta.get('page','?')}]\n{doc}")
            knowledge = "\n\n---\n\n".join(passages)
            chunks_used = len(passages)
        except Exception as e:
            print(f"RAG error: {e}")

    system = SYSTEM
    if knowledge:
        system += f"\n\n=== VRONSKY KNOWLEDGE BASE ===\n{knowledge}"

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": MODEL,
                "max_tokens": 2048,
                "system": system,
                "messages": messages
            }
        )

    data = r.json()
    reply = data["content"][0]["text"]
    return ChatRes(reply=reply, chunks_used=chunks_used)
