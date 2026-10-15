"""FastAPI app: upload, list, delete, ask. Also serves the plain-JS frontend at /."""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, answer, config, llm
from .service import DocStore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_docs"
FRONT = ROOT / "frontend"
MAX_BYTES = 15 * 1024 * 1024

app = FastAPI(title="DocQ", version=__version__)
store = DocStore()


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=config.TOP_K, ge=1, le=10)


@app.get("/api/health")
def health():
    r = store.retriever
    return {"status": "ok", "version": __version__, "retriever": r.name if r else store.mode,
            "embedding_model": config.EMBED_MODEL if (r and r.name != "bm25") else None,
            "vector_backend": getattr(r, "backend", None),
            "generator": "llm:" + config.LLM_MODEL if llm.enabled() else "extractive (no LLM configured)",
            "documents": len(store.docs)}


@app.get("/api/documents")
def documents():
    return store.list()


@app.post("/api/documents")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload").name
    if not name.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "Only .pdf, .txt and .md files are supported.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (15 MB max).")
    try:
        return store.add(name, data)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(422, f"Could not read this file: {e}")


@app.delete("/api/documents/{name}")
def delete(name: str):
    if not store.remove(name):
        raise HTTPException(404, "No such document.")
    return {"deleted": name}


@app.post("/api/load-samples")
def load_samples():
    out = []
    for p in sorted(SAMPLES.glob("*.pdf")):
        out.append(store.add(p.name, p.read_bytes()))
    return out


@app.post("/api/ask")
def ask(body: Ask):
    return answer.ask(store.retriever, body.question, body.k)


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


app.mount("/static", StaticFiles(directory=FRONT), name="static")
