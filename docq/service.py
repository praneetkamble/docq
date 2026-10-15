"""In-memory document store + index. Rebuilt when documents change (fine for small corpora)."""
from __future__ import annotations
import threading

from . import config
from .ingest import chunk_document, extract_pages
from .retrieval import Retriever, build_retriever


class DocStore:
    def __init__(self, mode: str | None = None):
        self.mode = mode or config.RETRIEVER
        self.docs: dict[str, list[str]] = {}      # filename -> pages
        self.retriever: Retriever | None = None
        self._lock = threading.Lock()

    def _rebuild(self):
        chunks = []
        for name, pages in self.docs.items():
            chunks.extend(chunk_document(name, pages, config.CHUNK_WORDS))
        self.retriever = build_retriever(chunks, self.mode, config.EMBED_MODEL) if chunks else None

    def add(self, filename: str, data: bytes) -> dict:
        pages = extract_pages(filename, data)
        if not any(pages):
            raise ValueError("No extractable text found (scanned PDFs need OCR, which DocQ does not do).")
        with self._lock:
            self.docs[filename] = pages
            self._rebuild()
        return self.describe(filename)

    def remove(self, filename: str) -> bool:
        with self._lock:
            if filename not in self.docs:
                return False
            del self.docs[filename]; self._rebuild(); return True

    def describe(self, filename: str) -> dict:
        pages = self.docs[filename]
        n = sum(1 for c in (self.retriever.chunks if self.retriever else []) if c.doc == filename)
        return {"name": filename, "pages": len(pages), "chunks": n}

    def list(self) -> list[dict]:
        return [self.describe(n) for n in self.docs]
