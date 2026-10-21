"""Document loading + chunking. Every chunk keeps (doc, page) so citations are exact."""
from __future__ import annotations
import io
import re
from dataclasses import dataclass, asdict

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    doc: str
    page: int
    idx: int          # chunk index inside the page
    text: str
    idx_global: int = -1   # position in the index (set by the retriever)

    @property
    def embed_text(self) -> str:
        # Contextual header: the file name gives the embedder/BM25 the document topic.
        return f"{self.doc.rsplit('.', 1)[0].replace('_', ' ')}. {self.text}"

    def to_dict(self):
        d = asdict(self); d.pop('idx_global', None); return d


def clean(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\r", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # de-hyphenate line wraps
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def extract_pages(filename: str, data: bytes) -> list[str]:
    """Return one string per page. PDFs use real pages; text files are paged by form-feed
    or, failing that, every ~2500 characters at a paragraph boundary."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [clean(p.extract_text() or "") for p in reader.pages]
    text = data.decode("utf-8-sig", errors="replace")   # -sig drops a leading BOM
    if "\f" in text:
        return [clean(p) for p in text.split("\f")]
    pages, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > 2500:
            pages.append(clean(cur)); cur = ""
        cur += para + "\n\n"
    if cur.strip():
        pages.append(clean(cur))
    return pages


def chunk_page(doc: str, page: int, text: str, max_words: int = 90) -> list[Chunk]:
    """Sentence-aligned chunks of <= max_words with a one-sentence overlap.
    Chunks never cross a page boundary."""
    sents = split_sentences(text)
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = cur[-1:], len(cur[-1].split())      # overlap: carry last sentence
        cur.append(s); n += w
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        chunks.append(" ".join(cur))
    return [Chunk(doc, page, i, c) for i, c in enumerate(chunks)]


def chunk_document(doc: str, pages: list[str], max_words: int = 90) -> list[Chunk]:
    out: list[Chunk] = []
    for pno, text in enumerate(pages, start=1):
        if text:
            out.extend(chunk_page(doc, pno, text, max_words))
    return out
