"""Retrievers: dense (MiniLM + FAISS), BM25 (pure python) and hybrid (reciprocal-rank fusion)."""
from __future__ import annotations
import math
import re
from collections import Counter
import numpy as np

from .ingest import Chunk

_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set("a an the of to in on at for and or is are was be it its this that with by from as do does how what when which who whom can i my you your we our there any if not no".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avg = (sum(len(t) for t in self.toks) / self.N) if self.N else 0.0
        df = Counter(w for t in self.toks for w in set(t))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N, dtype="float32")
        for i, tf in enumerate(self.tf):
            L = len(self.toks[i]) or 1
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / (self.avg or 1)))
            out[i] = s
        return out


class Embedder:
    """Wraps sentence-transformers. Raises ImportError if it is not installed."""
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        return np.asarray(v, dtype="float32")


def _topk(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
    return sorted(idx.tolist(), key=lambda i: -scores[i])


class Retriever:
    name = "base"
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError
    def rank_texts(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of arbitrary snippets (used to pick answer sentences)."""
        raise NotImplementedError
    # confidence in [0,1]-ish used for abstention; comparable across queries
    def confidence(self, query: str) -> float:
        raise NotImplementedError


class DenseRetriever(Retriever):
    name = "dense"
    def __init__(self, chunks, embedder: Embedder):
        super().__init__(chunks)
        self.emb = embedder
        self.mat = embedder.encode([c.embed_text for c in chunks]) if chunks else np.zeros((0, 384), "float32")
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.mat.shape[1]); self.index.add(self.mat); self.backend = "faiss"
        except ImportError:
            self.index = None; self.backend = "numpy"
    def _scores(self, query: str) -> np.ndarray:
        q = self.emb.encode([query])
        if self.index is not None:
            s, i = self.index.search(q, len(self.chunks))
            out = np.zeros(len(self.chunks), "float32"); out[i[0]] = s[0]; return out
        return (self.mat @ q[0])
    def search(self, query, k):
        s = self._scores(query)
        return [(self.chunks[i], float(s[i])) for i in _topk(s, k)]
    def rank_texts(self, query, texts):
        if not texts: return []
        return (self.emb.encode(texts) @ self.emb.encode([query])[0]).tolist()
    def confidence(self, query):
        s = self._scores(query); return float(s.max()) if len(s) else 0.0


class BM25Retriever(Retriever):
    name = "bm25"
    def __init__(self, chunks):
        super().__init__(chunks)
        self.bm = BM25([c.embed_text for c in chunks])
    def search(self, query, k):
        s = self.bm.scores(query)
        return [(self.chunks[i], float(s[i])) for i in _topk(s, k)]
    def rank_texts(self, query, texts):
        if not texts: return []
        return BM25(texts).scores(query).tolist()
    def confidence(self, query):
        # fraction of (non-stopword) query terms that occur anywhere in the corpus
        q = tokenize(query)
        if not q: return 0.0
        return sum(1 for w in q if w in self.bm.idf) / len(q)


class HybridRetriever(Retriever):
    """Reciprocal-rank fusion of dense and BM25 rankings (k=60)."""
    name = "hybrid"
    def __init__(self, chunks, embedder):
        super().__init__(chunks)
        self.dense = DenseRetriever(chunks, embedder)
        self.sparse = BM25Retriever(chunks)
        self.backend = self.dense.backend
    def search(self, query, k):
        n = len(self.chunks)
        rr: dict[int, float] = {}
        for ret in (self.dense, self.sparse):
            ranked = ret.search(query, n)
            for r, (c, _) in enumerate(ranked):
                rr[c.idx_global] = rr.get(c.idx_global, 0.0) + 1.0 / (60 + r + 1)
        top = sorted(rr, key=lambda i: -rr[i])[:k]
        dense_scores = {c.idx_global: s for c, s in self.dense.search(query, n)}
        # report the dense cosine as the displayed score (rank order is the fused one)
        return [(self.chunks[i], dense_scores[i]) for i in top]
    def rank_texts(self, query, texts):
        return self.dense.rank_texts(query, texts)
    def confidence(self, query):
        return self.dense.confidence(query)


_EMB: dict[str, Embedder] = {}

def get_embedder(model_name: str) -> Embedder:
    if model_name not in _EMB:
        _EMB[model_name] = Embedder(model_name)
    return _EMB[model_name]


def build_retriever(chunks: list[Chunk], mode: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Retriever:
    for i, c in enumerate(chunks):
        c.idx_global = i
    if mode in ("auto", "dense", "hybrid"):
        try:
            emb = get_embedder(model_name)
        except ImportError:
            if mode != "auto":
                raise
            return BM25Retriever(chunks)
        if mode == "dense":
            return DenseRetriever(chunks, emb)
        return HybridRetriever(chunks, emb)
    if mode == "bm25":
        return BM25Retriever(chunks)
    raise ValueError(f"unknown retriever mode: {mode}")
