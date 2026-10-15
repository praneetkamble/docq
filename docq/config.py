"""All runtime configuration comes from environment variables (no secrets in code)."""
import os

EMBED_MODEL = os.getenv("DOCQ_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# auto | dense | bm25 | hybrid.  auto = "hybrid" if sentence-transformers is importable, else "bm25".
RETRIEVER = os.getenv("DOCQ_RETRIEVER", "auto")
TOP_K = int(os.getenv("DOCQ_TOP_K", "5"))
# If the best dense cosine similarity is below this, DocQ says it cannot find the answer.
MIN_SCORE = float(os.getenv("DOCQ_MIN_SCORE", "0.25"))
# Optional LLM (any OpenAI-compatible /chat/completions endpoint). Unset => extractive answers.
LLM_BASE_URL = os.getenv("DOCQ_LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("DOCQ_LLM_API_KEY", "")
LLM_MODEL = os.getenv("DOCQ_LLM_MODEL", "")
CHUNK_WORDS = int(os.getenv("DOCQ_CHUNK_WORDS", "90"))
