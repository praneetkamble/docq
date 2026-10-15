"""Answer composition: extractive (default) or LLM (pluggable), always with page citations."""
from __future__ import annotations
import time

from . import config, llm
from .ingest import split_sentences
from .retrieval import Retriever

NOT_FOUND = "I could not find this in the uploaded documents."


def _cite(c, score):
    return {"doc": c.doc, "page": c.page, "score": round(float(score), 3), "text": c.text}


def ask(retriever: Retriever | None, question: str, k: int | None = None, min_score: float | None = None) -> dict:
    t0 = time.perf_counter()
    k = k or config.TOP_K
    min_score = config.MIN_SCORE if min_score is None else min_score
    if retriever is None or not retriever.chunks:
        return {"answer": "No documents are loaded yet. Upload a PDF or text file first.", "mode": "none",
                "abstained": True, "citations": [], "sources": [], "retriever": None, "latency_ms": 0}
    hits = retriever.search(question, k)
    conf = retriever.confidence(question)
    sources = [_cite(c, s) for c, s in hits]
    base = {"retriever": retriever.name, "confidence": round(conf, 3), "sources": sources}
    threshold = min_score if retriever.name != "bm25" else 0.5
    if conf < threshold:
        return {**base, "answer": NOT_FOUND, "mode": "abstain", "abstained": True, "citations": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    mode, err = "extractive", None
    if llm.enabled():
        try:
            passages = [f"({c.doc}, page {c.page}) {c.text}" for c, _ in hits]
            text = llm.generate(question, passages)
            nums = llm.cited_numbers(text, len(hits)) or []
            if text.lower().startswith("i could not find"):
                return {**base, "answer": NOT_FOUND, "mode": "llm", "abstained": True, "citations": [],
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            cites = [sources[n - 1] for n in nums]
            return {**base, "answer": text, "mode": "llm", "abstained": False,
                    "citations": cites or sources[:1],
                    "citations_verified": bool(nums),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        except llm.LLMError as e:
            err = str(e)                                       # fall back to extractive

    # ---- extractive: pick the best sentences from the retrieved chunks
    cand = []                                                  # (sentence, chunk, chunk_score)
    seen = set()
    for c, s in hits[:3]:
        for sent in split_sentences(c.text):
            if sent not in seen:
                seen.add(sent); cand.append((sent, c, s))
    sc = retriever.rank_texts(question, [x[0] for x in cand])
    order = sorted(range(len(cand)), key=lambda i: -sc[i])
    best = sc[order[0]] if order else 0
    chosen = [i for i in order[:2] if sc[i] >= best - 0.12 or i == order[0]]
    chosen.sort(key=lambda i: (cand[i][1].doc, cand[i][1].page))
    answer = " ".join(f"{cand[i][0]} [{cand[i][1].doc.rsplit('.',1)[0]}, p.{cand[i][1].page}]" for i in chosen)
    cites, seenc = [], set()
    for i in chosen:
        c = cand[i][1]
        if (c.doc, c.page) not in seenc:
            seenc.add((c.doc, c.page)); cites.append(_cite(c, cand[i][2]))
    out = {**base, "answer": answer, "mode": mode, "abstained": False, "citations": cites,
           "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if err:
        out["llm_error"] = err
    return out
