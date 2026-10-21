import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from docq import answer, llm, config
from docq.api import app, store, SAMPLES
from docq.ingest import chunk_document, chunk_page, extract_pages, split_sentences
from docq.retrieval import BM25, BM25Retriever, build_retriever


def sample_chunks():
    out = []
    for p in sorted(SAMPLES.glob("*.pdf")):
        out += chunk_document(p.name, extract_pages(p.name, p.read_bytes()))
    return out


# ---------- ingestion / chunking
def test_pdf_pages_are_real_pages():
    pages = extract_pages("m.pdf", (SAMPLES / "Greenfield_Apartments_Resident_Handbook.pdf").read_bytes())
    assert len(pages) == 3
    assert "security deposit" in pages[0].lower() and "clubhouse" in pages[2].lower()


def test_chunks_keep_page_and_never_cross_pages():
    chunks = sample_chunks()
    assert chunks and all(c.page >= 1 for c in chunks)
    pages = extract_pages("a.pdf", (SAMPLES / "AquaPure_W200_User_Manual.pdf").read_bytes())
    for c in chunks:
        if c.doc.startswith("AquaPure"):
            assert c.text in pages[c.page - 1]


def test_text_file_utf8_bom_is_stripped():
    # Notepad and Excel often save UTF-8 with a BOM; it must not leak into the first chunk.
    data = "\ufeffRefunds are issued within 14 days.\n\nContact support for help.".encode("utf-8")
    pages = extract_pages("policy.txt", data)
    assert pages[0].startswith("Refunds")
    assert "\ufeff" not in chunk_document("policy.txt", pages)[0].text


def test_chunk_size_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(30))
    ch = chunk_page("d.txt", 1, text, max_words=40)
    assert len(ch) > 3 and all(len(c.text.split()) <= 48 for c in ch)
    assert ch[0].text.split(". ")[-1] in ch[1].text          # one-sentence overlap


def test_split_sentences():
    assert len(split_sentences("Rent is Rs 5.5k. Pay by the 5th. Done!")) == 3


def test_text_upload_pages():
    assert extract_pages("a.txt", b"one\n\ntwo") == ["one two"]
    assert extract_pages("a.txt", b"p1\fp2") == ["p1", "p2"]


# ---------- retrieval
def test_bm25_prefers_matching_doc():
    bm = BM25(["the cat sat on the mat", "quarterly revenue grew twelve percent", "dogs bark"])
    assert int(np.argmax(bm.scores("revenue growth percent"))) == 1


@pytest.fixture(scope="module")
def dense():
    return build_retriever(sample_chunks(), "dense", config.EMBED_MODEL)


def test_dense_finds_correct_page(dense):
    c, _ = dense.search("How long is the warranty on the AquaPure unit?", 1)[0]
    assert c.doc.startswith("AquaPure") and c.page == 4


def test_hybrid_and_bm25_modes():
    for mode in ("bm25", "hybrid"):
        r = build_retriever(sample_chunks(), mode, config.EMBED_MODEL)
        c, _ = r.search("late fee for rent paid after the 5th", 1)[0]
        assert c.doc.startswith("Greenfield") and c.page == 1


def test_bm25_fallback_when_no_embedder(monkeypatch):
    import docq.retrieval as R
    monkeypatch.setattr(R, "get_embedder", lambda *_: (_ for _ in ()).throw(ImportError("no st")))
    r = R.build_retriever(sample_chunks(), "auto")
    assert isinstance(r, BM25Retriever)


# ---------- answering
def test_extractive_answer_cites_page(dense):
    res = answer.ask(dense, "How long should I fast before a cholesterol test?")
    assert res["mode"] == "extractive" and not res["abstained"]
    assert any(c["doc"].startswith("Sunrise") and c["page"] == 3 for c in res["citations"])
    assert "10 to 12 hours" in res["answer"]


def test_abstains_on_unrelated_question(dense):
    res = answer.ask(dense, "What is the capital of France?")
    assert res["abstained"] and res["citations"] == []


def test_no_documents():
    assert answer.ask(None, "anything at all?")["abstained"]


class FakeResp:
    def __init__(self, content): self._c = content
    def raise_for_status(self): pass
    def json(self): return {"choices": [{"message": {"content": self._c}}]}


class FakeClient:
    def __init__(self, content): self.content, self.last = content, None
    def post(self, url, json=None, headers=None):
        self.last = (url, json, headers); return FakeResp(self.content)


def test_llm_plugin_request_format_and_citation_mapping(dense, monkeypatch):
    fc = FakeClient("The warranty is 24 months [2].")
    out = llm.generate("q?", ["p1", "p2"], base_url="http://x/v1", api_key="k", model="m", client=fc)
    assert fc.last[0] == "http://x/v1/chat/completions" and fc.last[2]["Authorization"] == "Bearer k"
    assert "[2] p2" in fc.last[1]["messages"][1]["content"]
    assert llm.cited_numbers(out, 2) == [2]
    # end-to-end through answer.ask with a mocked (NOT real) LLM
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://x/v1"); monkeypatch.setattr(config, "LLM_MODEL", "m")
    monkeypatch.setattr(llm, "generate", lambda q, p, **kw: "The unit has a 24-month warranty [1].")
    res = answer.ask(dense, "How long is the warranty on the AquaPure unit?")
    assert res["mode"] == "llm" and res["citations_verified"] and len(res["citations"]) == 1


def test_llm_failure_falls_back_to_extractive(dense, monkeypatch):
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://x/v1"); monkeypatch.setattr(config, "LLM_MODEL", "m")
    def boom(*a, **k): raise llm.LLMError("down")
    monkeypatch.setattr(llm, "generate", boom)
    res = answer.ask(dense, "How long is the warranty on the AquaPure unit?")
    assert res["mode"] == "extractive" and res["llm_error"] == "down"


# ---------- API
def test_api_flow():
    store.docs.clear(); store.retriever = None
    c = TestClient(app)
    assert c.get("/api/health").json()["documents"] == 0
    assert c.post("/api/ask", json={"question": "anything?"}).json()["abstained"]
    pdf = (SAMPLES / "Sunrise_Family_Clinic_Patient_Guide.pdf").read_bytes()
    r = c.post("/api/documents", files={"file": ("clinic.pdf", pdf, "application/pdf")})
    assert r.status_code == 200 and r.json()["pages"] == 3
    a = c.post("/api/ask", json={"question": "How much is the flu shot?"}).json()
    assert a["citations"][0]["page"] == 3 and "850" in a["answer"]
    assert c.get("/api/documents").json()[0]["name"] == "clinic.pdf"
    assert c.post("/api/documents", files={"file": ("x.exe", b"zz")}).status_code == 400
    assert c.post("/api/documents", files={"file": ("e.txt", b"   ")}).status_code == 422
    assert c.post("/api/ask", json={"question": "hi"}).status_code == 422          # too short
    assert c.delete("/api/documents/clinic.pdf").status_code == 200
    assert c.delete("/api/documents/clinic.pdf").status_code == 404
    assert "DocQ" in c.get("/").text
