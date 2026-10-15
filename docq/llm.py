"""Pluggable LLM generation via any OpenAI-compatible /chat/completions endpoint.

Enabled only when DOCQ_LLM_BASE_URL and DOCQ_LLM_MODEL are set (DOCQ_LLM_API_KEY optional,
e.g. for a local Ollama / llama.cpp server). Without them DocQ answers extractively.
"""
from __future__ import annotations
import re
import httpx

from . import config

SYSTEM = ("You answer questions using ONLY the numbered passages provided. "
          "Cite the passages you used like [1] or [2]. If the passages do not contain the answer, "
          "reply exactly: I could not find this in the documents. Be concise.")


class LLMError(RuntimeError):
    pass


def enabled() -> bool:
    return bool(config.LLM_BASE_URL and config.LLM_MODEL)


def generate(question: str, passages: list[str], *, base_url=None, api_key=None, model=None, client: httpx.Client | None = None) -> str:
    base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
    model = model or config.LLM_MODEL
    api_key = api_key if api_key is not None else config.LLM_API_KEY
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}]}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        c = client or httpx.Client(timeout=60)
        r = c.post(f"{base_url}/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:                                   # network, HTTP, schema
        raise LLMError(str(e)) from e


def cited_numbers(text: str, n: int) -> list[int]:
    nums = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text) if 1 <= int(x) <= n})
    return nums
