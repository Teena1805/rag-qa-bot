"""
generator.py — LLM answer generation from retrieved context.

Default provider: Ollama (FREE, runs locally — no API key needed).
Optional providers: OpenAI, Anthropic (require API keys in .env).
"""

from __future__ import annotations

from typing import List, Tuple

from chunking import Chunk
from config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)


# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful document Q&A assistant.

Your job is to answer the user's question using ONLY the context documents provided below.
- Answer clearly and concisely using the information in the context.
- If the context does not contain the answer, say: "I could not find an answer to that question in the provided documents."
- Do NOT refer to "Context 1", "Context 2", or any context numbers in your answer. Just answer naturally.
- Do NOT add any "Sources:", "References:", or citation lines at the end — citations are handled separately.
- Do NOT use your own training knowledge. Only use what is in the context.
"""


def _build_context_block(chunks: List[Tuple[Chunk, float]]) -> str:
    """Format retrieved chunks — no numbered labels so LLM won't reference them."""
    blocks = []
    for chunk, _ in chunks:
        page_info = f", page {chunk.page_hint}" if chunk.page_hint else ""
        header    = f"Source: {chunk.source}{page_info}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def _build_user_message(question: str, context: str) -> str:
    return (
        f"Here are the context documents you must use to answer the question:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Based ONLY on the context above, answer this question:\n"
        f"{question}"
    )


# ─── LLM Backends ─────────────────────────────────────────────────────────────

def _call_ollama(system: str, user_msg: str) -> str:
    import json as _json
    import urllib.request

    payload = _json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }).encode("utf-8")

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"].strip()
    except Exception as e:
        raise ConnectionError(
            f"\n❌ Could not reach Ollama at {OLLAMA_BASE_URL}\n\n"
            f"  Quick fix:\n"
            f"    1. Install Ollama: https://ollama.com/download\n"
            f"    2. Pull the model: ollama pull {OLLAMA_MODEL}\n"
            f"    3. Ollama starts automatically — retry your question.\n\n"
            f"  Or switch to OpenAI/Anthropic: set LLM_PROVIDER in your .env\n\n"
            f"  Original error: {e}"
        )


def _call_openai(system: str, user_msg: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai is required: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set in your .env file.\n"
            "Tip: Switch to the free local option — set LLM_PROVIDER=ollama in .env"
        )

    client   = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(system: str, user_msg: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic is required: pip install anthropic")

    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set in your .env file.\n"
            "Tip: Switch to the free local option — set LLM_PROVIDER=ollama in .env"
        )

    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


# ─── Provider registry ────────────────────────────────────────────────────────

_PROVIDERS = {
    "ollama":    _call_ollama,
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
}

_PROVIDER_LABELS = {
    "ollama":    f"Ollama / {OLLAMA_MODEL} (local, FREE — no API key)",
    "openai":    f"OpenAI / {OPENAI_MODEL}",
    "anthropic": f"Anthropic / {ANTHROPIC_MODEL}",
}


# ─── Public API ───────────────────────────────────────────────────────────────

class AnswerGenerator:
    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider.lower()
        if self.provider not in _PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'.\n"
                f"Choose from: {list(_PROVIDERS.keys())}"
            )
        print(f"  🤖 LLM: {_PROVIDER_LABELS[self.provider]}")

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[str, List[str]]:
        """Generate a grounded answer. Returns (answer_text, source_citations)."""
        if not retrieved_chunks:
            return (
                "I could not find an answer to that question in the provided documents.",
                [],
            )

        context  = _build_context_block(retrieved_chunks)
        user_msg = _build_user_message(question, context)
        answer   = _PROVIDERS[self.provider](SYSTEM_PROMPT, user_msg)
        sources  = _extract_sources(retrieved_chunks)

        return answer, sources


def _extract_sources(chunks: List[Tuple[Chunk, float]]) -> List[str]:
    """Return only the single highest-scoring source."""
    if not chunks:
        return []
    best_chunk, _ = max(chunks, key=lambda x: x[1])
    page_info     = f", page {best_chunk.page_hint}" if best_chunk.page_hint else ""
    return [f"{best_chunk.source}{page_info}"]