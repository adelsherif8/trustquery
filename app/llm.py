"""Thin OpenAI wrapper with a deterministic MOCK fallback.

If OPENAI_API_KEY is set, we use real embeddings (text-embedding-3-small) and a
real chat model (gpt-4o-mini) with JSON output. If not, everything degrades to a
deterministic offline mode so the demo — and the evals — still run end to end.
The UI shows which mode is live.
"""

import os
import json
import math
import hashlib

CHAT_MODEL = os.getenv("TQ_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("TQ_EMBED_MODEL", "text-embedding-3-small")


def live() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def mode() -> str:
    return "openai" if live() else "mock"


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


# ── embeddings ───────────────────────────────────────────────────────────────
def embed(texts: list[str]) -> list[list[float]]:
    if live():
        resp = _get_client().embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    return [_mock_embed(t) for t in texts]


def _mock_embed(text: str, dim: int = 256) -> list[float]:
    """Deterministic hashed bag-of-words vector — good enough for offline ranking."""
    vec = [0.0] * dim
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if len(t) > 1]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ── chat (JSON) ──────────────────────────────────────────────────────────────
def chat_json(system: str, user: str) -> dict:
    """Return a parsed JSON object from the model (or raise). OpenAI only."""
    resp = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def chat_text(system: str, user: str) -> str:
    resp = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content
