"""
Thin wrapper around LiteLLM for text embeddings. Swap the underlying model
by changing EMBEDDING_MODEL in your .env — no code changes needed.
"""
from __future__ import annotations

from typing import Any

import litellm

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

litellm.suppress_debug_info = True


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, splitting into API-friendly chunks."""
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response: Any = litellm.embedding(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(item["embedding"] for item in response.data)

    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    response: Any = litellm.embedding(model=EMBEDDING_MODEL, input=[text])
    return response.data[0]["embedding"]