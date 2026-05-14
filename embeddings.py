"""
Embedding helpers — dense (LiteLLM / OpenAI), sparse (FastEmbed SPLADE),
and cross-encoder re-ranking (FastEmbed TextCrossEncoder).

Swap models via env vars: EMBEDDING_MODEL, SPARSE_EMBEDDING_MODEL, RERANK_MODEL.
"""
from __future__ import annotations

import logging
from typing import Any

import litellm
from fastembed import SparseTextEmbedding, SparseEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    SPARSE_EMBEDDING_BATCH_SIZE,
    SPARSE_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Dense embeddings (OpenAI / LiteLLM)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sparse embeddings (FastEmbed SPLADE)
# ---------------------------------------------------------------------------

_sparse_model: SparseTextEmbedding | None = None


def init_sparse_model() -> None:
    """Load the SPLADE model. Call once during server startup."""
    global _sparse_model
    if _sparse_model is not None:
        return
    logger.info("Loading sparse model: %s", SPARSE_EMBEDDING_MODEL)
    _sparse_model = SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL)
    logger.info("Sparse model loaded.")


def sparse_embed_texts(texts: list[str]) -> list[SparseEmbedding]:
    """Embed a batch of texts into sparse (SPLADE) vectors."""
    assert _sparse_model is not None, "Call init_sparse_model() at startup"
    all_sparse: list[SparseEmbedding] = []

    for start in range(0, len(texts), SPARSE_EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + SPARSE_EMBEDDING_BATCH_SIZE]
        all_sparse.extend(_sparse_model.embed(batch, batch_size=len(batch)))

    return all_sparse


def sparse_embed_query(text: str) -> SparseEmbedding:
    """Embed a single query string into a sparse (SPLADE) vector."""
    assert _sparse_model is not None, "Call init_sparse_model() at startup"
    return list(_sparse_model.embed([text]))[0]


# ---------------------------------------------------------------------------
# Cross-encoder re-ranking (FastEmbed)
# ---------------------------------------------------------------------------

_rerank_model: TextCrossEncoder | None = None


def init_rerank_model() -> None:
    """Load the cross-encoder reranker. Call once during server startup."""
    global _rerank_model
    if _rerank_model is not None:
        return
    logger.info("Loading rerank model: %s", RERANK_MODEL)
    _rerank_model = TextCrossEncoder(model_name=RERANK_MODEL)
    logger.info("Rerank model loaded.")


def rerank(query: str, documents: list[str]) -> list[float]:
    """Score each document against the query using the cross-encoder.
    Returns relevance scores in the same order as the input documents."""
    assert _rerank_model is not None, "Call init_rerank_model() at startup"
    return list(_rerank_model.rerank(query, documents))