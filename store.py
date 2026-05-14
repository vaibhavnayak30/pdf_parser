"""
Qdrant vector store — collection setup, chunk indexing, and hybrid search.

Uses named vectors ("dense" + "sparse") with Reciprocal Rank Fusion (RRF)
to combine semantic similarity with keyword-level matching at query time.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastembed import SparseEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from config import EMBEDDING_DIMENSION, QDRANT_COLLECTION, QDRANT_URL

logger = logging.getLogger(__name__)

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


class VectorStore:
    """Manages a single Qdrant collection for paper chunk embeddings."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
        dimension: int = EMBEDDING_DIMENSION,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dimension = dimension
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        names = [c.name for c in self.client.get_collections().collections]

        if self.collection in names:
            if self._has_hybrid_schema():
                logger.info("Qdrant collection '%s' ready.", self.collection)
                return
            logger.warning(
                "Collection '%s' lacks hybrid vector schema — recreating. "
                "You will need to re-index all papers.",
                self.collection,
            )
            self.client.delete_collection(collection_name=self.collection)

        logger.info("Creating Qdrant collection '%s'.", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        self._create_payload_indexes()

    def _has_hybrid_schema(self) -> bool:
        """Return True if the collection already has both named vectors."""
        info = self.client.get_collection(collection_name=self.collection)
        vectors_cfg = info.config.params.vectors or {}
        sparse_cfg = info.config.params.sparse_vectors or {}
        return (
            _DENSE_VECTOR_NAME in vectors_cfg
            and _SPARSE_VECTOR_NAME in sparse_cfg
        )

    def _create_payload_indexes(self) -> None:
        keyword_fields = ["paper_id", "authors", "section_name", "source_filename"]
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="title",
            field_schema=TextIndexParams(
                type=TextIndexType.TEXT,
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=30,
                lowercase=True,
            ),
        )

        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="page_numbers",
            field_schema=PayloadSchemaType.INTEGER,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(
        self,
        paper_id: str,
        chunks: list[dict[str, Any]],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseEmbedding],
        meta: dict[str, Any],
    ) -> int:
        """Upsert embedded chunks with both dense and sparse vectors.
        Returns how many points were stored."""
        points: list[PointStruct] = []
        for i, (chunk, dense_vec, sparse_emb) in enumerate(
            zip(chunks, dense_embeddings, sparse_embeddings)
        ):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{paper_id}_{i}")
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        _DENSE_VECTOR_NAME: dense_vec,
                        _SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                    },
                    payload={
                        "paper_id": paper_id,
                        "title": meta.get("title", ""),
                        "authors": meta.get("authors", []),
                        "section_name": chunk.get("section_name", ""),
                        "chunk_index": i,
                        "chunk_text": chunk.get("text", ""),
                        "source_filename": meta.get("source_filename", ""),
                        "indexed_at": meta.get("indexed_at", ""),
                        "page_numbers": chunk.get("page_numbers", []),
                        "is_abstract": chunk.get("is_abstract", False),
                    },
                )
            )

        batch_size = 100
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
            )
        logger.info(
            "Stored %d chunks for paper %s in Qdrant.", len(points), paper_id
        )
        return len(points)

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        query_sparse: SparseEmbedding,
        limit: int = 5,
        paper_id: str | None = None,
        author: str | None = None,
        section: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions: list[FieldCondition] = []
        if paper_id:
            conditions.append(
                FieldCondition(key="paper_id", match=MatchValue(value=paper_id))
            )
        if author:
            conditions.append(
                FieldCondition(
                    key="authors",
                    match=MatchAny(any=[author]),
                )
            )
        if section:
            conditions.append(
                FieldCondition(
                    key="section_name", match=MatchValue(value=section)
                )
            )

        query_filter = Filter(must=conditions) if conditions else None
        logger.debug(
            "Hybrid search: limit=%d, filters=%s",
            limit,
            [c.key for c in conditions] if conditions else "none",
        )

        prefetch_limit = max(limit * 4, 20)

        hits = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=SparseVector(
                        indices=query_sparse.indices.tolist(),
                        values=query_sparse.values.tolist(),
                    ),
                    using=_SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=query_vector,
                    using=_DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        logger.info("Hybrid search returned %d results.", len(hits.points))
        return [
            {
                "score": hit.score,
                "paper_id": (hit.payload or {}).get("paper_id"),
                "title": (hit.payload or {}).get("title"),
                "authors": (hit.payload or {}).get("authors", []),
                "section": (hit.payload or {}).get("section_name"),
                "chunk_index": (hit.payload or {}).get("chunk_index"),
                "text": (hit.payload or {}).get("chunk_text"),
                "page_numbers": (hit.payload or {}).get("page_numbers", []),
            }
            for hit in hits.points
        ]