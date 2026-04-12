"""
Qdrant vector store — collection setup, chunk indexing, and semantic search.
"""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from config import EMBEDDING_DIMENSION, QDRANT_COLLECTION, QDRANT_URL


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

    def _ensure_collection(self) -> None:
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection in names:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.dimension, distance=Distance.COSINE
            ),
        )
        self._create_payload_indexes()

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

    # Indexing chunks
    def index_chunks(
        self,
        paper_id: str,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        meta: dict[str, Any],
    ) -> int:
        """Upsert embedded chunks. Returns how many points were stored."""
        points: list[PointStruct] = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{paper_id}_{i}")
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
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
        return len(points)

    # Search chunks
    def search(
        self,
        query_vector: list[float],
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

        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

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