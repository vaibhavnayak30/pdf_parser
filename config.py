from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Embedding (LiteLLM)
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Qdrant
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "paper_chunks")

# MinIO
MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "research-papers")
MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Chunking
CHUNK_MAX_TOKENS: int = int(os.getenv("CHUNK_MAX_TOKENS", "512"))
SKIP_REFERENCE_SECTIONS: bool = (
    os.getenv("SKIP_REFERENCE_SECTIONS", "true").lower() == "true"
)

# Summarization
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "gpt-4o-mini")

# Feature flags
EXTRACT_IMAGES: bool = os.getenv("EXTRACT_IMAGES", "false").lower() == "true"
EXTRACT_TABLE_IMAGES: bool = (
    os.getenv("EXTRACT_TABLE_IMAGES", "false").lower() == "true"
)

_METADATA_MODEL: str = os.getenv("METADATA_MODEL", "gpt-4o-mini")

# Remote PDF ingestion
MAX_PDF_DOWNLOAD_BYTES: int = int(
    os.getenv("MAX_PDF_DOWNLOAD_BYTES", str(100 * 1024 * 1024))
)