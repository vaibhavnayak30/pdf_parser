"""
MinIO wrapper for persisting raw PDFs, parsed markdown, and metadata.
Also hands out presigned download URLs.
"""
from __future__ import annotations

import io
import json
from datetime import timedelta
from typing import Any

from minio import Minio
from minio.error import S3Error

from config import (
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
)


class ObjectStore:
    """Single-bucket MinIO client for the paper data lake."""

    def __init__(
        self,
        endpoint: str = MINIO_ENDPOINT,
        access_key: str = MINIO_ACCESS_KEY,
        secret_key: str = MINIO_SECRET_KEY,
        bucket: str = MINIO_BUCKET,
        secure: bool = MINIO_SECURE,
    ) -> None:
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )
        self.bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    # Uploads
    def upload_pdf(self, paper_id: str, pdf_path: str) -> str:
        object_name = f"{paper_id}/raw.pdf"
        self.client.fput_object(
            self.bucket, object_name, pdf_path, content_type="application/pdf"
        )
        return object_name

    def upload_markdown(self, paper_id: str, markdown_text: str) -> str:
        object_name = f"{paper_id}/parsed.md"
        data = markdown_text.encode("utf-8")
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="text/markdown",
        )
        return object_name

    def upload_metadata(self, paper_id: str, metadata: dict[str, Any]) -> str:
        object_name = f"{paper_id}/metadata.json"
        data = json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            len(data),
            content_type="application/json",
        )
        return object_name

    # Reads

    def paper_exists(self, paper_id: str) -> bool:
        try:
            self.client.stat_object(self.bucket, f"{paper_id}/raw.pdf")
            return True
        except S3Error:
            return False

    def get_metadata(self, paper_id: str) -> dict[str, Any] | None:
        response = None
        try:
            response = self.client.get_object(
                self.bucket, f"{paper_id}/metadata.json"
            )
            return json.loads(response.read().decode("utf-8"))
        except S3Error:
            return None
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def get_markdown(self, paper_id: str) -> str | None:
        response = None
        try:
            response = self.client.get_object(
                self.bucket, f"{paper_id}/parsed.md"
            )
            return response.read().decode("utf-8")
        except S3Error:
            return None
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def get_download_url(
        self, paper_id: str, file_type: str = "pdf", expires_hours: int = 1
    ) -> str:
        """Presigned GET URL, valid for *expires_hours*."""
        object_name = (
            f"{paper_id}/raw.pdf" if file_type == "pdf" else f"{paper_id}/parsed.md"
        )
        return self.client.presigned_get_object(
            self.bucket, object_name, expires=timedelta(hours=expires_hours)
        )

    # Listing

    def list_papers(self) -> list[dict[str, Any]]:
        """Return metadata for every paper in the bucket."""
        papers: list[dict[str, Any]] = []
        seen: set[str] = set()
        for obj in self.client.list_objects(self.bucket, recursive=False):
            if obj.is_dir:
                paper_id = obj.object_name.rstrip("/")
                if paper_id not in seen:
                    seen.add(paper_id)
                    meta = self.get_metadata(paper_id)
                    if meta:
                        papers.append(meta)
        return papers