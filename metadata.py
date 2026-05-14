"""
LLM-based metadata extraction for research papers.

Uses the configured metadata model to pull title, authors, and abstract
from the first page of a parsed Docling document.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import litellm

from config import _METADATA_MODEL

logger = logging.getLogger(__name__)

_METADATA_SYSTEM_PROMPT: str = """\
You are a research paper metadata extractor.  Given the raw items from the
first page of an academic PDF, return a JSON object with exactly these keys:

  {
    "title":    "<full paper title, without subtitle>",
    "authors":  ["First Last", ...],
    "abstract": "<the paper abstract text>",
    "publication_year": "<four-digit year or empty string>"
  }

Rules:
- "title" is the main paper title only.  Do NOT include a subtitle.
- "authors" is a JSON array of person names.  Strip affiliations, emails,
  superscript numbers, and markers like ∗ or †.  Return each name in its
  original casing.  If you cannot identify any authors return [].
- "abstract" is the full abstract text.  If there is no abstract return "".
- "publication_year" is the four-digit year the paper was published or
  submitted.  Look for dates in headers, footers, copyright lines, or
  conference / journal references.  If uncertain return "".
- Return ONLY valid JSON, no markdown fences, no commentary."""


def generate_paper_id(pdf_path: str) -> str:
    """SHA-256 of the file content, truncated to 16 hex chars. Same file
    always gives the same ID."""
    sha = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


def _collect_header_block(doc: Any, max_items: int = 20) -> str:
    """Grab the first few Docling items and format them as labelled lines
    for the LLM prompt."""
    lines: list[str] = []
    for i, (item, _level) in enumerate(doc.iterate_items()):
        if i >= max_items:
            break
        label = _label_str(item)
        text = _item_text(item) or ""
        if text:
            lines.append(f"[{label}] {text}")
    return "\n".join(lines)


def _llm_extract_metadata(doc: Any) -> dict[str, Any]:
    """Ask the configured metadata model to extract title / authors / abstract.
    Raises on failure."""
    logger.info("Extracting metadata via LLM (%s).", _METADATA_MODEL)
    header_text = _collect_header_block(doc)
    if not header_text:
        raise ValueError("No text items found on the first page of the document.")

    response: Any = litellm.completion(
        model=_METADATA_MODEL,
        messages=[
            {"role": "system", "content": _METADATA_SYSTEM_PROMPT},
            {"role": "user", "content": header_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1024,
    )

    raw: str | None = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM returned empty response for metadata extraction.")

    parsed: dict[str, Any] = json.loads(raw)

    title = (parsed.get("title") or "").strip()
    authors = parsed.get("authors") or []
    abstract = (parsed.get("abstract") or "").strip()
    publication_year = (parsed.get("publication_year") or "").strip()

    if not isinstance(authors, list):
        authors = []
    authors = [str(a).strip() for a in authors if str(a).strip()]

    logger.info(
        "Metadata extracted — title=%r, authors=%d, year=%s",
        title[:60] if title else "",
        len(authors),
        publication_year or "unknown",
    )
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "publication_year": publication_year,
    }


def extract_metadata(doc: Any, source_filename: str) -> dict[str, Any]:
    """Build a metadata dict from a Docling document using LLM extraction."""
    try:
        llm_meta = _llm_extract_metadata(doc)
        title = llm_meta["title"] or _title_from_filename(source_filename)
        authors = llm_meta["authors"]
        abstract = llm_meta["abstract"]
        publication_year = llm_meta["publication_year"]
    except Exception as exc:
        logger.warning("LLM metadata extraction failed: %s", exc)
        title = _title_from_filename(source_filename)
        authors = []
        abstract = ""
        publication_year = ""

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "publication_year": publication_year,
        "source_filename": source_filename,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


def _title_from_filename(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[_\-]+", " ", name).strip()


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------

def _label_str(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is None:
        return ""
    return str(label).rsplit(".", 1)[-1].strip().lower()


def _item_text(item: Any) -> str | None:
    if hasattr(item, "text") and item.text:
        return str(item.text).strip() or None
    if hasattr(item, "export_to_markdown"):
        try:
            md = item.export_to_markdown()
            return md.strip() or None
        except Exception:
            pass
    return None