from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

import litellm

from config import _METADATA_MODEL

_METADATA_SYSTEM_PROMPT: str = """\
You are a research paper metadata extractor.  Given the raw items from the
first page of an academic PDF, return a JSON object with exactly these keys:

  {
    "title":    "<full paper title, without subtitle>",
    "authors":  ["First Last", ...],
    "abstract": "<the paper abstract text>"
  }

Rules:
- "title" is the main paper title only.  Do NOT include a subtitle.
- "authors" is a JSON array of person names.  Strip affiliations, emails,
  superscript numbers, and markers like ∗ or †.  Return each name in its
  original casing.  If you cannot identify any authors return [].
- "abstract" is the full abstract text.  If there is no abstract return "".
- Return ONLY valid JSON, no markdown fences, no commentary."""


def generate_paper_id(pdf_path: str) -> str:
    """SHA-256 of the file content, truncated to 16 hex chars. Same file
    always gives the same ID."""
    sha = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


# LLM-based extraction (preferred)
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


def _llm_extract_metadata(doc: Any) -> dict[str, Any] | None:
    """Ask the configured metadata model to extract title / authors / abstract.
    Returns None on any failure so callers can fall back to heuristics."""
    header_text = _collect_header_block(doc)
    if not header_text:
        return None

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
        return None

    parsed: dict[str, Any] = json.loads(raw)

    title = (parsed.get("title") or "").strip()
    authors = parsed.get("authors") or []
    abstract = (parsed.get("abstract") or "").strip()

    if not isinstance(authors, list):
        authors = []
    authors = [str(a).strip() for a in authors if str(a).strip()]

    if not title and not authors:
        return None

    return {"title": title, "authors": authors, "abstract": abstract}


# Public API for extracting metadata from a Docling document
def extract_metadata(doc: Any, source_filename: str) -> dict[str, Any]:
    """Build a metadata dict from a Docling document. Tries the LLM first,
    falls back to heuristic parsing if that doesn't work out."""
    llm_meta: dict[str, Any] | None = None
    try:
        llm_meta = _llm_extract_metadata(doc)
    except Exception as exc:
        print(
            f"[metadata] LLM extraction failed, falling back to heuristics: {exc}",
            file=sys.stderr,
            flush=True,
        )

    if llm_meta:
        title = llm_meta["title"] or _title_from_filename(source_filename)
        authors = llm_meta["authors"]
        abstract = llm_meta["abstract"]
    else:
        title = _extract_title(doc) or _title_from_filename(source_filename)
        authors = _extract_authors(doc)
        abstract = _extract_abstract(doc)

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "source_filename": source_filename,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Heuristic: title
# ---------------------------------------------------------------------------

def _extract_title(doc: Any) -> str | None:
    if hasattr(doc, "title") and doc.title:
        title = doc.title if isinstance(doc.title, str) else str(doc.title)
        if title.strip():
            return title.strip()

    for item, _level in doc.iterate_items():
        label = _label_str(item)
        if label in ("title", "heading_0", "document_title"):
            text = _item_text(item)
            if text:
                return text

    # First section_header is usually the paper title (before Abstract)
    for item, _level in doc.iterate_items():
        label = _label_str(item)
        if label in ("section_header", "heading", "heading_1"):
            text = _item_text(item)
            if text and text.strip().lower() not in _SKIP_TITLE_TEXTS:
                return text

    return None


_SKIP_TITLE_TEXTS: set[str] = {"abstract", "summary", "introduction", "references", "bibliography"}


def _title_from_filename(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[_\-]+", " ", name).strip()


# ---------------------------------------------------------------------------
# Heuristic: authors
# ---------------------------------------------------------------------------

_AUTHOR_LABELS: set[str] = {"author", "authors", "creator", "creators"}


def _extract_authors(doc: Any) -> list[str]:
    if hasattr(doc, "metadata"):
        meta = doc.metadata
        for attr in ("authors", "creator", "author"):
            val = getattr(meta, attr, None) or (
                meta.get(attr) if isinstance(meta, dict) else None
            )
            if val:
                return _normalise_author_list(val)

    for item, _level in doc.iterate_items():
        label = _label_str(item)
        if label in _AUTHOR_LABELS:
            text = _item_text(item)
            if text:
                return _split_authors(text)

    # Last resort: grab text between the title heading and the next heading
    return _extract_authors_between_headers(doc)


def _extract_authors_between_headers(doc: Any) -> list[str]:
    """Grab text items sitting between the title header and the next heading.
    That region is where author names live in most paper layouts."""
    found_title_header = False
    candidates: list[str] = []

    for item, _level in doc.iterate_items():
        label = _label_str(item)

        if label in ("section_header", "heading", "heading_1"):
            if not found_title_header:
                found_title_header = True
                continue
            break

        if not found_title_header:
            continue

        if label in ("footnote", "list_item", "picture"):
            continue

        text = _item_text(item)
        if not text:
            continue

        if _is_non_author_block(text):
            break

        if _CORRESPONDING_RE.match(text.strip()):
            continue

        candidates.append(text)

    authors: list[str] = []
    for text in candidates:
        for name in _parse_author_names(text):
            if name and len(name) > 1 and not _is_duplicate_author(name, authors):
                authors.append(name)
    return authors


# Regexes for cleaning up author lines
_EMAIL_RE: re.Pattern[str] = re.compile(r"\S+@\S+\.\S+")
_DECORATION_RE: re.Pattern[str] = re.compile(r"[∗†‡§¶\*]+")
_PARENTHETICAL_RE: re.Pattern[str] = re.compile(r"\([^)]*\)")
_NUMBERED_SUPER_RE: re.Pattern[str] = re.compile(r"(?<=\w)\s+\d{1,2}\s*(?=[,\s]|$)")
_CORRESPONDING_RE: re.Pattern[str] = re.compile(r"corresponding\s+author", re.IGNORECASE)

_AFFILIATION_MARKERS: re.Pattern[str] = re.compile(
    r"\b(University|Institute|Google|Meta|Microsoft|DeepMind|OpenAI|Facebook|"
    r"Research|Brain|Lab|College|Department|Dept|School|Corp|Center|Centre|"
    r"Bowie|Stanford|MIT|ETH|INRIA|Academy)\b",
    re.IGNORECASE,
)

_NON_AUTHOR_PREFIXES: tuple[str, ...] = (
    "abstract", "index terms", "keywords", "key words",
    "summary", "this paper", "we propose", "in this",
    "received", "accepted", "published", "copyright",
    "digital object", "doi:",
)


def _is_non_author_block(text: str) -> bool:
    lower = text.strip().lower()
    if any(lower.startswith(p) for p in _NON_AUTHOR_PREFIXES):
        return True
    if len(text) > 500:
        return True
    return False


def _parse_author_names(text: str) -> list[str]:
    """Best-effort extraction of person names from a messy author line.
    Strips emails, affiliations, superscripts, etc. before splitting."""
    text = _EMAIL_RE.sub("", text)
    text = _DECORATION_RE.sub("", text)
    text = _PARENTHETICAL_RE.sub("", text)
    text = _NUMBERED_SUPER_RE.sub("", text)

    match = _AFFILIATION_MARKERS.search(text)
    if match:
        text = text[: match.start()]

    text = re.sub(r"\bAND\b", ",", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(",;")

    if not text:
        return []

    return _split_authors(text)


def _is_duplicate_author(name: str, existing: list[str]) -> bool:
    lower = name.lower()
    return any(lower == e.lower() for e in existing)


def _normalise_author_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(a).strip() for a in raw if str(a).strip()]
    if isinstance(raw, str):
        return _split_authors(raw)
    return []


def _split_authors(text: str) -> list[str]:
    for delimiter in [";", ",", " and ", "\n"]:
        parts = [p.strip() for p in text.split(delimiter) if p.strip()]
        if len(parts) > 1:
            return parts
    return [text.strip()] if text.strip() else []


# ---------------------------------------------------------------------------
# Heuristic: abstract
# ---------------------------------------------------------------------------

_ABSTRACT_HEADINGS: set[str] = {"abstract", "summary", "synopsis"}


def _extract_abstract(doc: Any) -> str:
    capturing = False
    parts: list[str] = []

    for item, _level in doc.iterate_items():
        label = _label_str(item)
        text = _item_text(item)

        if not capturing:
            if label in ("section_header", "heading", "heading_1") and text:
                if text.strip().lower() in _ABSTRACT_HEADINGS:
                    capturing = True
                    continue
            if label == "abstract":
                if text:
                    return text
                capturing = True
                continue
            # IEEE-style: text block that starts with "ABSTRACT"
            if text and _INLINE_ABSTRACT_RE.match(text.strip()):
                body = _INLINE_ABSTRACT_RE.sub("", text.strip(), count=1).strip()
                if body:
                    return body
                capturing = True
                continue
        else:
            if label in ("section_header", "heading", "heading_1", "title"):
                break
            if text:
                parts.append(text)

    return " ".join(parts).strip()


_INLINE_ABSTRACT_RE: re.Pattern[str] = re.compile(r"^ABSTRACT[\s:—–-]*", re.IGNORECASE)


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