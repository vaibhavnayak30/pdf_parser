"""
Splits a Docling document into chunks using the library's own HybridChunker.
"""
from __future__ import annotations

from typing import Any

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker import DocChunk
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from config import CHUNK_MAX_TOKENS, CHUNK_TOKENIZER_MODEL, SKIP_REFERENCE_SECTIONS

_REFERENCE_HEADINGS = frozenset(
    {"references", "bibliography", "works cited", "literature cited"}
)

_ABSTRACT_HEADINGS = frozenset({"abstract"})

_tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model(CHUNK_TOKENIZER_MODEL),
    max_tokens=CHUNK_MAX_TOKENS,
)

_chunker = HybridChunker(
    tokenizer=_tokenizer,
    merge_peers=True,
    repeat_table_header=True,
)


def chunk_document(
    doc: Any,
    title: str = "",
    publication_year: str = "",
) -> list[dict[str, Any]]:
    """Turn a DoclingDocument into a flat list of chunk dicts ready for
    embedding. Each dict has: text, enriched_text, section_name,
    chunk_index, page_numbers, is_abstract.

    ``title`` and ``publication_year`` are prepended to ``enriched_text``
    so the embedding captures document-level context, but they are NOT
    included in ``text`` (which is what the LLM sees as retrieved context).
    """

    doc_prefix = _build_doc_prefix(title, publication_year)

    raw_chunks: list[DocChunk] = list(_chunker.chunk(dl_doc=doc))
    results: list[dict[str, Any]] = []

    for chunk in raw_chunks:
        headings = chunk.meta.headings or []
        section_name = headings[-1] if headings else "Untitled"

        if SKIP_REFERENCE_SECTIONS and _is_reference_section(headings):
            continue

        is_abstract = _is_abstract_section(headings)
        page_numbers = _extract_page_numbers(chunk)

        base_enriched = _chunker.contextualize(chunk=chunk)
        enriched_text = f"{doc_prefix}{base_enriched}" if doc_prefix else base_enriched

        results.append(
            {
                "text": chunk.text,
                "enriched_text": enriched_text,
                "section_name": section_name,
                "chunk_index": len(results),
                "page_numbers": page_numbers,
                "is_abstract": is_abstract,
            }
        )

    return results


def _build_doc_prefix(title: str, publication_year: str) -> str:
    """Build the document-level prefix prepended to each chunk's enriched
    text for embedding.  Returns empty string if no metadata is available."""
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if publication_year:
        parts.append(f"Year: {publication_year}")
    if not parts:
        return ""
    return "\n".join(parts) + "\n---\n"


def _is_reference_section(headings: list[str]) -> bool:
    return any(h.strip().lower() in _REFERENCE_HEADINGS for h in headings)


def _is_abstract_section(headings: list[str]) -> bool:
    return any(h.strip().lower() in _ABSTRACT_HEADINGS for h in headings)


def _extract_page_numbers(chunk: DocChunk) -> list[int]:
    """Pull unique page numbers from the provenance of every doc_item."""
    pages: set[int] = set()
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no") and prov.page_no is not None:
                        pages.add(prov.page_no)
    return sorted(pages)