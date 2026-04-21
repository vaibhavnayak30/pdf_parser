"""
Research-paper RAG server built on FastMCP.

Exposes five tools over MCP:
  index_paper      – ingest a PDF into the vector store
  query_papers     – semantic search across papers (auto-indexes new PDFs)
  summarize_paper  – full-paper LLM summary (auto-indexes new PDFs)
  list_papers      – list everything in the index
  get_paper_info   – metadata + download links for a single paper

Backed by Qdrant (vectors), MinIO (raw files + markdown), Docling (PDF parsing),
and LiteLLM (embeddings + summarization).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import urllib.parse
import urllib.request
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from fastmcp import FastMCP, Context
import litellm

from chunker import chunk_document
from config import (
    EXTRACT_IMAGES,
    EXTRACT_TABLE_IMAGES,
    MAX_PDF_DOWNLOAD_BYTES,
    SUMMARIZATION_MODEL,
)
from embeddings import embed_query, embed_texts
from metadata import extract_metadata, generate_paper_id
from object_store import ObjectStore
from store import VectorStore

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Populated once during server startup
_converter: Any | None = None
_vector_store: VectorStore | None = None
_object_store: ObjectStore | None = None


# Startup / shutdown context manager
@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncGenerator[None, None]:
    global _converter, _vector_store, _object_store

    print("[startup] initialising Docling …", file=sys.stderr, flush=True)

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.generate_picture_images = EXTRACT_IMAGES
    pipeline_opts.generate_table_images = EXTRACT_TABLE_IMAGES

    _converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        },
    )

    try:
        _converter._get_pipeline(InputFormat.PDF)
        print("[startup] Docling models pre-loaded.", file=sys.stderr, flush=True)
    except Exception:
        print(
            "[startup] Docling ready (models will load on first PDF).",
            file=sys.stderr,
            flush=True,
        )

    _vector_store = VectorStore()
    print("[startup] Qdrant connected.", file=sys.stderr, flush=True)

    _object_store = ObjectStore()
    print("[startup] MinIO connected.", file=sys.stderr, flush=True)

    yield

    print("[shutdown] bye.", file=sys.stderr, flush=True)


mcp = FastMCP(name="pdf-parser", lifespan=lifespan)


# Helpers
def _validate_pdf_path(pdf_path: str) -> str | None:
    """Quick sanity check on a user-supplied PDF path."""
    if not pdf_path or not pdf_path.strip():
        return "Error: pdf_path must not be empty."
    pdf_path = pdf_path.strip()
    if not os.path.isfile(pdf_path):
        return f"Error: File not found: '{pdf_path}'."
    if not pdf_path.lower().endswith(".pdf"):
        return f"Error: '{pdf_path}' does not look like a PDF file."
    return None


def _is_public_pdf_url(value: str) -> bool:
    """Return True if *value* looks like an HTTP(S) URL."""
    parsed = urllib.parse.urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_public_pdf(url: str) -> str:
    """Download a public PDF URL to a temporary local file and return its path."""
    with urllib.request.urlopen(url, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            raise ValueError(
                "URL does not appear to be a PDF (expected .pdf or Content-Type: application/pdf)."
            )

        content_length = response.headers.get("Content-Length")
        if content_length:
            expected_bytes = int(content_length)
            if expected_bytes > MAX_PDF_DOWNLOAD_BYTES:
                raise ValueError(
                    f"PDF is too large ({expected_bytes} bytes). "
                    f"Maximum allowed is {MAX_PDF_DOWNLOAD_BYTES} bytes."
                )

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pdf", prefix="pdf_parser_", delete=False
            ) as tmp:
                temp_path = tmp.name
                total_downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total_downloaded += len(chunk)
                    if total_downloaded > MAX_PDF_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"PDF is too large (> {MAX_PDF_DOWNLOAD_BYTES} bytes)."
                        )
                    tmp.write(chunk)
                return tmp.name
        except Exception:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise


def _services_ready() -> str | None:
    """Returns an error string if any backend hasn't started yet."""
    if _converter is None:
        return "Error: Docling converter not initialised."
    if _vector_store is None:
        return "Error: Qdrant vector store not initialised."
    if _object_store is None:
        return "Error: MinIO object store not initialised."
    return None


def _resolve_paper_id(identifier: str) -> str | None:
    """Try to match a paper by its ID, title, or filename (substring ok)."""
    assert _object_store is not None
    
    identifier = identifier.strip()
    if not identifier:
        return None

    if _object_store.paper_exists(identifier):
        return identifier

    papers = _object_store.list_papers()
    id_lower = identifier.lower()
    for p in papers:
        if id_lower in p.get("title", "").lower():
            return p["paper_id"]
        if id_lower in p.get("source_filename", "").lower():
            return p["paper_id"]
    return None


def _ensure_indexed(source: str) -> str:
    """Resolve *source* to a paper_id. If it points to a PDF that hasn't
    been indexed yet, index it first. Raises ValueError on failure."""
    source = source.strip()

    paper_id = _resolve_paper_id(source)
    if paper_id:
        return paper_id

    if os.path.isfile(source) and source.lower().endswith(".pdf"):
        summary = _index_paper_sync(source)
        return summary["paper_id"]

    if _is_public_pdf_url(source):
        temp_path = _download_public_pdf(source)
        try:
            summary = _index_paper_sync(temp_path)
            return summary["paper_id"]
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    raise ValueError(
        f"No indexed paper matches '{source}' and it is not a valid PDF path/URL."
    )

def _index_paper_sync(pdf_path: str) -> dict[str, Any]:
    """Full pipeline: parse → metadata → chunk → embed → store."""
    assert _object_store is not None
    assert _converter is not None
    assert _vector_store is not None

    paper_id = generate_paper_id(pdf_path)
    source_filename = os.path.basename(pdf_path)

    if _object_store.paper_exists(paper_id):
        existing = _object_store.get_metadata(paper_id)
        return {
            "status": "already_indexed",
            "paper_id": paper_id,
            "title": existing.get("title", source_filename) if existing else source_filename,
        }

    result = _converter.convert(pdf_path)
    doc = result.document

    meta = extract_metadata(doc, source_filename)
    meta["paper_id"] = paper_id

    full_markdown = doc.export_to_markdown()

    _object_store.upload_pdf(paper_id, pdf_path)
    _object_store.upload_markdown(paper_id, full_markdown)

    chunks = chunk_document(doc)
    if not chunks:
        _object_store.upload_metadata(paper_id, {**meta, "total_chunks": 0})
        return {
            "status": "indexed",
            "paper_id": paper_id,
            "title": meta["title"],
            "authors": meta["authors"],
            "total_chunks": 0,
            "warning": "No text chunks produced — the PDF may be image-only.",
        }

    embed_inputs = [c["enriched_text"] for c in chunks]
    embeddings = embed_texts(embed_inputs)

    num_stored = _vector_store.index_chunks(paper_id, chunks, embeddings, meta)

    meta["total_chunks"] = num_stored
    _object_store.upload_metadata(paper_id, meta)

    return {
        "status": "indexed",
        "paper_id": paper_id,
        "title": meta["title"],
        "authors": meta["authors"],
        "total_chunks": num_stored,
    }


def _query_sync(
    question: str,
    top_k: int,
    paper_id: str | None,
    author: str | None,
    section: str | None,
) -> list[dict[str, Any]]:
    assert _vector_store is not None
    query_vec = embed_query(question)
    return _vector_store.search(
        query_vector=query_vec,
        limit=top_k,
        paper_id=paper_id or None,
        author=author or None,
        section=section or None,
    )


# ---------------------------------------------------------------------------
# Tool: index_paper
# ---------------------------------------------------------------------------

@mcp.tool()
async def index_paper(pdf_path: str, ctx: Context) -> str:
    """
    Ingest a PDF into the search index.

    Parses the file, pulls out metadata, splits into chunks, embeds them,
    and stores everything in Qdrant + MinIO. Duplicates are caught
    automatically.

    Args:
        pdf_path: Absolute local PDF path, or a public HTTP(S) URL to a PDF.
    """
    err = _services_ready()
    if err:
        return err

    source = pdf_path.strip()
    local_pdf = source
    cleanup_path: str | None = None

    if _is_public_pdf_url(source):
        await ctx.info(f"Downloading PDF from URL: {source}")
        try:
            local_pdf = await asyncio.to_thread(_download_public_pdf, source)
            cleanup_path = local_pdf
        except Exception as exc:
            return f"Error downloading PDF URL: {exc}"
    else:
        err = _validate_pdf_path(source)
        if err:
            return err

    await ctx.info(f"Indexing paper: {source}")

    try:
        summary = await asyncio.to_thread(_index_paper_sync, local_pdf)
    except Exception as exc:
        return f"Error during indexing: {exc}"
    finally:
        if cleanup_path:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass

    if summary["status"] == "already_indexed":
        return (
            f"Paper already indexed.\n"
            f"  ID:    {summary['paper_id']}\n"
            f"  Title: {summary['title']}"
        )

    lines = [
        "Paper indexed successfully.",
        f"  ID:     {summary['paper_id']}",
        f"  Title:  {summary['title']}",
        f"  Authors: {', '.join(summary.get('authors', [])) or 'unknown'}",
        f"  Chunks: {summary['total_chunks']}",
    ]
    if summary.get("warning"):
        lines.append(f"  Warning: {summary['warning']}")

    await ctx.info("Indexing complete.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: query_papers
# ---------------------------------------------------------------------------

@mcp.tool()
async def query_papers(
    question: str,
    ctx: Context,
    source: str = "",
    author: str = "",
    section: str = "",
    top_k: int = 5,
) -> str:
    """
    Search across indexed papers and return the most relevant chunks.

    When *source* is given the search is scoped to that paper. If it points
    to a PDF that hasn't been indexed yet, it gets indexed on the fly before
    the query runs.

    Args:
        question: Natural-language query.
        source: Paper title / filename / ID, or a PDF path (optional).
                Unindexed PDFs are auto-indexed.
        author: Filter by author name (optional).
        section: Filter by section heading (optional).
        top_k: How many results to return (default 5).
    """
    err = _services_ready()
    if err:
        return err

    if not question or not question.strip():
        return "Error: question must not be empty."

    resolved_paper_id: str | None = None
    if source and source.strip():
        try:
            resolved_paper_id = await asyncio.to_thread(
                _ensure_indexed, source.strip()
            )
            await ctx.info(f"Scoping search to paper {resolved_paper_id}")
        except ValueError as exc:
            return str(exc)

    await ctx.info(f"Searching: {question}")

    try:
        results = await asyncio.to_thread(
            _query_sync,
            question.strip(),
            top_k,
            resolved_paper_id or "",
            author,
            section,
        )
    except Exception as exc:
        return f"Error during search: {exc}"

    if not results:
        return "No results found."

    parts: list[str] = [f"Found {len(results)} result(s):\n"]
    for i, r in enumerate(results, 1):
        authors_str = ", ".join(r.get("authors", [])) or "unknown"
        pages = r.get("page_numbers", [])
        pages_str = ", ".join(str(p) for p in pages) if pages else "n/a"
        parts.append(
            f"--- Result {i} (score: {r['score']:.3f}) ---\n"
            f"Paper:   {r['title']}\n"
            f"Authors: {authors_str}\n"
            f"Section: {r['section']}\n"
            f"Page(s): {pages_str}\n"
            f"Paper ID: {r['paper_id']}\n\n"
            f"{r['text']}\n"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool: list_papers
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_papers(ctx: Context) -> str:
    """Show every paper currently in the index."""
    err = _services_ready()
    if err:
        return err

    assert _object_store is not None

    try:
        papers = await asyncio.to_thread(_object_store.list_papers)
    except Exception as exc:
        return f"Error listing papers: {exc}"

    if not papers:
        return "No papers indexed yet."

    lines: list[str] = [f"Indexed papers ({len(papers)}):\n"]
    for p in papers:
        authors = ", ".join(p.get("authors", [])) or "unknown"
        lines.append(
            f"  [{p.get('paper_id', '?')}] {p.get('title', 'Untitled')}\n"
            f"    Authors: {authors}\n"
            f"    File:    {p.get('source_filename', '?')}\n"
            f"    Indexed: {p.get('indexed_at', '?')}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: get_paper_info
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_paper_info(identifier: str, ctx: Context) -> str:
    """
    Detailed metadata and download links for one paper.

    Args:
        identifier: Paper ID, title, or filename (substring match works).
    """
    err = _services_ready()
    if err:
        return err

    if not identifier or not identifier.strip():
        return "Error: identifier must not be empty."

    assert _object_store is not None

    paper_id = await asyncio.to_thread(_resolve_paper_id, identifier)
    if not paper_id:
        return f"No paper found matching '{identifier.strip()}'."

    try:
        meta = await asyncio.to_thread(_object_store.get_metadata, paper_id)
    except Exception as exc:
        return f"Error fetching metadata: {exc}"

    if not meta:
        return f"Paper not found: '{paper_id}'."

    pdf_url = await asyncio.to_thread(
        _object_store.get_download_url, paper_id, "pdf"
    )
    md_url = await asyncio.to_thread(
        _object_store.get_download_url, paper_id, "md"
    )

    authors = ", ".join(meta.get("authors", [])) or "unknown"
    abstract = meta.get("abstract", "")
    abstract_display = (abstract[:500] + " …") if len(abstract) > 500 else abstract

    return (
        f"Paper: {meta.get('title', 'Untitled')}\n"
        f"ID:      {paper_id}\n"
        f"Authors: {authors}\n"
        f"File:    {meta.get('source_filename', '?')}\n"
        f"Chunks:  {meta.get('total_chunks', '?')}\n"
        f"Indexed: {meta.get('indexed_at', '?')}\n\n"
        f"Abstract:\n{abstract_display}\n\n"
        f"Download PDF:      {pdf_url}\n"
        f"Download Markdown: {md_url}"
    )


# ---------------------------------------------------------------------------
# Tool: summarize_paper
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM_PROMPT = """\
You are a research paper summarization assistant.  Given the full Markdown
text of an academic paper, produce a clear, structured summary with these
sections:

1. **Title & Authors**
2. **Problem / Motivation** — what gap or question the paper addresses
3. **Approach / Methodology** — how the authors tackle the problem
4. **Key Results** — the main findings, metrics, or contributions
5. **Limitations & Future Work** — acknowledged gaps or next steps
6. **One-paragraph TL;DR**

Be concise but thorough.  Use bullet points where appropriate."""


def _summarize_sync(markdown: str, title: str) -> str:
    """Feed the full paper text to the summarization model."""
    max_chars = 120_000
    if len(markdown) > max_chars:
        markdown = markdown[:max_chars] + "\n\n[... truncated for length ...]"

    response: Any = litellm.completion(
        model=SUMMARIZATION_MODEL,
        messages=[
            {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Paper: {title}\n\n{markdown}"},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return response.choices[0].message.content


@mcp.tool()
async def summarize_paper(source: str, ctx: Context) -> str:
    """
    Produce a structured LLM summary of a research paper.

    Pass a title, filename, or paper ID to summarize something already in
    the index, or pass a PDF path to auto-index it first and then summarize.

    Args:
        source: Paper title / filename / ID, or absolute path to a PDF.
    """
    err = _services_ready()
    if err:
        return err

    if not source or not source.strip():
        return "Error: source must not be empty."

    assert _object_store is not None

    try:
        paper_id = await asyncio.to_thread(_ensure_indexed, source.strip())
    except ValueError as exc:
        return str(exc)

    await ctx.info(f"Fetching paper {paper_id} for summarization …")

    md = await asyncio.to_thread(_object_store.get_markdown, paper_id)
    if not md:
        return f"Markdown not found in storage for paper '{paper_id}'."

    meta = await asyncio.to_thread(_object_store.get_metadata, paper_id)
    title = meta.get("title", source) if meta else source

    await ctx.info(f"Summarizing '{title}' with {SUMMARIZATION_MODEL} …")

    try:
        summary = await asyncio.to_thread(_summarize_sync, md, title)
    except Exception as exc:
        return f"Error during summarization: {exc}"

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)