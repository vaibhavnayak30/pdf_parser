"""
MCP Server: PDF Parser using IBM Docling
-----------------------------------------
- DocumentConverter is initialised ONCE in the async lifespan (server startup).
- Stored in a module-level variable — safe for a single-process stdio server.
- Blocking Docling calls run in asyncio.to_thread so concurrent requests
  never block each other.
- No images, no file I/O. Tables returned as Markdown.
"""

from __future__ import annotations
import sys
import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from fastmcp import FastMCP, Context

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Module-level converter — populated at startup, shared across all requests
# ---------------------------------------------------------------------------
_converter: Any = None


# ---------------------------------------------------------------------------
# Lifespan: load Docling once before any tool is called
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncGenerator[None, None]:
    global _converter

    print("[mcp-docling] Starting up — initialising Docling …", flush=True)

    # Keep heavy imports here so any ImportError surfaces at startup.
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = False
    pipeline_options.generate_table_images = False

    _converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    print("[mcp-docling] Docling ready.", file=sys.stderr, flush=True)
    yield                                     # server runs here
    print("[mcp-docling] Shutting down.", file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Initialise FastMCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="pdf-parser",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper: runs in a worker thread (blocking I/O + CPU)
# ---------------------------------------------------------------------------
def _parse_pdf_sync(pdf_path: str) -> str:
    """Blocking Docling work — called via asyncio.to_thread."""
    from docling.datamodel.document import TableItem

    result = _converter.convert(pdf_path)
    doc = result.document

    # Full document in Markdown (tables are already embedded inline)
    full_markdown: str = doc.export_to_markdown()

    # Also build an explicit table section for clarity
    table_sections: list[str] = []
    table_count = 0

    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            table_count += 1

            # Caption (optional)
            caption_text = ""
            if hasattr(item, "captions") and item.captions:
                parts: list[str] = []
                for cap in item.captions:
                    try:
                        parts.append(cap.resolve(doc).text)
                    except Exception:
                        parts.append(getattr(cap, "text", str(cap)))
                caption_text = " ".join(parts)

            # Prefer pandas → markdown; fall back to docling's own exporter
            try:
                df = item.export_to_dataframe(doc)
                md_table = df.to_markdown(index=False)   # needs tabulate
            except Exception:
                try:
                    md_table = item.export_to_markdown(doc)
                except Exception as exc2:
                    md_table = f"*(Table {table_count} could not be rendered: {exc2})*"

            header = f"### Table {table_count}"
            if caption_text:
                header += f": {caption_text}"
            table_sections.append(f"{header}\n\n{md_table}")

    output_parts: list[str] = ["# Parsed PDF Content\n", full_markdown]
    if table_sections:
        output_parts.append("\n\n---\n## Extracted Tables\n")
        output_parts.append("\n\n".join(table_sections))

    return "\n".join(output_parts)


# ---------------------------------------------------------------------------
# Tool: parse_pdf
# ---------------------------------------------------------------------------
@mcp.tool()
async def parse_pdf(pdf_path: str, ctx: Context) -> str:
    """
    Parses a PDF document from a local file path and extracts its text.
    Use this tool when the user uploads a PDF and asks for a summary or extraction.will 

    Args:
        pdf_path: Absolute path to the PDF file on disk.

    Returns:
        Markdown string containing the full document text and all tables.
    """
    if not pdf_path or not pdf_path.strip():
        return "Error: pdf_path must not be empty."

    pdf_path = pdf_path.strip()

    if not os.path.isfile(pdf_path):
        return (
            f"Error: File not found: '{pdf_path}'. "
            "Provide the full absolute path to the PDF."
        )

    if not pdf_path.lower().endswith(".pdf"):
        return f"Error: '{pdf_path}' does not look like a PDF file."

    if _converter is None:
        return "Error: Docling converter is not initialised. Server may not have started correctly."

    await ctx.info(f"Parsing PDF: {pdf_path}")

    try:
        output = await asyncio.to_thread(_parse_pdf_sync, pdf_path)
    except Exception as exc:
        return f"Error while parsing PDF: {exc}"

    await ctx.info("Parsing complete.")
    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")