"""
SourceSleuth MCP Server — Main Entry Point.

This module defines the Model Context Protocol server that exposes:
    - **Tools**: ``find_orphaned_quote``, ``ingest_pdfs``, ``get_store_stats``
    - **Resources**: ``sourcesleuth://pdfs/{filename}`` for reading raw PDF text
    - **Prompts**: ``cite_recovered_source`` for formatting recovered citations

Transport: stdio (standard input/output), the default for local MCP Hosts
such as Claude Desktop, Cursor, and Windsurf.

Usage:
    # Run directly
    python -m src.mcp_server

    # Or via the installed entry-point
    sourcesleuth
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.pdf_processor import (
    TextChunk,
    extract_text_from_pdf,
    process_pdf_directory,
)
from src.vector_store import VectorStore
from src.dataset_preprocessor import (
    preprocess_dataset,
    stream_arxiv_records,
)

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("sourcesleuth.server")

# Configuration
# Resolve paths relative to the project root (parent of `src/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(os.environ.get("SOURCESLEUTH_PDF_DIR", str(PROJECT_ROOT / "student_pdfs")))
DATA_DIR = Path(os.environ.get("SOURCESLEUTH_DATA_DIR", str(PROJECT_ROOT / "data")))

# Initialize MCP Server & Vector Store

mcp = FastMCP(
    "SourceSleuth",
    instructions=(
        "A local MCP server that helps students recover citations "
        "for orphaned quotes by semantically searching their academic PDFs."
    ),
)

store = VectorStore(data_dir=DATA_DIR)

# Attempt to load a previously persisted vector store on startup
_loaded = store.load()
if _loaded:
    logger.info("Restored vector store with %d chunks.", store.total_chunks)
else:
    logger.info("Starting with an empty vector store.")



# MCP TOOLS
@mcp.tool()
def find_orphaned_quote(quote: str, top_k: int = 5) -> str:
    """
    Find the original academic source for an orphaned quote or paraphrase.

    Embeds the student's text and performs a cosine-similarity search
    against all ingested PDF chunks to locate the most likely source
    documents, pages, and surrounding context.

    Args:
        quote: The text or paraphrase the student wants to find a source for.
        top_k: Number of top matching results to return (default 5).

    Returns:
        A formatted string listing the most similar PDF chunks, including
        the source filename, page number, confidence score, and context.
    """
    if store.total_chunks == 0:
        return (
            "No PDFs have been ingested yet.\n\n"
            "Please run the `ingest_pdfs` tool first to index your "
            "academic papers, then try again."
        )

    results = store.search(query=quote, top_k=top_k)

    if not results:
        return "No matching sources found for the given text."

    response_parts = [
        f"**Found {len(results)} potential source(s)** for your quote:\n"
    ]

    for i, result in enumerate(results, start=1):
        score = result["score"]
        # Determine confidence tier
        if score >= 0.75:
            badge = "High"
        elif score >= 0.50:
            badge = "Medium"
        else:
            badge = "Low"

        context_preview = result["text"][:300].replace("\n", " ")
        if len(result["text"]) > 300:
            context_preview += " …"

        response_parts.append(
            f"### Match {i}\n"
            f"- **Document**: `{result['filename']}`\n"
            f"- **Page**: {result['page']}\n"
            f"- **Confidence**: {badge} ({score})\n"
            f"- **Context**:\n"
            f"  > {context_preview}\n"
        )

    return "\n".join(response_parts)


@mcp.tool()
def ingest_pdfs(directory: str = "") -> str:
    """
    Ingest all PDF files from a directory into the local vector store.

    Extracts text from every PDF, splits it into 500-token chunks with
    50-token overlap, embeds each chunk using the all-MiniLM-L6-v2 model,
    and stores them in a FAISS index for fast similarity search.

    Args:
        directory: Path to the folder containing PDFs. If empty, defaults
                   to the `student_pdfs/` directory in the project root.
                   Supports both absolute and relative paths.

    Returns:
        A summary of how many PDFs and chunks were processed.
    """
    target_dir = Path(directory) if directory else PDF_DIR

    if not target_dir.is_dir():
        return f"Directory not found: `{target_dir}`"

    pdf_files = list(target_dir.glob("*.pdf"))
    if not pdf_files:
        return f"No PDF files found in `{target_dir}`."

    # Process all PDFs
    chunks = process_pdf_directory(target_dir)
    if not chunks:
        return "PDFs were found but no text could be extracted."

    # Add to vector store and persist
    added = store.add_chunks(chunks)
    store.save()

    files_set = {c.filename for c in chunks}

    return (
        f"**Ingestion complete!**\n\n"
        f"- **PDFs processed**: {len(files_set)}\n"
        f"- **Chunks created**: {added}\n"
        f"- **Total chunks in store**: {store.total_chunks}\n"
        f"- **Files**: {', '.join(sorted(files_set))}\n\n"
        f"You can now use `find_orphaned_quote` to search these documents."
    )


@mcp.tool()
def get_store_stats() -> str:
    """
    Get statistics about the current vector store.

    Returns information about how many documents and chunks are indexed,
    which files have been ingested, and which embedding model is in use.
    """
    stats = store.get_stats()

    if stats["total_chunks"] == 0:
        return (
            " **Vector Store Status**: Empty\n\n"
            "No PDFs have been ingested yet. Use `ingest_pdfs` to get started."
        )

    files_list = "\n".join(f"  - `{f}`" for f in stats["ingested_files"])

    return (
        f"**Vector Store Statistics**\n\n"
        f"- **Total chunks**: {stats['total_chunks']}\n"
        f"- **Number of files**: {stats['num_files']}\n"
        f"- **Embedding model**: `{stats['model_name']}`\n"
        f"- **Embedding dimensions**: {stats['embedding_dim']}\n"
        f"- **Index type**: {stats['index_type']}\n\n"
        f"**Ingested files**:\n{files_list}"
    )


@mcp.tool()
def ingest_arxiv(
    category_prefix: str = "cs.",
    max_records: int = 5000,
) -> str:
    """
    Preprocess and ingest arXiv paper abstracts into the vector store.

    Reads the arXiv metadata snapshot, filters by category, cleans the
    text, and embeds the title+abstract of each paper for semantic search.
    This lets students search across millions of academic paper abstracts
    to find the source of an orphaned quote.

    Args:
        category_prefix: arXiv category prefix to filter by (e.g. "cs."
                         for all Computer Science papers, "physics." for
                         Physics). Default is "cs." for CS papers.
        max_records: Maximum number of papers to ingest. Start small
                     (1000-5000) for quick tests; increase for thorough
                     searches. Default is 5000.

    Returns:
        A summary of how many arXiv records were preprocessed and ingested.
    """
    raw_path = DATA_DIR / "arxiv-metadata-oai-snapshot.json"
    if not raw_path.exists():
        return (
            "arXiv dataset not found.\n\n"
            f"Expected file at: `{raw_path}`\n"
            "Download it from: https://www.kaggle.com/Cornell-University/arxiv"
        )

    # Step 1: Preprocess
    preprocessed_path = DATA_DIR / "arxiv_preprocessed.jsonl"
    logger.info(
        "Preprocessing arXiv dataset (prefix=%s, max=%d) …",
        category_prefix, max_records,
    )

    prefixes = {p.strip() for p in category_prefix.split(",") if p.strip()}
    stats = preprocess_dataset(
        input_path=raw_path,
        output_path=preprocessed_path,
        category_prefix_filter=prefixes,
        max_records=max_records,
    )

    # Step 2: Convert to TextChunks and ingest
    chunks = []
    for record in stream_arxiv_records(
        preprocessed_path,
        max_records=max_records,
    ):
        # Use title + abstract as the searchable text
        text = f"{record.title}. {record.abstract}"
        chunk = TextChunk(
            text=text,
            filename=f"arxiv:{record.arxiv_id}",
            page=0,  # arXiv papers don't have page numbers here
            chunk_index=0,
            start_char=0,
            end_char=len(text),
        )
        chunks.append(chunk)

    if not chunks:
        return "No arXiv records matched the filter criteria."

    added = store.add_chunks(chunks)
    store.save()

    # Top categories from stats
    top_cats = sorted(
        stats.categories_seen.items(), key=lambda x: -x[1]
    )[:10]
    cats_str = ", ".join(f"{cat} ({n})" for cat, n in top_cats)

    return (
        f"**arXiv Ingestion Complete!**\n\n"
        f"- **Records preprocessed**: {stats.records_output:,}\n"
        f"- **Chunks added to store**: {added:,}\n"
        f"- **Total chunks in store**: {store.total_chunks:,}\n"
        f"- **Category filter**: `{category_prefix}`\n"
        f"- **Top categories**: {cats_str}\n"
        f"- **Preprocessing time**: {stats.elapsed_seconds:.1f}s\n\n"
        f"You can now use `find_orphaned_quote` to search across "
        f"both your PDFs and arXiv papers."
    )


# MCP RESOURCES

@mcp.resource("sourcesleuth://pdfs/{filename}")
def get_pdf_text(filename: str) -> str:
    """
    Read the full extracted text of a specific PDF.

    This resource allows the AI model to access the raw text content of
    any ingested PDF when it needs deeper context beyond the chunk-level
    results returned by find_orphaned_quote.

    Args:
        filename: Name of the PDF file (e.g., "research_paper.pdf").

    Returns:
        The full extracted text of the PDF.
    """
    pdf_path = PDF_DIR / filename
    if not pdf_path.exists():
        return f"Error: PDF '{filename}' not found in {PDF_DIR}"

    if not pdf_path.suffix.lower() == ".pdf":
        return f"Error: '{filename}' is not a PDF file."

    try:
        document = extract_text_from_pdf(pdf_path)
        return document.full_text
    except Exception as exc:
        return f"Error reading '{filename}': {exc}"


# MCP PROMPTS
@mcp.prompt()
def cite_recovered_source(
    quote: str,
    source_filename: str,
    page_number: int,
    citation_style: str = "APA",
) -> str:
    """
    Format a recovered source into a proper academic citation.

    This prompt instructs the LLM to take the raw recovery result and
    produce a clean, correctly formatted citation in the requested style.

    Args:
        quote: The original orphaned quote from the student's paper.
        source_filename: The PDF filename where the source was found.
        page_number: The page number in the source document.
        citation_style: Citation format — "APA", "MLA", or "Chicago".
    """
    return (
        f"You are an expert academic citation assistant.\n\n"
        f"A student had the following orphaned quote in their paper:\n"
        f"  \"{quote}\"\n\n"
        f"Our citation recovery tool found this quote in the document "
        f"`{source_filename}` on page {page_number}.\n\n"
        f"Please do the following:\n"
        f"1. Extract the likely author(s), title, publication year, and "
        f"   publisher from the document filename and any context available.\n"
        f"2. Format a complete **{citation_style}** citation.\n"
        f"3. Also provide the correct in-text citation the student should "
        f"   use in their paper.\n"
        f"4. If you cannot determine all fields from the filename alone, "
        f"   clearly indicate which fields need to be filled in manually "
        f"   with placeholders like [Author Last Name].\n\n"
        f"Respond with:\n"
        f"- **Full Citation** (for the bibliography/works cited page)\n"
        f"- **In-Text Citation** (for use within the paper)\n"
        f"- **Notes** (any caveats or fields that need manual verification)"
    )


# Entry Point

def main():
    """Run the SourceSleuth MCP server over stdio."""
    logger.info("Starting SourceSleuth MCP Server v1.0.0 …")
    logger.info("PDF directory : %s", PDF_DIR)
    logger.info("Data directory: %s", DATA_DIR)
    mcp.run()


if __name__ == "__main__":
    main()
