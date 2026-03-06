from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger("sourcesleuth.pdf_processor")

# Configuration

DEFAULT_CHUNK_SIZE = 500       # tokens (≈ 375 words)
DEFAULT_CHUNK_OVERLAP = 50     # tokens overlap between consecutive chunks
APPROX_CHARS_PER_TOKEN = 4    # rough estimate for English text


@dataclass
class TextChunk:
    """A single chunk of text extracted from a PDF."""

    text: str
    filename: str
    page: int          # 1-indexed page number
    chunk_index: int   # position within the document
    start_char: int    # character offset in the full document text
    end_char: int      # character offset in the full document text

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            "text": self.text,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextChunk":
        """Reconstruct from a dictionary."""
        return cls(**data)


@dataclass
class PageSpan:
    """Tracks which page a character range belongs to."""

    page: int          # 1-indexed
    start_char: int
    end_char: int


@dataclass
class PDFDocument:
    """Represents a fully extracted PDF document."""

    filename: str
    full_text: str
    page_spans: list[PageSpan] = field(default_factory=list)
    chunks: list[TextChunk] = field(default_factory=list)


# Extraction

def extract_text_from_pdf(pdf_path: str | Path) -> PDFDocument:
    """
    Extract all text from a PDF file, tracking per-page boundaries.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A PDFDocument with full_text and page_spans populated.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{pdf_path.name}': {exc}") from exc

    full_text_parts: list[str] = []
    page_spans: list[PageSpan] = []
    current_offset = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")

        if not page_text.strip():
            continue

        start = current_offset
        full_text_parts.append(page_text)
        current_offset += len(page_text)

        page_spans.append(PageSpan(
            page=page_num + 1,  # 1-indexed
            start_char=start,
            end_char=current_offset,
        ))

    doc.close()

    full_text = "".join(full_text_parts)
    logger.info(
        "Extracted %d characters from %d pages of '%s'",
        len(full_text), len(page_spans), pdf_path.name,
    )

    return PDFDocument(
        filename=pdf_path.name,
        full_text=full_text,
        page_spans=page_spans,
    )


# Chunking

def _char_size(token_count: int) -> int:
    """Convert a token count to an approximate character count."""
    return token_count * APPROX_CHARS_PER_TOKEN


def chunk_text(
    document: PDFDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split the document text into overlapping chunks.

    Uses a sliding-window approach over character offsets, then maps each
    chunk back to its originating page.

    Args:
        document: A previously extracted PDFDocument.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap between consecutive chunks in tokens.

    Returns:
        A list of TextChunk objects.
    """
    text = document.full_text
    if not text.strip():
        logger.warning("Document '%s' has no extractable text.", document.filename)
        return []

    char_chunk = _char_size(chunk_size)
    char_overlap = _char_size(chunk_overlap)
    stride = max(char_chunk - char_overlap, 1)

    chunks: list[TextChunk] = []
    idx = 0
    start = 0

    while start < len(text):
        end = min(start + char_chunk, len(text))
        chunk_text_str = text[start:end].strip()

        if chunk_text_str:
            page = _resolve_page(document.page_spans, start)
            chunks.append(TextChunk(
                text=chunk_text_str,
                filename=document.filename,
                page=page,
                chunk_index=idx,
                start_char=start,
                end_char=end,
            ))
            idx += 1

        start += stride

    document.chunks = chunks
    logger.info(
        "Chunked '%s' into %d chunks (size=%d, overlap=%d tokens).",
        document.filename, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def _resolve_page(page_spans: list[PageSpan], char_offset: int) -> int:
    """Map a character offset back to its 1-indexed page number."""
    for span in page_spans:
        if span.start_char <= char_offset < span.end_char:
            return span.page
    # Fallback: return last page if offset is at the very end
    return page_spans[-1].page if page_spans else 1


# Batch Processing

def process_pdf_directory(
    directory: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Process all PDFs in a directory and return a flat list of chunks.

    Args:
        directory: Path to the directory containing PDF files.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Overlap in tokens.

    Returns:
        Flat list of TextChunk objects from all PDFs in the directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", directory)
        return []

    all_chunks: list[TextChunk] = []

    for pdf_path in pdf_files:
        try:
            document = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(document, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            logger.info(
                "Processed '%s' -> %d chunks", pdf_path.name, len(chunks),
            )
        except Exception as exc:
            logger.error("Failed to process '%s': %s", pdf_path.name, exc)

    logger.info(
        "Total: processed %d PDFs -> %d chunks from '%s'.",
        len(pdf_files), len(all_chunks), directory,
    )
    return all_chunks
