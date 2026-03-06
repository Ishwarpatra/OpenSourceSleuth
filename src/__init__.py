"""
SourceSleuth — Context-Aware Source Recovery.

A local MCP server and library for recovering citations for orphaned
quotes using semantic search across academic PDFs.
"""

__version__ = "1.0.0"

from src.source_sleuth import SourceRetriever
from src.pdf_processor import TextChunk, PDFDocument
from src.vector_store import VectorStore

__all__ = [
    "SourceRetriever",
    "TextChunk",
    "PDFDocument",
    "VectorStore",
    "__version__",
]
