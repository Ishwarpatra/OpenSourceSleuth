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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("sourcesleuth.server")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(os.environ.get("SOURCESLEUTH_PDF_DIR", str(PROJECT_ROOT / "student_pdfs")))
DATA_DIR = Path(os.environ.get("SOURCESLEUTH_DATA_DIR", str(PROJECT_ROOT / "data")))


class CitationFormatter:
    def __init__(self, confidence_thresholds: dict = None):
        self.confidence_thresholds = confidence_thresholds or {
            'high': 0.75,
            'medium': 0.50,
        }

    def get_confidence_badge(self, score: float) -> str:
        if score >= self.confidence_thresholds['high']:
            return "High"
        elif score >= self.confidence_thresholds['medium']:
            return "Medium"
        return "Low"

    def format_context(self, text: str, max_length: int = 300) -> str:
        context = text[:max_length].replace("\n", " ")
        if len(text) > max_length:
            context += " …"
        return context

    def format_results(self, results: list[dict]) -> str:
        if not results:
            return "No matching sources found for the given text."

        response_parts = [
            f"**Found {len(results)} potential source(s)** for your quote:\n"
        ]

        for i, result in enumerate(results, start=1):
            score = result["score"]
            badge = self.get_confidence_badge(score)
            context_preview = self.format_context(result["text"])

            response_parts.append(
                f"### Match {i}\n"
                f"- **Document**: `{result['filename']}`\n"
                f"- **Page**: {result['page']}\n"
                f"- **Confidence**: {badge} ({score})\n"
                f"- **Context**:\n"
                f"  > {context_preview}\n"
            )

        return "\n".join(response_parts)


class PDFIngester:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def ingest(self, directory: Path) -> tuple[int, set[str]]:
        chunks = process_pdf_directory(directory)
        if not chunks:
            return 0, set()

        added = self.vector_store.add_chunks(chunks)
        self.vector_store.save()

        files_set = {c.filename for c in chunks}
        return added, files_set

    def format_result(self, added: int, files_set: set[str]) -> str:
        if added == 0:
            return "PDFs were found but no text could be extracted."

        return (
            f"**Ingestion complete!**\n\n"
            f"- **PDFs processed**: {len(files_set)}\n"
            f"- **Chunks created**: {added}\n"
            f"- **Total chunks in store**: {self.vector_store.total_chunks}\n"
            f"- **Files**: {', '.join(sorted(files_set))}\n\n"
            f"You can now use `find_orphaned_quote` to search these documents."
        )


class ArxivIngester:
    def __init__(self, vector_store: VectorStore, data_dir: Path):
        self.vector_store = vector_store
        self.data_dir = data_dir

    def ingest(self, category_prefix: str, max_records: int) -> str:
        raw_path = self.data_dir / "arxiv-metadata-oai-snapshot.json"
        if not raw_path.exists():
            return (
                "arXiv dataset not found.\n\n"
                f"Expected file at: `{raw_path}`\n"
                "Download it from: https://www.kaggle.com/Cornell-University/arxiv"
            )

        preprocessed_path = self.data_dir / "arxiv_preprocessed.jsonl"
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

        chunks = self._create_chunks(preprocessed_path, max_records)

        if not chunks:
            return "No arXiv records matched the filter criteria."

        added = self.vector_store.add_chunks(chunks)
        self.vector_store.save()

        top_cats = sorted(stats.categories_seen.items(), key=lambda x: -x[1])[:10]
        cats_str = ", ".join(f"{cat} ({n})" for cat, n in top_cats)

        return (
            f"**arXiv Ingestion Complete!**\n\n"
            f"- **Records preprocessed**: {stats.records_output:,}\n"
            f"- **Chunks added to store**: {added:,}\n"
            f"- **Total chunks in store**: {self.vector_store.total_chunks:,}\n"
            f"- **Category filter**: `{category_prefix}`\n"
            f"- **Top categories**: {cats_str}\n"
            f"- **Preprocessing time**: {stats.elapsed_seconds:.1f}s\n\n"
            f"You can now use `find_orphaned_quote` to search across "
            f"both your PDFs and arXiv papers."
        )

    def _create_chunks(self, preprocessed_path: Path, max_records: int) -> list[TextChunk]:
        chunks = []
        for record in stream_arxiv_records(preprocessed_path, max_records=max_records):
            text = f"{record.title}. {record.abstract}"
            chunk = TextChunk(
                text=text,
                filename=f"arxiv:{record.arxiv_id}",
                page=0,
                chunk_index=0,
                start_char=0,
                end_char=len(text),
            )
            chunks.append(chunk)
        return chunks


class SourceSleuthService:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.citation_formatter = CitationFormatter()
        self.pdf_ingester = PDFIngester(vector_store)
        self.arxiv_ingester = ArxivIngester(vector_store, DATA_DIR)

    def find_orphaned_quote(self, quote: str, top_k: int = 5) -> str:
        if self.vector_store.total_chunks == 0:
            return (
                "No PDFs have been ingested yet.\n\n"
                "Please run the `ingest_pdfs` tool first to index your "
                "academic papers, then try again."
            )

        results = self.vector_store.search(query=quote, top_k=top_k)
        return self.citation_formatter.format_results(results)

    def ingest_pdfs(self, directory: Path) -> str:
        if not directory.is_dir():
            return f"Directory not found: `{directory}`"

        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            return f"No PDF files found in `{directory}`."

        added, files_set = self.pdf_ingester.ingest(directory)
        return self.pdf_ingester.format_result(added, files_set)

    def get_store_stats(self) -> str:
        stats = self.vector_store.get_stats()

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

    def ingest_arxiv(self, category_prefix: str, max_records: int) -> str:
        return self.arxiv_ingester.ingest(category_prefix, max_records)

    def get_pdf_text(self, filename: str) -> str:
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

    def format_citation(
        self,
        quote: str,
        source_filename: str,
        page_number: int,
        citation_style: str = "APA",
    ) -> str:
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


mcp = FastMCP(
    "SourceSleuth",
    instructions=(
        "A local MCP server that helps students recover citations "
        "for orphaned quotes by semantically searching their academic PDFs."
    ),
)

store = VectorStore(data_dir=DATA_DIR)
_loaded = store.load()
if _loaded:
    logger.info("Restored vector store with %d chunks.", store.total_chunks)
else:
    logger.info("Starting with an empty vector store.")

service = SourceSleuthService(store)


@mcp.tool()
def find_orphaned_quote(quote: str, top_k: int = 5) -> str:
    return service.find_orphaned_quote(quote, top_k)


@mcp.tool()
def ingest_pdfs(directory: str = "") -> str:
    target_dir = Path(directory) if directory else PDF_DIR
    return service.ingest_pdfs(target_dir)


@mcp.tool()
def get_store_stats() -> str:
    return service.get_store_stats()


@mcp.tool()
def ingest_arxiv(category_prefix: str = "cs.", max_records: int = 5000) -> str:
    return service.ingest_arxiv(category_prefix, max_records)


@mcp.resource("sourcesleuth://pdfs/{filename}")
def get_pdf_text(filename: str) -> str:
    return service.get_pdf_text(filename)


@mcp.prompt()
def cite_recovered_source(
    quote: str,
    source_filename: str,
    page_number: int,
    citation_style: str = "APA",
) -> str:
    return service.format_citation(quote, source_filename, page_number, citation_style)


def main():
    logger.info("Starting SourceSleuth MCP Server v1.0.0 …")
    logger.info("PDF directory : %s", PDF_DIR)
    logger.info("Data directory: %s", DATA_DIR)
    mcp.run()


if __name__ == "__main__":
    main()
