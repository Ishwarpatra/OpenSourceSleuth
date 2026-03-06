from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.vector_store import VectorStore
from src.pdf_processor import process_pdf_directory, TextChunk
from src.dataset_preprocessor import preprocess_dataset, stream_arxiv_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("sourcesleuth.ingest")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(os.environ.get("SOURCESLEUTH_PDF_DIR", str(PROJECT_ROOT / "student_pdfs")))
DATA_DIR = Path(os.environ.get("SOURCESLEUTH_DATA_DIR", str(PROJECT_ROOT / "data")))


class PDFIngestionCommand:
    def __init__(self, pdf_dir: Path, data_dir: Path):
        self.pdf_dir = pdf_dir
        self.data_dir = data_dir

    def execute(self, directory: str) -> int:
        target_dir = Path(directory) if directory else self.pdf_dir

        if not target_dir.is_dir():
            logger.error("Directory not found: %s", target_dir)
            return 1

        pdf_files = list(target_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in '%s'.", target_dir)
            return 0

        logger.info("Found %d PDF(s) to ingest.", len(pdf_files))

        store = VectorStore(data_dir=self.data_dir)
        if store.load():
            logger.info("Loaded existing vector store (%d chunks).", store.total_chunks)

        chunks = process_pdf_directory(target_dir)
        if not chunks:
            logger.warning("No text could be extracted from PDFs.")
            return 0

        added = store.add_chunks(chunks)
        store.save()

        files_set = {c.filename for c in chunks}
        self._log_summary(len(files_set), added, store.total_chunks, files_set)

        return 0

    def _log_summary(self, pdf_count: int, added: int, total: int, files: set[str]) -> None:
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info("PDFs processed:     %d", pdf_count)
        logger.info("Chunks created:     %d", added)
        logger.info("Total chunks:       %d", total)
        logger.info("Files:              %s", ", ".join(sorted(files)))
        logger.info("=" * 60)


class ArxivIngestionCommand:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def execute(self, category: str, max_records: int) -> int:
        raw_path = self.data_dir / "arxiv-metadata-oai-snapshot.json"
        if not raw_path.exists():
            logger.error(
                "arXiv dataset not found at: %s\n"
                "Download from: https://www.kaggle.com/Cornell-University/arxiv",
                raw_path,
            )
            return 1

        preprocessed_path = self.data_dir / "arxiv_preprocessed.jsonl"

        logger.info("Preprocessing arXiv dataset (prefix=%s, max=%d)...",
                    category, max_records)

        prefixes = {p.strip() for p in category.split(",") if p.strip()}
        stats = preprocess_dataset(
            input_path=raw_path,
            output_path=preprocessed_path,
            category_prefix_filter=prefixes,
            max_records=max_records,
        )

        store = VectorStore(data_dir=self.data_dir)
        if store.load():
            logger.info("Loaded existing vector store (%d chunks).", store.total_chunks)

        chunks = self._create_chunks(preprocessed_path, max_records)

        if not chunks:
            logger.warning("No arXiv records matched the filter criteria.")
            return 0

        added = store.add_chunks(chunks)
        store.save()

        top_cats = sorted(stats.categories_seen.items(), key=lambda x: -x[1])[:10]
        cats_str = ", ".join(f"{cat} ({n})" for cat, n in top_cats)

        self._log_summary(stats.records_output, added, store.total_chunks, category, cats_str, stats.elapsed_seconds)

        return 0

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

    def _log_summary(self, records: int, added: int, total: int, category: str, categories: str, time_sec: float) -> None:
        logger.info("=" * 60)
        logger.info("ARXIV INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info("Records preprocessed:   %d", records)
        logger.info("Chunks added:           %d", added)
        logger.info("Total chunks:           %d", total)
        logger.info("Category filter:        %s", category)
        logger.info("Top categories:         %s", categories)
        logger.info("Preprocessing time:     %.1fs", time_sec)
        logger.info("=" * 60)


class StatsCommand:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def execute(self) -> int:
        store = VectorStore(data_dir=self.data_dir)

        if not store.load():
            logger.info("Vector store is empty. Use 'ingest pdfs' to add documents.")
            return 0

        stats = store.get_stats()

        print("\n" + "=" * 60)
        print("VECTOR STORE STATISTICS")
        print("=" * 60)
        print(f"Total chunks:         {stats['total_chunks']}")
        print(f"Number of files:      {stats['num_files']}")
        print(f"Embedding model:      {stats['model_name']}")
        print(f"Embedding dimensions: {stats['embedding_dim']}")
        print(f"Index type:           {stats['index_type']}")
        print("\nIngested files:")
        for f in stats["ingested_files"]:
            print(f"  - {f}")
        print("=" * 60)

        return 0


class ClearCommand:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def execute(self) -> int:
        store = VectorStore(data_dir=self.data_dir)

        if store.load():
            store.clear()
            self._remove_persisted_files()
            logger.info("Vector store cleared.")
        else:
            logger.info("Vector store was already empty.")

        return 0

    def _remove_persisted_files(self) -> None:
        index_path = self.data_dir / "sourcesleuth.index"
        meta_path = self.data_dir / "sourcesleuth_metadata.json"
        if index_path.exists():
            index_path.unlink()
        if meta_path.exists():
            meta_path.unlink()


class CLIApplication:
    def __init__(self, pdf_dir: Path, data_dir: Path):
        self.pdf_dir = pdf_dir
        self.data_dir = data_dir
        self.commands = {
            'pdfs': PDFIngestionCommand(pdf_dir, data_dir),
            'arxiv': ArxivIngestionCommand(data_dir),
            'stats': StatsCommand(data_dir),
            'clear': ClearCommand(data_dir),
        }

    def run(self, args: argparse.Namespace) -> int:
        if args.command is None:
            return 0

        if args.command == 'pdfs':
            return self.commands['pdfs'].execute(args.directory)
        elif args.command == 'arxiv':
            return self.commands['arxiv'].execute(args.category, args.max_records)
        elif args.command == 'stats':
            return self.commands['stats'].execute()
        elif args.command == 'clear':
            return self.commands['clear'].execute()

        return 0


def create_parser(pdf_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sourcesleuth-ingest",
        description="SourceSleuth CLI - Ingest PDFs and arXiv data for semantic search.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    pdf_parser = subparsers.add_parser("pdfs", help="Ingest PDFs from a directory")
    pdf_parser.add_argument(
        "-d", "--directory",
        type=str,
        default="",
        help=f"Directory containing PDFs (default: {pdf_dir})",
    )
    pdf_parser.set_defaults(func='pdfs')

    arxiv_parser = subparsers.add_parser("arxiv", help="Ingest arXiv paper abstracts")
    arxiv_parser.add_argument(
        "-c", "--category",
        type=str,
        default="cs.",
        help="arXiv category prefix (e.g., 'cs.', 'physics.')",
    )
    arxiv_parser.add_argument(
        "-n", "--max-records",
        type=int,
        default=5000,
        help="Maximum number of records to ingest",
    )
    arxiv_parser.set_defaults(func='arxiv')

    stats_parser = subparsers.add_parser("stats", help="Display vector store statistics")
    stats_parser.set_defaults(func='stats')

    clear_parser = subparsers.add_parser("clear", help="Clear the vector store")
    clear_parser.set_defaults(func='clear')

    return parser


def main() -> int:
    parser = create_parser(PDF_DIR)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    app = CLIApplication(PDF_DIR, DATA_DIR)
    return app.run(args)


if __name__ == "__main__":
    sys.exit(main())
