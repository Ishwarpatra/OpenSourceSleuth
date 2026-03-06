from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("sourcesleuth.preprocessor")


@dataclass
class ArxivRecord:
    arxiv_id: str
    title: str
    authors: str
    abstract: str
    categories: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    update_date: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def searchable_text(self) -> str:
        return f"{self.title}. {self.abstract}"


class TextCleaner:
    _LATEX_CMD_RE = re.compile(
        r"\\(?:textbf|textit|emph|text|mathrm|mathbf|mathcal|mathbb|operatorname)"
        r"\{([^}]*)\}",
    )
    _LATEX_ACCENT_RE = re.compile(r"\\['\"`~^=.uvHtcdb]\{?(\w)\}?")
    _LATEX_DOLLAR_RE = re.compile(r"\$([^$]*)\$")
    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines)

        text = self._LATEX_CMD_RE.sub(r"\1", text)
        text = self._LATEX_ACCENT_RE.sub(r"\1", text)
        text = self._LATEX_DOLLAR_RE.sub(r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
        text = re.sub(r"\\([a-zA-Z])", r"\1", text)

        text = self._MULTI_SPACE_RE.sub(" ", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)

        return text.strip()

    def clean_title(self, title: str) -> str:
        title = title.replace("\n", " ").replace("\r", " ")
        return self.clean(title)


class AuthorFormatter:
    def format(self, authors_parsed: list[list[str]] | None, authors_str: str) -> str:
        if not authors_parsed:
            return TextCleaner().clean(authors_str)

        names = []
        for parts in authors_parsed:
            name = self._format_author_parts(parts)
            names.append(name)

        return ", ".join(names)

    def _format_author_parts(self, parts: list[str]) -> str:
        last = parts[0].strip() if len(parts) > 0 else ""
        first = parts[1].strip() if len(parts) > 1 else ""
        suffix = parts[2].strip() if len(parts) > 2 else ""

        if suffix:
            return f"{first} {last} {suffix}".strip()
        elif first:
            return f"{first} {last}".strip()
        return last


class ArxivRecordBuilder:
    def __init__(self, text_cleaner: TextCleaner = None, author_formatter: AuthorFormatter = None):
        self.text_cleaner = text_cleaner or TextCleaner()
        self.author_formatter = author_formatter or AuthorFormatter()

    def build(self, raw: dict) -> ArxivRecord:
        return ArxivRecord(
            arxiv_id=raw.get("id", ""),
            title=self.text_cleaner.clean_title(raw.get("title", "")),
            authors=self.author_formatter.format(
                raw.get("authors_parsed"),
                raw.get("authors", ""),
            ),
            abstract=self.text_cleaner.clean(raw.get("abstract", "")),
            categories=raw.get("categories", ""),
            doi=raw.get("doi"),
            journal_ref=raw.get("journal-ref"),
            update_date=raw.get("update_date", ""),
        )


class RecordFilter:
    def __init__(
        self,
        categories_filter: set[str] | None = None,
        category_prefix_filter: set[str] | None = None,
        start_date: str | None = None,
    ):
        self.categories_filter = categories_filter
        self.category_prefix_filter = category_prefix_filter
        self.start_date = start_date

    def matches(self, raw: dict) -> bool:
        cats = raw.get("categories", "")

        if self.categories_filter:
            record_cats = set(cats.split())
            if not record_cats.intersection(self.categories_filter):
                return False

        if self.category_prefix_filter:
            record_cats = cats.split()
            if not any(
                cat.startswith(prefix)
                for cat in record_cats
                for prefix in self.category_prefix_filter
            ):
                return False

        update_date = raw.get("update_date", "")
        if self.start_date and update_date < self.start_date:
            return False

        return True


class ArxivRecordStream:
    def __init__(
        self,
        filepath: Path,
        record_filter: RecordFilter = None,
        max_records: int | None = None,
    ):
        self.filepath = filepath
        self.record_filter = record_filter or RecordFilter()
        self.max_records = max_records
        self.record_builder = ArxivRecordBuilder()

    def stream(self) -> Iterator[ArxivRecord]:
        if not self.filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.filepath}")

        yielded = 0
        skipped = 0

        with open(self.filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d", line_num)
                    skipped += 1
                    continue

                if not self.record_filter.matches(raw):
                    continue

                record = self.record_builder.build(raw)

                if not record.abstract:
                    skipped += 1
                    continue

                yield record
                yielded += 1

                if self.max_records and yielded >= self.max_records:
                    break

                if yielded % 100_000 == 0:
                    logger.info("  … streamed %d records so far …", yielded)

        logger.info(
            "Streaming complete: %d records yielded, %d skipped, %d lines read.",
            yielded, skipped, line_num,
        )


@dataclass
class PreprocessingStats:
    total_input_lines: int = 0
    records_output: int = 0
    records_skipped: int = 0
    categories_seen: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        top_cats = sorted(
            self.categories_seen.items(), key=lambda x: -x[1]
        )[:15]
        cat_lines = "\n".join(f"    {cat}: {count:,}" for cat, count in top_cats)
        return (
            f"Preprocessing Summary\n"
            f"{'=' * 40}\n"
            f"  Input lines read  : {self.total_input_lines:,}\n"
            f"  Records output    : {self.records_output:,}\n"
            f"  Records skipped   : {self.records_skipped:,}\n"
            f"  Unique categories : {len(self.categories_seen):,}\n"
            f"  Elapsed time      : {self.elapsed_seconds:.1f}s\n"
            f"\n  Top 15 categories:\n{cat_lines}"
        )


class ArxivPreprocessor:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def preprocess(
        self,
        input_path: Path,
        categories_filter: set[str] | None = None,
        category_prefix_filter: set[str] | None = None,
        start_date: str | None = None,
        max_records: int | None = None,
    ) -> PreprocessingStats:
        stats = PreprocessingStats()
        t0 = time.time()

        logger.info("Starting preprocessing: %s -> %s", input_path, self.output_path)
        if categories_filter:
            logger.info("  Category filter: %s", categories_filter)
        if category_prefix_filter:
            logger.info("  Category prefix filter: %s", category_prefix_filter)
        if start_date:
            logger.info("  Start date filter: %s", start_date)
        if max_records:
            logger.info("  Max records: %d", max_records)

        record_filter = RecordFilter(
            categories_filter=categories_filter,
            category_prefix_filter=category_prefix_filter,
            start_date=start_date,
        )

        stream = ArxivRecordStream(
            filepath=input_path,
            record_filter=record_filter,
            max_records=max_records,
        )

        with open(self.output_path, "w", encoding="utf-8") as out:
            for record in stream.stream():
                out.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                stats.records_output += 1

                for cat in record.categories.split():
                    stats.categories_seen[cat] = stats.categories_seen.get(cat, 0) + 1

        with open(input_path, "r", encoding="utf-8") as f:
            stats.total_input_lines = sum(1 for _ in f)

        stats.records_skipped = stats.total_input_lines - stats.records_output
        stats.elapsed_seconds = time.time() - t0

        logger.info(stats.summary())
        return stats


def stream_arxiv_records(
    filepath: str | Path,
    categories_filter: set[str] | None = None,
    category_prefix_filter: set[str] | None = None,
    start_date: str | None = None,
    max_records: int | None = None,
) -> Iterator[ArxivRecord]:
    filepath = Path(filepath)
    record_filter = RecordFilter(
        categories_filter=categories_filter,
        category_prefix_filter=category_prefix_filter,
        start_date=start_date,
    )
    stream = ArxivRecordStream(
        filepath=filepath,
        record_filter=record_filter,
        max_records=max_records,
    )
    return stream.stream()


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    categories_filter: set[str] | None = None,
    category_prefix_filter: set[str] | None = None,
    start_date: str | None = None,
    max_records: int | None = None,
) -> PreprocessingStats:
    preprocessor = ArxivPreprocessor(Path(output_path))
    return preprocessor.preprocess(
        input_path=Path(input_path),
        categories_filter=categories_filter,
        category_prefix_filter=category_prefix_filter,
        start_date=start_date,
        max_records=max_records,
    )


def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess arXiv metadata for SourceSleuth.",
    )
    parser.add_argument(
        "--input", "-i",
        default="data/arxiv-metadata-oai-snapshot.json",
        help="Path to raw arXiv JSON-Lines file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/arxiv_preprocessed.jsonl",
        help="Path for cleaned output file.",
    )
    parser.add_argument(
        "--categories", "-c",
        nargs="*",
        default=None,
        help="Exact arXiv categories to include (e.g. cs.AI cs.CL).",
    )
    parser.add_argument(
        "--category-prefix", "-p",
        nargs="*",
        default=None,
        help="Category prefixes to include (e.g. cs. stat. math.).",
    )
    parser.add_argument(
        "--start-date", "-d",
        default=None,
        help="Only include records updated on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--max-records", "-n",
        type=int,
        default=None,
        help="Maximum number of records to output.",
    )

    args = parser.parse_args()

    cats = set(args.categories) if args.categories else None
    prefixes = set(args.category_prefix) if args.category_prefix else None

    stats = preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        categories_filter=cats,
        category_prefix_filter=prefixes,
        start_date=args.start_date,
        max_records=args.max_records,
    )

    print("\n" + stats.summary())


if __name__ == "__main__":
    main()
