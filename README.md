# SourceSleuth

> Recover citations for orphaned quotes using local semantic search, powered by MCP.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)

---

## The Problem

Every student has been there: you're polishing your research paper and find a brilliant quote but you've lost the citation. Which paper was it from? Which page?

SourceSleuth solves this by running a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that semantically searches your academic PDFs. Connect it to your AI assistant (Claude Desktop, Cursor, Windsurf) and ask: *"Where did I get this quote?"*

Everything runs **locally on your machine**   no data leaves your laptop, no API keys needed.

---

## Features

| Capability | Type | Description |
|---|---|---|
| `find_orphaned_quote` | Tool | Semantic search across all your PDFs for a quote or paraphrase |
| `ingest_pdfs` | Tool | Batch-ingest a folder of PDFs into the local vector store |
| `ingest_arxiv` | Tool | Preprocess and ingest arXiv paper abstracts for citation recovery |
| `get_store_stats` | Tool | View statistics about indexed documents |
| `sourcesleuth://pdfs/{filename}` | Resource | Read the full text of any indexed PDF |
| `cite_recovered_source` | Prompt | Format recovered sources into proper APA / MLA / Chicago citations |

---

## Architecture

```
MCP Host (Claude Desktop / Cursor / Windsurf)
  └── MCP Client  ──stdio──>  SourceSleuth MCP Server
                                    |
                  ┌─────────────────┼─────────────────┐
                  |                 |                 |
           PDF Processor      Vector Store    SentenceTransformer
           (PyMuPDF)          (FAISS)         (all-MiniLM-L6-v2)
                  |                 |
           student_pdfs/         data/
           (your papers)     (persisted index)
```

| Module | Responsibility |
|---|---|
| `src/mcp_server.py` | FastMCP server   exposes tools, resources, and prompts |
| `src/pdf_processor.py` | PDF text extraction (PyMuPDF) and chunking |
| `src/vector_store.py` | FAISS index management, embedding, persistence |
| `src/dataset_preprocessor.py` | arXiv metadata preprocessing, LaTeX cleaning, filtering |

---

## Quick Start

### Prerequisites

- Python 3.10+
- An MCP-compatible host such as [Claude Desktop](https://claude.ai/desktop)

### 1. Clone and install

```bash
git clone https://github.com/your-username/sourcesleuth.git
cd sourcesleuth

python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

pip install -e ".[dev]"
```

### 2. Add your PDFs

Drop your academic PDF files into the `student_pdfs/` directory:

```bash
cp ~/Downloads/research_paper.pdf student_pdfs/
```

### 3. Configure your MCP host

**Claude Desktop**   add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sourcesleuth": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/sourcesleuth"
    }
  }
}
```

**Cursor / Windsurf**   add to your MCP settings:

```json
{
  "sourcesleuth": {
    "command": "python",
    "args": ["-m", "src.mcp_server"],
    "cwd": "/path/to/sourcesleuth"
  }
}
```

### 4. Use it

```
"Ingest my PDFs from the student_pdfs folder."
```

```
"Where did I get this quote: 'Attention is all you need for sequence transduction'?"
```

---

## AI/ML Documentation

*All model and data choices are documented here for reproducibility.*

### Embedding model

| Parameter | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Type | Sentence-Transformer (bi-encoder) |
| Embedding dimension | 384 |
| Model size | ~80 MB |
| Training data | 1B+ sentence pairs (NLI, paraphrase, QA) |
| Hardware requirement | CPU only   no GPU needed |

This model was chosen for its CPU efficiency, strong zero-shot performance on semantic similarity, small footprint, and active maintenance within the Sentence-Transformers library.

### PDF pipeline

| Parameter | Value | Rationale |
|---|---|---|
| Input data | Student's local PDF files | Privacy-first: no data leaves the machine |
| Text extraction | PyMuPDF (`fitz`) | Fast, accurate, handles complex layouts |
| Chunk size | 500 tokens (~2,000 chars) | Balances granularity with context retention |
| Chunk overlap | 50 tokens (~200 chars) | Ensures boundary sentences stay recoverable |
| Token estimation | ~4 chars / token | Approximation for English academic text |

### arXiv metadata pipeline

| Parameter | Value | Rationale |
|---|---|---|
| Source | Kaggle arXiv Dataset | ~2.97M papers, comprehensive academic coverage |
| Raw size | ~5 GB (JSON-Lines) | One JSON object per line |
| Processing | Stream-read, line-by-line | Never loads full file into memory |
| Text cleaning | Strip LaTeX, accents | Produces clean text suitable for embedding |
| Filtering | arXiv category prefix and date | Creates focused, manageable subsets |
| Output | Cleaned JSON-Lines | Fields: id, title, authors, abstract, categories, doi |

### Vector search

| Parameter | Value |
|---|---|
| Index type | FAISS `IndexFlatIP` |
| Similarity metric | Cosine similarity (L2-normalized inner product) |
| Search complexity | O(n) exact search |
| Persistence | Binary FAISS index + JSON metadata |

For the expected corpus size (fewer than 100k chunks), exact search is fast enough and guarantees the best possible results. Approximate indices such as IVF or HNSW add complexity without meaningful benefit at this scale.

---

## Configuration

SourceSleuth is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SOURCESLEUTH_PDF_DIR` | `./student_pdfs` | Directory containing PDF files |
| `SOURCESLEUTH_DATA_DIR` | `./data` | Directory for persisted vector store |

```bash
export SOURCESLEUTH_PDF_DIR="/home/student/papers"
export SOURCESLEUTH_DATA_DIR="/home/student/.sourcesleuth/data"
```

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific module
pytest tests/test_pdf_processor.py -v
```

---

## Project Structure

```
sourcesleuth/
├── src/
│   ├── __init__.py
│   ├── mcp_server.py            # MCP server: tools, resources, prompts
│   ├── pdf_processor.py         # PDF extraction and chunking
│   ├── vector_store.py          # FAISS vector store
│   └── dataset_preprocessor.py  # arXiv metadata preprocessing
├── student_pdfs/                # Place your PDF files here
├── data/                        # Persisted vector store and arXiv dataset
├── tests/
│   ├── test_pdf_processor.py
│   ├── test_vector_store.py
│   ├── test_mcp_server.py
│   └── test_dataset_preprocessor.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where help is appreciated:

- Bug fixes in PDF parsing edge cases
- Additional format support: EPUB, DOCX, Markdown
- Alternative embedding model support
- Improved citation output formatting
- Quote similarity comparison tooling
- Expanded test coverage

---

## License

Licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io)   open standard for AI tool integration
- [Sentence-Transformers](https://sbert.net)   state-of-the-art sentence embeddings
- [FAISS](https://github.com/facebookresearch/faiss)   efficient similarity search
- [PyMuPDF](https://pymupdf.readthedocs.io)   fast PDF text extraction
