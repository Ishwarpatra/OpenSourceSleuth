# SourceSleuth

> Recover citations for orphaned quotes using local semantic search, powered by MCP.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io)

---

## The Problem

Research involves handling vast amounts of data. Often, brilliant quotes are captured but their specific source or page number is lost. 

**SourceSleuth** solves this by running a local [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that semantically searches your academic PDFs. You can connect it to AI assistants like Claude Desktop, Cursor, and Windsurf, or use the dedicated Web UI to identify exactly where a quote originated.

Everything runs **locally on your machine**; no data leaves your hardware and no external API keys are required.

---

## Features

| Capability | Type | Description |
|---|---|---|
| `find_orphaned_quote` | Tool | Semantic search across all indexed PDFs for a quote or paraphrase |
| `ingest_pdfs` | Tool | Batch-ingest a folder of PDFs into the local vector store |
| `ingest_arxiv` | Tool | Preprocess and ingest arXiv paper abstracts for citation recovery |
| `get_store_stats` | Tool | View statistics about indexed documents and total chunks |
| `sourcesleuth://pdfs/{filename}` | Resource | Access and read the full text of any indexed PDF |
| **Web UI** | Interface | Interactive Streamlit browser-based search interface |
| **CLI** | Interface | Standalone command-line tool for ingestion and management |

---

## Architecture
MCP Host (Claude Desktop / Cursor / Windsurf)
└── MCP Client  ──stdio──>  SourceSleuth MCP Server
|
┌─────────────────┼─────────────────┐
|                 |                 |
PDF Processor      Vector Store      SentenceTransformer
(PyMuPDF)          (FAISS)           (all-MiniLM-L6-v2)
|                 |
student_pdfs/         data/
(your papers)     (persisted index)
| Module | Responsibility |
|---|---|
| `src/mcp_server.py` | FastMCP server exposing tools, resources, and prompts |
| `src/pdf_processor.py` | PDF text extraction and chunking |
| `src/vector_store.py` | FAISS index management and semantic embedding |
| `src/dataset_preprocessor.py` | arXiv metadata cleaning and filtering |
| `src/ingest.py` | CLI tool for standalone ingestion |
| `app.py` | Streamlit web UI |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- An MCP-compatible host (e.g., [Claude Desktop](https://claude.ai/desktop)) — optional

### 1. Installation
```bash
git clone [https://github.com/Ishwarpatra/OpenSourceSleuth.git](https://github.com/Ishwarpatra/OpenSourceSleuth.git)
cd sourcesleuth

python -m venv .venv
# Linux/macOS: source .venv/bin/activate | Windows: .venv\Scripts\activate

pip install -e ".[dev,ui]"
2. Add Your PDFs
Drop your academic PDF files into the student_pdfs/ directory.

3. Launch the Web UI
Bash
```
streamlit run app.py
Access the dashboard at http://localhost:8501 to search, upload PDFs, and view index statistics.

🖥️ CLI Usage
For power users, SourceSleuth includes a standalone CLI:
```
Bash
```
# Ingest PDFs from a specific directory
python -m src.ingest pdfs --directory /path/to/pdfs

# Ingest arXiv papers by category
python -m src.ingest arxiv --category cs.AI --max-records 5000

# View store statistics
python -m src.ingest stats
🧪 Testing
```
Bash
```
# Run the full test suite
pytest

# Run specific integration tests
pytest tests/test_integration_pdfs.py -v
```
📄 License
Licensed under the Apache 2.0 License. See LICENSE for details.


### Changes made:
* **Resolved Conflicts:** Removed all `<<<<<<<`, `=======`, and `>>>>>>>` markers.
* **Integrated Web UI:** Added the Web UI to the Features table, Module list, and Quick Start instructions.
* **Refined CLI Section:** Included the standalone CLI usage instructions originally found in the `main` branch version.
* **Professional Formatting:** Standardized table alignments and added icons for better