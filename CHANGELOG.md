# Changelog

All notable changes to SourceSleuth will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - March 2026

### Added

#### MCP Server
- MCP server with stdio transport for integration with AI assistants
- `find_orphaned_quote` tool — semantic search for orphaned quotes across indexed PDFs
- `ingest_pdfs` tool — batch ingestion of PDFs from a directory
- `ingest_arxiv` tool — ingestion of arXiv paper abstracts by category
- `get_store_stats` tool — view statistics about indexed documents and chunks
- `sourcesleuth://pdfs/{filename}` resource — access full text of indexed PDFs
- `cite_recovered_source` prompt — format recovered sources into APA/MLA/Chicago citations

#### Core Functionality
- PDF text extraction and chunking with PyMuPDF
- FAISS vector store with exact inner product search (`IndexFlatIP`)
- SentenceTransformers embeddings using `all-MiniLM-L6-v2`
- Local persistence of vector store and metadata
- Environment-based configuration

#### CLI Tools
- `sourcesleuth` — MCP server entry point
- `sourcesleuth-ingest` — standalone CLI for ingestion and management
- Commands: `pdfs`, `arxiv`, `stats`, `clear`

#### Web UI
- Streamlit-based web interface for interactive search
- PDF upload functionality via web interface
- Real-time index statistics dashboard
- Modern, dark-themed UI with confidence tier visualization

#### Documentation
- Comprehensive README.md with installation and usage guide
- MODEL_CARD.md — model architecture, dataset documentation, reproducibility
- EVALUATION.md — evaluation methodology and results
- ROADMAP.md — development roadmap through v2.0
- CONTRIBUTING.md — contribution guidelines
- CODE_OF_CONDUCT.md — community standards
- LICENSE — Apache 2.0

#### Testing
- Unit tests for `SourceRetriever` class
- Unit tests for PDF processing and chunking
- Unit tests for vector store operations
- MCP server integration tests
- End-to-end integration tests with sample PDFs

### Technical Stack
- **Python**: 3.10+
- **MCP**: mcp[cli]>=1.2.0
- **Embeddings**: sentence-transformers>=3.0.0
- **Vector Search**: faiss-cpu>=1.8.0
- **PDF Processing**: PyMuPDF>=1.24.0
- **Web UI**: streamlit>=1.30.0
- **Development**: pytest, ruff

### Known Issues
- English-only text support
- Math formulas and LaTeX equations embed poorly
- Scanned/image-only PDFs require OCR (planned for v1.1)
- Very short queries (<5 words) may produce unreliable results

---

## [Unreleased]

### Planned for v1.1
- OCR integration for scanned documents (Tesseract)
- Table extraction with structure preservation
- Figure caption recovery
- Configurable chunk size via `.env`
- Progress bars for CLI ingestion

### Under Consideration
- Hybrid search (BM25 + semantic)
- Multi-language support
- Alternative embedding models (configurable)
- Approximate nearest neighbor search for large corpora

---

## Version History

| Version | Release Date | Key Changes |
|---------|--------------|-------------|
| 1.0.0   | March 2026   | Initial release |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

To propose a change, please:
1. Check existing issues or create a new one
2. Create a branch following our naming conventions
3. Submit a pull request with a clear description

---

## License

SourceSleuth is licensed under the [Apache 2.0 License](LICENSE).
