# Contributing to SourceSleuth

Thank you for your interest in contributing to SourceSleuth! This document provides guidelines and instructions for contributing.

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/your-username/sourcesleuth.git
cd sourcesleuth
```

### 2. Set Up Your Development Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

pip install -e ".[dev]"
```

### 3. Run the Tests

```bash
pytest -v
```

Make sure all tests pass before making changes.

---

## 📋 How to Contribute

### Reporting Bugs

- Open a GitHub Issue with the **Bug Report** template.
- Include: steps to reproduce, expected behavior, actual behavior, and your environment (OS, Python version).

### Suggesting Features

- Open a GitHub Issue with the **Feature Request** template.
- Describe the use case and why it would benefit students.

### Submitting Code

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes** following the code style guidelines below.
3. **Write tests** for any new functionality.
4. **Run the tests**: `pytest -v`
5. **Run the linter**: `ruff check src/ tests/`
6. **Commit** with a clear message: `git commit -m "feat: add DOCX ingestion support"`
7. **Push & open a Pull Request** against `main`.

---

## 🎨 Code Style

- **Formatter/Linter**: We use [Ruff](https://docs.astral.sh/ruff/) for linting.
- **Line length**: 100 characters max.
- **Type hints**: Use type hints for all function signatures.
- **Docstrings**: Use Google-style docstrings for all public functions and classes.
- **Logging**: Use `logging.getLogger("sourcesleuth.<module>")` instead of `print()`.

### Example

```python
def process_document(path: str | Path, chunk_size: int = 500) -> list[TextChunk]:
    """
    Process a single document into text chunks.

    Args:
        path: Path to the document file.
        chunk_size: Target chunk size in tokens.

    Returns:
        A list of TextChunk objects ready for embedding.

    Raises:
        FileNotFoundError: If the document does not exist.
    """
    ...
```

---

## 🏗️ Project Architecture

Understanding the modular architecture helps you contribute to the right place:

```
src/
├── mcp_server.py      ← MCP interface (tools, resources, prompts)
├── pdf_processor.py   ← Text extraction & chunking logic
└── vector_store.py    ← FAISS index & embedding management
```

- **Want to add a new MCP tool?** → Edit `mcp_server.py`
- **Want to support a new file format?** → Edit `pdf_processor.py`
- **Want to change the embedding/search strategy?** → Edit `vector_store.py`

---

## 🧪 Testing Guidelines

- Place tests in the `tests/` directory.
- Name test files as `test_<module>.py`.
- Use `pytest` fixtures for shared setup.
- Aim for tests that are **fast** (no network calls) and **deterministic**.
- Use `tmp_path` fixture for temporary files.

### Running Specific Tests

```bash
# All tests
pytest

# Single module
pytest tests/test_pdf_processor.py -v

# Single test
pytest tests/test_vector_store.py::TestVectorStoreCore::test_search_relevance -v
```

---

## 📝 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use For |
|---|---|
| `feat:` | New features |
| `fix:` | Bug fixes |
| `docs:` | Documentation changes |
| `test:` | Adding or updating tests |
| `refactor:` | Code changes that don't add features or fix bugs |
| `chore:` | Maintenance tasks |

---

## 💡 Contribution Ideas

Here are some areas where contributions are especially welcome:

### 🟢 Good First Issues
- Add input validation to tool arguments
- Improve error messages for common failure modes
- Add more citation styles (IEEE, Vancouver)

### 🟡 Intermediate
- Support EPUB and DOCX file formats
- Add a `remove_pdf` tool to un-ingest a specific file
- Implement chunk deduplication

### 🔴 Advanced
- Support alternative embedding models (configurable)
- Implement approximate nearest neighbor search (IVF/HNSW) for large corpora
- Add a `compare_quotes` tool for plagiarism-style comparison
- Build a CLI for non-MCP usage

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
