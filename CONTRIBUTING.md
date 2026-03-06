# Contributing to SourceSleuth

Thank you for your interest in contributing to SourceSleuth! This document provides guidelines and instructions for contributing.

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers via [GitHub Issues](https://github.com/Ishwarpatra/OpenSourceSleuth/issues).

---

## Getting Started

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

# Install with all optional dependencies (dev tools, UI, OCR)
pip install -e ".[dev,ui,ocr]"
```

This installs:
- **pytest** - Testing framework
- **ruff** - Linting and formatting
- **streamlit** - Web UI development
- **nltk** - Query expansion
- **pytesseract, pdf2image, Pillow** - OCR support

### 3. Install Pre-commit Hooks (Recommended)

Pre-commit hooks automatically check your code for common issues before each commit:

```bash
pip install pre-commit
pre-commit install
```

This installs hooks that run on every `git commit`:
- **Ruff linter** - Catches bugs and style issues
- **Ruff formatter** - Ensures consistent code formatting
- **Trailing whitespace remover** - Cleans up whitespace
- **End-of-file fixer** - Ensures files end with a newline
- **YAML/JSON validator** - Checks config file syntax
- **Large file detector** - Prevents accidentally committing large files

Pre-commit will automatically fix issues where possible. If a fix can't be applied automatically, the commit will be blocked with an error message.

> **Note:** Pre-commit is also run in CI on all pull requests. The CI will auto-fix formatting issues on your PR branch.

### 4. Run the Tests

```bash
pytest -v
```

Make sure all tests pass before making changes.

---

## How to Contribute

### Reporting Bugs

- Open a GitHub Issue using the **[Bug Report](.github/ISSUE_TEMPLATE/bug_report.md)** template.
- Include: steps to reproduce, expected behavior, actual behavior, and your environment (OS, Python version).
- Attach relevant logs or screenshots if applicable.

### Suggesting Features

- Open a GitHub Issue using the **[Feature Request](.github/ISSUE_TEMPLATE/feature_request.md)** template.
- Describe the use case and why it would benefit students.
- Check the [ROADMAP.md](ROADMAP.md) to see if this is already planned.

### Submitting Code

1. **Find an issue**: Look for existing issues or create a new one describing your proposed change.
2. **Create a branch**: `git checkout -b feature/your-feature-name` (see [Branch Naming](#branch-naming-conventions)).
3. **Make your changes** following the code style guidelines below.
4. **Write tests** for any new functionality.
5. **Run the tests**: `pytest -v`
6. **Run the linter and formatter**: `ruff check src/ tests/ && ruff format src/ tests/ --check`
7. **Commit** with a clear message: `git commit -m "feat: add DOCX ingestion support"` (see [Commit Messages](#commit-message-format)).
8. **Push & open a Pull Request** against `main`.
9. **Fill out the PR template** at [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).

---

## Branch Naming Conventions

Use descriptive branch names with the following prefixes:

| Prefix | Use For | Example |
|--------|---------|---------|
| `feature/` | New features | `feature/docx-ingestion` |
| `bugfix/` | Bug fixes | `bugfix/pdf-chunk-overlap` |
| `docs/` | Documentation updates | `docs/readme-examples` |
| `test/` | Test additions | `test/vector-store-coverage` |
| `refactor/` | Code refactoring | `refactor/mcp-tool-validation` |
| `chore/` | Maintenance tasks | `chore/update-dependencies` |

---

## Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat:` | A new feature |
| `fix:` | A bug fix |
| `docs:` | Documentation only changes |
| `test:` | Adding or updating tests |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `chore:` | Changes to the build process, tooling, or configuration |
| `perf:` | Performance improvement |
| `style:` | Formatting, missing semi-colons, etc. (no code change) |

### Examples

```
feat(mcp): add remove_pdf tool for un-ingesting files
fix(vector): correct cosine similarity calculation edge case
docs(readme): add usage examples for CLI ingestion
test(pdf): add unit tests for chunk overlap logic
refactor(core): extract embedding logic into separate module
chore(deps): bump sentence-transformers to 3.1.0
```

---

## Pull Request Process

### Before Opening a PR

- [ ] Ensure your branch is up to date with `main`: `git pull origin main`
- [ ] Run all tests: `pytest -v`
- [ ] Run the linter and formatter: `ruff check src/ tests/ && ruff format src/ tests/ --check`
- [ ] Update documentation if needed
- [ ] Add tests for new functionality
- [ ] Verify pre-commit hooks pass: `pre-commit run --all-files`

### Continuous Integration (CI)

This repository uses GitHub Actions and pre-commit.ci for automated quality checks:

| Check | Description | Required |
|-------|-------------|----------|
| **Lint** | Ruff linter and formatter (check mode) | Yes |
| **Test** | Pytest across Python 3.10, 3.11, 3.12 | Yes |
| **Build** | Verify package builds correctly | Yes |
| **Security** | Bandit security scan, dependency check | Advisory |

All CI checks must pass before a PR can be merged. The CI runs automatically when you:
- Push to a branch with an open PR
- Push to the `main` branch

**Pre-commit.ci Auto-fix:** If your PR fails linting due to formatting issues, pre-commit.ci will automatically create a commit with fixes on your branch.

### CodeRabbit AI Review

This repository uses [CodeRabbit](https://coderabbit.ai) for AI-powered code review. CodeRabbit will:
- Review your PR automatically
- Provide line-by-line feedback on code quality
- Check for MCP best practices and local-first architecture compliance
- Suggest improvements for performance and readability

**Important:** CodeRabbit is configured to reject any changes that introduce external LLM API calls. This project must remain local-first.

### PR Review Criteria

Reviewers will evaluate your PR based on:

1. **Correctness**: Does the code work as intended?
2. **Testing**: Are there adequate tests? Do they pass?
3. **Code Style**: Does it follow the project's style guidelines?
4. **Documentation**: Is the code documented? Are user-facing changes reflected in README/docs?
5. **Impact**: Does this introduce breaking changes? Are they documented?
6. **CI Status**: Do all CI checks pass (lint, test, build)?

### Merge Process

1. All PRs require at least one maintainer approval.
2. All CI checks must pass (tests, linting, build).
3. CodeRabbit review completed (or manually overridden by maintainer).
4. The PR author should merge after approval (unless maintainer prefers to merge).
5. Delete the branch after merging (unless it's a long-lived branch).

---

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for all linting and formatting. Ruff is a fast Python linter and formatter that replaces black, flake8, isort, and other tools with a single tool.

### Configuration

All linting rules are defined in [`pyproject.toml`](pyproject.toml):

```toml
[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort (import sorting)
    "N",      # pep8-naming
    "UP",     # pyupgrade (modern Python syntax)
    "B",      # flake8-bugbear (common bugs)
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
    "ARG",    # flake8-unused-arguments
    "RUF",    # Ruff-specific rules
    "ANN",    # flake8-annotations (type hints)
]
```

### Running Linter and Formatter

```bash
# Check for linting issues
ruff check src/ tests/

# Auto-fix issues where possible
ruff check src/ tests/ --fix

# Check formatting (check mode)
ruff format src/ tests/ --check

# Format code (write mode)
ruff format src/ tests/
```

### Code Style Rules

- **Line length**: 100 characters max
- **Type hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Imports**: Sorted automatically by Ruff (isort)
- **Quotes**: Double quotes for strings
- **Logging**: Use `logging.getLogger("sourcesleuth.<module>")` instead of `print()`

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

### Disabling Rules

If you need to disable a rule for a specific line or file:

```python
# Disable for a line
result = some_function()  # noqa: ANN201

# Disable for a file
# ruff: noqa: ANN001, ANN201
```

Use sparingly and only when there's a good reason.

---

## Project Architecture

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

## Testing Guidelines

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

## Contribution Ideas

Here are some areas where contributions are especially welcome:

### Good First Issues
- Add input validation to tool arguments
- Improve error messages for common failure modes
- Add more citation styles (IEEE, Vancouver)
- Write unit tests for existing functionality
- Improve documentation or add examples

### Intermediate
- Support EPUB and DOCX file formats
- Add a `remove_pdf` tool to un-ingest a specific file
- Implement chunk deduplication
- Add hybrid search (BM25 + semantic)
- Create tutorials or blog posts

### Advanced
- Support alternative embedding models (configurable)
- Implement approximate nearest neighbor search (IVF/HNSW) for large corpora
- Add a `compare_quotes` tool for plagiarism-style comparison
- Build a CLI for non-MCP usage
- OCR integration for scanned documents

---

## First Contribution Guide

New to open source? Here's how to make your first contribution:

1. **Find a "good first issue"**: Look for issues labeled [`good first issue`](https://github.com/Ishwarpatra/OpenSourceSleuth/labels/good%20first%20issue) on GitHub.

2. **Comment on the issue**: Let others know you're working on it to avoid duplicate effort.

3. **Set up your environment**: Follow the [Getting Started](#-getting-started) guide above.

4. **Make your changes**: Start small and focus on one issue at a time.

5. **Ask for help**: If you get stuck, comment on the issue or open a draft PR to ask for guidance.

6. **Submit your PR**: Follow the [PR Process](#-pull-request-process) and don't worry about making mistakes—reviewers are here to help!

---

## Getting Help

- **Documentation**: Check the [README.md](README.md), [MODEL_CARD.md](MODEL_CARD.md), and [examples/](examples/) directory.
- **Issues**: Search existing issues or create a new one.
- **Discussions**: Use GitHub Discussions for questions and ideas.

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
