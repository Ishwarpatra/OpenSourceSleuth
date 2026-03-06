# SourceSleuth Examples

This directory contains practical examples demonstrating how to use SourceSleuth in different scenarios.

---

## Table of Contents

1. [MCP Integration with Claude Desktop](#1-mcp-integration-with-claude-desktop)
2. [CLI Usage for PDF Ingestion](#2-cli-usage-for-pdf-ingestion)
3. [Web UI Search Workflow](#3-web-ui-search-workflow)
4. [arXiv Paper Ingestion](#4-arxiv-paper-ingestion)
5. [Custom Embedding Model](#5-custom-embedding-model)

---

## 1. MCP Integration with Claude Desktop

### Setup

Add SourceSleuth to your Claude Desktop configuration:

**macOS/Linux:**
```json
{
  "mcpServers": {
    "sourcesleuth": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/OpenSourceSleuth"
    }
  }
}
```

**Windows:**
```json
{
  "mcpServers": {
    "sourcesleuth": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "C:\\Users\\YourName\\OpenSourceSleuth"
    }
  }
}
```

### Example Conversation

**You:** I found this quote in my draft but lost the source: "The attention mechanism allows models to focus on specific parts of the input sequence, enabling them to capture long-range dependencies more effectively than recurrent architectures." Can you help me find where this came from?

**Claude:** I'll search your indexed PDFs for this quote using SourceSleuth.

*[Claude uses the `find_orphaned_quote` tool]*

**Claude:** I found a potential match! This quote appears to come from:

- **File:** `vaswani2017_attention.pdf`
- **Page:** 3
- **Confidence:** High (0.89)
- **Context:** "We propose the Transformer, a model architecture based entirely on attention mechanisms. The attention mechanism allows models to focus on specific parts of the input sequence, enabling them to capture long-range dependencies more effectively than recurrent architectures without the constraints of sequential processing."

Would you like me to format this as a citation?

---

## 2. CLI Usage for PDF Ingestion

### Basic PDF Ingestion

```bash
# Ingest all PDFs from a directory
python -m src.ingest pdfs --directory ./student_pdfs

# Output:
# Processing PDFs from: ./student_pdfs
# Found 15 PDF files
# Extracted 1,247 chunks from 15 documents
# Vector store saved to ./data/sourcesleuth.index
```

### View Index Statistics

```bash
python -m src.ingest stats

# Output:
# Vector Store Statistics
# =======================
# Total chunks: 1,247
# Indexed files: 15
# Embedding dimension: 384
# Index type: IndexFlatIP
```

### Clear and Rebuild Index

```bash
# Clear the existing index
python -m src.ingest clear

# Output:
# Vector store cleared successfully.
```

---

## 3. Web UI Search Workflow

### Launch the Web UI

```bash
streamlit run app.py
```

### Example Search Session

1. **Upload PDFs:**
   - Navigate to the sidebar
   - Drag and drop your PDF files
   - Click "Process Uploaded PDFs"

2. **Search for a Quote:**
   - Paste your orphaned quote in the search box
   - Click "Find Sources"
   - Review ranked results with confidence scores

3. **Interpret Results:**
   - **High confidence (≥0.75):** Likely the exact source
   - **Medium confidence (0.50-0.75):** Possible paraphrase
   - **Low confidence (<0.50):** Weak match, verify manually

### Sample Search Query

**Input:**
```
The photoelectric effect demonstrates that light behaves as discrete packets of energy called photons, rather than purely as waves.
```

**Expected Output:**
```
#1 — modern_physics_textbook.pdf [HIGH - 0.87]
Page 142, Chunk #23
"The photoelectric effect demonstrates that light behaves as discrete packets 
of energy called photons, rather than purely as waves. This groundbreaking 
discovery by Einstein in 1905 established the quantum nature of light..."

#2 — quantum_mechanics_intro.pdf [MEDIUM - 0.62]
Page 8, Chunk #5
"Einstein's explanation of the photoelectric effect showed that light energy 
is quantized into discrete packets, later named photons..."
```

---

## 4. arXiv Paper Ingestion

### Ingest Papers by Category

```bash
# Ingest machine learning papers
python -m src.ingest arxiv --category cs.LG --max-records 1000

# Ingest AI papers
python -m src.ingest arxiv --category cs.AI --max-records 500

# Ingest multiple categories
python -m src.ingest arxiv --category cs.CL --max-records 2000
```

### Example Output

```
Fetching arXiv papers for category: cs.LG
Retrieved 1,000 papers
Preprocessing titles and abstracts...
Extracted 8,542 chunks from 1,000 arXiv papers
Vector store updated successfully.
```

### Use Case: Building a Domain-Specific Corpus

```bash
# Step 1: Clear existing index
python -m src.ingest clear

# Step 2: Ingest your local PDFs
python -m src.ingest pdfs --directory ./my_thesis_references

# Step 3: Supplement with arXiv papers
python -m src.ingest arxiv --category cs.IR --max-records 3000
python -m src.ingest arxiv --category cs.DL --max-records 2000

# Step 4: Verify the index
python -m src.ingest stats
```

---

## 5. Custom Embedding Model

### Using a Different Sentence Transformer

By default, SourceSleuth uses `all-MiniLM-L6-v2`. You can configure alternative models:

```bash
# Set environment variable before running
export SOURCESLEUTH_EMBEDDING_MODEL=all-mpnet-base-v2
python -m src.ingest pdfs --directory ./student_pdfs
```

### Recommended Models

| Model | Dimensions | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default, CPU-only |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher accuracy |
| `all-large-v2` | 1024 | Slow | Best | Maximum precision |
| `paraphrase-multilingual` | 768 | Medium | Good | Multi-language |

### Performance Comparison

```bash
# Benchmark with default model
time python -m src.ingest pdfs --directory ./test_pdfs
# ~30 seconds for 10 PDFs

# Benchmark with larger model
export SOURCESLEUTH_EMBEDDING_MODEL=all-mpnet-base-v2
time python -m src.ingest pdfs --directory ./test_pdfs
# ~90 seconds for 10 PDFs
```

---

## 6. Citation Formatting

### Using the `cite_recovered_source` Prompt

When connected via MCP, ask your AI assistant:

**You:** Format this source as an APA citation:
- Author: Vaswani et al.
- Title: Attention Is All You Need
- Year: 2017
- Venue: NeurIPS

**Claude:** Here's the citation in APA format:

```
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., 
Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is 
All You Need. In Advances in Neural Information Processing Systems 
(NeurIPS).
```

### Supported Citation Styles

- **APA 7th Edition** (default)
- **MLA 9th Edition**
- **Chicago 17th Edition**
- **IEEE**
- **Vancouver**

---

## Troubleshooting

### Issue: No Results Found

**Solutions:**
1. Ensure PDFs are ingested: `python -m src.ingest stats`
2. Lower the minimum score threshold in settings
3. Try a longer query (5+ words)
4. Check if the PDF text is extractable (not scanned images)

### Issue: Slow Search Performance

**Solutions:**
1. Reduce the number of results (top_k)
2. Use a smaller embedding model
3. For large corpora (>10k chunks), consider approximate search (planned for v2.0)

### Issue: Poor Match Quality

**Solutions:**
1. Use more specific queries with unique phrases
2. Try alternative embedding models
3. Check chunk size configuration (default: 500 tokens)

---

## Contributing Examples

Have a creative use case? Submit it as a pull request to add to this directory!

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
