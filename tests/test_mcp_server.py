"""Tests for the MCP Server tool functions."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.pdf_processor import TextChunk


# Tests: Tool Functions (unit-test the logic directly)

class TestFindOrphanedQuote:
    """Test the find_orphaned_quote tool logic."""

    def test_empty_store_returns_warning(self):
        """When no PDFs are ingested, should return a helpful message."""
        from src.mcp_server import find_orphaned_quote, store

        # Ensure store is empty
        store.clear()
        result = find_orphaned_quote("some random quote")
        assert "No PDFs have been ingested" in result

    def test_search_with_data(self):
        """When data is present, should return formatted results."""
        from src.mcp_server import find_orphaned_quote, store

        store.clear()
        chunks = [
            TextChunk(
                text="Machine learning is a subset of artificial intelligence.",
                filename="intro_to_ml.pdf",
                page=1,
                chunk_index=0,
                start_char=0,
                end_char=55,
            ),
        ]
        store.add_chunks(chunks)

        result = find_orphaned_quote("artificial intelligence and machine learning")
        assert "intro_to_ml.pdf" in result
        assert "Confidence" in result

        # Cleanup
        store.clear()


class TestGetStoreStats:
    """Test the get_store_stats tool."""

    def test_empty_stats(self):
        from src.mcp_server import get_store_stats, store

        store.clear()
        result = get_store_stats()
        assert "Empty" in result

    def test_stats_with_data(self):
        from src.mcp_server import get_store_stats, store

        store.clear()
        chunks = [
            TextChunk(
                text="Test chunk content.",
                filename="test.pdf",
                page=1,
                chunk_index=0,
                start_char=0,
                end_char=19,
            ),
        ]
        store.add_chunks(chunks)
        result = get_store_stats()
        assert "test.pdf" in result
        assert "all-MiniLM-L6-v2" in result

        store.clear()


class TestCiteRecoveredSource:
    """Test the cite_recovered_source prompt."""

    def test_prompt_output(self):
        from src.mcp_server import cite_recovered_source

        result = cite_recovered_source(
            quote="Deep learning has transformed NLP.",
            source_filename="smith2023_nlp.pdf",
            page_number=5,
            citation_style="APA",
        )
        assert "smith2023_nlp.pdf" in result
        assert "APA" in result
        assert "page 5" in result
        assert "Full Citation" in result
