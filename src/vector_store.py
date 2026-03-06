"""
Vector Store Module for SourceSleuth.

Manages embeddings using FAISS (Facebook AI Similarity Search) as the
local vector database. All data stays on the student's machine — no
network calls, no cloud dependencies.

Model Architecture Documentation (Hackathon AI/ML Requirement):
    - Model: ``all-MiniLM-L6-v2`` from Sentence-Transformers
    - Embedding dimension: 384
    - Why this model?
        1. Runs efficiently on CPU (no GPU required).
        2. Produces high-quality sentence embeddings for semantic similarity.
        3. Small footprint (~80 MB) — ideal for a student laptop.
        4. Trained on 1B+ sentence pairs, strong zero-shot performance.
    - Index type: FAISS IndexFlatIP (inner product on L2-normalized vectors,
      equivalent to cosine similarity). Flat index chosen for simplicity and
      exact results on small-to-medium corpora (< 100k chunks).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.pdf_processor import TextChunk

logger = logging.getLogger("sourcesleuth.vector_store")

# Configuration

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2

INDEX_FILENAME = "sourcesleuth.index"
METADATA_FILENAME = "sourcesleuth_metadata.json"


# Vector Store Class

class VectorStore:
    """
    FAISS-backed vector store for PDF chunk embeddings.

    Supports adding chunks, querying by text similarity, and persisting
    the index + metadata to disk for fast reloads.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        data_dir: str | Path = "data",
    ) -> None:
        """
        Initialize the vector store.

        Args:
            model_name: HuggingFace model identifier for the embedding model.
            data_dir: Directory where the FAISS index and metadata are persisted.
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-load the model to avoid slow startup when not needed
        self._model: Optional[SentenceTransformer] = None

        # FAISS index — inner-product on L2-normalized vectors = cosine sim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)

        # Parallel metadata list (same order as vectors in the index)
        self._metadata: list[dict] = []

        # Track ingested filenames to support re-ingestion
        self._ingested_files: set[str] = set()

    # Model

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer model on first access."""
        if self._model is None:
            logger.info("Loading embedding model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")
        return self._model

    # Core API

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """
        Embed and add text chunks to the vector store.

        Args:
            chunks: List of TextChunk objects to embed and index.

        Returns:
            Number of new chunks added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]

        logger.info("Encoding %d chunks …", len(texts))
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
            batch_size=64,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Add to FAISS index
        self._index.add(embeddings)

        # Store metadata in parallel
        for chunk in chunks:
            self._metadata.append(chunk.to_dict())
            self._ingested_files.add(chunk.filename)

        logger.info(
            "Added %d chunks to the vector store (total: %d).",
            len(chunks), self._index.ntotal,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Perform semantic search for a query text.

        Args:
            query: The text to search for (e.g., an orphaned quote).
            top_k: Number of top results to return.

        Returns:
            List of result dicts, each containing 'score' and chunk metadata.
        """
        if self._index.ntotal == 0:
            logger.warning("Vector store is empty — no results to return.")
            return []

        # Clamp top_k to the number of available vectors
        top_k = min(top_k, self._index.ntotal)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        scores, indices = self._index.search(query_embedding, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for missing entries
            meta = self._metadata[idx].copy()
            meta["score"] = round(float(score), 4)
            results.append(meta)

        return results

    # Persistence

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path = self.data_dir / INDEX_FILENAME
        meta_path = self.data_dir / METADATA_FILENAME

        faiss.write_index(self._index, str(index_path))

        payload = {
            "model_name": self.model_name,
            "ingested_files": sorted(self._ingested_files),
            "chunks": self._metadata,
        }
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info(
            "Saved vector store: %d vectors -> '%s'.",
            self._index.ntotal, self.data_dir,
        )

    def load(self) -> bool:
        """
        Load a previously persisted vector store from disk.

        Returns:
            True if loaded successfully, False if no saved data was found.
        """
        index_path = self.data_dir / INDEX_FILENAME
        meta_path = self.data_dir / METADATA_FILENAME

        if not index_path.exists() or not meta_path.exists():
            logger.info("No saved vector store found at '%s'.", self.data_dir)
            return False

        self._index = faiss.read_index(str(index_path))

        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        self._metadata = payload.get("chunks", [])
        self._ingested_files = set(payload.get("ingested_files", []))

        logger.info(
            "Loaded vector store: %d vectors from '%s'.",
            self._index.ntotal, self.data_dir,
        )
        return True

    # Utilities

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently in the store."""
        return self._index.ntotal

    @property
    def ingested_files(self) -> set[str]:
        """Set of filenames that have been ingested."""
        return self._ingested_files.copy()

    def clear(self) -> None:
        """Remove all vectors and metadata."""
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._metadata.clear()
        self._ingested_files.clear()
        logger.info("Vector store cleared.")

    def remove_file(self, filename: str) -> int:
        """
        Remove all chunks belonging to a specific file and rebuild the index.

        Args:
            filename: The filename to remove.

        Returns:
            Number of chunks removed.
        """
        if filename not in self._ingested_files:
            return 0

        # Filter out chunks belonging to this file
        keep_indices = [
            i for i, m in enumerate(self._metadata)
            if m["filename"] != filename
        ]
        removed_count = len(self._metadata) - len(keep_indices)

        if not keep_indices:
            self.clear()
            return removed_count

        # Rebuild index with remaining vectors
        remaining_texts = [self._metadata[i]["text"] for i in keep_indices]
        remaining_meta = [self._metadata[i] for i in keep_indices]

        # Re-encode remaining chunks
        embeddings = self.model.encode(
            remaining_texts,
            normalize_embeddings=True,
            batch_size=64,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._index.add(embeddings)
        self._metadata = remaining_meta
        self._ingested_files.discard(filename)

        logger.info("Removed %d chunks for '%s'.", removed_count, filename)
        return removed_count

    def get_stats(self) -> dict:
        """Return summary statistics about the vector store."""
        return {
            "total_chunks": self._index.ntotal,
            "ingested_files": sorted(self._ingested_files),
            "num_files": len(self._ingested_files),
            "model_name": self.model_name,
            "embedding_dim": EMBEDDING_DIM,
            "index_type": "IndexFlatIP (cosine similarity)",
        }
