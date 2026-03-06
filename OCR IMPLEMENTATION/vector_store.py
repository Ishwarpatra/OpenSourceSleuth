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

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
INDEX_FILENAME = "sourcesleuth.index"
METADATA_FILENAME = "sourcesleuth_metadata.json"


class EmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s' …", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully.")
        return self._model

    def encode(self, texts: list[str], normalize: bool = True, show_progress: bool = False) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            batch_size=64,
        )
        return np.asarray(embeddings, dtype=np.float32)


class VectorIndex:
    def __init__(self, dimension: int = EMBEDDING_DIM):
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.dimension = dimension

    def add(self, embeddings: np.ndarray) -> None:
        self._index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        top_k = min(top_k, self._index.ntotal)
        return self._index.search(query_embedding, top_k)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def save(self, path: Path) -> None:
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        self._index = faiss.read_index(str(path))

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(self.dimension)


class MetadataStore:
    def __init__(self):
        self._metadata: list[dict] = []
        self._ingested_files: set[str] = set()

    def add(self, chunk: TextChunk) -> None:
        self._metadata.append(chunk.to_dict())
        self._ingested_files.add(chunk.filename)

    def add_batch(self, chunks: list[TextChunk]) -> None:
        for chunk in chunks:
            self.add(chunk)

    def get_all(self) -> list[dict]:
        return self._metadata

    def set_all(self, metadata: list[dict]) -> None:
        self._metadata = metadata

    def filter_by_filename(self, filename: str) -> list[int]:
        return [
            i for i, m in enumerate(self._metadata)
            if m["filename"] != filename
        ]

    def remove_by_filename(self, filename: str) -> int:
        keep_indices = self.filter_by_filename(filename)
        removed_count = len(self._metadata) - len(keep_indices)

        if not keep_indices:
            self.clear()
        else:
            self._metadata = [self._metadata[i] for i in keep_indices]

        self._ingested_files.discard(filename)
        return removed_count

    def save(self, path: Path, model_name: str) -> None:
        payload = {
            "model_name": model_name,
            "ingested_files": sorted(self._ingested_files),
            "chunks": self._metadata,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False

        payload = json.loads(path.read_text(encoding="utf-8"))
        self._metadata = payload.get("chunks", [])
        self._ingested_files = set(payload.get("ingested_files", []))
        return True

    def clear(self) -> None:
        self._metadata.clear()
        self._ingested_files.clear()

    @property
    def ingested_files(self) -> set[str]:
        return self._ingested_files.copy()

    def get_texts_by_indices(self, indices: list[int]) -> list[str]:
        return [self._metadata[i]["text"] for i in indices]

    def get_metadata_by_indices(self, indices: list[int]) -> list[dict]:
        return [self._metadata[i] for i in indices]


class VectorStore:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        data_dir: str | Path = "data",
    ):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = EmbeddingModel(model_name)
        self.index = VectorIndex(EMBEDDING_DIM)
        self.metadata_store = MetadataStore()

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0

        texts = [c.text for c in chunks]

        logger.info("Encoding %d chunks …", len(texts))
        embeddings = self.embedding_model.encode(
            texts, normalize=True, show_progress=True
        )

        self.index.add(embeddings)
        self.metadata_store.add_batch(chunks)

        logger.info(
            "Added %d chunks to the vector store (total: %d).",
            len(chunks), self.index.total_vectors,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index.total_vectors == 0:
            logger.warning("Vector store is empty — no results to return.")
            return []

        query_embedding = self.embedding_model.encode([query], normalize=True)

        scores, indices = self.index.search(query_embedding, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata_store.get_all()[idx].copy()
            meta["score"] = round(float(score), 4)
            results.append(meta)

        return results

    def save(self) -> None:
        index_path = self.data_dir / INDEX_FILENAME
        meta_path = self.data_dir / METADATA_FILENAME

        self.index.save(index_path)
        self.metadata_store.save(meta_path, self.model_name)

        logger.info(
            "Saved vector store: %d vectors -> '%s'.",
            self.index.total_vectors, self.data_dir,
        )

    def load(self) -> bool:
        index_path = self.data_dir / INDEX_FILENAME
        meta_path = self.data_dir / METADATA_FILENAME

        if not index_path.exists() or not meta_path.exists():
            logger.info("No saved vector store found at '%s'.", self.data_dir)
            return False

        self.index.load(index_path)
        self.metadata_store.load(meta_path)

        logger.info(
            "Loaded vector store: %d vectors from '%s'.",
            self.index.total_vectors, self.data_dir,
        )
        return True

    @property
    def total_chunks(self) -> int:
        return self.index.total_vectors

    @property
    def ingested_files(self) -> set[str]:
        return self.metadata_store.ingested_files

    def clear(self) -> None:
        self.index.clear()
        self.metadata_store.clear()
        logger.info("Vector store cleared.")

    def remove_file(self, filename: str) -> int:
        if filename not in self.ingested_files:
            return 0

        removed_count = self.metadata_store.remove_by_filename(filename)

        if removed_count == 0:
            return 0

        if self.index.total_vectors == 0:
            self.clear()
            return removed_count

        keep_indices = self.metadata_store.filter_by_filename(filename)
        remaining_texts = self.metadata_store.get_texts_by_indices(keep_indices)

        embeddings = self.embedding_model.encode(remaining_texts, normalize=True)

        self.index.clear()
        self.index.add(embeddings)

        logger.info("Removed %d chunks for '%s'.", removed_count, filename)
        return removed_count

    def get_stats(self) -> dict:
        return {
            "total_chunks": self.index.total_vectors,
            "ingested_files": sorted(self.ingested_files),
            "num_files": len(self.ingested_files),
            "model_name": self.model_name,
            "embedding_dim": EMBEDDING_DIM,
            "index_type": "IndexFlatIP (cosine similarity)",
        }
