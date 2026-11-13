"""FAISS-backed vector store with the same methods as :class:`rag.vectorstore.VectorStore`.

Requires ``pip install faiss-cpu``. It uses ``IndexFlatIP`` (exact inner-product
search) on unit-length vectors, so scores are cosine similarities and match the
NumPy store. Swap it in by passing ``store_cls=FaissVectorStore`` to the
retriever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from rag.errors import EmbeddingModelUnavailableError, VectorStoreError
from rag.models import Chunk
from rag.vectorstore import (
    SearchResult,
    as_matrix,
    check_new_chunks,
    normalize_rows,
    read_metadata,
    write_metadata,
)

_INDEX_FILE = "index.faiss"


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise EmbeddingModelUnavailableError(
            "The 'faiss-cpu' package is not installed. Run: pip install faiss-cpu"
        ) from exc
    return faiss


class FaissVectorStore:
    """Exact cosine-similarity store on top of a FAISS flat inner-product index."""

    def __init__(self, dimension: int, embedder_name: str = "") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0.")
        self._faiss = _import_faiss()
        self.dimension = dimension
        self.embedder_name = embedder_name
        self._index = self._faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: object) -> None:
        chunks = list(chunks)
        if not chunks and len(vectors) == 0:  # type: ignore[arg-type]
            return
        matrix = as_matrix(vectors, self.dimension)
        if len(chunks) != len(matrix):
            raise ValueError(f"Got {len(chunks)} chunks but {len(matrix)} vectors.")
        check_new_chunks(chunks, self._chunks)
        self._index.add(np.ascontiguousarray(normalize_rows(matrix)))
        self._chunks.extend(chunks)

    def search(
        self, query_vector: object, k: int = 5, min_score: float | None = None
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        query = normalize_rows(as_matrix(np.atleast_2d(query_vector), self.dimension))
        if not self._chunks or not query.any():
            return []
        scores, positions = self._index.search(np.ascontiguousarray(query), min(k, len(self)))
        results = [
            SearchResult(self._chunks[int(pos)], float(score))
            for score, pos in zip(scores[0], positions[0])
            if pos >= 0
        ]
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

    def remove_source(self, source: str) -> int:
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.source != source]
        removed = len(self._chunks) - len(keep)
        if removed:
            everything = self._index.reconstruct_n(0, self._index.ntotal)
            self._index = self._faiss.IndexFlatIP(self.dimension)
            if keep:
                self._index.add(np.ascontiguousarray(everything[keep]))
            self._chunks = [self._chunks[i] for i in keep]
        return removed

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / _INDEX_FILE))
        write_metadata(path, self.dimension, self.embedder_name, self._chunks)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        path = Path(directory)
        dimension, embedder_name, chunks = read_metadata(path)
        store = cls(dimension, embedder_name)
        try:
            index = store._faiss.read_index(str(path / _INDEX_FILE))
        except (RuntimeError, OSError) as exc:
            raise VectorStoreError(f"Could not read FAISS index in '{path}': {exc}") from exc
        if index.ntotal != len(chunks) or index.d != dimension:
            raise VectorStoreError(
                f"Saved index in '{path}' is inconsistent: {len(chunks)} chunks but "
                f"{index.ntotal} vectors of dimension {index.d}."
            )
        store._index = index
        store._chunks = chunks
        return store
