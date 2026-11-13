"""Storing chunk vectors and finding the ones closest to a question.

:class:`VectorStore` does an exact search: it compares the query with every
stored vector and returns the best ``k``. This is the same computation as a
FAISS ``IndexFlatIP`` index and is fast enough for hundreds of thousands of
chunks, far beyond a handful of PDFs. Approximate indexes only pay off at much
larger scale. A FAISS-backed twin with the same methods lives in
``rag.faiss_store``.

Scores are cosine similarity: 1.0 means the same direction (very similar),
0 means unrelated, negative means opposed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rag.errors import VectorStoreError
from rag.models import Chunk

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk and how similar it is to the query (cosine, higher is closer)."""

    chunk: Chunk
    score: float


def as_matrix(vectors: object, dimension: int) -> np.ndarray:
    """Validate vectors and return them as a ``(n, dimension)`` float32 array."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(
            f"Expected vectors of dimension {dimension}, got array of shape {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Vectors must not contain NaN or infinite values.")
    return matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to length 1 so a dot product is a cosine similarity; zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def check_new_chunks(new: Sequence[Chunk], existing: Sequence[Chunk]) -> None:
    """Reject chunk ids that appear twice in ``new`` or already exist in the store."""
    seen = {chunk.chunk_id for chunk in existing}
    for chunk in new:
        if chunk.chunk_id in seen:
            raise ValueError(
                f"Duplicate chunk id '{chunk.chunk_id}'. To re-add a document, "
                "call remove_source() for it first."
            )
        seen.add(chunk.chunk_id)


def write_metadata(
    directory: Path, dimension: int, embedder_name: str, chunks: Sequence[Chunk]
) -> None:
    (directory / "meta.json").write_text(
        json.dumps(
            {"version": _FORMAT_VERSION, "dimension": dimension, "embedder": embedder_name}
        ),
        encoding="utf-8",
    )
    (directory / "chunks.json").write_text(
        json.dumps([asdict(chunk) for chunk in chunks]), encoding="utf-8"
    )


def read_metadata(directory: Path) -> tuple[int, str, list[Chunk]]:
    """Read ``(dimension, embedder_name, chunks)`` or raise VectorStoreError."""
    try:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        raw_chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if meta["version"] != _FORMAT_VERSION:
            raise VectorStoreError(f"Unsupported index format version: {meta['version']}.")
        chunks = [Chunk(**item) for item in raw_chunks]
        return int(meta["dimension"]), str(meta["embedder"]), chunks
    except FileNotFoundError as exc:
        raise VectorStoreError(f"No saved index found in '{directory}'.") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise VectorStoreError(f"Saved index in '{directory}' is damaged: {exc}") from exc


class VectorStore:
    """In-memory exact cosine-similarity store that can be saved to and loaded from disk."""

    def __init__(self, dimension: int, embedder_name: str = "") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0.")
        self.dimension = dimension
        self.embedder_name = embedder_name
        self._vectors = np.zeros((0, dimension), dtype=np.float32)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: object) -> None:
        """Store ``chunks`` with their vectors (row ``i`` belongs to ``chunks[i]``)."""
        chunks = list(chunks)
        if not chunks and len(vectors) == 0:  # type: ignore[arg-type]
            return
        matrix = as_matrix(vectors, self.dimension)
        if len(chunks) != len(matrix):
            raise ValueError(f"Got {len(chunks)} chunks but {len(matrix)} vectors.")
        check_new_chunks(chunks, self._chunks)
        self._vectors = np.vstack([self._vectors, normalize_rows(matrix)])
        self._chunks.extend(chunks)

    def search(
        self, query_vector: object, k: int = 5, min_score: float | None = None
    ) -> list[SearchResult]:
        """Return up to ``k`` most similar chunks, best first.

        ``min_score`` drops weak matches, which lets the caller say "I could not
        find this in the document" instead of answering from irrelevant text.
        """
        if k <= 0:
            raise ValueError("k must be greater than 0.")
        query = normalize_rows(as_matrix(np.atleast_2d(query_vector), self.dimension))[0]
        if not self._chunks or not query.any():
            return []
        scores = self._vectors @ query
        order = np.argsort(-scores, kind="stable")[:k]
        results = [SearchResult(self._chunks[i], float(scores[i])) for i in order]
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        return results

    def remove_source(self, source: str) -> int:
        """Delete every chunk that came from file ``source``; return how many were removed."""
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.source != source]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._vectors = self._vectors[keep]
            self._chunks = [self._chunks[i] for i in keep]
        return removed

    def save(self, directory: str | Path) -> None:
        """Write the index to ``directory`` (created if needed)."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        write_metadata(path, self.dimension, self.embedder_name, self._chunks)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Read an index written by :meth:`save`.

        Raises:
            VectorStoreError: files are missing, damaged, or disagree with each other.
        """
        path = Path(directory)
        dimension, embedder_name, chunks = read_metadata(path)
        try:
            vectors = np.load(path / "vectors.npy", allow_pickle=False)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise VectorStoreError(f"Could not read vectors in '{path}': {exc}") from exc
        if vectors.shape != (len(chunks), dimension):
            raise VectorStoreError(
                f"Saved index in '{path}' is inconsistent: {len(chunks)} chunks but "
                f"vectors of shape {vectors.shape}."
            )
        store = cls(dimension, embedder_name)
        store._vectors = vectors.astype(np.float32, copy=False)
        store._chunks = chunks
        return store
