"""Turning text into vectors.

An *embedding* is a list of numbers that represents the meaning of a text, so
that texts with similar meaning end up close together. Retrieval works by
embedding the question and finding the chunks whose vectors are closest.

Two implementations share one small interface (``name``, ``dimension``,
``embed``):

* :class:`SentenceTransformerEmbedder` uses a real neural model and captures
  *meaning* (a question about "automobiles" can match a chunk about "cars").
  This is the one to use for the real application.
* :class:`HashingEmbedder` needs no download and no model. It only matches
  *shared words*, so it is a lexical baseline. It exists so the pipeline and
  the tests run anywhere, offline, with deterministic results.

Every embedder returns float32 vectors scaled to length 1, so the dot product
of two vectors equals their cosine similarity.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Protocol, Sequence

import numpy as np

from rag.errors import EmbeddingModelUnavailableError

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "what when where which who why how with does do did".split()
)


class Embedder(Protocol):
    """What the rest of the code needs from an embedding model."""

    name: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a ``(len(texts), dimension)`` float32 array of unit-length rows."""
        ...


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1; all-zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class HashingEmbedder:
    """Bag-of-words embedder using the hashing trick (lexical, not semantic).

    Each word is hashed to one of ``dimension`` slots with a +1 or -1 sign;
    word counts are damped with ``1 + log(count)`` so repeating a word does not
    dominate. Common function words are ignored. Hashing uses CRC32 so results
    are identical across runs and machines (Python's built-in ``hash`` is not).
    """

    def __init__(self, dimension: int = 512) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0.")
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in _TOKEN.findall(text.lower()):
                if token not in _STOPWORDS:
                    counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                digest = zlib.crc32(token.encode("utf-8"))
                slot = digest % self.dimension
                sign = 1.0 if (digest >> 16) & 1 else -1.0
                matrix[row, slot] += sign * (1.0 + math.log(count))
        return _unit_rows(matrix)


class SentenceTransformerEmbedder:
    """Neural embedder backed by the ``sentence-transformers`` library.

    The model is downloaded from the Hugging Face hub the first time it is used
    (about 90 MB for the default), then cached. ``model`` can be passed to reuse
    an already-loaded model object, which is also how tests substitute a fake.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        model: object | None = None,
    ) -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingModelUnavailableError(
                    "The 'sentence-transformers' package is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            model = SentenceTransformer(model_name)
        self._model = model
        self._batch_size = batch_size
        self.name = model_name
        self.dimension = int(model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, show_progress_bar=False
        )
        return _unit_rows(np.asarray(vectors, dtype=np.float32))
