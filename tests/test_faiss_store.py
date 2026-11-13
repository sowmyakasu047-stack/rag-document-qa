"""Tests for the FAISS-backed store; skipped automatically when faiss-cpu is not installed."""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from rag.faiss_store import FaissVectorStore  # noqa: E402
from rag.models import Chunk  # noqa: E402
from rag.vectorstore import VectorStore  # noqa: E402


def _chunk(chunk_id: str, source: str = "a.pdf") -> Chunk:
    return Chunk(chunk_id=chunk_id, text=chunk_id, source=source, page=1, index=0)


VECTORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
CHUNKS = [_chunk("x"), _chunk("y"), _chunk("z"), _chunk("xy", "b.pdf")]


def _fill(store):
    store.add(CHUNKS, VECTORS)
    return store


def test_faiss_results_match_the_numpy_store():
    """TC-69: FAISS and NumPy stores return the same chunks and scores for the same query."""
    query = [0.3, 0.9, 0.1]

    faiss_results = _fill(FaissVectorStore(3)).search(query, k=4)
    numpy_results = _fill(VectorStore(3)).search(query, k=4)

    assert [r.chunk for r in faiss_results] == [r.chunk for r in numpy_results]
    assert [r.score for r in faiss_results] == pytest.approx([r.score for r in numpy_results], abs=1e-5)


def test_faiss_store_remove_source_and_k_larger_than_size():
    """TC-70: remove_source works on a FAISS store and k beyond the store size is accepted."""
    store = _fill(FaissVectorStore(3))

    assert store.remove_source("b.pdf") == 1
    assert len(store.search([1, 0, 0], k=50)) == 3


def test_faiss_store_save_and_load_round_trip(tmp_path):
    """TC-71: A saved FAISS store reloads with identical chunks and search results."""
    store = _fill(FaissVectorStore(3, "test"))
    store.save(tmp_path)

    loaded = FaissVectorStore.load(tmp_path)

    assert loaded.embedder_name == "test"
    assert loaded.search([0, 1, 0], k=4) == store.search([0, 1, 0], k=4)
