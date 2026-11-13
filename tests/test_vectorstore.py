"""Tests for the vector store (rag.vectorstore)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rag.errors import VectorStoreError
from rag.models import Chunk
from rag.vectorstore import VectorStore


def _chunk(chunk_id: str, source: str = "a.pdf", page: int = 1) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=f"text of {chunk_id}", source=source, page=page, index=0)


@pytest.fixture
def store() -> VectorStore:
    """Three chunks pointing along the x, y and z axes."""
    s = VectorStore(dimension=3, embedder_name="test")
    s.add([_chunk("x"), _chunk("y"), _chunk("z")], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return s


def test_search_returns_closest_chunk_first(store):
    """TC-40: A query near the y axis returns chunk y first with a score near 1."""
    results = store.search([0.1, 0.9, 0.0], k=3)

    assert [r.chunk.chunk_id for r in results][0] == "y"
    assert results[0].score > results[1].score >= results[2].score
    assert results[0].score == pytest.approx(0.9939, abs=1e-3)


def test_k_limits_results_and_larger_k_returns_everything(store):
    """TC-41: k=1 gives one result; k greater than the store size gives all stored chunks."""
    assert len(store.search([1, 0, 0], k=1)) == 1
    assert len(store.search([1, 0, 0], k=50)) == 3


def test_min_score_filters_weak_matches(store):
    """TC-42: Results below min_score are dropped, possibly leaving none."""
    assert [r.chunk.chunk_id for r in store.search([1, 0, 0], k=3, min_score=0.5)] == ["x"]
    assert store.search([1, 1, 1], k=3, min_score=0.9) == []


def test_scale_of_vectors_does_not_matter(store):
    """TC-43: Vectors are compared by direction (cosine), so [10, 0, 0] matches [1, 0, 0] fully."""
    assert store.search([10, 0, 0], k=1)[0].score == pytest.approx(1.0)


def test_empty_store_and_zero_query_return_no_results(store):
    """TC-44: Searching an empty store, or with an all-zero query, returns an empty list."""
    assert VectorStore(3).search([1, 0, 0]) == []
    assert store.search([0, 0, 0]) == []


def test_invalid_k_raises_value_error(store):
    """TC-45: k of zero or less is rejected."""
    with pytest.raises(ValueError):
        store.search([1, 0, 0], k=0)


def test_wrong_vector_dimension_is_rejected(store):
    """TC-46: Vectors of the wrong length are rejected on both add and search."""
    with pytest.raises(ValueError):
        store.add([_chunk("w")], [[1, 0]])
    with pytest.raises(ValueError):
        store.search([1, 0])


def test_mismatched_chunk_and_vector_counts_are_rejected(store):
    """TC-47: Adding two chunks with one vector raises ValueError and stores nothing."""
    with pytest.raises(ValueError):
        store.add([_chunk("p"), _chunk("q")], [[1, 0, 0]])

    assert len(store) == 3


def test_non_finite_vectors_are_rejected(store):
    """TC-48: NaN or infinite values raise ValueError."""
    with pytest.raises(ValueError):
        store.add([_chunk("n")], [[float("nan"), 0, 0]])


def test_duplicate_chunk_ids_are_rejected(store):
    """TC-49: Re-adding an existing chunk id raises ValueError and leaves the store unchanged."""
    with pytest.raises(ValueError, match="Duplicate chunk id 'x'"):
        store.add([_chunk("x")], [[1, 0, 0]])

    assert len(store) == 3


def test_remove_source_deletes_only_that_documents_chunks():
    """TC-50: remove_source drops one file's chunks and keeps results correct for the rest."""
    s = VectorStore(2)
    s.add(
        [_chunk("a1", "a.pdf"), _chunk("b1", "b.pdf"), _chunk("a2", "a.pdf")],
        [[1, 0], [0, 1], [1, 1]],
    )

    assert s.remove_source("a.pdf") == 2
    assert s.remove_source("missing.pdf") == 0
    assert [c.chunk_id for c in s.chunks] == ["b1"]
    assert s.search([0, 1], k=5)[0].chunk.chunk_id == "b1"


def test_save_and_load_round_trip(store, tmp_path):
    """TC-51: A saved and reloaded store gives identical search results and metadata."""
    store.save(tmp_path / "index")

    loaded = VectorStore.load(tmp_path / "index")

    assert loaded.embedder_name == "test"
    assert loaded.chunks == store.chunks
    assert loaded.search([0.2, 0.8, 0], k=3) == store.search([0.2, 0.8, 0], k=3)


def test_loading_a_missing_index_raises_vector_store_error(tmp_path):
    """TC-52: Loading from a folder with no saved index raises VectorStoreError."""
    with pytest.raises(VectorStoreError):
        VectorStore.load(tmp_path / "nothing-here")


def test_loading_an_inconsistent_index_raises_vector_store_error(store, tmp_path):
    """TC-53: If chunks.json lists fewer chunks than there are vectors, loading fails clearly."""
    store.save(tmp_path)
    chunks = json.loads((tmp_path / "chunks.json").read_text())
    (tmp_path / "chunks.json").write_text(json.dumps(chunks[:2]))

    with pytest.raises(VectorStoreError, match="inconsistent"):
        VectorStore.load(tmp_path)


def test_loading_a_damaged_metadata_file_raises_vector_store_error(store, tmp_path):
    """TC-54: Unreadable JSON in the saved index raises VectorStoreError, not a raw crash."""
    store.save(tmp_path)
    (tmp_path / "meta.json").write_text("{not json")

    with pytest.raises(VectorStoreError, match="damaged"):
        VectorStore.load(tmp_path)


def test_saved_vectors_are_loaded_without_pickle(store, tmp_path):
    """TC-55: The vectors file is a plain numeric array that loads with pickle disabled."""
    store.save(tmp_path)

    assert np.load(tmp_path / "vectors.npy", allow_pickle=False).shape == (3, 3)
