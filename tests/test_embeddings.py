"""Tests for embedders (rag.embeddings)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from rag.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from rag.errors import EmbeddingModelUnavailableError


def test_hashing_embedder_returns_unit_vectors_of_right_shape():
    """TC-31: embed() returns one float32 row per text, each scaled to length 1."""
    vectors = HashingEmbedder(dimension=64).embed(["apples and oranges", "vector databases"])

    assert vectors.shape == (2, 64)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_hashing_embedder_is_deterministic():
    """TC-32: The same text always gives the same vector (no randomness, no per-run salt)."""
    embedder = HashingEmbedder()

    assert np.array_equal(embedder.embed(["chunk overlap"]), embedder.embed(["chunk overlap"]))


def test_similar_texts_score_higher_than_unrelated_texts():
    """TC-33: Texts sharing words are closer than texts sharing none."""
    embedder = HashingEmbedder()
    query, related, unrelated = embedder.embed(
        [
            "how are chunks of text embedded",
            "each chunk of text is embedded as a vector",
            "the stadium hosted a football match on Sunday",
        ]
    )

    assert float(query @ related) > float(query @ unrelated)


def test_case_and_punctuation_do_not_change_the_vector():
    """TC-34: 'Hello, World!' and 'hello world' embed identically."""
    embedder = HashingEmbedder()

    assert np.array_equal(embedder.embed(["Hello, World!"]), embedder.embed(["hello world"]))


def test_text_with_no_content_words_gives_zero_vector():
    """TC-35: Text made only of ignored words or symbols yields an all-zero vector, not NaN."""
    vector = HashingEmbedder(dimension=32).embed(["the of and !!!"])[0]

    assert not vector.any()
    assert np.isfinite(vector).all()


def test_empty_input_gives_empty_matrix():
    """TC-36: Embedding an empty list returns shape (0, dimension)."""
    assert HashingEmbedder(dimension=16).embed([]).shape == (0, 16)


def test_hashing_embedder_rejects_invalid_dimension():
    """TC-37: A dimension of zero or less raises ValueError."""
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=0)


class _FakeModel:
    """Stands in for a SentenceTransformer so the wrapper can be tested without a download."""

    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size, show_progress_bar):
        self.calls.append((list(texts), batch_size))
        return np.array([[3.0, 4.0, 0.0]] * len(texts))


def test_sentence_transformer_wrapper_normalises_and_batches():
    """TC-38: The neural wrapper passes texts and batch size through and returns unit vectors."""
    model = _FakeModel()
    embedder = SentenceTransformerEmbedder("fake-model", batch_size=8, model=model)

    vectors = embedder.embed(["a", "b"])

    assert embedder.dimension == 3
    assert embedder.name == "fake-model"
    assert model.calls == [(["a", "b"], 8)]
    assert np.allclose(vectors, [[0.6, 0.8, 0.0]] * 2)


def test_sentence_transformer_missing_library_gives_clear_error(monkeypatch):
    """TC-39: Without sentence-transformers installed, a readable error explains how to fix it."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(EmbeddingModelUnavailableError, match="pip install"):
        SentenceTransformerEmbedder("any-model")
