"""Tests for chunking (rag.chunker)."""

from __future__ import annotations

import pytest

from rag.chunker import chunk_pages
from rag.models import PageText


def _sentences(count: int) -> str:
    return " ".join(f"Sentence {n} talks about topic {n}." for n in range(count))


def test_short_page_becomes_a_single_chunk():
    """TC-16: A page shorter than chunk_size produces exactly one chunk with the full text."""
    pages = [PageText("doc.pdf", 1, "A short page.")]

    chunks = chunk_pages(pages, chunk_size=500, overlap=50)

    assert [c.text for c in chunks] == ["A short page."]


def test_no_chunk_exceeds_chunk_size():
    """TC-17: Every chunk is at most chunk_size characters long."""
    pages = [PageText("doc.pdf", 1, _sentences(60))]

    chunks = chunk_pages(pages, chunk_size=200, overlap=60)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_consecutive_chunks_overlap():
    """TC-18: The start of each chunk repeats the end of the previous one."""
    pages = [PageText("doc.pdf", 1, _sentences(60))]

    chunks = chunk_pages(pages, chunk_size=200, overlap=60)

    for previous, following in zip(chunks, chunks[1:]):
        first_sentence = following.text.split(". ")[0]
        assert first_sentence in previous.text


def test_chunks_keep_source_and_page_and_never_cross_pages():
    """TC-19: Chunks carry the right source/page, and text from two pages is never mixed."""
    pages = [
        PageText("doc.pdf", 2, "ALPHA " * 100),
        PageText("doc.pdf", 5, "OMEGA " * 100),
    ]

    chunks = chunk_pages(pages, chunk_size=150, overlap=20)

    for chunk in chunks:
        assert chunk.source == "doc.pdf"
        assert ("ALPHA" in chunk.text) != ("OMEGA" in chunk.text)
        assert chunk.page == (2 if "ALPHA" in chunk.text else 5)


def test_chunk_ids_are_unique_and_index_is_sequential():
    """TC-20: chunk_id values are unique and index counts 0, 1, 2... across the document."""
    pages = [PageText("report.pdf", 1, _sentences(40)), PageText("report.pdf", 2, _sentences(40))]

    chunks = chunk_pages(pages, chunk_size=200, overlap=40)

    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "report-p1-c0"


def test_no_text_is_lost():
    """TC-21: Every word of the original page appears in at least one chunk."""
    text = _sentences(60)
    chunks = chunk_pages([PageText("doc.pdf", 1, text)], chunk_size=200, overlap=60)

    covered = set(" ".join(c.text for c in chunks).split())

    assert set(text.split()) <= covered


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-10, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_parameters_raise_value_error(chunk_size, overlap):
    """TC-22: Non-positive chunk_size, negative overlap, or overlap >= chunk_size is rejected."""
    with pytest.raises(ValueError):
        chunk_pages([PageText("doc.pdf", 1, "text")], chunk_size=chunk_size, overlap=overlap)


def test_word_longer_than_chunk_size_is_hard_split():
    """TC-23: A single unbroken string longer than chunk_size is cut so no chunk is oversized."""
    chunks = chunk_pages([PageText("doc.pdf", 1, "x" * 50)], chunk_size=20, overlap=5)

    assert all(len(c.text) <= 20 for c in chunks)
    assert sum(len(c.text) for c in chunks) == 50


def test_no_pages_gives_no_chunks():
    """TC-24: An empty list of pages produces an empty list of chunks."""
    assert chunk_pages([]) == []
