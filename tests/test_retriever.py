"""Tests for the retrieval layer (rag.retriever) and its command line tool (rag.query)."""

from __future__ import annotations

import pytest

from rag.embeddings import HashingEmbedder
from rag.errors import EmbedderMismatchError, InvalidPDFError
from rag.query import main
from rag.retriever import Retriever

THREE_TOPIC_PAGES = [
    ["Chunking splits long documents into small overlapping passages", "so that each passage covers one topic."],
    ["Embeddings turn passages into vectors", "and vector databases find the nearest vectors quickly."],
    ["Football season begins in September.", "The stadium seats seventy thousand fans for each match."],
]


@pytest.fixture
def retriever(make_pdf) -> Retriever:
    r = Retriever(HashingEmbedder())
    r.add_pdf(make_pdf(THREE_TOPIC_PAGES, name="guide.pdf"))
    return r


def test_search_finds_the_page_that_answers_the_question(retriever):
    """TC-56: A question about stadiums retrieves the football page, with source and page."""
    best = retriever.search("How many fans does the stadium seat?", k=3)[0]

    assert (best.chunk.source, best.chunk.page) == ("guide.pdf", 3)


def test_different_questions_retrieve_different_pages(retriever):
    """TC-57: Questions about chunking and about vector databases hit pages 1 and 2."""
    chunking = retriever.search("what does chunking do to documents", k=1)[0]
    vectors = retriever.search("how do vector databases find nearest vectors", k=1)[0]

    assert (chunking.chunk.page, vectors.chunk.page) == (1, 2)


def test_add_pdf_returns_chunk_count_and_fills_store(make_pdf):
    """TC-58: add_pdf reports how many chunks were indexed and the store holds exactly those."""
    r = Retriever(HashingEmbedder())

    added = r.add_pdf(make_pdf(THREE_TOPIC_PAGES, name="guide.pdf"))

    assert added == 3
    assert len(r.store) == 3


def test_adding_the_same_file_again_replaces_instead_of_duplicating(retriever, make_pdf):
    """TC-59: Re-indexing 'guide.pdf' keeps one copy of its chunks, not two."""
    retriever.add_pdf(make_pdf(THREE_TOPIC_PAGES, name="guide.pdf"))

    assert len(retriever.store) == 3


def test_a_failed_reupload_keeps_the_existing_index(retriever, tmp_path):
    """TC-60: If a new version of a file cannot be read, the old chunks stay searchable."""
    broken = tmp_path / "guide.pdf"
    broken.write_bytes(b"not a pdf")

    with pytest.raises(InvalidPDFError):
        retriever.add_pdf(broken)

    assert len(retriever.store) == 3


def test_two_documents_can_be_indexed_together(retriever, make_pdf):
    """TC-61: Chunks from two files are searchable together and keep their own source names."""
    retriever.add_pdf(make_pdf([["Tides are caused by the moon's gravity pulling on oceans."]], name="sea.pdf"))

    best = retriever.search("what causes ocean tides", k=1)[0]

    assert best.chunk.source == "sea.pdf"


def test_blank_query_is_rejected(retriever):
    """TC-62: An empty or whitespace-only question raises ValueError."""
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_unrelated_question_can_be_filtered_with_min_score(retriever):
    """TC-63: A question sharing no words with the document returns nothing above min_score."""
    assert retriever.search("quantum chromodynamics lagrangian", k=3, min_score=0.2) == []


def test_save_and_load_round_trip(retriever, tmp_path):
    """TC-64: A saved retriever reloads with the same embedder and gives the same results."""
    retriever.save(tmp_path / "idx")

    loaded = Retriever.load(tmp_path / "idx", HashingEmbedder())

    question = "how many fans does the stadium seat"
    assert loaded.search(question, k=2) == retriever.search(question, k=2)


def test_loading_with_a_different_embedder_is_refused(retriever, tmp_path):
    """TC-65: An index built with one embedder cannot be loaded with another."""
    retriever.save(tmp_path / "idx")

    with pytest.raises(EmbedderMismatchError):
        Retriever.load(tmp_path / "idx", HashingEmbedder(dimension=128))


def test_store_and_embedder_dimensions_must_match():
    """TC-66: Building a retriever from a store and an embedder of different sizes fails early."""
    from rag.vectorstore import VectorStore

    with pytest.raises(ValueError):
        Retriever(HashingEmbedder(dimension=64), VectorStore(dimension=32))


def test_cli_prints_ranked_passages_with_pages(make_pdf, capsys):
    """TC-67: The query tool exits 0 and lists the best passage first with its page number."""
    path = make_pdf(THREE_TOPIC_PAGES, name="guide.pdf")

    exit_code = main([str(path), "how many fans does the stadium seat", "--k", "2"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Indexed 3 chunks from guide.pdf" in out
    assert "1. score" in out and "page 3" in out.split("\n")[1]


def test_cli_reports_error_for_missing_file(tmp_path, capsys):
    """TC-68: The query tool exits 1 with a readable error when the PDF does not exist."""
    exit_code = main([str(tmp_path / "missing.pdf"), "anything"])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err
