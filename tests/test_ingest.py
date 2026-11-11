"""End-to-end tests for the ingestion pipeline and its command line entry point."""

from __future__ import annotations

import pytest

from rag.errors import NoExtractableTextError
from rag.ingest import ingest_pdf, main


def test_ingest_returns_source_tagged_chunks(make_pdf):
    """TC-25: A PDF goes in and chunks come out, each tagged with its file and page."""
    path = make_pdf([["Vector databases store embeddings."], ["Chunking splits long text."]])

    chunks = ingest_pdf(path)

    assert [(c.source, c.page) for c in chunks] == [("sample.pdf", 1), ("sample.pdf", 2)]
    assert "embeddings" in chunks[0].text


def test_ingest_removes_running_header(make_pdf):
    """TC-26: A header repeated on every page does not end up inside any chunk."""
    path = make_pdf(
        [
            ["CONFIDENTIAL DRAFT", "Retrieval finds relevant passages."],
            ["CONFIDENTIAL DRAFT", "Prompts combine context and question."],
            ["CONFIDENTIAL DRAFT", "Citations point back to pages."],
        ]
    )

    chunks = ingest_pdf(path)

    assert chunks
    assert all("CONFIDENTIAL" not in c.text for c in chunks)
    assert "Retrieval" in chunks[0].text


def test_ingest_raises_when_nothing_is_left_after_cleaning(make_pdf):
    """TC-30: If cleaning removes every line, ingestion fails loudly instead of returning nothing."""
    path = make_pdf([["Identical line."], ["Identical line."], ["Identical line."]])

    with pytest.raises(NoExtractableTextError):
        ingest_pdf(path)


def test_cli_prints_summary_for_valid_pdf(make_pdf, capsys):
    """TC-27: The command line tool exits 0 and reports the number of chunks."""
    path = make_pdf([["Hello from the command line."]])

    exit_code = main([str(path)])

    assert exit_code == 0
    assert "1 chunks from 1 page(s)" in capsys.readouterr().out


def test_cli_reports_error_for_missing_file(tmp_path, capsys):
    """TC-28: The command line tool exits 1 and prints a readable error for a missing file."""
    exit_code = main([str(tmp_path / "missing.pdf")])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err
