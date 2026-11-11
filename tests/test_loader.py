"""Tests for PDF loading and text extraction (rag.loader)."""

from __future__ import annotations

import pytest

from rag.errors import EncryptedPDFError, InvalidPDFError, NoExtractableTextError
from rag.loader import load_pdf


def test_extracts_text_with_one_based_page_numbers(make_pdf):
    """TC-01: Every page with text is returned with its 1-based page number."""
    path = make_pdf([["First page about apples."], ["Second page about oranges."]])

    pages = load_pdf(path)

    assert [p.page for p in pages] == [1, 2]
    assert "apples" in pages[0].text
    assert "oranges" in pages[1].text


def test_blank_pages_are_skipped_but_numbering_is_kept(make_pdf):
    """TC-02: A page without text is dropped and later pages keep their true number."""
    path = make_pdf([["Alpha content."], [], ["Gamma content."]])

    pages = load_pdf(path)

    assert [p.page for p in pages] == [1, 3]


def test_source_is_the_file_name(make_pdf):
    """TC-03: Each page records the file name (not the full path) as its source."""
    path = make_pdf([["Some text."]], name="annual-report.pdf")

    assert load_pdf(path)[0].source == "annual-report.pdf"


def test_missing_file_raises_file_not_found(tmp_path):
    """TC-04: A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")


def test_wrong_extension_is_rejected(tmp_path):
    """TC-05: A file that is not named *.pdf raises InvalidPDFError."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("just text")

    with pytest.raises(InvalidPDFError):
        load_pdf(text_file)


def test_corrupt_pdf_raises_invalid_pdf_error(tmp_path):
    """TC-06: A .pdf file whose bytes are not a valid PDF raises InvalidPDFError."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is definitely not a pdf")

    with pytest.raises(InvalidPDFError):
        load_pdf(broken)


def test_pdf_without_text_layer_raises_no_extractable_text(make_pdf):
    """TC-07: A PDF whose pages have no text (like a scan) raises NoExtractableTextError."""
    path = make_pdf([[], []])

    with pytest.raises(NoExtractableTextError):
        load_pdf(path)


def test_password_protected_pdf_raises_encrypted_error(make_pdf, encrypt_pdf):
    """TC-08: A PDF that needs a password raises EncryptedPDFError."""
    locked = encrypt_pdf(make_pdf([["Secret content."]]), "hunter2")

    with pytest.raises(EncryptedPDFError):
        load_pdf(locked)
