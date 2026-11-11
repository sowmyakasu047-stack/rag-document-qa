"""Tests for text cleaning and header/footer removal (rag.preprocess)."""

from __future__ import annotations

from rag.models import PageText
from rag.preprocess import clean_text, strip_repeated_lines


def test_hyphenated_line_break_is_rejoined():
    """TC-09: A word split with a hyphen at a line end is restored."""
    assert clean_text("The exam-\nple is simple.") == "The example is simple."


def test_wrapped_lines_are_joined_into_one_paragraph():
    """TC-10: Lines that were only wrapped by page width become a single line."""
    assert clean_text("one two\nthree four\nfive") == "one two three four five"


def test_paragraph_breaks_are_preserved():
    """TC-11: Blank-line separated paragraphs stay separate, with exactly one blank line."""
    raw = "First paragraph\ncontinues here.\n\n\n\nSecond paragraph."

    assert clean_text(raw) == "First paragraph continues here.\n\nSecond paragraph."


def test_ligatures_are_normalised():
    """TC-12: Unicode ligatures such as 'ﬁ' are expanded so searches match plain text."""
    assert clean_text("ﬁnancial ﬂow") == "financial flow"


def test_control_characters_and_extra_spaces_are_removed():
    """TC-13: Control characters are dropped and runs of spaces/tabs collapse to one space."""
    assert clean_text("a\x00b\x0c   c\t\td") == "ab c d"


def test_repeated_header_and_footer_are_removed():
    """TC-14: A header and a numbered footer repeated on most pages are stripped."""
    bodies = ["apples grow here", "oranges grow there", "pears fall down", "grapes hang high"]
    pages = [
        PageText("doc.pdf", number, f"ACME Annual Report\n{body}\nPage {number}")
        for number, body in enumerate(bodies, start=1)
    ]

    result = strip_repeated_lines(pages)

    assert [p.text for p in result] == bodies
    assert [p.page for p in result] == [1, 2, 3, 4]


def test_repeated_body_text_is_not_mistaken_for_a_header():
    """TC-29: A line repeated in the middle of every page is body text and is kept."""
    pages = [
        PageText("doc.pdf", n, f"top {w}\nline two {w}\nNote: see appendix\nline four {w}\nend {w}")
        for n, w in enumerate(["aa", "bb", "cc"], start=1)
    ]

    result = strip_repeated_lines(pages)

    assert all("Note: see appendix" in p.text for p in result)


def test_short_documents_are_left_unchanged():
    """TC-15: Documents under three pages are not touched, as repetition proves nothing."""
    pages = [
        PageText("doc.pdf", 1, "Title\nbody one"),
        PageText("doc.pdf", 2, "Title\nbody two"),
    ]

    assert strip_repeated_lines(pages) == pages
