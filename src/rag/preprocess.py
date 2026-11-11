"""Text cleaning applied between extraction and chunking.

PDF extraction returns text laid out for a page, not for a reader: lines are
wrapped at the page width, words are hyphenated at line ends, and headers or
footers repeat on every page. Left alone, these artefacts hurt embeddings
(a broken word embeds differently from the whole word) and waste chunk space.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from rag.models import PageText

# Control characters except tab (\x09), newline (\x0a) and carriage return (\x0d).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# "exam-\nple" -> "example": a lowercase letter, hyphen, line break, lowercase letter.
_HYPHEN_LINEBREAK = re.compile(r"([a-z])-\n([a-z])")
_BLANK_LINES = re.compile(r"\n\s*\n+")
_SPACES = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    """Normalise raw extracted text while keeping paragraph boundaries.

    Steps: unicode normalisation (ligatures like "ﬁ" become "fi"), control
    character removal, de-hyphenation across line breaks, joining of lines
    that were only wrapped by the page width, and whitespace collapsing.
    Paragraphs (blank-line separated) are preserved as a single blank line.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    paragraphs = []
    for block in _BLANK_LINES.split(text):
        lines = [_SPACES.sub(" ", line).strip() for line in block.split("\n")]
        joined = " ".join(line for line in lines if line)
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def strip_repeated_lines(
    pages: list[PageText],
    min_fraction: float = 0.6,
    edge_lines: int = 2,
) -> list[PageText]:
    """Remove running headers and footers.

    Only the first and last ``edge_lines`` non-empty lines of each page are
    candidates, because that is where headers and footers live; body text
    that happens to repeat is never touched. A candidate is removed when it
    appears (after replacing digits, so "Page 3" and "Page 4" match) among
    the edge lines of at least ``min_fraction`` of the pages.

    Documents with fewer than three pages are returned unchanged, because
    with so few pages repetition is not meaningful evidence.
    """
    if len(pages) < 3:
        return list(pages)

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    def edge_positions(lines: list[str]) -> set[int]:
        non_empty = [i for i, line in enumerate(lines) if line.strip()]
        return set(non_empty[:edge_lines] + non_empty[-edge_lines:])

    split_pages = [page.text.split("\n") for page in pages]

    seen_on_pages: Counter[str] = Counter()
    for lines in split_pages:
        seen_on_pages.update({key(lines[i]) for i in edge_positions(lines)})

    threshold = min_fraction * len(pages)
    repeated = {k for k, count in seen_on_pages.items() if count >= threshold}

    cleaned = []
    for page, lines in zip(pages, split_pages):
        edges = edge_positions(lines)
        kept = [
            line for i, line in enumerate(lines) if not (i in edges and key(line) in repeated)
        ]
        text = "\n".join(kept).strip()
        if text:
            cleaned.append(PageText(source=page.source, page=page.page, text=text))
    return cleaned
