"""Plain data containers passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Text extracted from a single PDF page.

    Attributes:
        source: File name the text came from (used for source references).
        page: 1-based page number, matching what a reader sees in a PDF viewer.
        text: Extracted text for that page.
    """

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document.

    A chunk never spans two pages, so ``source`` and ``page`` are exact and
    can be shown to the user as a citation next to a generated answer.

    Attributes:
        chunk_id: Stable identifier, e.g. ``report-p3-c2``.
        text: The chunk content that will be embedded and searched.
        source: File name the chunk came from.
        page: 1-based page number the chunk came from.
        index: Position of the chunk within the whole document (0-based).
    """

    chunk_id: str
    text: str
    source: str
    page: int
    index: int
