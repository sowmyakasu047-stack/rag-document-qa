"""Recursive character chunking with overlap.

Why chunk at all? Embedding models and LLM context windows work best on
focused passages. A whole page mixes several topics into one vector, so the
search cannot tell which topic a question is about. Chunks that are too
small lose the context needed to answer.

Strategy: split on the largest natural boundary that keeps pieces under
``chunk_size`` (paragraph, then line, then sentence, then word, then raw
characters), then merge neighbouring pieces back up to ``chunk_size``.
Consecutive chunks share up to ``overlap`` characters so an answer that sits
on a boundary is still fully contained in at least one chunk.
"""

from __future__ import annotations

from pathlib import PurePath

from rag.models import Chunk, PageText

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150

# Tried in order; separators stay attached to the end of the piece they close.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split pages into overlapping chunks that keep their source and page.

    Chunks never cross a page boundary, so every chunk has an exact citation.

    Raises:
        ValueError: ``chunk_size`` is not positive, or ``overlap`` is negative
            or not smaller than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least 0 and smaller than chunk_size.")

    chunks: list[Chunk] = []
    for page in pages:
        stem = PurePath(page.source).stem
        units = _split_into_units(page.text, chunk_size, _SEPARATORS)
        for page_chunk_number, text in enumerate(_merge_units(units, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}-p{page.page}-c{page_chunk_number}",
                    text=text,
                    source=page.source,
                    page=page.page,
                    index=len(chunks),
                )
            )
    return chunks


def _split_into_units(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than ``chunk_size``; joining them restores the text."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, remaining = separators[0], separators[1:]
    if separator not in text:
        return _split_into_units(text, chunk_size, remaining)

    parts = text.split(separator)
    units: list[str] = []
    for position, part in enumerate(parts):
        piece = part + separator if position < len(parts) - 1 else part
        units.extend(_split_into_units(piece, chunk_size, remaining))
    return units


def _merge_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack units into chunks, carrying a tail of each chunk into the next."""
    merged: list[str] = []
    current: list[str] = []
    current_length = 0

    for unit in units:
        if current and current_length + len(unit) > chunk_size:
            merged.append("".join(current))
            # Drop units from the front until the carried-over tail fits the
            # overlap budget and leaves room for the incoming unit.
            while current and (current_length > overlap or current_length + len(unit) > chunk_size):
                current_length -= len(current.pop(0))
        current.append(unit)
        current_length += len(unit)

    if current:
        merged.append("".join(current))

    stripped = (chunk.strip() for chunk in merged)
    return [chunk for chunk in stripped if chunk]
