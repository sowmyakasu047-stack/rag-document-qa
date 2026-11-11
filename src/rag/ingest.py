"""End-to-end ingestion: PDF file in, source-tagged chunks out.

Run as a script to inspect how a PDF gets split::

    PYTHONPATH=src python -m rag path/to/file.pdf --chunk-size 800 --overlap 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_pages
from rag.errors import NoExtractableTextError, RagError
from rag.loader import load_pdf
from rag.models import Chunk, PageText
from rag.preprocess import clean_text, strip_repeated_lines


def ingest_pdf(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Load, clean and chunk a PDF.

    Raises the same exceptions as :func:`rag.loader.load_pdf` and
    :func:`rag.chunker.chunk_pages`.
    """
    # Header/footer removal must run first: it relies on the raw line
    # structure, which clean_text later flattens by joining wrapped lines.
    pages = strip_repeated_lines(load_pdf(path))
    cleaned = [PageText(p.source, p.page, clean_text(p.text)) for p in pages]
    cleaned = [p for p in cleaned if p.text]
    if not cleaned:
        raise NoExtractableTextError(
            f"No usable text left in '{Path(path).name}' after removing headers and footers."
        )
    return chunk_pages(cleaned, chunk_size=chunk_size, overlap=overlap)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show how a PDF is chunked.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF file")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--show", type=int, default=3, help="Number of chunks to preview")
    args = parser.parse_args(argv)

    try:
        chunks = ingest_pdf(args.pdf, args.chunk_size, args.overlap)
    except (RagError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    pages = len({c.page for c in chunks})
    print(f"{args.pdf.name}: {len(chunks)} chunks from {pages} page(s)")
    for chunk in chunks[: args.show]:
        preview = chunk.text[:120].replace("\n", " ")
        print(f"  [{chunk.chunk_id}] page {chunk.page}, {len(chunk.text)} chars: {preview}...")
    return 0
