"""Try retrieval from the command line.

::

    PYTHONPATH=src python -m rag.query file.pdf "What is chunking?" --k 3
    PYTHONPATH=src python -m rag.query file.pdf "What is chunking?" --embedder minilm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.embeddings import DEFAULT_MODEL, HashingEmbedder, SentenceTransformerEmbedder
from rag.errors import RagError
from rag.retriever import Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find the passages of a PDF relevant to a question.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF file")
    parser.add_argument("question", help="The question to search for")
    parser.add_argument("--k", type=int, default=3, help="Number of passages to return")
    parser.add_argument(
        "--embedder",
        choices=["hashing", "minilm"],
        default="hashing",
        help="hashing: offline word-overlap baseline; minilm: neural model (needs sentence-transformers)",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args(argv)

    try:
        embedder = (
            SentenceTransformerEmbedder(DEFAULT_MODEL)
            if args.embedder == "minilm"
            else HashingEmbedder()
        )
        retriever = Retriever(embedder)
        count = retriever.add_pdf(args.pdf, args.chunk_size, args.overlap)
        results = retriever.search(args.question, k=args.k)
    except (RagError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed {count} chunks from {args.pdf.name} with '{embedder.name}'.")
    if not results:
        print("No matching passages found.")
    for rank, result in enumerate(results, start=1):
        preview = result.chunk.text[:160].replace("\n", " ")
        print(f"{rank}. score {result.score:.3f} | page {result.chunk.page} | {preview}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
