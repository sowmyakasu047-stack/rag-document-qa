"""Ask a question about a PDF from the command line.

::

    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=src python -m rag.ask file.pdf "What is the main conclusion?" --embedder minilm

    # or with a local model through Ollama (no API key):
    PYTHONPATH=src python -m rag.ask file.pdf "What is the main conclusion?" --llm ollama
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.answerer import DEFAULT_K, DEFAULT_MIN_SCORE, Answerer
from rag.embeddings import DEFAULT_MODEL as EMBEDDING_MODEL
from rag.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from rag.errors import RagError
from rag.llm import make_llm
from rag.retriever import Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about a PDF.")
    parser.add_argument("pdf", type=Path, help="Path to a PDF file")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Excerpts given to the model")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--embedder", choices=["hashing", "minilm"], default="minilm")
    parser.add_argument("--llm", choices=["claude", "ollama"], default="claude", help="Which model provider answers")
    parser.add_argument("--model", help="Model name (default: claude-sonnet-5-5, or llama3.2 for ollama)")
    parser.add_argument("--ollama-url", help="Ollama address (default: OLLAMA_HOST or http://127.0.0.1:11434)")
    args = parser.parse_args(argv)

    try:
        llm = make_llm(args.llm, args.model, args.ollama_url)
        embedder = (
            HashingEmbedder()
            if args.embedder == "hashing"
            else SentenceTransformerEmbedder(EMBEDDING_MODEL)
        )
        retriever = Retriever(embedder)
        retriever.add_pdf(args.pdf)
        answer = Answerer(retriever, llm, k=args.k, min_score=args.min_score).ask(args.question)
    except (RagError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(answer.text)
    for warning in answer.warnings:
        print(f"\nWarning: {warning}")
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            print(f"  [{source.number}] {source.source}, page {source.page} (score {source.score:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
