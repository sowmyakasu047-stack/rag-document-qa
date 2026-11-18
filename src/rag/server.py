"""Start the web application.

::

    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=src python -m rag.server --embedder minilm

    # or with a local model through Ollama (no API key; run 'ollama pull llama3.2' once first):
    PYTHONPATH=src python -m rag.server --llm ollama

Then open http://127.0.0.1:8000. The index is saved in ``data/index`` and reloaded on
the next start, so uploaded documents survive a restart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from starlette.applications import Starlette

from rag.answerer import DEFAULT_MIN_SCORE
from rag.api import create_app
from rag.embeddings import DEFAULT_MODEL, Embedder, HashingEmbedder, SentenceTransformerEmbedder
from rag.errors import RagError
from rag.llm import ClaudeLLM, make_llm
from rag.retriever import Retriever

DEFAULT_INDEX_DIR = Path("data") / "index"
DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def make_embedder(kind: str) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(DEFAULT_MODEL)


def build_app(
    embedder: Embedder,
    llm_factory=None,
    index_dir: str | Path | None = DEFAULT_INDEX_DIR,
    frontend_dist: str | Path | None = DEFAULT_FRONTEND_DIST,
    min_score: float | None = DEFAULT_MIN_SCORE,
) -> Starlette:
    """Create the app, loading a previously saved index when one exists.

    Raises:
        EmbedderMismatchError: the saved index was built with a different embedder.
        VectorStoreError: the saved index is damaged.
    """
    index_path = Path(index_dir) if index_dir is not None else None
    if index_path is not None and (index_path / "meta.json").is_file():
        retriever = Retriever.load(index_path, embedder)
    else:
        retriever = Retriever(embedder)
    return create_app(
        retriever,
        llm_factory if llm_factory is not None else ClaudeLLM,
        persist_dir=index_path,
        frontend_dist=frontend_dist,
        min_score=min_score,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the document Q&A web application.")
    parser.add_argument("--embedder", choices=["hashing", "minilm"], default="minilm")
    parser.add_argument("--llm", choices=["claude", "ollama"], default="claude", help="Which model provider answers")
    parser.add_argument("--model", help="Model name (default: claude-sonnet-5-5, or llama3.2 for ollama)")
    parser.add_argument("--ollama-url", help="Ollama address (default: OLLAMA_HOST or http://127.0.0.1:11434)")
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 only on a trusted network")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--frontend-dist", type=Path, default=DEFAULT_FRONTEND_DIST)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args(argv)

    try:
        app = build_app(
            make_embedder(args.embedder),
            llm_factory=lambda: make_llm(args.llm, args.model, args.ollama_url),
            index_dir=args.index_dir,
            frontend_dist=args.frontend_dist,
            min_score=args.min_score,
        )
    except RagError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.frontend_dist.is_dir():
        print(
            f"Note: no built frontend at '{args.frontend_dist}'. The API works, but for the "
            "interface run 'npm run build' in frontend/ (or 'npm run dev' for development).",
            file=sys.stderr,
        )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
