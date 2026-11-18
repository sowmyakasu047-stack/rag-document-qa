"""Run the real web app with a stub LLM, for browser checks that need no API key.

The stub answers by quoting the first sentence of the best excerpt and citing it, so
this exercises upload, retrieval, citations and the interface, but says nothing
about the quality of real Claude answers.

    python e2e/serve_stub.py --dist frontend/dist --port 8765
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from rag.embeddings import HashingEmbedder  # noqa: E402
from rag.prompt import NOT_FOUND_MESSAGE  # noqa: E402
from rag.server import build_app  # noqa: E402


class StubLLM:
    def generate(self, system: str, prompt: str) -> str:
        match = re.search(r'<excerpt id="1"[^>]*>\n(.*?)\n</excerpt>', prompt, re.S)
        if not match:
            return NOT_FOUND_MESSAGE
        first_sentence = match.group(1).split(". ")[0].strip()
        return f"According to the document: {first_sentence} [1]."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True, help="Built frontend folder")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as index_dir:
        app = build_app(
            HashingEmbedder(),
            lambda: StubLLM(),
            index_dir=index_dir,
            frontend_dist=args.dist,
            min_score=0.1,
        )
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
