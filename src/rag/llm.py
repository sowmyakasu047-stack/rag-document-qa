"""The language model that writes the final answer.

Only one method is needed from a model, so :class:`LLM` is a tiny interface. That
keeps the answering logic independent of any vendor and lets tests use a fake.
"""

from __future__ import annotations

import os
from typing import Protocol

from rag.errors import LLMError, LLMUnavailableError

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str:
        """Return the model's reply to ``prompt`` under the instructions in ``system``."""
        ...


class ClaudeLLM:
    """Answers with Claude through the Anthropic API.

    Needs ``pip install anthropic`` and the ``ANTHROPIC_API_KEY`` environment
    variable. ``client`` can be passed to reuse a configured client (or a fake).
    """

    def __init__(
        self, model: str = DEFAULT_MODEL, max_tokens: int = 1024, client: object | None = None
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            if client is None:
                raise LLMUnavailableError(
                    "The 'anthropic' package is not installed. Run: pip install anthropic"
                ) from exc
            anthropic = None

        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMUnavailableError(
                    "ANTHROPIC_API_KEY is not set. Create a key in the Anthropic Console and "
                    "export it, e.g. export ANTHROPIC_API_KEY=sk-ant-..."
                )
            client = anthropic.Anthropic()

        self._client = client
        self._api_errors: tuple[type[Exception], ...] = (
            (anthropic.APIError,) if anthropic is not None else ()
        )
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except self._api_errors as exc:
            raise LLMError(f"The language model request failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LLMError("The language model returned an empty answer.")
        return text
