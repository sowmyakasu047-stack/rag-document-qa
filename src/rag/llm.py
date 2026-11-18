"""The language model that writes the final answer.

Only one method is needed from a model, so :class:`LLM` is a tiny interface. That
keeps the answering logic independent of any vendor and lets tests use a fake.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Protocol

from rag.errors import LLMError, LLMUnavailableError

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


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


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _normalize_base_url(url: str) -> str:
    """Accept ``host:port`` (the form used by Ollama's OLLAMA_HOST) as well as a full URL."""
    url = url.strip().rstrip("/")
    return url if url.startswith(("http://", "https://")) else f"http://{url}"


class OllamaLLM:
    """Answers with a model running on your own computer through Ollama (no API key).

    Install Ollama from https://ollama.com, then download a model once::

        ollama pull llama3.2

    The constructor checks that Ollama is running and the model is installed, and
    raises :class:`LLMUnavailableError` with the exact fix if not. Because the web
    app only creates the model on the first question and retries after a failure,
    starting Ollama or pulling the model later needs no restart.

    Small local models follow instructions less reliably than Claude, so expect
    more answers without citations (the app flags those with a warning).

    Args:
        model: Name as shown by ``ollama list``.
        base_url: Ollama's address; defaults to ``OLLAMA_HOST`` or ``http://127.0.0.1:11434``.
        timeout: Seconds to wait for an answer. CPU-only machines can be slow.
        num_ctx: Context window requested from Ollama; the prompt holds up to ten excerpts.
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str | None = None,
        timeout: float = 300.0,
        num_ctx: int = 4096,
    ) -> None:
        self.model = model
        self.base_url = _normalize_base_url(base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL)
        self.timeout = timeout
        self.num_ctx = num_ctx
        self._check_ready()

    def _open(self, request: urllib.request.Request, timeout: float):
        # Local requests must not go through a system proxy, which would not reach localhost.
        if urllib.parse.urlparse(self.base_url).hostname in _LOCAL_HOSTS:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(request, timeout=timeout)
        return urllib.request.urlopen(request, timeout=timeout)

    def _call(self, path: str, payload: dict | None, timeout: float) -> object:
        request = urllib.request.Request(
            self.base_url + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with self._open(request, timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _check_ready(self) -> None:
        try:
            tags = self._call("/api/tags", None, timeout=5.0)
            installed = [item["name"] for item in tags["models"]]
        except urllib.error.HTTPError as exc:
            raise LLMUnavailableError(
                f"Ollama at {self.base_url} answered with HTTP {exc.code}. Is that the right address?"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.base_url}. Install it from https://ollama.com and "
                "make sure it is running (open the Ollama app, or run 'ollama serve')."
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMUnavailableError(f"The service at {self.base_url} does not look like Ollama.") from exc

        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in installed and self.model not in installed:
            have = ", ".join(installed) if installed else "none"
            raise LLMUnavailableError(
                f"The model '{self.model}' is not installed in Ollama (installed: {have}). "
                f"Run: ollama pull {self.model}"
            )

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        try:
            body = self._call("/api/chat", payload, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise LLMError(f"Ollama returned an error: {_error_text(exc)}") from exc
        except TimeoutError as exc:
            raise LLMError(
                f"Ollama did not answer within {self.timeout:.0f} seconds. A smaller model, or a "
                "machine with a GPU, will be faster."
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(f"Could not get an answer from Ollama at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise LLMError("Ollama returned a response that could not be read.") from exc

        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        text = _THINK_BLOCK.sub("", content).strip() if isinstance(content, str) else ""
        if not text:
            raise LLMError("The language model returned an empty answer.")
        return text


def _error_text(exc: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(exc.read().decode("utf-8"))["error"])
    except (ValueError, KeyError, TypeError, OSError):
        return f"HTTP {exc.code}"


def make_llm(provider: str = "claude", model: str | None = None, ollama_url: str | None = None) -> LLM:
    """Create the language model client for ``provider`` ('claude' or 'ollama')."""
    if provider == "claude":
        return ClaudeLLM(model=model or DEFAULT_MODEL)
    if provider == "ollama":
        return OllamaLLM(model=model or DEFAULT_OLLAMA_MODEL, base_url=ollama_url)
    raise ValueError(f"Unknown LLM provider '{provider}'. Use 'claude' or 'ollama'.")
