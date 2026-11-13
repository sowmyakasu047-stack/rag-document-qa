"""Tests for the Claude client wrapper (rag.llm), using fakes instead of the real API."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from rag.errors import LLMError, LLMUnavailableError
from rag.llm import ClaudeLLM


class _FakeMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _client(outcome):
    return SimpleNamespace(messages=_FakeMessages(outcome))


def _response(*blocks):
    return SimpleNamespace(content=list(blocks))


def _text(value):
    return SimpleNamespace(type="text", text=value)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Install a stand-in 'anthropic' module so the wrapper can be tested without the package."""
    module = SimpleNamespace(APIError=ConnectionError, Anthropic=lambda: _client(_response(_text("hi"))))
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return module


def test_generate_sends_model_system_and_prompt_and_returns_text(fake_anthropic):
    """TC-77: The request carries the model, system prompt and user prompt; text blocks are joined."""
    client = _client(_response(_text("Hello "), SimpleNamespace(type="tool_use"), _text("world")))
    llm = ClaudeLLM(model="m-1", max_tokens=50, client=client)

    assert llm.generate("SYS", "PROMPT") == "Hello world"
    assert client.messages.kwargs == {
        "model": "m-1",
        "max_tokens": 50,
        "system": "SYS",
        "messages": [{"role": "user", "content": "PROMPT"}],
    }


def test_api_failure_becomes_llm_error(fake_anthropic):
    """TC-78: A failure from the API library is raised as LLMError with the cause attached."""
    llm = ClaudeLLM(client=_client(ConnectionError("network down")))

    with pytest.raises(LLMError, match="network down"):
        llm.generate("s", "p")


def test_empty_reply_becomes_llm_error(fake_anthropic):
    """TC-79: A reply with no text raises LLMError instead of returning an empty answer."""
    llm = ClaudeLLM(client=_client(_response()))

    with pytest.raises(LLMError, match="empty"):
        llm.generate("s", "p")


def test_missing_package_gives_clear_error(monkeypatch):
    """TC-80: Without the anthropic package, a readable error explains how to install it."""
    monkeypatch.setitem(sys.modules, "anthropic", None)

    with pytest.raises(LLMUnavailableError, match="pip install anthropic"):
        ClaudeLLM()


def test_missing_api_key_gives_clear_error(fake_anthropic, monkeypatch):
    """TC-81: Without ANTHROPIC_API_KEY, a readable error says how to set it."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMUnavailableError, match="ANTHROPIC_API_KEY"):
        ClaudeLLM()
