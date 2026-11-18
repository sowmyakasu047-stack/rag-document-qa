"""Tests for the local-model option (rag.llm.OllamaLLM) against a fake Ollama server."""

from __future__ import annotations

import socket

import pytest

from fake_ollama import FakeOllama
from rag import ask
from rag.answerer import Answerer
from rag.embeddings import HashingEmbedder
from rag.errors import LLMError, LLMUnavailableError
from rag.llm import ClaudeLLM, OllamaLLM, make_llm
from rag.prompt import NOT_FOUND_MESSAGE
from rag.retriever import Retriever

PAGES = [
    ["Chunking splits long documents into small overlapping passages", "so that each passage covers one topic."],
    ["Football season begins in September.", "The stadium seats seventy thousand fans for each match."],
]


def test_generate_sends_the_chat_request_and_returns_the_reply():
    """TC-129: The request has the model, system and user messages, no streaming, temperature 0."""
    with FakeOllama(reply="Seventy thousand [1].") as server:
        llm = OllamaLLM("llama3.2", base_url=server.url, num_ctx=2048)

        assert llm.generate("SYSTEM RULES", "THE PROMPT") == "Seventy thousand [1]."

    request = server.chat_requests[0]
    assert request["model"] == "llama3.2"
    assert request["messages"] == [
        {"role": "system", "content": "SYSTEM RULES"},
        {"role": "user", "content": "THE PROMPT"},
    ]
    assert request["stream"] is False
    assert request["options"] == {"temperature": 0, "num_ctx": 2048}


def test_model_name_without_a_tag_matches_the_latest_tag():
    """TC-130: 'llama3.2' is found when Ollama lists 'llama3.2:latest'; a tagged name must match exactly."""
    with FakeOllama(models=["llama3.2:latest", "qwen2.5:7b"]) as server:
        OllamaLLM("llama3.2", base_url=server.url)
        OllamaLLM("qwen2.5:7b", base_url=server.url)
        with pytest.raises(LLMUnavailableError):
            OllamaLLM("qwen2.5:3b", base_url=server.url)


def test_ollama_not_running_gives_a_clear_error():
    """TC-131: With nothing listening, the error says Ollama cannot be reached and how to start it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]  # a port that is free, so nothing answers

    with pytest.raises(LLMUnavailableError, match="Could not reach Ollama") as info:
        OllamaLLM(base_url=f"http://127.0.0.1:{closed_port}")

    assert "ollama serve" in str(info.value)


def test_missing_model_tells_you_the_pull_command():
    """TC-132: If the model is not installed, the error lists what is installed and the fix."""
    with FakeOllama(models=["mistral:latest"]) as server:
        with pytest.raises(LLMUnavailableError) as info:
            OllamaLLM("llama3.2", base_url=server.url)

    assert "ollama pull llama3.2" in str(info.value)
    assert "mistral:latest" in str(info.value)


def test_error_from_the_chat_endpoint_becomes_llm_error_with_ollamas_message():
    """TC-133: An HTTP error from Ollama is raised as LLMError containing Ollama's own message."""
    with FakeOllama(chat_status=404) as server:
        llm = OllamaLLM("llama3.2", base_url=server.url)

        with pytest.raises(LLMError, match="model 'x' not found"):
            llm.generate("s", "p")


def test_empty_reply_becomes_llm_error():
    """TC-134: A reply with no text raises LLMError."""
    with FakeOllama(reply="   ") as server:
        llm = OllamaLLM("llama3.2", base_url=server.url)

        with pytest.raises(LLMError, match="empty"):
            llm.generate("s", "p")


def test_unreadable_response_becomes_llm_error():
    """TC-135: A response that is not valid JSON raises LLMError instead of crashing."""
    with FakeOllama(chat_body=b"<html>oops</html>") as server:
        llm = OllamaLLM("llama3.2", base_url=server.url)

        with pytest.raises(LLMError, match="could not be read"):
            llm.generate("s", "p")


def test_thinking_blocks_are_removed_from_the_reply():
    """TC-136: Reasoning models that wrap their thoughts in <think> tags do not leak them into answers."""
    with FakeOllama(reply="<think>The user wants a number.\nHmm.</think>\nSeventy thousand [1].") as server:
        llm = OllamaLLM("llama3.2", base_url=server.url)

        assert llm.generate("s", "p") == "Seventy thousand [1]."


@pytest.mark.parametrize("form", ["{url}", "{url}/", "{host}"])
def test_address_may_be_a_url_or_host_and_port(form):
    """TC-137: 'http://host:port', a trailing slash, and the bare 'host:port' form of OLLAMA_HOST all work."""
    with FakeOllama() as server:
        address = form.format(url=server.url, host=server.url.removeprefix("http://"))

        assert OllamaLLM("llama3.2", base_url=address).generate("s", "p") == "Hello."


def test_address_defaults_to_the_ollama_host_variable(monkeypatch):
    """TC-138: Without an explicit address, OLLAMA_HOST is used."""
    with FakeOllama() as server:
        monkeypatch.setenv("OLLAMA_HOST", server.url.removeprefix("http://"))

        assert OllamaLLM("llama3.2").generate("s", "p") == "Hello."


def test_make_llm_chooses_the_provider(monkeypatch):
    """TC-139: make_llm builds Ollama or Claude by name and rejects an unknown provider."""
    with FakeOllama() as server:
        llm = make_llm("ollama", model=None, ollama_url=server.url)
        assert isinstance(llm, OllamaLLM) and llm.model == "llama3.2"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setitem(__import__("sys").modules, "anthropic", type("M", (), {"APIError": Exception, "Anthropic": lambda: object()}))
    assert isinstance(make_llm("claude", model="m-1"), ClaudeLLM)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        make_llm("openai")


def test_full_answer_flow_over_http(make_pdf):
    """TC-140: Retrieval, prompt, the HTTP call to Ollama and citation parsing work together."""
    retriever = Retriever(HashingEmbedder())
    retriever.add_pdf(make_pdf(PAGES, name="guide.pdf"))
    with FakeOllama(reply="The stadium seats seventy thousand fans [1].") as server:
        llm = OllamaLLM("llama3.2", base_url=server.url)

        answer = Answerer(retriever, llm, k=2, min_score=None).ask("How many fans does the stadium seat?")

    assert answer.answered and [(s.source, s.page) for s in answer.sources] == [("guide.pdf", 2)]
    prompt = server.chat_requests[0]["messages"][1]["content"]
    assert '<excerpt id="1" source="guide.pdf" page="2">' in prompt


@pytest.mark.parametrize(
    "reply",
    [NOT_FOUND_MESSAGE, f'"{NOT_FOUND_MESSAGE}"', f"  `{NOT_FOUND_MESSAGE.upper()}`  ", f"“{NOT_FOUND_MESSAGE}”"],
)
def test_not_found_is_recognised_even_when_quoted_or_recased(make_pdf, fake_llm, reply):
    """TC-141: Small models often add quotes or change case; the not-found reply is still recognised."""
    retriever = Retriever(HashingEmbedder())
    retriever.add_pdf(make_pdf(PAGES, name="guide.pdf"))

    answer = Answerer(retriever, fake_llm(reply), k=2, min_score=None).ask("How many fans does the stadium seat?")

    assert not answer.answered and answer.sources == []


def test_ask_command_works_with_ollama(make_pdf, capsys):
    """TC-142: 'ask --llm ollama' answers through a (fake) Ollama server and prints sources."""
    path = make_pdf(PAGES, name="guide.pdf")
    with FakeOllama(reply="Seventy thousand fans [1].") as server:
        code = ask.main(
            [str(path), "How many fans does the stadium seat?", "--embedder", "hashing", "--min-score", "0",
             "--llm", "ollama", "--ollama-url", server.url]
        )

    out = capsys.readouterr().out
    assert code == 0 and "Seventy thousand fans [1]." in out and "guide.pdf, page 2" in out


def test_ask_command_explains_when_ollama_is_not_installed_or_running(make_pdf, capsys, monkeypatch):
    """TC-143: With no Ollama reachable, 'ask --llm ollama' exits 1 with the instructions."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    code = ask.main([str(make_pdf(PAGES)), "anything", "--embedder", "hashing", "--llm", "ollama",
                     "--ollama-url", f"http://127.0.0.1:{port}"])

    assert code == 1
    assert "Could not reach Ollama" in capsys.readouterr().err
