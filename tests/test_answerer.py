"""Tests for answering with citations (rag.answerer) and the ask command (rag.ask)."""

from __future__ import annotations

import pytest

from rag import ask
from rag.answerer import Answerer
from rag.embeddings import HashingEmbedder
from rag.errors import LLMError
from rag.prompt import NOT_FOUND_MESSAGE
from rag.retriever import Retriever

PAGES = [
    ["Chunking splits long documents into small overlapping passages", "so that each passage covers one topic."],
    ["Embeddings turn passages into vectors", "and vector databases find the nearest vectors quickly."],
    ["Football season begins in September.", "The stadium seats seventy thousand fans for each match."],
]
STADIUM_Q = "How many fans does the stadium seat?"


@pytest.fixture
def retriever(make_pdf) -> Retriever:
    r = Retriever(HashingEmbedder())
    r.add_pdf(make_pdf(PAGES, name="guide.pdf"))
    return r


def test_cited_answer_returns_only_the_cited_sources(retriever, fake_llm):
    """TC-82: A reply citing [1] returns that excerpt's file and page and nothing else."""
    llm = fake_llm("The stadium seats seventy thousand fans [1].")

    answer = Answerer(retriever, llm, k=3, min_score=None).ask(STADIUM_Q)

    assert answer.answered
    assert [(s.number, s.source, s.page) for s in answer.sources] == [(1, "guide.pdf", 3)]
    assert len(answer.retrieved) == 3
    assert answer.warnings == []


def test_grouped_and_repeated_citations_are_understood(retriever, fake_llm):
    """TC-83: '[1, 2]' and repeated '[2]' count as citations of excerpts 1 and 2, once each."""
    llm = fake_llm("First point [1, 2]. Second point [2].")

    answer = Answerer(retriever, llm, k=3, min_score=None).ask(STADIUM_Q)

    assert [s.number for s in answer.sources] == [1, 2]


def test_citation_to_a_missing_excerpt_is_ignored_with_a_warning(retriever, fake_llm):
    """TC-84: Citing [9] when only 3 excerpts exist drops it and adds a warning."""
    llm = fake_llm("Seventy thousand [1][9].")

    answer = Answerer(retriever, llm, k=3, min_score=None).ask(STADIUM_Q)

    assert [s.number for s in answer.sources] == [1]
    assert any("[9]" in w for w in answer.warnings)


def test_answer_without_citations_is_flagged(retriever, fake_llm):
    """TC-85: A reply that cites nothing is still returned but carries a warning and no sources."""
    answer = Answerer(retriever, fake_llm("About seventy thousand."), k=3, min_score=None).ask(STADIUM_Q)

    assert answer.answered
    assert answer.sources == []
    assert any("does not cite" in w for w in answer.warnings)


def test_model_saying_not_found_gives_unanswered_result(retriever, fake_llm):
    """TC-86: When the model replies with the not-found sentence, answered is False and no sources."""
    llm = fake_llm(NOT_FOUND_MESSAGE)

    answer = Answerer(retriever, llm, k=3, min_score=None).ask(STADIUM_Q)

    assert not answer.answered
    assert answer.text == NOT_FOUND_MESSAGE
    assert answer.sources == []


def test_nothing_relevant_skips_the_model_call(retriever, fake_llm):
    """TC-87: If every chunk is below min_score, the not-found message is returned and the
    LLM is never called."""
    llm = fake_llm("should never be used")

    answer = Answerer(retriever, llm, k=3, min_score=0.2).ask("quantum chromodynamics lagrangian")

    assert not answer.answered and answer.text == NOT_FOUND_MESSAGE
    assert llm.calls == []


def test_blank_question_is_rejected_before_any_work(retriever, fake_llm):
    """TC-88: An empty question raises ValueError and does not call the model."""
    llm = fake_llm()

    with pytest.raises(ValueError):
        Answerer(retriever, llm).ask("   ")

    assert llm.calls == []


def test_llm_failure_propagates(retriever, fake_llm):
    """TC-89: An LLMError from the model reaches the caller instead of being swallowed."""
    with pytest.raises(LLMError):
        Answerer(retriever, fake_llm(LLMError("boom")), k=3, min_score=None).ask(STADIUM_Q)


def test_model_receives_at_most_k_excerpts_and_the_question(retriever, fake_llm):
    """TC-90: The prompt contains k excerpts (here 2) plus the question, with the system rules."""
    llm = fake_llm("Answer [1].")

    Answerer(retriever, llm, k=2, min_score=None).ask(STADIUM_Q)

    system, prompt = llm.calls[0]
    assert prompt.count("<excerpt ") == 2
    assert f"Question: {STADIUM_Q}" in prompt
    assert NOT_FOUND_MESSAGE in system


def test_ask_command_prints_answer_and_sources(make_pdf, fake_llm, monkeypatch, capsys):
    """TC-91: The ask command prints the answer, then a Sources list with file and page."""
    monkeypatch.setattr(ask, "make_llm", lambda *args: fake_llm("Seventy thousand fans [1]."))
    path = make_pdf(PAGES, name="guide.pdf")

    code = ask.main([str(path), STADIUM_Q, "--embedder", "hashing", "--min-score", "0"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Seventy thousand fans [1]." in out
    assert "Sources:" in out and "guide.pdf, page" in out


def test_ask_command_reports_missing_api_key(make_pdf, monkeypatch, capsys):
    """TC-92: Without an API key the ask command exits 1 with a readable error on stderr."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    code = ask.main([str(make_pdf(PAGES)), "anything", "--embedder", "hashing"])

    assert code == 1
    assert "Error:" in capsys.readouterr().err


PTO_PAGES = [["Employees get 18 days of paid time off each year.", "After three years this rises to 23 days."]]


@pytest.fixture
def pto_retriever(make_pdf) -> Retriever:
    r = Retriever(HashingEmbedder())
    r.add_pdf(make_pdf(PTO_PAGES, name="pto.pdf"))
    return r


def test_invented_figure_adds_a_warning_but_keeps_the_answer(pto_retriever, fake_llm):
    """TC-155: 'You get 25 days [1]' when the document says 18 returns the answer with a figures warning."""
    answer = Answerer(pto_retriever, fake_llm("You get 25 days of paid time off [1]."), k=2, min_score=None).ask(
        "How many days of paid time off do employees get?"
    )

    assert answer.answered and [s.page for s in answer.sources] == [1]
    assert any("25" in w and "do not appear in the excerpts" in w for w in answer.warnings)


def test_supported_figures_add_no_warning(pto_retriever, fake_llm):
    """TC-156: '18 days [1]' with 18 in the excerpt produces no warnings."""
    answer = Answerer(pto_retriever, fake_llm("Employees get 18 days [1]."), k=2, min_score=None).ask(
        "How many days of paid time off do employees get?"
    )

    assert answer.answered and answer.warnings == []


def test_require_citations_withholds_an_uncited_answer(pto_retriever, fake_llm):
    """TC-157: With require_citations, a reply that cites nothing is withheld and reported."""
    answer = Answerer(
        pto_retriever, fake_llm("Employees get 18 days."), k=2, min_score=None, require_citations=True
    ).ask("How many days of paid time off do employees get?")

    assert not answer.answered and answer.text == NOT_FOUND_MESSAGE and answer.sources == []
    assert any("withheld" in w for w in answer.warnings)
    assert len(answer.retrieved) == 1


def test_require_citations_keeps_a_cited_answer(pto_retriever, fake_llm):
    """TC-158: With require_citations, a properly cited reply is returned as normal."""
    answer = Answerer(
        pto_retriever, fake_llm("Employees get 18 days [1]."), k=2, min_score=None, require_citations=True
    ).ask("How many days of paid time off do employees get?")

    assert answer.answered and [s.number for s in answer.sources] == [1]


def test_ask_command_can_withhold_uncited_answers(make_pdf, fake_llm, monkeypatch, capsys):
    """TC-159: 'ask --require-citations' prints the not-found message and the reason for an uncited reply."""
    monkeypatch.setattr(ask, "make_llm", lambda *args: fake_llm("Employees get 18 days."))
    path = make_pdf(PTO_PAGES, name="pto.pdf")

    code = ask.main([str(path), "How many days of paid time off?", "--embedder", "hashing", "--min-score", "0",
                     "--require-citations"])

    out = capsys.readouterr().out
    assert code == 0 and NOT_FOUND_MESSAGE in out and "withheld" in out
