"""Question in, grounded answer with source references out.

Flow: retrieve chunks, drop weak matches, ask the LLM to answer from them, then
check the reply instead of trusting it: find which excerpts it cited, ignore
citations to excerpts that do not exist, and detect the "not found" sentence.

Guards against hallucination, in the order they act:

1. nothing relevant was retrieved (every chunk is below ``min_score``): the fixed not-found
   message is returned without any model call;
2. the prompt limits the model to the excerpts and gives it an exact "not found" sentence;
3. the model saying "not found" is recognised, and no sources are shown for it;
4. citations that point at excerpts which do not exist are dropped and reported;
5. figures in the answer that appear in no excerpt are reported (``rag.grounding``);
6. optionally (``require_citations``) an answer that cites nothing is withheld.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from rag.grounding import unsupported_numbers
from rag.llm import LLM
from rag.prompt import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from rag.retriever import Retriever
from rag.vectorstore import SearchResult

DEFAULT_K = 4
DEFAULT_MIN_SCORE = 0.2

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Source:
    """An excerpt shown to the model, numbered as in the prompt."""

    number: int
    source: str
    page: int
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    """The result of asking a question.

    Attributes:
        text: The answer, or the fixed not-found message.
        answered: False when the document does not contain the answer.
        sources: The excerpts the answer cites, in number order (empty if not answered).
        retrieved: Every excerpt the model was given, for inspection and debugging.
        warnings: Problems found when checking the reply (uncited answer, bad citations).
    """

    text: str
    answered: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _cited_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        numbers.extend(int(part) for part in match.group(1).split(","))
    return numbers


def _says_not_found(reply: str) -> bool:
    """True if the reply is the not-found sentence. Smaller models often wrap it in quotes."""
    cleaned = reply.strip().strip("\"'`\u201c\u201d\u2018\u2019 ").casefold()
    return cleaned.startswith(NOT_FOUND_MESSAGE.casefold())


def _to_sources(results: Sequence[SearchResult]) -> list[Source]:
    return [
        Source(n, r.chunk.source, r.chunk.page, r.chunk.text, r.score)
        for n, r in enumerate(results, start=1)
    ]


class Answerer:
    """Answers questions about the documents indexed in a :class:`Retriever`."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        k: int = DEFAULT_K,
        min_score: float | None = DEFAULT_MIN_SCORE,
        require_citations: bool = False,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_score = min_score
        self.require_citations = require_citations

    def ask(self, question: str) -> Answer:
        """Answer ``question`` from the indexed documents.

        Raises:
            ValueError: the question is empty.
            LLMError / LLMUnavailableError: the model could not be used.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")

        results = self.retriever.search(question, k=self.k, min_score=self.min_score)
        if not results:
            return Answer(text=NOT_FOUND_MESSAGE, answered=False)

        retrieved = _to_sources(results)
        reply = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, results))

        if _says_not_found(reply):
            return Answer(text=NOT_FOUND_MESSAGE, answered=False, retrieved=retrieved)

        cited = _cited_numbers(reply)
        valid = sorted({n for n in cited if 1 <= n <= len(retrieved)})
        invalid = sorted({n for n in cited if n not in valid})

        if self.require_citations and not valid:
            return Answer(
                text=NOT_FOUND_MESSAGE,
                answered=False,
                retrieved=retrieved,
                warnings=["The model's reply did not cite any excerpt, so it was withheld."],
            )

        warnings = []
        if not valid:
            warnings.append("The answer does not cite any excerpt; treat it with caution.")
        if invalid:
            listed = ", ".join(f"[{n}]" for n in invalid)
            warnings.append(f"The answer cites excerpts that were not provided: {listed}.")
        unsupported = unsupported_numbers(
            reply, question, [source.text for source in retrieved], [source.page for source in retrieved]
        )
        if unsupported:
            warnings.append(
                "The answer contains figures that do not appear in the excerpts "
                f"({', '.join(unsupported)}). Check them against the document."
            )

        return Answer(
            text=reply.strip(),
            answered=True,
            sources=[retrieved[n - 1] for n in valid],
            retrieved=retrieved,
            warnings=warnings,
        )
