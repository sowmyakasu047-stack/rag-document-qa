"""A cheap, deterministic check for invented figures in an answer.

Hallucinated numbers (a wrong date, an amount, a percentage) are the most damaging kind of
mistake in document question answering, and the easiest to catch without another model call:
every figure in the answer should appear somewhere in the excerpts the model was given.

What counts as supported
------------------------
A figure in the answer is supported when it appears in any excerpt, or in the question (an
answer to "after 4 years?" may repeat the 4), or is the page number of an excerpt ("page 2").
Citation markers like ``[1]`` and list numbering like ``1.`` are ignored. Commas, currency
signs and percent signs are ignored when comparing, so ``$2,500`` matches ``2500``.

Limits
------
Numbers written as words ("seventy thousand") are not recognised, so an answer that turns
words into digits is flagged even when it is right, and a model that restates a figure in
words escapes the check. This is a warning signal, not a proof of correctness. It cannot tell
whether a supported figure is attached to the right fact.
"""

from __future__ import annotations

import re
from typing import Iterable

_CITATION = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _normalize(token: str) -> str:
    return token.replace(",", "").rstrip(".")


def numbers_in(text: str) -> list[str]:
    """Figures in ``text`` in order of appearance, normalised (no commas, signs or trailing dots)."""
    return [_normalize(match) for match in _NUMBER.findall(text)]


def unsupported_numbers(
    answer: str,
    question: str,
    excerpts: Iterable[str],
    pages: Iterable[int] = (),
) -> list[str]:
    """Return the figures in ``answer`` that are in no excerpt, the question or a page number.

    Each figure is listed once, in the order it first appears.
    """
    cleaned = _LIST_MARKER.sub("", _CITATION.sub("", answer))
    supported = set(numbers_in(question))
    supported.update(str(page) for page in pages)
    for excerpt in excerpts:
        supported.update(numbers_in(excerpt))

    missing: list[str] = []
    for figure in numbers_in(cleaned):
        if figure and figure not in supported and figure not in missing:
            missing.append(figure)
    return missing
