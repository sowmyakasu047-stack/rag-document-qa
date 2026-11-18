"""Tests for the figure-grounding check (rag.grounding)."""

from __future__ import annotations

from rag.grounding import numbers_in, unsupported_numbers


def check(answer, excerpts=("",), question="", pages=()):
    return unsupported_numbers(answer, question, excerpts, pages)


def test_figures_found_in_an_excerpt_are_supported():
    """TC-144: Every figure in the answer appears in an excerpt, so nothing is reported."""
    excerpts = ["New employees get 18 days of paid time off, rising to 23 days after three years."]

    assert check("You get 18 days, and 23 days later [1].", excerpts) == []


def test_a_figure_that_is_in_no_excerpt_is_reported():
    """TC-145: An answer saying 25 days when the document says 18 is flagged."""
    assert check("You get 25 days of PTO [1].", ["New employees get 18 days of PTO."]) == ["25"]


def test_commas_currency_and_percent_signs_do_not_matter():
    """TC-146: '$2,500' in an excerpt supports '2500' and '2,500'; '4%' supports '4 percent'."""
    excerpts = ["Any expense above $2,500 needs approval. The match is 100% of the first 4% saved."]

    assert check("Above 2500 a director must approve. Also 2,500 is the limit.", excerpts) == []
    assert check("It matches the first 4 percent.", excerpts) == []


def test_figures_repeated_from_the_question_are_allowed():
    """TC-147: 'After 4 years you get 23 days' is fine when the 4 came from the question."""
    result = check("After 4 years you get 23 days [1].", ["23 days after three years"], question="I joined 4 years ago")

    assert result == []


def test_page_numbers_of_the_excerpts_are_allowed():
    """TC-148: 'see page 2' is supported when an excerpt came from page 2."""
    assert check("This is stated on page 2 [1].", ["no digits here"], pages=[2]) == []
    assert check("This is stated on page 7 [1].", ["no digits here"], pages=[2]) == ["7"]


def test_citation_markers_and_list_numbering_are_ignored():
    """TC-149: '[1]', '[2, 3]' and leading '1.' / '2)' list markers are not treated as figures."""
    answer = "1. First point [1]\n2) Second point [2, 3]"

    assert check(answer, ["text without any digits"]) == []


def test_decimals_are_compared_exactly():
    """TC-150: '4.5' is supported by '4.5 percent' but '4.6' is flagged."""
    assert check("It is 4.5%.", ["rate: 4.5 percent"]) == []
    assert check("It is 4.6%.", ["rate: 4.5 percent"]) == ["4.6"]


def test_each_missing_figure_is_listed_once_in_order():
    """TC-151: Repeated figures are listed once, in order of first appearance."""
    assert check("7 and 9 and 7 and 12", ["nothing"]) == ["7", "9", "12"]


def test_answer_without_figures_reports_nothing():
    """TC-152: Text with no digits can never be flagged."""
    assert check("The stadium seats many fans [1].", ["seventy thousand"]) == []


def test_a_trailing_full_stop_is_not_part_of_the_figure():
    """TC-153: '... costs 600.' matches an excerpt's '$600'."""
    assert check("The stipend costs 600.", ["a $600 stipend"]) == []


def test_numbers_written_as_words_are_a_known_limitation():
    """TC-154: Digits in the answer are flagged when the excerpt spells the number out. This documents
    the limit stated in the module: the check is a warning signal, not proof."""
    assert check("The stadium seats 70,000 fans.", ["seats seventy thousand fans"]) == ["70000"]
    assert numbers_in("a 1,200.50 fee") == ["1200.50"]
