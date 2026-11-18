"""Tests for the evaluation harness (rag.evaluate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag import evaluate
from rag.answerer import Answer, Answerer, Source
from rag.embeddings import HashingEmbedder
from rag.evaluate import Case, RetrievalRow
from rag.prompt import NOT_FOUND_MESSAGE
from rag.retriever import Retriever

PAGES = [
    ["Chunking splits long documents into small overlapping passages", "so that each passage covers one topic."],
    ["Embeddings turn passages into vectors", "and vector databases find the nearest vectors quickly."],
    ["Football season begins in September.", "The stadium seats seventy thousand fans for each match."],
]
STADIUM = "How many fans does the stadium seat?"
FRANCE = "What is the capital of France?"


class ScriptedLLM:
    """Answers the stadium question, and says 'not found' to the France question."""

    def generate(self, system: str, prompt: str) -> str:
        if "France" in prompt:
            return NOT_FOUND_MESSAGE
        return "The stadium seats seventy thousand fans [1]."


def _case(**overrides) -> Case:
    base = dict(id=1, group="fact", question="q?", answerable=True, pages=(3,), expect=(("seventy",),))
    base.update(overrides)
    return Case(**base)


def _answer(text: str, pages: tuple[int, ...] = (3,), answered: bool = True) -> Answer:
    sources = [Source(n, "d.pdf", page, "t", 0.5) for n, page in enumerate(pages, start=1)]
    return Answer(text=text, answered=answered, sources=sources if answered else [])


def _write_dataset(path: Path, questions: list[dict], document: str | None = None) -> Path:
    data = {"questions": questions}
    if document is not None:
        data["document"] = document
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _questions() -> list[dict]:
    return [
        {"id": 1, "group": "fact", "question": STADIUM, "answerable": True, "pages": [3],
         "expect": [["seventy thousand"]]},
        {"id": 2, "group": "missing", "question": FRANCE, "answerable": False, "forbid": ["Paris"]},
    ]


@pytest.fixture
def pdf(make_pdf) -> Path:
    return make_pdf(PAGES, name="guide.pdf")


@pytest.fixture
def dataset(tmp_path) -> Path:
    return _write_dataset(tmp_path / "questions.json", _questions())


# --- dataset --------------------------------------------------------------------------


def test_load_cases_reads_fields(dataset):
    """TC-161: A valid dataset loads into cases with pages, expectations and flags."""
    document, cases = evaluate.load_cases(dataset)

    assert document is None
    assert [c.id for c in cases] == [1, 2]
    assert cases[0].pages == (3,) and cases[0].expect == (("seventy thousand",),)
    assert cases[1].answerable is False and cases[1].forbid == ("Paris",)


@pytest.mark.parametrize(
    "questions",
    [
        [{"id": 1, "group": "g", "question": "  ", "answerable": False}],
        [{"id": 1, "group": "g", "question": "q", "answerable": True}],
        [{"id": 1, "group": "g", "question": "a", "answerable": False},
         {"id": 1, "group": "g", "question": "b", "answerable": False}],
        [{"group": "g", "question": "q", "answerable": False}],
    ],
    ids=["blank question", "answerable without pages", "duplicate id", "missing id"],
)
def test_load_cases_rejects_malformed_questions(tmp_path, questions):
    """TC-162: Blank text, an answerable question with no pages, a repeated id or a missing field is rejected."""
    path = _write_dataset(tmp_path / "bad.json", questions)

    with pytest.raises(ValueError):
        evaluate.load_cases(path)


def test_load_cases_rejects_bad_files(tmp_path):
    """TC-163: A missing file, invalid JSON and JSON without 'questions' each raise ValueError."""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")

    for path in (tmp_path / "absent.json", broken, empty):
        with pytest.raises(ValueError):
            evaluate.load_cases(path)


def test_shipped_dataset_is_valid():
    """TC-164: The dataset shipped in eval/ loads, has 15+ cases and both answerable and unanswerable ones."""
    document, cases = evaluate.load_cases(evaluate.DEFAULT_DATASET)

    assert document == "examples/alderbrook-handbook.pdf"
    assert len(cases) >= 15
    assert any(c.answerable for c in cases) and any(not c.answerable for c in cases)


# --- judging --------------------------------------------------------------------------


def test_judge_passes_a_correct_cited_answer():
    """TC-165: Right page cited and expected text present gives 'pass'."""
    assert evaluate.judge(_case(), _answer("About SEVENTY thousand [1].")) == ("pass", [])


def test_judge_fails_wrong_page_missing_fact_and_forbidden_text():
    """TC-166: Wrong cited page, missing expected text and forbidden text are each reported."""
    case = _case(forbid=("HACKED",))

    verdict, reasons = evaluate.judge(case, _answer("HACKED", pages=(1,)))

    assert verdict == "fail"
    assert any("cited pages [1]" in r for r in reasons)
    assert any("missing any of" in r for r in reasons)
    assert any("forbidden" in r for r in reasons)


def test_judge_accepts_any_alternative_in_an_expect_group():
    """TC-167: An expect group is satisfied by any one of its alternatives."""
    case = _case(expect=(("6 weeks", "six weeks"),))

    assert evaluate.judge(case, _answer("Six weeks [1]."))[0] == "pass"


def test_judge_fails_not_found_on_an_answerable_question():
    """TC-168: Declining a question the document answers is a failure."""
    verdict, reasons = evaluate.judge(_case(), _answer(NOT_FOUND_MESSAGE, answered=False))

    assert verdict == "fail" and "not found" in reasons[0]


def test_judge_unanswerable_questions():
    """TC-169: Declining passes; answering fails, or is flagged for review when the case allows partial answers."""
    case = _case(answerable=False, pages=(), expect=())

    assert evaluate.judge(case, _answer(NOT_FOUND_MESSAGE, answered=False)) == ("pass", [])
    assert evaluate.judge(case, _answer("Paris [1]."))[0] == "fail"
    assert evaluate.judge(_case(answerable=False, pages=(), expect=(), review_if_answered=True), _answer("x [1]."))[0] == "review"
    forbidden = _case(answerable=False, pages=(), expect=(), review_if_answered=True, forbid=("Paris",))
    assert evaluate.judge(forbidden, _answer("Paris [1]."))[0] == "fail"


# --- retrieval metrics ----------------------------------------------------------------


def test_evaluate_retrieval_finds_the_expected_page(pdf):
    """TC-170: The question about the stadium ranks page 3 first; rows record rank and scores."""
    retriever = Retriever(HashingEmbedder())
    retriever.add_pdf(pdf)
    _, cases = evaluate.load_cases(_write_dataset(pdf.with_name("q.json"), _questions()))

    rows = evaluate.evaluate_retrieval(retriever, cases, k=3)

    assert rows[0].rank == 1 and rows[0].pages_found == (3,)
    assert rows[0].best_expected_score == rows[0].top_score > 0
    assert rows[1].rank is None and rows[1].best_expected_score is None


def test_retrieval_summary_computes_hit_rates_and_mrr():
    """TC-171: hit@1, hit@k, mean reciprocal rank and multi-page coverage come out as expected."""
    one_page = _case(pages=(1,))
    two_pages = _case(id=2, pages=(1, 2))
    rows = [
        RetrievalRow(one_page, 0.9, 1, 0.9, (1,)),
        RetrievalRow(_case(id=3, pages=(1,)), 0.5, 2, 0.4, (1,)),
        RetrievalRow(_case(id=4, pages=(1,)), 0.5, None, None, ()),
        RetrievalRow(two_pages, 0.7, 1, 0.7, (1,)),
        RetrievalRow(_case(id=5, answerable=False, pages=()), 0.1, None, None, ()),
    ]

    summary = evaluate.retrieval_summary(rows, k=4)

    assert summary["answerable_questions"] == 4
    assert summary["hit_at_1"] == pytest.approx(2 / 4)
    assert summary["hit_at_4"] == pytest.approx(3 / 4)
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert summary["multi_page_fully_covered"] == "0/1"


def test_recommend_min_score_is_recall_first():
    """TC-172: The recommended cut-off keeps 90% of answerable questions and blocks the weak unanswerable ones."""
    scores = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]
    rows = [RetrievalRow(_case(id=i), s, 1, s, (3,)) for i, s in enumerate(scores, start=1)]
    rows += [RetrievalRow(_case(id=20 + i, answerable=False, pages=()), s, None, None, ()) for i, s in enumerate([0.18, 0.10])]

    rec = evaluate.recommend_min_score(rows)

    assert rec["min_score"] == pytest.approx(0.20)
    assert rec["kept"] == pytest.approx(0.9)
    assert rec["blocked"] == pytest.approx(1.0)
    assert rec["table"][0]["min_score"] == 0 and rec["table"][0]["answerable_kept"] == 1.0


def test_recommend_min_score_without_answerable_questions():
    """TC-173: With no answerable questions there is nothing to recommend, and nothing crashes."""
    rows = [RetrievalRow(_case(answerable=False, pages=()), 0.3, None, None, ())]

    assert evaluate.recommend_min_score(rows)["min_score"] is None


# --- answers --------------------------------------------------------------------------


def test_evaluate_answers_runs_the_pipeline_and_summarises(pdf, dataset):
    """TC-174: With a scripted model both cases pass and the summary counts them per group."""
    retriever = Retriever(HashingEmbedder())
    retriever.add_pdf(pdf)
    _, cases = evaluate.load_cases(dataset)

    rows = evaluate.evaluate_answers(Answerer(retriever, ScriptedLLM(), k=3, min_score=None), cases)
    summary = evaluate.answer_summary(rows)

    assert [r.verdict for r in rows] == ["pass", "pass"]
    assert rows[0].cited_pages == [3]
    assert summary["pass"] == 2 and summary["fail"] == 0
    assert summary["by_group"] == {"fact": "1/1", "missing": "1/1"}


def test_evaluate_answers_catches_a_model_that_ignores_the_rules(pdf, dataset, fake_llm):
    """TC-175: A model that answers every question the same way fails the unanswerable case."""
    retriever = Retriever(HashingEmbedder())
    retriever.add_pdf(pdf)
    _, cases = evaluate.load_cases(dataset)
    llm = fake_llm("The stadium seats seventy thousand fans [1].")

    rows = evaluate.evaluate_answers(Answerer(retriever, llm, k=3, min_score=None), cases)

    assert [r.verdict for r in rows] == ["pass", "fail"]


# --- command line ---------------------------------------------------------------------


def test_cli_retrieval_mode_prints_and_writes_json(pdf, dataset, tmp_path, capsys):
    """TC-176: Retrieval mode needs no model, prints the table and recommendation, and exits 0."""
    out = tmp_path / "retrieval.json"

    code = evaluate.main(["--dataset", str(dataset), "--pdf", str(pdf), "--embedder", "hashing", "--json", str(out)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "hit@1" in printed and "Recommended min_score" in printed
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["settings"]["embedder"] and len(saved["rows"]) == 2


def test_cli_answers_mode_writes_json_and_markdown(pdf, dataset, tmp_path, monkeypatch, capsys):
    """TC-177: Answers mode prints verdicts, writes both reports, and exits 0 when nothing fails."""
    monkeypatch.setattr(evaluate, "make_llm", lambda *args: ScriptedLLM())
    as_json, as_md = tmp_path / "r.json", tmp_path / "r.md"

    code = evaluate.main([
        "--mode", "answers", "--dataset", str(dataset), "--pdf", str(pdf), "--embedder", "hashing",
        "--min-score", "0", "--json", str(as_json), "--markdown", str(as_md),
    ])

    assert code == 0
    assert "pass 2" in capsys.readouterr().out
    assert json.loads(as_json.read_text(encoding="utf-8"))["summary"]["pass"] == 2
    markdown = as_md.read_text(encoding="utf-8")
    assert markdown.startswith("# Evaluation report") and "| 1 | fact | pass" in markdown


def test_cli_answers_mode_exits_1_when_a_case_fails(pdf, dataset, monkeypatch, fake_llm, capsys):
    """TC-178: A failing case makes the command exit 1, so it can gate a build."""
    monkeypatch.setattr(evaluate, "make_llm", lambda *args: fake_llm("The stadium seats seventy thousand fans [1]."))

    code = evaluate.main(["--mode", "answers", "--dataset", str(dataset), "--pdf", str(pdf), "--embedder", "hashing", "--min-score", "0"])

    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_reports_errors_without_a_traceback(pdf, tmp_path, capsys):
    """TC-179: A missing dataset, a missing PDF and a dataset without a document each print 'Error:' and exit 1."""
    no_document = _write_dataset(tmp_path / "nodoc.json", _questions())

    assert evaluate.main(["--dataset", str(tmp_path / "absent.json"), "--embedder", "hashing"]) == 1
    assert evaluate.main(["--dataset", str(no_document), "--embedder", "hashing"]) == 1
    assert evaluate.main(["--dataset", str(no_document), "--pdf", str(tmp_path / "absent.pdf"), "--embedder", "hashing"]) == 1
    assert capsys.readouterr().err.count("Error:") == 3


def test_baseline_on_the_shipped_handbook_does_not_regress():
    """TC-180: The word-overlap baseline still finds the right page in the top 4 for most answerable questions."""
    pdf = evaluate.REPO_ROOT / "examples" / "alderbrook-handbook.pdf"
    if not pdf.is_file():
        pytest.skip("example handbook is not present")
    _, cases = evaluate.load_cases(evaluate.DEFAULT_DATASET)
    retriever = Retriever(HashingEmbedder())
    retriever.add_pdf(pdf)

    summary = evaluate.retrieval_summary(evaluate.evaluate_retrieval(retriever, cases, 4), 4)

    assert summary["hit_at_4"] >= 0.75
