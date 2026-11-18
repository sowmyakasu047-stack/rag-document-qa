"""Measure how well retrieval and answering work on a fixed set of questions.

Two modes, because they answer different questions and cost differently:

``retrieval``  needs no language model. It checks whether the passage that answers each
               question is among the top results, and recommends a ``min_score`` cut-off.
``answers``    runs the whole pipeline with a real model and judges each reply: right page
               cited, expected facts present, forbidden text absent, unanswerable questions
               declined.

::

    PYTHONPATH=src python -m rag.evaluate --mode retrieval --embedder minilm
    PYTHONPATH=src python -m rag.evaluate --mode answers --embedder minilm --llm ollama

The dataset format is described in ``eval/handbook_questions.json``.

The verdicts are mechanical checks on the text, not a human reading. A ``pass`` means the
reply satisfied the written expectations; ``review`` marks replies a person should read.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag.answerer import DEFAULT_K, DEFAULT_MIN_SCORE, Answer, Answerer
from rag.embeddings import DEFAULT_MODEL as EMBEDDING_MODEL
from rag.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from rag.errors import RagError
from rag.llm import make_llm
from rag.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "eval" / "handbook_questions.json"
SEARCH_DEPTH = 10
RECALL_TARGET = 0.9


@dataclass(frozen=True)
class Case:
    id: int
    group: str
    question: str
    answerable: bool
    pages: tuple[int, ...] = ()
    expect: tuple[tuple[str, ...], ...] = ()
    forbid: tuple[str, ...] = ()
    review_if_answered: bool = False


def load_cases(path: str | Path) -> tuple[str | None, list[Case]]:
    """Read a dataset file; return ``(document_path_or_None, cases)``.

    Raises:
        ValueError: the file is not valid JSON or a question is malformed.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_cases = data["questions"]
    except FileNotFoundError as exc:
        raise ValueError(f"Dataset not found: {path}") from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Dataset '{path}' is not valid: {exc!r}") from exc

    cases: list[Case] = []
    seen: set[int] = set()
    for position, item in enumerate(raw_cases, start=1):
        try:
            case = Case(
                id=int(item["id"]),
                group=str(item["group"]),
                question=str(item["question"]).strip(),
                answerable=bool(item["answerable"]),
                pages=tuple(int(p) for p in item.get("pages", [])),
                expect=tuple(tuple(str(a) for a in group) for group in item.get("expect", [])),
                forbid=tuple(str(f) for f in item.get("forbid", [])),
                review_if_answered=bool(item.get("review_if_answered", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Question #{position} in '{path}' is malformed: {exc!r}") from exc
        if not case.question:
            raise ValueError(f"Question #{position} in '{path}' has no text.")
        if case.answerable and not case.pages:
            raise ValueError(f"Question {case.id} is answerable but lists no pages.")
        if case.id in seen:
            raise ValueError(f"Question id {case.id} appears twice.")
        seen.add(case.id)
        cases.append(case)
    return data.get("document"), cases


# --- retrieval ------------------------------------------------------------------------


@dataclass
class RetrievalRow:
    case: Case
    top_score: float
    rank: int | None  # 1-based position of the first chunk from an expected page
    best_expected_score: float | None
    pages_found: tuple[int, ...]  # expected pages that appear within the top k


def evaluate_retrieval(retriever: Retriever, cases: list[Case], k: int = DEFAULT_K) -> list[RetrievalRow]:
    rows = []
    for case in cases:
        results = retriever.search(case.question, k=max(k, SEARCH_DEPTH))
        top = results[0].score if results else 0.0
        expected = [r for r in results if r.chunk.page in case.pages]
        rank = next((i for i, r in enumerate(results, start=1) if r.chunk.page in case.pages), None)
        found = tuple(sorted({r.chunk.page for r in results[:k]} & set(case.pages)))
        rows.append(RetrievalRow(case, top, rank, expected[0].score if expected else None, found))
    return rows


def retrieval_summary(rows: list[RetrievalRow], k: int) -> dict:
    answerable = [r for r in rows if r.case.answerable]
    multi = [r for r in answerable if len(r.case.pages) > 1]
    return {
        "answerable_questions": len(answerable),
        "hit_at_1": sum(r.rank == 1 for r in answerable) / len(answerable) if answerable else None,
        f"hit_at_{k}": sum(r.rank is not None and r.rank <= k for r in answerable) / len(answerable) if answerable else None,
        "mrr": sum(1 / r.rank for r in answerable if r.rank) / len(answerable) if answerable else None,
        "multi_page_fully_covered": f"{sum(len(r.pages_found) == len(r.case.pages) for r in multi)}/{len(multi)}",
    }


def recommend_min_score(rows: list[RetrievalRow]) -> dict:
    """Suggest the highest ``min_score`` that still lets the right passage through.

    The cut-off is recall-first: it should almost never block a question the document can
    answer (the model and the other guards handle the rest), and only then block as many
    unanswerable questions as that allows. The result is a guide, not a guarantee: with a
    small question set the number is only as good as the questions.
    """
    answerable = [r for r in rows if r.case.answerable]
    unanswerable = [r for r in rows if not r.case.answerable]
    if not answerable:
        return {"min_score": None, "kept": None, "blocked": None, "table": []}

    def kept(t: float) -> float:
        return sum(r.best_expected_score is not None and r.best_expected_score >= t for r in answerable) / len(answerable)

    def blocked(t: float) -> float | None:
        return sum(r.top_score < t for r in unanswerable) / len(unanswerable) if unanswerable else None

    target = min(RECALL_TARGET, kept(0.0))
    thresholds = [i / 100 for i in range(0, 101)]
    best = max(t for t in thresholds if kept(t) >= target)
    top = max([r.top_score for r in rows] + [0.0])
    table = [
        {"min_score": t, "answerable_kept": kept(t), "unanswerable_blocked": blocked(t)}
        for t in [i / 20 for i in range(0, int(top * 20) + 2)]
    ]
    return {"min_score": best, "kept": kept(best), "blocked": blocked(best), "table": table}


# --- answers --------------------------------------------------------------------------


@dataclass
class AnswerRow:
    case: Case
    verdict: str  # "pass", "fail" or "review"
    reasons: list[str]
    answered: bool
    text: str
    cited_pages: list[int]
    warnings: list[str] = field(default_factory=list)


def _contains(text: str, needle: str) -> bool:
    return needle.casefold() in text.casefold()


def judge(case: Case, answer: Answer) -> tuple[str, list[str]]:
    """Decide whether ``answer`` meets the expectations in ``case``."""
    reasons: list[str] = []
    forbidden = [f for f in case.forbid if _contains(answer.text, f)]

    if not case.answerable:
        if not answer.answered:
            return "pass", []
        if forbidden:
            return "fail", [f"contains forbidden text {f!r}" for f in forbidden]
        if case.review_if_answered:
            return "review", ["answered a question the document only partly covers; check for invented facts"]
        return "fail", ["answered a question the document cannot answer"]

    if not answer.answered:
        return "fail", ["said 'not found' although the document contains the answer"]
    cited = sorted({s.page for s in answer.sources})
    if not cited:
        reasons.append("cited no excerpt")
    elif not set(cited) & set(case.pages):
        reasons.append(f"cited pages {cited}, expected one of {list(case.pages)}")
    for group in case.expect:
        if not any(_contains(answer.text, alternative) for alternative in group):
            reasons.append(f"missing any of {list(group)}")
    reasons.extend(f"contains forbidden text {f!r}" for f in forbidden)
    return ("fail" if reasons else "pass"), reasons


def evaluate_answers(answerer: Answerer, cases: list[Case]) -> list[AnswerRow]:
    rows = []
    for case in cases:
        answer = answerer.ask(case.question)
        verdict, reasons = judge(case, answer)
        rows.append(
            AnswerRow(case, verdict, reasons, answer.answered, answer.text,
                      sorted({s.page for s in answer.sources}), list(answer.warnings))
        )
    return rows


def answer_summary(rows: list[AnswerRow]) -> dict:
    def rate(selected: list[AnswerRow]) -> str:
        return f"{sum(r.verdict == 'pass' for r in selected)}/{len(selected)}"

    groups = sorted({r.case.group for r in rows})
    return {
        "pass": sum(r.verdict == "pass" for r in rows),
        "fail": sum(r.verdict == "fail" for r in rows),
        "review": sum(r.verdict == "review" for r in rows),
        "with_warnings": sum(bool(r.warnings) for r in rows),
        "by_group": {g: rate([r for r in rows if r.case.group == g]) for g in groups},
    }


# --- reporting ------------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_retrieval(rows: list[RetrievalRow], k: int) -> str:
    summary = retrieval_summary(rows, k)
    rec = recommend_min_score(rows)
    lines = [f"{'id':>3}  {'group':<16} {'rank':>4} {'best match':>10} {'top score':>9}  question"]
    for r in rows:
        rank = "-" if r.rank is None else str(r.rank)
        best = "-" if r.best_expected_score is None else f"{r.best_expected_score:.2f}"
        lines.append(f"{r.case.id:>3}  {r.case.group:<16} {rank:>4} {best:>10} {r.top_score:>9.2f}  {r.case.question}")
    lines += [
        "",
        "rank = position of the first passage from a correct page; 'best match' = its score;",
        "'top score' = score of the best passage overall (for unanswerable questions, lower is better).",
        "",
        f"Answerable questions: {summary['answerable_questions']}",
        f"  correct page ranked first (hit@1): {_percent(summary['hit_at_1'])}",
        f"  correct page in the top {k} (hit@{k}): {_percent(summary[f'hit_at_{k}'])}",
        f"  mean reciprocal rank: {summary['mrr']:.2f}" if summary["mrr"] is not None else "  mean reciprocal rank: n/a",
        f"  questions needing several pages, all found in the top {k}: {summary['multi_page_fully_covered']}",
        "",
        "min_score sweep (share of answerable questions whose right passage survives / share of",
        "unanswerable questions whose best passage is cut off):",
    ]
    for entry in rec["table"]:
        lines.append(
            f"  min_score {entry['min_score']:.2f}:  kept {_percent(entry['answerable_kept']):>4}   "
            f"blocked {_percent(entry['unanswerable_blocked']):>4}"
        )
    if rec["min_score"] is not None:
        lines += [
            "",
            f"Recommended min_score: {rec['min_score']:.2f} "
            f"(keeps {_percent(rec['kept'])} of answerable questions, blocks {_percent(rec['blocked'])} of unanswerable ones).",
            "Pass it with --min-score when starting the server.",
        ]
    return "\n".join(lines)


def format_answers(rows: list[AnswerRow], retrieval: list[RetrievalRow]) -> str:
    scores = {r.case.id: r.top_score for r in retrieval}
    lines = [f"{'id':>3}  {'group':<16} {'verdict':<7} {'score':>5} {'cited':<8}  question / reasons"]
    for r in rows:
        cited = ",".join(map(str, r.cited_pages)) or "-"
        lines.append(f"{r.case.id:>3}  {r.case.group:<16} {r.verdict.upper():<7} {scores.get(r.case.id, 0):>5.2f} {cited:<8}  {r.case.question}")
        for reason in r.reasons:
            lines.append(f"{'':>45}-> {reason}")
        for warning in r.warnings:
            lines.append(f"{'':>45}!  {warning}")
    summary = answer_summary(rows)
    lines += [
        "",
        f"pass {summary['pass']}   fail {summary['fail']}   review {summary['review']}   "
        f"answers with warnings {summary['with_warnings']}   (of {len(rows)})",
        "by group: " + ", ".join(f"{g} {v}" for g, v in summary["by_group"].items()),
        "",
        "A 'review' reply needs a human read. 'score' is the best passage's relevance.",
    ]
    return "\n".join(lines)


def format_markdown(rows: list[AnswerRow], retrieval: list[RetrievalRow], settings: dict) -> str:
    scores = {r.case.id: r.top_score for r in retrieval}
    summary = answer_summary(rows)
    head = ", ".join(f"{key}: {value}" for key, value in settings.items())
    out = [
        "# Evaluation report", "", f"Settings: {head}", "",
        f"**Pass {summary['pass']}, fail {summary['fail']}, review {summary['review']}** of {len(rows)} questions.",
        "", "| # | Group | Verdict | Score | Cited pages | Question | Notes |", "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        notes = "; ".join(r.reasons + r.warnings).replace("|", "/")
        cited = ", ".join(map(str, r.cited_pages)) or "-"
        out.append(f"| {r.case.id} | {r.case.group} | {r.verdict} | {scores.get(r.case.id, 0):.2f} | {cited} | {r.case.question} | {notes} |")
    out += ["", "By group: " + ", ".join(f"{g} {v}" for g, v in summary["by_group"].items()), ""]
    return "\n".join(out)


# --- command line ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers on a question set.")
    parser.add_argument("--mode", choices=["retrieval", "answers"], default="retrieval")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pdf", type=Path, help="Document to index (default: the dataset's 'document')")
    parser.add_argument("--embedder", choices=["hashing", "minilm"], default="minilm")
    parser.add_argument("--llm", choices=["claude", "ollama"], default="claude")
    parser.add_argument("--model")
    parser.add_argument("--ollama-url")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--require-citations", action="store_true")
    parser.add_argument("--json", type=Path, help="Also write the results to this JSON file")
    parser.add_argument("--markdown", type=Path, help="Also write a Markdown report (answers mode)")
    args = parser.parse_args(argv)

    try:
        document, cases = load_cases(args.dataset)
        pdf = args.pdf or (REPO_ROOT / document if document else None)
        if pdf is None:
            raise ValueError("No document given: use --pdf or set 'document' in the dataset.")
        embedder = HashingEmbedder() if args.embedder == "hashing" else SentenceTransformerEmbedder(EMBEDDING_MODEL)
        retriever = Retriever(embedder)
        retriever.add_pdf(pdf)
        retrieval = evaluate_retrieval(retriever, cases, args.k)
        settings = {"document": Path(pdf).name, "embedder": embedder.name, "k": args.k, "min_score": args.min_score}

        if args.mode == "retrieval":
            print(format_retrieval(retrieval, args.k))
            if args.json:
                args.json.write_text(json.dumps({
                    "settings": settings, "summary": retrieval_summary(retrieval, args.k),
                    "recommendation": recommend_min_score(retrieval),
                    "rows": [{"id": r.case.id, "rank": r.rank, "top_score": r.top_score,
                              "best_expected_score": r.best_expected_score} for r in retrieval]}, indent=2), encoding="utf-8")
            return 0

        llm = make_llm(args.llm, args.model, args.ollama_url)
        settings.update({"llm": f"{args.llm}:{args.model or 'default'}", "require_citations": args.require_citations})
        answerer = Answerer(retriever, llm, k=args.k, min_score=args.min_score, require_citations=args.require_citations)
        rows = evaluate_answers(answerer, cases)
    except (RagError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_answers(rows, retrieval))
    if args.json:
        args.json.write_text(json.dumps({
            "settings": settings, "summary": answer_summary(rows),
            "rows": [{**{k: v for k, v in asdict(r).items() if k != "case"}, "id": r.case.id, "question": r.case.question}
                     for r in rows]}, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(format_markdown(rows, retrieval, settings), encoding="utf-8")
    return 1 if any(r.verdict == "fail" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
