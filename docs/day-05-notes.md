# Day 5: testing, hallucination reduction, evaluation

## The problem

A language model produces fluent text whether or not it is right. In a document Q&A system the
dangerous failures are quiet ones: a wrong number stated confidently, an answer to a question the
document never addressed, a citation that points nowhere. The goal today was to make those failures
**visible or impossible**, and to **measure** the system instead of judging it by a few good-looking answers.

## Defenses, in the order they act

| # | Defense | Where | Stops |
| --- | --- | --- | --- |
| 1 | Relevance cut-off (`min_score`) | `retriever.search`, `answerer` | Asking the model about passages that are not related; no model call at all |
| 2 | Prompt rules: answer only from the excerpts, use an exact not-found sentence, treat document text as untrusted | `prompt.py` | Answers from the model's memory; instructions planted in a PDF |
| 3 | Not-found recognition (tolerates quotes and case) | `answerer._says_not_found` | A refusal being shown as an answer with sources |
| 4 | Citation verification | `answerer` | `[7]` when only four excerpts exist; sources are built from verified citations only |
| 5 | Figure grounding (new) | `grounding.py` | Numbers that appear in no excerpt (for example "25 days" when the document says 18) |
| 6 | `--require-citations` (new, optional) | `answerer` | An answer that cites nothing is withheld |

Defenses 1 to 4 existed before; 5 and 6 are new today.

### Figure grounding

`unsupported_numbers(answer, question, excerpts, pages)` extracts every number from the answer and
reports those found nowhere in the excerpts, the question, or the page numbers of the excerpts.
It normalises commas, `$` and `%`, ignores `[1]` citations and list markers like `1.`, and accepts
a figure the user supplied in the question ("after 4 years..."). The result is a **warning**, not a
refusal: a correct calculation such as 18 + 5 = 23 is legitimate, so the user is told to check
instead of the answer being blocked.

**Known limit:** numbers written as words ("eighteen") are not compared with digits. The test
TC-154 documents this on purpose so the limit is visible, not hidden.

### Why a warning and not a block

Blocking has a cost: each wrongly blocked answer is a failure the user can see. Warnings keep
recall high and put the judgement with the reader, who has the source passage one click away.
`--require-citations` is the stricter choice for settings where an uncited answer is not acceptable;
small local models may fail it often, so it is off by default.

## Measuring: the evaluation harness

`python -m rag.evaluate` runs 17 fixed questions (`eval/handbook_questions.json`) against the example handbook.

- **Retrieval mode** (no model needed): for each answerable question, where does a chunk from the
  correct page rank? Reports hit@1, hit@k, mean reciprocal rank, and a `min_score` sweep that shows
  how many answerable questions survive a cut-off against how many unanswerable ones are blocked.
  It recommends the highest cut-off that keeps at least 90% of answerable questions (recall first:
  blocking a question the document can answer is worse than letting the model decline).
- **Answers mode** (needs a model): runs the whole pipeline and gives each reply a mechanical
  verdict: right page cited, expected facts present, forbidden text absent, unanswerable questions
  declined. A reply the document only partly covers (dental insurance) is marked `review` for a human.
  The command exits 1 if any case fails, so it can gate a build.

### Measured baseline (word-overlap embedder, no model)

These numbers were measured on this repository with `--embedder hashing`:

| Measure | Value |
| --- | --- |
| Answerable questions | 13 |
| Correct page ranked first (hit@1) | 69% |
| Correct page in the top 4 | 85% |
| Mean reciprocal rank | 0.79 |
| Multi-page questions with all pages found | 0 of 1 |

What the baseline shows:

- Questions that share words with the document do well; questions that paraphrase ("vacation days if I quit") do badly. That gap is what the neural embedder is for.
- The default `min_score` of 0.2 **would block 54% of answerable questions** with this embedder, because word overlap yields low scores even for right passages. The cut-off is specific to the embedder and must be tuned for it. The baseline's own recommendation (0.00) is not meaningful for MiniLM.

> **To update after running on a real setup:** run both commands from the README with `--embedder minilm`
> and put the numbers here. They have not been measured in this environment (no model download was
> possible), so no MiniLM or LLM numbers are claimed.

## Testing

| Layer | Count | Notes |
| --- | --- | --- |
| Python (`python -m pytest`) | 180 documented cases (TC-01 to TC-180) | Fake LLM, hand-written ASGI client, fake Ollama server over real HTTP |
| Frontend logic (`npm test`) | 14 | API client and citation parsing |
| Browser (`e2e/browser_check.py`) | 28 checks | Real browser against the real server with a stub LLM |
| Evaluation (`rag.evaluate`) | 17 questions | Measures quality rather than correctness of code |

Tests were also checked by **mutation**: breaking the code on purpose (for example removing the citation check)
and confirming a test fails. A test that cannot fail proves nothing.

## What we learned

1. **Search always returns something.** Nearest-neighbour search has no concept of "no answer", so the system needs an explicit cut-off, and the right value depends on the embedder.
2. **Trust the check, not the reply.** Citations, refusals and figures are verified in code after the model answers.
3. **A silent bug is worse than a crash.** Early header/footer removal wiped an entire PDF and reported success; now it removes at most the top and bottom two lines of a page and raises an error if nothing is left.
4. **Fakes make the logic testable, real runs make it believable.** Most behaviour is tested with fake models; the parts that touch real services are listed as unverified in the README.
5. **A browser test finds what unit tests cannot.** It caught two elements sharing one accessible label.
6. **Measure before tuning.** The evaluation harness turned "it seems to work" into numbers, and showed the default cut-off is wrong for the baseline embedder.

## Limits that remain

- Numbers written as words are not checked; tables and multi-column PDFs can extract badly; scanned PDFs are rejected.
- No conversation memory: each question stands alone.
- The checks are mechanical. A fluent but wrong answer that cites a real excerpt and uses its figures can still pass.
- Evaluation covers one fictional document; it shows the method, not general accuracy.
- `min_score` is untuned for MiniLM until the evaluation is run on a machine that can download it.
