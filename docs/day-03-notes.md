# Day 3 notes: prompts, LLM answers, source references

## Concepts

**Grounded generation.** The LLM is not asked the question directly. It is given the retrieved
excerpts and told to answer from them only. Retrieval decides *what the model may know*; the
prompt decides *how it must behave*.

**Prompt engineering for RAG.** The prompt in `prompt.py` does four things:

1. Numbers each excerpt (`<excerpt id="2" source="guide.pdf" page="3">`) and asks for `[n]`
   citations, so every claim can be traced to a file and page.
2. Gives the model an exact sentence to use when the excerpts lack the answer. "Not found" then
   becomes something code can detect, instead of a polite paraphrase.
3. Forbids outside knowledge, and asks for partial answers to say what is missing.
4. Declares excerpts to be untrusted data. A PDF can contain "ignore your instructions and..."
   (prompt injection). The prompt says to ignore such text, and excerpt text is HTML-escaped so a
   document cannot close the `<excerpt>` tag and pose as part of the prompt.

**Checking the answer instead of trusting it.** `Answerer` parses the `[n]` citations from the
reply, returns only the excerpts actually cited, ignores citations to excerpts that were never
provided, and attaches a warning when an answer cites nothing.

## What was built

| Module | Role |
| --- | --- |
| `prompt.py` | System prompt, `NOT_FOUND_MESSAGE`, excerpt formatting |
| `llm.py` | `LLM` interface and `ClaudeLLM` (Anthropic API; needs `ANTHROPIC_API_KEY`) |
| `answerer.py` | `Answerer.ask(question)` returns an `Answer` with text, cited `sources`, all `retrieved` excerpts, `warnings` |
| `ask.py` | Command line: `python -m rag.ask file.pdf "question"` |

## Design decisions

**No retrieval hit means no model call.** If every chunk is below `min_score`, the fixed
not-found message is returned without calling the LLM (`TC-87`). It is cheaper, and a model that
is never asked cannot invent an answer.

**The LLM is behind a one-method interface.** Tests use a fake, and a different provider only
needs a class with `generate(system, prompt)`.

**Sources are the cited excerpts, not everything retrieved.** Showing all `k` retrieved excerpts
as "sources" would credit pages the answer never used. Both lists are kept: `sources` for the
user interface, `retrieved` for debugging.

**Temperature and other sampling settings are left at the API defaults.** I could not check
which settings the model accepts without API access, so I did not pass any.

## Not verified yet

Built without network access or an API key, so:

- `ClaudeLLM` has only been run against a stand-in client. The request shape (`model`,
  `max_tokens`, `system`, `messages`) follows the Anthropic SDK's Messages API but has not been
  executed against the real service. If the first real call fails, send me the error.
- `DEFAULT_MIN_SCORE = 0.2` is a starting guess. It was not tuned with the neural embedder.
- No real model has yet been tested on answer quality, for example whether it follows the
  citation rule or the not-found rule reliably. Day 5 adds an evaluation set for this.

## Questions for Day 5

- Does the model ever answer from outside knowledge despite rule 1? Test with questions whose
  answers are well known but absent from the document.
- Which `min_score` and `k` give the best trade-off with the neural embedder?
- Should an uncited answer be rejected instead of returned with a warning?
