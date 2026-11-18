# Error handling

Every failure the application can foresee ends in a readable message and a state that is still
usable. Nothing is added to the index by a failed upload, and no failure needs a restart.

All HTTP errors use one shape: `{"error": {"code": "...", "message": "..."}}`. The interface shows
the `message`; code and tests rely on the `code`.

## What the user sees

| Situation | Code (HTTP) | Message shown | State afterwards |
| --- | --- | --- | --- |
| No file chosen | `missing_file` (400) | Choose a PDF file to upload. | Unchanged |
| File is not named `*.pdf` | `not_a_pdf` (400) | Only .pdf files can be uploaded. | Unchanged |
| Upload body cannot be read | `invalid_upload` (400) | The upload could not be read: ... | Unchanged |
| Corrupt "PDF" | `invalid_pdf` (400) | Reason from the loader | Nothing indexed |
| Password-protected PDF | `encrypted_pdf` (400) | This PDF is password protected. Remove the password and try again. | Nothing indexed |
| Scanned PDF, no text layer (or text removed entirely by cleaning) | `no_text` (422) | Reason from the loader | Nothing indexed |
| File over 20 MB | `file_too_large` (413) | The PDF is larger than the 20 MB limit. | Nothing indexed |
| Embedding model cannot be loaded | `embedding_unavailable` (503) | Reason from the embedder | Index unchanged |
| Delete a document that is not indexed | `not_found` (404) | No document named '...' is indexed. | Unchanged |
| Question body is not JSON, or not an object | `invalid_json` (400) | The request body must be JSON. | Unchanged |
| Blank question | `empty_question` (400) | Type a question first. | Unchanged |
| Question over 2000 characters | `question_too_long` (400) | Questions can be at most 2000 characters. | Unchanged |
| `k` not a whole number from 1 to 10 | `invalid_k` (400) | k must be a whole number from 1 to 10. | Unchanged |
| Ask before any upload | `no_documents` (409) | Upload a PDF before asking questions. | Unchanged |
| No API key, or Ollama not running or model missing | `llm_unavailable` (503) | What is missing and the fix (for example `ollama pull llama3.2`) | Works again as soon as it is fixed, no restart |
| Model call fails (rate limit, HTTP error, empty or unreadable reply) | `llm_error` (502) | Reason from the provider | Unchanged |
| Anything unforeseen | `internal_error` (500) | Something went wrong on the server. | Details are logged, none are leaked |

Startup errors (command line) print `Error: ...` and exit with code 1 instead of a traceback:
a saved index built with a different embedder (`EmbedderMismatchError`) and a damaged saved index
(`VectorStoreError`).

## Answer-level checks (not errors, but warnings or withheld answers)

These happen after the model has replied. They are covered in `docs/day-05-notes.md`.

| Situation | What the user sees |
| --- | --- |
| Nothing in the document scores above `min_score` | The fixed not-found sentence, no model call, no sources |
| The model replies with the not-found sentence (even in quotes) | The fixed not-found sentence, no sources |
| The answer cites no excerpt | Answer shown with a warning (or withheld with `--require-citations`) |
| The answer cites `[n]` for an excerpt that was not provided | The bad citation is dropped; a warning names it |
| The answer contains a figure found in no excerpt | Warning listing the figures |

## Design rules followed

- **Validate at the edge.** Size, type, JSON shape, question length and `k` are checked before any work is done.
- **Fail without side effects.** A rejected upload never reaches the index; the index is saved only after a successful change.
- **Say what to do next.** Messages for fixable problems (password, API key, missing Ollama model) name the fix.
- **Never leak internals.** Unexpected exceptions return a generic message; the traceback goes to the server log.
- **Lazy model creation.** The server starts without an API key so documents can still be uploaded and managed.
