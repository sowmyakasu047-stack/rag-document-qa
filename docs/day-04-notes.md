# Day 4 notes: web API and React interface

## Concepts

**Client and server.** The browser runs the React app (the *client*). It cannot read PDFs into the
vector index or call the LLM itself, so it sends HTTP requests to the Python program (the *server*),
which does that work and returns JSON. The two sides only agree on the request and response
shapes, listed in the README.

**Uploading a file.** A file travels as `multipart/form-data`: one request body with named parts,
here a part called `file`. The server must not trust anything in it: the file name can contain
directory parts (`../../x.pdf`), the size can be huge, and the content may not be a PDF at all.

**Why a proxy in development.** The React dev server runs on port 5173 and the Python server on
8000. A browser blocks calls between different origins unless the server opts in (CORS). Vite's
`server.proxy` forwards `/api` to Python, so the browser sees one origin and no CORS settings are
needed. In production-style use, Python serves the built React files itself, again one origin.

**State in React.** The interface keeps three pieces of state: the document list, the conversation,
and upload/ask progress. Every server call either updates that state or sets an error message, so
the screen always matches what the server reported.

## What was built

| Part | Role |
| --- | --- |
| `src/rag/api.py` | Starlette app: `/api/health`, `/api/documents` (list, upload, delete), `/api/ask` |
| `src/rag/server.py` | Starts the app with uvicorn; loads the saved index; serves `frontend/dist` |
| `src/rag/retriever.py` | Gained `list_documents()` and `remove_document()` |
| `frontend/src/api.js` | Server calls; turns failures into one `ApiError` with a user-safe message |
| `frontend/src/citations.js` | Splits `[1]` / `[1, 2]` markers out of an answer so they become links |
| `frontend/src/components/` | `DocumentsPanel` (upload, list, remove), `ChatPanel` (conversation, input), `Message` (answer, citation links, source cards) |
| `e2e/` | Optional browser check with a stub LLM (needs Playwright) |

## Design decisions

**Starlette instead of FastAPI.** Day 1's plan named FastAPI. FastAPI is Starlette plus automatic
validation and generated API docs. This project has five endpoints with hand-written validation, so
the extra layer buys little, and the build environment had Starlette and uvicorn but not FastAPI,
which meant FastAPI code could not have been run or tested. Moving to FastAPI later is mostly
changing route decorators.

**One error shape for every failure.** `{"error": {"code", "message"}}`. The `message` is written for
the user and shown as is; the `code` (`encrypted_pdf`, `no_text`, `llm_unavailable`, ...) lets the
interface or tests branch without parsing text.

**The server starts without an API key.** The LLM client is created on the first question, not at
startup. Uploading and indexing work regardless, `/api/health` reports `llm_ready: false` with the
reason, and the interface shows a banner instead of the server refusing to start.

**One lock around the index; the LLM call stays outside it.** Requests run on worker threads and the
vector index is not safe for concurrent changes, so `SafeRetriever` serialises uploads, deletes and
searches. The slow network call to the LLM is made after retrieval, outside the lock, so a pending
answer never blocks an upload (`TC-119` sends eight uploads at once).

**The index is saved after every change and reloaded on start.** Uploaded documents survive a restart
(`TC-122`, `TC-123`). Starting with a different embedder than the saved index is refused with a
clear message instead of silently returning wrong results (`TC-125`, `TC-126`).

**Uploaded PDFs are not kept.** Only the extracted passages are stored; the file is deleted after
indexing. Sources show the passage text and page, not the PDF itself.

**Upload safety.** File names are reduced to a base name and unsafe characters replaced; only
`.pdf` is accepted; size is limited to 20 MB, checked first from the declared length and again while
copying; a failed re-upload never removes the existing copy (Day 2, `TC-60`).

**Citations are only links when they match a returned source.** If the model writes `[7]` but only
three excerpts exist, the marker stays plain text instead of pointing at nothing.

**Plain JavaScript, not TypeScript.** The build environment could not type-check TypeScript against
React's types, so the code stays in JavaScript where every file could be run and tested.

## Local model option (Ollama)

Added after the first real run, because an API key is a barrier. `OllamaLLM` talks to Ollama's HTTP API
(`/api/tags` to check readiness, `/api/chat` to answer) using only Python's standard library, so there
is no new dependency. `make_llm("claude" | "ollama")` picks the provider; `--llm` selects it for
`rag.server` and `rag.ask`.

- **Readiness check with fix-it messages.** The constructor checks that Ollama answers and that the model is
  installed, and the error names the exact command (`ollama pull llama3.2`). Failures are not cached, so
  fixing the problem needs no restart.
- **Loopback requests bypass system proxies**, which would otherwise send `127.0.0.1` traffic to a proxy.
  The default address is `127.0.0.1`, not `localhost`, because on some Windows setups `localhost` resolves to
  IPv6 first while Ollama listens on IPv4.
- **Small models are less obedient.** The answer check now accepts the "not found" sentence in quotes or other
  case, and `<think>…</think>` blocks from reasoning models are stripped. Citations are still only trusted
  when they match a real excerpt.
- **Verified against a fake Ollama server over real HTTP** (`TC-129` to `TC-143`), not against Ollama itself.

## Observations from testing

- Driving the real app in a browser found a defect that unit tests did not: the user's message bubble
  and the question box were both labelled "Your question", so a screen reader would announce two
  controls with one name. The bubble is now labelled "You asked".
- Source rows looked like plain text because a CSS layout choice removed the browser's expand
  triangle; a chevron was added so it is clear they open.
- The browser check with a stub LLM passes 28 of 28 checks, including upload errors, citation
  links, reload persistence, removal, phone width and a network failure.

## Not verified yet

- **`npm install`, `vite` and `vite build` were never run** (no network access during the build). The
  same source files were bundled with esbuild and exercised in Chromium, but the Vite config and the
  dev-server proxy are unexecuted. If `npm run dev` or `npm run build` fails, send me the output.
- **Real Ollama** has not run here (not installed). The fake server follows Ollama's documented endpoints, but a
  version difference would only show up on your machine.
- **The real Claude API and the real MiniLM embedder** have still not run. The browser check used a
  stub LLM and the word-overlap embedder, so it proves the plumbing, not answer quality.
- The server has been tested on Linux only. File-name handling for Windows paths is unit tested, but
  the server itself has not run on Windows.

## Questions for Day 5

- Which `min_score` and `k` work with MiniLM and Claude? The interface shows each source's relevance
  score, which makes this easy to eyeball.
- Questions are answered independently; follow-up questions ("and what about page 3?") get no memory
  of the earlier turn. Is that acceptable for the demo?
- Should an answer with no valid citation be rejected rather than shown with a warning?
