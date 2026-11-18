# Test cases

Run all with `python -m pytest`. Every case has an ID that also appears in the docstring of
the matching test, so a failure can be traced back to this table.

Day 1 total: **30 test cases** (34 pytest cases; TC-22 is parametrized over 5 inputs). Day 2 cases are at the end of this file.

## PDF loading: `tests/test_loader.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-01 | Two-page PDF with text on each page | Two pages returned, numbered 1 and 2 |
| TC-02 | PDF with a blank page between two text pages | Blank page skipped; remaining pages keep numbers 1 and 3 |
| TC-03 | PDF named `annual-report.pdf` | `source` is the file name `annual-report.pdf` |
| TC-04 | Path that does not exist | `FileNotFoundError` |
| TC-05 | File named `notes.txt` | `InvalidPDFError` |
| TC-06 | File named `broken.pdf` containing random bytes | `InvalidPDFError` |
| TC-07 | PDF where no page has text (like a scan) | `NoExtractableTextError` |
| TC-08 | Password-protected PDF | `EncryptedPDFError` |

## Text cleaning: `tests/test_preprocess.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-09 | `"exam-\nple"` (word hyphenated at line end) | `"example"` |
| TC-10 | Lines wrapped only by page width | Joined into one line with single spaces |
| TC-11 | Paragraphs separated by several blank lines | Exactly one blank line between paragraphs |
| TC-12 | Ligature characters `ﬁ` and `ﬂ` | Plain `fi` and `fl` |
| TC-13 | Control characters, repeated spaces and tabs | Control characters removed, whitespace collapsed |
| TC-14 | Four pages with the same header and a numbered footer ("Page 1", "Page 2", ...) | Header and footer lines removed; body text kept; page numbers 1 to 4 preserved |
| TC-15 | Document with two pages sharing a first line | Returned unchanged (too few pages to judge) |
| TC-29 | Line repeated in the middle of every page | Kept, because it is body text and not at a page edge |

## Chunking: `tests/test_chunker.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-16 | Page shorter than `chunk_size` | One chunk with the full text |
| TC-17 | Long page, `chunk_size=200` | Several chunks, none longer than 200 characters |
| TC-18 | Long page, `chunk_size=200`, `overlap=60` | Start of each chunk repeats the end of the previous chunk |
| TC-19 | Two pages (2 and 5) with different content | Each chunk has the right `source` and `page`; no chunk mixes pages |
| TC-20 | Two pages of text | Unique `chunk_id` values, sequential `index`, first id is `report-p1-c0` |
| TC-21 | Long page chunked with overlap | Every word of the original appears in some chunk |
| TC-22 | `chunk_size` 0 or negative; `overlap` negative, equal to or larger than `chunk_size` | `ValueError` for each of the 5 inputs |
| TC-23 | 50 characters with no spaces, `chunk_size=20` | Hard-split; no chunk over 20 characters; no characters lost |
| TC-24 | Empty list of pages | Empty list of chunks |

## End-to-end ingestion: `tests/test_ingest.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-25 | Generated two-page PDF | Chunks tagged `sample.pdf` with pages 1 and 2 |
| TC-26 | Three pages sharing the header "CONFIDENTIAL DRAFT" | Header absent from every chunk; body text present |
| TC-27 | CLI run on a valid PDF | Exit code 0 and a "1 chunks from 1 page(s)" summary |
| TC-28 | CLI run on a missing file | Exit code 1 and an `Error:` message on stderr |
| TC-30 | Three pages that each contain only the same line | `NoExtractableTextError` instead of an empty result |

# Day 2 test cases

Day 2 adds **41 test cases** (TC-31 to TC-71), bringing the total to **71**. The FAISS cases
(TC-69 to TC-71) are skipped automatically when `faiss-cpu` is not installed.

## Embedders: `tests/test_embeddings.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-31 | Embed two texts with `HashingEmbedder(64)` | Shape (2, 64), float32, every row has length 1 |
| TC-32 | Embed the same text twice | Identical vectors (deterministic) |
| TC-33 | A question, a passage sharing its words, and an unrelated sentence | Question is closer to the related passage |
| TC-34 | `"Hello, World!"` vs `"hello world"` | Identical vectors |
| TC-35 | Text made only of ignored words and symbols | All-zero vector with no NaN values |
| TC-36 | Embed an empty list | Shape (0, dimension) |
| TC-37 | `HashingEmbedder(dimension=0)` | `ValueError` |
| TC-38 | Neural wrapper with a stand-in model returning `[3, 4, 0]` | Texts and batch size passed through; output normalised to `[0.6, 0.8, 0]` |
| TC-39 | `sentence_transformers` not installed | `EmbeddingModelUnavailableError` mentioning `pip install` |

## Vector store: `tests/test_vectorstore.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-40 | Query `[0.1, 0.9, 0]` against x, y, z axis vectors | y ranks first; scores descending; top score about 0.994 |
| TC-41 | `k=1`, and `k=50` with 3 chunks stored | 1 result, and 3 results |
| TC-42 | `min_score=0.5` on an axis query; `min_score=0.9` on `[1, 1, 1]` | Only the matching chunk; empty list |
| TC-43 | Query `[10, 0, 0]` | Score 1.0 (direction matters, not length) |
| TC-44 | Search an empty store; search with an all-zero query | Empty list both times |
| TC-45 | `k=0` | `ValueError` |
| TC-46 | Vectors of wrong dimension on add and on search | `ValueError` both times |
| TC-47 | Two chunks with one vector | `ValueError`; nothing stored |
| TC-48 | A vector containing NaN | `ValueError` |
| TC-49 | Add a chunk id that already exists | `ValueError` naming the id; store unchanged |
| TC-50 | `remove_source("a.pdf")` with chunks from two files | Returns 2; only the other file's chunk remains and is still searchable; unknown file returns 0 |
| TC-51 | Save, then load | Same chunks, metadata and search results |
| TC-52 | Load from a folder with no index | `VectorStoreError` |
| TC-53 | `chunks.json` lists fewer chunks than there are vectors | `VectorStoreError` ("inconsistent") |
| TC-54 | `meta.json` contains invalid JSON | `VectorStoreError` ("damaged") |
| TC-55 | Load `vectors.npy` with `allow_pickle=False` | Loads as a (3, 3) numeric array |

## Retriever and query tool: `tests/test_retriever.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-56 | Three-topic PDF; question about stadium seating | Best chunk is from `guide.pdf` page 3 |
| TC-57 | Questions about chunking and about vector databases | Best chunks are pages 1 and 2 |
| TC-58 | `add_pdf` on a 3-page PDF | Returns 3; store holds 3 chunks |
| TC-59 | Index the same file name twice | Still 3 chunks, no duplicates |
| TC-60 | Re-upload of an existing file name that is corrupt | `InvalidPDFError`; the old 3 chunks remain |
| TC-61 | Index a second PDF | Questions retrieve chunks from the right file |
| TC-62 | Whitespace-only question | `ValueError` |
| TC-63 | Question sharing no words with the document, `min_score=0.2` | Empty list |
| TC-64 | Save and reload a retriever | Same search results |
| TC-65 | Load an index with an embedder of a different size or name | `EmbedderMismatchError` |
| TC-66 | Embedder of dimension 64 with a store of dimension 32 | `ValueError` |
| TC-67 | Query CLI on a valid PDF | Exit 0; reports "Indexed 3 chunks"; best passage listed first with its page |
| TC-68 | Query CLI on a missing file | Exit 1 and an `Error:` message on stderr |

## FAISS store: `tests/test_faiss_store.py` (needs `faiss-cpu`)

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-69 | Same data and query in the FAISS and NumPy stores | Same chunks in the same order; scores equal within 1e-5 |
| TC-70 | `remove_source` and `k` larger than the store | Removal count correct; all remaining chunks returned |
| TC-71 | Save and reload a FAISS store | Same metadata and search results |

# Day 3 test cases

Day 3 adds **21 test cases** (TC-72 to TC-92), bringing the total to **92**. They use a fake LLM,
so they need no network and no API key.

## Prompt building: `tests/test_prompt.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-72 | Two retrieved chunks | Numbered excerpts that carry their file name and page |
| TC-73 | Question with surrounding spaces | Trimmed question placed after the excerpts block |
| TC-74 | Document text containing `</excerpt>` and a fake `<excerpt>` | Escaped; exactly one real closing tag remains |
| TC-75 | File name containing a quote | Escaped inside the `source` attribute |
| TC-76 | System prompt content | Has the not-found sentence, citation rule, "untrusted" rule and no-outside-knowledge rule |

## Claude client: `tests/test_llm.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-77 | `generate("SYS", "PROMPT")` with a reply of two text blocks and one other block | Request has model, max tokens, system and user message; text blocks are joined |
| TC-78 | The API library raises a connection error | `LLMError` containing the cause |
| TC-79 | Reply with no text | `LLMError` |
| TC-80 | `anthropic` package not installed | `LLMUnavailableError` mentioning `pip install anthropic` |
| TC-81 | No `ANTHROPIC_API_KEY` | `LLMUnavailableError` mentioning the variable |

## Answering: `tests/test_answerer.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-82 | Reply cites `[1]` | `sources` lists only excerpt 1 with file and page 3; no warnings |
| TC-83 | Reply uses `[1, 2]` and repeats `[2]` | Sources 1 and 2, once each |
| TC-84 | Reply cites `[9]` with 3 excerpts | `[9]` ignored; warning added |
| TC-85 | Reply cites nothing | Answer returned, no sources, "does not cite" warning |
| TC-86 | Model replies with the not-found sentence | `answered` is False; no sources |
| TC-87 | All chunks below `min_score` | Not-found message; the LLM is never called |
| TC-88 | Blank question | `ValueError`; LLM not called |
| TC-89 | LLM raises `LLMError` | Error reaches the caller |
| TC-90 | `k=2` | Prompt holds exactly 2 excerpts, the question and the system rules |
| TC-91 | `ask` command with a fake LLM | Exit 0; prints the answer and a Sources list with file and page |
| TC-92 | `ask` command without an API key | Exit 1; `Error:` on stderr |

# Day 4 test cases

Day 4 adds **36 Python test cases** (TC-93 to TC-128, 128 in total), **14 frontend test cases**
(FE-01 to FE-14) and a **28-check browser run**. The Python tests call the application through its
full stack with real multipart requests, using a fake LLM, so they need no network and no API key.

## HTTP API: `tests/test_api.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-93 | `GET /api/health` | `status: ok`, document count, `llm_ready: true` |
| TC-94 | Health check when no API key is set | `llm_ready: false` with the reason; server still responds |
| TC-95 | `GET /api/documents` before any upload | Empty list |
| TC-96 | Upload a valid 3-page PDF | 201 with `name`, `chunks: 3`, `replaced: false`; then listed with 3 pages |
| TC-97 | Upload the same file name twice | `replaced: true`; one document in the list |
| TC-98 | Upload `notes.txt` | 400 `not_a_pdf` |
| TC-99 | Upload a `.pdf` containing garbage | 400 `invalid_pdf` with a message naming the file |
| TC-100 | Upload a PDF with no text layer | 422 `no_text` |
| TC-101 | Upload a password-protected PDF | 400 `encrypted_pdf` |
| TC-102 | Multipart request without a `file` field | 400 `missing_file` |
| TC-103 | JSON body sent to the upload endpoint | 400 `missing_file` (no server error) |
| TC-104 | File slightly over the size limit | 413 `file_too_large`; nothing indexed |
| TC-105 | Request whose declared size is far over the limit | 413 `file_too_large`, refused up front |
| TC-106 | Names `../../evil.pdf` and `C:\docs\report.pdf` | Stored as `evil.pdf` and `report.pdf` |
| TC-107 | `safe_filename` on 9 inputs (paths, unsafe characters, non-PDF, empty, `..`, 300 characters) | Safe base name, or refused |
| TC-108 | `DELETE` an indexed document (name containing a space) | 204; list is empty |
| TC-109 | `DELETE` an unknown name | 404 `not_found` |
| TC-110 | Ask a question the document answers | 200; answer text; `answered: true`; sources with number, file, page, text, score; no warnings |
| TC-111 | Ask about something not in the document | `answered: false`, not-found message, no sources, LLM never called |
| TC-112 | Ask with nothing uploaded | 409 `no_documents` |
| TC-113 | Ask with empty, blank, non-text, missing, non-object or unparsable body (6 cases) | 400 for each; LLM never called |
| TC-114 | Question over 2000 characters | 400 `question_too_long` |
| TC-115 | `k` of 0, 11, -1, `"3"`, 2.5, `true`, `null` | 400 `invalid_k` for each |
| TC-116 | Ask when no API key is configured | 503 `llm_unavailable` with the reason |
| TC-117 | Ask when the model call fails | 502 `llm_error` |
| TC-118 | Upload when no API key is configured | Still 201 |
| TC-119 | Eight uploads sent at the same time | All 201; all eight listed once with 3 chunks each |

## Document listing: `tests/test_retriever.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-120 | Two indexed files | Name, chunk count and page count for each, first-added first |
| TC-121 | `remove_document` on one of two files, then on a missing file | Returns 3 then 0; the other file remains |

## Server start-up: `tests/test_server.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-122 | Upload, then build a new app on the same index folder | Document still listed and answerable |
| TC-123 | Delete a document, then restart | Document stays deleted |
| TC-124 | Start with a folder that does not exist | Empty list; folder and `meta.json` created on first upload |
| TC-125 | Start with a different embedder than the saved index | `EmbedderMismatchError` |
| TC-126 | `python -m rag.server` with a mismatched index | Exit 1 with "Rebuild the index" on stderr |
| TC-127 | Built frontend folder exists | `/` serves its `index.html`; API still works |
| TC-128 | No frontend folder | `/` returns 404; API still works |

## Frontend logic: `frontend/src/*.test.js` (run with `npm test`)

| ID | Scenario | Expected result |
| --- | --- | --- |
| FE-01 | `listDocuments` | Returns the `documents` array |
| FE-02 | `getHealth` | Returns the health object |
| FE-03 | `uploadDocument(file)` | POST of multipart form data with the file under `file` |
| FE-04 | `deleteDocument("my report #1.pdf")` | URL-encoded path; resolves on 204 |
| FE-05 | `askQuestion` with and without `k` | JSON body; `k` only included when given |
| FE-06 | Server error response | `ApiError` carrying the server's message, code and status |
| FE-07 | Error response that is not JSON | Generic "Request failed (502)." message |
| FE-08 | Network failure | `ApiError` "Could not reach the server" with code `network` |
| FE-09 | Answer without citations | One text segment |
| FE-10 | `Fans [1] sit.` | Text, citation 1, text |
| FE-11 | `[1, 2,3]` | One citation with numbers 1, 2, 3 |
| FE-12 | `Claim [1][2].` | Two separate citations |
| FE-13 | `[note]` and `[a, b]` | Left as plain text |
| FE-14 | Empty string | No segments |

## Browser check: `e2e/browser_check.py` (optional, 28 checks)

Run against the real server with a stub LLM in headless Chromium. Covers: the empty state and the
disabled question box; a non-PDF and a text-less PDF each showing their error and listing nothing;
a successful upload (listed with page and passage counts, status message); asking with Enter;
the answer, its citation link, and the source's file and page; the source passage opening when the
citation is clicked; the input clearing after sending; a second question citing page 2; an
unanswerable question showing the not-found note with no sources; Shift+Enter making a new line;
documents surviving a page reload; removing a document and the question box disabling again; no
horizontal scroll at phone width; a network failure shown as an error message; and no JavaScript
errors in the console.

# Day 4 addendum: local model (Ollama)

Adds **15 Python test cases** (TC-129 to TC-143, 143 in total). They run against `tests/fake_ollama.py`,
a small server that implements Ollama's `/api/tags` and `/api/chat` over real HTTP.

## Ollama client: `tests/test_ollama.py`

| ID | Scenario | Expected result |
| --- | --- | --- |
| TC-129 | `generate` against the fake server | Request has the model, system and user messages, `stream: false`, temperature 0 and the context size; the reply text is returned |
| TC-130 | `llama3.2` when `llama3.2:latest` is installed; `qwen2.5:3b` when only `qwen2.5:7b` is | Untagged name matches `:latest`; a tagged name must match exactly (else error) |
| TC-131 | Nothing listening at the address | `LLMUnavailableError` "Could not reach Ollama" mentioning `ollama serve` |
| TC-132 | Model not installed | `LLMUnavailableError` listing installed models and `ollama pull llama3.2` |
| TC-133 | Chat endpoint returns an HTTP error | `LLMError` containing Ollama's own message |
| TC-134 | Reply is only whitespace | `LLMError` "empty" |
| TC-135 | Response is not valid JSON | `LLMError` "could not be read" |
| TC-136 | Reply wrapped in `<think>…</think>` | Thinking removed; only the answer returned |
| TC-137 | Address as `http://host:port`, with a trailing slash, and as bare `host:port` (3 cases) | All work |
| TC-138 | No address given, `OLLAMA_HOST` set | The variable's address is used |
| TC-139 | `make_llm` with `ollama`, `claude` and an unknown name | Ollama client with default model `llama3.2`; Claude client; `ValueError` |
| TC-140 | Full flow: retrieval, prompt, HTTP call, citation parsing | Answer with source `guide.pdf` page 2; prompt carries the numbered excerpt |
| TC-141 | Not-found reply plain, in quotes, upper case in backticks, in curly quotes (4 cases) | Recognised as not answered, no sources |
| TC-142 | `rag.ask --llm ollama --ollama-url ...` | Exit 0, answer and Sources list printed |
| TC-143 | `rag.ask --llm ollama` with no Ollama reachable | Exit 1 and "Could not reach Ollama" on stderr |

# Day 5 addendum: guards and evaluation

Adds **37 Python test cases** (TC-144 to TC-180, 180 in total). The first group covers the
figure-grounding check and `--require-citations`; the second covers the evaluation harness
(`rag.evaluate`). In these tables each row gives the test file and what the case checks.

| ID | File | What the case checks |
| --- | --- | --- |
| TC-144 | `test_grounding.py` | Every figure in the answer appears in an excerpt, so nothing is reported. |
| TC-145 | `test_grounding.py` | An answer saying 25 days when the document says 18 is flagged. |
| TC-146 | `test_grounding.py` | '$2,500' in an excerpt supports '2500' and '2,500'; '4%' supports '4 percent'. |
| TC-147 | `test_grounding.py` | 'After 4 years you get 23 days' is fine when the 4 came from the question. |
| TC-148 | `test_grounding.py` | 'see page 2' is supported when an excerpt came from page 2. |
| TC-149 | `test_grounding.py` | '[1]', '[2, 3]' and leading '1.' / '2)' list markers are not treated as figures. |
| TC-150 | `test_grounding.py` | '4.5' is supported by '4.5 percent' but '4.6' is flagged. |
| TC-151 | `test_grounding.py` | Repeated figures are listed once, in order of first appearance. |
| TC-152 | `test_grounding.py` | Text with no digits can never be flagged. |
| TC-153 | `test_grounding.py` | '... costs 600.' matches an excerpt's '$600'. |
| TC-154 | `test_grounding.py` | Digits in the answer are flagged when the excerpt spells the number out. This documents the limit stated in the module: the check is a warning signal, not proof. |
| TC-155 | `test_answerer.py` | 'You get 25 days [1]' when the document says 18 returns the answer with a figures warning. |
| TC-156 | `test_answerer.py` | '18 days [1]' with 18 in the excerpt produces no warnings. |
| TC-157 | `test_answerer.py` | With require_citations, a reply that cites nothing is withheld and reported. |
| TC-158 | `test_answerer.py` | With require_citations, a properly cited reply is returned as normal. |
| TC-159 | `test_answerer.py` | 'ask --require-citations' prints the not-found message and the reason for an uncited reply. |
| TC-160 | `test_api.py` | An app built with require_citations withholds an uncited answer (answered false). |
| TC-161 | `test_evaluate.py` | A valid dataset loads into cases with pages, expectations and flags. |
| TC-162 | `test_evaluate.py` | Blank text, an answerable question with no pages, a repeated id or a missing field is rejected. |
| TC-163 | `test_evaluate.py` | A missing file, invalid JSON and JSON without 'questions' each raise ValueError. |
| TC-164 | `test_evaluate.py` | The dataset shipped in eval/ loads, has 15+ cases and both answerable and unanswerable ones. |
| TC-165 | `test_evaluate.py` | Right page cited and expected text present gives 'pass'. |
| TC-166 | `test_evaluate.py` | Wrong cited page, missing expected text and forbidden text are each reported. |
| TC-167 | `test_evaluate.py` | An expect group is satisfied by any one of its alternatives. |
| TC-168 | `test_evaluate.py` | Declining a question the document answers is a failure. |
| TC-169 | `test_evaluate.py` | Declining passes; answering fails, or is flagged for review when the case allows partial answers. |
| TC-170 | `test_evaluate.py` | The question about the stadium ranks page 3 first; rows record rank and scores. |
| TC-171 | `test_evaluate.py` | hit@1, hit@k, mean reciprocal rank and multi-page coverage come out as expected. |
| TC-172 | `test_evaluate.py` | The recommended cut-off keeps 90% of answerable questions and blocks the weak unanswerable ones. |
| TC-173 | `test_evaluate.py` | With no answerable questions there is nothing to recommend, and nothing crashes. |
| TC-174 | `test_evaluate.py` | With a scripted model both cases pass and the summary counts them per group. |
| TC-175 | `test_evaluate.py` | A model that answers every question the same way fails the unanswerable case. |
| TC-176 | `test_evaluate.py` | Retrieval mode needs no model, prints the table and recommendation, and exits 0. |
| TC-177 | `test_evaluate.py` | Answers mode prints verdicts, writes both reports, and exits 0 when nothing fails. |
| TC-178 | `test_evaluate.py` | A failing case makes the command exit 1, so it can gate a build. |
| TC-179 | `test_evaluate.py` | A missing dataset, a missing PDF and a dataset without a document each print 'Error:' and exit 1. |
| TC-180 | `test_evaluate.py` | The word-overlap baseline still finds the right page in the top 4 for most answerable questions. |
