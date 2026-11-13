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
