# Test cases

Run all with `python -m pytest`. Every case has an ID that also appears in the docstring of
the matching test, so a failure can be traced back to this table.

Day 1 total: **30 test cases** (34 pytest cases; TC-22 is parametrized over 5 inputs).

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
