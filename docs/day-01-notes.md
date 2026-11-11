# Day 1 notes: RAG basics, PDF extraction, chunking

## What RAG is

Retrieval-Augmented Generation answers a question in two steps:

1. **Retrieve** the passages of a document collection that are most related to the question.
2. **Generate** an answer with an LLM that is given only those passages as context.

This addresses two weaknesses of a plain LLM: it cannot know the contents of a private
document, and it may produce confident but invented statements (hallucination). Because
every answer is built from retrieved passages, each one can point back to a source page.

## Pipeline stages and where Day 1 fits

| Stage | Purpose | Day |
| --- | --- | --- |
| Load | Turn the PDF into text, page by page | 1 |
| Clean | Remove extraction artefacts that damage search quality | 1 |
| Chunk | Cut text into passages sized for embedding and for the LLM context | 1 |
| Embed | Convert each chunk into a vector that captures its meaning | 2 |
| Store | Keep vectors in FAISS or ChromaDB for fast similarity search | 2 |
| Retrieve | Embed the question and fetch the nearest chunks | 3 |
| Generate | Prompt the LLM with the chunks and the question; cite sources | 3 |

## Design decisions

**Text is extracted per page, and chunks never cross pages.** The assignment requires source
references for answers. If a chunk could span pages 4 and 5, its citation would be
ambiguous. The cost is that a sentence running over a page break is split in two; the
overlap between chunks softens this within a page only.

**Cleaning happens before chunking, in a fixed order.** Header/footer removal needs the
original line structure, so it runs first. `clean_text` then joins wrapped lines, and the
chunker works on the result. (A first version did these in the opposite order, which glued
headers onto body text; the end-to-end test `TC-26` guards against that.)

**Header/footer detection is limited to page edges.** An earlier version removed any line
repeated on most pages. Testing it on a PDF with repeated body lines removed the entire
document and reported success with zero chunks. It now only considers the top and bottom
two lines of each page, and `ingest_pdf` raises `NoExtractableTextError` if nothing is
left (`TC-29`, `TC-30`).

**Chunking is recursive on natural boundaries.** The splitter prefers paragraph breaks,
then line breaks, then sentence ends, then spaces, and only cuts mid-word as a last resort.
Splitting on meaning boundaries keeps each chunk readable, which helps both the embedding
and the LLM.

**Each failure has its own exception.** A corrupt file, a password-protected file and a
scanned file need different messages for the user, so they are distinct classes rather than
one generic error.

## Questions to revisit on later days

- Are 800/150 characters a good default? Compare retrieval quality for several settings
  once embeddings exist (Day 2/3).
- Would token-based sizes match the embedding model's limits better than characters?
- Should chunks keep a short heading or document title as extra context?
