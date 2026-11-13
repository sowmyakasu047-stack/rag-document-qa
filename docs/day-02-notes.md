# Day 2 notes: embeddings, vector search, retrieval

## Concepts

**Embedding.** A function that maps a text to a vector (a list of numbers, 384 of them for
`all-MiniLM-L6-v2`) so that texts with similar meaning get vectors pointing in similar
directions. This is what lets a question like "How do plants make sugar?" find a passage that
says "photosynthesis converts sunlight into glucose" even though they share almost no words.

**Cosine similarity.** The angle between two vectors: 1 means same direction, 0 unrelated,
negative opposite. Length is irrelevant, only direction. If every vector is scaled to length 1,
cosine similarity is just the dot product, which is one fast matrix multiplication for the
whole store. That is why every embedder here returns unit-length vectors and the store
normalises again on the way in (a different model, or a caller, may not).

**Vector store.** Holds one vector per chunk and answers "which stored vectors are closest to
this one?". FAISS and ChromaDB are libraries for this. Both are optional here: the flat (exact)
index compares the query with every vector, which is perfectly fast up to hundreds of thousands
of chunks. Approximate indexes (IVF, HNSW) trade a little accuracy for speed at millions of
vectors and would be over-engineering for a handful of PDFs.

**Top-k retrieval.** Return the `k` highest-scoring chunks. `k` is a trade-off: too small and
the answer may sit in chunk k+1; too large and the LLM prompt fills with noise.

## What was built

| Module | Role |
| --- | --- |
| `embeddings.py` | `Embedder` interface; `SentenceTransformerEmbedder` (neural, for real use); `HashingEmbedder` (offline word-overlap baseline) |
| `vectorstore.py` | `VectorStore`: exact cosine search, `remove_source`, save/load; shared validation helpers |
| `faiss_store.py` | `FaissVectorStore`: same methods on a FAISS `IndexFlatIP` |
| `retriever.py` | `Retriever`: PDF in (Day 1 ingestion), embed, store, `search(question)` out |
| `query.py` | Command line to try retrieval on a PDF |

## Design decisions

**Embedder and store are swappable behind small interfaces.** The assignment allows FAISS or
ChromaDB and a choice of embedding model; keeping both behind a few methods means the choice can
change without touching retrieval code, and tests can run without any model download.

**A lexical baseline embedder ships alongside the neural one.** `HashingEmbedder` only matches
shared words, so it is *not* a substitute for a real model and does not capture meaning. It
exists because the build and test environment has no network or GPU, and because a deterministic
embedder makes retrieval tests exact. It also doubles as a baseline for comparing against the
neural model later.

**Re-uploading a file replaces its chunks.** Otherwise the same document indexed twice returns
duplicate passages and crowds out other results. The old chunks are only removed after the new
file has been read successfully, so a corrupt re-upload cannot wipe a good index (`TC-60`).

**An index remembers which embedder built it.** Vectors from two models are not comparable;
searching one with the other produces plausible-looking but meaningless scores. Loading an
index with a different embedder raises `EmbedderMismatchError` (`TC-65`).

**Saved data uses plain formats.** Vectors in a `.npy` file loaded with `allow_pickle=False`,
chunks and metadata in JSON. Loading checks that the counts agree and raises
`VectorStoreError` for missing, damaged or inconsistent files (`TC-52` to `TC-55`).

## Observations from running it

Three-topic PDF (photosynthesis, French Revolution, transistors), baseline embedder:

- The right page ranked first for all three questions (scores 0.41, 0.35, 0.25). Scores are low
  because the baseline only compares words, not meaning.
- A question about a topic absent from the document ("Who won the football match?") still
  returned a result, with score 0.000. **Search always returns the nearest chunks, even if none
  is relevant.** A `min_score` cutoff is how the system can later say "this is not in the
  document" instead of letting the LLM answer from irrelevant text. The threshold depends on the
  embedder, so it must be tuned with the real model on Day 3 and 5.
- One irrelevant page scored 0.144 against the French Revolution question although they share
  no words. With only 512 hash slots, unrelated words occasionally collide. A neural embedder
  does not have this problem; it is the cost of the baseline.

## Not verified yet

The environment used to build Day 2 had no network, so these were written but **not run
against the real libraries**: `SentenceTransformerEmbedder` with a real model (tested only with
a stand-in model object), and `FaissVectorStore` (its three tests skip when `faiss` is absent).
Run `python -m pytest` after `pip install -r requirements.txt` and confirm nothing is skipped.

## Questions for Day 3

- What `min_score` and `k` give good answers with the neural embedder?
- Compare baseline vs MiniLM retrieval on the same questions (a useful result for the
  presentation).
- How should retrieved chunks be ordered and labelled inside the prompt so the LLM can cite them?
