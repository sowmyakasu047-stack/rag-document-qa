"""The retrieval layer: documents in, relevant chunks out.

:class:`Retriever` joins the three earlier pieces: Day 1's ingestion (PDF to
chunks), an embedder (chunks to vectors) and a vector store (vectors to
similarity search). Day 3 hands the chunks it returns to the LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from rag.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from rag.embeddings import Embedder
from rag.errors import EmbedderMismatchError
from rag.ingest import ingest_pdf
from rag.models import Chunk
from rag.vectorstore import SearchResult, VectorStore

_EMBED_BATCH = 64


class Retriever:
    """Index documents and answer "which passages are relevant to this question?"."""

    def __init__(self, embedder: Embedder, store: VectorStore | None = None) -> None:
        if store is not None and store.dimension != embedder.dimension:
            raise ValueError(
                f"Store dimension {store.dimension} does not match embedder "
                f"'{embedder.name}' dimension {embedder.dimension}."
            )
        self.embedder = embedder
        self.store = store if store is not None else VectorStore(embedder.dimension, embedder.name)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed ``chunks`` in batches and add them to the store."""
        chunks = list(chunks)
        for start in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[start : start + _EMBED_BATCH]
            self.store.add(batch, self.embedder.embed([chunk.text for chunk in batch]))

    def add_pdf(
        self,
        path: str | Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> int:
        """Index a PDF and return the number of chunks added.

        A file with the same name that was indexed before is replaced, so
        uploading a new version does not leave stale passages behind. The old
        copy is only removed after the new one was read successfully, so a bad
        upload never destroys a good index.
        """
        chunks = ingest_pdf(path, chunk_size=chunk_size, overlap=overlap)
        self.store.remove_source(Path(path).name)
        self.add_chunks(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 5, min_score: float | None = None) -> list[SearchResult]:
        """Return the ``k`` chunks most relevant to ``query``, best first."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        vector = self.embedder.embed([query])[0]
        return self.store.search(vector, k=k, min_score=min_score)

    def save(self, directory: str | Path) -> None:
        self.store.save(directory)

    @classmethod
    def load(
        cls, directory: str | Path, embedder: Embedder, store_cls: type = VectorStore
    ) -> "Retriever":
        """Load a saved index, refusing one that was built with a different embedder."""
        store = store_cls.load(directory)
        if store.embedder_name != embedder.name:
            raise EmbedderMismatchError(
                f"The index was built with '{store.embedder_name}' but the current embedder is "
                f"'{embedder.name}'. Rebuild the index with the current embedder."
            )
        return cls(embedder, store)
