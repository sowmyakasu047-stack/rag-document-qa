"""Retrieval-augmented question answering over PDFs: ingestion and retrieval."""

from rag.chunker import chunk_pages
from rag.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from rag.errors import (
    EmbedderMismatchError,
    EmbeddingModelUnavailableError,
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    RagError,
    VectorStoreError,
)
from rag.ingest import ingest_pdf
from rag.loader import load_pdf
from rag.models import Chunk, PageText
from rag.preprocess import clean_text, strip_repeated_lines
from rag.retriever import Retriever
from rag.vectorstore import SearchResult, VectorStore

__all__ = [
    "Chunk",
    "Embedder",
    "EmbedderMismatchError",
    "EmbeddingModelUnavailableError",
    "EncryptedPDFError",
    "HashingEmbedder",
    "InvalidPDFError",
    "NoExtractableTextError",
    "PageText",
    "RagError",
    "Retriever",
    "SearchResult",
    "SentenceTransformerEmbedder",
    "VectorStore",
    "VectorStoreError",
    "chunk_pages",
    "clean_text",
    "ingest_pdf",
    "load_pdf",
    "strip_repeated_lines",
]
