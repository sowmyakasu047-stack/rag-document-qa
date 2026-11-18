"""Retrieval-augmented question answering over PDFs."""

from rag.answerer import Answer, Answerer, Source
from rag.chunker import chunk_pages
from rag.embeddings import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from rag.errors import (
    EmbedderMismatchError,
    EmbeddingModelUnavailableError,
    EncryptedPDFError,
    InvalidPDFError,
    LLMError,
    LLMUnavailableError,
    NoExtractableTextError,
    RagError,
    VectorStoreError,
)
from rag.ingest import ingest_pdf
from rag.llm import LLM, ClaudeLLM, OllamaLLM, make_llm
from rag.loader import load_pdf
from rag.models import Chunk, PageText
from rag.preprocess import clean_text, strip_repeated_lines
from rag.prompt import NOT_FOUND_MESSAGE
from rag.retriever import Retriever
from rag.vectorstore import SearchResult, VectorStore

__all__ = [
    "Answer",
    "Answerer",
    "ClaudeLLM",
    "LLM",
    "LLMError",
    "LLMUnavailableError",
    "NOT_FOUND_MESSAGE",
    "OllamaLLM",
    "Source",
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
    "make_llm",
    "strip_repeated_lines",
]
