"""Document ingestion pipeline for the RAG question-answering system."""

from rag.chunker import chunk_pages
from rag.errors import (
    EncryptedPDFError,
    InvalidPDFError,
    NoExtractableTextError,
    RagError,
)
from rag.ingest import ingest_pdf
from rag.loader import load_pdf
from rag.models import Chunk, PageText
from rag.preprocess import clean_text, strip_repeated_lines

__all__ = [
    "Chunk",
    "EncryptedPDFError",
    "InvalidPDFError",
    "NoExtractableTextError",
    "PageText",
    "RagError",
    "chunk_pages",
    "clean_text",
    "ingest_pdf",
    "load_pdf",
    "strip_repeated_lines",
]
