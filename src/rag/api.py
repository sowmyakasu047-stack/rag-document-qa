"""HTTP API used by the React interface.

Built on Starlette (the toolkit FastAPI itself is built on), so it adds only the
pieces it needs: routing, multipart uploads and JSON responses.

Endpoints
---------
``GET    /api/health``              service status and whether the LLM is configured
``GET    /api/documents``           list indexed documents
``POST   /api/documents``           upload a PDF (multipart field ``file``) and index it
``DELETE /api/documents/{name}``    remove a document from the index
``POST   /api/ask``                 ``{"question": "...", "k": 4}`` -> answer with sources

Every error uses one shape, ``{"error": {"code": "...", "message": "..."}}``, so the
interface can show ``message`` directly and branch on ``code``.

The service has no login and is meant to run on your own machine (it listens on
127.0.0.1 by default). Do not expose it to a network without adding authentication.
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Callable

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from rag.answerer import DEFAULT_K, DEFAULT_MIN_SCORE, Answer, Answerer
from rag.errors import (
    EmbeddingModelUnavailableError,
    EncryptedPDFError,
    InvalidPDFError,
    LLMError,
    LLMUnavailableError,
    NoExtractableTextError,
    RagError,
)
from rag.llm import LLM
from rag.retriever import DocumentInfo, Retriever

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_QUESTION_CHARS = 2000
MAX_K = 10
_UPLOAD_OVERHEAD = 64 * 1024  # multipart framing around the file itself
_READ_BLOCK = 1024 * 1024
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def safe_filename(raw: str | None) -> str | None:
    """Reduce an uploaded file name to a plain, safe ``*.pdf`` name, or None if unusable.

    Browsers send only the base name, but other clients may send paths such as
    ``../../x.pdf`` or ``C:\\docs\\x.pdf``; directories are always dropped.
    """
    if not raw:
        return None
    name = PurePosixPath(raw.replace("\\", "/")).name.strip()
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)
    if name in {"", ".", ".."} or len(name) > 200 or not name.lower().endswith(".pdf"):
        return None
    return name


class SafeRetriever:
    """Serialises access to a :class:`Retriever` and saves the index after every change.

    Requests run in worker threads, and the retriever's index is not safe for
    concurrent modification. One lock keeps uploads, deletes and searches from
    interleaving. The slow LLM call is made outside the lock, so a pending answer
    does not block an upload.
    """

    def __init__(self, retriever: Retriever, persist_dir: Path | None = None) -> None:
        self._retriever = retriever
        self._persist_dir = persist_dir
        self._lock = threading.RLock()

    def index_pdf(self, path: Path) -> tuple[int, bool]:
        """Index a PDF; return ``(chunk_count, replaced_existing_document)``."""
        with self._lock:
            replaced = any(d.name == path.name for d in self._retriever.list_documents())
            count = self._retriever.add_pdf(path)
            self._persist()
            return count, replaced

    def remove_document(self, name: str) -> int:
        with self._lock:
            removed = self._retriever.remove_document(name)
            if removed:
                self._persist()
            return removed

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            return self._retriever.list_documents()

    def search(self, query: str, k: int = 5, min_score: float | None = None):
        with self._lock:
            return self._retriever.search(query, k=k, min_score=min_score)

    def _persist(self) -> None:
        if self._persist_dir is not None:
            self._retriever.save(self._persist_dir)


def answer_to_dict(answer: Answer) -> dict:
    return {
        "answer": answer.text,
        "answered": answer.answered,
        "sources": [asdict(source) for source in answer.sources],
        "warnings": answer.warnings,
    }


async def _save_limited(upload: UploadFile, destination: Path, limit: int) -> bool:
    """Copy an upload to ``destination``; return False if it exceeds ``limit`` bytes."""
    written = 0
    with destination.open("wb") as handle:
        while block := await upload.read(_READ_BLOCK):
            written += len(block)
            if written > limit:
                return False
            handle.write(block)
    return True


def create_app(
    retriever: Retriever,
    llm_factory: Callable[[], LLM],
    *,
    persist_dir: str | Path | None = None,
    frontend_dist: str | Path | None = None,
    min_score: float | None = DEFAULT_MIN_SCORE,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> Starlette:
    """Build the web application.

    Args:
        retriever: Holds the index. Wrapped so concurrent requests are safe.
        llm_factory: Called on the first question to create the LLM. Failing here
            (for example, no API key) only affects ``/api/ask``; uploads still work.
        persist_dir: If set, the index is saved here after every upload or delete.
        frontend_dist: If this folder exists, it is served at ``/`` (the built React app).
        min_score: Similarity below which excerpts are ignored when answering.
        max_upload_bytes: Largest accepted PDF.
    """
    service = SafeRetriever(retriever, Path(persist_dir) if persist_dir else None)
    llm_lock = threading.Lock()
    llm_cache: list[LLM] = []

    def get_llm() -> LLM:
        with llm_lock:
            if not llm_cache:
                llm_cache.append(llm_factory())  # raises LLMUnavailableError if not configured
            return llm_cache[0]

    async def health(request: Request) -> Response:
        try:
            get_llm()
            llm_ready, llm_message = True, None
        except LLMUnavailableError as exc:
            llm_ready, llm_message = False, str(exc)
        documents = await run_in_threadpool(service.list_documents)
        return JSONResponse(
            {
                "status": "ok",
                "documents": len(documents),
                "llm_ready": llm_ready,
                "llm_message": llm_message,
                "max_upload_mb": max_upload_bytes // (1024 * 1024),
            }
        )

    async def list_documents(request: Request) -> Response:
        documents = await run_in_threadpool(service.list_documents)
        return JSONResponse({"documents": [asdict(d) for d in documents]})

    async def upload_document(request: Request) -> Response:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_upload_bytes + _UPLOAD_OVERHEAD:
            return _too_large(max_upload_bytes)
        try:
            async with request.form(max_files=1, max_fields=5) as form:
                upload = form.get("file")
                if not isinstance(upload, UploadFile):
                    return error_response(400, "missing_file", "Choose a PDF file to upload.")
                name = safe_filename(upload.filename)
                if name is None:
                    return error_response(400, "not_a_pdf", "Only .pdf files can be uploaded.")
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / name
                    if not await _save_limited(upload, path, max_upload_bytes):
                        return _too_large(max_upload_bytes)
                    chunks, replaced = await run_in_threadpool(service.index_pdf, path)
        except MultiPartException as exc:
            return error_response(400, "invalid_upload", f"The upload could not be read: {exc}")
        except InvalidPDFError as exc:
            return error_response(400, "invalid_pdf", str(exc))
        except EncryptedPDFError:
            return error_response(
                400, "encrypted_pdf", "This PDF is password protected. Remove the password and try again."
            )
        except NoExtractableTextError as exc:
            return error_response(422, "no_text", str(exc))
        except EmbeddingModelUnavailableError as exc:
            return error_response(503, "embedding_unavailable", str(exc))
        except RagError as exc:
            logger.exception("Indexing failed")
            return error_response(500, "internal_error", str(exc))
        return JSONResponse({"name": name, "chunks": chunks, "replaced": replaced}, status_code=201)

    async def delete_document(request: Request) -> Response:
        name = request.path_params["name"]
        removed = await run_in_threadpool(service.remove_document, name)
        if not removed:
            return error_response(404, "not_found", f"No document named '{name}' is indexed.")
        return Response(status_code=204)

    async def ask(request: Request) -> Response:
        try:
            payload = await request.json()
        except ValueError:
            return error_response(400, "invalid_json", "The request body must be JSON.")
        if not isinstance(payload, dict):
            return error_response(400, "invalid_json", "The request body must be a JSON object.")

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            return error_response(400, "empty_question", "Type a question first.")
        if len(question) > MAX_QUESTION_CHARS:
            return error_response(
                400, "question_too_long", f"Questions can be at most {MAX_QUESTION_CHARS} characters."
            )
        k = payload.get("k", DEFAULT_K)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
            return error_response(400, "invalid_k", f"k must be a whole number from 1 to {MAX_K}.")

        if not await run_in_threadpool(service.list_documents):
            return error_response(409, "no_documents", "Upload a PDF before asking questions.")

        try:
            llm = get_llm()
            answerer = Answerer(service, llm, k=k, min_score=min_score)  # type: ignore[arg-type]
            answer = await run_in_threadpool(answerer.ask, question)
        except LLMUnavailableError as exc:
            return error_response(503, "llm_unavailable", str(exc))
        except LLMError as exc:
            return error_response(502, "llm_error", str(exc))
        except EmbeddingModelUnavailableError as exc:
            return error_response(503, "embedding_unavailable", str(exc))
        return JSONResponse(answer_to_dict(answer))

    async def unexpected_error(request: Request, exc: Exception) -> Response:
        logger.exception("Unhandled error", exc_info=exc)
        return error_response(500, "internal_error", "Something went wrong on the server.")

    routes: list = [
        Route("/api/health", health, methods=["GET"]),
        Route("/api/documents", list_documents, methods=["GET"]),
        Route("/api/documents", upload_document, methods=["POST"]),
        Route("/api/documents/{name}", delete_document, methods=["DELETE"]),
        Route("/api/ask", ask, methods=["POST"]),
    ]
    if frontend_dist is not None and Path(frontend_dist).is_dir():
        routes.append(Mount("/", app=StaticFiles(directory=str(frontend_dist), html=True)))

    return Starlette(routes=routes, exception_handlers={Exception: unexpected_error})


def _too_large(limit: int) -> JSONResponse:
    return error_response(
        413, "file_too_large", f"The PDF is larger than the {limit // (1024 * 1024)} MB limit."
    )
