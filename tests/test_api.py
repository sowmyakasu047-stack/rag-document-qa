"""Tests for the HTTP API (rag.api)."""

from __future__ import annotations

import asyncio

import pytest

from asgi_client import arequest, request
from rag.api import create_app, safe_filename
from rag.embeddings import HashingEmbedder
from rag.errors import LLMError, LLMUnavailableError
from rag.prompt import NOT_FOUND_MESSAGE
from rag.retriever import Retriever

PAGES = [
    ["Chunking splits long documents into small overlapping passages", "so that each passage covers one topic."],
    ["Embeddings turn passages into vectors", "and vector databases find the nearest vectors quickly."],
    ["Football season begins in September.", "The stadium seats seventy thousand fans for each match."],
]


@pytest.fixture
def make_app(fake_llm):
    def _make(llm=None, **kwargs):
        """Build an app. ``llm`` is a fake model, or a factory function to simulate setup failures."""
        llm = llm if llm is not None else fake_llm("The stadium seats seventy thousand fans [1].")
        factory = (lambda: llm) if hasattr(llm, "generate") else llm
        return create_app(Retriever(HashingEmbedder()), factory, **kwargs), llm

    return _make


@pytest.fixture
def pdf_bytes(make_pdf):
    def _bytes(pages=PAGES, name="guide.pdf") -> bytes:
        return make_pdf(pages, name=name).read_bytes()

    return _bytes


def upload(app, content: bytes, name: str = "guide.pdf"):
    return request(app, "POST", "/api/documents", files={"file": (name, content)})


def error_code(response) -> str:
    return response.json()["error"]["code"]


# --- health and listing -------------------------------------------------------------


def test_health_reports_status_documents_and_llm_readiness(make_app):
    """TC-93: /api/health returns ok, the document count and that the LLM is ready."""
    app, _ = make_app()

    body = request(app, "GET", "/api/health").json()

    assert body["status"] == "ok" and body["documents"] == 0
    assert body["llm_ready"] is True and body["llm_message"] is None


def test_health_explains_when_the_llm_is_not_configured(make_app):
    """TC-94: Without an API key the health check says so, and the server still answers."""
    def no_key():
        raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")

    app, _ = make_app(llm=no_key)

    body = request(app, "GET", "/api/health").json()

    assert body["llm_ready"] is False and "ANTHROPIC_API_KEY" in body["llm_message"]


def test_document_list_starts_empty(make_app):
    """TC-95: Before any upload the document list is empty."""
    app, _ = make_app()

    assert request(app, "GET", "/api/documents").json() == {"documents": []}


# --- upload -------------------------------------------------------------------------


def test_upload_indexes_the_pdf_and_lists_it(make_app, pdf_bytes):
    """TC-96: A valid PDF returns 201 with its chunk count and then appears in the list."""
    app, _ = make_app()

    response = upload(app, pdf_bytes())

    assert response.status == 201
    assert response.json() == {"name": "guide.pdf", "chunks": 3, "replaced": False}
    assert request(app, "GET", "/api/documents").json()["documents"] == [
        {"name": "guide.pdf", "chunks": 3, "pages": 3}
    ]


def test_uploading_the_same_name_again_replaces_it(make_app, pdf_bytes):
    """TC-97: A second upload with the same name reports replaced and leaves one copy."""
    app, _ = make_app()
    upload(app, pdf_bytes())

    again = upload(app, pdf_bytes())

    assert again.json()["replaced"] is True
    assert len(request(app, "GET", "/api/documents").json()["documents"]) == 1


def test_non_pdf_extension_is_rejected(make_app):
    """TC-98: A file not named *.pdf returns 400 not_a_pdf."""
    app, _ = make_app()

    response = upload(app, b"hello", name="notes.txt")

    assert response.status == 400 and error_code(response) == "not_a_pdf"


def test_corrupt_pdf_is_rejected_with_a_message(make_app):
    """TC-99: A .pdf file with garbage inside returns 400 invalid_pdf and a readable message."""
    app, _ = make_app()

    response = upload(app, b"this is not a pdf")

    assert response.status == 400 and error_code(response) == "invalid_pdf"
    assert "guide.pdf" in response.json()["error"]["message"]


def test_pdf_without_text_returns_422(make_app, pdf_bytes):
    """TC-100: A scanned-style PDF with no text layer returns 422 no_text."""
    app, _ = make_app()

    response = upload(app, pdf_bytes([[], []]))

    assert response.status == 422 and error_code(response) == "no_text"


def test_password_protected_pdf_is_rejected(make_app, make_pdf, encrypt_pdf):
    """TC-101: A password-protected PDF returns 400 encrypted_pdf."""
    app, _ = make_app()
    locked = encrypt_pdf(make_pdf([["Secret."]]), "pw")

    response = upload(app, locked.read_bytes())

    assert response.status == 400 and error_code(response) == "encrypted_pdf"


def test_upload_without_a_file_field_is_rejected(make_app):
    """TC-102: A multipart request that has no 'file' field returns 400 missing_file."""
    app, _ = make_app()

    response = request(app, "POST", "/api/documents", files={"note": "hello"})

    assert response.status == 400 and error_code(response) == "missing_file"


def test_upload_with_a_json_body_is_rejected(make_app):
    """TC-103: Posting JSON instead of a file returns 400 missing_file, not a server error."""
    app, _ = make_app()

    response = request(app, "POST", "/api/documents", json_body={"file": "x"})

    assert response.status == 400 and error_code(response) == "missing_file"


def test_file_over_the_size_limit_is_rejected_while_reading(make_app, pdf_bytes):
    """TC-104: A file slightly over the limit returns 413 and nothing is indexed."""
    app, _ = make_app(max_upload_bytes=100)

    response = upload(app, pdf_bytes())

    assert response.status == 413 and error_code(response) == "file_too_large"
    assert request(app, "GET", "/api/documents").json()["documents"] == []


def test_file_far_over_the_limit_is_rejected_from_the_declared_size(make_app):
    """TC-105: A request whose declared size is far over the limit is refused immediately."""
    app, _ = make_app(max_upload_bytes=1000)

    response = upload(app, b"x" * 200_000)

    assert response.status == 413 and error_code(response) == "file_too_large"


def test_uploaded_file_names_lose_any_directory_part(make_app, pdf_bytes):
    """TC-106: '../../evil.pdf' and a Windows path are stored under their plain base names."""
    app, _ = make_app()

    first = upload(app, pdf_bytes(), name="../../evil.pdf")
    second = upload(app, pdf_bytes(), name="C:\\docs\\report.pdf")

    assert first.json()["name"] == "evil.pdf"
    assert second.json()["name"] == "report.pdf"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../a.pdf", "a.pdf"),
        ("C:\\docs\\b.PDF", "b.PDF"),
        ('we<ir>d:"name?.pdf', "we_ir_d__name_.pdf"),
        ("notes.txt", None),
        ("", None),
        (None, None),
        ("..", None),
        ("a" * 300 + ".pdf", None),
    ],
)
def test_safe_filename(raw, expected):
    """TC-107: File names are reduced to a safe base name, or refused when unusable."""
    assert safe_filename(raw) == expected


# --- delete -------------------------------------------------------------------------


def test_delete_removes_the_document(make_app, pdf_bytes):
    """TC-108: DELETE returns 204 and the document disappears from the list."""
    app, _ = make_app()
    upload(app, pdf_bytes(), name="my notes.pdf")

    assert request(app, "DELETE", "/api/documents/my notes.pdf").status == 204
    assert request(app, "GET", "/api/documents").json()["documents"] == []


def test_deleting_an_unknown_document_returns_404(make_app):
    """TC-109: DELETE of a name that is not indexed returns 404 not_found."""
    app, _ = make_app()

    response = request(app, "DELETE", "/api/documents/missing.pdf")

    assert response.status == 404 and error_code(response) == "not_found"


# --- ask ----------------------------------------------------------------------------


def test_ask_returns_the_answer_with_cited_sources(make_app, pdf_bytes):
    """TC-110: A question returns the answer text and the cited file, page and score."""
    app, llm = make_app()
    upload(app, pdf_bytes())

    response = request(app, "POST", "/api/ask", json_body={"question": "How many fans does the stadium seat?"})

    body = response.json()
    assert response.status == 200
    assert body["answer"] == "The stadium seats seventy thousand fans [1]."
    assert body["answered"] is True and body["warnings"] == []
    assert [(s["number"], s["source"], s["page"]) for s in body["sources"]] == [(1, "guide.pdf", 3)]
    assert set(body["sources"][0]) == {"number", "source", "page", "text", "score"}


def test_ask_about_something_not_in_the_document_skips_the_model(make_app, pdf_bytes):
    """TC-111: A question with no relevant excerpt returns the not-found answer, no LLM call."""
    app, llm = make_app()
    upload(app, pdf_bytes())

    body = request(app, "POST", "/api/ask", json_body={"question": "quantum chromodynamics lagrangian"}).json()

    assert body["answered"] is False and body["answer"] == NOT_FOUND_MESSAGE
    assert body["sources"] == [] and llm.calls == []


def test_ask_before_any_upload_returns_409(make_app):
    """TC-112: Asking with nothing indexed returns 409 no_documents."""
    app, _ = make_app()

    response = request(app, "POST", "/api/ask", json_body={"question": "anything"})

    assert response.status == 409 and error_code(response) == "no_documents"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json_body": {"question": ""}},
        {"json_body": {"question": "   "}},
        {"json_body": {"question": 42}},
        {"json_body": {}},
        {"json_body": ["question"]},
        {"raw_body": b"{not json", "content_type": "application/json"},
    ],
)
def test_ask_rejects_malformed_requests(make_app, pdf_bytes, kwargs):
    """TC-113: Blank, non-text, missing or unparsable questions return 400, never a crash."""
    app, llm = make_app()
    upload(app, pdf_bytes())

    response = request(app, "POST", "/api/ask", **kwargs)

    assert response.status == 400
    assert llm.calls == []


def test_ask_rejects_an_over_long_question(make_app, pdf_bytes):
    """TC-114: A question over 2000 characters returns 400 question_too_long."""
    app, _ = make_app()
    upload(app, pdf_bytes())

    response = request(app, "POST", "/api/ask", json_body={"question": "x" * 2001})

    assert response.status == 400 and error_code(response) == "question_too_long"


@pytest.mark.parametrize("k", [0, 11, -1, "3", 2.5, True, None])
def test_ask_rejects_invalid_k(make_app, pdf_bytes, k):
    """TC-115: k must be a whole number from 1 to 10."""
    app, _ = make_app()
    upload(app, pdf_bytes())

    response = request(app, "POST", "/api/ask", json_body={"question": "stadium", "k": k})

    assert response.status == 400 and error_code(response) == "invalid_k"


def test_ask_reports_a_missing_api_key_as_503(make_app, pdf_bytes):
    """TC-116: With no LLM configured, asking returns 503 llm_unavailable with the reason."""
    def no_key():
        raise LLMUnavailableError("ANTHROPIC_API_KEY is not set.")

    app, _ = make_app(llm=no_key)
    upload(app, pdf_bytes())

    response = request(app, "POST", "/api/ask", json_body={"question": "How many fans does the stadium seat?"})

    assert response.status == 503 and error_code(response) == "llm_unavailable"
    assert "ANTHROPIC_API_KEY" in response.json()["error"]["message"]


def test_ask_reports_a_failing_model_as_502(make_app, pdf_bytes, fake_llm):
    """TC-117: If the model call fails, the API returns 502 llm_error."""
    app, _ = make_app(llm=fake_llm(LLMError("The language model request failed: timeout")))
    upload(app, pdf_bytes())

    response = request(app, "POST", "/api/ask", json_body={"question": "How many fans does the stadium seat?"})

    assert response.status == 502 and error_code(response) == "llm_error"


def test_uploads_still_work_when_the_llm_is_not_configured(make_app, pdf_bytes):
    """TC-118: A missing API key does not stop documents from being uploaded."""
    def no_key():
        raise LLMUnavailableError("no key")

    app, _ = make_app(llm=no_key)

    assert upload(app, pdf_bytes()).status == 201


def test_concurrent_uploads_do_not_corrupt_the_index(make_app, make_pdf):
    """TC-119: Eight uploads sent at the same time all succeed and are all listed once."""
    app, _ = make_app()
    names = [f"doc{i}.pdf" for i in range(8)]
    contents = {n: make_pdf(PAGES, name=n).read_bytes() for n in names}

    async def run():
        return await asyncio.gather(
            *(arequest(app, "POST", "/api/documents", files={"file": (n, contents[n])}) for n in names)
        )

    responses = asyncio.run(run())

    assert [r.status for r in responses] == [201] * 8
    listed = request(app, "GET", "/api/documents").json()["documents"]
    assert sorted(d["name"] for d in listed) == sorted(names)
    assert all(d["chunks"] == 3 for d in listed)


def test_require_citations_setting_reaches_the_api(make_app, pdf_bytes, fake_llm):
    """TC-160: An app built with require_citations withholds an uncited answer (answered false)."""
    app, _ = make_app(llm=fake_llm("About seventy thousand."), require_citations=True)
    upload(app, pdf_bytes())

    body = request(app, "POST", "/api/ask", json_body={"question": "How many fans does the stadium seat?"}).json()

    assert body["answered"] is False and body["sources"] == []
    assert any("withheld" in w for w in body["warnings"])
