"""Tests for starting the application (rag.server): persistence and static files."""

from __future__ import annotations

import pytest

from asgi_client import request
from rag.embeddings import HashingEmbedder
from rag.errors import EmbedderMismatchError
from rag.server import build_app, main

PAGES = [["Football season begins in September.", "The stadium seats seventy thousand fans."]]


def _app(index_dir, fake_llm, embedder=None, **kwargs):
    llm = fake_llm("ok [1]")
    return build_app(embedder or HashingEmbedder(), lambda: llm, index_dir=index_dir, frontend_dist=None, **kwargs)


def _upload(app, make_pdf, name="guide.pdf"):
    return request(app, "POST", "/api/documents", files={"file": (name, make_pdf(PAGES, name=name).read_bytes())})


def _names(app):
    return [d["name"] for d in request(app, "GET", "/api/documents").json()["documents"]]


def test_uploaded_documents_survive_a_restart(tmp_path, make_pdf, fake_llm):
    """TC-122: A new app built on the same index folder still lists and answers from the document."""
    first = _app(tmp_path / "idx", fake_llm)
    assert _upload(first, make_pdf).status == 201

    second = _app(tmp_path / "idx", fake_llm)

    assert _names(second) == ["guide.pdf"]
    answer = request(second, "POST", "/api/ask", json_body={"question": "how many fans does the stadium seat"})
    assert answer.status == 200 and answer.json()["sources"][0]["page"] == 1


def test_a_deleted_document_stays_deleted_after_restart(tmp_path, make_pdf, fake_llm):
    """TC-123: Deleting a document is saved, so it does not come back on restart."""
    first = _app(tmp_path / "idx", fake_llm)
    _upload(first, make_pdf)
    assert request(first, "DELETE", "/api/documents/guide.pdf").status == 204

    assert _names(_app(tmp_path / "idx", fake_llm)) == []


def test_a_new_index_folder_starts_empty_and_is_created_on_first_upload(tmp_path, make_pdf, fake_llm):
    """TC-124: With no saved index the app starts empty and creates the folder when needed."""
    app = _app(tmp_path / "does" / "not" / "exist", fake_llm)

    assert _names(app) == []
    _upload(app, make_pdf)
    assert (tmp_path / "does" / "not" / "exist" / "meta.json").is_file()


def test_loading_an_index_built_with_another_embedder_is_refused(tmp_path, make_pdf, fake_llm):
    """TC-125: Starting with a different embedder than the saved index raises EmbedderMismatchError."""
    _upload(_app(tmp_path / "idx", fake_llm), make_pdf)

    with pytest.raises(EmbedderMismatchError):
        _app(tmp_path / "idx", fake_llm, embedder=HashingEmbedder(dimension=128))


def test_server_command_reports_an_embedder_mismatch_and_exits(tmp_path, make_pdf, fake_llm, capsys):
    """TC-126: The server command prints a clear error and exits 1 instead of crashing."""
    _upload(_app(tmp_path / "idx", fake_llm, embedder=HashingEmbedder(dimension=128)), make_pdf)

    code = main(["--embedder", "hashing", "--index-dir", str(tmp_path / "idx")])

    assert code == 1
    assert "Rebuild the index" in capsys.readouterr().err


def test_built_frontend_is_served_next_to_the_api(tmp_path, fake_llm):
    """TC-127: When a built frontend folder exists, '/' serves its index.html and the API still works."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><body>Hello UI</body></html>")
    llm = fake_llm()
    app = build_app(HashingEmbedder(), lambda: llm, index_dir=None, frontend_dist=dist)

    page = request(app, "GET", "/")

    assert page.status == 200 and b"Hello UI" in page.body
    assert request(app, "GET", "/api/health").json()["status"] == "ok"


def test_without_a_built_frontend_the_root_is_not_found(tmp_path, fake_llm):
    """TC-128: If there is no frontend folder, '/' returns 404 while the API keeps working."""
    app = _app(tmp_path / "idx", fake_llm)

    assert request(app, "GET", "/").status == 404
    assert request(app, "GET", "/api/health").status == 200
