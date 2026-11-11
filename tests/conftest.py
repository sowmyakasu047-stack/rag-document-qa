"""Shared fixtures that build small PDFs on the fly, so tests need no binary files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MakePdf = Callable[..., Path]


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Build a PDF where each list item becomes one page.

    Each page's text is a list of lines drawn top to bottom. An empty list
    produces a page with no text layer (like a scanned page).
    """

    def _make(pages: list[list[str]], name: str = "sample.pdf") -> Path:
        path = tmp_path / name
        pdf = canvas.Canvas(str(path), pagesize=A4)
        for lines in pages:
            y = 800
            for line in lines:
                pdf.drawString(50, y, line)
                y -= 18
            pdf.showPage()
        pdf.save()
        return path

    return _make


@pytest.fixture
def encrypt_pdf() -> Callable[[Path, str], Path]:
    """Return a helper that writes a password-protected copy of a PDF."""

    def _encrypt(source: Path, password: str) -> Path:
        writer = PdfWriter(clone_from=PdfReader(str(source)))
        writer.encrypt(password)
        target = source.with_name(f"locked-{source.name}")
        with target.open("wb") as handle:
            writer.write(handle)
        return target

    return _encrypt
