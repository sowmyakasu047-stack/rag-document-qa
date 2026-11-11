"""PDF loading and raw text extraction."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

from rag.errors import EncryptedPDFError, InvalidPDFError, NoExtractableTextError
from rag.models import PageText


def load_pdf(path: str | Path) -> list[PageText]:
    """Extract the text of every page that has any.

    Pages with no extractable text are skipped, but page numbers of the
    remaining pages are preserved so citations stay correct.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        InvalidPDFError: not a ``.pdf`` file, or the file is corrupt.
        EncryptedPDFError: the PDF needs a password.
        NoExtractableTextError: no page contains text (likely a scan).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError(f"Expected a .pdf file, got '{path.suffix or 'no extension'}'.")

    try:
        reader = PdfReader(str(path))
        _ensure_readable(reader)
        pages = _extract_pages(reader, source=path.name)
    except (PdfReadError, PyPdfError, ValueError, KeyError, OSError) as exc:
        raise InvalidPDFError(f"Could not read '{path.name}': {exc}") from exc

    if not pages:
        raise NoExtractableTextError(
            f"'{path.name}' has no extractable text. It may be a scanned document; "
            "run OCR on it first."
        )
    return pages


def _ensure_readable(reader: PdfReader) -> None:
    """Try an empty password on encrypted files; fail clearly if that is not enough."""
    if not reader.is_encrypted:
        return
    # Many PDFs are "encrypted" only to restrict printing and open with an empty password.
    if reader.decrypt("") == 0:
        raise EncryptedPDFError("This PDF is password protected and cannot be processed.")


def _extract_pages(reader: PdfReader, source: str) -> list[PageText]:
    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(source=source, page=number, text=text))
    return pages
