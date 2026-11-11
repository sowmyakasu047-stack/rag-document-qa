"""Exceptions raised by the ingestion pipeline.

Every failure a user can cause (bad file, locked file, scanned file) gets its
own exception type so the API layer can map each one to a clear message
instead of a generic 500 error.
"""


class RagError(Exception):
    """Base class for all errors raised by this project."""


class InvalidPDFError(RagError):
    """The file is not a readable PDF (wrong extension, corrupt, truncated)."""


class EncryptedPDFError(RagError):
    """The PDF is password protected and cannot be opened without a password."""


class NoExtractableTextError(RagError):
    """The PDF opened fine but contains no text layer (e.g. a scanned image)."""
