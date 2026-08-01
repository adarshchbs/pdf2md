from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pymupdf

_PYMUPDF_LOCK = threading.RLock()


@contextmanager
def pymupdf_session() -> Generator[None, None, None]:
    """Serialize guarded modern-pipeline access while allowing nested calls."""
    with _PYMUPDF_LOCK:
        yield


@contextmanager
def open_document(
    filename: str | Path | None = None,
    *,
    stream: bytes | bytearray | memoryview | None = None,
    filetype: str | None = None,
) -> Generator[pymupdf.Document, None, None]:
    """Open a document while holding the process-wide lock through its close."""
    with pymupdf_session():
        document = pymupdf.open(filename, stream=stream, filetype=filetype)
        try:
            yield document
        finally:
            document.close()
