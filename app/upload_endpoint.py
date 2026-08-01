import os
import tempfile
from pathlib import Path

import anyio
from anyio import CapacityLimiter, to_thread
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.pdf2md.engine import extract_document_elements, render_document

router = APIRouter()
WORK_LIMITER = CapacityLimiter(4)
_UPLOAD_CHUNK_SIZE = 1024 * 1024


def _make_temp_path() -> Path:
    descriptor, name = tempfile.mkstemp(suffix=".pdf")
    os.close(descriptor)
    return Path(name)


def _append_chunk(path: Path, chunk: bytes) -> None:
    with path.open("ab") as output:
        output.write(chunk)


def _remove_temp_path(path: Path) -> None:
    path.unlink(missing_ok=True)


@router.post("/")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    filename = file.filename
    if filename is None or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File type not supported. Please upload a PDF file.",
        )

    temp_file_path = await to_thread.run_sync(_make_temp_path, limiter=WORK_LIMITER)
    try:
        while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
            await to_thread.run_sync(_append_chunk, temp_file_path, chunk, limiter=WORK_LIMITER)
        elements = await to_thread.run_sync(extract_document_elements, temp_file_path, limiter=WORK_LIMITER)
        rendered = await to_thread.run_sync(render_document, elements, limiter=WORK_LIMITER)
        return {"message": f"Successfully uploaded {filename}", "processed_text": rendered}
    finally:
        with anyio.CancelScope(shield=True):
            await to_thread.run_sync(_remove_temp_path, temp_file_path, limiter=WORK_LIMITER)
