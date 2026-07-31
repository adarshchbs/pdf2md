import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.pdf2md.engine import extract_document_elements, render_document

router = APIRouter()


@router.post("/")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    filename = file.filename
    if filename is None or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="File type not supported. Please upload a PDF file.",
        )

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = Path(temp_file.name)
    try:
        elements = await run_in_threadpool(extract_document_elements, temp_file_path)
        return {
            "message": f"Successfully uploaded {filename}",
            "processed_text": render_document(elements),
        }
    finally:
        os.unlink(temp_file_path)
