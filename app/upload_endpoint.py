from fastapi import APIRouter, File, UploadFile, HTTPException
from .pymupdf_parser.main import main
import shutil
import os
from tempfile import NamedTemporaryFile

router = APIRouter()


@router.post("/")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="File type not supported. Please upload a PDF file."
        )
    with NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    try:
        processed_text = main(temp_file_path)
        return {
            "message": f"Successfully uploaded {file.filename}",
            "processed_text": processed_text,
        }
    finally:
        os.unlink(temp_file_path)
