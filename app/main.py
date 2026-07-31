import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .comparison_router import router as comparison_router
from .upload_endpoint import router as upload_router

app = FastAPI()

configured_origins = tuple(
    origin
    for origin in os.environ.get("PDF2MD_CORS_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080").split(
        ","
    )
    if origin
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(configured_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/upload")
app.include_router(comparison_router)


@app.get("/api/health")
@app.get("/health")
async def health() -> dict[str, str]:
    return {"message": "PDF Parser v4 is online"}


# API routes are registered first, so they retain precedence over the frontend mount.
app.frontend(
    "/",
    directory=Path(__file__).resolve().parent.parent / "frontend/dist",
    fallback="index.html",
    check_dir=False,
)
