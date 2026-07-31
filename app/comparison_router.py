from __future__ import annotations

import hashlib
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Literal

import anyio
from anyio import CapacityLimiter, to_thread
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.pdf2md.comparison import ComparisonCatalog, ResolvedArtifact, Split, roots_from_environment

router = APIRouter(prefix="/comparison", tags=["comparison"])
WORK_LIMITER = CapacityLimiter(4)
ACTIVE_STREAM_LIMITER = CapacityLimiter(4)
_CHUNK_SIZE = 1024 * 1024


def _catalog() -> ComparisonCatalog:
    root = Path(os.environ.get("PDF2MD_PROJECT_ROOT", "."))
    return ComparisonCatalog(
        root,
        candidate_roots=roots_from_environment("PDF2MD_CANDIDATE_ROOTS", root=root),
        reference_roots=roots_from_environment("PDF2MD_REFERENCE_ROOTS", root=root),
    )


def _not_found(error: FileNotFoundError | KeyError | ValueError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error))


@router.get("/metadata")
async def metadata() -> dict[str, object]:
    return {
        "tool": "pdf2md comparison dashboard backend",
        "mode": "non-blind engineering comparison",
        "review_status": "not human review",
        "gold": None,
        "partition": "partition-v7",
    }


@router.get("/catalog")
async def catalog(split: Split | None = None) -> dict[str, object]:
    return await documents(split)


@router.get("/documents")
async def documents(split: Split | None = None) -> dict[str, object]:
    try:
        catalog_instance = await to_thread.run_sync(_catalog, limiter=WORK_LIMITER)
        entries = await to_thread.run_sync(
            lambda: catalog_instance.comparison_summaries(split), limiter=WORK_LIMITER
        )
        return {"partition": "partition-v7", "split": split, "documents": entries}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/documents/{document_id}")
async def document(document_id: str) -> dict[str, object]:
    try:
        catalog_instance = await to_thread.run_sync(_catalog, limiter=WORK_LIMITER)
        return await to_thread.run_sync(
            lambda: catalog_instance.comparison(document_id), limiter=WORK_LIMITER
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise _not_found(error) from error


@router.get("/documents/{document_id}/elements/{source}")
async def elements(document_id: str, source: Literal["candidate", "reference"]) -> dict[str, object]:
    try:
        catalog_instance = await to_thread.run_sync(_catalog, limiter=WORK_LIMITER)
        result = await to_thread.run_sync(
            lambda: catalog_instance.elements(document_id, source), limiter=WORK_LIMITER
        )
        result["document_id"] = document_id
        return result
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise _not_found(error) from error


@router.get("/documents/{document_id}/evaluation")
async def evaluation(document_id: str) -> dict[str, object]:
    try:
        catalog_instance = await to_thread.run_sync(_catalog, limiter=WORK_LIMITER)
        report = await to_thread.run_sync(
            lambda: catalog_instance.evaluation(document_id), limiter=WORK_LIMITER
        )
        return report.model_dump(mode="json")
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise _not_found(error) from error


def _open_descriptor(artifact: ResolvedArtifact, expected: os.stat_result) -> tuple[int, int]:
    path, root = artifact.path, artifact.root
    if path is None or root is None:
        raise FileNotFoundError("source PDF is unavailable")
    relative = path.relative_to(root)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise OSError("source PDF is outside its trusted root")
    directory_fds: list[int] = []
    file_fd: int | None = None
    try:
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory_fds.append(directory_fd)
        for component in relative.parts[:-1]:
            directory_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd
            )
            directory_fds.append(directory_fd)
        file_fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fds[-1])
        stat = os.fstat(file_fd)
        identity = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        expected_identity = (expected.st_dev, expected.st_ino, expected.st_mtime_ns, expected.st_size)
        if identity != expected_identity:
            os.close(file_fd)
            file_fd = None
            raise OSError("source PDF changed during validation")
        if artifact.sha256 is not None:
            digest = hashlib.sha256()
            os.lseek(file_fd, 0, os.SEEK_SET)
            while chunk := os.read(file_fd, _CHUNK_SIZE):
                digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise OSError("source PDF changed during validation")
        result = file_fd, stat.st_size
        file_fd = None
        return result  # ownership transfers to the async generator
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)


async def stream_artifact(
    artifact: ResolvedArtifact, expected: os.stat_result, start: int, length: int
) -> AsyncGenerator[bytes, None]:
    fd: int | None = None
    acquired = False
    try:
        await ACTIVE_STREAM_LIMITER.acquire()
        acquired = True
        with anyio.CancelScope(shield=True):
            fd, _ = await to_thread.run_sync(_open_descriptor, artifact, expected, limiter=WORK_LIMITER)
        if fd is None:
            raise OSError("descriptor open returned no file descriptor")
        await to_thread.run_sync(lambda: os.lseek(fd, start, os.SEEK_SET), limiter=WORK_LIMITER)
        remaining = length
        while remaining:
            chunk = await to_thread.run_sync(
                lambda: os.read(fd, min(_CHUNK_SIZE, remaining)), limiter=WORK_LIMITER
            )
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk
    finally:
        if fd is not None:
            with anyio.CancelScope(shield=True):
                await to_thread.run_sync(lambda: os.close(fd), limiter=WORK_LIMITER)
        if acquired:
            with anyio.CancelScope(shield=True):
                ACTIVE_STREAM_LIMITER.release()


def parse_range(header: str | None, size: int) -> tuple[int, int, int] | None:
    if header is None:
        return None
    match = re.fullmatch(r"bytes=(?:([0-9]+)-([0-9]*)|-[0-9]+)", header, flags=re.ASCII)
    if match is None or "," in header:
        raise HTTPException(
            status_code=416, detail="invalid byte range", headers={"Content-Range": f"bytes */{size}"}
        )
    value = header[6:]
    first, last = value.split("-", 1)
    try:
        if first == "":
            suffix = int(last, 10)
            if suffix <= 0:
                raise ValueError
            start, end = max(0, size - suffix), size - 1
        else:
            start = int(first, 10)
            end = size - 1 if last == "" else int(last, 10)
            if end < start:
                raise ValueError
            end = min(end, size - 1)
        if start >= size:
            raise ValueError
    except ValueError as error:
        raise HTTPException(
            status_code=416, detail="unsatisfiable byte range", headers={"Content-Range": f"bytes */{size}"}
        ) from error
    return start, end, end - start + 1


async def _source_artifact(
    document_id: str, source: Literal["candidate", "reference", "original"]
) -> ResolvedArtifact:
    catalog_instance = await to_thread.run_sync(_catalog, limiter=WORK_LIMITER)
    return await to_thread.run_sync(
        lambda: catalog_instance.artifact(document_id, source, "pdf"), limiter=WORK_LIMITER
    )


async def _serve_source(
    document_id: str,
    source: Literal["candidate", "reference", "original"],
    request: Request,
    *,
    head: bool = False,
) -> Response:
    try:
        artifact = await _source_artifact(document_id, source)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise _not_found(error) from error
    if not artifact.available or artifact.path is None:
        raise HTTPException(status_code=404, detail=artifact.reason or "source PDF is unavailable")
    try:
        expected = await to_thread.run_sync(artifact.path.stat, limiter=WORK_LIMITER)
    except OSError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    size = expected.st_size
    selected = parse_range(request.headers.get("range"), size)
    if selected is None:
        start, length, status = 0, size, 200
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(size)}
    else:
        start, end, length = selected
        status = 206
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Range": f"bytes {start}-{end}/{size}",
        }
    if head:
        return Response(status_code=status, media_type="application/pdf", headers=headers)
    return StreamingResponse(
        stream_artifact(artifact, expected, start, length),
        status_code=status,
        media_type="application/pdf",
        headers=headers,
    )


@router.api_route("/documents/{document_id}/source/{source}", methods=["GET"])
async def source_pdf(
    document_id: str, source: Literal["candidate", "reference", "original"], request: Request
) -> Response:
    return await _serve_source(document_id, source, request)


@router.head("/documents/{document_id}/source/{source}")
async def source_pdf_head(
    document_id: str, source: Literal["candidate", "reference", "original"], request: Request
) -> Response:
    return await _serve_source(document_id, source, request, head=True)
