from __future__ import annotations

import hashlib
import importlib.metadata
import time
from pathlib import Path
from typing import Final, cast

import pymupdf
from pydantic import JsonValue

from app.pdf2md.pymupdf_runtime import open_document
from benchmarks.adapters.base import (
    ImmutableFileSnapshot,
    config_sha256,
    failed_run,
    snapshot_regular_file,
)
from benchmarks.canonical import (
    AdapterRun,
    CanonicalElement,
    CanonicalPage,
    CanonicalProvenance,
    PageSize,
    RuntimeStats,
    canonical_json,
)

_CONFIG: Final[dict[str, object]] = {
    "api": "Page.get_text(text)",
    "bbox_policy": "unsupported-null",
    "empty_page_policy": "no-elements",
    "sort": False,
}


class PyMuPDFTextAdapter:
    """Native-text-only PyMuPDF baseline with no structural inference."""

    adapter_id = "pymupdf_text"

    def process(self, pdf_path: Path) -> AdapterRun:
        started = time.perf_counter()
        snapshot = snapshot_regular_file(pdf_path)
        return self.process_snapshot(snapshot, started=started)

    def process_snapshot(
        self, snapshot: ImmutableFileSnapshot, *, started: float | None = None
    ) -> AdapterRun:
        """Extract from an already verified immutable source-byte snapshot."""
        if started is None:
            started = time.perf_counter()
        try:
            with open_document(stream=snapshot.data, filetype="pdf") as document:
                records = [
                    {
                        "page_index": page_index,
                        "width": float(page.rect.width),
                        "height": float(page.rect.height),
                        "text": str(page.get_text("text", sort=False)),
                    }
                    for page_index in range(document.page_count)
                    for page in (document[page_index],)
                ]
        except pymupdf.FileDataError as error:
            return failed_run(error, time.perf_counter() - started)

        input_sha256 = snapshot.sha256
        pending_runtime = RuntimeStats(wall_seconds=0, peak_rss_bytes=None)
        pages = [
            _canonical_native_text_page(
                page_index=int(record["page_index"]),
                width=float(record["width"]),
                height=float(record["height"]),
                text=str(record["text"]),
                runtime=pending_runtime,
                input_sha256=input_sha256,
            )
            for record in records
        ]
        runtime = RuntimeStats(wall_seconds=time.perf_counter() - started, peak_rss_bytes=None)
        pages = [page.model_copy(update={"runtime": runtime}) for page in pages]
        return AdapterRun(
            status="success",
            pages=pages,
            raw_records=cast(list[JsonValue], records),
            runtime=runtime,
        )


def _native_text_provenance(input_sha256: str) -> CanonicalProvenance:
    """Return the provenance locked to this native adapter implementation."""
    return CanonicalProvenance(
        tool_name="PyMuPDF",
        tool_version=importlib.metadata.version("pymupdf"),
        mode="native-page-text",
        input_sha256=input_sha256,
        config_sha256=config_sha256(_CONFIG),
    )


def _canonical_native_text_page(
    *,
    page_index: int,
    width: float,
    height: float,
    text: str,
    runtime: RuntimeStats,
    input_sha256: str,
) -> CanonicalPage:
    """Build the sole canonical page shape emitted by the native adapter."""
    elements = (
        [
            CanonicalElement(
                id=f"page-{page_index + 1}-paragraph-1",
                element_type="paragraph",
                reading_order=0,
                text=text,
                markdown=text,
                bbox=None,
                confidence=None,
            )
        ]
        if text.strip()
        else []
    )
    page = CanonicalPage(
        document_id=input_sha256,
        page_index=page_index,
        page_size=PageSize(width=width, height=height, unit="point"),
        markdown=text if elements else "",
        elements=elements,
        tables=[],
        figures=[],
        runtime=runtime,
        provenance=_native_text_provenance(input_sha256),
    )
    payload = canonical_json(page.model_dump(mode="json", exclude={"runtime", "output_sha256"})).encode()
    return page.model_copy(update={"output_sha256": hashlib.sha256(payload).hexdigest()})
