from __future__ import annotations

import hashlib
import importlib.metadata
import time
from pathlib import Path
from typing import Final, cast

import pymupdf
from pydantic import JsonValue

from benchmarks.adapters.base import config_sha256, failed_run, sha256_file
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
        input_sha256 = sha256_file(pdf_path)
        try:
            with pymupdf.open(pdf_path) as document:
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

        provenance = CanonicalProvenance(
            tool_name="PyMuPDF",
            tool_version=importlib.metadata.version("pymupdf"),
            mode="native-page-text",
            input_sha256=input_sha256,
            config_sha256=config_sha256(_CONFIG),
        )
        pending_runtime = RuntimeStats(wall_seconds=0, peak_rss_bytes=None)
        pages = [_page_from_record(input_sha256, record, pending_runtime, provenance) for record in records]
        runtime = RuntimeStats(wall_seconds=time.perf_counter() - started, peak_rss_bytes=None)
        pages = [page.model_copy(update={"runtime": runtime}) for page in pages]
        return AdapterRun(
            status="success",
            pages=pages,
            raw_records=cast(list[JsonValue], records),
            runtime=runtime,
        )


def _page_from_record(
    document_id: str,
    record: dict[str, int | float | str],
    runtime: RuntimeStats,
    provenance: CanonicalProvenance,
) -> CanonicalPage:
    page_index = int(record["page_index"])
    text = str(record["text"])
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
        document_id=document_id,
        page_index=page_index,
        page_size=PageSize(width=float(record["width"]), height=float(record["height"]), unit="point"),
        markdown=text if elements else "",
        elements=elements,
        tables=[],
        figures=[],
        runtime=runtime,
        provenance=provenance,
    )
    payload = canonical_json(page.model_dump(mode="json", exclude={"runtime", "output_sha256"})).encode()
    return page.model_copy(update={"output_sha256": hashlib.sha256(payload).hexdigest()})
