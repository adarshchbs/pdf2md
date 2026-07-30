from __future__ import annotations

import json
from pathlib import Path

from benchmarks.canonical import CanonicalPage, validate_page_collection

UPSTREAM_INGESTION_STATUS = "requires_pinned_checkout"
UPSTREAM_INGESTION_NOTE = (
    "Exact ParseBench upstream ingestion requires a pinned upstream checkout; "
    "this credential-free export is a deterministic interchange record, not an official upstream schema."
)


def export_parsebench_records(pages: list[CanonicalPage]) -> list[dict[str, object]]:
    """Return deterministic, JSON-compatible page records without losing structure."""
    validate_page_collection(pages, require_canonical_order=False)
    records: list[dict[str, object]] = []
    for page in sorted(pages, key=lambda item: (item.document_id, item.page_index)):
        records.append({
            "adapter": "parsebench",
            "format": "canonical-parsebench-interchange-v1",
            "upstream_ingestion": {
                "status": UPSTREAM_INGESTION_STATUS,
                "note": UPSTREAM_INGESTION_NOTE,
            },
            "document_id": page.document_id,
            "page_index": page.page_index,
            "page_size": page.page_size.model_dump(mode="json"),
            "bbox_origin": page.bbox_origin,
            "markdown": page.markdown,
            "elements": [element.model_dump(mode="json") for element in page.elements],
            "tables": [table.model_dump(mode="json") for table in page.tables],
            "figures": [figure.model_dump(mode="json") for figure in page.figures],
        })
    return records


def write_parsebench_jsonl(pages: list[CanonicalPage], output_path: Path, *, overwrite: bool = False) -> None:
    """Write interchange records as stable UTF-8 JSONL."""
    if output_path.suffix != ".jsonl":
        raise ValueError("ParseBench interchange output must use the .jsonl suffix")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)

    records = export_parsebench_records(pages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            f"{json.dumps(record, ensure_ascii=False, separators=(',', ':'), sort_keys=True, allow_nan=False)}\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
