from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from benchmarks.canonical import CanonicalPage, validate_page_collection

type PageId = tuple[str, int]


def export_omnidocbench(
    pages: list[CanonicalPage],
    output_dir: Path,
    basenames: Mapping[PageId, str],
    *,
    include_layout_json: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Write one Markdown file per page using only caller-supplied immutable names."""
    ordered = _ordered_pages(pages)
    page_ids = {(page.document_id, page.page_index) for page in ordered}
    mapped_ids = set(basenames)
    if mapped_ids != page_ids:
        missing = sorted(page_ids - mapped_ids)
        unexpected = sorted(mapped_ids - page_ids)
        raise ValueError(
            f"basename mapping IDs do not match pages; missing={missing}, unexpected={unexpected}"
        )

    normalized = {page_id: _normalize_basename(name) for page_id, name in basenames.items()}
    if len(set(normalized.values())) != len(normalized):
        raise ValueError("basename mapping values must be unique")

    planned: list[tuple[Path, str]] = []
    for page in ordered:
        page_id = (page.document_id, page.page_index)
        basename = normalized[page_id]
        planned.append((output_dir / f"{basename}.md", page.markdown))
        if include_layout_json:
            planned.append((output_dir / f"{basename}.layout.json", _layout_json(page)))
    _write_planned(planned, overwrite=overwrite)
    return [path for path, _ in planned]


def _ordered_pages(pages: list[CanonicalPage]) -> list[CanonicalPage]:
    validate_page_collection(pages, require_canonical_order=False)
    return sorted(pages, key=lambda page: (page.document_id, page.page_index))


def _normalize_basename(name: str) -> str:
    if not name or name in {".", ".."}:
        raise ValueError("output basename must be non-empty")
    path = Path(name)
    if path.name != name:
        raise ValueError(f"output basename must not contain a directory: {name!r}")
    if path.suffix == ".md":
        name = path.stem
    elif path.suffix:
        raise ValueError(f"output basename must be extensionless or end in .md: {name!r}")
    if not name or name in {".", ".."}:
        raise ValueError("output basename must contain a stem")
    return name


def _layout_json(page: CanonicalPage) -> str:
    record = {
        "bbox_origin": page.bbox_origin,
        "document_id": page.document_id,
        "elements": [
            {
                "bbox": element.bbox,
                "id": element.id,
                "reading_order": element.reading_order,
                "type": element.element_type,
            }
            for element in page.elements
        ],
        "figures": [{"bbox": figure.bbox, "id": figure.id} for figure in page.figures],
        "page_index": page.page_index,
        "page_size": page.page_size.model_dump(mode="json"),
        "tables": [
            {
                "bbox": table.bbox,
                "cells": [
                    {
                        "bbox": cell.bbox,
                        "column_index": cell.column_index,
                        "colspan": cell.colspan,
                        "id": cell.id,
                        "row_index": cell.row_index,
                        "rowspan": cell.rowspan,
                    }
                    for cell in table.cells
                ],
                "id": table.id,
            }
            for table in page.tables
        ],
    }
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _write_planned(planned: list[tuple[Path, str]], *, overwrite: bool) -> None:
    if not overwrite:
        existing = [path for path, _ in planned if path.exists()]
        if existing:
            raise FileExistsError(existing[0])
    if planned:
        planned[0][0].parent.mkdir(parents=True, exist_ok=True)
    for path, content in planned:
        path.write_text(content, encoding="utf-8", newline="\n")
