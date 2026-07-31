from __future__ import annotations

import hashlib
import importlib.metadata
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

import pymupdf
from pydantic import JsonValue

from app.pdf2md.engine import ExtractedDocument, extract_document_with_catalog
from app.pdf2md.pymupdf_runtime import open_document
from app.pdf2md.schema import DocumentElement, PageFragment
from benchmarks.adapters.base import config_sha256, failed_run, sha256_file
from benchmarks.canonical import (
    AdapterRun,
    BoundingBox,
    CanonicalCell,
    CanonicalElement,
    CanonicalFigure,
    CanonicalPage,
    CanonicalProvenance,
    CanonicalTable,
    ElementType,
    PageSize,
    RuntimeStats,
    canonical_json,
)

_CONFIG: Final[dict[str, object]] = {
    "annotator": "pdf2md-benchmark-adapter",
    "bbox_origin": "top-left",
    "cross_page_projection": "repeat-element-with-page-local-geometry",
    "page_selection": "all",
}
_ELEMENT_TYPES: Final[dict[str, ElementType]] = {
    "paragraph": "paragraph",
    "heading": "heading",
    "list_item": "list_item",
    "table": "table",
    "figure": "figure",
    "caption": "caption",
    "footnote": "footnote",
    "header": "header",
    "footer": "footer",
}


class OurParserAdapter:
    adapter_id = "our_parser"

    def process(self, pdf_path: Path) -> AdapterRun:
        started = time.perf_counter()
        input_sha256 = sha256_file(pdf_path)
        try:
            extracted = extract_document_with_catalog(
                pdf_path,
                annotator=str(_CONFIG["annotator"]),
            )
            with open_document(pdf_path) as document:
                page_sizes = [
                    PageSize(width=float(page.rect.width), height=float(page.rect.height), unit="point")
                    for page in document
                ]
        except pymupdf.FileDataError as error:
            return failed_run(error, time.perf_counter() - started)

        provenance = CanonicalProvenance(
            tool_name="pdf2md",
            tool_version=importlib.metadata.version("pdf2md"),
            mode="typed-pymupdf-parser",
            input_sha256=input_sha256,
            config_sha256=config_sha256(_CONFIG),
        )
        pending_runtime = RuntimeStats(wall_seconds=0, peak_rss_bytes=None)
        pages = project_extracted_document(extracted, page_sizes, pending_runtime, provenance)
        raw_records = [
            {"record_type": "document_element", "value": element.model_dump(mode="json")}
            for element in extracted.elements
        ]
        raw_records.append({
            "record_type": "source_catalog",
            "value": extracted.source_catalog.model_dump(mode="json"),
        })
        runtime = RuntimeStats(wall_seconds=time.perf_counter() - started, peak_rss_bytes=None)
        pages = [page.model_copy(update={"runtime": runtime}) for page in pages]
        return AdapterRun(
            status="success",
            pages=pages,
            raw_records=cast(list[JsonValue], raw_records),
            runtime=runtime,
        )


def project_extracted_document(
    extracted: ExtractedDocument,
    page_sizes: Sequence[PageSize],
    runtime: RuntimeStats,
    provenance: CanonicalProvenance,
) -> list[CanonicalPage]:
    """Project document elements onto every source page they touch.

    A cross-page element is repeated with the same stable source ID and exact
    representation on each touched page. Element, table, and cell boxes are unions
    of only that page's fragments; a cell without a fragment on the projected page
    is retained for grid/span fidelity with a null box. Reading order is compacted
    independently per page. Empty source pages are emitted with empty collections.
    """
    elements_by_page: list[list[DocumentElement]] = [[] for _ in page_sizes]
    for element in extracted.elements:
        page_numbers = sorted({fragment.page_number for fragment in element.fragments})
        for page_number in page_numbers:
            if page_number > len(page_sizes):
                raise ValueError(f"element fragment page {page_number} exceeds source page count")
            elements_by_page[page_number - 1].append(element)

    captions = {element.element_id: element.content for element in extracted.elements}
    pages: list[CanonicalPage] = []
    for page_index, (page_size, source_elements) in enumerate(zip(page_sizes, elements_by_page, strict=True)):
        page_number = page_index + 1
        canonical_elements: list[CanonicalElement] = []
        tables: list[CanonicalTable] = []
        figures: list[CanonicalFigure] = []
        output_fragments: list[str] = []
        for source in source_elements:
            table_id: str | None = None
            figure_id: str | None = None
            if source.structure.table is not None:
                table_id = source.element_id
                tables.append(_project_table(source, page_number))
            if source.structure.figure is not None:
                figure_id = source.element_id
                figure = source.structure.figure
                figures.append(
                    CanonicalFigure(
                        id=figure_id,
                        bbox=_page_bbox(source.fragments, page_number),
                        caption=(
                            captions.get(figure.caption_element_id)
                            if figure.caption_element_id is not None
                            else None
                        ),
                        confidence=None,
                    )
                )
            canonical_elements.append(
                CanonicalElement(
                    id=source.element_id,
                    element_type=_ELEMENT_TYPES.get(source.element_type, "other"),
                    reading_order=len(canonical_elements),
                    text=_plain_text(source),
                    markdown=source.content,
                    bbox=_page_bbox(source.fragments, page_number),
                    confidence=None,
                    include_in_output=source.include_in_output,
                    table_id=table_id,
                    figure_id=figure_id,
                )
            )
            if source.include_in_output and source.content.strip():
                output_fragments.append(source.content)

        page = CanonicalPage(
            document_id=extracted.source_catalog.document_id,
            page_index=page_index,
            page_size=page_size,
            markdown="\n\n".join(output_fragments),
            elements=canonical_elements,
            tables=tables,
            figures=figures,
            runtime=runtime,
            provenance=provenance,
        )
        payload = canonical_json(page.model_dump(mode="json", exclude={"runtime", "output_sha256"})).encode()
        pages.append(page.model_copy(update={"output_sha256": hashlib.sha256(payload).hexdigest()}))
    return pages


def _project_table(element: DocumentElement, page_number: int) -> CanonicalTable:
    structure = element.structure.table
    if structure is None:
        raise ValueError("table projection requires table structure")
    cells = [
        CanonicalCell(
            id=f"{element.element_id}:cell:{cell.row_index}:{cell.column_index}",
            row_index=cell.row_index,
            column_index=cell.column_index,
            rowspan=cell.rowspan,
            colspan=cell.colspan,
            text=cell.text,
            bbox=_page_bbox(cell.fragments, page_number),
            confidence=None,
        )
        for cell in structure.cells
    ]
    return CanonicalTable(
        id=element.element_id,
        row_count=structure.row_count,
        column_count=structure.column_count,
        cells=cells,
        bbox=_page_bbox(element.fragments, page_number),
        markdown=element.content if source_is_markdown(element) else None,
        html=element.content if element.format == "html" else None,
        confidence=None,
    )


def source_is_markdown(element: DocumentElement) -> bool:
    return element.format == "markdown"


def _plain_text(element: DocumentElement) -> str:
    table = element.structure.table
    if table is not None:
        return "\n".join(cell.text for cell in table.cells)
    return element.content


def _page_bbox(fragments: Sequence[PageFragment], page_number: int) -> BoundingBox | None:
    local = [fragment.bbox for fragment in fragments if fragment.page_number == page_number]
    if not local:
        return None
    return (
        min(bbox.x0 for bbox in local),
        min(bbox.y0 for bbox in local),
        max(bbox.x1 for bbox in local),
        max(bbox.y1 for bbox in local),
    )
