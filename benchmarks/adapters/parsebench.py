from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from app.pdf2md.engine import render_document_elements
from app.pdf2md.pymupdf_geometry import unrotated_page_extent
from app.pdf2md.pymupdf_runtime import open_document
from app.pdf2md.schema import (
    DocumentElement,
    PageFragment,
    TableCell,
    inline_footnote_marker_ranges,
)
from app.pdf2md.semantic_text import normalize_page_size, normalize_point
from app.pdf2md.source_catalog import SourceCatalog, SourceItem
from benchmarks.canonical import CanonicalPage, validate_page_collection

UPSTREAM_INGESTION_STATUS = "requires_pinned_checkout"
UPSTREAM_INGESTION_NOTE = (
    "Exact ParseBench upstream ingestion requires a pinned upstream checkout; "
    "this credential-free export is a deterministic interchange record, not an official upstream schema."
)
ATTRIBUTION_PROJECTION_VERSION = "parsebench-layout-attribution-v2"
_LABELS: Final[dict[str, str]] = {
    "paragraph": "text",
    "heading": "heading",
    "list_item": "text",
    "table": "table",
    "figure": "image",
    "caption": "figure_title",
    "footnote": "footnote",
    "header": "header",
    "footer": "footer",
}


@dataclass(frozen=True, slots=True)
class ParseBenchPageGeometry:
    source_width: float
    source_height: float
    rotation: int = 0
    output_width: float | None = None
    output_height: float | None = None
    crop_x0: float = 0.0
    crop_y0: float = 0.0

    def __post_init__(self) -> None:
        source = _validate_page_size((self.source_width, self.source_height), 1)
        display = normalize_page_size(*source, self.rotation)
        if (self.output_width is None) != (self.output_height is None):
            raise ValueError("ParseBench output dimensions must both be present or both be null")
        output = (
            display
            if self.output_width is None or self.output_height is None
            else _validate_page_size((self.output_width, self.output_height), 1)
        )
        if not all(math.isfinite(value) for value in (self.crop_x0, self.crop_y0)):
            raise ValueError("crop origin must be finite")
        object.__setattr__(self, "source_width", source[0])
        object.__setattr__(self, "source_height", source[1])
        object.__setattr__(self, "output_width", output[0])
        object.__setattr__(self, "output_height", output[1])

    @property
    def display_size(self) -> tuple[float, float]:
        return normalize_page_size(self.source_width, self.source_height, self.rotation)

    @property
    def output_size(self) -> tuple[float, float]:
        if self.output_width is None or self.output_height is None:
            raise AssertionError("validated page geometry lost output dimensions")
        return self.output_width, self.output_height


def parsebench_page_geometries(pdf_path: Path) -> list[ParseBenchPageGeometry]:
    """Describe source-to-upright page transforms without using rotation-adjusted input extents."""
    with open_document(pdf_path) as document:
        result: list[ParseBenchPageGeometry] = []
        for page in document:
            source_width, source_height = unrotated_page_extent(page)
            cropbox = page.cropbox
            result.append(
                ParseBenchPageGeometry(
                    source_width=source_width,
                    source_height=source_height,
                    rotation=int(page.rotation),
                    crop_x0=float(cropbox.x0),
                    crop_y0=float(cropbox.y0),
                )
            )
        return result


def parsebench_source_page_extents(pdf_path: Path) -> list[tuple[float, float]]:
    """Legacy helper returning only unrotated source extents."""
    return [(page.source_width, page.source_height) for page in parsebench_page_geometries(pdf_path)]


def project_parsebench_layout_pages(
    elements: Sequence[DocumentElement],
    source_catalog: SourceCatalog | None,
    page_sizes: Sequence[tuple[float, float] | ParseBenchPageGeometry],
) -> list[dict[str, Any]]:
    """Project only source-grounded attribution into ParseBench's local layout boundary.

    Display-frame fragments and source-frame native items are converted to one
    normalized upright frame. Native spans become ``layoutAwareBbox`` records
    with exact inclusive semantic-source offsets. Tables
    retain one geometry-only parent item and emit one source-grounded text item
    per non-empty logical cell, because ParseBench's table attribution adapter
    applies a table item's complete value to every segment.

    This v2 projection is deliberately strict and separate from the legacy v1
    canonical interchange exporter below. It never infers text from geometry or
    reuses a native text item across attributed owners.
    """
    if source_catalog is None:
        raise ValueError("a source catalog is required for ParseBench attribution projection")
    if source_catalog.migrated_from_schema_version is not None:
        raise ValueError("legacy source catalogs are incompatible with attribution projection v2")
    if any(item.id_scheme != "native-content-v1" for item in source_catalog.items):
        raise ValueError("attribution projection requires native-content-v1 source item IDs")
    geometries = [
        page if isinstance(page, ParseBenchPageGeometry) else ParseBenchPageGeometry(*page)
        for page in page_sizes
    ]
    if not geometries:
        raise ValueError("ParseBench attribution projection requires at least one source page")

    ordered_elements = sorted(elements, key=lambda element: (element.order, element.element_id))
    _validate_projection_document(ordered_elements, source_catalog, len(geometries))
    items_by_id = {item.source_item_id: item for item in source_catalog.items}
    rendered_elements = render_document_elements(ordered_elements)
    rendered_by_id = {item.element_id: item.markdown for item in rendered_elements}
    canonical_document_markdown = "\n\n".join(item.markdown for item in rendered_elements)
    projected_by_page: list[list[dict[str, Any]]] = [[] for _ in geometries]
    claimed_source_ids: dict[str, str] = {}

    for element in ordered_elements:
        label = _LABELS.get(element.element_type, "text")
        page_numbers = sorted({fragment.page_number for fragment in element.fragments})
        if element.structure.table is not None:
            _validate_fragment_references(
                element.fragments,
                items_by_id,
                source_catalog.document_id,
                expected_kind=None,
                owner=f"table element {element.element_id}",
            )
            for page_number in page_numbers:
                local_fragments = _fragments_on_page(element.fragments, page_number)
                parent_bbox = _normalized_fragment_bbox(
                    local_fragments,
                    geometries[page_number - 1],
                    f"table {element.element_id}",
                )
                projected_by_page[page_number - 1].append({
                    "id": f"{element.element_id}:page:{page_number:06d}",
                    "type": "geometry",
                    "value": "",
                    "md": "",
                    "html": "",
                    "bBox": {**_bbox_payload(parent_bbox), "label": "table"},
                    "layoutAwareBbox": [],
                    "projectionVersion": ATTRIBUTION_PROJECTION_VERSION,
                })
            table = element.structure.table
            for cell in sorted(
                table.cells,
                key=lambda value: (
                    value.row_index,
                    value.column_index,
                    value.rowspan,
                    value.colspan,
                ),
            ):
                _project_table_cell(
                    element,
                    cell,
                    source_catalog,
                    items_by_id,
                    geometries,
                    projected_by_page,
                    claimed_source_ids,
                )
            continue

        references = _ordered_references(
            element.fragments,
            items_by_id,
            source_catalog.document_id,
            expected_kind="span",
            owner=f"element {element.element_id}",
        )
        if not element.content.strip() and references:
            raise ValueError(f"empty element {element.element_id} cannot cite native source items")
        attribution_value = _source_semantic_value(element)
        spans = _attribution_segments(
            value=attribution_value,
            references=references,
            label=label,
            owner=f"element {element.element_id}",
            page_geometries=geometries,
            claimed_source_ids=claimed_source_ids,
        )
        spans_by_page: dict[int, list[dict[str, Any]]] = {}
        for page_number, segment in spans:
            spans_by_page.setdefault(page_number, []).append(segment)
        for page_number in page_numbers:
            local_fragments = _fragments_on_page(element.fragments, page_number)
            element_bbox = _normalized_fragment_bbox(
                local_fragments,
                geometries[page_number - 1],
                f"element {element.element_id}",
            )
            projected_by_page[page_number - 1].append({
                "id": f"{element.element_id}:page:{page_number:06d}",
                "type": "image" if element.element_type == "figure" else "text",
                "value": attribution_value if spans else "",
                "md": rendered_by_id.get(element.element_id, ""),
                "html": "",
                "bBox": _bbox_payload(element_bbox),
                "layoutAwareBbox": spans_by_page.get(page_number, []),
                "projectionVersion": ATTRIBUTION_PROJECTION_VERSION,
            })

    pages: list[dict[str, Any]] = []
    for page_number, (geometry, items) in enumerate(zip(geometries, projected_by_page, strict=True), start=1):
        width, height = geometry.output_size
        page_elements = [
            element
            for element in ordered_elements
            if any(fragment.page_number == page_number for fragment in element.fragments)
        ]
        markdown = "\n\n".join(
            rendered_by_id[element.element_id]
            for element in page_elements
            if element.element_id in rendered_by_id
            and page_number == min(fragment.page_number for fragment in element.fragments)
        )
        pages.append({
            "page": page_number,
            "width": width,
            "height": height,
            "md": markdown,
            "text": markdown,
            "items": items,
            "projectionVersion": ATTRIBUTION_PROJECTION_VERSION,
        })
    projected_document_markdown = "\n\n".join(page["md"] for page in pages if page["md"])
    if projected_document_markdown != canonical_document_markdown:
        raise ValueError("page-local Markdown does not preserve canonical document rendering")
    return pages


def _project_table_cell(
    element: DocumentElement,
    cell: TableCell,
    catalog: SourceCatalog,
    items_by_id: dict[str, SourceItem],
    page_geometries: list[ParseBenchPageGeometry],
    projected_by_page: list[list[dict[str, Any]]],
    claimed_source_ids: dict[str, str],
) -> None:
    owner = (
        f"table {element.element_id} cell "
        f"({cell.row_index},{cell.column_index},{cell.rowspan},{cell.colspan})"
    )
    references = _ordered_references(
        cell.fragments,
        items_by_id,
        catalog.document_id,
        expected_kind="word",
        owner=owner,
    )
    if not cell.text.strip():
        if references:
            raise ValueError("empty table cells cannot cite native source items")
        return
    if not references:
        return
    spans = _attribution_segments(
        value=cell.text,
        references=references,
        label="text",
        owner=owner,
        page_geometries=page_geometries,
        claimed_source_ids=claimed_source_ids,
    )
    spans_by_page: dict[int, list[dict[str, Any]]] = {}
    for page_number, segment in spans:
        spans_by_page.setdefault(page_number, []).append(segment)
    for page_number in sorted(spans_by_page):
        local_fragments = _fragments_on_page(cell.fragments, page_number)
        cell_bbox = _normalized_fragment_bbox(
            local_fragments,
            page_geometries[page_number - 1],
            owner,
        )
        projected_by_page[page_number - 1].append({
            "id": (
                f"{element.element_id}:cell:{cell.row_index:06d}:{cell.column_index:06d}:"
                f"{cell.rowspan:06d}:{cell.colspan:06d}:page:{page_number:06d}"
            ),
            "type": "text",
            "value": cell.text,
            "md": cell.text,
            "html": "",
            "bBox": _bbox_payload(cell_bbox),
            "layoutAwareBbox": spans_by_page[page_number],
            "tableId": element.element_id,
            "rowIndex": cell.row_index,
            "columnIndex": cell.column_index,
            "rowspan": cell.rowspan,
            "colspan": cell.colspan,
            "projectionVersion": ATTRIBUTION_PROJECTION_VERSION,
        })


def _validate_projection_document(
    elements: list[DocumentElement], catalog: SourceCatalog, page_count: int
) -> None:
    document_ids = {element.document_id for element in elements}
    if document_ids and document_ids != {catalog.document_id}:
        raise ValueError("elements and source catalog must have exactly one matching document_id")
    orders = [element.order for element in elements]
    if sorted(orders) != list(range(len(elements))):
        raise ValueError("element order must be unique and contiguous from zero")
    ids = [element.element_id for element in elements]
    if len(ids) != len(set(ids)):
        raise ValueError("element IDs must be unique within the projected document")
    for element in elements:
        for fragment in element.fragments:
            if fragment.page_number > page_count:
                raise ValueError(
                    f"element {element.element_id} fragment page {fragment.page_number} exceeds source page count"
                )


def _ordered_references(
    fragments: Sequence[PageFragment],
    items_by_id: dict[str, SourceItem],
    document_id: str,
    *,
    expected_kind: str,
    owner: str,
) -> list[SourceItem]:
    _validate_fragment_references(
        fragments,
        items_by_id,
        document_id,
        expected_kind=expected_kind,
        owner=owner,
    )
    source_ids = [source_id for fragment in fragments for source_id in fragment.source_item_ids]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"a native source item cannot be repeated within {owner}")
    return sorted(
        (items_by_id[source_id] for source_id in source_ids),
        key=lambda item: (
            item.page_number,
            item.y0,
            item.x0,
            item.y1,
            item.x1,
            item.source_item_id,
        ),
    )


def _validate_fragment_references(
    fragments: Sequence[PageFragment],
    items_by_id: dict[str, SourceItem],
    document_id: str,
    *,
    expected_kind: str | None,
    owner: str,
) -> None:
    for fragment in fragments:
        for source_id in fragment.source_item_ids:
            item = items_by_id.get(source_id)
            if item is None:
                raise ValueError(f"{owner} references unknown source item: {source_id}")
            if item.document_id != document_id:
                raise ValueError(f"{owner} source item document_id mismatch: {source_id}")
            if item.page_number != fragment.page_number:
                raise ValueError(f"{owner} source item page mismatch: {source_id}")
            if expected_kind is not None and item.coordinate_frame != "source_page":
                raise ValueError("attribution projection requires the source_page coordinate frame")
            if expected_kind is not None and item.kind != expected_kind:
                requirement = (
                    "paragraph attribution requires native spans"
                    if expected_kind == "span"
                    else "table cell attribution requires native words"
                )
                raise ValueError(f"{requirement}: {source_id}")


def _attribution_segments(
    *,
    value: str,
    references: Sequence[SourceItem],
    label: str,
    owner: str,
    page_geometries: list[ParseBenchPageGeometry],
    claimed_source_ids: dict[str, str],
) -> list[tuple[int, dict[str, Any]]]:
    if not references:
        return []
    normalized_references: list[tuple[SourceItem, str]] = []
    for item in references:
        raw_text = item.text
        if raw_text is None or not raw_text:
            raise ValueError(f"{owner} cites a native text item without content: {item.source_item_id}")
        normalized_text, normalized_boundaries = _nfkc_raw_to_normalized_boundaries(raw_text)
        if not normalized_text or normalized_boundaries[-1] != len(normalized_text):
            raise ValueError(
                f"{owner} cites native text that normalizes to empty or inconsistent content: "
                f"{item.source_item_id}"
            )
        normalized_references.append((item, normalized_text))
    ranges = _exact_source_ranges(
        value,
        tuple(text for _, text in normalized_references),
        owner=owner,
    )

    result: list[tuple[int, dict[str, Any]]] = []
    for (item, _normalized_text), (start, end_exclusive) in zip(
        normalized_references,
        ranges,
        strict=True,
    ):
        previous_owner = claimed_source_ids.get(item.source_item_id)
        if previous_owner is not None:
            raise ValueError(
                f"native source item {item.source_item_id} cannot be shared by {previous_owner} and {owner}"
            )
        claimed_source_ids[item.source_item_id] = owner
        end = end_exclusive - 1
        coordinates = _normalized_source_bbox(
            item,
            page_geometries[item.page_number - 1],
            owner,
        )
        result.append((
            item.page_number,
            {
                **_bbox_payload(coordinates),
                "label": label,
                "startIndex": start,
                "endIndex": end,
                "sourceItemId": item.source_item_id,
            },
        ))
    return result


def _exact_source_ranges(
    value: str,
    pieces: tuple[str, ...],
    *,
    owner: str,
) -> tuple[tuple[int, int], ...]:
    solutions: list[tuple[tuple[int, int], ...]] = []

    def visit(piece_index: int, cursor: int, ranges: tuple[tuple[int, int], ...]) -> None:
        if len(solutions) > 1:
            return
        if piece_index == len(pieces):
            if value[cursor:].isspace() or cursor == len(value):
                solutions.append(ranges)
            return
        piece = pieces[piece_index]
        for start in range(cursor, len(value) - len(piece) + 1):
            if start > cursor and not value[cursor:start].isspace():
                break
            if value.startswith(piece, start):
                visit(
                    piece_index + 1,
                    start + len(piece),
                    (*ranges, (start, start + len(piece))),
                )

    visit(0, 0, ())
    if not solutions:
        raise ValueError(f"native source text for {owner} cannot be projected exactly")
    if len(solutions) != 1:
        raise ValueError(f"native source text projection for {owner} is ambiguous")
    return solutions[0]


def _source_semantic_value(element: DocumentElement) -> str:
    ranges = inline_footnote_marker_ranges(element)
    if not ranges:
        return element.content
    parts: list[str] = []
    cursor = 0
    for marker in ranges:
        parts.append(element.content[cursor : marker.start])
        parts.append(marker.label)
        cursor = marker.end
    parts.append(element.content[cursor:])
    return "".join(parts)


def _normalized_source_bbox(
    item: SourceItem,
    geometry: ParseBenchPageGeometry,
    owner: str,
) -> tuple[float, float, float, float]:
    source_bbox = (item.x0, item.y0, item.x1, item.y1)
    _validate_bbox_on_page(
        source_bbox,
        (geometry.source_width, geometry.source_height),
        owner,
        allow_zero_width=True,
    )
    points = (
        normalize_point(item.x0, item.y0, geometry.source_width, geometry.source_height, geometry.rotation),
        normalize_point(item.x1, item.y0, geometry.source_width, geometry.source_height, geometry.rotation),
        normalize_point(item.x0, item.y1, geometry.source_width, geometry.source_height, geometry.rotation),
        normalize_point(item.x1, item.y1, geometry.source_width, geometry.source_height, geometry.rotation),
    )
    display_width, display_height = geometry.display_size
    xs = [point[0] / display_width for point in points]
    ys = [point[1] / display_height for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _normalized_fragment_bbox(
    fragments: Sequence[PageFragment],
    geometry: ParseBenchPageGeometry,
    owner: str,
) -> tuple[float, float, float, float]:
    bbox = _fragments_bbox(fragments)
    dimensions = {(fragment.page_width, fragment.page_height) for fragment in fragments}
    if len(dimensions) != 1 or None in next(iter(dimensions)):
        raise ValueError(f"{owner} fragments require one explicit page coordinate extent")
    page_width, page_height = next(iter(dimensions))
    if page_width is None or page_height is None:
        raise AssertionError("validated fragment dimensions became null")
    _validate_bbox_on_page(bbox, (page_width, page_height), owner)
    display_width, display_height = geometry.display_size
    width_scale = page_width / display_width
    height_scale = page_height / display_height
    if not math.isclose(width_scale, height_scale, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{owner} fragment coordinates are incompatible with page rotation and scaling")
    return (
        bbox[0] / page_width,
        bbox[1] / page_height,
        bbox[2] / page_width,
        bbox[3] / page_height,
    )


def _nfkc_raw_to_normalized_boundaries(raw_text: str) -> tuple[str, tuple[int, ...]]:
    """Map every raw prefix boundary into the exact NFKC-normalized string."""
    normalized = unicodedata.normalize("NFKC", raw_text)
    boundaries = tuple(
        len(unicodedata.normalize("NFKC", raw_text[:raw_index])) for raw_index in range(len(raw_text) + 1)
    )
    if boundaries[0] != 0 or boundaries[-1] != len(normalized):
        raise ValueError("NFKC raw-to-normalized boundary mapping is inconsistent")
    if any(right < left for left, right in zip(boundaries, boundaries[1:], strict=False)):
        raise ValueError("NFKC raw-to-normalized boundaries must be monotonic")
    return normalized, boundaries


def _fragments_on_page(fragments: Sequence[PageFragment], page_number: int) -> list[PageFragment]:
    return [fragment for fragment in fragments if fragment.page_number == page_number]


def _fragments_bbox(fragments: Sequence[PageFragment]) -> tuple[float, float, float, float]:
    if not fragments:
        raise ValueError("cannot project geometry from an empty page-fragment collection")
    return (
        min(fragment.bbox.x0 for fragment in fragments),
        min(fragment.bbox.y0 for fragment in fragments),
        max(fragment.bbox.x1 for fragment in fragments),
        max(fragment.bbox.y1 for fragment in fragments),
    )


def _bbox_payload(bbox: tuple[float, float, float, float]) -> dict[str, float]:
    x0, y0, x1, y1 = bbox
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _validate_page_size(size: tuple[float, float], page_number: int) -> tuple[float, float]:
    width, height = size
    if not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError(f"source page {page_number} dimensions must be positive and finite")
    return float(width), float(height)


def _validate_bbox_on_page(
    bbox: tuple[float, float, float, float],
    page_size: tuple[float, float],
    owner: str,
    *,
    allow_zero_width: bool = False,
) -> None:
    x0, y0, x1, y1 = bbox
    width, height = page_size
    if (
        not all(math.isfinite(value) for value in bbox)
        or x1 < x0
        or (x1 == x0 and not allow_zero_width)
        or y1 <= y0
        or x0 < 0
        or y0 < 0
        or x1 > width
        or y1 > height
    ):
        raise ValueError(f"{owner} has geometry outside its declared source page")


def export_parsebench_records(pages: list[CanonicalPage]) -> list[dict[str, Any]]:
    """Return deterministic, JSON-compatible page records without losing structure."""
    validate_page_collection(pages, require_canonical_order=False)
    records: list[dict[str, Any]] = []
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
