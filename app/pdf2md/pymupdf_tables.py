from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import median
from typing import Protocol, cast

import pymupdf

from app.pdf2md.canonical_table_view import canonical_table_view
from app.pdf2md.continuations import merge_table_continuations
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    PageFragment,
    StructureProperty,
    TableCell,
    TableStructure,
)
from app.pdf2md.source_catalog import SourceItem, make_source_item
from app.pdf2md.table_provenance import NativeRule, TableProvenance, TableReconstruction, reconstruct_table
from app.pdf2md.table_quality import validate_table_candidate
from app.pdf2md.tables import TableFeatures, classify_table, render_table

_MAX_BOUNDARY_TOLERANCE = 0.5
_CONTAINER_TOLERANCE = 0.5
_RELATIVE_BOUNDARY_TOLERANCE = 0.05
_DEGENERATE_RELATIVE_EPSILON = 1e-7


class DetectedHeader(Protocol):
    @property
    def external(self) -> bool: ...

    @property
    def names(self) -> Sequence[str | None]: ...


class DetectedTable(Protocol):
    @property
    def bbox(self) -> Sequence[float]: ...

    @property
    def cells(self) -> Sequence[Sequence[float]]: ...

    @property
    def header(self) -> DetectedHeader: ...

    def extract(self) -> list[list[str | None]]: ...


def extract_document_table_elements(
    pdf_path: Path,
    document_id: str,
    *,
    pages: list[int] | None = None,
    annotator: str = "pymupdf",
) -> list[DocumentElement]:
    with pymupdf.open(pdf_path) as document:
        selected_pages = pages or list(range(1, document.page_count + 1))
        if selected_pages != sorted(set(selected_pages)):
            raise ValueError("pages must be sorted and unique")
        invalid_pages = [page for page in selected_pages if page < 1 or page > document.page_count]
        if invalid_pages:
            raise ValueError(f"pages outside the document range: {invalid_pages}")

        elements: list[DocumentElement] = []
        for page_number in selected_pages:
            elements.extend(
                extract_page_table_elements(
                    document[page_number - 1],
                    document_id,
                    starting_order=len(elements),
                    annotator=annotator,
                )
            )
        return merge_table_continuations(elements)


def extract_page_table_elements(
    page: pymupdf.Page,
    document_id: str,
    *,
    starting_order: int = 0,
    annotator: str = "pymupdf",
    source_items: list[SourceItem] | None = None,
) -> list[DocumentElement]:
    view = canonical_table_view(page, document_id=document_id if source_items is not None else None)
    elements: list[DocumentElement] = []
    for canonical_table in view.tables:
        detected_table = cast(DetectedTable, canonical_table.detected)
        structure = table_structure_from_pymupdf(
            detected_table,
            view.page_number,
            page_width=view.page_width,
            page_height=view.page_height,
            output_bbox=canonical_table.output_bbox,
        )
        if not validate_table_candidate(
            structure,
            nearby_text=canonical_table.nearby_text,
        ).accepted:
            continue
        table_number = len(elements) + 1
        raw_evidence = getattr(detected_table, "provenance", None)
        evidence = raw_evidence if isinstance(raw_evidence, TableProvenance) else None
        if source_items is not None and evidence is not None:
            _append_table_source_items(
                source_items,
                evidence,
                document_id=document_id,
                page_number=view.page_number,
            )
        reconstructed_bbox = reconstruct_table(detected_table, evidence).bbox
        elements.append(
            DocumentElement(
                document_id=document_id,
                element_id=f"page-{view.page_number}-table-{table_number}",
                order=starting_order + table_number - 1,
                element_type="table",
                content=render_table(structure),
                format=structure.representation,
                fragments=[
                    _fragment(
                        view.page_number,
                        canonical_table.output_bbox(reconstructed_bbox),
                        page_width=view.page_width,
                        page_height=view.page_height,
                        source_item_ids=_table_source_ids(structure),
                    )
                ],
                structure=ElementStructure(
                    table=structure,
                    properties=_table_structural_support(evidence),
                ),
                annotation=AnnotationMetadata(
                    stage="candidate",
                    revision=1,
                    annotator=annotator,
                    confidence=1.0,
                    adjudication_status="unreviewed",
                ),
            )
        )
    return elements


def table_structure_from_pymupdf(
    detected_table: DetectedTable,
    page_number: int,
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    output_bbox: Callable[[Sequence[float]], Sequence[float]] | None = None,
) -> TableStructure:
    map_output_bbox = output_bbox or _identity_bbox
    raw_evidence = getattr(detected_table, "provenance", None)
    evidence = raw_evidence if isinstance(raw_evidence, TableProvenance) else None
    reconstruction = reconstruct_table(detected_table, evidence)
    table_bbox = reconstruction.bbox
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    epsilon = max(min(table_width, table_height) * _DEGENERATE_RELATIVE_EPSILON, 1e-9)

    raw_cells = list(
        dict.fromkeys(
            bbox
            for bbox in reconstruction.cells
            if bbox[2] - bbox[0] > epsilon and bbox[3] - bbox[1] > epsilon
        )
    )
    cells = _remove_container_cells(raw_cells)
    if not cells:
        raise ValueError("PyMuPDF table has no cells")

    extracted_text: dict[tuple[float, float, float, float], str] = {}
    logical_positions: dict[tuple[float, float, float, float], tuple[int, int, int]] = {}
    for logical_cell in reconstruction.logical_cells:
        extracted_text.setdefault(logical_cell.bbox, logical_cell.text)
        logical_positions.setdefault(
            logical_cell.bbox,
            (logical_cell.row_index, logical_cell.column_index, logical_cell.rowspan),
        )
    x_values = [value for cell in cells for value in (cell[0], cell[2])]
    y_values = [value for cell in cells for value in (cell[1], cell[3])]
    x_tolerance = _boundary_tolerance(x_values)
    y_tolerance = _boundary_tolerance(y_values)
    x_boundaries = _cluster_boundaries(x_values, x_tolerance)
    y_boundaries = _cluster_boundaries(y_values, y_tolerance)

    cell_source_ids = _assign_cell_source_ids(cells, extracted_text, evidence)
    table_cells: list[TableCell] = []
    for bbox in cells:
        if reconstruction.logical_grid_indices:
            position = logical_positions.get(bbox)
            if position is None:
                raise ValueError(f"logical reconstruction is missing a cell position: {bbox}")
            row_index, column_index, rowspan = position
            row_end = row_index + rowspan
        else:
            row_index = _boundary_index(y_boundaries, float(bbox[1]), y_tolerance)
            column_index = _boundary_index(x_boundaries, float(bbox[0]), x_tolerance)
            row_end = _boundary_index(y_boundaries, float(bbox[3]), y_tolerance)
        column_end = _boundary_index(x_boundaries, float(bbox[2]), x_tolerance)
        if row_end <= row_index or column_end <= column_index:
            continue
        text = extracted_text.get(bbox, "")
        table_cells.append(
            TableCell(
                row_index=row_index,
                column_index=column_index,
                rowspan=row_end - row_index,
                colspan=column_end - column_index,
                role="body",
                text=text,
                fragments=[
                    _fragment(
                        page_number,
                        map_output_bbox(bbox),
                        page_width=page_width,
                        page_height=page_height,
                        source_item_ids=cell_source_ids.get(bbox, ()),
                    )
                ],
            )
        )

    table_cells, geometry_ambiguous = _remove_overlapping_table_cells(table_cells)
    if not table_cells:
        raise ValueError("PyMuPDF table has no non-degenerate cells")
    row_count = (
        len(reconstruction.extracted_rows) if reconstruction.logical_grid_indices else len(y_boundaries) - 1
    )
    column_count = (
        max((len(row) for row in reconstruction.extracted_rows), default=0)
        if reconstruction.logical_grid_indices
        else len(x_boundaries) - 1
    )
    rule_backed_header_row_count: int | None = None
    if evidence is not None and not reconstruction.logical_grid_indices:
        table_cells, row_count, rule_backed_header_row_count = _collapse_rule_backed_physical_rows(
            table_cells,
            row_count,
            column_count,
            table_bbox,
            evidence.native_rules,
        )
    header_row_count = (
        rule_backed_header_row_count
        if rule_backed_header_row_count is not None
        else _infer_header_row_count(reconstruction, table_cells, row_count)
    )
    if rule_backed_header_row_count is None:
        table_cells = _coalesce_empty_header_continuations(table_cells, header_row_count)
    table_cells = _assign_cell_roles(table_cells, header_row_count, column_count)
    features = TableFeatures(
        row_count=row_count,
        column_count=column_count,
        header_row_count=header_row_count,
        cells=table_cells,
    )
    representation, reasons = classify_table(features)
    if geometry_ambiguous:
        representation = "html"
        reasons = [*reasons, "geometry_ambiguous"]
    return TableStructure(
        row_count=row_count,
        column_count=column_count,
        header_row_count=header_row_count,
        representation=representation,
        classification_reasons=reasons,
        cells=table_cells,
    )


def _coalesce_empty_header_continuations(cells: list[TableCell], header_row_count: int) -> list[TableCell]:
    by_position = {(cell.row_index, cell.column_index): cell for cell in cells}
    consumed: set[tuple[int, int]] = set()
    result: list[TableCell] = []
    for cell in sorted(cells, key=lambda value: (value.row_index, value.column_index)):
        position = (cell.row_index, cell.column_index)
        if position in consumed:
            continue
        merged = cell
        while merged.text.strip() and merged.row_index + merged.rowspan < header_row_count:
            next_position = (merged.row_index + merged.rowspan, merged.column_index)
            continuation = by_position.get(next_position)
            if (
                continuation is None
                or continuation.text.strip()
                or continuation.colspan != merged.colspan
                or continuation.row_index + continuation.rowspan > header_row_count
            ):
                break
            if not _vertically_aligned(merged, continuation):
                break
            consumed.add(next_position)
            merged = merged.model_copy(
                update={
                    "rowspan": merged.rowspan + continuation.rowspan,
                    "fragments": [*merged.fragments, *continuation.fragments],
                }
            )
        result.append(merged)
    return result


def _vertically_aligned(upper: TableCell, lower: TableCell) -> bool:
    upper_fragment = upper.fragments[-1]
    lower_fragment = lower.fragments[0]
    upper_bbox = upper_fragment.bbox
    lower_bbox = lower_fragment.bbox
    tolerance = (
        max(
            upper_bbox.x1 - upper_bbox.x0,
            upper_bbox.y1 - upper_bbox.y0,
            lower_bbox.x1 - lower_bbox.x0,
            lower_bbox.y1 - lower_bbox.y0,
        )
        * 1e-4
    )
    same_x_span = (
        abs(upper_bbox.x0 - lower_bbox.x0) <= tolerance and abs(upper_bbox.x1 - lower_bbox.x1) <= tolerance
    )
    same_y_span = (
        abs(upper_bbox.y0 - lower_bbox.y0) <= tolerance and abs(upper_bbox.y1 - lower_bbox.y1) <= tolerance
    )
    touching_y_edge = (
        min(
            abs(upper_bbox.y1 - lower_bbox.y0),
            abs(upper_bbox.y0 - lower_bbox.y1),
        )
        <= tolerance
    )
    touching_x_edge = (
        min(
            abs(upper_bbox.x1 - lower_bbox.x0),
            abs(upper_bbox.x0 - lower_bbox.x1),
        )
        <= tolerance
    )
    return upper_fragment.page_number == lower_fragment.page_number and (
        (same_x_span and touching_y_edge) or (same_y_span and touching_x_edge)
    )


def _assign_cell_roles(cells: list[TableCell], header_row_count: int, column_count: int) -> list[TableCell]:
    has_numeric_value_rows = _has_numeric_value_rows(cells, header_row_count, column_count)
    by_row: dict[int, list[TableCell]] = {}
    for cell in cells:
        by_row.setdefault(cell.row_index, []).append(cell)
    section_rows = {
        row_index
        for row_index, row in by_row.items()
        if row_index >= header_row_count
        and sum(cell.colspan for cell in row) == column_count
        and len(populated := [cell for cell in row if cell.text.strip()]) == 1
        and populated[0].colspan >= column_count - 1
    }
    assigned: list[TableCell] = []
    for cell in cells:
        role = cell.role
        has_stub_text = cell.column_index == 0 and bool(cell.text.strip())
        is_full_width_body_cell = (
            cell.row_index >= header_row_count and cell.column_index == 0 and cell.colspan == column_count
        )
        if cell.row_index < header_row_count or (
            has_stub_text and is_full_width_body_cell and not has_numeric_value_rows
        ):
            role = "header"
        elif cell.row_index in section_rows or (
            has_stub_text and column_count > 1 and (header_row_count > 0 or is_full_width_body_cell)
        ):
            role = "row_header"
        assigned.append(cell.model_copy(update={"role": role}))
    return assigned


def _has_numeric_value_rows(cells: Sequence[TableCell], header_row_count: int, column_count: int) -> bool:
    if column_count < 3:
        return False
    by_row: dict[int, list[TableCell]] = {}
    for cell in cells:
        if cell.row_index >= header_row_count:
            by_row.setdefault(cell.row_index, []).append(cell)
    return any(
        any(cell.column_index == 0 and bool(cell.text.strip()) for cell in row)
        and sum(
            _is_numeric_value(cell.text) for cell in row if cell.column_index > 0 and bool(cell.text.strip())
        )
        >= 2
        for row in by_row.values()
    )


def _is_numeric_value(value: str) -> bool:
    normalized = value.strip().replace(",", "").replace(" ", "")
    return bool(re.fullmatch(r"[$€£¥]?\(?[+-]?\d+(?:\.\d+)?%?\)?", normalized))


def _looks_like_data_row(values: Sequence[str | None]) -> bool:
    nonempty = [str(value).strip() for value in values if value and str(value).strip()]
    return bool(
        nonempty
        and _is_numeric_value(nonempty[0])
        and sum(_is_numeric_value(value) for value in nonempty) / len(nonempty) >= 0.5
    )


def _looks_like_numeric_body_row(values: Sequence[str | None], column_count: int) -> bool:
    nonempty = [str(value).strip() for value in values if value and str(value).strip()]
    if column_count <= 0 or not nonempty:
        return False
    numeric_count = sum(_is_numeric_value(value) for value in nonempty)
    return bool(
        len(nonempty) / column_count >= 0.5
        and not _is_numeric_value(nonempty[0])
        and numeric_count >= 2
        and numeric_count / len(nonempty) >= 0.5
    )


def _looks_like_sparse_section_row(values: Sequence[str | None], column_count: int) -> bool:
    populated_indices = [index for index, value in enumerate(values) if value and str(value).strip()]
    return column_count > 1 and populated_indices == [0]


def _infer_header_row_count(
    reconstruction: TableReconstruction, cells: list[TableCell], row_count: int
) -> int:
    if reconstruction.recovered_header_row_count is not None:
        return min(reconstruction.recovered_header_row_count, row_count)
    header = reconstruction.header
    if header.external:
        return 0
    names = header.names
    if not any(name and str(name).strip() for name in names):
        return 0
    if _looks_like_data_row(names):
        return 0

    top_cells = [cell for cell in cells if cell.row_index == 0]
    extracted_rows = reconstruction.extracted_rows
    header_rows = _leading_header_row_count(extracted_rows)
    if row_count > 1 and any(cell.colspan > 1 for cell in top_cells):
        second_logical_row = [cell for cell in cells if cell.row_index == 1]
        if any(cell.rowspan >= 3 and cell.text.strip() for cell in second_logical_row):
            return 1
        if len(extracted_rows) > 1:
            second_row = extracted_rows[1]
            first_value = next(iter(second_row), None)
            if first_value and first_value.strip() and _looks_like_data_row(second_row):
                return 0
        header_rows = 2
        extracted_column_count = max((len(row) for row in extracted_rows), default=0)
        for index, row in enumerate(extracted_rows[2:6], start=2):
            if _looks_like_sparse_section_row(row, extracted_column_count):
                header_rows = index
                break
            if _looks_like_numeric_body_row(row, extracted_column_count):
                header_rows = index
                break
    if top_cells:
        header_rows = max(header_rows, max(cell.rowspan for cell in top_cells))
    return min(header_rows, row_count)


def _leading_header_row_count(rows: Sequence[Sequence[str | None]]) -> int:
    if len(rows) < 2:
        return 1

    first = tuple(_normalize_grid_value(value) for value in rows[0])
    second = tuple(_normalize_grid_value(value) for value in rows[1])
    if first == second and sum(bool(value) for value in first) >= 2:
        return 2

    column_count = max((len(row) for row in rows), default=0)
    for index, row in enumerate(rows[1:8], start=1):
        if _looks_like_sparse_section_row(row, column_count):
            return 1
        if _looks_like_numeric_body_row(row, column_count):
            return index
    return 1


def _normalize_grid_value(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _collapse_rule_backed_physical_rows(
    cells: list[TableCell],
    row_count: int,
    column_count: int,
    table_bbox: tuple[float, float, float, float],
    rules: Sequence[NativeRule],
) -> tuple[list[TableCell], int, int | None]:
    """Collapse text-line rows only when full-width native rules prove a smaller grid."""
    if row_count < 5 or column_count < 10 or not rules:
        return cells, row_count, None

    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    level_tolerance = max(0.75, table_height * 0.004)
    grouped: list[tuple[list[float], list[tuple[float, float]]]] = []
    for rule in sorted(rules, key=lambda value: (value.y, value.x0, value.x1)):
        if not table_bbox[1] - level_tolerance <= rule.y <= table_bbox[3] + level_tolerance:
            continue
        interval = (max(table_bbox[0], rule.x0), min(table_bbox[2], rule.x1))
        if interval[1] <= interval[0]:
            continue
        if grouped and rule.y - grouped[-1][0][-1] <= level_tolerance:
            grouped[-1][0].append(rule.y)
            grouped[-1][1].append(interval)
        else:
            grouped.append(([rule.y], [interval]))

    boundaries: list[float] = []
    boundary_coverages: list[float] = []
    for ys, intervals in grouped:
        merged: list[list[float]] = []
        for left, right in sorted(intervals):
            if merged and left <= merged[-1][1] + level_tolerance:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        coverage = sum(right - left for left, right in merged) / table_width
        if coverage >= 0.8:
            boundaries.append(median(ys))
            boundary_coverages.append(coverage)

    logical_row_count = len(boundaries) - 1
    if (
        logical_row_count < 3
        or logical_row_count > row_count - 2
        or abs(boundaries[0] - table_bbox[1]) > level_tolerance * 2
        or abs(boundaries[-1] - table_bbox[3]) > level_tolerance * 2
    ):
        return cells, row_count, None

    edge_tolerance = level_tolerance * 1.5
    grouped_cells: dict[tuple[int, int, int, int], list[TableCell]] = {}
    for cell in cells:
        bbox = cell.fragments[0].bbox
        center = (bbox.y0 + bbox.y1) / 2
        center_row = next(
            (
                index
                for index, (top, bottom) in enumerate(zip(boundaries, boundaries[1:], strict=False))
                if top <= center <= bottom
            ),
            None,
        )
        if center_row is None:
            return cells, row_count, None
        start_index = min(range(len(boundaries)), key=lambda index: abs(boundaries[index] - bbox.y0))
        end_index = min(range(len(boundaries)), key=lambda index: abs(boundaries[index] - bbox.y1))
        row_start = start_index if abs(boundaries[start_index] - bbox.y0) <= edge_tolerance else center_row
        row_end = end_index if abs(boundaries[end_index] - bbox.y1) <= edge_tolerance else center_row + 1
        if row_end <= row_start:
            row_start, row_end = center_row, center_row + 1
        key = (row_start, cell.column_index, row_end - row_start, cell.colspan)
        grouped_cells.setdefault(key, []).append(cell)

    occupied: set[tuple[int, int]] = set()
    for row_index, column_index, rowspan, colspan in grouped_cells:
        positions = {
            (row, column)
            for row in range(row_index, row_index + rowspan)
            for column in range(column_index, column_index + colspan)
        }
        if occupied.intersection(positions):
            return cells, row_count, None
        occupied.update(positions)

    collapsed: list[TableCell] = []
    for (row_index, column_index, rowspan, colspan), pieces in sorted(grouped_cells.items()):
        ordered_pieces = sorted(
            pieces,
            key=lambda piece: (
                piece.fragments[0].bbox.y0,
                piece.fragments[0].bbox.x0,
                piece.fragments[0].bbox.y1,
            ),
        )
        text_parts = [piece.text.strip() for piece in ordered_pieces if piece.text.strip()]
        merged_cell = ordered_pieces[0].model_copy(
            update={
                "row_index": row_index,
                "rowspan": rowspan,
                "text": " ".join(dict.fromkeys(text_parts)),
                "fragments": [fragment for piece in ordered_pieces for fragment in piece.fragments],
            }
        )
        split_boundaries = [
            boundary
            for boundary in range(row_index + 1, row_index + rowspan)
            if boundary_coverages[boundary] >= 0.98
        ]
        if not split_boundaries:
            collapsed.append(merged_cell)
            continue
        starts = [row_index, *split_boundaries]
        ends = [*split_boundaries, row_index + rowspan]
        collapsed.extend(
            merged_cell.model_copy(
                update={
                    "row_index": start,
                    "rowspan": end - start,
                    "text": merged_cell.text if segment == 0 else "",
                }
            )
            for segment, (start, end) in enumerate(zip(starts, ends, strict=True))
        )
    populated_by_row = [
        sum(
            bool(cell.text.strip()) and cell.row_index <= row < cell.row_index + cell.rowspan
            for cell in collapsed
        )
        for row in range(logical_row_count)
    ]
    dense_header_row = next(
        (
            row
            for row, populated in enumerate(populated_by_row[: min(logical_row_count, 8)])
            if populated >= column_count * 0.6
        ),
        None,
    )
    if dense_header_row is None or dense_header_row < 1:
        return cells, row_count, None
    header_row_count = dense_header_row + 1
    return collapsed, logical_row_count, header_row_count


def _remove_overlapping_table_cells(cells: list[TableCell]) -> tuple[list[TableCell], bool]:
    retained: list[TableCell] = []
    occupied: set[tuple[int, int]] = set()
    ambiguous = False
    for cell in sorted(
        cells,
        key=lambda value: (
            value.rowspan * value.colspan,
            not bool(value.text.strip()),
            value.row_index,
            value.column_index,
        ),
    ):
        positions = {
            (row, column)
            for row in range(cell.row_index, cell.row_index + cell.rowspan)
            for column in range(cell.column_index, cell.column_index + cell.colspan)
        }
        if occupied.intersection(positions):
            ambiguous = True
            continue
        retained.append(cell)
        occupied.update(positions)
    retained.sort(key=lambda value: (value.row_index, value.column_index))
    return retained, ambiguous


def _remove_container_cells(
    cells: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, float, float]]:
    retained: list[tuple[float, float, float, float]] = []
    for outer in cells:
        contained = sum(_contains_bbox(outer, inner) for inner in cells if inner != outer)
        if contained < 2:
            retained.append(outer)
    return retained


def _contains_bbox(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    return (
        outer[0] <= inner[0] + _CONTAINER_TOLERANCE
        and outer[1] <= inner[1] + _CONTAINER_TOLERANCE
        and outer[2] >= inner[2] - _CONTAINER_TOLERANCE
        and outer[3] >= inner[3] - _CONTAINER_TOLERANCE
    )


def _boundary_tolerance(values: list[float]) -> float:
    distinct = sorted(set(values))
    positive_gaps = [
        right - left for left, right in zip(distinct, distinct[1:], strict=False) if right > left
    ]
    if not positive_gaps:
        return 1e-9
    return max(
        1e-9,
        min(_MAX_BOUNDARY_TOLERANCE, median(positive_gaps) * _RELATIVE_BOUNDARY_TOLERANCE),
    )


def _cluster_boundaries(values: list[float], tolerance: float) -> list[float]:
    sorted_values = sorted(set(values))
    groups: list[list[float]] = []
    for value in sorted_values:
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _boundary_index(boundaries: Sequence[float], value: float, tolerance: float) -> int:
    index, distance = min(
        enumerate(abs(boundary - value) for boundary in boundaries),
        key=lambda item: item[1],
    )
    if distance > tolerance:
        raise ValueError(f"cell edge {value} does not align with a table boundary")
    return index


def _identity_bbox(bbox: Sequence[float]) -> Sequence[float]:
    return bbox


def _append_table_source_items(
    destination: list[SourceItem],
    evidence: TableProvenance,
    *,
    document_id: str,
    page_number: int,
) -> None:
    destination.extend(
        make_source_item(
            source_item_id=token.token_id.value,
            document_id=document_id,
            page_number=page_number,
            kind="word",
            coordinate_frame="detector_page",
            coordinates=token.bbox,
            canonical_coordinates=token.canonical_bbox or token.bbox,
            text=token.text,
        )
        for token in evidence.native_tokens
    )
    destination.extend(
        make_source_item(
            source_item_id=rule.rule_id.value,
            document_id=document_id,
            page_number=page_number,
            kind="rule",
            coordinate_frame="detector_page",
            coordinates=(rule.x0, rule.y, rule.x1, rule.y),
            canonical_coordinates=rule.canonical_geometry or (rule.x0, rule.y, rule.x1, rule.y),
            text=None,
        )
        for rule in evidence.native_rules
    )


def _normalized_provenance_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+|[^\w\s]", normalized))


def _token_text_compatible(token_text: str, cell_text: str) -> bool:
    token = _normalized_provenance_text(token_text)
    cell = _normalized_provenance_text(cell_text)
    if not token or not cell:
        return False
    return token == cell or f" {token} " in f" {cell} "


def _assign_cell_source_ids(
    cells: Sequence[tuple[float, float, float, float]],
    extracted_text: dict[tuple[float, float, float, float], str],
    evidence: TableProvenance | None,
) -> dict[tuple[float, float, float, float], tuple[str, ...]]:
    """Assign each native word to at most one independently supported nonempty cell."""
    if evidence is None:
        return {}
    assignments: dict[tuple[float, float, float, float], list[str]] = {}
    nonempty_cells = tuple(bbox for bbox in cells if extracted_text.get(bbox, "").strip())
    for token in evidence.native_tokens:
        token_width = token.bbox[2] - token.bbox[0]
        token_height = token.bbox[3] - token.bbox[1]
        token_area = token_width * token_height
        if token_area <= 0:
            continue
        center_x = (token.bbox[0] + token.bbox[2]) / 2
        center_y = (token.bbox[1] + token.bbox[3]) / 2
        candidates: list[tuple[float, float, tuple[float, float, float, float]]] = []
        for bbox in nonempty_cells:
            if not (bbox[0] < center_x < bbox[2] and bbox[1] < center_y < bbox[3]):
                continue
            overlap_width = max(0.0, min(token.bbox[2], bbox[2]) - max(token.bbox[0], bbox[0]))
            overlap_height = max(0.0, min(token.bbox[3], bbox[3]) - max(token.bbox[1], bbox[1]))
            overlap_ratio = overlap_width * overlap_height / token_area
            if overlap_ratio < 0.5 or not _token_text_compatible(token.text, extracted_text[bbox]):
                continue
            edge_clearance = min(
                center_x - bbox[0], bbox[2] - center_x, center_y - bbox[1], bbox[3] - center_y
            )
            candidates.append((overlap_ratio, edge_clearance, bbox))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        best = candidates[0]
        if len(candidates) > 1 and best[:2] == candidates[1][:2]:
            # A boundary tie is not positive evidence for either logical cell.
            continue
        assignments.setdefault(best[2], []).append(token.token_id.value)
    return {bbox: tuple(dict.fromkeys(ids)) for bbox, ids in assignments.items()}


def _table_structural_support(evidence: TableProvenance | None) -> list[StructureProperty]:
    if evidence is None or not evidence.native_rule_ids:
        return []
    return [
        StructureProperty(
            key="table_structural_source_item_ids",
            value=json.dumps(
                sorted(rule_id.value for rule_id in evidence.native_rule_ids),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        )
    ]


def _table_source_ids(structure: TableStructure) -> list[str]:
    return list(
        dict.fromkeys(
            source_item_id
            for cell in structure.cells
            for fragment in cell.fragments
            for source_item_id in fragment.source_item_ids
        )
    )


def _fragment(
    page_number: int,
    bbox: Sequence[float],
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    source_item_ids: Sequence[str] = (),
) -> PageFragment:
    return PageFragment(
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
        source_item_ids=list(dict.fromkeys(source_item_ids)),
    )
