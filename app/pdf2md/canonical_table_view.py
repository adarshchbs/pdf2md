from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pymupdf

from app.pdf2md.pymupdf_geometry import unrotated_page_extent
from app.pdf2md.pymupdf_runtime import pymupdf_session
from app.pdf2md.pymupdf_table_detection import DetectedTable, detect_page_tables
from app.pdf2md.semantic_text import normalize_page_size
from app.pdf2md.table_provenance import TableDiagnostic

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class CanonicalTable:
    """A table snapshot and its validation context with an output-coordinate mapping."""

    detected: DetectedTable
    nearby_text: str
    _output_matrix: tuple[float, float, float, float, float, float]

    def output_bbox(self, bbox: Sequence[float]) -> BBox:
        return _map_bbox(bbox, self._output_matrix)


@dataclass(frozen=True)
class CanonicalTableView:
    """Closed-lifetime table detection results with upright output mapping."""

    page_number: int
    page_width: float
    page_height: float
    tables: tuple[CanonicalTable, ...]
    _output_matrix: tuple[float, float, float, float, float, float]

    def output_bbox(self, bbox: Sequence[float]) -> BBox:
        """Map one canonical detection fragment to upright output coordinates."""
        return _map_bbox(bbox, self._output_matrix)


@dataclass(frozen=True)
class _DetectionView:
    tables: tuple[DetectedTable, ...]
    contexts: tuple[str, ...]
    output_matrix: tuple[float, float, float, float, float, float]
    diagnostics: tuple[TableDiagnostic, ...]


@dataclass(frozen=True)
class _DetectionCoherence:
    table_count: int
    populated_count: int
    capacity: int
    character_count: int


_IDENTITY_MATRIX = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def canonical_table_view(
    page: pymupdf.Page,
    *,
    document_id: str | None = None,
    diagnostics: list[TableDiagnostic] | None = None,
) -> CanonicalTableView:
    """Detect in canonical coordinates, reconcile rotated-source pathologies, and restore the page."""
    with pymupdf_session():
        return _canonical_table_view(page, document_id=document_id, diagnostics=diagnostics)


def _canonical_table_view(
    page: pymupdf.Page,
    *,
    document_id: str | None,
    diagnostics: list[TableDiagnostic] | None,
) -> CanonicalTableView:
    page_index = page.number
    if page_index is None:
        raise ValueError("page must belong to an open PyMuPDF document")
    rotation = int(page.rotation)
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"page rotation must be a multiple of 90 degrees: {rotation}")

    output_matrix = _matrix_tuple(page.rotation_matrix)
    page_width, page_height = normalize_page_size(*unrotated_page_extent(page), rotation)

    display_view: _DetectionView | None = None
    if rotation:
        display_view = _detect_view(
            page,
            _IDENTITY_MATRIX,
            document_id=document_id,
            collect_diagnostics=diagnostics is not None,
        )
        page.set_rotation(0)
    try:
        canonical_view = _detect_view(
            page,
            output_matrix,
            document_id=document_id,
            collect_diagnostics=diagnostics is not None,
        )
    finally:
        if rotation:
            page.set_rotation(rotation)

    selected_view = canonical_view
    if display_view is not None and _is_materially_more_coherent(display_view, canonical_view):
        selected_view = display_view
    if diagnostics is not None:
        diagnostics.extend(selected_view.diagnostics)
    tables = tuple(
        CanonicalTable(
            detected=table,
            nearby_text=context,
            _output_matrix=selected_view.output_matrix,
        )
        for table, context in zip(selected_view.tables, selected_view.contexts, strict=True)
    )
    return CanonicalTableView(
        page_number=page_index + 1,
        page_width=page_width,
        page_height=page_height,
        tables=tables,
        _output_matrix=selected_view.output_matrix,
    )


def _detect_view(
    page: pymupdf.Page,
    output_matrix: tuple[float, float, float, float, float, float],
    *,
    document_id: str | None,
    collect_diagnostics: bool,
) -> _DetectionView:
    diagnostics: list[TableDiagnostic] = []
    tables = tuple(
        detect_page_tables(page, document_id=document_id, diagnostics=diagnostics)
        if collect_diagnostics
        else detect_page_tables(page, document_id=document_id)
    )
    return _DetectionView(
        tables=tables,
        contexts=tuple(_table_context_text(page, table.bbox) for table in tables),
        output_matrix=output_matrix,
        diagnostics=tuple(diagnostics),
    )


def _is_materially_more_coherent(alternate: _DetectionView, canonical: _DetectionView) -> bool:
    alternate_coherence = _detection_coherence(alternate.tables)
    canonical_coherence = _detection_coherence(canonical.tables)
    if alternate_coherence.table_count != canonical_coherence.table_count:
        return False
    if (
        alternate_coherence.table_count == 0
        or alternate_coherence.capacity == 0
        or canonical_coherence.capacity == 0
        or alternate_coherence.populated_count == 0
        or canonical_coherence.populated_count == 0
    ):
        return False

    alternate_density = alternate_coherence.populated_count / alternate_coherence.capacity
    canonical_density = canonical_coherence.populated_count / canonical_coherence.capacity
    alternate_characters_per_cell = alternate_coherence.character_count / alternate_coherence.populated_count
    canonical_characters_per_cell = canonical_coherence.character_count / canonical_coherence.populated_count
    return (
        alternate_coherence.character_count >= canonical_coherence.character_count * 0.75
        and alternate_density >= canonical_density + 0.10
        and alternate_characters_per_cell >= canonical_characters_per_cell * 1.25
    )


def _detection_coherence(tables: Sequence[DetectedTable]) -> _DetectionCoherence:
    populated_count = 0
    capacity = 0
    character_count = 0
    for table in tables:
        rows = table.extract()
        column_count = max((len(row) for row in rows), default=0)
        capacity += len(rows) * column_count
        for row in rows:
            for value in row:
                normalized = " ".join(str(value or "").split())
                if normalized:
                    populated_count += 1
                    character_count += len(normalized)
    return _DetectionCoherence(
        table_count=len(tables),
        populated_count=populated_count,
        capacity=capacity,
        character_count=character_count,
    )


def _table_context_text(page: pymupdf.Page, bbox: Sequence[float]) -> str:
    if len(bbox) != 4:
        raise ValueError(f"PyMuPDF table bbox must have four coordinates: {bbox}")
    vertical_context = max(72.0, float(page.rect.height) * 0.1)
    clip = (
        pymupdf.Rect(
            float(bbox[0]),
            max(float(page.rect.y0), float(bbox[1]) - vertical_context),
            float(bbox[2]),
            float(bbox[3]),
        )
        & page.rect
    )
    return str(page.get_text("text", clip=clip))


def _map_bbox(bbox: Sequence[float], matrix: tuple[float, float, float, float, float, float]) -> BBox:
    if len(bbox) != 4:
        raise ValueError(f"table fragment bbox must have four coordinates: {bbox}")
    rectangle = pymupdf.Rect(*(float(value) for value in bbox))
    if rectangle.is_empty or rectangle.is_infinite:
        raise ValueError(f"table fragment bbox must have positive dimensions: {bbox}")
    mapped = rectangle * pymupdf.Matrix(*matrix)
    return (float(mapped.x0), float(mapped.y0), float(mapped.x1), float(mapped.y1))


def _matrix_tuple(matrix: pymupdf.Matrix) -> tuple[float, float, float, float, float, float]:
    values = cast(Sequence[float], matrix)
    if len(values) != 6:
        raise ValueError(f"PyMuPDF matrix must have six values: {values}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]
