from __future__ import annotations

from collections.abc import Sequence

import pymupdf
import pytest

import app.pdf2md.canonical_table_view as canonical_table_view_module
from app.pdf2md.canonical_table_view import canonical_table_view
from app.pdf2md.pymupdf_geometry import unrotated_page_extent
from app.pdf2md.pymupdf_tables import extract_page_table_elements
from app.pdf2md.schema import BoundingBox
from app.pdf2md.tables import render_table


def test_unrotated_page_extent_is_derived_exactly_from_fractional_cropbox() -> None:
    document = pymupdf.open()
    page = document.new_page(width=700, height=900)
    page.set_cropbox(pymupdf.Rect(100.25, 50.5, 550.75, 700.75))
    page.set_rotation(90)

    assert unrotated_page_extent(page) == (450.5, 650.25)
    document.close()


def _draw_merged_table(page: pymupdf.Page) -> None:
    x_boundaries = (40.0, 110.0, 180.0, 260.0)
    y_boundaries = (80.0, 120.0, 160.0, 200.0)
    page.draw_rect((x_boundaries[0], y_boundaries[0], x_boundaries[-1], y_boundaries[-1]))
    for y in y_boundaries[1:-1]:
        page.draw_line((x_boundaries[0], y), (x_boundaries[-1], y))
    for x in x_boundaries[1:-1]:
        page.draw_line((x, y_boundaries[1]), (x, y_boundaries[-1]))
    for point, text in (
        ((48.0, 105.0), "Summary"),
        ((48.0, 145.0), "A"),
        ((118.0, 145.0), "10"),
        ((188.0, 145.0), "20"),
        ((48.0, 185.0), "B"),
        ((118.0, 185.0), "30"),
        ((188.0, 185.0), "40"),
    ):
        page.insert_text(point, text)


def _draw_borderless_table(page: pymupdf.Page) -> None:
    for row, values in enumerate((("Item", "Count"), ("Alpha", "10"), ("Beta", "20"))):
        for column, text in enumerate(values):
            page.insert_text((40.0 + column * 100.0, 80.0 + row * 35.0), text)


def _bbox_tuple(bbox: BoundingBox) -> tuple[float, float, float, float]:
    return bbox.x0, bbox.y0, bbox.x1, bbox.y1


def _assert_bbox_close(actual: Sequence[float], expected: Sequence[float]) -> None:
    assert tuple(actual) == pytest.approx(tuple(expected), abs=1e-4)


@pytest.mark.parametrize(
    ("rotation", "expected_page_size", "expected_table_bbox", "expected_summary_bbox"),
    [
        (0, (300.0, 280.0), (40.0, 80.0, 260.0, 200.0), (40.0, 80.0, 260.0, 120.0)),
        (90, (280.0, 300.0), (80.0, 40.0, 200.0, 260.0), (160.0, 40.0, 200.0, 260.0)),
        (180, (300.0, 280.0), (40.0, 80.0, 260.0, 200.0), (40.0, 160.0, 260.0, 200.0)),
        (270, (280.0, 300.0), (80.0, 40.0, 200.0, 260.0), (80.0, 40.0, 120.0, 260.0)),
    ],
)
def test_canonical_view_preserves_topology_and_maps_only_emitted_fragments(
    rotation: int,
    expected_page_size: tuple[float, float],
    expected_table_bbox: tuple[float, float, float, float],
    expected_summary_bbox: tuple[float, float, float, float],
) -> None:
    document = pymupdf.open()
    page = document.new_page(width=300, height=280)
    _draw_merged_table(page)
    page.set_rotation(rotation)

    elements = extract_page_table_elements(page, "rotation-probe")

    assert len(elements) == 1
    element = elements[0]
    structure = element.structure.table
    assert structure is not None
    assert (structure.row_count, structure.column_count) == (3, 3)
    assert [(cell.text, cell.rowspan, cell.colspan) for cell in structure.cells] == [
        ("Summary", 1, 3),
        ("A", 1, 1),
        ("10", 1, 1),
        ("20", 1, 1),
        ("B", 1, 1),
        ("30", 1, 1),
        ("40", 1, 1),
    ]
    assert (element.fragments[0].page_width, element.fragments[0].page_height) == expected_page_size
    _assert_bbox_close(_bbox_tuple(element.fragments[0].bbox), expected_table_bbox)
    summary = next(cell for cell in structure.cells if cell.text == "Summary")
    _assert_bbox_close(_bbox_tuple(summary.fragments[0].bbox), expected_summary_bbox)
    document.close()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_borderless_detection_and_conversion_are_quarter_turn_invariant(rotation: int) -> None:
    document = pymupdf.open()
    page = document.new_page(width=300, height=280)
    _draw_borderless_table(page)
    page.set_rotation(rotation)

    elements = extract_page_table_elements(page, "borderless-rotation-probe")

    assert len(elements) == 1
    table = elements[0].structure.table
    assert table is not None
    assert (table.row_count, table.column_count) == (3, 2)
    assert [cell.text for cell in table.cells if cell.text] == [
        "Item",
        "Count",
        "Alpha",
        "10",
        "Beta",
        "20",
    ]
    document.close()


def test_rendered_table_is_quarter_turn_invariant() -> None:
    rendered: list[str] = []
    for rotation in (0, 90, 180, 270):
        document = pymupdf.open()
        page = document.new_page(width=300, height=280)
        _draw_merged_table(page)
        page.set_rotation(rotation)
        elements = extract_page_table_elements(page, "render-rotation-probe")
        assert len(elements) == 1
        table = elements[0].structure.table
        assert table is not None
        rendered.append(render_table(table))
        document.close()

    assert len(set(rendered)) == 1


def test_nonzero_rotation_restores_the_source_and_returns_closed_lifetime_snapshots() -> None:
    document = pymupdf.open()
    page = document.new_page(width=300, height=280)
    _draw_merged_table(page)
    page.set_rotation(90)
    source_xref = page.xref
    source_page_count = document.page_count

    view = canonical_table_view(page)

    assert page.rotation == 90
    assert page.xref == source_xref
    assert document.page_count == source_page_count
    assert tuple(page.rect) == (0.0, 0.0, 280.0, 300.0)
    assert len(view.tables) == 1
    assert view.tables[0].detected.extract() == [
        ["Summary", None, None],
        ["A", "10", "20"],
        ["B", "30", "40"],
    ]
    document.close()

    assert view.tables[0].detected.extract()[2] == ["B", "30", "40"]
    _assert_bbox_close(view.output_bbox(view.tables[0].detected.bbox), (80.0, 40.0, 200.0, 260.0))


class _SyntheticDetectedTable:
    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        *,
        row_count: int,
        column_count: int,
        populated_count: int,
        value: str,
    ) -> None:
        self.bbox = bbox
        values = [value] * populated_count + [None] * (row_count * column_count - populated_count)
        self._rows = [values[index : index + column_count] for index in range(0, len(values), column_count)]

    def extract(self) -> list[list[str | None]]:
        return [list(row) for row in self._rows]


def test_rotated_source_prefers_coherent_display_detection_over_majority_empty_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((40, 40), "Synthetic rotated source")
    page.set_rotation(90)
    display_table = _SyntheticDetectedTable(
        (60.0, 150.0, 730.0, 415.0),
        row_count=25,
        column_count=9,
        populated_count=151,
        value="abcdef",
    )
    expanded_canonical_table = _SyntheticDetectedTable(
        (55.0, 60.0, 545.0, 735.0),
        row_count=67,
        column_count=20,
        populated_count=503,
        value="ab",
    )

    def synthetic_detection(
        source_page: pymupdf.Page, *, document_id: str | None = None
    ) -> list[_SyntheticDetectedTable]:
        assert document_id is None
        return [display_table] if source_page.rotation == 90 else [expanded_canonical_table]

    monkeypatch.setattr(canonical_table_view_module, "detect_page_tables", synthetic_detection)

    view = canonical_table_view(page)

    assert page.rotation == 90
    assert [table.detected for table in view.tables] == [display_table]
    assert view.output_bbox(display_table.bbox) == display_table.bbox
    document.close()
