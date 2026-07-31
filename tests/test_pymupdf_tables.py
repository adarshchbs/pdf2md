import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pymupdf
import pytest

from app.pdf2md.engine import extract_document_elements
from app.pdf2md.pymupdf_table_detection import detect_page_tables
from app.pdf2md.pymupdf_tables import (
    _assign_cell_source_ids,  # pyright: ignore[reportPrivateUsage]
    _table_structural_support,  # pyright: ignore[reportPrivateUsage]
    extract_document_table_elements,
    extract_page_table_elements,
    table_structure_from_pymupdf,
)
from app.pdf2md.schema import DocumentElement
from app.pdf2md.table_provenance import (
    CoordinateFrame,
    FinderProvenance,
    FinderSnapshot,
    GeometricBand,
    GeometricBandKind,
    NativeRule,
    NativeRuleId,
    NativeToken,
    NativeTokenId,
    TableDiagnostic,
    TableDiagnosticOutcome,
    TableDiagnosticStage,
    TableProvenance,
    TableReconstruction,
    reconstruct_table,
)
from app.pdf2md.tables import render_table


class FakeHeader:
    def __init__(self, *, external: bool, names: list[str | None]) -> None:
        self.external = external
        self.names = names


class FakeTable:
    bbox = (0.0, 0.0, 30.0, 30.0)
    cells = [
        (0.0, 0.0, 10.0, 20.0),
        (10.0, 0.0, 30.0, 10.0),
        (10.0, 10.0, 20.0, 20.0),
        (20.0, 10.0, 30.0, 20.0),
        (0.0, 20.0, 10.0, 30.0),
        (10.0, 20.0, 20.0, 30.0),
        (20.0, 20.0, 30.0, 30.0),
    ]
    header = FakeHeader(external=False, names=["Region", "Revenue", None])

    def extract(self) -> list[list[str | None]]:
        return [
            ["Region", "Revenue", None],
            [None, "2025", "2026"],
            ["North", "10", "12"],
        ]


def _assert_native_word_centers_are_structurally_covered(
    pdf_path: Path,
    page_number: int,
    elements: Sequence[DocumentElement],
) -> None:
    fragment_bboxes = [
        fragment.bbox
        for element in elements
        for fragment in element.fragments
        if fragment.page_number == page_number
    ]
    assert fragment_bboxes
    with pymupdf.open(pdf_path) as document:
        words = cast(Sequence[Sequence[object]], document[page_number - 1].get_text("words", sort=True))
    for word in words:
        assert len(word) >= 5
        center_x = (float(str(word[0])) + float(str(word[2]))) / 2
        center_y = (float(str(word[1])) + float(str(word[3]))) / 2
        assert any(
            bbox.x0 <= center_x <= bbox.x1 and bbox.y0 <= center_y <= bbox.y1 for bbox in fragment_bboxes
        ), word


def test_cell_provenance_is_exclusive_text_compatible_and_excludes_rules_and_empty_cells() -> None:
    cells = ((0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0))
    alpha = NativeToken(
        token_id=NativeTokenId("alpha"),
        frame=CoordinateFrame.DETECTOR_PAGE,
        bbox=(1.0, 1.0, 5.0, 5.0),
        baseline=5.0,
        text="Alpha",
    )
    incompatible = NativeToken(
        token_id=NativeTokenId("incompatible"),
        frame=CoordinateFrame.DETECTOR_PAGE,
        bbox=(6.0, 1.0, 9.0, 5.0),
        baseline=5.0,
        text="Other",
    )
    empty_cell_word = NativeToken(
        token_id=NativeTokenId("empty"),
        frame=CoordinateFrame.DETECTOR_PAGE,
        bbox=(12.0, 1.0, 16.0, 5.0),
        baseline=5.0,
        text="Ghost",
    )
    boundary = NativeToken(
        token_id=NativeTokenId("boundary"),
        frame=CoordinateFrame.DETECTOR_PAGE,
        bbox=(9.0, 1.0, 11.0, 5.0),
        baseline=5.0,
        text="Alpha",
    )
    rule = NativeRule(
        rule_id=NativeRuleId("rule"),
        frame=CoordinateFrame.DETECTOR_PAGE,
        x0=0,
        x1=20,
        y=10,
    )
    evidence = TableProvenance(
        finder=FinderProvenance.DEFAULT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_token_ids=(alpha.token_id, incompatible.token_id, empty_cell_word.token_id, boundary.token_id),
        native_rule_ids=(rule.rule_id,),
        native_tokens=(alpha, incompatible, empty_cell_word, boundary),
        native_rules=(rule,),
    )

    diagnostics = []
    assigned = _assign_cell_source_ids(
        cells,
        {cells[0]: "Alpha", cells[1]: ""},
        evidence,
        diagnostics=diagnostics,
    )

    assert assigned == {cells[0]: ("alpha",)}
    assert len(diagnostics) == 1
    assert diagnostics[0].stage == TableDiagnosticStage.CELL_ASSIGNMENT
    assert diagnostics[0].outcome == TableDiagnosticOutcome.APPLIED
    assert diagnostics[0].metric("assigned_cells") == 1
    assert diagnostics[0].metric("assigned_source_items") == 1
    assert diagnostics[0].metric("native_tokens") == 4
    cited = [source_id for ids in assigned.values() for source_id in ids]
    assert len(cited) == len(set(cited))
    properties = _table_structural_support(evidence)
    assert [property.value for property in properties] == ['["rule"]']
    assert "rule" not in cited


def test_pymupdf_merged_cells_become_spans() -> None:
    diagnostics = []
    table = table_structure_from_pymupdf(FakeTable(), page_number=2, diagnostics=diagnostics)

    assert table.header_row_count == 2
    assert table.representation == "html"
    assert table.classification_reasons == ["multiple_header_rows", "spanning_cells"]
    assert table.cells[0].rowspan == 2
    assert table.cells[1].colspan == 2
    assert next(cell for cell in table.cells if cell.text == "North").role == "row_header"
    assert all(fragment.page_number == 2 for cell in table.cells for fragment in cell.fragments)
    assert render_table(table).startswith("<table><thead>")
    assert [record.stage for record in diagnostics[-3:]] == [
        TableDiagnosticStage.CELL_ASSIGNMENT,
        TableDiagnosticStage.SPAN_RECONSTRUCTION,
        TableDiagnosticStage.RENDERING_CLASSIFICATION,
    ]
    assert diagnostics[-2].metric("spanning_cells") == 2
    assert diagnostics[-2].metric("preserved_spans") == 2
    assert diagnostics[-2].metric("created_spans") == 0
    assert diagnostics[-2].metric("removed_spans") == 0
    assert diagnostics[-2].outcome == TableDiagnosticOutcome.UNCHANGED
    assert diagnostics[-2].reason == "finder_spans_preserved"
    assert diagnostics[-1].reason == "multiple_header_rows,spanning_cells"


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pymupdf_coalesces_empty_stub_below_multilevel_header(rotation: int) -> None:
    class StackedStubTable(FakeTable):
        cells = [
            (0.0, 0.0, 10.0, 10.0),
            (0.0, 10.0, 10.0, 20.0),
            (10.0, 0.0, 30.0, 10.0),
            (10.0, 10.0, 20.0, 20.0),
            (20.0, 10.0, 30.0, 20.0),
            (0.0, 20.0, 10.0, 30.0),
            (10.0, 20.0, 20.0, 30.0),
            (20.0, 20.0, 30.0, 30.0),
        ]
        header = FakeHeader(external=False, names=["Country", "Years", None])

        def extract(self) -> list[list[str | None]]:
            return [["Country", "Years", None], [None, "2025", "2026"], ["North", "10", "12"]]

    matrix = pymupdf.Matrix(1, 1).prerotate(rotation)

    def output_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
        mapped = pymupdf.Rect(*bbox) * matrix
        return mapped.x0, mapped.y0, mapped.x1, mapped.y1

    diagnostics = []
    table = table_structure_from_pymupdf(
        StackedStubTable(), page_number=1, output_bbox=output_bbox, diagnostics=diagnostics
    )
    country = next(cell for cell in table.cells if cell.text == "Country")
    span_diagnostic = next(
        record for record in diagnostics if record.stage == TableDiagnosticStage.SPAN_RECONSTRUCTION
    )

    assert country.rowspan == 2
    assert len([cell for cell in table.cells if cell.column_index == 0 and cell.row_index < 2]) == 1
    assert span_diagnostic.outcome == TableDiagnosticOutcome.APPLIED
    assert span_diagnostic.reason == "post_finder_stages_created_spans"
    assert span_diagnostic.metric("finder_spanning_cells") == 1
    assert span_diagnostic.metric("preserved_spans") == 1
    assert span_diagnostic.metric("created_spans") == 1
    assert span_diagnostic.metric("removed_spans") == 0
    assert span_diagnostic.metric("spanning_cells") == 2


def test_pymupdf_external_header_does_not_invent_table_header() -> None:
    source = FakeTable()
    source.header = FakeHeader(external=True, names=["External title"])

    table = table_structure_from_pymupdf(source, page_number=1)

    assert table.header_row_count == 0
    assert "no_header" in table.classification_reasons


def test_pymupdf_numeric_continuation_row_is_not_a_header() -> None:
    source = FakeTable()
    source.header = FakeHeader(external=False, names=["63", "Malaysia", "0.807"])

    table = table_structure_from_pymupdf(source, page_number=1)

    assert table.header_row_count == 0
    assert "no_header" in table.classification_reasons


def test_pymupdf_numeric_years_below_named_header_remain_headers() -> None:
    table = table_structure_from_pymupdf(FakeTable(), page_number=1)

    assert table.header_row_count == 2


def test_pymupdf_section_band_above_data_is_not_a_header() -> None:
    class SectionBandTable(FakeTable):
        bbox = (0.0, 0.0, 30.0, 20.0)
        cells = [
            (0.0, 0.0, 30.0, 10.0),
            (0.0, 10.0, 10.0, 20.0),
            (10.0, 10.0, 20.0, 20.0),
            (20.0, 10.0, 30.0, 20.0),
        ]
        header = FakeHeader(external=False, names=["Very high human development", None, None])

        def extract(self) -> list[list[str | None]]:
            return [["Very high human development", None, None], ["63", "Malaysia", "0.807"]]

    table = table_structure_from_pymupdf(SectionBandTable(), page_number=1)

    assert table.header_row_count == 0
    assert "no_header" in table.classification_reasons
    assert table.cells[0].role == "header"


@pytest.mark.parametrize(
    "section_label",
    ["ASSETS:", "Liabilities and equity", "OPERATING SEGMENTS"],
)
def test_full_width_body_section_role_is_independent_of_label_text(section_label: str) -> None:
    class FullWidthBodySectionTable(FakeTable):
        bbox = (0.0, 0.0, 30.0, 30.0)
        cells = [
            (0.0, 0.0, 10.0, 10.0),
            (10.0, 0.0, 20.0, 10.0),
            (20.0, 0.0, 30.0, 10.0),
            (0.0, 10.0, 30.0, 20.0),
            (0.0, 20.0, 10.0, 30.0),
            (10.0, 20.0, 20.0, 30.0),
            (20.0, 20.0, 30.0, 30.0),
        ]
        header = FakeHeader(external=False, names=["Item", "2024", "2023"])

        def extract(self) -> list[list[str | None]]:
            return [
                ["Item", "2024", "2023"],
                [section_label, None, None],
                ["Cash", "10", "9"],
            ]

    table = table_structure_from_pymupdf(FullWidthBodySectionTable(), page_number=1)
    section = next(cell for cell in table.cells if cell.text == section_label)

    assert table.header_row_count == 1
    assert section.row_index == table.header_row_count
    assert section.colspan == table.column_count
    assert section.role == "row_header"
    assert render_table(table).count('<th scope="row" colspan="3">') == 1


def test_split_stub_section_band_assigns_row_header_roles_across_the_row() -> None:
    class SplitStubSectionTable(FakeTable):
        bbox = (0.0, 0.0, 30.0, 40.0)
        cells = [
            (0.0, 0.0, 10.0, 10.0),
            (10.0, 0.0, 20.0, 10.0),
            (20.0, 0.0, 30.0, 10.0),
            (0.0, 10.0, 10.0, 20.0),
            (10.0, 10.0, 20.0, 20.0),
            (20.0, 10.0, 30.0, 20.0),
            (0.0, 20.0, 10.0, 30.0),
            (10.0, 20.0, 30.0, 30.0),
            (0.0, 30.0, 10.0, 40.0),
            (10.0, 30.0, 20.0, 40.0),
            (20.0, 30.0, 30.0, 40.0),
        ]
        header = FakeHeader(external=False, names=["Item", "2024", "2023"])

        def extract(self) -> list[list[str | None]]:
            return [
                ["Item", "2024", "2023"],
                ["Cash", "10", "9"],
                [None, "Civic & Institutional", None],
                ["Debt", "8", "7"],
            ]

    table = table_structure_from_pymupdf(SplitStubSectionTable(), page_number=1)
    section = [cell for cell in table.cells if cell.row_index == 2]

    assert [cell.role for cell in section] == ["row_header", "row_header"]
    assert next(cell for cell in section if cell.text).colspan == 2


def test_pymupdf_discards_phantom_cell_containing_real_grid() -> None:
    class PhantomTable(FakeTable):
        bbox = (0.0, 0.0, 100.0, 20.0)
        cells = [
            (0.0, 0.0, 100.0, 20.0),
            (5.0, 0.0, 95.0, 10.0),
            (5.0, 10.0, 95.0, 20.0),
        ]
        header = FakeHeader(external=False, names=[None, "Header"])

        def extract(self) -> list[list[str | None]]:
            return [["phantom", "Header"], [None, "Body"]]

    table = table_structure_from_pymupdf(PhantomTable(), page_number=1)

    assert table.row_count == 2
    assert table.column_count == 1
    assert [cell.text for cell in table.cells] == ["Header", "Body"]


def test_pymupdf_filters_degenerate_geometry_cells() -> None:
    source = FakeTable()
    source.cells = [*source.cells, (0.0, 0.0, 0.0, 30.0)]

    table = table_structure_from_pymupdf(source, page_number=1)

    assert all(cell.colspan >= 1 and cell.rowspan >= 1 for cell in table.cells)


def test_pymupdf_uses_row_cell_association_for_ragged_columns() -> None:
    class FakeRow:
        def __init__(self, cells: list[tuple[float, float, float, float]]) -> None:
            self.cells = cells

    class RaggedTable(FakeTable):
        bbox = (0.0, 0.0, 90.0, 20.0)
        cells = [
            (0.0, 0.0, 30.0, 10.0),
            (30.0, 0.0, 60.0, 10.0),
            (60.0, 0.0, 90.0, 10.0),
            (0.0, 10.0, 45.0, 20.0),
            (45.0, 10.0, 60.0, 20.0),
            (60.0, 10.0, 90.0, 20.0),
        ]
        rows = [FakeRow(cells[:3]), FakeRow(cells[3:])]
        header = FakeHeader(external=False, names=["A", "B", "C"])

        def extract(self) -> list[list[str | None]]:
            return [["A", "B", "C"], ["D", "E", "F"]]

    table = table_structure_from_pymupdf(RaggedTable(), page_number=1)
    cell_c = next(
        cell for cell in table.cells if cell.fragments[0].bbox.x0 == 60 and cell.fragments[0].bbox.y0 == 0
    )

    assert cell_c.text == "C"


def test_reconstruction_preserves_conversion_output_for_finder_grid_v1() -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float] | None]) -> None:
            self.cells = list(cells)

    class AssociatedTable(FakeTable):
        rows = [
            GeometryRow([FakeTable.cells[0], FakeTable.cells[1], None]),
            GeometryRow([None, FakeTable.cells[2], FakeTable.cells[3]]),
            GeometryRow(FakeTable.cells[4:]),
        ]

    token = NativeToken(
        token_id=NativeTokenId("default-token"),
        frame=CoordinateFrame.DETECTOR_PAGE,
        bbox=(1.0, 1.0, 5.0, 5.0),
        baseline=5.0,
        text="ignored",
    )

    class ProvenancedAssociatedTable(AssociatedTable):
        provenance = TableProvenance(
            finder=FinderProvenance.DEFAULT,
            frame=CoordinateFrame.DETECTOR_PAGE,
            native_token_ids=(token.token_id,),
            native_tokens=(token,),
        )

    legacy = table_structure_from_pymupdf(FakeTable(), page_number=2)
    associated = table_structure_from_pymupdf(AssociatedTable(), page_number=2)
    authoritative = table_structure_from_pymupdf(ProvenancedAssociatedTable(), page_number=2)

    assert associated.model_dump_json() == legacy.model_dump_json()
    without_source_ids = lambda table: table.model_copy(  # noqa: E731
        update={
            "cells": [
                cell.model_copy(
                    update={
                        "fragments": [
                            fragment.model_copy(update={"source_item_ids": []}) for fragment in cell.fragments
                        ]
                    }
                )
                for cell in table.cells
            ]
        }
    )
    assert without_source_ids(authoritative) == legacy
    # Geometry alone cannot authorize a content citation: the native word text is
    # incompatible with the logical cell, so assignment fails closed.
    assert all(not fragment.source_item_ids for cell in authoritative.cells for fragment in cell.fragments)
    assert all(not fragment.source_item_ids for cell in associated.cells for fragment in cell.fragments)


def test_reconstruction_fails_fast_for_malformed_and_dangling_finder_grids() -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float] | None]) -> None:
            self.cells = list(cells)

    class Grid:
        bbox = (0.0, 0.0, 20.0, 10.0)
        cells = [(0.0, 0.0, 10.0, 10.0)]
        header = FakeHeader(external=True, names=[])
        rows = [GeometryRow(cells)]

        def extract(self) -> list[list[str | None]]:
            return [["A", "B"]]

    with pytest.raises(ValueError, match="geometry width does not match extracted width"):
        reconstruct_table(Grid(), None)

    Grid.rows = [GeometryRow([(10.0, 0.0, 20.0, 10.0)])]
    Grid.extract = lambda _self: [["B"]]  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="references unknown cell bbox"):
        reconstruct_table(Grid(), None)

    Grid.cells = [(0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)]
    Grid.rows = [GeometryRow([Grid.cells[0]])]
    with pytest.raises(ValueError, match="cells not referenced by geometry rows"):
        reconstruct_table(Grid(), None)


@pytest.mark.parametrize(
    ("update", "error"),
    [
        ({"stage": "row_recovery"}, TypeError),
        ({"outcome": "accepted"}, TypeError),
        ({"metrics": ((1, "value"),)}, TypeError),
        ({"metrics": (("value", object()),)}, TypeError),
        ({"metrics": (("value", float("nan")),)}, ValueError),
        ({"metrics": (("value", float("inf")),)}, ValueError),
    ],
)
def test_table_diagnostics_reject_non_typed_or_non_json_scalar_values(
    update: dict[str, object], error: type[Exception]
) -> None:
    values: dict[str, object] = {
        "stage": TableDiagnosticStage.ROW_RECOVERY,
        "outcome": TableDiagnosticOutcome.ACCEPTED,
    }
    values.update(update)

    with pytest.raises(error):
        TableDiagnostic(**values)  # type: ignore[arg-type]


def test_reconstruction_closes_snapshot_lifetime() -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float] | None]) -> None:
            self.cells = list(cells)

    class MutableGrid:
        def __init__(self) -> None:
            self.bbox = [0.0, 0.0, 10.0, 10.0]
            self.cells = [[0.0, 0.0, 10.0, 10.0]]
            self.header = FakeHeader(external=False, names=["Before"])
            self.rows = [GeometryRow([(0.0, 0.0, 10.0, 10.0)])]
            self.extracted: list[list[str | None]] = [["Before"]]

        def extract(self) -> list[list[str | None]]:
            return self.extracted

    source = MutableGrid()
    reconstruction = reconstruct_table(source, None)
    source.bbox[:] = [1.0, 1.0, 2.0, 2.0]
    source.cells[0][:] = [1.0, 1.0, 2.0, 2.0]
    source.header.names[0] = "After"
    source.rows[0].cells[0] = (1.0, 1.0, 2.0, 2.0)
    source.extracted[0][0] = "After"

    assert reconstruction.adapter == "finder_grid_v1"
    assert reconstruction.bbox == (0.0, 0.0, 10.0, 10.0)
    assert reconstruction.header.names == ("Before",)
    assert reconstruction.extracted_rows == (("Before",),)
    assert reconstruction.logical_cells[0].text == "Before"


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.5])
@pytest.mark.parametrize(("dx", "dy"), [(0.0, 0.0), (37.0, -19.0)])
def test_rule_backed_inset_tracks_collapse_to_logical_grid_at_any_scale_and_translation(
    scale: float, dx: float, dy: float
) -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float] | None]) -> None:
            self.cells = list(cells)

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)

    header_cells = [
        box(left, 0, right, 10)
        for left, right in zip((0, 1, 49, 50, 51, 99), (1, 49, 50, 51, 99, 100), strict=True)
    ]
    first = [box(0, 10, 50, 30), None, None, box(50, 10, 100, 30), None, None]
    second = [box(0, 30, 50, 50), None, None, box(50, 30, 100, 50), None, None]

    class Grid:
        bbox = box(0, 0, 100, 50)
        header = FakeHeader(external=False, names=[None, "Name", None, None, "Value", None])
        rows = [GeometryRow(header_cells), GeometryRow(first), GeometryRow(second)]
        cells = [cell for row in rows for cell in row.cells if cell is not None]

        def extract(self) -> list[list[str | None]]:
            return [
                [None, "Name", None, None, "Value", None],
                ["Alpha", None, None, "One", None, None],
                ["Beta", None, None, "Two", None, None],
            ]

    rules = tuple(
        NativeRule(
            rule_id=NativeRuleId(f"rule-{row}-{column}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=column * 50 * scale + dx,
            x1=(column + 1) * 50 * scale + dx,
            y=(row * 10 if row == 0 else 10 + (row - 1) * 20) * scale + dy,
        )
        for row in range(4)
        for column in range(2)
    )
    evidence = TableProvenance(
        finder=FinderProvenance.DEFAULT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_rules=rules,
    )

    reconstruction = reconstruct_table(cast(FinderSnapshot, Grid()), evidence)

    assert reconstruction.logical_grid_indices
    assert reconstruction.extracted_rows == (("Name", "Value"), ("Alpha", "One"), ("Beta", "Two"))
    assert [(cell.row_index, cell.column_index, cell.text) for cell in reconstruction.logical_cells] == [
        (0, 0, "Name"),
        (0, 1, "Value"),
        (1, 0, "Alpha"),
        (1, 1, "One"),
        (2, 0, "Beta"),
        (2, 1, "Two"),
    ]
    strategy_records = [
        record
        for record in reconstruction.diagnostics
        if record.stage == TableDiagnosticStage.RECONSTRUCTION_STRATEGY
    ]
    assert [(record.outcome, record.reason) for record in strategy_records] == [
        (TableDiagnosticOutcome.REJECTED, "external_financial_header"),
        (TableDiagnosticOutcome.REJECTED, "rule_connected_external_header"),
        (TableDiagnosticOutcome.APPLIED, "ruled_inset_grid"),
    ]
    assert all(
        record.metric("failure_reason") == "gates_not_satisfied"
        for record in strategy_records
        if record.outcome == TableDiagnosticOutcome.REJECTED
    )
    assert any(
        record.stage == TableDiagnosticStage.ROW_RECOVERY and record.metric("after") == 3
        for record in reconstruction.diagnostics
    )
    assert any(
        record.stage == TableDiagnosticStage.COLUMN_RECOVERY and record.metric("after") == 2
        for record in reconstruction.diagnostics
    )


@pytest.mark.parametrize("failure", ["missing_helper", "nonuniform_helper", "unruled"])
def test_inset_track_reconstruction_fails_closed_without_uniform_rule_backed_helpers(
    failure: str,
) -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float] | None]) -> None:
            self.cells = list(cells)

    boundaries = [0.0, 1.0, 49.0, 50.0, 51.0, 99.0, 100.0]
    if failure == "nonuniform_helper":
        boundaries = [0.0, 4.0, 49.0, 50.0, 51.0, 99.0, 100.0]
    header_cells: list[tuple[float, float, float, float] | None] = [
        (left, 0.0, right, 10.0) for left, right in zip(boundaries, boundaries[1:], strict=False)
    ]
    if failure == "missing_helper":
        header_cells[2] = None
    body_cells = [
        (0.0, 10.0, 50.0, 30.0),
        None,
        None,
        (50.0, 10.0, 100.0, 30.0),
        None,
        None,
    ]

    class Grid:
        bbox = (0.0, 0.0, 100.0, 30.0)
        header = FakeHeader(external=False, names=[None, "Name", None, None, "Value", None])
        rows = [GeometryRow(header_cells), GeometryRow(body_cells)]
        cells = [cell for row in rows for cell in row.cells if cell is not None]

        def extract(self) -> list[list[str | None]]:
            return [
                [None, "Name", None, None, "Value", None],
                ["Alpha", None, None, "One", None, None],
            ]

    rules = ()
    if failure != "unruled":
        rules = tuple(
            NativeRule(
                rule_id=NativeRuleId(f"rule-{row}-{column}"),
                frame=CoordinateFrame.DETECTOR_PAGE,
                x0=column * 50,
                x1=(column + 1) * 50,
                y=row * 10 if row == 0 else 10 + (row - 1) * 20,
            )
            for row in range(3)
            for column in range(2)
        )
    evidence = TableProvenance(
        finder=FinderProvenance.DEFAULT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_rules=rules,
    )

    reconstruction = reconstruct_table(cast(FinderSnapshot, Grid()), evidence)

    assert not reconstruction.logical_grid_indices
    assert len(reconstruction.extracted_rows[0]) == 6


@pytest.mark.parametrize(("scale", "dx", "dy"), [(0.5, 0.0, 0.0), (1.0, 37.0, -19.0), (2.5, 0.0, 41.0)])
def test_full_width_rules_collapse_physical_text_rows_at_any_scale_and_translation(
    scale: float, dx: float, dy: float
) -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float]]) -> None:
            self.cells = list(cells)

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)

    physical_boundaries = (0.0, 5.0, 10.0, 30.0, 40.0, 50.0)
    geometry = [
        [box(column * 10, top, (column + 1) * 10, bottom) for column in range(10)]
        for top, bottom in zip(physical_boundaries, physical_boundaries[1:], strict=False)
    ]
    extracted: list[list[str | None]] = [
        ["Group", *(None for _ in range(9))],
        [None for _ in range(10)],
        [f"H{column}" for column in range(10)],
        [f"B{column}" for column in range(10)],
        [None for _ in range(10)],
    ]
    rules = tuple(
        NativeRule(
            rule_id=NativeRuleId(f"rule-{index}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=dx,
            x1=100 * scale + dx,
            y=value * scale + dy,
        )
        for index, value in enumerate((0.0, 10.0, 30.0, 50.0))
    )
    evidence = TableProvenance(
        finder=FinderProvenance.DEFAULT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_rules=rules,
    )

    class Grid:
        bbox = box(0.0, 0.0, 100.0, 50.0)
        header = FakeHeader(external=False, names=extracted[0])
        rows = [GeometryRow(row) for row in geometry]
        cells = [cell for row in geometry for cell in row]
        provenance = evidence

        def extract(self) -> list[list[str | None]]:
            return extracted

    table = table_structure_from_pymupdf(Grid(), page_number=1)

    assert (table.row_count, table.column_count, table.header_row_count) == (3, 10, 2)
    assert len(table.cells) == 30
    assert next(cell for cell in table.cells if cell.text == "Group").row_index == 0
    assert next(cell for cell in table.cells if cell.text == "H9").row_index == 1


def test_physical_text_rows_fail_closed_without_complete_rule_boundaries() -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float]]) -> None:
            self.cells = list(cells)

    physical_boundaries = (0.0, 5.0, 10.0, 30.0, 40.0, 50.0)
    geometry = [
        [(column * 10.0, top, (column + 1) * 10.0, bottom) for column in range(10)]
        for top, bottom in zip(physical_boundaries, physical_boundaries[1:], strict=False)
    ]
    extracted: list[list[str | None]] = [[f"R{row}C{column}" for column in range(10)] for row in range(5)]
    rules = tuple(
        NativeRule(
            rule_id=NativeRuleId(f"rule-{index}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=0.0,
            x1=100.0,
            y=value,
        )
        for index, value in enumerate((0.0, 10.0, 50.0))
    )
    evidence = TableProvenance(
        finder=FinderProvenance.DEFAULT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_rules=rules,
    )

    class Grid:
        bbox = (0.0, 0.0, 100.0, 50.0)
        header = FakeHeader(external=False, names=extracted[0])
        rows = [GeometryRow(row) for row in geometry]
        cells = [cell for row in geometry for cell in row]
        provenance = evidence

        def extract(self) -> list[list[str | None]]:
            return extracted

    table = table_structure_from_pymupdf(Grid(), page_number=1)

    assert (table.row_count, table.column_count) == (5, 10)
    assert len(table.cells) == 50


def _synthetic_compressed_baseline_reconstruction(
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    reverse_evidence: bool = False,
    complete_rules: bool = True,
) -> TableReconstruction:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float]]) -> None:
            self.cells = list(cells)

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)

    physical_cells = [
        box(column * 150, row * 30, (column + 1) * 150, (row + 1) * 30)
        for row in range(3)
        for column in range(2)
    ]

    class Grid:
        bbox = box(0, 0, 300, 90)
        header = FakeHeader(external=False, names=["Method", "Metric A Metric B"])
        rows = [GeometryRow(physical_cells[row * 2 : (row + 1) * 2]) for row in range(3)]
        cells = physical_cells

        def extract(self) -> list[list[str | None]]:
            return [
                ["Method", "Metric A Metric B"],
                ["Alpha\nBeta\nGamma", "10 20\n11 21\n12 22"],
                ["Delta\nEpsilon\nZeta", "13 23\n14 24\n15 25"],
            ]

    rows = [
        ((5, 8, 55, 14), "Method"),
        ((165, 8, 205, 14), "Metric A"),
        ((250, 8, 290, 14), "Metric B"),
    ]
    for index, label in enumerate(("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"), start=1):
        baseline = 8 + index * 11
        rows.extend([
            ((5, baseline, 60, baseline + 6), label),
            ((170, baseline, 195, baseline + 6), str(9 + index)),
            ((255, baseline, 280, baseline + 6), str(19 + index)),
        ])
    tokens = tuple(
        NativeToken(
            token_id=NativeTokenId(f"compressed-token-{index:02d}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=box(*bbox),
            baseline=(bbox[3] * scale + dy),
            text=text,
        )
        for index, (bbox, text) in enumerate(rows)
    )
    rule_ys = (0, 15, 90) if complete_rules else (0, 90)
    rules = tuple(
        NativeRule(
            rule_id=NativeRuleId(f"compressed-rule-{index}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=dx,
            x1=300 * scale + dx,
            y=y * scale + dy,
        )
        for index, y in enumerate(rule_ys)
    )
    if reverse_evidence:
        tokens = tuple(reversed(tokens))
        rules = tuple(reversed(rules))
    evidence = TableProvenance(
        finder=FinderProvenance.DEFAULT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_token_ids=tuple(token.token_id for token in tokens),
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_tokens=tokens,
        native_rules=rules,
    )
    return reconstruct_table(cast(FinderSnapshot, Grid()), evidence)


@pytest.mark.parametrize(
    ("scale", "dx", "dy", "reverse_evidence"),
    [(0.5, 17.0, -9.0, False), (1.0, 0.0, 0.0, False), (2.5, -31.0, 23.0, True)],
)
def test_rule_backed_compressed_rows_reconstruct_by_baseline_under_affine_and_permutation(
    scale: float, dx: float, dy: float, reverse_evidence: bool
) -> None:
    reconstruction = _synthetic_compressed_baseline_reconstruction(
        scale=scale,
        dx=dx,
        dy=dy,
        reverse_evidence=reverse_evidence,
    )

    assert reconstruction.logical_grid_indices
    assert reconstruction.extracted_rows == (
        ("Method", "Metric A", "Metric B"),
        ("Alpha", "10", "20"),
        ("Beta", "11", "21"),
        ("Gamma", "12", "22"),
        ("Delta", "13", "23"),
        ("Epsilon", "14", "24"),
        ("Zeta", "15", "25"),
    )


def test_compressed_rows_fail_closed_without_three_rule_levels() -> None:
    reconstruction = _synthetic_compressed_baseline_reconstruction(complete_rules=False)

    assert not reconstruction.logical_grid_indices
    assert len(reconstruction.extracted_rows) == 3
    assert len(reconstruction.extracted_rows[0]) == 2


def _synthetic_borderless_reconstruction(
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    reverse_tokens: bool = False,
    reverse_rules: bool = False,
    duplicate_geometry: bool = False,
    duplicate_geometry_text: str = "40",
    sparse: bool = False,
) -> TableReconstruction:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float]]) -> None:
            self.cells = list(cells)

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (
            x0 * scale + dx,
            y0 * scale + dy,
            x1 * scale + dx,
            y1 * scale + dy,
        )

    cells = [
        box(column * 100, row * 10, (column + 1) * 100, (row + 1) * 10)
        for row in range(5)
        for column in range(3)
    ]

    class Grid:
        bbox = box(0, 0, 300, 50)
        header = FakeHeader(external=False, names=["Item", "Amounts", None])
        rows = [GeometryRow(cells[row * 3 : (row + 1) * 3]) for row in range(5)]

        def extract(self) -> list[list[str | None]]:
            return [
                ["Item", "Amounts", None],
                [None, None, None],
                ["Long Header", "$ 10", "20"],
                [None, None, None],
                ["Body", "30", "40"],
            ]

    Grid.cells = cells  # type: ignore[attr-defined]
    raw_tokens = [
        (box(8, 1, 28, 8), "Item"),
        (box(155, 1, 205, 8), "Amounts"),
        (box(8, 21, 28, 28), "Long"),
        (box(30, 21, 58, 28), "Header"),
        (box(112, 21, 118, 28), "$"),
        (box(124, 21, 142, 28), "10"),
        (box(224, 21, 242, 28), "20"),
        (box(8, 41, 28, 48), "Body"),
        (box(124, 41, 142, 48), "30"),
        (box(224, 41, 242, 48), "40"),
    ]
    if sparse:
        raw_tokens = [raw_tokens[index] for index in (0, 2, 3, 7, 8)]
    if duplicate_geometry:
        raw_tokens.append((raw_tokens[-1][0], duplicate_geometry_text))
    tokens = tuple(
        NativeToken(
            token_id=NativeTokenId(f"token-{index:02d}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=bbox,
            baseline=bbox[3],
            text=text,
        )
        for index, (bbox, text) in enumerate(raw_tokens)
    )
    if reverse_tokens:
        tokens = tuple(reversed(tokens))
    rules = (
        NativeRule(
            rule_id=NativeRuleId("rule-01"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=100 * scale + dx,
            x1=300 * scale + dx,
            y=9 * scale + dy,
        ),
        NativeRule(
            rule_id=NativeRuleId("rule-02"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=0 * scale + dx,
            x1=100 * scale + dx,
            y=29 * scale + dy,
        ),
    )
    if reverse_rules:
        rules = tuple(reversed(rules))
    bands = tuple(
        GeometricBand(
            band_id=f"band-{row}",
            kind=GeometricBandKind.FINDER_ROW,
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=box(0, row * 10, 300, (row + 1) * 10),
        )
        for row in range(5)
    )
    evidence = TableProvenance(
        finder=FinderProvenance.BORDERLESS_TEXT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        geometric_bands=bands,
        native_token_ids=tuple(token.token_id for token in tokens),
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_tokens=tokens,
        native_rules=rules,
    )
    return reconstruct_table(cast(FinderSnapshot, Grid()), evidence)


def test_borderless_reconstruction_suppresses_empty_bands_and_uses_positive_span_evidence() -> None:
    reconstruction = _synthetic_borderless_reconstruction()

    assert reconstruction.extracted_rows == (
        ("Item", "Amounts", None),
        ("Long Header", "$ 10", "20"),
        ("Body", "30", "40"),
    )
    assert reconstruction.provenance is not None
    assert len(reconstruction.provenance.geometric_band_ids) == 5
    amounts = next(cell for cell in reconstruction.logical_cells if cell.text == "Amounts")
    assert amounts.column_index == 1
    assert amounts.bbox[2] == 300


def test_borderless_reconstruction_falls_back_for_sparse_rows() -> None:
    reconstruction = _synthetic_borderless_reconstruction(sparse=True)

    assert len(reconstruction.extracted_rows) == 5
    assert reconstruction.extracted_rows[1] == (None, None, None)


def test_borderless_reconstruction_rejects_duplicate_geometry_and_is_transform_permutation_invariant() -> (
    None
):
    baseline = _synthetic_borderless_reconstruction()
    transformed = _synthetic_borderless_reconstruction(
        scale=2.5,
        dx=17,
        dy=31,
        reverse_tokens=True,
        reverse_rules=True,
    )
    ambiguous = _synthetic_borderless_reconstruction(duplicate_geometry=True)
    conflicting_overlay = _synthetic_borderless_reconstruction(
        duplicate_geometry=True,
        duplicate_geometry_text="conflict",
    )

    assert transformed.extracted_rows == baseline.extracted_rows
    assert len(transformed.logical_cells) == len(baseline.logical_cells)
    for rejected in (ambiguous, conflicting_overlay):
        assert len(rejected.extracted_rows) == 5
        assert rejected.extracted_rows[1] == (None, None, None)


@pytest.mark.parametrize(
    ("scale", "dx", "dy", "reverse_evidence"),
    [(0.5, 17.0, -9.0, False), (1.0, 0.0, 0.0, False), (2.5, -31.0, 23.0, True)],
)
def test_borderless_reconstruction_compacts_helper_columns_and_clips_isolated_margin_text(
    scale: float,
    dx: float,
    dy: float,
    reverse_evidence: bool,
) -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float]]) -> None:
            self.cells = list(cells)

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)

    cells = [
        box(column * 50, row * 10, (column + 1) * 50, (row + 1) * 10)
        for row in range(10)
        for column in range(6)
    ]

    class ExpandedGrid:
        bbox = box(0, 0, 300, 100)
        header = FakeHeader(external=False, names=["", "", "", "", "2024", "2023"])
        rows = [GeometryRow(cells[row * 6 : (row + 1) * 6]) for row in range(10)]

        def extract(self) -> list[list[str | None]]:
            return [["", "", "", "", "", ""] for _ in range(10)]

    ExpandedGrid.cells = cells  # type: ignore[attr-defined]
    raw_tokens = [
        (box(80, 2, 220, 8), "Detached centered title"),
        (box(10, 22, 45, 28), "Years Ended"),
        (box(218, 22, 238, 28), "2024"),
        (box(268, 22, 288, 28), "2023"),
        (box(10, 32, 28, 38), "Long"),
        (box(30, 32, 55, 38), "adjustment"),
        (box(57, 32, 78, 38), "phrase"),
        (box(80, 32, 98, 38), "that"),
        (box(100, 32, 125, 38), "wraps"),
        (box(127, 32, 145, 38), "onto"),
        (box(30, 42, 72, 48), "continuation"),
        (box(10, 52, 45, 58), "Alpha"),
        (box(218, 52, 238, 58), "100"),
        (box(268, 52, 288, 58), "90"),
        (box(10, 62, 45, 68), "Beta"),
        (box(218, 62, 238, 68), "80"),
        (box(268, 62, 288, 68), "70"),
        (box(10, 72, 45, 78), "Total"),
        (box(218, 72, 238, 78), "180"),
        (box(268, 72, 288, 78), "160"),
        (box(70, 94, 110, 100), "Detached"),
        (box(112, 94, 152, 100), "centered"),
        (box(154, 94, 195, 100), "margin"),
        (box(197, 94, 230, 100), "text"),
    ]
    tokens = tuple(
        NativeToken(
            token_id=NativeTokenId(f"expanded-token-{index:02d}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=bbox,
            baseline=bbox[3],
            text=text,
        )
        for index, (bbox, text) in enumerate(raw_tokens)
    )
    rules = (
        NativeRule(
            rule_id=NativeRuleId("expanded-top-rule"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=0 * scale + dx,
            x1=300 * scale + dx,
            y=20 * scale + dy,
        ),
    )
    if reverse_evidence:
        tokens = tuple(reversed(tokens))
        rules = tuple(reversed(rules))
    evidence = TableProvenance(
        finder=FinderProvenance.BORDERLESS_TEXT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_token_ids=tuple(token.token_id for token in tokens),
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_tokens=tokens,
        native_rules=rules,
    )

    reconstruction = reconstruct_table(cast(FinderSnapshot, ExpandedGrid()), evidence)

    assert reconstruction.logical_grid_indices
    assert reconstruction.bbox == box(0, 20, 300, 78)
    assert reconstruction.extracted_rows == (
        ("Years Ended", "2024", "2023"),
        ("Long adjustment phrase that wraps onto continuation", "", ""),
        ("Alpha", "100", "90"),
        ("Beta", "80", "70"),
        ("Total", "180", "160"),
    )
    assert len(reconstruction.logical_cells) == 15


def test_multilevel_header_depth_stops_before_sparse_body_section_row() -> None:
    class SectionedBodyTable(FakeTable):
        bbox = (0.0, 0.0, 30.0, 40.0)
        cells = [
            (0.0, 0.0, 10.0, 20.0),
            (10.0, 0.0, 30.0, 10.0),
            (10.0, 10.0, 20.0, 20.0),
            (20.0, 10.0, 30.0, 20.0),
            (0.0, 20.0, 10.0, 30.0),
            (10.0, 10.0, 20.0, 30.0),
            (20.0, 10.0, 30.0, 30.0),
            (0.0, 30.0, 10.0, 40.0),
            (10.0, 30.0, 20.0, 40.0),
            (20.0, 30.0, 30.0, 40.0),
        ]
        header = FakeHeader(external=False, names=["Region", "Revenue", None])

        def extract(self) -> list[list[str | None]]:
            return [
                ["Region", "Revenue", None],
                [None, "2025", "2026"],
                ["World", None, None],
                ["North", "10", "12"],
            ]

    table = table_structure_from_pymupdf(SectionedBodyTable(), page_number=1)

    assert table.header_row_count == 2
    assert next(cell for cell in table.cells if cell.text == "World").role == "row_header"


def test_multilevel_header_depth_stops_at_first_dense_numeric_body_row() -> None:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float] | None]) -> None:
            self.cells = list(cells)

    class ThreeRowHeader:
        bbox = (0.0, 0.0, 90.0, 40.0)
        header = FakeHeader(
            external=False,
            names=["Years Ended", "2024", None, None, None, "2023", None, None, None],
        )
        rows = [
            GeometryRow([
                (0.0, 0.0, 10.0, 10.0),
                (10.0, 0.0, 50.0, 10.0),
                None,
                None,
                None,
                (50.0, 0.0, 90.0, 10.0),
                None,
                None,
                None,
            ]),
            GeometryRow([(column * 10.0, 10.0, (column + 1) * 10.0, 20.0) for column in range(9)]),
            GeometryRow([(column * 10.0, 20.0, (column + 1) * 10.0, 30.0) for column in range(9)]),
            GeometryRow([(column * 10.0, 30.0, (column + 1) * 10.0, 40.0) for column in range(9)]),
        ]
        cells = [cell for row in rows for cell in row.cells if cell is not None]

        def extract(self) -> list[list[str | None]]:
            return [
                [None, "2024", None, None, None, "2023", None, None, None],
                [None, None, "Management", None, None, None, "Management", None, None],
                [
                    "Item",
                    "Program",
                    "and General",
                    "Fundraising",
                    "Total",
                    "Program",
                    "and General",
                    "Fundraising",
                    "Total",
                ],
                ["Salaries", "240", "174", "3", "417", "202", "123", "2", "328"],
            ]

    table = table_structure_from_pymupdf(ThreeRowHeader(), page_number=1)

    assert (table.row_count, table.column_count, table.header_row_count) == (4, 9, 3)
    assert all(cell.role == "header" for cell in table.cells if cell.row_index < 3)
    assert next(cell for cell in table.cells if cell.text == "Salaries").role == "row_header"


def test_page_extraction_rejects_empty_tables_without_id_gaps() -> None:
    class EmptyTable:
        bbox = (40.0, 40.0, 50.0, 50.0)
        cells = [(40.0, 40.0, 50.0, 50.0)]
        header = FakeHeader(external=True, names=[])

        def extract(self) -> list[list[str | None]]:
            return [[""]]

    class Finder:
        tables = [EmptyTable(), FakeTable()]

    class FakePage:
        number = 0
        rotation = 0
        rotation_matrix = pymupdf.Matrix(1, 0, 0, 1, 0, 0)
        rect = pymupdf.Rect(0, 0, 100, 100)
        cropbox = pymupdf.Rect(0, 0, 100, 100)

        def find_tables(self, **_kwargs: str) -> Finder:
            return Finder()

        def get_drawings(self) -> list[dict[str, object]]:
            return []

        def get_text(
            self, mode: str, **_kwargs: object
        ) -> str | list[tuple[float, float, float, float, str]]:
            if mode == "blocks":
                return []
            return ""

    diagnostics = []
    elements = extract_page_table_elements(cast(pymupdf.Page, FakePage()), "doc", diagnostics=diagnostics)
    without_diagnostics = extract_page_table_elements(cast(pymupdf.Page, FakePage()), "doc")

    assert len(elements) == 1
    assert elements[0].element_id == "page-1-table-1"
    assert elements[0].order == 0
    assert elements[0].content == without_diagnostics[0].content
    assert without_diagnostics[0].structure.properties == elements[0].structure.properties
    diagnostic_properties = [
        json.loads(prop.value)
        for prop in elements[0].structure.properties
        if prop.key == "table_diagnostic_v1"
    ]
    assert diagnostic_properties
    boundary_stages = {
        TableDiagnosticStage.BOUNDARY_DETECTION,
        TableDiagnosticStage.BOUNDARY_FILTER,
        TableDiagnosticStage.BOUNDARY_DEDUPLICATION,
    }
    collected_boundaries = [record for record in diagnostics if record.stage in boundary_stages]
    property_boundaries = [
        record
        for record in diagnostic_properties
        if record["stage"] in {stage.value for stage in boundary_stages}
    ]
    assert len(collected_boundaries) == len(property_boundaries)
    assert len(collected_boundaries) == len(set(collected_boundaries))
    assert any(
        record.stage == TableDiagnosticStage.TABLE_QUALITY
        and record.outcome == TableDiagnosticOutcome.REJECTED
        and record.reason == "empty"
        for record in diagnostics
    )


def test_pymupdf_irregular_grid_uses_html() -> None:
    source = FakeTable()
    source.cells = [
        (0.0, 0.0, 10.0, 10.0),
        (20.0, 0.0, 30.0, 10.0),
    ]
    source.bbox = (0.0, 0.0, 40.0, 10.0)

    table = table_structure_from_pymupdf(source, page_number=1)

    assert table.representation == "html"
    assert "incomplete_grid" in table.classification_reasons


def test_fragmented_academic_grid_preserves_ambiguous_rule_fragments() -> None:
    pdf_path = Path("data/corpus/candidates/technical/academic-deep-residual-learning.pdf")
    elements = extract_document_elements(pdf_path, pages=[5])
    tables = [element.structure.table for element in elements if element.structure.table is not None]

    assert [(table.row_count, table.column_count) for table in tables] == [
        (1, 4),
        (5, 2),
    ]
    assert Counter(element.element_type for element in elements) == Counter({
        "paragraph": 122,
        "table": 2,
        "caption": 3,
        "figure": 1,
        "footnote": 1,
    })
    assert Counter(
        element.structure.paragraph.role for element in elements if element.structure.paragraph is not None
    ) == Counter({"body": 103, "figure_text": 19, "caption": 3, "footnote": 1})
    source_ids = [
        source_id
        for element in elements
        for fragment in element.fragments
        for source_id in fragment.source_item_ids
    ]
    assert source_ids and len(source_ids) == len(set(source_ids))
    _assert_native_word_centers_are_structurally_covered(pdf_path, 5, elements)
    preserved_text = " ".join(element.content for element in elements)
    for text in (
        "layer name",
        "152-layer",
        "conv2 x",
        "11.3×109",
        "Architectures for ImageNet",
        "Training on ImageNet",
        "Top-1 error",
        "reducing of the training error",
    ):
        assert text in preserved_text


def test_rule_backed_academic_tables_split_compressed_baselines_into_logical_rows() -> None:
    pdf_path = Path("data/corpus/candidates/technical/academic-deep-residual-learning.pdf")
    elements = extract_document_table_elements(pdf_path, "drl", pages=[6])
    reconstructed = [
        element.structure.table
        for element in elements
        if element.structure.table is not None
        and element.structure.table.row_count == 11
        and element.structure.table.column_count == 3
    ]

    assert len(reconstructed) == 2
    assert all(table.header_row_count == 1 for table in reconstructed)
    assert all(table.representation == "markdown" for table in reconstructed)
    assert [[cell.text for cell in table.cells if cell.row_index == 0] for table in reconstructed] == [
        ["model", "top-1 err.", "top-5 err."],
        ["method", "top-1 err.", "top-5 err."],
    ]
    assert all(len(table.cells) == 33 for table in reconstructed)


def test_rule_connected_repeated_header_is_recovered_and_merged_across_shifted_pages() -> None:
    pdf_path = Path("data/corpus/candidates/institutional/undp-hdr2024.pdf")
    elements = extract_document_table_elements(pdf_path, "undp", pages=[288, 289, 290])

    assert len(elements) == 1
    element = elements[0]
    table = element.structure.table
    assert table is not None
    assert (table.row_count, table.column_count, table.header_row_count) == (193, 9, 4)
    assert [fragment.page_number for fragment in element.fragments] == [288, 289, 290]
    assert [fragment.bbox.x0 for fragment in element.fragments] == pytest.approx([43.2, 50.4, 43.2])
    assert [
        [cell.text for cell in table.cells if cell.row_index == row] for row in range(table.header_row_count)
    ] == [
        ["", "", "", "SDG 3", "SDG 4.3", "SDG 4.4", "SDG 8.5", "", ""],
        [
            "",
            "",
            "Human Development Index (HDI)",
            "Life expectancy at birth",
            "Expected years of schooling",
            "Mean years of schooling",
            "Gross national income (GNI) per capita",
            "GNI per capita rank minus HDI rank",
            "HDI rank",
        ],
        ["", "", "Value", "(years)", "(years)", "(years)", "(2017 PPP $)"],
        ["HDI", "RANK", "2022", "2022", "2022a", "2022a", "2022", "2022b", "2021"],
    ]


def test_rule_connected_external_header_recovery_fails_closed_without_every_tier_rule() -> None:
    pdf_path = Path("data/corpus/candidates/institutional/undp-hdr2024.pdf")
    with pymupdf.open(pdf_path) as document:
        detected = detect_page_tables(document[287])
    assert len(detected) == 1
    snapshot = cast(FinderSnapshot, detected[0])
    evidence = cast(TableProvenance, getattr(snapshot, "provenance"))
    retained_rules = tuple(rule for rule in evidence.native_rules if not 129.5 <= rule.y <= 131.5)
    retained_ids = tuple(rule.rule_id for rule in retained_rules)
    weakened = TableProvenance(
        finder=evidence.finder,
        frame=evidence.frame,
        geometric_bands=evidence.geometric_bands,
        native_token_ids=evidence.native_token_ids,
        native_rule_ids=retained_ids,
        native_tokens=evidence.native_tokens,
        native_rules=retained_rules,
    )

    reconstruction = reconstruct_table(snapshot, weakened)

    assert reconstruction.recovered_header_row_count is None
    assert len(reconstruction.extracted_rows) == 63
    assert reconstruction.bbox[1] == pytest.approx(163.418)


def test_apple_balance_sheet_full_width_sections_are_row_headers() -> None:
    pdf_path = Path("data/corpus/candidates/institutional/apple-10k2024.pdf")
    elements = extract_document_table_elements(pdf_path, "apple", pages=[34])

    assert len(elements) == 1
    table = elements[0].structure.table
    assert table is not None
    section_cells = [
        cell for cell in table.cells if cell.text in {"ASSETS:", "LIABILITIES AND SHAREHOLDERS’ EQUITY:"}
    ]
    assert len(section_cells) == 2
    assert all(cell.row_index >= table.header_row_count for cell in section_cells)
    assert all(cell.column_index == 0 and cell.colspan == table.column_count for cell in section_cells)
    assert all(cell.role == "row_header" for cell in section_cells)
    assert all(
        f'<th scope="row" colspan="{table.column_count}">{cell.text}</th>' in elements[0].content
        for cell in section_cells
    )


def test_nist_helper_tracks_reconstruct_and_merge_without_token_loss() -> None:
    pdf_path = Path("data/corpus/candidates/institutional/nist-sp800-218.pdf")
    pages = list(range(14, 29))
    elements = extract_document_table_elements(pdf_path, "nist", pages=pages)

    assert len(elements) == 1
    element = elements[0]
    table = element.structure.table
    assert table is not None
    assert (table.row_count, table.column_count, table.header_row_count) == (52, 4, 1)
    assert len(table.cells) == 169
    assert Counter(cell.role for cell in table.cells) == Counter({"header": 8, "row_header": 20, "body": 141})
    assert [fragment.page_number for fragment in element.fragments] == pages
    assert element.structure.linked_element_ids == [f"page-{page}-table-1" for page in pages[1:]]
    assert [
        (cell.row_index, cell.rowspan) for cell in table.cells if cell.column_index == 0 and cell.rowspan > 1
    ] == [
        (2, 3),
        (5, 3),
        (8, 3),
        (11, 2),
        (13, 2),
        (18, 2),
        (21, 3),
        (25, 2),
        (27, 5),
        (32, 2),
        (34, 2),
        (36, 2),
        (38, 2),
        (40, 2),
        (43, 3),
        (46, 2),
        (48, 4),
    ]
    assert [
        (cell.row_index, cell.text) for cell in table.cells if cell.role == "header" and cell.row_index > 0
    ] == [
        (1, "Prepare the Organization (PO)"),
        (15, "Protect Software (PS)"),
        (20, "Produce Well-Secured Software (PW)"),
        (42, "Respond to Vulnerabilities (RV)"),
    ]
    final_task = next(cell for cell in table.cells if cell.row_index == 51 and cell.column_index == 1)
    final_reference = next(cell for cell in table.cells if cell.row_index == 51 and cell.column_index == 3)
    assert final_task.text.startswith("RV.3.4: Review the SDLC process")
    assert final_reference.text.endswith("SP800181: K0009, K0039, K0070")
    assert [fragment.page_number for fragment in final_reference.fragments] == [27, 28]

    unmerged_tokens: Counter[str] = Counter()
    with pymupdf.open(pdf_path) as document:
        fragments = [
            element for page in pages for element in extract_page_table_elements(document[page - 1], "nist")
        ]
    assert len(fragments) == 15
    assert (
        sum(
            fragment.structure.table.row_count - fragment.structure.table.header_row_count
            for fragment in fragments
            if fragment.structure.table is not None
        )
        - (table.row_count - table.header_row_count)
        == 6
    )
    for fragment in fragments:
        fragment_table = fragment.structure.table
        assert fragment_table is not None
        unmerged_tokens.update(
            token
            for cell in fragment_table.cells
            if cell.row_index >= fragment_table.header_row_count
            for token in re.findall(r"\S+", cell.text)
        )
    merged_tokens = Counter(
        token
        for cell in table.cells
        if cell.row_index >= table.header_row_count
        for token in re.findall(r"\S+", cell.text)
    )
    assert merged_tokens == unmerged_tokens


def _synthetic_rule_partitioned_form(
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    reverse_evidence: bool = False,
    missing_boundary: int | None = None,
    finder: FinderProvenance = FinderProvenance.DEFAULT,
) -> tuple[FinderSnapshot, TableProvenance]:
    class GeometryRow:
        def __init__(self, cells: Sequence[tuple[float, float, float, float] | None]) -> None:
            self.cells = list(cells)

    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)

    giant = box(0, 10, 60, 80)
    geometry = [
        [box(0, 0, 60, 10), box(60, 0, 80, 10), box(80, 0, 100, 10)],
        *[
            [
                giant if row == 1 else None,
                box(60, row * 10, 80, (row + 1) * 10),
                box(80, row * 10, 100, (row + 1) * 10),
            ]
            for row in range(1, 8)
        ],
    ]
    extracted: list[list[str | None]] = [
        ["Item", "Left", "Right"],
        ["Row 1 Row 2 Row 3 Row 4 Row 5 Row 6 Row 7", "1", "11"],
        *[[None, str(row), str(row + 10)] for row in range(2, 8)],
    ]

    class FormGrid:
        bbox = box(0, 0, 100, 80)
        header = FakeHeader(external=False, names=extracted[0])
        rows = [GeometryRow(row) for row in geometry]
        cells = list(dict.fromkeys(cell for row in geometry for cell in row if cell is not None))

        def extract(self) -> list[list[str | None]]:
            return extracted

    raw_tokens = [(box(5, row * 10 + 2, 20, row * 10 + 8), f"Row {row}") for row in range(1, 8)]
    tokens = tuple(
        NativeToken(
            token_id=NativeTokenId(f"form-token-{row}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=bbox,
            baseline=bbox[3],
            text=text,
        )
        for row, (bbox, text) in enumerate(raw_tokens, start=1)
    )
    rules = tuple(
        NativeRule(
            rule_id=NativeRuleId(f"form-rule-{boundary}"),
            frame=CoordinateFrame.DETECTOR_PAGE,
            x0=60 * scale + dx,
            x1=100 * scale + dx,
            y=boundary * 10 * scale + dy,
        )
        for boundary in range(1, 8)
        if boundary != missing_boundary
    )
    if reverse_evidence:
        tokens = tuple(reversed(tokens))
        rules = tuple(reversed(rules))
    evidence = TableProvenance(
        finder=finder,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_token_ids=tuple(token.token_id for token in tokens),
        native_rule_ids=tuple(rule.rule_id for rule in rules),
        native_tokens=tokens,
        native_rules=rules,
    )
    return cast(FinderSnapshot, FormGrid()), evidence


@pytest.mark.parametrize(
    ("scale", "dx", "dy", "reverse_evidence"),
    [(0.5, 17.0, -9.0, False), (1.0, 0.0, 0.0, False), (2.5, -31.0, 23.0, True)],
)
def test_rule_partitioned_form_splits_oversized_stub_with_transform_and_permutation_invariance(
    scale: float, dx: float, dy: float, reverse_evidence: bool
) -> None:
    snapshot, evidence = _synthetic_rule_partitioned_form(
        scale=scale,
        dx=dx,
        dy=dy,
        reverse_evidence=reverse_evidence,
    )

    reconstruction = reconstruct_table(snapshot, evidence)

    assert reconstruction.logical_grid_indices
    assert reconstruction.extracted_rows == (
        ("Item", "Left", "Right"),
        ("Row 1", "1", "11"),
        ("Row 2", "2", "12"),
        ("Row 3", "3", "13"),
        ("Row 4", "4", "14"),
        ("Row 5", "5", "15"),
        ("Row 6", "6", "16"),
        ("Row 7", "7", "17"),
    )


@pytest.mark.parametrize(
    ("missing_boundary", "finder"),
    [(4, FinderProvenance.DEFAULT), (None, FinderProvenance.LINES_STRICT)],
)
def test_rule_partitioned_form_fails_closed_without_complete_native_support(
    missing_boundary: int | None, finder: FinderProvenance
) -> None:
    snapshot, evidence = _synthetic_rule_partitioned_form(
        missing_boundary=missing_boundary,
        finder=finder,
    )

    reconstruction = reconstruct_table(snapshot, evidence)

    assert not reconstruction.logical_grid_indices
    assert reconstruction.extracted_rows[1][0] == "Row 1 Row 2 Row 3 Row 4 Row 5 Row 6 Row 7"


def test_form_page_external_sentence_is_prose_and_grid_keeps_reference_shape() -> None:
    pdf_path = Path("data/corpus/candidates/institutional/irs-form990.pdf")
    elements = extract_document_elements(pdf_path, pages=[10])
    tables = [element for element in elements if element.structure.table is not None]

    assert Counter(element.element_type for element in elements) == Counter({"paragraph": 17, "heading": 2})
    assert all(
        element.structure.paragraph is not None for element in elements if element.element_type == "paragraph"
    )
    source_ids = [
        source_id
        for element in elements
        for fragment in element.fragments
        for source_id in fragment.source_item_ids
    ]
    assert source_ids and len(source_ids) == len(set(source_ids))
    _assert_native_word_centers_are_structurally_covered(pdf_path, 10, elements)
    assert tables == []
    preserved = " ".join(element.content for element in elements)
    for text in (
        "Part IX",
        "Statement of Functional Expenses",
        "Check if Schedule O",
        "Do not include amounts",
        "1 Grants and other assistance",
        "26 Joint costs",
    ):
        assert text in preserved


def test_rule_partitioned_form_recovers_target_rows_without_splitting_vertical_section_labels() -> None:
    pdf_path = Path("data/corpus/candidates/institutional/irs-form990.pdf")
    elements = extract_document_table_elements(pdf_path, "form", pages=[9, 10, 11])
    tables = [element.structure.table for element in elements if element.structure.table is not None]

    assert [(table.row_count, table.column_count, table.header_row_count) for table in tables] == [
        (46, 10, 1),
        (38, 5, 1),
        (40, 7, 1),
    ]
    revenue = tables[0]
    balance = tables[-1]
    assert [
        next(cell.text for cell in revenue.cells if cell.row_index == row and cell.column_index == 1)
        for row in range(1, 5)
    ] == [
        "1a Federated campaigns",
        "b Membership dues",
        "c Fundraising events",
        "d Related organizations",
    ]
    assets = next(cell for cell in balance.cells if cell.text == "Assets")
    assert (assets.row_index, assets.rowspan, assets.column_index) == (1, 19, 0)
    assert next(
        cell.text for cell in balance.cells if cell.row_index == 2 and cell.column_index == 1
    ).startswith("2 Savings and temporary cash investments")


@pytest.mark.parametrize(
    ("slug", "page_number"),
    [
        ("us-fema-ics-203-v3", 1),
        ("us-fema-ics-206-v3", 1),
        ("us-uscis-i-90", 1),
    ],
)
def test_rule_partitioned_form_does_not_rewrite_heterogeneous_emergency_or_regulatory_grids(
    slug: str, page_number: int
) -> None:
    pdf_path = Path("data/corpus-expansion/difficult-strata/pdfs") / f"{slug}.pdf"
    with pymupdf.open(pdf_path) as document:
        detected = detect_page_tables(document[page_number - 1])

    assert detected
    assert all(
        not reconstruct_table(
            cast(FinderSnapshot, table), cast(TableProvenance, getattr(table, "provenance"))
        ).logical_grid_indices
        for table in detected
    )


def test_rule_partitioned_form_keeps_titles_instructions_and_grid_content_separate() -> None:
    pdf_path = Path("data/corpus/candidates/institutional/irs-form990.pdf")

    elements = extract_document_elements(pdf_path, pages=[9, 10, 11])
    by_page = {
        page: [element for element in elements if element.fragments[0].page_number == page]
        for page in (9, 10, 11)
    }

    assert [(element.element_type, element.content) for element in by_page[9][:3]] == [
        ("header", "Form 990 (2025) Page 9"),
        ("heading", "Part VIII Statement of Revenue"),
        (
            "paragraph",
            "Check if Schedule O contains a response or note to any line in this Part VIII "
            ". . . . . . . . . . . . .",
        ),
    ]
    assert [(element.element_type, element.content) for element in by_page[10][:4]] == [
        ("header", "Form 990 (2025) Page 10"),
        ("heading", "Part IX Statement of Functional Expenses"),
        (
            "paragraph",
            "Section 501(c)(3) and 501(c)(4) organizations must complete all columns. "
            "All other organizations must complete column (A).",
        ),
        (
            "paragraph",
            "Check if Schedule O contains a response or note to any line in this Part IX "
            ". . . . . . . . . . . . . Do not include amounts reported on lines 6b, 7b, "
            "8b, 9b, and 10b of Part VIII.",
        ),
    ]
    assert all(by_page.values())
    assert sum(element.element_type == "heading" for element in by_page[11]) == 1
    retained_page_11 = [element for element in by_page[11] if element.element_type == "paragraph"]
    assert retained_page_11
    assert all(element.content.strip() for element in retained_page_11)


def test_rule_partitioned_form_generalizes_to_independent_accepted_financial_form() -> None:
    pdf_path = Path("data/corpus-expansion/difficult-strata/pdfs/us-irs-f1120.pdf")
    with pymupdf.open(pdf_path) as document:
        detected = detect_page_tables(document[5])

    reconstruction = reconstruct_table(
        cast(FinderSnapshot, detected[0]),
        cast(TableProvenance, getattr(detected[0], "provenance")),
    )
    assert reconstruction.logical_grid_indices
    assert len(reconstruction.extracted_rows) == 37
    assert any(row[0] and "Cash" in row[0] for row in reconstruction.extracted_rows)
