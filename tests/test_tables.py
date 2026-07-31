import pytest

from app.pdf2md.schema import BoundingBox, PageFragment, TableCell, TableStructure
from app.pdf2md.table_provenance import TableDiagnosticOutcome, TableDiagnosticStage
from app.pdf2md.tables import TableFeatures, classify_table, render_table, strict_table_classification


def fragment(page_number: int = 1) -> PageFragment:
    return PageFragment(
        page_number=page_number,
        bbox=BoundingBox(x0=1, y0=1, x1=2, y1=2),
    )


def cell(
    row: int,
    column: int,
    text: str,
    *,
    role: str = "body",
    rowspan: int = 1,
    colspan: int = 1,
) -> TableCell:
    return TableCell.model_validate({
        "row_index": row,
        "column_index": column,
        "rowspan": rowspan,
        "colspan": colspan,
        "role": role,
        "text": text,
        "fragments": [fragment()],
    })


def test_simple_table_is_markdown_representable() -> None:
    cells = [
        cell(0, 0, "Name", role="header"),
        cell(0, 1, "Value", role="header"),
        cell(1, 0, "A|B"),
        cell(1, 1, "first\nsecond"),
    ]
    representation, reasons = classify_table(
        TableFeatures(row_count=2, column_count=2, header_row_count=1, cells=cells)
    )
    table = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation=representation,
        classification_reasons=reasons,
        cells=cells,
    )

    assert render_table(table) == ("| Name | Value |\n| --- | --- |\n| A\\|B | first<br>second |")


def test_multi_header_spanning_table_uses_complete_minified_html() -> None:
    cells = [
        cell(0, 0, "Region", role="header", rowspan=2),
        cell(0, 1, "Revenue", role="header", colspan=2),
        cell(1, 1, "2025", role="header"),
        cell(1, 2, "2026", role="header"),
        cell(2, 0, "North", role="row_header"),
        cell(2, 1, "10 & <1"),
        cell(2, 2, "12"),
    ]
    representation, reasons = classify_table(
        TableFeatures(row_count=3, column_count=3, header_row_count=2, cells=cells)
    )
    table = TableStructure(
        row_count=3,
        column_count=3,
        header_row_count=2,
        representation=representation,
        classification_reasons=reasons,
        cells=cells,
    )

    assert reasons == ["multiple_header_rows", "spanning_cells"]
    assert render_table(table) == (
        '<table><thead><tr><th scope="col" rowspan="2">Region</th>'
        '<th scope="colgroup" colspan="2">Revenue</th></tr><tr><th scope="col">2025</th>'
        '<th scope="col">2026</th></tr></thead><tbody><tr><th scope="row">North</th>'
        "<td>10 &amp; &lt;1</td><td>12</td></tr></tbody></table>"
    )


def test_no_header_table_uses_empty_thead() -> None:
    cells = [cell(0, 0, "A"), cell(0, 1, "1")]
    representation, reasons = classify_table(
        TableFeatures(row_count=1, column_count=2, header_row_count=0, cells=cells)
    )
    table = TableStructure(
        row_count=1,
        column_count=2,
        header_row_count=0,
        representation=representation,
        classification_reasons=reasons,
        cells=cells,
    )

    assert render_table(table) == (
        "<table><thead></thead><tbody><tr><td>A</td><td>1</td></tr></tbody></table>"
    )


def test_overlapping_cells_fail_fast() -> None:
    with pytest.raises(ValueError, match="overlap"):
        classify_table(
            TableFeatures(
                row_count=1,
                column_count=2,
                header_row_count=1,
                cells=[
                    cell(0, 0, "All", role="header", colspan=2),
                    cell(0, 1, "Duplicate", role="header"),
                ],
            )
        )


@pytest.mark.parametrize(
    ("feature_updates", "reason"),
    [
        ({"has_nested_content": True}, "nested_content"),
        ({"has_multi_paragraph_cells": True}, "multi_paragraph_cell"),
        ({"has_ambiguous_continuation": True}, "ambiguous_continuation"),
    ],
)
def test_semantically_complex_rectangular_grids_are_not_markdown(
    feature_updates: dict[str, bool], reason: str
) -> None:
    cells = [
        cell(0, 0, "Name", role="header"),
        cell(0, 1, "Value", role="header"),
        cell(1, 0, "A", role="row_header"),
        cell(1, 1, "B"),
    ]
    features = TableFeatures(
        row_count=2,
        column_count=2,
        header_row_count=1,
        cells=cells,
    ).model_copy(update=feature_updates)

    assert classify_table(features) == ("html", [reason])


def test_sparse_body_section_has_dedicated_reason_not_markdown() -> None:
    cells = [
        cell(0, 0, "Item", role="header"),
        cell(0, 1, "Value", role="header"),
        cell(0, 2, "Unit", role="header"),
        cell(1, 0, "Operating activities", role="header"),
        cell(1, 1, ""),
        cell(1, 2, ""),
        cell(2, 0, "Cash", role="row_header"),
        cell(2, 1, "10"),
        cell(2, 2, "USD"),
    ]

    assert classify_table(TableFeatures(row_count=3, column_count=3, header_row_count=1, cells=cells)) == (
        "html",
        ["sparse_section"],
    )


def test_strict_classification_recovers_sparse_semantics_from_stale_markdown_label() -> None:
    cells = [
        cell(0, 0, "Item", role="header"),
        cell(0, 1, "Value", role="header"),
        cell(0, 2, "Unit", role="header"),
        cell(1, 0, "Operating activities", role="row_header"),
        cell(1, 1, ""),
        cell(1, 2, ""),
        cell(2, 0, "Cash", role="row_header"),
        cell(2, 1, "10"),
        cell(2, 2, "USD"),
    ]
    stale = TableStructure(
        row_count=3,
        column_count=3,
        header_row_count=1,
        representation="markdown",
        cells=cells,
    )

    assert strict_table_classification(stale) == ("html", ["sparse_section"])


def test_strict_classification_preserves_recorded_non_grid_evidence() -> None:
    table = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation="html",
        classification_reasons=["ambiguous_continuation"],
        cells=[
            cell(0, 0, "Name", role="header"),
            cell(0, 1, "Value", role="header"),
            cell(1, 0, "A", role="row_header"),
            cell(1, 1, "1"),
        ],
    )

    assert strict_table_classification(table) == ("html", ["ambiguous_continuation"])


def test_strict_classification_does_not_treat_unknown_stale_reasons_as_policy() -> None:
    table = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation="html",
        classification_reasons=["legacy_markdown_override"],
        cells=[
            cell(0, 0, "Name", role="header"),
            cell(0, 1, "Value", role="header"),
            cell(1, 0, "A", role="row_header"),
            cell(1, 1, "1"),
        ],
    )

    assert strict_table_classification(table) == ("markdown", [])


def test_markdown_escapes_html_metacharacters_and_all_line_endings() -> None:
    cells = [
        cell(0, 0, "<Name>", role="header"),
        cell(0, 1, "Value & unit", role="header"),
        cell(1, 0, "A\\B|C"),
        cell(1, 1, "one\r\ntwo\rthree\nfour"),
    ]
    table = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation="markdown",
        cells=cells,
    )

    assert render_table(table) == (
        "| &lt;Name&gt; | Value &amp; unit |\n| --- | --- |\n| A\\\\B\\|C | one<br>two<br>three<br>four |"
    )


def test_html_escapes_text_preserves_empty_and_multiline_cells_and_has_no_css() -> None:
    cells = [
        cell(0, 0, "Group <all>", role="header", colspan=2),
        cell(1, 0, "", role="row_header"),
        cell(1, 1, "A & B\r\nnext"),
    ]
    table = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation="html",
        classification_reasons=["spanning_cells"],
        cells=cells,
    )

    rendered = render_table(table)
    assert rendered == (
        '<table><thead><tr><th scope="colgroup" colspan="2">Group &lt;all&gt;</th></tr></thead>'
        '<tbody><tr><th scope="row"></th><td>A &amp; B<br>next</td></tr></tbody></table>'
    )
    assert "style=" not in rendered and "class=" not in rendered


def test_rendering_is_deterministic_for_shuffled_cell_input() -> None:
    cells = [
        cell(0, 0, "A", role="header"),
        cell(0, 1, "B", role="header"),
        cell(1, 0, "1", role="row_header"),
        cell(1, 1, "2"),
    ]
    expected = render_table(
        TableStructure(
            row_count=2,
            column_count=2,
            header_row_count=1,
            representation="markdown",
            cells=cells,
        )
    )
    baseline_diagnostics = []
    assert classify_table(
        TableFeatures(row_count=2, column_count=2, header_row_count=1, cells=cells),
        diagnostics=baseline_diagnostics,
    ) == ("markdown", [])

    for ordering in (list(reversed(cells)), [cells[2], cells[0], cells[3], cells[1]]):
        table = TableStructure(
            row_count=2,
            column_count=2,
            header_row_count=1,
            representation="markdown",
            cells=ordering,
        )
        diagnostics = []
        assert render_table(table) == expected
        assert classify_table(
            TableFeatures(row_count=2, column_count=2, header_row_count=1, cells=ordering),
            diagnostics=diagnostics,
        ) == ("markdown", [])
        assert diagnostics == baseline_diagnostics
    assert baseline_diagnostics[0].stage == TableDiagnosticStage.RENDERING_CLASSIFICATION
    assert baseline_diagnostics[0].outcome == TableDiagnosticOutcome.ACCEPTED


def test_malformed_header_row_role_fails_schema_validation() -> None:
    with pytest.raises(ValueError, match="header role"):
        TableStructure(
            row_count=1,
            column_count=1,
            header_row_count=1,
            representation="markdown",
            cells=[cell(0, 0, "not a header")],
        )


def test_renderer_revalidates_models_created_without_schema_validation() -> None:
    malformed = TableStructure.model_construct(
        row_count=1,
        column_count=1,
        header_row_count=1,
        representation="markdown",
        classification_reasons=[],
        cells=[cell(0, 0, "not a header")],
    )

    with pytest.raises(ValueError, match="header role"):
        render_table(malformed)


@pytest.mark.parametrize(
    "malformed_cell",
    [
        TableCell.model_construct(
            row_index=-1,
            column_index=0,
            rowspan=1,
            colspan=1,
            role="header",
            text="A",
            fragments=[fragment()],
        ),
        TableCell.model_construct(
            row_index=0,
            column_index=0,
            rowspan=1,
            colspan=1,
            role="header",
            text=None,
            fragments=[fragment()],
        ),
    ],
)
def test_renderer_rejects_malformed_constructed_cells(malformed_cell: TableCell) -> None:
    malformed = TableStructure.model_construct(
        row_count=1,
        column_count=1,
        header_row_count=1,
        representation="markdown",
        classification_reasons=[],
        cells=[malformed_cell],
    )

    with pytest.raises(ValueError):
        render_table(malformed)


def test_incomplete_html_grid_renders_placeholders_without_duplicating_spans() -> None:
    table = TableStructure(
        row_count=3,
        column_count=3,
        header_row_count=1,
        representation="html",
        classification_reasons=["spanning_cells", "incomplete_grid"],
        cells=[
            cell(0, 0, "A", role="header"),
            cell(0, 2, "C", role="header"),
            cell(1, 0, "Rows", role="row_header", rowspan=2),
            cell(1, 2, "X"),
            cell(2, 1, "Y"),
        ],
    )

    assert render_table(table) == (
        '<table><thead><tr><th scope="col">A</th><th scope="col"></th><th scope="col">C</th></tr>'
        '</thead><tbody><tr><th scope="rowgroup" rowspan="2">Rows</th><td></td><td>X</td></tr>'
        "<tr><td>Y</td><td></td></tr></tbody></table>"
    )
