from __future__ import annotations

import re
from html import escape
from typing import Literal, get_args

from pydantic import Field

from app.pdf2md.schema import SchemaModel, TableCell, TableStructure
from app.pdf2md.table_provenance import TableDiagnostic, TableDiagnosticOutcome, TableDiagnosticStage

RENDERER_SEMANTICS_VERSION = "1.3.0"

HtmlReason = Literal[
    "no_header",
    "multiple_header_rows",
    "spanning_cells",
    "incomplete_grid",
    "nested_content",
    "sparse_section",
    "multi_paragraph_cell",
    "ambiguous_continuation",
    "geometry_ambiguous",
]

_POLICY_HTML_REASONS = frozenset(get_args(HtmlReason))


class TableFeatures(SchemaModel):
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    header_row_count: int = Field(ge=0)
    cells: list[TableCell] = Field(min_length=1)
    has_nested_content: bool = False
    has_multi_paragraph_cells: bool = False
    has_ambiguous_continuation: bool = False


def classify_table(
    features: TableFeatures,
    *,
    diagnostics: list[TableDiagnostic] | None = None,
) -> tuple[Literal["markdown", "html"], list[str]]:
    features = TableFeatures.model_validate(features.model_dump(), strict=True)
    occupied = _validate_layout(
        features.row_count,
        features.column_count,
        features.header_row_count,
        features.cells,
    )
    reasons: list[str] = []

    if features.header_row_count == 0:
        reasons.append("no_header")
    elif features.header_row_count > 1:
        reasons.append("multiple_header_rows")
    if any(cell.rowspan > 1 or cell.colspan > 1 for cell in features.cells):
        reasons.append("spanning_cells")

    expected = {(row, column) for row in range(features.row_count) for column in range(features.column_count)}
    if occupied != expected:
        reasons.append("incomplete_grid")
    if features.has_nested_content:
        reasons.append("nested_content")
    if _has_sparse_body_section(features):
        reasons.append("sparse_section")
    if features.has_multi_paragraph_cells:
        reasons.append("multi_paragraph_cell")
    if features.has_ambiguous_continuation:
        reasons.append("ambiguous_continuation")

    representation: Literal["markdown", "html"] = "html" if reasons else "markdown"
    if diagnostics is not None:
        diagnostics.append(
            TableDiagnostic(
                stage=TableDiagnosticStage.RENDERING_CLASSIFICATION,
                outcome=TableDiagnosticOutcome.ACCEPTED,
                reason=",".join(reasons) or representation,
                metrics=(
                    ("column_count", features.column_count),
                    ("header_row_count", features.header_row_count),
                    ("representation", representation),
                    ("row_count", features.row_count),
                ),
            )
        )
    return (representation, reasons) if reasons else (representation, [])


def strict_table_classification(table: TableStructure) -> tuple[Literal["markdown", "html"], list[str]]:
    """Recompute rendering eligibility from structure and recorded semantic evidence."""
    table = _validated_table(table)
    recorded_reasons = list(
        dict.fromkeys(reason for reason in table.classification_reasons if reason in _POLICY_HTML_REASONS)
    )
    representation, inferred_reasons = classify_table(
        TableFeatures(
            row_count=table.row_count,
            column_count=table.column_count,
            header_row_count=table.header_row_count,
            cells=table.cells,
            has_nested_content="nested_content" in recorded_reasons,
            has_multi_paragraph_cells="multi_paragraph_cell" in recorded_reasons,
            has_ambiguous_continuation="ambiguous_continuation" in recorded_reasons,
        )
    )
    reasons = list(dict.fromkeys([*inferred_reasons, *recorded_reasons]))
    return ("html", reasons) if reasons else (representation, [])


def _has_sparse_body_section(features: TableFeatures) -> bool:
    populated_by_row: dict[int, int] = {}
    for cell in features.cells:
        if cell.row_index >= features.header_row_count and cell.text.strip():
            populated_by_row[cell.row_index] = populated_by_row.get(cell.row_index, 0) + 1
    return features.column_count > 2 and any(count == 1 for count in populated_by_row.values())


def render_table(table: TableStructure) -> str:
    table = _validated_table(table)
    if table.representation == "markdown":
        return render_markdown_table(table)
    return render_html_table(table)


def render_markdown_table(table: TableStructure) -> str:
    table = _validated_table(table)
    if table.representation != "markdown":
        raise ValueError("render_markdown_table requires a Markdown-representable table")
    occupied = _validate_layout(
        table.row_count,
        table.column_count,
        table.header_row_count,
        table.cells,
    )
    expected = {(row, column) for row in range(table.row_count) for column in range(table.column_count)}
    if table.header_row_count != 1:
        raise ValueError("Markdown tables require exactly one header row")
    if any(cell.rowspan != 1 or cell.colspan != 1 for cell in table.cells):
        raise ValueError("Markdown tables cannot contain spanning cells")
    if occupied != expected:
        raise ValueError("Markdown tables require a complete rectangular grid")
    if _has_sparse_body_section(
        TableFeatures(
            row_count=table.row_count,
            column_count=table.column_count,
            header_row_count=table.header_row_count,
            cells=table.cells,
        )
    ):
        raise ValueError("Markdown tables cannot contain nested section rows")
    if table.classification_reasons:
        raise ValueError("Markdown tables cannot have HTML classification reasons")

    cells = {(cell.row_index, cell.column_index): cell for cell in table.cells}
    rows = [
        [_escape_markdown_cell(cells[(row, column)].text) for column in range(table.column_count)]
        for row in range(table.row_count)
    ]
    header = _markdown_row(rows[0])
    separator = _markdown_row(["---"] * table.column_count)
    body = [_markdown_row(row) for row in rows[1:]]
    return "\n".join([header, separator, *body])


def render_html_table(table: TableStructure) -> str:
    table = _validated_table(table)
    if table.representation != "html":
        raise ValueError("render_html_table requires an HTML-classified table")
    _validate_layout(
        table.row_count,
        table.column_count,
        table.header_row_count,
        table.cells,
    )

    starts = {(cell.row_index, cell.column_index): cell for cell in table.cells}
    covered_nonstarts = {
        (row, column)
        for cell in table.cells
        for row in range(cell.row_index, cell.row_index + cell.rowspan)
        for column in range(cell.column_index, cell.column_index + cell.colspan)
        if (row, column) != (cell.row_index, cell.column_index)
    }
    rows = [
        _html_grid_row(
            row,
            table.column_count,
            starts=starts,
            covered_nonstarts=covered_nonstarts,
            in_header=row < table.header_row_count,
        )
        for row in range(table.row_count)
    ]
    header_rows = "".join(rows[: table.header_row_count])
    body_rows = "".join(rows[table.header_row_count :])
    return f"<table><thead>{header_rows}</thead><tbody>{body_rows}</tbody></table>"


def _markdown_row(values: list[str]) -> str:
    return f"| {' | '.join(values)} |"


def _escape_markdown_cell(value: str) -> str:
    return "<br>".join(
        escape(line.replace("\\", "\\\\").replace("|", "\\|"), quote=False)
        for line in re.split(r"\r\n|\r|\n", value)
    )


def _html_grid_row(
    row: int,
    column_count: int,
    *,
    starts: dict[tuple[int, int], TableCell],
    covered_nonstarts: set[tuple[int, int]],
    in_header: bool,
) -> str:
    rendered: list[str] = []
    for column in range(column_count):
        position = (row, column)
        cell = starts.get(position)
        if cell is not None:
            rendered.append(_html_cell(cell, in_header=in_header))
        elif position not in covered_nonstarts:
            rendered.append('<th scope="col"></th>' if in_header else "<td></td>")
    return f"<tr>{''.join(rendered)}</tr>"


def _html_cell(cell: TableCell, *, in_header: bool) -> str:
    tag = "th" if cell.role in {"header", "row_header"} else "td"
    attributes: list[str] = []
    if cell.role == "header":
        scope = "colgroup" if in_header and cell.colspan > 1 else "col" if in_header else "rowgroup"
        attributes.append(f' scope="{scope}"')
    elif cell.role == "row_header":
        scope = "rowgroup" if cell.rowspan > 1 else "row"
        attributes.append(f' scope="{scope}"')
    if cell.rowspan > 1:
        attributes.append(f' rowspan="{cell.rowspan}"')
    if cell.colspan > 1:
        attributes.append(f' colspan="{cell.colspan}"')
    content = "<br>".join(escape(line, quote=False) for line in re.split(r"\r\n|\r|\n", cell.text))
    return f"<{tag}{''.join(attributes)}>{content}</{tag}>"


def _validated_table(table: TableStructure) -> TableStructure:
    return TableStructure.model_validate(table.model_dump(), strict=True)


def _validate_layout(
    row_count: int,
    column_count: int,
    header_row_count: int,
    cells: list[TableCell],
) -> set[tuple[int, int]]:
    if header_row_count > row_count:
        raise ValueError("header_row_count cannot exceed row_count")

    occupied: set[tuple[int, int]] = set()
    for cell in cells:
        if cell.row_index + cell.rowspan > row_count:
            raise ValueError("cell rowspan exceeds table row_count")
        if cell.column_index + cell.colspan > column_count:
            raise ValueError("cell colspan exceeds table column_count")
        if cell.row_index < header_row_count and cell.role != "header":
            raise ValueError("cells starting in header rows require the header role")
        positions = {
            (row, column)
            for row in range(cell.row_index, cell.row_index + cell.rowspan)
            for column in range(cell.column_index, cell.column_index + cell.colspan)
        }
        if occupied.intersection(positions):
            raise ValueError("table cells overlap")
        occupied.update(positions)
    return occupied
