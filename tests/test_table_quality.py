from app.pdf2md.schema import BoundingBox, PageFragment, TableCell, TableStructure
from app.pdf2md.table_provenance import TableDiagnosticOutcome, TableDiagnosticStage
from app.pdf2md.table_quality import validate_table_candidate


def table_with_populated_cells(rows: int, columns: int, populated: int) -> TableStructure:
    cells = [
        TableCell(
            row_index=index // columns,
            column_index=index % columns,
            role="body",
            text=f"v{index}" if index < populated else "",
            fragments=[
                PageFragment(
                    page_number=1,
                    bbox=BoundingBox(
                        x0=float(index % columns),
                        y0=float(index // columns),
                        x1=float(index % columns + 1),
                        y1=float(index // columns + 1),
                    ),
                )
            ],
        )
        for index in range(max(populated, 1))
    ]
    return TableStructure(
        row_count=rows,
        column_count=columns,
        header_row_count=0,
        representation="html",
        classification_reasons=["no_header", "incomplete_grid"],
        cells=cells,
    )


def test_rejects_empty_table() -> None:
    decision = validate_table_candidate(table_with_populated_cells(4, 4, 0))

    assert not decision.accepted
    assert decision.rejection_reason == "empty"
    assert decision.diagnostic.stage == TableDiagnosticStage.TABLE_QUALITY
    assert decision.diagnostic.outcome == TableDiagnosticOutcome.REJECTED
    assert decision.diagnostic.reason == "empty"


def test_rejects_sparse_diagram_grid() -> None:
    decision = validate_table_candidate(table_with_populated_cells(3, 3, 2))

    assert not decision.accepted
    assert decision.rejection_reason == "diagram_sparse"


def test_preserves_small_one_row_table() -> None:
    decision = validate_table_candidate(table_with_populated_cells(1, 3, 3))

    assert decision.accepted


def test_rejects_table_of_contents_with_context_heading() -> None:
    table = table_with_populated_cells(6, 2, 12)
    values = [
        "Introduction",
        "1",
        "Methods",
        "3",
        "Results",
        "7",
        "Discussion",
        "12",
        "Appendix",
        "18",
        "References",
        "21",
    ]
    cells = [cell.model_copy(update={"text": text}) for cell, text in zip(table.cells, values, strict=True)]

    decision = validate_table_candidate(
        table.model_copy(update={"cells": cells}),
        nearby_text="TABLE OF CONTENTS",
    )

    assert not decision.accepted
    assert decision.rejection_reason == "table_of_contents"


def test_preserves_title_and_integer_table_without_contents_heading() -> None:
    table = table_with_populated_cells(6, 2, 12)
    values = [
        "Product A",
        "1",
        "Product B",
        "3",
        "Product C",
        "7",
        "Product D",
        "12",
        "Product E",
        "18",
        "Product F",
        "21",
    ]
    cells = [cell.model_copy(update={"text": text}) for cell, text in zip(table.cells, values, strict=True)]

    decision = validate_table_candidate(table.model_copy(update={"cells": cells}))

    assert decision.accepted


def test_rejects_pathological_large_sparse_grid() -> None:
    decision = validate_table_candidate(table_with_populated_cells(20, 20, 19))

    assert not decision.accepted
    assert decision.rejection_reason == "pathological_sparse"


def test_preserves_sparse_grid_with_substantial_text() -> None:
    table = table_with_populated_cells(39, 7, 8)
    cells = list(table.cells)
    cells[0] = cells[0].model_copy(update={"text": "substantial table text " * 100})

    decision = validate_table_candidate(table.model_copy(update={"cells": cells}))

    assert decision.accepted


def test_preserves_sparse_density_boundary() -> None:
    decision = validate_table_candidate(table_with_populated_cells(20, 20, 20))

    assert decision.accepted


def test_preserves_sparse_grid_below_size_boundary() -> None:
    decision = validate_table_candidate(table_with_populated_cells(9, 11, 3))

    assert decision.accepted


def test_preserves_low_density_known_positive_envelope() -> None:
    decision = validate_table_candidate(table_with_populated_cells(47, 10, 75))

    assert decision.accepted
