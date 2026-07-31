import pytest

from app.pdf2md.continuations import (
    _table_caption_evidence,  # pyright: ignore[reportPrivateUsage]
    mark_selection_boundary_tables,
    merge_table_continuations,
)
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    PageFragment,
    ParagraphStructure,
    TableCell,
    TableStructure,
)
from app.pdf2md.tables import render_table


def table_element(page_number: int, y0: float, y1: float, order: int, *, x0: float = 0) -> DocumentElement:
    table_bbox = BoundingBox(x0=x0, y0=y0, x1=x0 + 100, y1=y1)
    row_height = (y1 - y0) / 2
    fragments = [
        PageFragment(
            page_number=page_number,
            page_width=100,
            page_height=800,
            bbox=BoundingBox(
                x0=x0 + column * 50,
                y0=y0 + row * row_height,
                x1=x0 + (column + 1) * 50,
                y1=y0 + (row + 1) * row_height,
            ),
        )
        for row in range(2)
        for column in range(2)
    ]
    values = (("Name", "Value"), (f"row-{page_number}", str(page_number)))
    cells = [
        TableCell(
            row_index=row,
            column_index=column,
            role="header" if row == 0 else "body",
            text=values[row][column],
            fragments=[fragments[row * 2 + column]],
        )
        for row in range(2)
        for column in range(2)
    ]
    table = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation="markdown",
        cells=cells,
    )
    return DocumentElement(
        document_id="doc",
        element_id=f"table-{page_number}",
        order=order,
        element_type="table",
        content=render_table(table),
        format="markdown",
        fragments=[
            PageFragment(
                page_number=page_number,
                page_width=100,
                page_height=800,
                bbox=table_bbox,
            )
        ],
        structure=ElementStructure(table=table),
        annotation=AnnotationMetadata(
            stage="candidate",
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        ),
    )


@pytest.mark.parametrize(
    ("page_number", "y0", "y1", "selected_pages"),
    [
        (291, 80, 140, [291, 292, 293]),
        (293, 620, 770, [291, 292, 293]),
        (5, 80, 140, [1, 2, 3, 5]),
    ],
)
def test_selection_boundary_table_is_marked_ambiguous(
    page_number: int, y0: float, y1: float, selected_pages: list[int]
) -> None:
    source = table_element(page_number, y0, y1, 0)

    result = mark_selection_boundary_tables([source], selected_pages, document_page_count=400)
    table = result[0].structure.table

    assert table is not None
    assert table.representation == "html"
    assert table.classification_reasons == ["ambiguous_continuation"]


@pytest.mark.parametrize(
    ("page_number", "y0", "y1", "selected_pages", "page_count"),
    [
        (291, 300, 500, [291, 292, 293], 400),
        (291, 80, 140, [290, 291, 292], 400),
        (293, 620, 770, [292, 293, 294], 400),
        (1, 80, 140, [1], 1),
    ],
)
def test_selection_boundary_marking_preserves_non_ambiguous_table(
    page_number: int,
    y0: float,
    y1: float,
    selected_pages: list[int],
    page_count: int,
) -> None:
    source = table_element(page_number, y0, y1, 0)

    result = mark_selection_boundary_tables([source], selected_pages, page_count)

    assert result == [source]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(("x_factor", "y_factor"), [(1.0, 1.0), (1.7, 0.6)])
def test_substantive_selected_page_context_proves_standalone_boundary_tables(
    rotation: int,
    x_factor: float,
    y_factor: float,
) -> None:
    transform = lambda element: _quarter_turn(  # noqa: E731
        _affine_scale(element, x_factor, y_factor), rotation
    )
    top_table = transform(table_element(5, 180, 240, 0))
    bottom_table = transform(table_element(5, 620, 770, 1))
    heading = transform(body_element(5, "Standalone table heading", 0, (0, 130, 100, 150)))
    trailing_note = transform(body_element(5, "Source: independently complete table.", 2, (0, 775, 100, 790)))

    result = mark_selection_boundary_tables(
        [top_table, bottom_table],
        [5],
        document_page_count=10,
        context_elements=[heading, trailing_note],
    )

    assert result == [top_table, bottom_table]


def test_top_margin_caption_proves_standalone_selected_boundary_table() -> None:
    table = table_element(5, 80, 240, 0)
    caption = caption_element(5, "Table 7: Standalone results", 0)

    result = mark_selection_boundary_tables(
        [table],
        [5],
        document_page_count=10,
        context_elements=[caption],
    )

    assert result == [table]


def test_explicit_continued_caption_overrides_repeated_heading_context() -> None:
    table = table_element(5, 160, 550, 0)
    context = [
        caption_element(5, "Table 7: (continued)", 0),
        body_element(5, "Repeated table heading", 1, (0, 145, 100, 158)),
    ]

    result = mark_selection_boundary_tables(
        [table],
        [5],
        document_page_count=10,
        context_elements=context,
    )

    structure = result[0].structure.table
    assert structure is not None
    assert structure.representation == "html"
    assert "ambiguous_continuation" in structure.classification_reasons


def test_adjoining_top_margin_section_heading_proves_standalone_table() -> None:
    table = table_element(5, 72, 500, 0)
    heading = body_element(5, "Part VIII Statement of Revenue", 0, (0, 35, 100, 71.5))

    result = mark_selection_boundary_tables(
        [table],
        [5],
        document_page_count=10,
        context_elements=[heading],
    )

    assert result == [table]


def test_page_scoped_part_heading_proves_form_table_completion() -> None:
    table = table_element(5, 72, 700, 0)
    heading = body_element(5, "Part VIII Statement of Revenue", 0, (0, 35, 100, 71.5))

    result = mark_selection_boundary_tables(
        [table],
        [5],
        document_page_count=10,
        context_elements=[heading],
    )

    assert result == [table]


def test_compact_sparse_titled_table_is_standalone_near_page_bottom() -> None:
    source = table_element(5, 620, 700, 0)
    table = source.structure.table
    assert table is not None
    cells = [
        cell.model_copy(update={"text": "Awards and Acknowledgements"})
        if cell.row_index == 0 and cell.column_index == 0
        else cell.model_copy(update={"text": ""})
        if cell.row_index == 0
        else cell
        for cell in table.cells
    ]
    updated = table.model_copy(update={"cells": cells})
    source = source.model_copy(
        update={
            "structure": source.structure.model_copy(update={"table": updated}),
            "content": render_table(updated),
        }
    )

    result = mark_selection_boundary_tables([source], [5], document_page_count=10)

    assert result == [source]


def test_next_page_continuation_note_overrides_adjoining_completion_context() -> None:
    table = table_element(5, 300, 750, 0)
    context = [
        body_element(5, "Continued on next page", 1, (0, 752, 100, 765)),
        body_element(5, "Explanatory footer", 2, (0, 766, 100, 775)),
    ]

    result = mark_selection_boundary_tables(
        [table],
        [5],
        document_page_count=10,
        context_elements=context,
    )

    structure = result[0].structure.table
    assert structure is not None
    assert structure.representation == "html"
    assert "ambiguous_continuation" in structure.classification_reasons


def caption_element(page_number: int, content: str, order: int) -> DocumentElement:
    return DocumentElement(
        document_id="doc",
        element_id=f"caption-{page_number}-{order}",
        order=order,
        element_type="caption",
        content=content,
        format="text",
        fragments=[
            PageFragment(
                page_number=page_number,
                page_width=100,
                page_height=800,
                bbox=BoundingBox(x0=20, y0=5, x1=80, y1=15),
            )
        ],
        structure=ElementStructure(paragraph=ParagraphStructure(role="caption")),
        annotation=AnnotationMetadata(
            stage="candidate",
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        ),
    )


def body_element(
    page_number: int,
    content: str,
    order: int,
    bbox: tuple[float, float, float, float],
) -> DocumentElement:
    element = caption_element(page_number, content, order)
    return element.model_copy(
        update={
            "element_id": f"body-{page_number}-{order}",
            "element_type": "paragraph",
            "fragments": [
                element.fragments[0].model_copy(
                    update={"bbox": BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])}
                )
            ],
            "structure": ElementStructure(paragraph=ParagraphStructure(role="body")),
        }
    )


def _quarter_turn(element: DocumentElement, rotation: int) -> DocumentElement:
    def rotate_fragment(fragment: PageFragment) -> PageFragment:
        assert fragment.page_width is not None and fragment.page_height is not None
        width, height = fragment.page_width, fragment.page_height
        bbox = fragment.bbox
        match rotation:
            case 0:
                rotated = bbox
                page_width, page_height = width, height
            case 90:
                rotated = BoundingBox(x0=height - bbox.y1, y0=bbox.x0, x1=height - bbox.y0, y1=bbox.x1)
                page_width, page_height = height, width
            case 180:
                rotated = BoundingBox(
                    x0=width - bbox.x1,
                    y0=height - bbox.y1,
                    x1=width - bbox.x0,
                    y1=height - bbox.y0,
                )
                page_width, page_height = width, height
            case 270:
                rotated = BoundingBox(x0=bbox.y0, y0=width - bbox.x1, x1=bbox.y1, y1=width - bbox.x0)
                page_width, page_height = height, width
            case _:
                raise ValueError(f"unsupported test rotation: {rotation}")
        return fragment.model_copy(
            update={"page_width": page_width, "page_height": page_height, "bbox": rotated}
        )

    table = element.structure.table
    if table is not None:
        table = table.model_copy(
            update={
                "cells": [
                    cell.model_copy(
                        update={"fragments": [rotate_fragment(value) for value in cell.fragments]}
                    )
                    for cell in table.cells
                ]
            }
        )
    return element.model_copy(
        update={
            "fragments": [rotate_fragment(fragment) for fragment in element.fragments],
            "structure": element.structure.model_copy(update={"table": table}),
        }
    )


def _scale(element: DocumentElement, factor: float) -> DocumentElement:
    return _affine_scale(element, factor, factor)


def _affine_scale(element: DocumentElement, x_factor: float, y_factor: float) -> DocumentElement:
    def scale_fragment(fragment: PageFragment) -> PageFragment:
        assert fragment.page_width is not None and fragment.page_height is not None
        bbox = fragment.bbox
        return fragment.model_copy(
            update={
                "page_width": fragment.page_width * x_factor,
                "page_height": fragment.page_height * y_factor,
                "bbox": BoundingBox(
                    x0=bbox.x0 * x_factor,
                    y0=bbox.y0 * y_factor,
                    x1=bbox.x1 * x_factor,
                    y1=bbox.y1 * y_factor,
                ),
            }
        )

    table = element.structure.table
    scaled_table = (
        table.model_copy(
            update={
                "cells": [
                    cell.model_copy(update={"fragments": [scale_fragment(value) for value in cell.fragments]})
                    for cell in table.cells
                ]
            }
        )
        if table is not None
        else None
    )
    return element.model_copy(
        update={
            "fragments": [scale_fragment(fragment) for fragment in element.fragments],
            "structure": element.structure.model_copy(update={"table": scaled_table}),
        }
    )


def _append_uniform_foot(element: DocumentElement, label: str = "Foot") -> DocumentElement:
    table = element.structure.table
    assert table is not None
    page = element.fragments[0].page_number
    cells = [
        *table.cells,
        *[
            TableCell(
                row_index=table.row_count,
                column_index=column,
                role="body",
                text=label,
                fragments=[
                    PageFragment(
                        page_number=page,
                        page_width=100,
                        page_height=800,
                        bbox=BoundingBox(
                            x0=column * 50,
                            y0=element.fragments[0].bbox.y1 - 10,
                            x1=(column + 1) * 50,
                            y1=element.fragments[0].bbox.y1,
                        ),
                    )
                ],
            )
            for column in range(2)
        ],
    ]
    updated = table.model_copy(update={"row_count": table.row_count + 1, "cells": cells})
    return element.model_copy(
        update={
            "content": render_table(updated),
            "structure": element.structure.model_copy(update={"table": updated}),
        }
    )


def _replace_body_row(element: DocumentElement, values: tuple[str, str]) -> DocumentElement:
    table = element.structure.table
    assert table is not None
    cells = [
        cell.model_copy(update={"text": values[cell.column_index]}) if cell.row_index == 1 else cell
        for cell in table.cells
    ]
    updated = table.model_copy(update={"cells": cells})
    return element.model_copy(
        update={
            "content": render_table(updated),
            "structure": element.structure.model_copy(update={"table": updated}),
        }
    )


def _replace_header(element: DocumentElement, values: tuple[str, str]) -> DocumentElement:
    table = element.structure.table
    assert table is not None
    cells = [
        cell.model_copy(update={"text": values[cell.column_index]}) if cell.row_index == 0 else cell
        for cell in table.cells
    ]
    updated_table = table.model_copy(update={"cells": cells})
    return element.model_copy(
        update={
            "content": render_table(updated_table),
            "structure": element.structure.model_copy(update={"table": updated_table}),
        }
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_merge_adjacency_and_columns_are_scale_and_quarter_turn_invariant(
    rotation: int, scale: float
) -> None:
    first = _quarter_turn(_scale(table_element(page_number=1, y0=700, y1=790, order=0), scale), rotation)
    second = _quarter_turn(_scale(table_element(page_number=2, y0=10, y1=100, order=1), scale), rotation)

    result = merge_table_continuations([first, second])

    assert len(result) == 1
    table = result[0].structure.table
    assert table is not None
    assert [cell.text for cell in table.cells if cell.column_index == 0] == [
        "Name",
        "row-1",
        "row-2",
    ]


def test_merge_repeated_header_across_page_boundary() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = table_element(page_number=2, y0=10, y1=100, order=1)

    result = merge_table_continuations([first, second])

    assert len(result) == 1
    table = result[0].structure.table
    assert table is not None
    assert table.row_count == 3
    assert [cell.text for cell in table.cells if cell.column_index == 0] == [
        "Name",
        "row-1",
        "row-2",
    ]
    assert [fragment.page_number for fragment in result[0].fragments] == [1, 2]
    assert result[0].structure.linked_element_ids == ["table-2"]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_explicit_matching_continued_caption_allows_distinct_first_page_header_under_affine_transform(
    rotation: int,
) -> None:
    first = _quarter_turn(
        _affine_scale(table_element(page_number=1, y0=700, y1=790, order=0), 1.7, 0.6),
        rotation,
    )
    second = _quarter_turn(
        _affine_scale(
            _replace_header(table_element(page_number=2, y0=10, y1=100, order=1), ("First", "Second")),
            1.7,
            0.6,
        ),
        rotation,
    )
    captions = [
        caption_element(1, "Table 7: Initial caption", 0),
        caption_element(2, "Table 7: (continued)", 1),
    ]

    result = merge_table_continuations([first, second], captions=captions)

    assert len(result) == 1
    table = result[0].structure.table
    assert table is not None
    assert table.row_count == 3
    assert [cell.text for cell in table.cells if cell.column_index == 0] == [
        "Name",
        "row-1",
        "row-2",
    ]


@pytest.mark.parametrize(
    ("captions", "expected_count"),
    [
        ([], 2),
        ([caption_element(1, "Table 7: Initial", 0), caption_element(2, "Table 8: (continued)", 1)], 2),
        ([caption_element(1, "Table 7: Initial", 0), caption_element(2, "Table 7: New part", 1)], 2),
        (
            [
                caption_element(1, "Table 7: Initial", 0),
                caption_element(2, "Table 7: (continued)", 1),
                caption_element(2, "Table 8: (continued)", 2),
            ],
            2,
        ),
    ],
)
def test_distinct_headers_fail_closed_without_unique_matching_continued_caption(
    captions: list[DocumentElement], expected_count: int
) -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = _replace_header(
        table_element(page_number=2, y0=10, y1=100, order=1),
        ("First", "Second"),
    )

    assert len(merge_table_continuations([first, second], captions=captions)) == expected_count


@pytest.mark.parametrize(
    "current_caption",
    ["Table 8: (continued)", "Table 7: A separate table"],
)
def test_explicit_caption_identity_vetoes_geometrically_plausible_repeated_header_merge(
    current_caption: str,
) -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = table_element(page_number=2, y0=10, y1=100, order=1)
    captions = [
        caption_element(1, "Table 7: Baseline", 0),
        caption_element(2, current_caption, 1),
    ]

    assert len(merge_table_continuations([first, second], captions=captions)) == 2


def test_caption_identifier_and_continued_marker_normalization_is_exact() -> None:
    assert _table_caption_evidence("TABLE 30. FOOD NET TRADE (CONTINUED)") == ("30", True)
    assert _table_caption_evidence("Table A.1 — continued") == ("a.1", True)
    assert _table_caption_evidence("Table (IV): Baseline") == ("iv", False)

    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = _replace_header(
        table_element(page_number=2, y0=10, y1=100, order=1),
        ("First", "Second"),
    )
    captions = [
        caption_element(1, "Ｔａｂｌｅ　Ａ – １： Baseline", 0),
        caption_element(2, "table a-1: [ CONTINUED ]", 1),
    ]

    assert len(merge_table_continuations([first, second], captions=captions)) == 1


@pytest.mark.parametrize(
    "current_caption",
    [
        "Table 7: continued growth",
        "Table 7: not continued",
        "Table 7: not (continued)",
        "Table 7: a continued discussion",
        "Table7: (continued)",
        "Table of contents (continued)",
    ],
)
def test_non_marker_uses_of_continued_do_not_override_distinct_headers(
    current_caption: str,
) -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = _replace_header(
        table_element(page_number=2, y0=10, y1=100, order=1),
        ("First", "Second"),
    )
    captions = [
        caption_element(1, "Table 7: Baseline", 0),
        caption_element(2, current_caption, 1),
    ]

    assert len(merge_table_continuations([first, second], captions=captions)) == 2


@pytest.mark.parametrize(
    "intervening",
    [
        body_element(1, "Unrelated trailing prose", 2, (0, 791, 100, 799)),
        body_element(2, "Unrelated page introduction", 2, (0, 1, 100, 4)),
        body_element(2, "Unrelated narrow prose", 2, (0, 20, 40, 30)),
    ],
)
def test_intervening_content_blocks_caption_override(intervening: DocumentElement) -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = _replace_header(
        table_element(page_number=2, y0=40, y1=130, order=1),
        ("First", "Second"),
    )
    context = [
        caption_element(1, "Table 7: Baseline", 0),
        caption_element(2, "Table 7: (continued)", 1),
        intervening,
    ]

    assert len(merge_table_continuations([first, second], context_elements=context)) == 2


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(("x_factor", "y_factor"), [(0.5, 2.0), (1.7, 0.6)])
def test_caption_context_association_is_affine_and_quarter_turn_invariant(
    rotation: int,
    x_factor: float,
    y_factor: float,
) -> None:
    transform = lambda element: _quarter_turn(  # noqa: E731
        _affine_scale(element, x_factor, y_factor), rotation
    )
    first = transform(table_element(page_number=1, y0=700, y1=790, order=0))
    second = transform(
        _replace_header(
            table_element(page_number=2, y0=40, y1=130, order=1),
            ("First", "Second"),
        )
    )
    context = [
        transform(caption_element(1, "Table 7: Baseline", 0)),
        transform(caption_element(2, "Table 7: (continued)", 1)),
        transform(body_element(2, "Repeated table heading missed by detection", 2, (0, 16, 100, 35))),
    ]

    assert len(merge_table_continuations([first, second], context_elements=context)) == 1


def test_same_number_caption_after_previous_table_is_not_associated() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = _replace_header(
        table_element(page_number=2, y0=40, y1=130, order=1),
        ("First", "Second"),
    )
    previous_caption = caption_element(1, "Table 7: Unrelated", 0).model_copy(
        update={
            "fragments": [
                caption_element(1, "unused", 3)
                .fragments[0]
                .model_copy(update={"bbox": BoundingBox(x0=20, y0=792, x1=80, y1=799)})
            ]
        }
    )
    context = [previous_caption, caption_element(2, "Table 7: (continued)", 1)]

    assert len(merge_table_continuations([first, second], context_elements=context)) == 2


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(("x_factor", "y_factor"), [(1.0, 1.0), (1.7, 0.6)])
def test_rp2040_distinct_caption_after_top_page_continuation_is_affine_and_quarter_turn_invariant(
    rotation: int,
    x_factor: float,
    y_factor: float,
) -> None:
    transform = lambda element: _quarter_turn(  # noqa: E731
        _affine_scale(element, x_factor, y_factor), rotation
    )
    first = transform(table_element(page_number=292, y0=700, y1=790, order=0))
    continuation = transform(table_element(page_number=293, y0=10, y1=530, order=1))
    following = transform(table_element(page_number=293, y0=620, y1=760, order=2))
    previous_caption = caption_element(292, "Table 329. INTR Register", 0).model_copy(
        update={
            "fragments": [
                caption_element(292, "unused", 3)
                .fragments[0]
                .model_copy(update={"bbox": BoundingBox(x0=20, y0=720, x1=80, y1=730)})
            ]
        }
    )
    following_caption = caption_element(293, "Table 330. PROC0_INTE Register", 1).model_copy(
        update={
            "fragments": [
                caption_element(293, "unused", 3)
                .fragments[0]
                .model_copy(update={"bbox": BoundingBox(x0=20, y0=622, x1=80, y1=632)})
            ]
        }
    )

    result = merge_table_continuations(
        [first, continuation, following],
        context_elements=[transform(previous_caption), transform(following_caption)],
    )

    assert len(result) == 2
    merged_table = result[0].structure.table
    following_table = result[1].structure.table
    assert merged_table is not None and following_table is not None
    assert merged_table.row_count == 3
    assert [fragment.page_number for fragment in result[0].fragments] == [292, 293]
    assert [fragment.page_number for fragment in result[1].fragments] == [293]


def test_distinct_caption_before_current_table_retains_identity_veto() -> None:
    first = table_element(page_number=292, y0=700, y1=790, order=0)
    current = table_element(page_number=293, y0=40, y1=530, order=1)
    context = [
        caption_element(292, "Table 329. INTR Register", 0),
        caption_element(293, "Table 330. PROC0_INTE Register", 1),
    ]

    assert len(merge_table_continuations([first, current], context_elements=context)) == 2


def test_intervening_content_blocks_merge_when_distinct_caption_follows_candidate() -> None:
    first = table_element(page_number=292, y0=700, y1=790, order=0)
    current = table_element(page_number=293, y0=40, y1=530, order=1)
    following_caption = caption_element(293, "Table 330. PROC0_INTE Register", 1).model_copy(
        update={
            "fragments": [
                caption_element(293, "unused", 3)
                .fragments[0]
                .model_copy(update={"bbox": BoundingBox(x0=20, y0=620, x1=80, y1=630)})
            ]
        }
    )
    context = [
        caption_element(292, "Table 329. INTR Register", 0),
        following_caption,
        body_element(293, "Unrelated page introduction", 2, (0, 16, 100, 30)),
    ]

    assert len(merge_table_continuations([first, current], context_elements=context)) == 2


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_distinct_identifiers_veto_composite_table_merge_at_every_quarter_turn(rotation: int) -> None:
    transform = lambda element: _quarter_turn(element, rotation)  # noqa: E731
    first = transform(table_element(page_number=1, y0=700, y1=790, order=0))
    second = transform(table_element(page_number=2, y0=10, y1=100, order=1))
    embedded_caption = caption_element(1, "Ｔａｂｌｅ 329: First register", 0).model_copy(
        update={
            "fragments": [
                caption_element(1, "unused", 3)
                .fragments[0]
                .model_copy(update={"bbox": BoundingBox(x0=20, y0=750, x1=80, y1=760)})
            ]
        }
    )
    context = [
        transform(embedded_caption),
        transform(caption_element(2, "Table 330: Next register", 1)),
    ]

    assert len(merge_table_continuations([first, second], context_elements=context)) == 2


@pytest.mark.parametrize(
    ("previous_y1", "current_y0", "expected_count"),
    [
        (600, 200, 1),
        (599.99, 200, 2),
        (600, 200.01, 2),
    ],
)
def test_explicit_caption_does_not_relax_page_boundary_geometry(
    previous_y1: float, current_y0: float, expected_count: int
) -> None:
    first = table_element(page_number=1, y0=510, y1=previous_y1, order=0)
    second = _replace_header(
        table_element(page_number=2, y0=current_y0, y1=current_y0 + 90, order=1),
        ("First", "Second"),
    )
    captions = [
        caption_element(1, "Table A.1: Initial", 0),
        caption_element(2, "Table A.1: continued", 1),
    ]

    assert len(merge_table_continuations([first, second], captions=captions)) == expected_count


def test_merge_rule_aligned_sparse_first_body_row_as_split_cross_page_row() -> None:
    def four_column_element(
        page_number: int,
        y0: float,
        y1: float,
        values: tuple[tuple[str, str, str, str], tuple[str, str, str, str]],
    ) -> DocumentElement:
        row_height = (y1 - y0) / 2
        cells = [
            TableCell(
                row_index=row,
                column_index=column,
                role="header" if row == 0 else "body",
                text=values[row][column],
                fragments=[
                    PageFragment(
                        page_number=page_number,
                        page_width=400,
                        page_height=800,
                        bbox=BoundingBox(
                            x0=column * 100,
                            y0=y0 + row * row_height,
                            x1=(column + 1) * 100,
                            y1=y0 + (row + 1) * row_height,
                        ),
                    )
                ],
            )
            for row in range(2)
            for column in range(4)
        ]
        table = TableStructure(
            row_count=2,
            column_count=4,
            header_row_count=1,
            representation="markdown",
            cells=cells,
        )
        return DocumentElement(
            document_id="doc",
            element_id=f"wide-{page_number}",
            order=page_number - 1,
            element_type="table",
            content=render_table(table),
            format="markdown",
            fragments=[
                PageFragment(
                    page_number=page_number,
                    page_width=400,
                    page_height=800,
                    bbox=BoundingBox(x0=0, y0=y0, x1=400, y1=y1),
                )
            ],
            structure=ElementStructure(table=table),
            annotation=AnnotationMetadata(
                stage="candidate",
                revision=1,
                annotator="test",
                confidence=1,
                adjudication_status="unreviewed",
            ),
        )

    header = ("Practice", "Task", "Example", "Reference")
    first = four_column_element(1, 10, 550, (header, ("P1", "T1", "Example first", "Ref first")))
    second = four_column_element(2, 10, 100, (header, ("", "", "Example continued", "Ref continued")))

    result = merge_table_continuations([first, second])

    assert len(result) == 1
    table = result[0].structure.table
    assert table is not None
    assert table.row_count == 2
    assert next(cell for cell in table.cells if cell.text.startswith("Example first")).text == (
        "Example first Example continued"
    )
    assert next(cell for cell in table.cells if cell.text.startswith("Ref first")).text == (
        "Ref first Ref continued"
    )


def test_merge_three_pages_using_adjoining_fragment_geometry() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = table_element(page_number=2, y0=10, y1=790, order=1, x0=2)
    third = table_element(page_number=3, y0=10, y1=100, order=2, x0=2)

    result = merge_table_continuations([first, second, third])

    assert len(result) == 1
    assert [fragment.page_number for fragment in result[0].fragments] == [1, 2, 3]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(("x_factor", "y_factor"), [(1.0, 1.0), (1.7, 0.6)])
def test_three_page_continuation_collapses_uniform_repeated_table_foot(
    rotation: int,
    x_factor: float,
    y_factor: float,
) -> None:
    transform = lambda element: _quarter_turn(  # noqa: E731
        _affine_scale(_append_uniform_foot(element), x_factor, y_factor), rotation
    )
    first = transform(table_element(page_number=1, y0=700, y1=790, order=0))
    second = transform(table_element(page_number=2, y0=10, y1=790, order=1))
    third = transform(table_element(page_number=3, y0=10, y1=100, order=2))

    result = merge_table_continuations([first, second, third])

    assert len(result) == 1
    table = result[0].structure.table
    assert table is not None
    assert table.row_count == 5
    feet = [cell for cell in table.cells if cell.text == "Foot"]
    assert len(feet) == 2
    assert {fragment.page_number for cell in feet for fragment in cell.fragments} == {3}
    assert result[0].content == render_table(table)


def test_two_page_equal_table_feet_remain_ambiguous_and_are_not_collapsed() -> None:
    first = _append_uniform_foot(table_element(page_number=1, y0=700, y1=790, order=0))
    second = _append_uniform_foot(table_element(page_number=2, y0=10, y1=100, order=1))

    result = merge_table_continuations([first, second])

    table = result[0].structure.table
    assert table is not None
    assert table.row_count == 5
    assert len([cell for cell in table.cells if cell.text == "Foot"]) == 4


def test_three_page_repeated_multivalue_data_rows_are_not_collapsed() -> None:
    first = _replace_body_row(
        table_element(page_number=1, y0=700, y1=790, order=0),
        ("Repeated", "7"),
    )
    second = _replace_body_row(
        table_element(page_number=2, y0=10, y1=790, order=1),
        ("Repeated", "7"),
    )
    third = _replace_body_row(
        table_element(page_number=3, y0=10, y1=100, order=2),
        ("Repeated", "7"),
    )

    result = merge_table_continuations([first, second, third])

    table = result[0].structure.table
    assert table is not None
    assert table.row_count == 4
    assert len([cell for cell in table.cells if cell.text == "Repeated"]) == 3


def test_merge_headerless_table_across_page_boundary() -> None:
    def without_header(element: DocumentElement) -> DocumentElement:
        table = element.structure.table
        assert table is not None
        cells = [cell.model_copy(update={"role": "body"}) for cell in table.cells]
        updated_table = table.model_copy(
            update={
                "header_row_count": 0,
                "representation": "html",
                "classification_reasons": ["no_header"],
                "cells": cells,
            }
        )
        return element.model_copy(
            update={
                "content": render_table(updated_table),
                "format": "html",
                "structure": element.structure.model_copy(update={"table": updated_table}),
            }
        )

    first = without_header(table_element(page_number=1, y0=700, y1=790, order=0))
    second = without_header(table_element(page_number=2, y0=10, y1=100, order=1))

    result = merge_table_continuations([first, second])

    assert len(result) == 1
    table = result[0].structure.table
    assert table is not None
    assert table.header_row_count == 0
    assert table.row_count == 4


def test_do_not_merge_when_only_one_fragment_has_a_header() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = table_element(page_number=2, y0=10, y1=100, order=1)
    second_table = second.structure.table
    assert second_table is not None
    cells = [cell.model_copy(update={"role": "body"}) for cell in second_table.cells]
    second_table = second_table.model_copy(
        update={
            "header_row_count": 0,
            "representation": "html",
            "classification_reasons": ["no_header"],
            "cells": cells,
        }
    )
    second = second.model_copy(
        update={
            "content": render_table(second_table),
            "format": "html",
            "structure": second.structure.model_copy(update={"table": second_table}),
        }
    )

    assert len(merge_table_continuations([first, second])) == 2


def test_mismatched_headers_at_aligned_page_boundaries_are_ambiguous_html() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = table_element(page_number=2, y0=10, y1=100, order=1)
    second_table = second.structure.table
    assert second_table is not None
    second_cells = [
        table_cell.model_copy(update={"text": "Different"})
        if table_cell.row_index == 0 and table_cell.column_index == 0
        else table_cell
        for table_cell in second_table.cells
    ]
    second_table = second_table.model_copy(update={"cells": second_cells})
    second = second.model_copy(
        update={
            "content": render_table(second_table),
            "structure": second.structure.model_copy(update={"table": second_table}),
        }
    )

    result = merge_table_continuations([first, second])

    assert len(result) == 2
    assert all(element.format == "html" for element in result)
    assert all(
        "ambiguous_continuation" in element.structure.table.classification_reasons
        for element in result
        if element.structure.table is not None
    )
    assert all(element.content.startswith("<table><thead>") for element in result)


def test_ambiguous_boundary_survives_merging_later_continuation_pages() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    second = _replace_header(
        table_element(page_number=2, y0=10, y1=790, order=1),
        ("First", "Second"),
    )
    third = _replace_header(
        table_element(page_number=3, y0=10, y1=100, order=2),
        ("First", "Second"),
    )

    result = merge_table_continuations([first, second, third])

    assert len(result) == 2
    assert [fragment.page_number for fragment in result[1].fragments] == [2, 3]
    assert all(element.format == "html" for element in result)
    assert all(
        element.structure.table is not None
        and "ambiguous_continuation" in element.structure.table.classification_reasons
        for element in result
    )


def test_do_not_merge_table_away_from_page_boundary() -> None:
    first = table_element(page_number=1, y0=100, y1=200, order=0)
    second = table_element(page_number=2, y0=10, y1=100, order=1)

    assert len(merge_table_continuations([first, second])) == 2


@pytest.mark.parametrize(
    ("previous_y0", "previous_y1", "current_y0", "expected_count"),
    [
        (600, 700, 200, 1),
        (600, 700, 200.01, 2),
        (200, 520, 200, 1),
        (200.01, 520, 200, 2),
        (200, 519.99, 200, 2),
    ],
)
def test_continuation_boundary_thresholds_are_inclusive_only_at_the_declared_edges(
    previous_y0: float,
    previous_y1: float,
    current_y0: float,
    expected_count: int,
) -> None:
    first = table_element(page_number=1, y0=previous_y0, y1=previous_y1, order=0)
    second = table_element(page_number=2, y0=current_y0, y1=current_y0 + 90, order=1)

    assert len(merge_table_continuations([first, second])) == expected_count


def test_unrelated_nearby_table_prevents_nonadjacent_continuation_merge() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    unrelated = table_element(page_number=2, y0=10, y1=100, order=1, x0=20)
    continuation = table_element(page_number=2, y0=110, y1=200, order=2)

    result = merge_table_continuations([first, unrelated, continuation])

    assert len(result) == 3


def test_merge_extends_row_header_rowspan_across_appended_continuation_rows() -> None:
    first = table_element(page_number=1, y0=700, y1=790, order=0)
    first_table = first.structure.table
    assert first_table is not None
    first_cells = [
        cell.model_copy(update={"role": "row_header"})
        if cell.row_index == 1 and cell.column_index == 0
        else cell
        for cell in first_table.cells
    ]
    first_table = first_table.model_copy(update={"cells": first_cells})
    first = first.model_copy(
        update={
            "structure": first.structure.model_copy(update={"table": first_table}),
            "content": render_table(first_table),
        }
    )
    second = table_element(page_number=2, y0=10, y1=100, order=1)
    second_table = second.structure.table
    assert second_table is not None
    second_cells = [
        cell.model_copy(update={"text": ""}) if cell.row_index == 1 and cell.column_index == 0 else cell
        for cell in second_table.cells
    ]
    second_table = second_table.model_copy(update={"cells": second_cells})
    second = second.model_copy(
        update={
            "structure": second.structure.model_copy(update={"table": second_table}),
            "content": render_table(second_table),
        }
    )

    result = merge_table_continuations([first, second])

    table = result[0].structure.table
    assert table is not None
    row_header = next(cell for cell in table.cells if cell.role == "row_header")
    assert row_header.rowspan == 2
    assert [fragment.page_number for fragment in row_header.fragments] == [1, 2]
    assert table.row_count == 3
