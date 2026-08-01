import json
from pathlib import Path
from typing import cast

import pymupdf
import pytest

from app.pdf2md.engine import (
    _classify_recurring_margins,  # pyright: ignore[reportPrivateUsage]
    _consolidate_compound_margin_lines,  # pyright: ignore[reportPrivateUsage]
    _consolidate_compound_margin_semantics,  # pyright: ignore[reportPrivateUsage]
    _extract_page_text_blocks,  # pyright: ignore[reportPrivateUsage]
    _insert_tables_in_reading_order,  # pyright: ignore[reportPrivateUsage]
    _order_encompassed_figure_panel_leads,  # pyright: ignore[reportPrivateUsage]
    _page_reading_rotation,  # pyright: ignore[reportPrivateUsage]
    _semantic_elements,  # pyright: ignore[reportPrivateUsage]
    _TableSpanLedger,  # pyright: ignore[reportPrivateUsage]
    _TableWordSupport,  # pyright: ignore[reportPrivateUsage]
    _text_blocks_from_pymupdf_dict,  # pyright: ignore[reportPrivateUsage]
    extract_document_elements,
    extract_document_with_catalog,
    join_text_lines,
    render_document,
)
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FigureStructure,
    PageFragment,
    StructureProperty,
    TableCell,
    TableStructure,
    read_document_elements,
)
from app.pdf2md.semantic_text import (
    SemanticBlock,
    SemanticKind,
    TextBlock,
    TextLine,
    TextSpan,
    classify_semantic_blocks,
    infer_page_column_splits,
    link_footnotes,
    normalize_bbox,
)
from app.pdf2md.source_catalog import (
    SourceItem,
    build_source_catalog,
    make_source_item,
    read_document_with_source_catalog,
    source_catalog_path,
    validate_source_catalog,
    write_document_with_source_catalog,
)


def test_latex_longtable_continuation_uses_caption_identity_without_absorbing_next_table() -> None:
    pdf_path = Path("data/corpus/candidates/technical/latex-longtable.pdf")

    elements = extract_document_elements(pdf_path, pages=[2, 3, 4])

    tables = [element for element in elements if element.element_type == "table"]
    assert len(tables) == 2
    assert [[fragment.page_number for fragment in table.fragments] for table in tables] == [[2], [4]]
    first_structure = tables[0].structure.table
    second_structure = tables[1].structure.table
    assert first_structure is not None and second_structure is not None
    assert (first_structure.row_count, first_structure.column_count) == (4, 2)
    assert (second_structure.row_count, second_structure.column_count) == (2, 3)
    assert "Table 1: (continued)" in [
        element.content for element in elements if element.element_type == "caption"
    ]
    assert "Table 2: A floating table" in [
        element.content for element in elements if element.element_type == "caption"
    ]


def test_mozilla_ungrounded_boundary_tables_fall_back_to_native_text() -> None:
    elements = extract_document_elements(
        Path("data/corpus/candidates/institutional/mozilla-fin2024.pdf"),
        pages=[6, 7, 8],
    )
    tables = [element for element in elements if element.element_type == "table"]

    assert tables == []
    assert {6, 7, 8}.issubset({element.fragments[0].page_number for element in elements})
    assert all(
        table.structure.table is not None
        and "ambiguous_continuation" not in table.structure.table.classification_reasons
        for table in tables
    )


def test_rp2040_table_329_continues_exactly_without_absorbing_table_330() -> None:
    pdf_path = Path("data/corpus/candidates/technical/product-rp2040-datasheet.pdf")
    silver_path = Path("data/silver-cycle2/rp2040-datasheet.parquet")

    elements = extract_document_elements(pdf_path, pages=[291, 292, 293])
    silver = read_document_elements(silver_path)
    tables = [element for element in elements if element.element_type == "table"]

    table_329 = next(
        table for table in tables if [fragment.page_number for fragment in table.fragments] == [292, 293]
    )
    table_330 = next(
        table
        for table in tables
        if [fragment.page_number for fragment in table.fragments] == [293]
        and table.structure.table is not None
        and table.structure.table.row_count == 8
    )
    silver_329 = next(
        table
        for table in silver
        if table.structure.table is not None
        and [fragment.page_number for fragment in table.fragments] == [292, 293]
    )
    candidate_structure = table_329.structure.table
    silver_structure = silver_329.structure.table
    table_330_structure = table_330.structure.table
    assert candidate_structure is not None and silver_structure is not None
    assert table_330_structure is not None

    assert (candidate_structure.row_count, candidate_structure.column_count) == (26, 4)
    assert len(candidate_structure.cells) == 104
    assert (table_330_structure.row_count, table_330_structure.column_count) == (8, 4)
    assert table_329.content == silver_329.content
    without_geometry = lambda table: table.model_copy(  # noqa: E731
        update={
            "cells": [cell.model_copy(update={"fragments": []}) for cell in table.cells],
        }
    )
    assert without_geometry(candidate_structure) == without_geometry(silver_structure)


def test_rp2040_table_330_continues_without_absorbing_table_331() -> None:
    elements = extract_document_elements(
        Path("data/corpus/candidates/technical/product-rp2040-datasheet.pdf"),
        pages=[292, 293, 294],
    )
    tables = [element for element in elements if element.structure.table is not None]

    table_330 = next(
        table for table in tables if [fragment.page_number for fragment in table.fragments] == [293, 294]
    )
    table_331 = next(
        table
        for table in tables
        if [fragment.page_number for fragment in table.fragments] == [294]
        and table.structure.table is not None
        and table.structure.table.row_count == 13
    )
    table_330_structure = table_330.structure.table
    table_331_structure = table_331.structure.table
    assert table_330_structure is not None and table_331_structure is not None
    assert (table_330_structure.row_count, table_330_structure.column_count) == (26, 4)
    assert (table_331_structure.row_count, table_331_structure.column_count) == (13, 4)


def create_structured_pdf(path: Path) -> None:
    document = pymupdf.open()
    for page_number in (1, 2):
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 20), f"Quarterly Report {page_number}")
        page.insert_text((40, 80), f"Body paragraph from page {page_number} with native text.")
        for x in (50, 150, 250):
            page.draw_line((x, 120), (x, 220))
        for y in (120, 170, 220):
            page.draw_line((50, y), (250, y))
        page.insert_text((60, 145), "Name")
        page.insert_text((160, 145), "Value")
        page.insert_text((60, 195), f"Item {page_number}")
        page.insert_text((160, 195), str(page_number))
        page.insert_text((130, 385), f"Page {page_number}")
    document.save(path)
    document.close()


def create_fractional_cropbox_pdf(path: Path, rotation: int) -> None:
    document = pymupdf.open()
    page = document.new_page(width=500, height=500)
    page.set_cropbox(pymupdf.Rect(20.25, 30.5, 320.75, 330.75))
    page.insert_text((40, 60), "Paragraph above fractional crop table.")
    for x in (40.0, 150.0, 260.0):
        page.draw_line((x, 100.0), (x, 220.0))
    for y in (100.0, 160.0, 220.0):
        page.draw_line((40.0, y), (260.0, y))
    for point, text in (
        ((50.0, 130.0), "Name"),
        ((160.0, 130.0), "Value"),
        ((50.0, 190.0), "Alpha"),
        ((160.0, 190.0), "1"),
    ):
        page.insert_text(point, text)
    page.set_rotation(rotation)
    document.save(path)
    document.close()


@pytest.mark.parametrize(
    ("rotation", "expected_extent"),
    [(0, (300.5, 300.25)), (90, (300.25, 300.5))],
)
def test_fractional_cropbox_origin_uses_exact_canonical_extent_for_text_and_table(
    tmp_path: Path,
    rotation: int,
    expected_extent: tuple[float, float],
) -> None:
    pdf_path = tmp_path / f"fractional-crop-{rotation}.pdf"
    create_fractional_cropbox_pdf(pdf_path, rotation)

    extracted = extract_document_with_catalog(pdf_path)

    assert any(element.content == "Paragraph above fractional crop table." for element in extracted.elements)
    tables = [element for element in extracted.elements if element.structure.table is not None]
    assert len(tables) == 1
    table = tables[0].structure.table
    assert table is not None
    assert {cell.text for cell in table.cells} == {"Name", "Value", "Alpha", "1"}
    assert {
        (fragment.page_width, fragment.page_height)
        for element in extracted.elements
        for fragment in element.fragments
    } == {expected_extent}
    assert {
        (fragment.page_width, fragment.page_height) for cell in table.cells for fragment in cell.fragments
    } == {expected_extent}


def create_degenerate_rectangle_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=300, height=300)
    page.insert_text((40, 60), "Paragraph survives degenerate vector geometry.")
    page.draw_rect((100, 100, 100, 200))
    page.draw_rect((100, 220, 200, 220))
    document.save(path)
    document.close()


def test_whole_engine_tolerates_degenerate_rectangle_items(tmp_path: Path) -> None:
    pdf_path = tmp_path / "degenerate-rectangles.pdf"
    create_degenerate_rectangle_pdf(pdf_path)

    extracted = extract_document_with_catalog(pdf_path)

    assert render_document(list(extracted.elements)) == "Paragraph survives degenerate vector geometry."
    assert extracted.source_catalog.items


def create_rotated_table_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=300, height=400)
    page.insert_text((40, 80), "Body text outside the table.")
    for x in (50, 150, 250):
        page.draw_line((x, 120), (x, 220))
    for y in (120, 170, 220):
        page.draw_line((50, y), (250, y))
    page.insert_text((60, 145), "Name")
    page.insert_text((160, 145), "Value")
    page.insert_text((60, 195), "Item")
    page.insert_text((160, 195), "7")
    page.set_rotation(90)
    document.save(path)
    document.close()


def create_external_financial_header_pdf(
    path: Path,
    *,
    rotation: int = 0,
    header_mode: str = "adjacent",
    scale: float = 1.0,
    translation: tuple[float, float] = (0.0, 0.0),
) -> None:
    dx, dy = translation

    def point(x: float, y: float) -> tuple[float, float]:
        return x * scale + dx, y * scale + dy

    document = pymupdf.open()
    page = document.new_page(width=400 * scale + 2 * dx, height=400 * scale + 2 * dy)
    page.insert_text(point(150, 55), "Financial Summary", fontsize=12 * scale)
    page.insert_text(point(165, 72), "(In millions)", fontsize=9 * scale)
    if header_mode == "adjacent":
        page.insert_text(point(255, 94), "Years ended", fontsize=9 * scale)
        page.draw_line(point(200, 98), point(380, 98))
        page.insert_text(point(220, 108), "December 31,", fontsize=9 * scale)
        page.insert_text(point(310, 108), "December 31,", fontsize=9 * scale)
        page.insert_text(point(235, 119), "2024", fontsize=9 * scale)
        page.insert_text(point(325, 119), "2023", fontsize=9 * scale)
        page.draw_line(point(200, 122), point(380, 122))
    elif header_mode == "separated":
        page.insert_text(point(220, 82), "Table 7. Historical context", fontsize=9 * scale)
        page.insert_text(point(235, 100), "2024", fontsize=9 * scale)
        page.insert_text(point(325, 100), "2023", fontsize=9 * scale)
    elif header_mode == "one-period":
        page.insert_text(point(235, 108), "December 31, 2024", fontsize=9 * scale)
        page.draw_line(point(200, 122), point(380, 122))
    elif header_mode == "ambiguous":
        page.insert_text(point(205, 108), "Management expects 2024 and 2023 growth", fontsize=9 * scale)
        page.draw_line(point(200, 122), point(380, 122))
    elif header_mode != "none":
        raise ValueError(f"unsupported synthetic header mode: {header_mode}")

    for x in (20, 200, 210, 285, 290, 300, 380):
        page.draw_line(point(x, 135), point(x, 225))
    for y in (135, 165, 195, 225):
        page.draw_line(point(20, y), point(380, y))
    first_label = "1" if header_mode == "none" else "Item"
    for x, y, text in (
        (30, 155, first_label),
        (202, 155, "$"),
        (250, 155, "10"),
        (292, 155, "$"),
        (340, 155, "20"),
        (30, 185, "Other"),
        (250, 185, "11"),
        (340, 185, "21"),
        (30, 215, "Total"),
        (250, 215, "21"),
        (340, 215, "41"),
    ):
        page.insert_text(point(x, y), text, fontsize=9 * scale)
    page.set_rotation(rotation)
    document.save(path)
    document.close()


def create_borderless_grid(path: Path, *, numeric: bool) -> None:
    document = pymupdf.open()
    page = document.new_page(width=320, height=400)
    values = (
        [("Region", "2025", "2026"), ("North", "10", "12"), ("South", "8", "9")]
        if numeric
        else [("First", "Second", "Third"), ("Alpha", "Beta", "Gamma"), ("Delta", "Epsilon", "Zeta")]
    )
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            page.insert_text((40 + column_index * 90, 80 + row_index * 30), value)
    document.save(path)
    document.close()


def create_two_column_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    for x, y, text in (
        (40, 100, "Left column first."),
        (40, 500, "Left column second."),
        (330, 100, "Right column first."),
        (330, 500, "Right column second."),
    ):
        page.insert_text((x, y), text, fontsize=10)
    document.save(path)
    document.close()


def _raw_span(
    text: object,
    bbox: tuple[float, float, float, float],
    *,
    size: float = 10,
    flags: int = 0,
) -> dict[str, object]:
    return {
        "text": text,
        "bbox": bbox,
        "size": size,
        "font": "SyntheticFont",
        "flags": flags,
    }


def _raw_text_block(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    size: float = 10,
    flags: int = 0,
) -> dict[str, object]:
    return {
        "type": 0,
        "bbox": bbox,
        "lines": [{"bbox": bbox, "spans": [_raw_span(text, bbox, size=size, flags=flags)]}],
    }


def test_join_text_lines_preserves_semantic_hyphens_and_removes_soft_hyphens() -> None:
    assert join_text_lines(["whole-", "document options"]) == "whole-document options"
    assert join_text_lines(["develop\N{SOFT HYPHEN}", "ment"]) == "development"


def test_engine_builds_semantic_elements_with_flags_and_all_source_fragments() -> None:
    payload = {
        "blocks": [
            _raw_text_block("Semantic Extraction", (40, 40, 300, 65), size=18, flags=18),
            _raw_text_block("Semantic extraction continues", (40, 100, 300, 112)),
            _raw_text_block("across retained source blocks.", (40, 115, 300, 127)),
            _raw_text_block("Figure 2. Semantic pipeline", (80, 300, 300, 312), size=9),
            _raw_text_block("1. Supporting qualification", (40, 700, 300, 712), size=8),
            _raw_text_block("Accepted table text", (100, 400, 240, 412)),
        ]
    }

    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[BoundingBox(x0=90, y0=390, x1=250, y1=420)],
    )
    semantic = classify_semantic_blocks(blocks, body_font_size=10)
    elements = _semantic_elements(semantic, "document-id", "test")

    assert blocks[0].lines[0].spans[0].font_name == "SyntheticFont"
    assert blocks[0].lines[0].spans[0].is_bold
    assert blocks[0].lines[0].spans[0].is_italic
    assert "Accepted table text" in {block.text for block in blocks}
    paragraph_structures = [element.structure.paragraph for element in elements]
    assert all(structure is not None for structure in paragraph_structures)
    assert [
        (element.element_type, structure.role)
        for element, structure in zip(elements, paragraph_structures, strict=True)
        if structure is not None
    ] == [
        ("heading", "heading"),
        ("paragraph", "body"),
        ("caption", "caption"),
        ("paragraph", "body"),
        ("footnote", "footnote"),
    ]
    heading_structure = elements[0].structure.paragraph
    assert heading_structure is not None
    assert heading_structure.heading_level == 1
    assert elements[1].content == "Semantic extraction continues across retained source blocks."
    assert len(elements[1].fragments) == 2
    assert elements[1].fragments[0].bbox.y0 == 100
    assert elements[1].fragments[1].bbox.y0 == 115
    assert elements[4].structure.footnote is not None


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_footnotes_link_reciprocally_and_render_as_canonical_markdown(scale: float) -> None:
    def semantic(
        kind: str,
        spans: tuple[tuple[str, float, tuple[float, float, float, float]], ...],
        *,
        page: int,
        bbox: tuple[float, float, float, float],
    ) -> SemanticBlock:
        scaled_bbox = tuple(value * scale for value in bbox)
        scaled_bbox = cast(tuple[float, float, float, float], scaled_bbox)
        text_spans = tuple(
            TextSpan(
                text=text,
                font_size=size * scale,
                bbox=cast(
                    tuple[float, float, float, float],
                    tuple(value * scale for value in span_bbox),
                ),
            )
            for text, size, span_bbox in spans
        )
        line = TextLine(spans=text_spans, bbox=scaled_bbox)
        source = TextBlock(
            lines=(line,),
            bbox=scaled_bbox,
            page_number=page,
            page_width=600 * scale,
            page_height=800 * scale,
        )
        return SemanticBlock(kind=cast(SemanticKind, kind), text=line.text, source_blocks=(source,))

    reference = semantic(
        "paragraph",
        (
            ("The measured result", 10, (40, 300, 130, 310)),
            ("69", 5.2, (130, 299, 136, 305)),
            (" was stable.", 10, (136, 300, 200, 310)),
        ),
        page=1,
        bbox=(40, 299, 200, 310),
    )
    note = semantic(
        "footnote",
        (("69 GHG emission metrics use a common unit.", 8, (40, 700, 300, 712)),),
        page=1,
        bbox=(40, 700, 300, 712),
    )
    semantics = (reference, note)
    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    footnote = elements[1].structure.footnote
    assert footnote is not None
    assert elements[0].content == "The measured result[^69] was stable."
    assert elements[0].structure.linked_element_ids == [elements[1].element_id]
    assert footnote.label == "69"
    assert footnote.reference_element_ids == [elements[0].element_id]
    assert footnote.association_confident
    assert elements[1].structure.linked_element_ids == [elements[0].element_id]
    assert render_document(elements) == (
        "The measured result[^69] was stable.\n\n[^69]: GHG emission metrics use a common unit."
    )


def _one_line_semantic(
    kind: SemanticKind,
    text: str,
    spans: tuple[TextSpan, ...],
    *,
    page: int,
    bbox: tuple[float, float, float, float],
    direction: tuple[float, float] = (1.0, 0.0),
) -> SemanticBlock:
    line = TextLine(spans=spans, bbox=bbox, direction=direction)
    source = TextBlock(
        lines=(line,),
        bbox=bbox,
        page_number=page,
        page_width=800,
        page_height=800,
    )
    return SemanticBlock(kind=kind, text=text, source_blocks=(source,))


def _plain_semantic(kind: SemanticKind, text: str, *, page: int, y: float) -> SemanticBlock:
    bbox = (40.0, y, 500.0, y + 12)
    return _one_line_semantic(
        kind,
        text,
        (TextSpan(text=text, bbox=bbox, font_size=8 if kind == "footnote" else 10),),
        page=page,
        bbox=bbox,
    )


def test_footnote_linking_handles_repeated_labels_cross_page_and_false_positive_numbers() -> None:
    def one_line(
        kind: str,
        text: str,
        spans: tuple[TextSpan, ...],
        *,
        page: int,
        bbox: tuple[float, float, float, float],
    ) -> SemanticBlock:
        line = TextLine(spans=spans, bbox=bbox)
        source = TextBlock(
            lines=(line,),
            bbox=bbox,
            page_number=page,
            page_width=600,
            page_height=800,
        )
        return SemanticBlock(kind=cast(SemanticKind, kind), text=text, source_blocks=(source,))

    def plain(kind: str, text: str, page: int, y: float) -> SemanticBlock:
        bbox = (40.0, y, 400.0, y + 12)
        return one_line(
            kind,
            text,
            (TextSpan(text=text, bbox=bbox, font_size=8 if kind == "footnote" else 10),),
            page=page,
            bbox=bbox,
        )

    repeated_reference = one_line(
        "paragraph",
        "Local claim1.",
        (
            TextSpan(text="Local claim", bbox=(40, 200, 100, 210), font_size=10),
            TextSpan(text="1", bbox=(100, 199, 104, 205), font_size=6),
            TextSpan(text=".", bbox=(104, 200, 107, 210), font_size=10),
        ),
        page=2,
        bbox=(40, 199, 107, 210),
    )
    cross_page_reference = one_line(
        "paragraph",
        "Cross-page claim³.",
        (TextSpan(text="Cross-page claim³.", bbox=(40, 300, 150, 310), font_size=10),),
        page=3,
        bbox=(40, 300, 150, 310),
    )
    equation = one_line(
        "paragraph",
        "The equation is W1x + x.",
        (
            TextSpan(text="The equation is W", bbox=(40, 400, 125, 410), font_size=10),
            TextSpan(text="1", bbox=(125, 404, 129, 410), font_size=6),
            TextSpan(text="x + x.", bbox=(129, 400, 160, 410), font_size=10),
        ),
        page=2,
        bbox=(40, 400, 160, 410),
    )
    semantics = (
        plain("footnote", "1 First-page note.", 1, 700),
        repeated_reference,
        equation,
        plain("footnote", "1 Second-page note.", 2, 700),
        cross_page_reference,
        plain("footnote", "³ Cross-page note.", 4, 700),
    )
    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    first_note = elements[0].structure.footnote
    second_note = elements[3].structure.footnote
    cross_page_note = elements[5].structure.footnote
    assert first_note is not None and first_note.reference_element_ids == []
    assert second_note is not None and second_note.reference_element_ids == [elements[1].element_id]
    assert elements[1].content == "Local claim[^1]."
    assert elements[2].content == "The equation is W1x + x."
    assert elements[2].structure.linked_element_ids == []
    assert cross_page_note is not None and cross_page_note.label == "3"
    assert cross_page_note.reference_element_ids == [elements[4].element_id]
    assert elements[4].content == "Cross-page claim[^3]."


@pytest.mark.parametrize(
    ("direction", "regular_bbox", "marker_bbox", "following_bbox"),
    [
        ((1.0, 0.0), (40, 100, 100, 110), (100, 98, 104, 104), (104, 100, 110, 110)),
        ((0.0, -1.0), (100, 100, 110, 160), (98, 96, 104, 100), (100, 90, 110, 96)),
        ((-1.0, 0.0), (100, 100, 160, 110), (96, 106, 100, 112), (90, 100, 96, 110)),
        ((0.0, 1.0), (100, 40, 110, 100), (106, 100, 112, 104), (100, 104, 110, 110)),
    ],
)
def test_reference_geometry_is_quarter_turn_invariant(
    direction: tuple[float, float],
    regular_bbox: tuple[float, float, float, float],
    marker_bbox: tuple[float, float, float, float],
    following_bbox: tuple[float, float, float, float],
) -> None:
    reference = _one_line_semantic(
        "paragraph",
        "Claim1.",
        (
            TextSpan(text="Claim", bbox=regular_bbox, font_size=10),
            TextSpan(text="1", bbox=marker_bbox, font_size=6),
            TextSpan(text=".", bbox=following_bbox, font_size=10),
        ),
        page=1,
        bbox=(
            min(regular_bbox[0], marker_bbox[0], following_bbox[0]),
            min(regular_bbox[1], marker_bbox[1], following_bbox[1]),
            max(regular_bbox[2], marker_bbox[2], following_bbox[2]),
            max(regular_bbox[3], marker_bbox[3], following_bbox[3]),
        ),
        direction=direction,
    )
    note = _plain_semantic("footnote", "1 Rotated supporting detail.", page=1, y=700)
    semantics = (reference, note)

    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    assert elements[0].content == "Claim[^1]."
    assert elements[0].structure.linked_element_ids == [elements[1].element_id]


def test_cross_page_math_scripts_do_not_borrow_an_adjoining_page_note() -> None:
    equation = _one_line_semantic(
        "paragraph",
        "∂2f(x)",
        (
            TextSpan(text="∂", bbox=(40, 100, 50, 110), font_size=10),
            TextSpan(text="2", bbox=(50, 97, 54, 103), font_size=6),
            TextSpan(text="f(x)", bbox=(54, 100, 75, 110), font_size=10),
        ),
        page=2,
        bbox=(40, 97, 75, 110),
    )
    note = _plain_semantic("footnote", "2 https://example.test/source", page=1, y=700)
    semantics = (note, equation)

    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    assert elements[1].content == "∂2f(x)"
    assert elements[1].structure.linked_element_ids == []
    footnote = elements[0].structure.footnote
    assert footnote is not None and not footnote.association_confident


def test_same_page_repeated_equation_markers_link_to_one_note_reciprocally() -> None:
    references = tuple(
        _one_line_semantic(
            "paragraph",
            f"M{index}=x2.",
            (
                TextSpan(text=f"M{index}=x", bbox=(40, y, 85, y + 10), font_size=10),
                TextSpan(text="2", bbox=(85, y - 3, 89, y + 3), font_size=6),
                TextSpan(text=".", bbox=(89, y, 92, y + 10), font_size=10),
            ),
            page=5,
            bbox=(40, y - 3, 92, y + 10),
        )
        for index, y in ((1, 100.0), (2, 140.0))
    )
    note = _plain_semantic("footnote", "2 Normalization applies to both equations.", page=5, y=700)
    semantics = (*references, note)

    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    assert [element.content for element in elements[:2]] == ["M1=x[^2].", "M2=x[^2]."]
    footnote = elements[2].structure.footnote
    assert footnote is not None
    assert footnote.reference_element_ids == [elements[0].element_id, elements[1].element_id]
    assert elements[2].structure.linked_element_ids == [elements[0].element_id, elements[1].element_id]


def test_repeated_source_labels_render_as_unique_document_global_markdown_labels() -> None:
    semantics = (
        _one_line_semantic(
            "paragraph",
            "First1.",
            (
                TextSpan(text="First", bbox=(40, 100, 70, 110), font_size=10),
                TextSpan(text="1", bbox=(70, 97, 74, 103), font_size=6),
                TextSpan(text=".", bbox=(74, 100, 77, 110), font_size=10),
            ),
            page=1,
            bbox=(40, 97, 77, 110),
        ),
        _plain_semantic("footnote", "1 First note.", page=1, y=700),
        _one_line_semantic(
            "paragraph",
            "Second1.",
            (
                TextSpan(text="Second", bbox=(40, 100, 78, 110), font_size=10),
                TextSpan(text="1", bbox=(78, 97, 82, 103), font_size=6),
                TextSpan(text=".", bbox=(82, 100, 85, 110), font_size=10),
            ),
            page=2,
            bbox=(40, 97, 85, 110),
        ),
        _plain_semantic("footnote", "1 Second note.", page=2, y=700),
    )
    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    markdown = render_document(elements)

    assert "First[^1-p1]." in markdown
    assert "[^1-p1]: First note." in markdown
    assert "Second[^1-p2]." in markdown
    assert "[^1-p2]: Second note." in markdown
    assert "[^1]:" not in markdown


def test_unlinked_footnote_candidate_renders_as_extractive_text() -> None:
    semantic = _plain_semantic("footnote", "3 768 12 5.84 77.9 79.8 88.4", page=9, y=700)
    elements = link_footnotes(_semantic_elements((semantic,), "doc", "test"), (semantic,))

    assert render_document(elements) == semantic.text


def test_marker_replacement_tolerates_normalized_spacing_between_native_spans() -> None:
    reference = _one_line_semantic(
        "paragraph",
        "Method [10]1 is effective.",
        (
            TextSpan(text="Method [10]", bbox=(40, 100, 100, 110), font_size=10),
            TextSpan(text="1", bbox=(100, 97, 104, 103), font_size=6),
            TextSpan(text=" is effective.", bbox=(104, 100, 165, 110), font_size=10),
        ),
        page=1,
        bbox=(40, 97, 165, 110),
    )
    note = _plain_semantic("footnote", "1 Terminology note.", page=1, y=700)
    semantics = (reference, note)

    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    assert elements[0].content == "Method [10][^1] is effective."
    assert elements[1].structure.linked_element_ids == [elements[0].element_id]


def test_chemical_superscript_and_stripped_list_marker_do_not_create_links() -> None:
    chemical = _one_line_semantic(
        "paragraph",
        "Fe2+ remains stable.",
        (
            TextSpan(text="Fe", bbox=(40, 100, 52, 110), font_size=10),
            TextSpan(text="2", bbox=(52, 97, 56, 103), font_size=6),
            TextSpan(text="+ remains stable.", bbox=(56, 100, 130, 110), font_size=10),
        ),
        page=1,
        bbox=(40, 97, 130, 110),
    )
    list_source = _one_line_semantic(
        "paragraph",
        "First item.",
        (
            TextSpan(text="1", bbox=(40, 137, 44, 143), font_size=6),
            TextSpan(text="First item.", bbox=(44, 140, 100, 150), font_size=10),
        ),
        page=1,
        bbox=(40, 137, 100, 150),
    )
    note = _plain_semantic("footnote", "1 Unrelated note.", page=1, y=700)
    note_two = _plain_semantic("footnote", "2 Another unrelated note.", page=1, y=720)
    semantics = (chemical, list_source, note, note_two)

    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    assert all(element.structure.linked_element_ids == [] for element in elements)
    assert [element.content for element in elements[:2]] == ["Fe2+ remains stable.", "First item."]


def test_symbol_label_renders_without_markdown_emphasis_escaping() -> None:
    reference = _one_line_semantic(
        "paragraph",
        "Qualified*.",
        (
            TextSpan(text="Qualified", bbox=(40, 100, 90, 110), font_size=10),
            TextSpan(text="*", bbox=(90, 97, 94, 103), font_size=6),
            TextSpan(text=".", bbox=(94, 100, 97, 110), font_size=10),
        ),
        page=1,
        bbox=(40, 97, 97, 110),
    )
    note = _plain_semantic("footnote", "*. Symbol-labeled detail.", page=1, y=700)
    semantics = (reference, note)
    elements = link_footnotes(_semantic_elements(semantics, "doc", "test"), semantics)

    assert render_document(elements) == "Qualified[^*].\n\n[^*]: Symbol-labeled detail."


def test_footnote_linking_rejects_duplicate_element_ids() -> None:
    semantics = (
        _plain_semantic("paragraph", "Body text.", page=1, y=100),
        _plain_semantic("footnote", "1 Note.", page=1, y=700),
    )
    elements = _semantic_elements(semantics, "doc", "test")
    elements[1] = elements[1].model_copy(update={"element_id": elements[0].element_id})

    with pytest.raises(ValueError, match="unique element IDs"):
        link_footnotes(elements, semantics)


def test_gnu_make_native_paragraph_continues_across_pages_112_and_113() -> None:
    elements = extract_document_elements(
        Path("data/corpus/candidates/technical/manual-gnu-make.pdf"),
        pages=[112, 113, 114, 115],
    )

    continuation = next(
        element for element in elements if element.content.startswith("The operator op can be >")
    )
    assert [fragment.page_number for fragment in continuation.fragments] == [112, 113]
    assert "contents of the file will be read in" in continuation.content


def test_gnu_make_cross_page_join_is_stable_for_exact_requested_subset() -> None:
    pdf_path = Path("data/corpus/candidates/technical/manual-gnu-make.pdf")

    exact = extract_document_elements(pdf_path, pages=[112, 113])
    repeated = extract_document_elements(pdf_path, pages=[112, 113])
    continuation = next(
        element for element in exact if element.content.startswith("The operator op can be >")
    )
    repeated_continuation = next(
        element for element in repeated if element.content.startswith("The operator op can be >")
    )

    assert continuation == repeated_continuation
    assert [fragment.page_number for fragment in continuation.fragments] == [112, 113]
    assert continuation.content.endswith(
        "There may optionally be whitespace between the operator and the file name."
    )
    assert continuation.model_dump(mode="json") == repeated_continuation.model_dump(mode="json")


def test_atlas_native_boundaries_join_only_proven_column_continuations() -> None:
    pdf_path = Path(
        "data/corpus/candidates/expansion-2026-07-30/scientific/arxiv-1207-7214-higgs-boson-atlas.pdf"
    )
    selected_pages = (4, 5, 15, 16, 21, 22, 34, 35)
    blocks = []
    page_column_splits: dict[int, tuple[float, ...]] = {}
    with pymupdf.open(pdf_path) as document:
        for page_number in selected_pages:
            page_blocks = _extract_page_text_blocks(document[page_number - 1], [])
            reading_rotation = _page_reading_rotation(page_blocks)
            page_column_splits[page_number] = infer_page_column_splits([
                block for block in page_blocks if block.reading_rotation == reading_rotation
            ])
            blocks.extend(page_blocks)

    semantics = classify_semantic_blocks(
        blocks,
        body_font_size=9,
        recurring_min_pages=4,
        page_column_splits=page_column_splits,
    )
    joins = {semantic.page_numbers: semantic.text for semantic in semantics if len(semantic.page_numbers) > 1}

    assert set(joins) == {(4, 5)}
    assert joins[(4, 5)].startswith("The data are selected using single-lepton or dilepton triggers.")
    assert joins[(4, 5)].endswith("leading and sub-leading muon, respectively.")
    assert not any("yield analysis include" in text for text in joins.values())
    assert not any("excess is driven butions" in text for text in joins.values())


def test_engine_emits_one_element_with_fragments_for_a_proven_cross_page_paragraph() -> None:
    previous = _text_blocks_from_pymupdf_dict(
        {"blocks": [_raw_text_block("The file operator reads the contents of", (40, 700, 560, 750))]},
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )
    current = _text_blocks_from_pymupdf_dict(
        {"blocks": [_raw_text_block("the named file.", (40, 45, 560, 75))]},
        page_number=2,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )

    semantic = classify_semantic_blocks([*previous, *current], body_font_size=10)
    elements = _semantic_elements(semantic, "document-id", "test")

    assert len(elements) == 1
    assert elements[0].content == "The file operator reads the contents of the named file."
    assert [fragment.page_number for fragment in elements[0].fragments] == [1, 2]


def test_engine_preserves_overprinted_zero_advance_native_characters() -> None:
    line_bbox = (89.0, 149.0, 150.0, 161.0)
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "dir": (1.0, 0.0),
                        "spans": [
                            _raw_span("S", (89.0, 150.0, 92.0, 161.0)),
                            _raw_span("t", (89.0, 149.0, 89.0, 161.0)),
                            _raw_span("euergerät", (92.0, 149.0, 150.0, 161.0)),
                        ],
                        "bbox": line_bbox,
                    }
                ],
                "bbox": line_bbox,
            }
        ]
    }

    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )

    assert len(blocks) == 1
    assert blocks[0].text == "Steuergerät"
    assert [span.text for span in blocks[0].lines[0].spans] == ["St", "euergerät"]


def test_engine_maps_semantic_list_metadata_to_paragraph_structure() -> None:
    blocks = _text_blocks_from_pymupdf_dict(
        {"blocks": [_raw_text_block("(a) Synthetic list item", (40, 100, 300, 112))]},
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )

    elements = _semantic_elements(classify_semantic_blocks(blocks, body_font_size=10), "doc", "test")

    structure = elements[0].structure.paragraph
    assert structure is not None
    assert elements[0].content == "Synthetic list item"
    assert structure.role == "list_item"
    assert structure.list_label == "(a)"
    assert structure.list_depth == 0


def test_engine_preserves_native_text_direction_for_compensated_page_rotation() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "dir": (0.0, -1.0),
                        "spans": [_raw_span("Visually upright text", (20, 100, 35, 180))],
                    }
                ],
            }
        ]
    }

    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=200,
        page_height=100,
        rotation=90,
        table_bboxes=[],
    )

    assert blocks[0].lines[0].direction == (0.0, -1.0)
    assert blocks[0].reading_rotation == 90
    assert blocks[0].reading_bbox == pytest.approx((-80, 20, 0, 35))


def test_engine_canonicalizes_rotated_text_and_fragments_once() -> None:
    blocks = _text_blocks_from_pymupdf_dict(
        {"blocks": [_raw_text_block("Rotated text", (10, 20, 80, 35))]},
        page_number=3,
        page_width=200,
        page_height=100,
        rotation=90,
        table_bboxes=[],
    )
    elements = _semantic_elements(
        (SemanticBlock(kind="paragraph", text=blocks[0].text, source_blocks=(blocks[0],)),),
        "document-id",
        "test",
    )

    assert blocks[0].rotation == 90
    assert blocks[0].bbox == (10, 20, 80, 35)
    assert blocks[0].normalized_bbox == (65, 10, 80, 80)
    assert blocks[0].normalized_page_size == (100, 200)
    assert elements[0].fragments[0].bbox == BoundingBox(x0=65, y0=10, x1=80, y1=80)
    assert elements[0].fragments[0].page_width == 100
    assert elements[0].fragments[0].page_height == 200


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_quarter_turns_preserve_unsupported_overlap_semantics_and_native_fragments(
    rotation: int,
) -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            _raw_span("18 layers", (10, 20, 60, 30)),
                            _raw_span("27.94", (90, 20, 120, 30)),
                        ]
                    },
                    {
                        "spans": [
                            _raw_span("34 layers", (10, 40, 60, 50)),
                            _raw_span("28.54", (90, 40, 120, 50)),
                        ]
                    },
                ],
            }
        ]
    }
    page_width, page_height = 200.0, 100.0
    displayed_table_bbox = normalize_bbox((80, 15, 130, 55), page_width, page_height, rotation)
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=page_width,
        page_height=page_height,
        rotation=rotation,
        table_bboxes=[
            BoundingBox(
                x0=displayed_table_bbox[0],
                y0=displayed_table_bbox[1],
                x1=displayed_table_bbox[2],
                y1=displayed_table_bbox[3],
            )
        ],
    )
    semantic = classify_semantic_blocks(blocks, body_font_size=10)
    elements = _semantic_elements(semantic, "document-id", "test")

    expected = "18 layers 27.94 34 layers 28.54"
    assert [item.text for item in blocks] == [expected]
    assert [(item.kind, item.text) for item in semantic] == [("paragraph", expected)]
    assert len(elements[0].fragments) == 1
    fragment = elements[0].fragments[0]
    assert fragment.page_width is not None
    assert fragment.page_height is not None
    restored_bbox = normalize_bbox(
        (fragment.bbox.x0, fragment.bbox.y0, fragment.bbox.x1, fragment.bbox.y1),
        fragment.page_width,
        fragment.page_height,
        -rotation,
    )
    assert restored_bbox == pytest.approx((10, 20, 120, 50))


def test_engine_preserves_column_aware_text_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "two-column.pdf"
    create_two_column_pdf(pdf_path)

    elements = extract_document_elements(pdf_path)

    assert [element.content for element in elements] == [
        "Left column first.",
        "Left column second.",
        "Right column first.",
        "Right column second.",
    ]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_engine_absorbs_rule_connected_external_financial_headers_once(tmp_path: Path, rotation: int) -> None:
    pdf_path = tmp_path / f"external-financial-header-{rotation}.pdf"
    create_external_financial_header_pdf(pdf_path, rotation=rotation)

    elements = extract_document_elements(pdf_path)
    tables = [element for element in elements if element.element_type == "table"]

    assert len(tables) == 1
    structure = tables[0].structure.table
    assert structure is not None
    assert (structure.row_count, structure.column_count, structure.header_row_count) == (5, 3, 2)
    assert "Years ended" in tables[0].content
    assert "December 31, 2024" in tables[0].content
    assert "December 31, 2023" in tables[0].content
    assert "$ 10" in tables[0].content
    assert "$ 20" in tables[0].content
    assert "Financial Summary" in {element.content for element in elements if element.element_type != "table"}
    assert "(In millions)" in {element.content for element in elements if element.element_type != "table"}
    residual = " ".join(element.content for element in elements if element.element_type != "table")
    assert "Years ended" not in residual
    assert "December 31" not in residual


@pytest.mark.parametrize(
    ("scale", "translation"),
    [(0.75, (0.0, 0.0)), (2.5, (17.0, 31.0))],
)
def test_external_financial_header_recovery_is_scale_and_translation_invariant(
    tmp_path: Path,
    scale: float,
    translation: tuple[float, float],
) -> None:
    pdf_path = tmp_path / f"external-financial-header-{scale}.pdf"
    create_external_financial_header_pdf(
        pdf_path,
        scale=scale,
        translation=translation,
    )

    elements = extract_document_elements(pdf_path)
    table = next(element for element in elements if element.element_type == "table")
    structure = table.structure.table

    assert structure is not None
    assert (structure.row_count, structure.column_count, structure.header_row_count) == (5, 3, 2)
    assert "December 31, 2024" in table.content
    assert "December 31, 2023" in table.content
    residual = " ".join(element.content for element in elements if element.element_type != "table")
    assert "Financial Summary" in residual
    assert "(In millions)" in residual
    assert "Years ended" not in residual
    assert "December 31" not in residual


@pytest.mark.parametrize(
    ("header_mode", "retained_text", "expected_header_rows"),
    [
        ("separated", "Table 7. Historical context", 1),
        ("one-period", "December 31, 2024", 1),
        ("ambiguous", "Management expects 2024 and 2023 growth", 1),
        ("none", "Financial Summary", 0),
    ],
)
def test_engine_does_not_absorb_separated_ambiguous_or_missing_headers(
    tmp_path: Path,
    header_mode: str,
    retained_text: str,
    expected_header_rows: int,
) -> None:
    pdf_path = tmp_path / f"financial-control-{header_mode}.pdf"
    create_external_financial_header_pdf(pdf_path, header_mode=header_mode)

    elements = extract_document_elements(pdf_path)
    table = next(element for element in elements if element.element_type == "table")
    structure = table.structure.table

    assert structure is not None
    assert (structure.row_count, structure.column_count, structure.header_row_count) == (
        3,
        6,
        expected_header_rows,
    )
    assert retained_text in " ".join(
        element.content for element in elements if element.element_type != "table"
    )


def test_engine_recovers_numeric_borderless_table(tmp_path: Path) -> None:
    pdf_path = tmp_path / "borderless-numeric.pdf"
    create_borderless_grid(pdf_path, numeric=True)

    elements = extract_document_elements(pdf_path)

    assert sum(element.element_type == "table" for element in elements) == 1


def test_engine_rejects_aligned_prose_as_borderless_table(tmp_path: Path) -> None:
    pdf_path = tmp_path / "aligned-prose.pdf"
    create_borderless_grid(pdf_path, numeric=False)

    elements = extract_document_elements(pdf_path)

    assert all(element.element_type != "table" for element in elements)


def test_engine_inserts_tables_and_omits_recurring_margins(tmp_path: Path) -> None:
    pdf_path = tmp_path / "structured.pdf"
    create_structured_pdf(pdf_path)

    elements = extract_document_elements(pdf_path)
    markdown = render_document(elements)

    assert [element.order for element in elements] == list(range(len(elements)))
    assert sum(element.element_type == "table" for element in elements) == 2
    assert any(element.element_type == "header" for element in elements)
    assert any(element.element_type == "footer" for element in elements)
    margin_roles = {
        element.structure.paragraph.role
        for element in elements
        if element.structure.paragraph is not None and not element.include_in_output
    }
    assert margin_roles == {"running_header", "page_number"}
    assert "Body paragraph from page 1" in markdown
    assert "| Name | Value |" in markdown
    assert "Quarterly Report" not in markdown
    assert "Page 1" not in markdown


def _table_element(
    bbox: tuple[float, float, float, float], *, page_width: float = 600, page_height: float = 800
) -> DocumentElement:
    fragment = PageFragment(
        page_number=1,
        page_width=page_width,
        page_height=page_height,
        bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
    )
    cell = TableCell(
        row_index=0,
        column_index=0,
        role="header",
        text="Value",
        fragments=[fragment],
    )
    table = TableStructure(
        row_count=1,
        column_count=1,
        header_row_count=1,
        representation="markdown",
        cells=[cell],
    )
    return DocumentElement(
        document_id="document-id",
        element_id="table-1",
        order=0,
        element_type="table",
        content="| Value |\n| --- |",
        format="markdown",
        fragments=[fragment],
        structure=ElementStructure(table=table),
        annotation=AnnotationMetadata(
            stage="candidate",
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        ),
    )


def _suppression_property(
    table_id: str,
    page_number: int,
    *,
    supported: list[str] | None = None,
    suppressed: list[str] | None = None,
    retained: list[str] | None = None,
) -> StructureProperty:
    supported = sorted(supported or [])
    suppressed = sorted(suppressed or [])
    retained = sorted(retained or [])
    disposition = (
        "suppressed_supported_spans"
        if suppressed
        else "retained_ambiguous_support"
        if supported
        else "retained_unsupported_overlap"
        if retained
        else "no_overlap"
    )
    return StructureProperty(
        key="table_span_suppression_v1",
        value=json.dumps({
            "counts": {
                "retained_overlap": len(retained),
                "supported": len(supported),
                "suppressed": len(suppressed),
            },
            "disposition": disposition,
            "page_number": page_number,
            "retained_overlap_span_ids": retained,
            "supported_span_ids": supported,
            "suppressed_span_ids": suppressed,
            "table_id": table_id,
        }),
    )


def _table_with_cell_provenance(text: str, source_ids: list[str]) -> DocumentElement:
    element = _table_element((0, 0, 100, 100))
    table = element.structure.table
    assert table is not None
    cell = table.cells[0]
    updated_cell = cell.model_copy(
        update={
            "text": text,
            "fragments": [cell.fragments[0].model_copy(update={"source_item_ids": source_ids})],
        }
    )
    ledger = _suppression_property(element.element_id, 1)
    return element.model_copy(
        update={
            "structure": element.structure.model_copy(
                update={
                    "table": table.model_copy(update={"cells": [updated_cell]}),
                    "properties": [ledger],
                }
            )
        }
    )


def _native_test_word(
    text: str,
    coordinates: tuple[float, float, float, float],
    *,
    page_number: int = 1,
) -> SourceItem:
    return make_source_item(
        document_id="document-id",
        page_number=page_number,
        kind="word",
        coordinate_frame="source_page",
        coordinates=coordinates,
        text=text,
    )


def _native_test_span(
    text: str,
    coordinates: tuple[float, float, float, float],
    *,
    page_number: int = 1,
) -> SourceItem:
    return make_source_item(
        document_id="document-id",
        page_number=page_number,
        kind="span",
        coordinate_frame="source_page",
        coordinates=coordinates,
        text=text,
    )


def test_source_catalog_accepts_complete_multiline_cell_in_geometry_reading_order() -> None:
    alpha = _native_test_word("Alpha", (0, 0, 10, 10))
    beta = _native_test_word("beta", (20, 0, 30, 10))
    gamma = _native_test_word("Gamma", (0, 20, 12, 30))
    words = [alpha, beta, gamma]
    element = _table_with_cell_provenance(
        "Alpha beta\nGamma",
        [gamma.source_item_id, beta.source_item_id, alpha.source_item_id],
    )

    validate_source_catalog([element], build_source_catalog("document-id", words))


def test_source_catalog_rejects_table_structural_ids_on_non_table_consumer() -> None:
    rule = make_source_item(
        document_id="document-id",
        page_number=1,
        kind="rule",
        coordinate_frame="detector_page",
        coordinates=(0, 0, 10, 0),
        text=None,
    )
    element = _table_element((0, 0, 100, 100)).model_copy(
        update={
            "element_type": "paragraph",
            "content": "Paragraph",
            "format": "text",
            "structure": ElementStructure(
                properties=[
                    StructureProperty(
                        key="table_structural_source_item_ids",
                        value=json.dumps([rule.source_item_id]),
                    )
                ]
            ),
        }
    )

    with pytest.raises(ValueError, match="require a table consumer"):
        validate_source_catalog([element], build_source_catalog("document-id", [rule]))


def test_legacy_catalog_explicitly_preserves_pre_ledger_candidate_compatibility() -> None:
    word = _native_test_word("Alpha", (0, 0, 10, 10))
    element = _table_with_cell_provenance("Alpha", [word.source_item_id])
    element = element.model_copy(
        update={"structure": element.structure.model_copy(update={"properties": []})}
    )
    catalog = build_source_catalog("document-id", [word]).model_copy(
        update={"migrated_from_schema_version": "1.0.0"}
    )

    validate_source_catalog([element], catalog)


def test_current_candidate_table_requires_page_suppression_ledger() -> None:
    word = _native_test_word("Alpha", (0, 0, 10, 10))
    element = _table_with_cell_provenance("Alpha", [word.source_item_id])
    element = element.model_copy(
        update={"structure": element.structure.model_copy(update={"properties": []})}
    )

    with pytest.raises(ValueError, match="require exactly one.*missing=.*table-1"):
        validate_source_catalog([element], build_source_catalog("document-id", [word]))


def test_current_candidate_table_accepts_explicit_empty_no_overlap_ledger() -> None:
    word = _native_test_word("Alpha", (0, 0, 10, 10))
    element = _table_with_cell_provenance("Alpha", [word.source_item_id])

    validate_source_catalog([element], build_source_catalog("document-id", [word]))


@pytest.mark.parametrize("span_bbox", [(1, 1, 9, 9), (110, 110, 120, 120)])
def test_current_candidate_table_rejects_incomplete_or_extra_span_disposition(
    span_bbox: tuple[float, float, float, float],
) -> None:
    word = _native_test_word("Alpha", (0, 0, 10, 10))
    span = _native_test_span("Alpha", span_bbox)
    element = _table_with_cell_provenance("Alpha", [word.source_item_id])
    if span_bbox[0] > 100:
        element = element.model_copy(
            update={
                "structure": element.structure.model_copy(
                    update={
                        "properties": [
                            _suppression_property(element.element_id, 1, retained=[span.source_item_id])
                        ]
                    }
                )
            }
        )

    expected = "missing=" if span_bbox[0] < 100 else "extra="
    with pytest.raises(ValueError, match=f"completely account.*{expected}"):
        validate_source_catalog([element], build_source_catalog("document-id", [word, span]))


@pytest.mark.parametrize(
    ("second_disposition", "detail"),
    [("suppressed", "double-consumed"), ("retained", "exactly one")],
)
def test_current_candidate_tables_reject_double_or_conflicting_span_dispositions(
    second_disposition: str, detail: str
) -> None:
    first_word = _native_test_word("Alpha", (0, 0, 10, 10))
    second_word = _native_test_word("Alpha", (0, 0, 11, 10))
    span = _native_test_span("Alpha", (1, 1, 9, 9))
    first = _table_with_cell_provenance("Alpha", [first_word.source_item_id])
    first = first.model_copy(
        update={
            "structure": first.structure.model_copy(
                update={
                    "properties": [
                        _suppression_property(first.element_id, 1, suppressed=[span.source_item_id])
                    ]
                }
            )
        }
    )
    second = _table_with_cell_provenance("Alpha", [second_word.source_item_id]).model_copy(
        update={"element_id": "table-2", "order": 1}
    )
    second = second.model_copy(
        update={
            "structure": second.structure.model_copy(
                update={
                    "properties": [
                        _suppression_property(
                            second.element_id,
                            1,
                            suppressed=[span.source_item_id] if second_disposition == "suppressed" else None,
                            retained=[span.source_item_id] if second_disposition == "retained" else None,
                        )
                    ]
                }
            )
        }
    )

    with pytest.raises(ValueError, match=detail):
        validate_source_catalog(
            [first, second],
            build_source_catalog("document-id", [first_word, second_word, span]),
        )


@pytest.mark.parametrize(
    ("case", "detail"),
    [
        ("valid", None),
        ("extra", "extra="),
        ("incomplete", "missing="),
        ("double", "ID lists must be disjoint"),
    ],
)
def test_current_candidate_nonempty_multipage_table_span_accounting(case: str, detail: str | None) -> None:
    element = _table_element((0, 0, 100, 100))
    first_fragment = element.fragments[0]
    second_fragment = first_fragment.model_copy(update={"page_number": 2})
    first_word = _native_test_word("Alpha", (10, 10, 30, 20), page_number=1)
    second_word = _native_test_word("Beta", (10, 10, 30, 20), page_number=2)
    first_span = _native_test_span("Alpha", (10, 10, 30, 20), page_number=1)
    second_span = _native_test_span("Beta", (10, 10, 30, 20), page_number=2)
    extra_span = _native_test_span("Outside", (110, 110, 130, 120), page_number=2)
    table = element.structure.table
    assert table is not None
    cell = table.cells[0].model_copy(
        update={
            "text": "Alpha Beta",
            "fragments": [
                first_fragment.model_copy(update={"source_item_ids": [first_word.source_item_id]}),
                second_fragment.model_copy(update={"source_item_ids": [second_word.source_item_id]}),
            ],
        }
    )
    page_two_suppressed = [] if case == "incomplete" else [second_span.source_item_id]
    page_two_ledger = _suppression_property(
        element.element_id,
        2,
        suppressed=page_two_suppressed,
        retained=[extra_span.source_item_id] if case == "extra" else None,
    )
    if case == "double":
        payload = json.loads(page_two_ledger.value)
        payload["supported_span_ids"] = [second_span.source_item_id]
        payload["counts"]["supported"] = 1
        page_two_ledger = StructureProperty(
            key=page_two_ledger.key,
            value=json.dumps(payload),
        )
    multipage = element.model_copy(
        update={
            "content": "| Alpha Beta |",
            "fragments": [first_fragment, second_fragment],
            "structure": element.structure.model_copy(
                update={
                    "table": table.model_copy(update={"cells": [cell]}),
                    "properties": [
                        _suppression_property(element.element_id, 1, suppressed=[first_span.source_item_id]),
                        page_two_ledger,
                    ],
                }
            ),
        }
    )
    catalog = build_source_catalog(
        "document-id",
        [first_word, second_word, first_span, second_span, extra_span],
    )

    if detail is None:
        validate_source_catalog([multipage], catalog)
    else:
        with pytest.raises(ValueError, match=detail):
            validate_source_catalog([multipage], catalog)


@pytest.mark.parametrize(
    ("cell_text", "word_specs", "source_indices"),
    [
        ("Alpha FABRICATED_NONCE", [("Alpha", (0, 0, 10, 10))], [0]),
        ("Alpha", [("Alpha", (0, 0, 10, 10)), ("extra", (20, 0, 30, 10))], [0, 1]),
        ("Alpha beta", [("Alpha", (20, 0, 30, 10)), ("beta", (0, 0, 10, 10))], [0, 1]),
        ("Alpha", [("wrong", (0, 0, 10, 10))], [0]),
        ("Alpha", [], []),
    ],
)
def test_source_catalog_rejects_incomplete_or_wrong_cell_word_reconstruction(
    cell_text: str,
    word_specs: list[tuple[str, tuple[float, float, float, float]]],
    source_indices: list[int],
) -> None:
    words = [_native_test_word(text, bbox) for text, bbox in word_specs]
    element = _table_with_cell_provenance(
        cell_text,
        [words[index].source_item_id for index in source_indices],
    )

    with pytest.raises(ValueError, match="require native word|reconstruct complete"):
        validate_source_catalog(
            [element],
            build_source_catalog("document-id", words),
        )


def test_source_catalog_rejects_empty_cell_with_native_word() -> None:
    word = _native_test_word("Alpha", (0, 0, 10, 10))
    element = _table_with_cell_provenance("", [word.source_item_id])

    with pytest.raises(ValueError, match="empty table cells"):
        validate_source_catalog(
            [element],
            build_source_catalog("document-id", [word]),
        )


@pytest.mark.parametrize(
    "counts",
    [
        [],
        {"retained_overlap": False, "supported": 0, "suppressed": 0},
        {"retained_overlap": 0.0, "supported": 0, "suppressed": 0},
        {"retained_overlap": "0", "supported": 0, "suppressed": 0},
        {"retained_overlap": 0, "supported": 0},
        {"retained_overlap": 0, "supported": 0, "suppressed": 0, "extra": 0},
        {"retained_overlap": -1, "supported": 0, "suppressed": 0},
    ],
)
def test_source_catalog_rejects_noncanonical_table_suppression_counts(
    counts: object,
) -> None:
    word = _native_test_word("Value", (0, 0, 10, 10))
    element = _table_with_cell_provenance("Value", [word.source_item_id])
    ledger = StructureProperty(
        key="table_span_suppression_v1",
        value=json.dumps({
            "counts": counts,
            "disposition": "no_overlap",
            "page_number": 1,
            "retained_overlap_span_ids": [],
            "supported_span_ids": [],
            "suppressed_span_ids": [],
            "table_id": element.element_id,
        }),
    )
    element = element.model_copy(
        update={"structure": element.structure.model_copy(update={"properties": [ledger]})}
    )

    with pytest.raises(ValueError, match="counts must contain exactly non-negative integer values"):
        validate_source_catalog([element], build_source_catalog("document-id", [word]))


def test_rotated_table_suppresses_cell_text_without_double_rotating_boxes(tmp_path: Path) -> None:
    pdf_path = tmp_path / "rotated-table.pdf"
    create_rotated_table_pdf(pdf_path)

    elements = extract_document_elements(pdf_path)

    tables = [element for element in elements if element.element_type == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert table.fragments[0].page_width == 400
    assert table.fragments[0].page_height == 300
    assert table.structure.table is not None
    assert all(
        fragment.page_width == 400 and fragment.page_height == 300
        for cell in table.structure.table.cells
        for fragment in cell.fragments
    )
    residual_text = " ".join(element.content for element in elements if element.element_type != "table")
    assert residual_text == "Body text outside the table."
    assert {cell.text for cell in table.structure.table.cells} == {"Name", "Value", "Item", "7"}


def _all_source_item_ids(elements: tuple[DocumentElement, ...]) -> set[str]:
    result = {
        source_item_id
        for element in elements
        for fragment in element.fragments
        for source_item_id in fragment.source_item_ids
    }
    result.update(
        source_item_id
        for element in elements
        if element.structure.table is not None
        for cell in element.structure.table.cells
        for fragment in cell.fragments
        for source_item_id in fragment.source_item_ids
    )
    return result


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_candidate_source_catalog_is_closed_lifetime_repeatable_and_rotation_stable(
    tmp_path: Path, rotation: int
) -> None:
    baseline_path = tmp_path / "source-catalog-baseline.pdf"
    rotated_path = tmp_path / f"source-catalog-{rotation}.pdf"
    create_rotated_table_pdf(baseline_path)
    with pymupdf.open(baseline_path) as document:
        document[0].set_rotation(rotation)
        document.save(rotated_path)

    first = extract_document_with_catalog(rotated_path)
    repeated = extract_document_with_catalog(rotated_path)
    baseline = extract_document_with_catalog(baseline_path)

    assert first == repeated
    assert first.source_catalog.sha256 == repeated.source_catalog.sha256
    assert _all_source_item_ids(first.elements)
    assert _all_source_item_ids(first.elements) <= {
        item.source_item_id for item in first.source_catalog.items
    }
    assert all(
        item.source_item_id.startswith(f"pymupdf:d{first.source_catalog.document_id}:")
        for item in first.source_catalog.items
    )
    # Saving a rotated copy changes the document hash, so document-scoped IDs
    # intentionally differ. The canonical native observations remain equivalent.
    native_observations = lambda catalog: {  # noqa: E731
        (
            item.page_number,
            item.kind,
            item.canonical_x0,
            item.canonical_y0,
            item.canonical_x1,
            item.canonical_y1,
            item.text,
            item.style_sha256,
            item.duplicate_index,
        )
        for item in catalog.items
    }
    assert native_observations(first.source_catalog) == native_observations(baseline.source_catalog)
    assert all(fragment.source_item_ids for element in first.elements for fragment in element.fragments)
    assert all(
        not fragment.source_item_ids or cell.text.strip()
        for element in first.elements
        if element.structure.table is not None
        for cell in element.structure.table.cells
        for fragment in cell.fragments
    )


def test_candidate_source_catalog_parquet_roundtrip_and_dangling_ids_fail_closed(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "catalog-roundtrip.pdf"
    output_path = tmp_path / "catalog-roundtrip.parquet"
    create_rotated_table_pdf(pdf_path)
    extracted = extract_document_with_catalog(pdf_path)

    catalog_path = write_document_with_source_catalog(
        list(extracted.elements), extracted.source_catalog, output_path
    )
    elements, catalog = read_document_with_source_catalog(output_path)

    assert catalog_path == source_catalog_path(output_path)
    assert elements == list(extracted.elements)
    assert catalog == extracted.source_catalog
    dangling = elements[0].model_copy(
        update={
            "fragments": [
                elements[0]
                .fragments[0]
                .model_copy(update={"source_item_ids": ["pymupdf:p000001:span:dangling"]})
            ]
        }
    )
    with pytest.raises(ValueError, match="dangling source_item_id"):
        validate_source_catalog([dangling, *elements[1:]], catalog)

    table_index, table_element = next(
        (index, element) for index, element in enumerate(elements) if element.structure.table is not None
    )
    table = table_element.structure.table
    assert table is not None
    cited_cells = [cell for cell in table.cells if cell.fragments[0].source_item_ids]
    assert len(cited_cells) >= 2
    shared_id = cited_cells[0].fragments[0].source_item_ids[0]
    duplicated_cells = [
        cell.model_copy(
            update={
                "text": cited_cells[0].text,
                "fragments": [cell.fragments[0].model_copy(update={"source_item_ids": [shared_id]})],
            }
        )
        if cell == cited_cells[1]
        else cell
        for cell in table.cells
    ]
    duplicated = table_element.model_copy(
        update={
            "structure": table_element.structure.model_copy(
                update={"table": table.model_copy(update={"cells": duplicated_cells})}
            )
        }
    )
    duplicated_elements = [*elements]
    duplicated_elements[table_index] = duplicated
    with pytest.raises(ValueError, match="more than one logical table cell"):
        validate_source_catalog(duplicated_elements, catalog)

    empty_cells = [
        cell.model_copy(update={"text": ""}) if cell == cited_cells[0] else cell for cell in table.cells
    ]
    empty = table_element.model_copy(
        update={
            "structure": table_element.structure.model_copy(
                update={"table": table.model_copy(update={"cells": empty_cells})}
            )
        }
    )
    empty_elements = [*elements]
    empty_elements[table_index] = empty
    with pytest.raises(ValueError, match="empty table cells"):
        validate_source_catalog(empty_elements, catalog)


def test_grounded_table_span_suppression_requires_complete_text_and_records_ledger() -> None:
    document_id = "grounded-document"
    word = make_source_item(
        document_id=document_id,
        page_number=1,
        kind="word",
        coordinate_frame="source_page",
        coordinates=(80.0, 20.0, 130.0, 35.0),
        text="table",
    )
    support = (_TableWordSupport("page-1-table-1", 1, (word,)),)
    ledger = _TableSpanLedger("page-1-table-1", 1, set(), set(), set())
    source_items = [word]
    blocks = _text_blocks_from_pymupdf_dict(
        {"blocks": [_raw_text_block("table", (80, 20, 130, 35))]},
        page_number=1,
        page_width=300,
        page_height=400,
        rotation=0,
        table_bboxes=[BoundingBox(x0=10, y0=10, x1=200, y1=50)],
        document_id=document_id,
        source_items=source_items,
        table_support=support,
        suppression_ledgers={ledger.table_id: ledger},
    )

    assert blocks == []
    span_ids = [item.source_item_id for item in source_items if item.kind == "span"]
    assert sorted(ledger.suppressed_span_ids) == span_ids
    assert not ledger.supported_span_ids
    assert not ledger.retained_overlap_span_ids


def test_table_span_text_mismatch_and_bbox_only_overlap_retain_native_text() -> None:
    document_id = "retained-document"
    word = make_source_item(
        document_id=document_id,
        page_number=1,
        kind="word",
        coordinate_frame="source_page",
        coordinates=(80.0, 20.0, 130.0, 35.0),
        text="table",
    )
    ledger = _TableSpanLedger("page-1-table-1", 1, set(), set(), set())
    source_items = [word]
    blocks = _text_blocks_from_pymupdf_dict(
        {"blocks": [_raw_text_block("header table", (20, 20, 130, 35))]},
        page_number=1,
        page_width=300,
        page_height=400,
        rotation=0,
        table_bboxes=[BoundingBox(x0=10, y0=10, x1=200, y1=50)],
        document_id=document_id,
        source_items=source_items,
        table_support=(_TableWordSupport(ledger.table_id, 1, (word,)),),
        suppression_ledgers={ledger.table_id: ledger},
    )
    bbox_only = _text_blocks_from_pymupdf_dict(
        {"blocks": [_raw_text_block("bbox only", (20, 20, 130, 35))]},
        page_number=1,
        page_width=300,
        page_height=400,
        rotation=0,
        table_bboxes=[BoundingBox(x0=10, y0=10, x1=200, y1=50)],
    )

    assert [block.text for block in blocks] == ["header table"]
    assert [block.text for block in bbox_only] == ["bbox only"]
    assert ledger.retained_overlap_span_ids
    assert not ledger.suppressed_span_ids


def test_bbox_only_table_overlap_does_not_split_native_text() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            _raw_span("before", (10, 20, 60, 35)),
                            _raw_span("table", (80, 20, 130, 35)),
                            _raw_span("after", (150, 20, 200, 35)),
                        ]
                    }
                ],
            }
        ]
    }

    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=300,
        page_height=400,
        rotation=0,
        table_bboxes=[BoundingBox(x0=75, y0=15, x1=135, y1=40)],
    )

    assert [(item.text, item.bbox) for item in blocks] == [
        ("before table after", (10, 20, 200, 35)),
    ]


def test_bbox_only_overlap_retains_nonempty_text_and_ignores_whitespace() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            _raw_span("table", (10, 20, 60, 35)),
                            _raw_span("                  ", (80, 20, 112, 35)),
                        ]
                    }
                ],
            }
        ]
    }

    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=7,
        page_width=612,
        page_height=792,
        rotation=0,
        table_bboxes=[BoundingBox(x0=5, y0=15, x1=65, y1=40)],
    )

    assert [block.text for block in blocks] == ["table"]


def test_bbox_only_overlaps_do_not_split_consecutive_lines() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            _raw_span("table-a", (10, 20, 40, 30)),
                            _raw_span("B", (50, 20, 60, 30)),
                        ]
                    },
                    {
                        "spans": [
                            _raw_span("C", (10, 40, 20, 50)),
                            _raw_span("table-b", (70, 40, 100, 50)),
                        ]
                    },
                ],
            }
        ]
    }

    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=300,
        page_height=400,
        rotation=0,
        table_bboxes=[
            BoundingBox(x0=5, y0=15, x1=45, y1=35),
            BoundingBox(x0=65, y0=35, x1=105, y1=55),
        ],
    )

    assert [block.text for block in blocks] == ["table-a B C table-b"]


def test_atomic_insertion_does_not_reorder_an_already_semantic_text_page() -> None:
    payload = {
        "blocks": [
            _raw_text_block("Left margin marker", (40, 50, 160, 65)),
            _raw_text_block("Right margin marker", (520, 44, 560, 65)),
        ]
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )
    text_elements = _semantic_elements(
        tuple(SemanticBlock(kind="paragraph", text=item.text, source_blocks=(item,)) for item in blocks),
        "document-id",
        "test",
    )

    result = _insert_tables_in_reading_order(list(reversed(text_elements)), [])

    assert [element.content for element in result] == ["Right margin marker", "Left margin marker"]


def test_table_insertion_preserves_column_aware_order() -> None:
    payload = {
        "blocks": [
            _raw_text_block("left top.", (40, 100, 270, 125)),
            _raw_text_block("left bottom.", (40, 500, 270, 525)),
            _raw_text_block("right top.", (330, 100, 560, 125)),
            _raw_text_block("right bottom.", (330, 500, 560, 525)),
        ]
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )
    text_elements = _semantic_elements(
        classify_semantic_blocks(blocks, body_font_size=10), "document-id", "test"
    )

    result = _insert_tables_in_reading_order(text_elements, [_table_element((40, 300, 270, 400))])

    assert [element.content for element in result] == [
        "left top.",
        "| Value |\n| --- |",
        "left bottom.",
        "right top.",
        "right bottom.",
    ]


@pytest.mark.parametrize("column_splits", [None, (300,)])
def test_wide_atomic_insertion_restores_bottom_margin_y_order_and_inline_ltr(
    column_splits: tuple[float, ...] | None,
) -> None:
    payload = {
        "blocks": [
            _raw_text_block("left body.", (40, 100, 270, 125)),
            _raw_text_block("left lower.", (40, 500, 270, 525)),
            _raw_text_block("right body.", (330, 100, 560, 125)),
            _raw_text_block("right lower.", (330, 500, 560, 525)),
            _raw_text_block("Continued →", (470, 730, 560, 744), size=8),
            _raw_text_block("42", (40, 770, 60, 784), size=8),
            _raw_text_block("Running footer", (420, 769, 520, 785), size=8),
        ]
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )
    text_elements = _semantic_elements(
        tuple(SemanticBlock(kind="paragraph", text=item.text, source_blocks=(item,)) for item in blocks),
        "document-id",
        "test",
    )
    table = _table_element((20, 300, 580, 400))
    expected_content = [element.content for element in [*text_elements, table]]

    result = _insert_tables_in_reading_order(
        text_elements,
        [table],
        column_splits=column_splits,
    )

    assert sorted(element.content for element in result) == sorted(expected_content)
    assert [element.content for element in result[-3:]] == [
        "Continued →",
        "42",
        "Running footer",
    ]


def test_atomic_insertion_uses_pre_suppression_column_plan() -> None:
    payload = {
        "blocks": [
            _raw_text_block("left top.", (40, 100, 270, 125)),
            _raw_text_block("left bottom.", (40, 500, 270, 525)),
            _raw_text_block("right top.", (330, 100, 560, 125)),
            _raw_text_block("right bottom.", (330, 500, 560, 525)),
            _raw_text_block("table first", (250, 300, 350, 325)),
            _raw_text_block("table second", (250, 340, 350, 365)),
        ]
    }
    atomic_bbox = BoundingBox(x0=240, y0=280, x1=360, y1=400)
    layout_blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )
    splits = infer_page_column_splits(
        layout_blocks, atomic_bboxes=[(atomic_bbox.x0, atomic_bbox.y0, atomic_bbox.x1, atomic_bbox.y1)]
    )
    residual_blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[atomic_bbox],
    )
    text_elements = _semantic_elements(
        classify_semantic_blocks(residual_blocks, body_font_size=10), "document-id", "test"
    )
    table = _table_element((240, 280, 360, 400))

    result = _insert_tables_in_reading_order(text_elements, [table], column_splits=splits)

    assert splits == pytest.approx((300,))
    assert [element.content for element in result] == [
        "left top.",
        "right top.",
        "| Value |\n| --- |",
        "table first table second",
        "left bottom.",
        "right bottom.",
    ]


@pytest.mark.parametrize(("scale", "reverse_blocks"), [(1.0, False), (1.5, True)])
def test_encompassing_figure_keeps_same_row_panel_headings_with_their_leads(
    scale: float, reverse_blocks: bool
) -> None:
    raw_blocks = [
        _raw_text_block("c) Left panel", (80 * scale, 100 * scale, 270 * scale, 120 * scale)),
        _raw_text_block("d) Right panel", (330 * scale, 100 * scale, 520 * scale, 120 * scale)),
        _raw_text_block("Right panel lead text", (330 * scale, 130 * scale, 520 * scale, 150 * scale)),
        _raw_text_block("Left panel lead text", (80 * scale, 150 * scale, 270 * scale, 170 * scale)),
    ]
    if reverse_blocks:
        raw_blocks.reverse()
    blocks = _text_blocks_from_pymupdf_dict(
        {"blocks": raw_blocks},
        page_number=1,
        page_width=600 * scale,
        page_height=800 * scale,
        rotation=0,
        table_bboxes=[],
    )
    by_text = {block.text: block for block in blocks}
    semantic = (
        SemanticBlock(
            kind="heading",
            text="c) Left panel",
            source_blocks=(by_text["c) Left panel"],),
            heading_level=3,
        ),
        SemanticBlock(
            kind="heading",
            text="d) Right panel",
            source_blocks=(by_text["d) Right panel"],),
            heading_level=3,
        ),
        SemanticBlock(
            kind="paragraph",
            text="Right panel lead text",
            source_blocks=(by_text["Right panel lead text"],),
        ),
        SemanticBlock(
            kind="paragraph",
            text="Left panel lead text",
            source_blocks=(by_text["Left panel lead text"],),
        ),
    )
    text_elements = _semantic_elements(semantic, "document-id", "test")
    figure = _table_element(
        (50 * scale, 80 * scale, 550 * scale, 500 * scale),
        page_width=600 * scale,
        page_height=800 * scale,
    ).model_copy(
        update={
            "element_id": "figure-1",
            "element_type": "figure",
            "content": "",
            "format": "text",
            "structure": ElementStructure(
                figure=FigureStructure(
                    asset_path="pdf://page/1/vector/test",
                    sha256="0" * 64,
                    width=500,
                    height=420,
                )
            ),
        }
    )

    result = _insert_tables_in_reading_order(text_elements, [figure], column_splits=())
    interleaved = _order_encompassed_figure_panel_leads(
        [text_elements[0], text_elements[1], figure, text_elements[2], text_elements[3]],
        0,
    )

    expected = [
        ("figure", ""),
        ("heading", "c) Left panel"),
        ("paragraph", "Left panel lead text"),
        ("heading", "d) Right panel"),
        ("paragraph", "Right panel lead text"),
    ]
    assert [(element.element_type, element.content) for element in result] == expected
    assert [(element.element_type, element.content) for element in interleaved] == expected


def test_table_insertion_preserves_three_column_order() -> None:
    payload = {
        "blocks": [
            _raw_text_block("left top.", (40, 100, 270, 125)),
            _raw_text_block("left bottom.", (40, 500, 270, 525)),
            _raw_text_block("middle top.", (335, 100, 565, 125)),
            _raw_text_block("middle bottom.", (335, 500, 565, 525)),
            _raw_text_block("right top.", (630, 100, 860, 125)),
            _raw_text_block("right bottom.", (630, 500, 860, 525)),
        ]
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=900,
        page_height=800,
        rotation=0,
        table_bboxes=[],
    )
    text_elements = _semantic_elements(
        classify_semantic_blocks(blocks, body_font_size=10), "document-id", "test"
    )
    table = _table_element((335, 300, 565, 400), page_width=900)

    result = _insert_tables_in_reading_order(text_elements, [table])

    assert [element.content for element in result] == [
        "left top.",
        "left bottom.",
        "middle top.",
        "| Value |\n| --- |",
        "middle bottom.",
        "right top.",
        "right bottom.",
    ]


def test_same_baseline_recurring_header_and_page_label_consolidate_atomically() -> None:
    blocks = []
    for page_number in (9, 10, 11):
        blocks.extend(
            _text_blocks_from_pymupdf_dict(
                {
                    "blocks": [
                        _raw_text_block("Synthetic Form (2025)", (36, 35, 150, 46), size=8),
                        _raw_text_block(f"Page {page_number}", (530, 34, 578, 47), size=8),
                        _raw_text_block(f"Unique Part {page_number}", (40, 100, 180, 115), size=12),
                    ]
                },
                page_number=page_number,
                page_width=612,
                page_height=792,
                rotation=0,
                table_bboxes=[],
            )
        )

    consolidated = _consolidate_compound_margin_lines(blocks, min_pages=2)
    elements = _semantic_elements(
        classify_semantic_blocks(consolidated, body_font_size=10, recurring_min_pages=2),
        "document-id",
        "test",
    )

    headers = [element for element in elements if element.element_type == "header"]
    assert [element.content for element in headers] == [
        "Synthetic Form (2025) Page 9",
        "Synthetic Form (2025) Page 10",
        "Synthetic Form (2025) Page 11",
    ]
    assert all(element.structure.paragraph is not None for element in headers)
    assert all(
        element.structure.paragraph.role == "running_header"
        for element in headers
        if element.structure.paragraph
    )
    assert all(not element.include_in_output for element in headers)
    assert not any(
        element.structure.paragraph is not None and element.structure.paragraph.role == "page_number"
        for element in elements
    )
    assert [element.content for element in elements if element.include_in_output] == [
        "Unique Part 9",
        "Unique Part 10",
        "Unique Part 11",
    ]


def test_semantic_margin_consolidation_rejoins_split_native_block() -> None:
    blocks = []
    for page_number in (9, 10, 11):
        blocks.extend(
            _text_blocks_from_pymupdf_dict(
                {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [
                                {"spans": [_raw_span("Synthetic Form (2025)", (36, 35, 150, 46), size=8)]},
                                {"spans": [_raw_span(f"Page {page_number}", (530, 34, 578, 47), size=8)]},
                                {
                                    "spans": [
                                        _raw_span(f"Unique Part {page_number}", (40, 100, 180, 115), size=12)
                                    ]
                                },
                            ],
                        }
                    ]
                },
                page_number=page_number,
                page_width=612,
                page_height=792,
                rotation=0,
                table_bboxes=[],
            )
        )

    classified = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)
    consolidated = _consolidate_compound_margin_semantics(classified)

    assert [(item.kind, item.margin_role, item.text) for item in consolidated if item.margin_role] == [
        ("recurring_margin", "running_header", "Synthetic Form (2025) Page 9"),
        ("recurring_margin", "running_header", "Synthetic Form (2025) Page 10"),
        ("recurring_margin", "running_header", "Synthetic Form (2025) Page 11"),
    ]
    assert [item.text for item in consolidated if item.margin_role is None] == [
        "Unique Part 9",
        "Unique Part 10",
        "Unique Part 11",
    ]


def test_compound_margin_consolidation_rejects_other_baselines_and_unique_headings() -> None:
    different_baselines = []
    unique_headings = []
    for page_number in (1, 2, 3):
        different_baselines.extend(
            _text_blocks_from_pymupdf_dict(
                {
                    "blocks": [
                        _raw_text_block("Recurring Report", (36, 10, 150, 21), size=8),
                        _raw_text_block(f"Page {page_number}", (530, 35, 578, 48), size=8),
                    ]
                },
                page_number=page_number,
                page_width=612,
                page_height=792,
                rotation=0,
                table_bboxes=[],
            )
        )
        unique_headings.extend(
            _text_blocks_from_pymupdf_dict(
                {
                    "blocks": [
                        _raw_text_block(f"Unique Section {page_number}", (36, 35, 170, 46), size=8),
                        _raw_text_block(f"Page {page_number}", (530, 34, 578, 47), size=8),
                    ]
                },
                page_number=page_number,
                page_width=612,
                page_height=792,
                rotation=0,
                table_bboxes=[],
            )
        )

    assert _consolidate_compound_margin_lines(different_baselines, min_pages=2) == different_baselines
    assert _consolidate_compound_margin_lines(unique_headings, min_pages=2) == unique_headings


def test_margin_semantics_map_to_exact_element_types_roles_and_output_flags() -> None:
    blocks = []
    for page, printed in zip(range(10, 13), range(201, 204), strict=True):
        blocks.extend(
            _text_blocks_from_pymupdf_dict(
                {
                    "blocks": [
                        _raw_text_block("Synthetic Report", (40, 10, 200, 22), size=8),
                        _raw_text_block(str(printed), (280, 780, 320, 792), size=8),
                        _raw_text_block(f"Canonical body {page}.", (40, 200, 300, 220)),
                    ]
                },
                page_number=page,
                page_width=600,
                page_height=800,
                rotation=0,
                table_bboxes=[],
            )
        )

    elements = _semantic_elements(
        classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2),
        "document-id",
        "test",
    )

    assert [
        (element.element_type, element.structure.paragraph.role, element.include_in_output)
        for element in elements
        if element.structure.paragraph is not None
        and element.structure.paragraph.role in {"running_header", "page_number"}
    ] == [
        ("header", "running_header", False),
        ("footer", "page_number", False),
    ] * 3
    assert [element.content for element in elements if element.include_in_output] == [
        "Canonical body 10.",
        "Canonical body 11.",
        "Canonical body 12.",
    ]


def test_engine_fallback_never_suppresses_continuation_markers() -> None:
    blocks = tuple(
        _text_blocks_from_pymupdf_dict(
            {"blocks": [_raw_text_block("Continued →", (40, 770, 120, 785), size=8)]},
            page_number=page,
            page_width=600,
            page_height=800,
            rotation=0,
            table_bboxes=[],
        )[0]
        for page in (1, 2)
    )
    elements = _semantic_elements(
        tuple(SemanticBlock(kind="paragraph", text=item.text, source_blocks=(item,)) for item in blocks),
        "document-id",
        "test",
    )

    result = _classify_recurring_margins(elements, selected_page_count=2)

    assert all(element.structure.paragraph is not None for element in result)
    assert [
        (
            element.element_type,
            element.structure.paragraph.role if element.structure.paragraph is not None else None,
            element.include_in_output,
        )
        for element in result
    ] == [
        ("paragraph", "body", True),
        ("paragraph", "body", True),
    ]


def test_recurring_margin_count_requires_distinct_pages() -> None:
    blocks = [
        _text_blocks_from_pymupdf_dict(
            {"blocks": [_raw_text_block("Page 1", (40, 10 + offset, 100, 20 + offset))]},
            page_number=1,
            page_width=600,
            page_height=800,
            rotation=0,
            table_bboxes=[],
        )[0]
        for offset in (0, 5)
    ]
    elements = _semantic_elements(
        tuple(SemanticBlock(kind="paragraph", text=item.text, source_blocks=(item,)) for item in blocks),
        "document-id",
        "test",
    )

    result = _classify_recurring_margins(elements, selected_page_count=2)

    assert all(element.element_type == "paragraph" for element in result)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_recurring_margin_fallback_uses_native_orientation(rotation: int) -> None:
    blocks = tuple(
        _text_blocks_from_pymupdf_dict(
            {"blocks": [_raw_text_block(f"Page {page_number}", (40, 10, 100, 20))]},
            page_number=page_number,
            page_width=600,
            page_height=800,
            rotation=rotation,
            table_bboxes=[],
        )[0]
        for page_number in (1, 2)
    )
    elements = _semantic_elements(
        tuple(SemanticBlock(kind="paragraph", text=item.text, source_blocks=(item,)) for item in blocks),
        "document-id",
        "test",
    )

    result = _classify_recurring_margins(
        elements,
        selected_page_count=2,
        page_rotations={1: rotation, 2: rotation},
    )

    assert [element.element_type for element in result] == ["header", "header"]
    assert [element.structure.paragraph.role for element in result if element.structure.paragraph] == [
        "running_header",
        "running_header",
    ]


def test_malformed_text_payload_and_non_finite_dimensions_fail_fast() -> None:
    def parse(payload: dict[str, object], *, page_width: float = 600) -> None:
        _text_blocks_from_pymupdf_dict(
            payload,
            page_number=1,
            page_width=page_width,
            page_height=800,
            rotation=0,
            table_bboxes=[],
        )

    with pytest.raises(TypeError, match="blocks sequence"):
        parse({"blocks": "not-blocks"})
    with pytest.raises(TypeError, match="lines must be a sequence"):
        parse({"blocks": [{"type": 0}]})
    malformed_span = {
        "type": 0,
        "lines": [{"spans": [_raw_span(7, (10, 10, 30, 20))]}],
    }
    with pytest.raises(TypeError, match="span text must be a string"):
        parse({"blocks": [malformed_span]})
    with pytest.raises(ValueError, match="dimensions must be finite"):
        parse({"blocks": []}, page_width=float("nan"))
    bool_coordinate = _raw_span("invalid", (10, 10, 30, 20))
    bool_coordinate["bbox"] = (True, 10, 30, 20)
    with pytest.raises(TypeError, match="numeric and not boolean"):
        parse({"blocks": [{"type": 0, "lines": [{"spans": [bool_coordinate]}]}]})
    bool_size = _raw_span("invalid", (10, 10, 30, 20))
    bool_size["size"] = True
    with pytest.raises(TypeError, match="numeric and not boolean"):
        parse({"blocks": [{"type": 0, "lines": [{"spans": [bool_size]}]}]})


def test_empty_page_selection_does_not_mean_all_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "selected-pages.pdf"
    create_two_column_pdf(pdf_path)

    assert extract_document_elements(pdf_path, pages=[]) == []


@pytest.mark.parametrize("invalid_pages", [[True], [1.0], ["1"]])
def test_page_selection_rejects_non_integer_runtime_values(tmp_path: Path, invalid_pages: object) -> None:
    pdf_path = tmp_path / "invalid-pages.pdf"
    create_two_column_pdf(pdf_path)

    with pytest.raises(TypeError, match="integers and not booleans"):
        extract_document_elements(pdf_path, pages=cast(list[int], invalid_pages))
