from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf

from app.pdf2md.engine import ExtractedDocument
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
    TableCell,
    TableStructure,
)
from app.pdf2md.source_catalog import SourceCatalog, SourceItem, make_source_item
from benchmarks.adapters import ADAPTER_REGISTRY, BenchmarkAdapter, get_adapter
from benchmarks.adapters.our_parser import OurParserAdapter, project_extracted_document
from benchmarks.adapters.pymupdf_text import PyMuPDFTextAdapter
from benchmarks.canonical import CanonicalProvenance, PageSize, RuntimeStats, semantic_hash

_SHA = "a" * 64


def _write_text_pdf(path: Path) -> None:
    with pymupdf.open() as document:
        first = document.new_page(width=300, height=400)
        first.insert_text((30, 50), "Synthetic first page")
        document.new_page(width=320, height=420)
        third = document.new_page(width=300, height=400)
        third.insert_text((30, 50), "Synthetic third page")
        document.save(path)


def _draw_table_pdf(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=300)
        xs = (30.0, 150.0, 270.0)
        ys = (50.0, 90.0, 130.0, 170.0)
        page.draw_rect((xs[0], ys[0], xs[-1], ys[-1]))
        for x in xs[1:-1]:
            page.draw_line((x, ys[0]), (x, ys[-1]))
        for y in ys[1:-1]:
            page.draw_line((xs[0], y), (xs[-1], y))
        for row, values in enumerate((("Item", "Count"), ("Alpha", "10"), ("Beta", "20"))):
            for column, value in enumerate(values):
                page.insert_text((xs[column] + 8, ys[row] + 25), value)
        document.save(path)


def _fragment(
    page: int,
    bbox: tuple[float, float, float, float],
    *,
    source_item_ids: tuple[str, ...] = (),
) -> PageFragment:
    return PageFragment(
        page_number=page,
        page_width=200,
        page_height=300,
        bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
        source_item_ids=list(source_item_ids),
    )


def _word_item(page: int, bbox: tuple[float, float, float, float], text: str) -> SourceItem:
    return make_source_item(
        document_id=_SHA,
        page_number=page,
        kind="word",
        coordinate_frame="source_page",
        coordinates=bbox,
        text=text,
    )


def _annotation() -> AnnotationMetadata:
    return AnnotationMetadata(
        stage="candidate",
        revision=1,
        annotator="synthetic",
        confidence=1,
        adjudication_status="unreviewed",
    )


def _empty_suppression_ledger(table_id: str, page_number: int) -> StructureProperty:
    return StructureProperty(
        key="table_span_suppression_v1",
        value=json.dumps({
            "counts": {"retained_overlap": 0, "supported": 0, "suppressed": 0},
            "disposition": "no_overlap",
            "page_number": page_number,
            "retained_overlap_span_ids": [],
            "supported_span_ids": [],
            "suppressed_span_ids": [],
            "table_id": table_id,
        }),
    )


def test_registry_is_typed_and_lazy() -> None:
    assert list(ADAPTER_REGISTRY) == ["our_parser", "pymupdf_text"]
    assert isinstance(get_adapter("our_parser"), BenchmarkAdapter)
    assert isinstance(get_adapter("pymupdf_text"), BenchmarkAdapter)


def test_pymupdf_native_baseline_emits_raw_pages_and_nullable_bbox(tmp_path: Path) -> None:
    pdf_path = tmp_path / "three-pages.pdf"
    _write_text_pdf(pdf_path)

    first = PyMuPDFTextAdapter().process(pdf_path)
    second = PyMuPDFTextAdapter().process(pdf_path)

    assert first.status == "success"
    assert len(first.pages) == 3
    assert first.pages[0].elements[0].bbox is None
    assert first.pages[0].elements[0].element_type == "paragraph"
    assert first.pages[1].markdown == ""
    assert first.pages[1].elements == []
    assert first.pages[1].page_size == PageSize(width=320, height=420, unit="point")
    assert first.pages[0].tables == first.pages[0].figures == []
    assert first.raw_records[1] == {
        "page_index": 1,
        "width": 320.0,
        "height": 420.0,
        "text": "",
    }
    assert [element.id for page in first.pages for element in page.elements] == [
        element.id for page in second.pages for element in page.elements
    ]
    assert first.pages[0].provenance == second.pages[0].provenance
    assert semantic_hash(first.pages) == semantic_hash(second.pages)


def test_our_parser_emits_every_source_page_with_stable_grounded_ids(tmp_path: Path) -> None:
    pdf_path = tmp_path / "three-pages.pdf"
    _write_text_pdf(pdf_path)

    first = OurParserAdapter().process(pdf_path)
    second = OurParserAdapter().process(pdf_path)

    assert first.status == "success"
    assert len(first.pages) == 3
    assert first.pages[1].elements == []
    assert first.pages[1].markdown == ""
    assert first.pages[0].bbox_origin == "top-left"
    assert first.pages[0].elements[0].bbox is not None
    assert first.pages[0].elements[0].confidence is None
    assert [element.id for page in first.pages for element in page.elements] == [
        element.id for page in second.pages for element in page.elements
    ]
    assert semantic_hash(first.pages) == semantic_hash(second.pages)


def test_our_parser_preserves_reliably_detected_simple_table(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    _draw_table_pdf(pdf_path)

    run = OurParserAdapter().process(pdf_path)

    assert run.status == "success"
    assert len(run.pages[0].tables) == 1
    table = run.pages[0].tables[0]
    assert (table.row_count, table.column_count) == (3, 2)
    assert table.markdown == "| Item | Count |\n| --- | --- |\n| Alpha | 10 |\n| Beta | 20 |"
    assert table.markdown in run.pages[0].markdown
    assert {(cell.row_index, cell.column_index, cell.text) for cell in table.cells} == {
        (0, 0, "Item"),
        (0, 1, "Count"),
        (1, 0, "Alpha"),
        (1, 1, "10"),
        (2, 0, "Beta"),
        (2, 1, "20"),
    }
    assert all(cell.confidence is None and cell.bbox is not None for cell in table.cells)


def test_model_projection_preserves_compact_html_spans_and_cross_page_geometry() -> None:
    html = (
        '<table><thead><tr><th colspan="2">Summary</th></tr></thead>'
        "<tbody><tr><td>A</td><td>10</td></tr></tbody></table>"
    )
    summary_word = _word_item(1, (10, 40, 190, 70), "Summary")
    label_word = _word_item(2, (10, 20, 100, 50), "A")
    value_word = _word_item(2, (100, 20, 190, 50), "10")
    source_items = tuple(sorted((summary_word, label_word, value_word), key=lambda item: item.source_item_id))
    cells = [
        TableCell(
            row_index=0,
            column_index=0,
            colspan=2,
            role="header",
            text="Summary",
            fragments=[
                _fragment(
                    1,
                    (10, 40, 190, 70),
                    source_item_ids=(summary_word.source_item_id,),
                )
            ],
        ),
        TableCell(
            row_index=1,
            column_index=0,
            role="body",
            text="A",
            fragments=[
                _fragment(
                    2,
                    (10, 20, 100, 50),
                    source_item_ids=(label_word.source_item_id,),
                )
            ],
        ),
        TableCell(
            row_index=1,
            column_index=1,
            role="body",
            text="10",
            fragments=[
                _fragment(
                    2,
                    (100, 20, 190, 50),
                    source_item_ids=(value_word.source_item_id,),
                )
            ],
        ),
    ]
    table = DocumentElement(
        document_id=_SHA,
        element_id="stable-table",
        order=0,
        element_type="table",
        content=html,
        format="html",
        fragments=[_fragment(1, (10, 40, 190, 70)), _fragment(2, (10, 20, 190, 50))],
        structure=ElementStructure(
            table=TableStructure(
                row_count=2,
                column_count=2,
                header_row_count=1,
                representation="html",
                classification_reasons=["spanning_cells"],
                cells=cells,
            ),
            properties=[
                _empty_suppression_ledger("stable-table", 1),
                _empty_suppression_ledger("stable-table", 2),
            ],
        ),
        annotation=_annotation(),
    )
    header = DocumentElement(
        document_id=_SHA,
        element_id="stable-header",
        order=1,
        element_type="header",
        content="Running header",
        format="text",
        include_in_output=False,
        fragments=[_fragment(1, (10, 5, 100, 15))],
        structure=ElementStructure(paragraph=ParagraphStructure(role="running_header")),
        annotation=_annotation(),
    )
    extracted = ExtractedDocument(
        elements=(table, header),
        source_catalog=SourceCatalog(document_id=_SHA, items=source_items),
    )
    provenance = CanonicalProvenance(
        tool_name="synthetic",
        tool_version="1",
        mode="model",
        input_sha256=_SHA,
        config_sha256=hashlib.sha256(b"model").hexdigest(),
    )

    pages = project_extracted_document(
        extracted,
        [PageSize(width=200, height=300, unit="point")] * 2,
        RuntimeStats(wall_seconds=0),
        provenance,
    )

    assert pages[0].markdown == html
    assert "Running header" not in pages[0].markdown
    assert pages[0].elements[1].include_in_output is False
    assert pages[0].tables[0].html == html
    assert pages[1].tables[0].html == html
    assert pages[0].tables[0].cells[0].colspan == 2
    assert pages[0].tables[0].cells[1].bbox is None
    assert pages[1].tables[0].cells[0].bbox is None
    assert pages[1].tables[0].bbox == (10, 20, 190, 50)
    assert pages[0].elements[0].id == pages[1].elements[0].id == "stable-table"


def test_malformed_pdf_returns_failed_run(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"not a PDF")

    for adapter in (PyMuPDFTextAdapter(), OurParserAdapter()):
        run = adapter.process(malformed)
        assert run.status == "failed"
        assert run.pages == []
        assert run.error is not None
        assert "FileDataError" in run.error
