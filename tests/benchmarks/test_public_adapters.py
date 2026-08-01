from __future__ import annotations

import json
import runpy
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pymupdf
import pytest

from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FootnoteStructure,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
    TableCell,
    TableStructure,
    TextStyleRun,
)
from app.pdf2md.source_catalog import SourceItemKind, build_source_catalog, make_source_item
from benchmarks.adapters.olmocr_bench import export_olmocr_bench
from benchmarks.adapters.omnidocbench import export_omnidocbench
from benchmarks.adapters.parsebench import (
    ParseBenchPageGeometry,
    export_parsebench_records,
    parsebench_page_geometries,
    parsebench_source_page_extents,
    project_parsebench_layout_pages,
    write_parsebench_jsonl,
)
from benchmarks.adapters.rd_tablebench import (
    UnsupportedTableRepresentationError,
    export_rd_tablebench,
    render_rd_tablebench_html,
)
from benchmarks.canonical import (
    CanonicalCell,
    CanonicalElement,
    CanonicalPage,
    CanonicalProvenance,
    CanonicalTable,
    PageSize,
    RuntimeStats,
)


def _page(*, html_table: bool = False) -> CanonicalPage:
    representation = (
        "<table><tr><th>都市</th><th>値</th></tr><tr><td>München</td><td>١٢</td></tr></table>"
        if html_table
        else "| 都市 | 値 |\n| --- | --- |\n| München | ١٢ |"
    )
    cells = [
        CanonicalCell(
            id=f"cell-{row}-{column}",
            row_index=row,
            column_index=column,
            text=text,
            bbox=(10 + 90 * column, 30 + 30 * row, 100 + 90 * column, 60 + 30 * row),
        )
        for row, values in enumerate((("都市", "値"), ("München", "١٢")))
        for column, text in enumerate(values)
    ]
    table = CanonicalTable(
        id="table-β",
        row_count=2,
        column_count=2,
        bbox=(10, 30, 190, 90),
        cells=cells,
        html=representation if html_table else None,
        markdown=None if html_table else representation,
    )
    return CanonicalPage(
        document_id="doc-α",
        page_index=0,
        page_size=PageSize(width=200, height=300, unit="point"),
        markdown=f"# Résumé\n\n{representation}",
        elements=[
            CanonicalElement(
                id="heading",
                element_type="heading",
                reading_order=0,
                text="Résumé",
                markdown="# Résumé",
                bbox=(1, 2, 80, 20),
            ),
            CanonicalElement(
                id="table-element",
                element_type="table",
                reading_order=1,
                text="都市 値 München ١٢",
                markdown=representation,
                bbox=(10, 30, 190, 90),
                table_id="table-β",
            ),
        ],
        tables=[table],
        figures=[],
        runtime=RuntimeStats(wall_seconds=0),
        provenance=CanonicalProvenance(
            tool_name="synthetic",
            tool_version="1",
            mode="test",
            input_sha256="a" * 64,
            config_sha256="b" * 64,
        ),
    )


def test_parsebench_records_preserve_unicode_types_boxes_tables_and_cells(tmp_path: Path) -> None:
    page = _page()

    records = export_parsebench_records([page])

    assert records[0]["upstream_ingestion"] == {
        "status": "requires_pinned_checkout",
        "note": (
            "Exact ParseBench upstream ingestion requires a pinned upstream checkout; "
            "this credential-free export is a deterministic interchange record, not an official upstream schema."
        ),
    }
    elements = records[0]["elements"]
    tables = records[0]["tables"]
    assert isinstance(elements, list) and elements[0]["element_type"] == "heading"
    assert elements[0]["bbox"] == [1.0, 2.0, 80.0, 20.0]
    assert isinstance(tables, list) and tables[0]["cells"][3]["text"] == "١٢"
    assert tables[0]["cells"][3]["bbox"] == [100.0, 60.0, 190.0, 90.0]

    output = tmp_path / "parsebench.jsonl"
    write_parsebench_jsonl([page], output)
    first = output.read_bytes()
    write_parsebench_jsonl([page], output, overwrite=True)
    assert output.read_bytes() == first
    assert "München" in json.loads(first)["tables"][0]["cells"][2]["text"]


def test_omnidocbench_uses_mapping_preserves_layout_and_rejects_mismatch_and_overwrite(
    tmp_path: Path,
) -> None:
    page = _page()
    mapping = {(page.document_id, page.page_index): "immutable-name"}

    paths = export_omnidocbench([page], tmp_path, mapping, include_layout_json=True)

    assert paths == [tmp_path / "immutable-name.md", tmp_path / "immutable-name.layout.json"]
    assert paths[0].read_text(encoding="utf-8") == page.markdown
    layout = json.loads(paths[1].read_text(encoding="utf-8"))
    assert layout["elements"][0]["bbox"] == [1.0, 2.0, 80.0, 20.0]
    assert layout["tables"][0]["cells"][3]["bbox"] == [100.0, 60.0, 190.0, 90.0]

    with pytest.raises(FileExistsError):
        export_omnidocbench([page], tmp_path, mapping, include_layout_json=True)
    export_omnidocbench([page], tmp_path, mapping, include_layout_json=True, overwrite=True)
    with pytest.raises(ValueError, match="mapping IDs do not match"):
        export_omnidocbench([page], tmp_path / "other", {("wrong", 0): "name"})


def test_olmocr_writes_verbatim_unicode_and_compact_html_with_stable_name(tmp_path: Path) -> None:
    page = _page(html_table=True)

    paths = export_olmocr_bench([page], tmp_path)

    assert paths == [tmp_path / "doc-α__page_000000.md"]
    assert paths[0].read_text(encoding="utf-8") == page.markdown
    assert "</td></tr></table>" in paths[0].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_olmocr_bench([page], tmp_path)
    export_olmocr_bench([page], tmp_path, overwrite=True)


def test_olmocr_writes_official_candidate_name_and_mirrors_pdf_subdirectory(tmp_path: Path) -> None:
    page = _page(html_table=True)
    mapping = {("doc-α", 0): Path("headers_footers/source.pdf")}

    paths = export_olmocr_bench([page], tmp_path, file_mapping=mapping)

    assert paths == [tmp_path / "headers_footers/source_pg1_repeat1.md"]
    assert paths[0].read_text(encoding="utf-8") == page.markdown
    with pytest.raises(ValueError, match="mapping IDs do not match"):
        export_olmocr_bench([page], tmp_path / "bad", file_mapping={("wrong", 0): Path("x.pdf")})
    with pytest.raises(ValueError, match="relative PDF path"):
        export_olmocr_bench([page], tmp_path / "absolute", file_mapping={("doc-α", 0): Path("/x.pdf")})


def test_rd_tablebench_exports_exact_compact_html_and_rejects_markdown() -> None:
    html_page = _page(html_table=True)

    first = export_rd_tablebench([html_page])
    second = export_rd_tablebench([html_page])

    assert first == second
    assert first[0]["id"] == "doc-α::table-table-β"
    assert first[0]["first_page_index"] == 0
    assert first[0]["html"] == html_page.tables[0].html
    continuation_page = html_page.model_copy(update={"page_index": 1})
    assert export_rd_tablebench([html_page, continuation_page]) == first
    with pytest.raises(UnsupportedTableRepresentationError, match="unsupported for markdown-only"):
        export_rd_tablebench([_page()])


def test_rd_tablebench_evaluation_projection_renders_canonical_grid_without_reference() -> None:
    table = _page().tables[0]

    rendered = render_rd_tablebench_html(table)

    assert rendered == (
        "<table><tr><th>都市</th><th>値</th></tr><tr><td>München</td><td>١٢</td></tr></table>"
    )


_ATTRIBUTION_DOCUMENT_ID = "c" * 64


def _attribution_annotation() -> AnnotationMetadata:
    return AnnotationMetadata(
        stage="candidate",
        revision=1,
        annotator="synthetic-private",
        confidence=1,
        adjudication_status="unreviewed",
    )


def _source_item(
    text: str,
    *,
    page_number: int,
    kind: SourceItemKind = "span",
    x0: float = 10,
    y0: float = 10,
):
    return make_source_item(
        document_id=_ATTRIBUTION_DOCUMENT_ID,
        page_number=page_number,
        kind=kind,
        coordinate_frame="source_page",
        coordinates=(x0, y0, x0 + 20, y0 + 10),
        text=text,
    )


def _attributed_paragraph(
    element_id: str,
    content: str,
    order: int,
    fragments: list[tuple[int, tuple[float, float, float, float], list[str]]],
) -> DocumentElement:
    return DocumentElement(
        document_id=_ATTRIBUTION_DOCUMENT_ID,
        element_id=element_id,
        order=order,
        element_type="paragraph",
        content=content,
        format="text",
        fragments=[
            PageFragment(
                page_number=page,
                page_width=200,
                page_height=300,
                bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                source_item_ids=source_ids,
            )
            for page, bbox, source_ids in fragments
        ],
        structure=ElementStructure(paragraph=ParagraphStructure(role="body")),
        annotation=_attribution_annotation(),
    )


def _attributed_table(source_ids: tuple[str, str]) -> DocumentElement:
    page_fragment = PageFragment(
        page_number=1,
        page_width=200,
        page_height=300,
        bbox=BoundingBox(x0=10, y0=50, x1=190, y1=100),
    )
    cells = [
        TableCell(
            row_index=0,
            column_index=0,
            colspan=2,
            role="header",
            text="Total",
            fragments=[page_fragment.model_copy(update={"source_item_ids": [source_ids[0]]})],
        ),
        TableCell(
            row_index=1,
            column_index=0,
            role="body",
            text="10",
            fragments=[page_fragment.model_copy(update={"source_item_ids": [source_ids[1]]})],
        ),
        TableCell(
            row_index=1,
            column_index=1,
            role="body",
            text="",
            fragments=[page_fragment],
        ),
    ]
    structure = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation="html",
        classification_reasons=["spanning_cells"],
        cells=cells,
    )
    html = (
        '<table><thead><tr><th colspan="2">Total</th></tr></thead>'
        "<tbody><tr><td>10</td><td></td></tr></tbody></table>"
    )
    return DocumentElement(
        document_id=_ATTRIBUTION_DOCUMENT_ID,
        element_id="table-one",
        order=0,
        element_type="table",
        content=html,
        format="html",
        fragments=[page_fragment],
        structure=ElementStructure(table=structure),
        annotation=_attribution_annotation(),
    )


def test_parsebench_attribution_projects_native_spans_and_cross_page_elements() -> None:
    alpha = _source_item("Alpha", page_number=1)
    beta = _source_item("Beta", page_number=2)
    element = _attributed_paragraph(
        "paragraph-one",
        "Alpha Beta",
        0,
        [
            (1, (5, 5, 50, 30), [alpha.source_item_id]),
            (2, (6, 6, 51, 31), [beta.source_item_id]),
        ],
    )
    catalog = build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [beta, alpha])

    pages = project_parsebench_layout_pages([element], catalog, [(200, 300), (200, 300)])

    assert [page["page"] for page in pages] == [1, 2]
    assert [page["md"] for page in pages] == ["Alpha Beta", ""]
    assert "\n\n".join(page["md"] for page in pages if page["md"]) == "Alpha Beta"
    first_item = pages[0]["items"][0]
    second_item = pages[1]["items"][0]
    assert first_item["id"] == "paragraph-one:page:000001"
    assert second_item["id"] == "paragraph-one:page:000002"
    assert first_item["bBox"] == pytest.approx({
        "x": 5 / 200,
        "y": 5 / 300,
        "w": 45 / 200,
        "h": 25 / 300,
    })
    first_segment = first_item["layoutAwareBbox"][0]
    assert {key: first_segment[key] for key in ("x", "y", "w", "h")} == pytest.approx({
        "x": 10 / 200,
        "y": 10 / 300,
        "w": 20 / 200,
        "h": 10 / 300,
    })
    assert {key: first_segment[key] for key in ("label", "startIndex", "endIndex", "sourceItemId")} == {
        "label": "text",
        "startIndex": 0,
        "endIndex": 4,
        "sourceItemId": alpha.source_item_id,
    }
    assert second_item["layoutAwareBbox"][0]["startIndex"] == 6
    assert second_item["layoutAwareBbox"][0]["endIndex"] == 9


def test_parsebench_attribution_projects_exclusive_spanning_table_cells_and_omits_empty_cell() -> None:
    total = _source_item("Total", page_number=1, kind="word", x0=20, y0=55)
    ten = _source_item("10", page_number=1, kind="word", x0=20, y0=80)
    table = _attributed_table((total.source_item_id, ten.source_item_id))

    pages = project_parsebench_layout_pages(
        [table],
        build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [total, ten]),
        [(200, 300)],
    )

    items = pages[0]["items"]
    assert {key: value for key, value in items[0].items() if key != "bBox"} == {
        "id": "table-one:page:000001",
        "type": "geometry",
        "value": "",
        "md": "",
        "html": "",
        "layoutAwareBbox": [],
        "projectionVersion": "parsebench-layout-attribution-v2",
    }
    assert items[0]["bBox"]["label"] == "table"
    assert {key: items[0]["bBox"][key] for key in ("x", "y", "w", "h")} == pytest.approx({
        "x": 10 / 200,
        "y": 50 / 300,
        "w": 180 / 200,
        "h": 50 / 300,
    })
    cell_items = items[1:]
    assert [item["id"] for item in cell_items] == [
        "table-one:cell:000000:000000:000001:000002:page:000001",
        "table-one:cell:000001:000000:000001:000001:page:000001",
    ]
    assert [item["value"] for item in cell_items] == ["Total", "10"]
    assert [item["layoutAwareBbox"][0]["sourceItemId"] for item in cell_items] == [
        total.source_item_id,
        ten.source_item_id,
    ]


def test_parsebench_attribution_rejects_missing_catalog_and_incompatible_document_page_or_kind() -> None:
    span = _source_item("Alpha", page_number=1)
    element = _attributed_paragraph("paragraph-one", "Alpha", 0, [(1, (5, 5, 50, 30), [span.source_item_id])])
    catalog = build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [span])

    with pytest.raises(ValueError, match="source catalog is required"):
        project_parsebench_layout_pages([element], None, [(200, 300)])
    with pytest.raises(ValueError, match="document_id"):
        project_parsebench_layout_pages(
            [element.model_copy(update={"document_id": "wrong"})], catalog, [(200, 300)]
        )

    page_two = _source_item("Alpha", page_number=2)
    wrong_page = element.model_copy(
        update={
            "fragments": [
                element.fragments[0].model_copy(update={"source_item_ids": [page_two.source_item_id]})
            ]
        }
    )
    with pytest.raises(ValueError, match="page mismatch"):
        project_parsebench_layout_pages(
            [wrong_page], build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [page_two]), [(200, 300), (200, 300)]
        )

    word = _source_item("Alpha", page_number=1, kind="word")
    wrong_kind = element.model_copy(
        update={
            "fragments": [element.fragments[0].model_copy(update={"source_item_ids": [word.source_item_id]})]
        }
    )
    with pytest.raises(ValueError, match="paragraph attribution requires native spans"):
        project_parsebench_layout_pages(
            [wrong_kind], build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [word]), [(200, 300)]
        )

    table = _attributed_table((span.source_item_id, span.source_item_id))
    with pytest.raises(ValueError, match="table cell attribution requires native words"):
        project_parsebench_layout_pages(
            [table], build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [span]), [(200, 300)]
        )


def test_parsebench_attribution_rejects_shared_sources_empty_cell_sources_and_text_mismatch() -> None:
    span = _source_item("Alpha", page_number=1)
    first = _attributed_paragraph("first", "Alpha", 0, [(1, (5, 5, 50, 30), [span.source_item_id])])
    second = _attributed_paragraph("second", "Alpha", 1, [(1, (60, 5, 100, 30), [span.source_item_id])])
    catalog = build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [span])
    with pytest.raises(ValueError, match="cannot be shared"):
        project_parsebench_layout_pages([first, second], catalog, [(200, 300)])
    with pytest.raises(ValueError, match="cannot be projected exactly"):
        project_parsebench_layout_pages(
            [first.model_copy(update={"content": "Different"})], catalog, [(200, 300)]
        )
    with pytest.raises(ValueError, match="empty element first cannot cite"):
        project_parsebench_layout_pages([first.model_copy(update={"content": ""})], catalog, [(200, 300)])

    total = _source_item("Total", page_number=1, kind="word", x0=20, y0=55)
    ten = _source_item("10", page_number=1, kind="word", x0=20, y0=80)
    table = _attributed_table((total.source_item_id, ten.source_item_id))
    structure = table.structure.table
    assert structure is not None
    cells = [
        cell.model_copy(
            update={
                "fragments": [cell.fragments[0].model_copy(update={"source_item_ids": [ten.source_item_id]})]
            }
        )
        if cell.text == ""
        else cell
        for cell in structure.cells
    ]
    invalid = table.model_copy(
        update={"structure": ElementStructure(table=structure.model_copy(update={"cells": cells}))}
    )
    with pytest.raises(ValueError, match="empty table cells cannot cite"):
        project_parsebench_layout_pages(
            [invalid], build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [total, ten]), [(200, 300)]
        )

    shared_cells = [
        cell.model_copy(
            update={
                "text": "10",
                "fragments": [cell.fragments[0].model_copy(update={"source_item_ids": [ten.source_item_id]})],
            }
        )
        if cell.text == ""
        else cell
        for cell in structure.cells
    ]
    shared = table.model_copy(
        update={"structure": ElementStructure(table=structure.model_copy(update={"cells": shared_cells}))}
    )
    with pytest.raises(ValueError, match="cannot be shared"):
        project_parsebench_layout_pages(
            [shared], build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [total, ten]), [(200, 300)]
        )


def test_parsebench_attribution_is_permutation_stable_and_serializes_deterministically() -> None:
    alpha = _source_item("Alpha", page_number=1, x0=10)
    beta = _source_item("Beta", page_number=1, x0=40)
    first = _attributed_paragraph(
        "first",
        "Alpha Beta",
        0,
        [(1, (5, 5, 80, 30), [alpha.source_item_id, beta.source_item_id])],
    )
    second = _attributed_paragraph("second", "Plain", 1, [(1, (5, 40, 80, 60), [])])
    catalog = build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [beta, alpha])

    canonical = project_parsebench_layout_pages([first, second], catalog, [(200, 300)])
    permuted_first = first.model_copy(
        update={
            "fragments": [
                first.fragments[0].model_copy(
                    update={"source_item_ids": first.fragments[0].source_item_ids[::-1]}
                )
            ]
        }
    )
    permuted = project_parsebench_layout_pages([second, permuted_first], catalog, [(200, 300)])

    assert permuted == canonical
    first_serialization = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    second_serialization = json.dumps(permuted, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    assert second_serialization == first_serialization
    unattributed = canonical[0]["items"][1]
    assert unattributed["value"] == ""
    assert unattributed["md"] == "Plain"
    assert unattributed["layoutAwareBbox"] == []


def test_parsebench_source_page_extents_stay_unrotated_for_rotated_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "rotated-private.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=200, height=300)
        page.set_rotation(90)
        document.save(pdf_path)

    with pymupdf.open(pdf_path) as document:
        assert (float(document[0].rect.width), float(document[0].rect.height)) == (300, 200)

    assert parsebench_source_page_extents(pdf_path) == [(200, 300)]


def test_parsebench_attribution_maps_nfkc_expansions_and_repeated_raw_spans() -> None:
    ligature = _source_item("ﬃ", page_number=1, x0=10)
    first_full_width = _source_item("Ａ", page_number=1, x0=40)
    second_full_width = _source_item("Ａ", page_number=1, x0=70)
    element = _attributed_paragraph(
        "normalized",
        "ffi A A",
        0,
        [
            (
                1,
                (5, 5, 100, 30),
                [
                    second_full_width.source_item_id,
                    ligature.source_item_id,
                    first_full_width.source_item_id,
                ],
            )
        ],
    )

    pages = project_parsebench_layout_pages(
        [element],
        build_source_catalog(
            _ATTRIBUTION_DOCUMENT_ID,
            [second_full_width, ligature, first_full_width],
        ),
        [(200, 300)],
    )

    segments = pages[0]["items"][0]["layoutAwareBbox"]
    assert [(segment["startIndex"], segment["endIndex"]) for segment in segments] == [
        (0, 2),
        (4, 4),
        (6, 6),
    ]
    assert [segment["sourceItemId"] for segment in segments] == [
        ligature.source_item_id,
        first_full_width.source_item_id,
        second_full_width.source_item_id,
    ]


def test_parsebench_attribution_requires_source_frame_and_native_content_ids() -> None:
    span = _source_item("Alpha", page_number=1)
    element = _attributed_paragraph("paragraph-one", "Alpha", 0, [(1, (5, 5, 50, 30), [span.source_item_id])])
    detector_rule = make_source_item(
        document_id=_ATTRIBUTION_DOCUMENT_ID,
        page_number=1,
        kind="rule",
        coordinate_frame="detector_page",
        coordinates=(0, 100, 100, 100),
        text=None,
    )
    assert (
        project_parsebench_layout_pages(
            [element],
            build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [span, detector_rule]),
            [(200, 300)],
        )[0]["items"][0]["value"]
        == "Alpha"
    )

    detector_item = span.model_copy(update={"coordinate_frame": "detector_page"})
    with pytest.raises(ValueError, match="source_page coordinate frame"):
        project_parsebench_layout_pages(
            [element],
            build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [detector_item]),
            [(200, 300)],
        )

    legacy = make_source_item(
        source_item_id="pymupdf:p000001:span:000001",
        document_id=_ATTRIBUTION_DOCUMENT_ID,
        page_number=1,
        kind="span",
        coordinate_frame="source_page",
        coordinates=(10, 10, 30, 20),
        text="Alpha",
    )
    legacy_element = element.model_copy(
        update={
            "fragments": [
                element.fragments[0].model_copy(update={"source_item_ids": [legacy.source_item_id]})
            ]
        }
    )
    with pytest.raises(ValueError, match="native-content-v1"):
        project_parsebench_layout_pages(
            [legacy_element],
            build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [legacy]),
            [(200, 300)],
        )


def test_parsebench_attribution_accepts_zero_width_native_text_but_rejects_out_of_page() -> None:
    zero_width = make_source_item(
        document_id=_ATTRIBUTION_DOCUMENT_ID,
        page_number=1,
        kind="span",
        coordinate_frame="source_page",
        coordinates=(10, 10, 10, 20),
        text="A",
    )
    element = _attributed_paragraph(
        "zero-width",
        "A",
        0,
        [(1, (5, 5, 50, 30), [zero_width.source_item_id])],
    )

    pages = project_parsebench_layout_pages(
        [element],
        build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [zero_width]),
        [(200, 300)],
    )
    assert pages[0]["items"][0]["layoutAwareBbox"][0]["w"] == 0

    outside = _source_item("A", page_number=1, x0=195)
    outside_element = element.model_copy(
        update={
            "fragments": [
                element.fragments[0].model_copy(update={"source_item_ids": [outside.source_item_id]})
            ]
        }
    )
    with pytest.raises(ValueError, match="outside its declared source page"):
        project_parsebench_layout_pages(
            [outside_element],
            build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [outside]),
            [(200, 300)],
        )

    nonfinite = outside.model_copy(update={"x0": float("nan")})
    with pytest.raises(ValueError, match="coordinates must be finite"):
        project_parsebench_layout_pages(
            [outside_element],
            build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [nonfinite]),
            [(200, 300)],
        )


@pytest.mark.parametrize(
    ("rotation", "display_size", "display_bbox"),
    [
        (90, (300.0, 200.0), (260.0, 10.0, 280.0, 30.0)),
        (180, (200.0, 300.0), (170.0, 260.0, 190.0, 280.0)),
        (270, (300.0, 200.0), (20.0, 170.0, 40.0, 190.0)),
    ],
)
def test_parsebench_rotated_source_and_scaled_fragment_geometry_share_normalized_frame(
    rotation: int,
    display_size: tuple[float, float],
    display_bbox: tuple[float, float, float, float],
) -> None:
    source = make_source_item(
        document_id=_ATTRIBUTION_DOCUMENT_ID,
        page_number=1,
        kind="span",
        coordinate_frame="source_page",
        coordinates=(10, 20, 30, 40),
        text="Text",
    )
    element = _attributed_paragraph(
        "rotated",
        "Text",
        0,
        [(1, display_bbox, [source.source_item_id])],
    )
    scale = 2.0
    scaled_bbox = tuple(value * scale for value in display_bbox)
    element = element.model_copy(
        update={
            "fragments": [
                element.fragments[0].model_copy(
                    update={
                        "page_width": display_size[0] * scale,
                        "page_height": display_size[1] * scale,
                        "bbox": BoundingBox(
                            x0=scaled_bbox[0],
                            y0=scaled_bbox[1],
                            x1=scaled_bbox[2],
                            y1=scaled_bbox[3],
                        ),
                    }
                )
            ]
        }
    )
    geometry = ParseBenchPageGeometry(
        source_width=200,
        source_height=300,
        rotation=rotation,
        output_width=display_size[0] * 3,
        output_height=display_size[1] * 3,
        crop_x0=17,
        crop_y0=29,
    )

    page = project_parsebench_layout_pages(
        [element],
        build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [source]),
        [geometry],
    )[0]

    item = page["items"][0]
    segment = item["layoutAwareBbox"][0]
    assert page["width"] == display_size[0] * 3
    assert page["height"] == display_size[1] * 3
    assert {key: item["bBox"][key] for key in ("x", "y", "w", "h")} == pytest.approx({
        key: segment[key] for key in ("x", "y", "w", "h")
    })
    assert 0 <= segment["x"] <= 1
    assert 0 <= segment["y"] <= 1
    assert 0 <= segment["x"] + segment["w"] <= 1
    assert 0 <= segment["y"] + segment["h"] <= 1


def test_parsebench_rotation_rejects_incompatible_fragment_frame_and_partial_output_scale() -> None:
    source = _source_item("Text", page_number=1)
    element = _attributed_paragraph(
        "wrong-frame",
        "Text",
        0,
        [(1, (10, 20, 30, 40), [source.source_item_id])],
    )
    with pytest.raises(ValueError, match="incompatible with page rotation and scaling"):
        project_parsebench_layout_pages(
            [element],
            build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [source]),
            [ParseBenchPageGeometry(source_width=200, source_height=300, rotation=90)],
        )
    with pytest.raises(ValueError, match="both be present or both be null"):
        ParseBenchPageGeometry(
            source_width=200,
            source_height=300,
            output_width=600,
        )


def test_parsebench_exact_footnote_ranges_preserve_same_label_literal_syntax() -> None:
    source = _source_item("Literal [^1] Word¹", page_number=1)
    element = _attributed_paragraph(
        "mixed-reference",
        "Literal [^1] Word[^1]",
        0,
        [(1, (10, 10, 180, 30), [source.source_item_id])],
    ).model_copy(
        update={
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="body"),
                properties=[
                    StructureProperty(
                        key="inline_footnote_markers",
                        value='[{"start":17,"end":21,"label":"1"}]',
                    )
                ],
            )
        }
    )

    page = project_parsebench_layout_pages(
        [element],
        build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [source]),
        [(200, 300)],
    )[0]
    item = page["items"][0]

    assert item["md"] == "Literal \\[^1\\] Word[^1]"
    assert item["value"] == "Literal [^1] Word1"
    assert item["layoutAwareBbox"][0]["endIndex"] == 17


def test_parsebench_legacy_footnote_labels_make_no_generated_syntax_claim() -> None:
    source = _source_item("Literal [^1]", page_number=1)
    element = _attributed_paragraph(
        "legacy-literal",
        "Literal [^1]",
        0,
        [(1, (10, 10, 120, 30), [source.source_item_id])],
    ).model_copy(
        update={
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="body"),
                properties=[StructureProperty(key="inline_footnote_markers", value='["1"]')],
            )
        }
    )

    page = project_parsebench_layout_pages(
        [element],
        build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [source]),
        [(200, 300)],
    )[0]
    item = page["items"][0]

    assert item["md"] == "Literal \\[^1\\]"
    assert item["value"] == "Literal [^1]"


def test_parsebench_page_geometry_preserves_crop_origin_and_upright_extent(tmp_path: Path) -> None:
    pdf_path = tmp_path / "crop-rotation-private.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=400, height=500)
        page.set_cropbox((50, 100, 350, 400))
        page.set_rotation(90)
        document.save(pdf_path)

    geometry = parsebench_page_geometries(pdf_path)[0]

    assert geometry.source_width == 300
    assert geometry.source_height == 300
    assert geometry.display_size == (300, 300)
    assert (geometry.crop_x0, geometry.crop_y0) == (50, 100)


def test_parsebench_markdown_uses_shared_semantic_renderer_and_excludes_generated_syntax_from_attribution() -> (
    None
):
    title_source = _source_item("Title", page_number=1, x0=10, y0=10)
    list_source = _source_item("Item", page_number=1, x0=10, y0=40)
    escaped_source = _source_item("a*b", page_number=1, x0=10, y0=70)
    reference_source = _source_item("Word¹", page_number=1, x0=10, y0=100)
    note_source = _source_item("1. Note.", page_number=1, x0=10, y0=130)

    title = _attributed_paragraph(
        "title", "Title", 0, [(1, (10, 10, 80, 30), [title_source.source_item_id])]
    ).model_copy(
        update={
            "element_type": "heading",
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="heading", heading_level=1),
                style_runs=[TextStyleRun(start=0, end=5, bold=True)],
            ),
        }
    )
    list_item = _attributed_paragraph(
        "item", "Item", 1, [(1, (10, 40, 80, 60), [list_source.source_item_id])]
    ).model_copy(
        update={
            "structure": ElementStructure(
                paragraph=ParagraphStructure(
                    role="list_item",
                    list_depth=0,
                    list_label="•",
                )
            )
        }
    )
    escaped = _attributed_paragraph(
        "escaped", "a*b", 2, [(1, (10, 70, 80, 90), [escaped_source.source_item_id])]
    )
    reference = _attributed_paragraph(
        "reference",
        "Word[^1]",
        3,
        [(1, (10, 100, 80, 120), [reference_source.source_item_id])],
    ).model_copy(
        update={
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="body"),
                linked_element_ids=["note"],
                properties=[
                    StructureProperty(
                        key="inline_footnote_markers",
                        value='[{"start":4,"end":8,"label":"1"}]',
                    )
                ],
            )
        }
    )
    note = _attributed_paragraph(
        "note", "1. Note.", 4, [(1, (10, 130, 80, 150), [note_source.source_item_id])]
    ).model_copy(
        update={
            "element_type": "footnote",
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="footnote"),
                footnote=FootnoteStructure(
                    label="1",
                    reference_element_ids=["reference"],
                    association_confident=True,
                ),
                linked_element_ids=["reference"],
            ),
        }
    )

    pages = project_parsebench_layout_pages(
        [title, list_item, escaped, reference, note],
        build_source_catalog(
            _ATTRIBUTION_DOCUMENT_ID,
            [title_source, list_source, escaped_source, reference_source, note_source],
        ),
        [(200, 300)],
    )

    expected = "# **Title**\n\n- Item\n\na\\*b\n\nWord[^1]\n\n[^1]: Note."
    assert pages[0]["md"] == expected
    assert pages[0]["text"] == expected
    items = {item["id"].split(":page:")[0]: item for item in pages[0]["items"]}
    assert items["title"]["md"] == "# **Title**"
    assert items["title"]["value"] == "Title"
    assert items["item"]["md"] == "- Item"
    assert items["escaped"]["md"] == "a\\*b"
    assert items["escaped"]["value"] == "a*b"
    assert items["escaped"]["layoutAwareBbox"][0]["endIndex"] == 2
    assert items["reference"]["md"] == "Word[^1]"
    assert items["reference"]["value"] == "Word1"
    assert items["reference"]["layoutAwareBbox"][0]["startIndex"] == 0
    assert items["reference"]["layoutAwareBbox"][0]["endIndex"] == 4
    assert items["note"]["md"] == "[^1]: Note."

    literal = reference.model_copy(
        update={
            "order": 0,
            "structure": reference.structure.model_copy(update={"properties": []}),
        }
    )
    with pytest.raises(ValueError, match="cannot be projected exactly"):
        project_parsebench_layout_pages(
            [literal],
            build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [reference_source]),
            [(200, 300)],
        )


def test_parsebench_provider_template_executes_with_pinned_compatible_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Record:
        def __init__(self, **values: object) -> None:
            for key, value in values.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, value: object) -> _Record:
            return cls(payload=value)

    class _Provider:
        pass

    class _ProviderPermanentError(ValueError):
        pass

    class _ProductType(Enum):
        PARSE = "parse"
        OTHER = "other"

    def _register_provider(_name: str) -> Any:
        def decorate(provider: Any) -> Any:
            return provider

        return decorate

    modules = {
        name: ModuleType(name)
        for name in (
            "parse_bench",
            "parse_bench.inference",
            "parse_bench.inference.providers",
            "parse_bench.inference.providers.base",
            "parse_bench.inference.providers.registry",
            "parse_bench.schemas",
            "parse_bench.schemas.parse_output",
            "parse_bench.schemas.pipeline",
            "parse_bench.schemas.pipeline_io",
            "parse_bench.schemas.product",
        )
    }
    for name in (
        "parse_bench",
        "parse_bench.inference",
        "parse_bench.inference.providers",
        "parse_bench.schemas",
    ):
        setattr(modules[name], "__path__", [])
    stub_attributes: dict[str, dict[str, object]] = {
        "parse_bench.inference.providers.base": {
            "Provider": _Provider,
            "ProviderPermanentError": _ProviderPermanentError,
        },
        "parse_bench.inference.providers.registry": {
            "register_provider": _register_provider,
        },
        "parse_bench.schemas.parse_output": {
            "PageIR": _Record,
            "ParseLayoutPageIR": _Record,
            "ParseOutput": _Record,
        },
        "parse_bench.schemas.pipeline": {"PipelineSpec": _Record},
        "parse_bench.schemas.pipeline_io": {
            "InferenceRequest": _Record,
            "InferenceResult": _Record,
            "RawInferenceResult": _Record,
        },
        "parse_bench.schemas.product": {"ProductType": _ProductType},
    }
    for module_name, attributes in stub_attributes.items():
        for attribute, value in attributes.items():
            setattr(modules[module_name], attribute, value)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    pdf_path = tmp_path / "provider-private.pdf"
    with pymupdf.open() as document:
        document.new_page(width=200, height=300)
        document.new_page(width=200, height=300)
        document.save(pdf_path)
    first_source = _source_item("First", page_number=1)
    second_source = _source_item("Second", page_number=2)
    extracted = SimpleNamespace(
        elements=(
            _attributed_paragraph(
                "first",
                "First",
                0,
                [(1, (10, 10, 80, 30), [first_source.source_item_id])],
            ),
            _attributed_paragraph(
                "second",
                "Second",
                1,
                [(2, (10, 10, 80, 30), [second_source.source_item_id])],
            ),
        ),
        source_catalog=build_source_catalog(_ATTRIBUTION_DOCUMENT_ID, [first_source, second_source]),
    )
    import app.pdf2md.engine as engine_module

    monkeypatch.setattr(
        engine_module,
        "extract_document_with_catalog",
        lambda _path, *, annotator: extracted,
    )
    template_path = Path(__file__).parents[2] / "benchmarks/integrations/parsebench/pdf2md_local.py.template"
    namespace = runpy.run_path(str(template_path))
    provider_class: Any = namespace["Pdf2mdLocalProvider"]
    provider = provider_class()
    pipeline = _Record(pipeline_name="private-pipeline")
    request = _Record(
        product_type=_ProductType.PARSE,
        source_file_path=str(pdf_path),
        example_id="private-example",
    )

    raw = provider.run_inference(pipeline, request)
    normalized = provider.normalize(raw)

    assert [page["md"] for page in raw.raw_output["pages"]] == ["First", "Second"]
    assert raw.raw_output["source_catalog"] == extracted.source_catalog.model_dump(mode="json")
    assert len(raw.raw_output["elements"]) == 2
    assert normalized.output.markdown == "First\n\nSecond"
    assert [page.markdown for page in normalized.output.pages] == ["First", "Second"]
    assert [page.payload["page"] for page in normalized.output.layout_pages] == [1, 2]

    with pytest.raises(_ProviderPermanentError, match="only supports PARSE"):
        provider.run_inference(
            pipeline,
            _Record(
                product_type=_ProductType.OTHER,
                source_file_path=str(pdf_path),
                example_id="private-example",
            ),
        )
    with pytest.raises(_ProviderPermanentError, match="PDF file not found"):
        provider.run_inference(
            pipeline,
            _Record(
                product_type=_ProductType.PARSE,
                source_file_path=str(tmp_path / "missing.pdf"),
                example_id="private-example",
            ),
        )
