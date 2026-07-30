from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.adapters.olmocr_bench import export_olmocr_bench
from benchmarks.adapters.omnidocbench import export_omnidocbench
from benchmarks.adapters.parsebench import export_parsebench_records, write_parsebench_jsonl
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
