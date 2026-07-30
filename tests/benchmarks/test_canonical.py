from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.canonical import (
    AdapterRun,
    CanonicalCell,
    CanonicalElement,
    CanonicalFigure,
    CanonicalPage,
    CanonicalProvenance,
    CanonicalTable,
    PageSize,
    RuntimeStats,
    read_pages_jsonl,
    semantic_hash,
    validate_page_collection,
    write_pages_jsonl,
)

_SHA = "a" * 64


def _provenance() -> CanonicalProvenance:
    return CanonicalProvenance(
        tool_name="synthetic-parser",
        tool_version="1.2.3",
        mode="local-cpu",
        input_sha256=_SHA,
        config_sha256="b" * 64,
    )


def _table() -> CanonicalTable:
    return CanonicalTable(
        id="table-1",
        row_count=2,
        column_count=2,
        bbox=(10, 40, 190, 100),
        markdown="| Name | Value |\n| --- | --- |\n| A | 1 |",
        cells=[
            CanonicalCell(
                id=f"cell-{row}-{column}",
                row_index=row,
                column_index=column,
                text=text,
                bbox=(10 + column * 90, 40 + row * 30, 100 + column * 90, 70 + row * 30),
            )
            for row, values in enumerate((("Name", "Value"), ("A", "1")))
            for column, text in enumerate(values)
        ],
    )


def _page(*, page_index: int = 0, document_id: str = "synthetic-document") -> CanonicalPage:
    return CanonicalPage(
        document_id=document_id,
        page_index=page_index,
        page_size=PageSize(width=200, height=300, unit="point"),
        markdown="# Heading\n\n| Name | Value |\n| --- | --- |\n| A | 1 |\n\nFigure 1",
        elements=[
            CanonicalElement(
                id=f"heading-{page_index}",
                element_type="heading",
                reading_order=0,
                text="Heading",
                markdown="# Heading",
                bbox=(10, 10, 100, 30),
            ),
            CanonicalElement(
                id=f"table-element-{page_index}",
                element_type="table",
                reading_order=1,
                text="Name Value A 1",
                markdown="| Name | Value |\n| --- | --- |\n| A | 1 |",
                bbox=(10, 40, 190, 100),
                table_id="table-1",
            ),
            CanonicalElement(
                id=f"figure-element-{page_index}",
                element_type="figure",
                reading_order=2,
                text="Figure 1",
                markdown="![Figure 1](figure.png)",
                bbox=None,
                figure_id="figure-1",
            ),
        ],
        tables=[_table()],
        figures=[CanonicalFigure(id="figure-1", bbox=None, caption="Figure 1")],
        runtime=RuntimeStats(wall_seconds=0.25, peak_rss_bytes=1024),
        provenance=_provenance(),
        output_sha256="c" * 64,
    )


def test_valid_page_round_trips_through_deterministic_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "canonical.jsonl"
    pages = [_page(page_index=1), _page(page_index=0)]

    write_pages_jsonl(pages, path)

    assert read_pages_jsonl(path) == [_page(page_index=0), _page(page_index=1)]
    first_bytes = path.read_bytes()
    write_pages_jsonl(list(reversed(pages)), path, overwrite=True)
    assert path.read_bytes() == first_bytes
    assert first_bytes.endswith(b"\n")


def test_nullable_bbox_represents_unsupported_grounding_without_fabricated_coordinates() -> None:
    page = _page()

    assert page.figures[0].bbox is None
    assert page.elements[2].bbox is None


@pytest.mark.parametrize(
    "bbox",
    [
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (2, 0, 1, 1),
        (0, float("nan"), 1, 1),
        (0, 0, float("inf"), 1),
    ],
)
def test_rejects_malformed_real_bboxes(bbox: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValidationError, match="bounding box"):
        CanonicalFigure(id="figure", bbox=bbox)


def test_rejects_bounding_boxes_outside_page_geometry() -> None:
    page = _page()
    out_of_bounds = page.elements[0].model_copy(update={"bbox": (-1, 10, 100, 30)})

    with pytest.raises(ValidationError, match="outside the page"):
        CanonicalPage.model_validate({
            **page.model_dump(),
            "elements": [out_of_bounds, *page.elements[1:]],
        })


def test_rejects_duplicate_ids_and_noncontiguous_reading_order() -> None:
    page = _page()
    duplicate = page.elements[0].model_copy(update={"reading_order": 1})

    with pytest.raises(ValidationError, match="element ids"):
        CanonicalPage.model_validate({
            **page.model_dump(),
            "elements": [page.elements[0], duplicate, page.elements[2]],
        })

    reordered = page.elements[1].model_copy(update={"reading_order": 3})
    with pytest.raises(ValidationError, match="reading_order"):
        CanonicalPage.model_validate({**page.model_dump(), "elements": [page.elements[0], reordered]})


def test_rejects_bad_spans_overlap_and_duplicate_cell_ids() -> None:
    cells = _table().cells
    with pytest.raises(ValidationError, match="rowspan exceeds"):
        CanonicalTable(
            id="bad-span",
            row_count=2,
            column_count=2,
            markdown="table",
            cells=[cells[0].model_copy(update={"rowspan": 3})],
        )

    with pytest.raises(ValidationError, match="overlap"):
        CanonicalTable(
            id="overlap",
            row_count=2,
            column_count=2,
            markdown="table",
            cells=[cells[0], cells[1].model_copy(update={"row_index": 0, "column_index": 0})],
        )

    with pytest.raises(ValidationError, match="cell ids"):
        CanonicalTable(
            id="duplicate",
            row_count=1,
            column_count=2,
            markdown="table",
            cells=[cells[0], cells[1].model_copy(update={"id": cells[0].id})],
        )


def test_rejects_missing_or_multiply_referenced_typed_collections() -> None:
    page = _page()
    with pytest.raises(ValidationError, match="every table"):
        CanonicalPage.model_validate({**page.model_dump(), "tables": []})

    second_reference = page.elements[1].model_copy(update={"id": "second-table", "reading_order": 2})
    figure = page.elements[2].model_copy(update={"reading_order": 3})
    with pytest.raises(ValidationError, match="every table"):
        CanonicalPage.model_validate({
            **page.model_dump(),
            "elements": [page.elements[0], page.elements[1], second_reference, figure],
        })


def test_table_representation_must_be_retained_exactly() -> None:
    page = _page()
    changed_element = page.elements[1].model_copy(update={"markdown": "flattened table text"})

    with pytest.raises(ValidationError, match="retained"):
        CanonicalPage.model_validate({
            **page.model_dump(),
            "elements": [page.elements[0], changed_element, page.elements[2]],
        })


def test_strict_models_reject_extra_fields_and_wrong_schema_version() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PageSize(width=1, height=1, unit="point", dpi=72)  # pyright: ignore[reportCallIssue]

    record = _page().model_dump(mode="json")
    record["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="1.0"):
        CanonicalPage.model_validate(record)


def test_rejects_nonfinite_runtime_and_failed_run_without_error() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        RuntimeStats(wall_seconds=float("nan"))

    with pytest.raises(ValidationError, match="requires an error"):
        AdapterRun(
            status="failed",
            pages=[],
            raw_records=[],
            runtime=RuntimeStats(wall_seconds=0),
        )


def test_semantic_hash_excludes_runtime_and_output_hash() -> None:
    page = _page()
    changed = page.model_copy(
        update={
            "runtime": RuntimeStats(wall_seconds=99, peak_rss_bytes=999_999),
            "output_sha256": "d" * 64,
        }
    )

    assert semantic_hash([page]) == semantic_hash([changed])
    assert semantic_hash([page]) != semantic_hash([
        page.model_copy(update={"markdown": "semantically changed"})
    ])


def test_complete_collection_requires_unique_contiguous_canonical_page_keys() -> None:
    with pytest.raises(ValueError, match="must contain at least one"):
        validate_page_collection([])
    with pytest.raises(ValueError, match="must be unique"):
        validate_page_collection([_page(), _page()])
    with pytest.raises(ValueError, match="contiguous"):
        validate_page_collection([_page(page_index=1)])
    with pytest.raises(ValueError, match="canonical"):
        validate_page_collection([_page(page_index=1), _page(page_index=0)])

    changed_provenance = _page(page_index=1).model_copy(
        update={
            "provenance": _provenance().model_copy(update={"tool_version": "different"}),
        }
    )
    with pytest.raises(ValueError, match="provenance"):
        validate_page_collection([_page(), changed_provenance])

    validate_page_collection([
        _page(document_id="a"),
        _page(document_id="b"),
    ])


def test_reader_fails_closed_on_malformed_competitor_records(tmp_path: Path) -> None:
    path = tmp_path / "competitor.jsonl"
    record = _page().model_dump(mode="json")
    record["fabricated_field"] = True
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        read_pages_jsonl(path)

    path.write_text(json.dumps(_page().model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="newline-terminated"):
        read_pages_jsonl(path)
