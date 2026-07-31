import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from app.pdf2md.engine import render_document
from app.pdf2md.schema import (
    DOCUMENT_SCHEMA,
    LEGACY_DOCUMENT_SCHEMA,
    PREVIOUS_DOCUMENT_SCHEMA,
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FigureStructure,
    InlineFootnoteMarkerRange,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
    TableCell,
    TableStructure,
    inline_footnote_marker_ranges,
    read_document_elements,
    write_document_elements,
)
from app.pdf2md.source_catalog import (
    _LEGACY_SOURCE_CATALOG_SCHEMA,  # pyright: ignore[reportPrivateUsage]
    _legacy_catalog_sha256,  # pyright: ignore[reportPrivateUsage]
    make_source_item,
    read_source_catalog,
)


def fragment() -> PageFragment:
    return PageFragment(
        page_number=1,
        bbox=BoundingBox(x0=10, y0=20, x1=100, y1=40),
        source_item_ids=["liteparse:1:2"],
    )


def annotation() -> AnnotationMetadata:
    return AnnotationMetadata(
        stage="silver",
        revision=1,
        annotator="test",
        confidence=0.9,
        adjudication_status="unreviewed",
    )


def markdown_table() -> TableStructure:
    return TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation="markdown",
        cells=[
            TableCell(
                row_index=row,
                column_index=column,
                role="header" if row == 0 else "body",
                text=text,
                fragments=[fragment()],
            )
            for row, values in enumerate((("Name", "Value"), ("A", "1")))
            for column, text in enumerate(values)
        ],
    )


def table_element() -> DocumentElement:
    return DocumentElement(
        document_id="document-1",
        element_id="table-1",
        order=0,
        element_type="table",
        content="| Name | Value |\n| --- | --- |\n| A | 1 |",
        format="markdown",
        fragments=[fragment()],
        structure=ElementStructure(table=markdown_table()),
        annotation=annotation(),
    )


def test_internal_table_provenance_does_not_change_public_schema() -> None:
    assert tuple(TableStructure.model_fields) == (
        "row_count",
        "column_count",
        "header_row_count",
        "representation",
        "classification_reasons",
        "cells",
    )
    assert tuple(TableCell.model_fields) == (
        "row_index",
        "column_index",
        "rowspan",
        "colspan",
        "role",
        "text",
        "fragments",
    )
    assert tuple(PageFragment.model_fields) == (
        "page_number",
        "page_width",
        "page_height",
        "bbox",
        "source_item_ids",
    )
    structure_type = DOCUMENT_SCHEMA.field("structure").type
    assert [field.name for field in structure_type] == [
        "paragraph",
        "table",
        "figure",
        "footnote",
        "style_runs",
        "linked_element_ids",
        "properties",
    ]
    assert table_element().structure.properties == []


def test_markdown_table_requires_rectangular_unspanned_grid() -> None:
    cells = list(markdown_table().cells)
    cells[0] = cells[0].model_copy(update={"colspan": 2})

    with pytest.raises(ValidationError, match="spanning cells|overlap"):
        TableStructure(
            row_count=2,
            column_count=2,
            header_row_count=1,
            representation="markdown",
            cells=cells,
        )


def test_html_table_requires_classification_reason() -> None:
    with pytest.raises(ValidationError, match="classification reason"):
        TableStructure(
            row_count=1,
            column_count=1,
            header_row_count=0,
            representation="html",
            cells=[
                TableCell(
                    row_index=0,
                    column_index=0,
                    role="body",
                    text="value",
                    fragments=[fragment()],
                )
            ],
        )


def test_source_catalog_v1_migrates_in_memory_and_new_ids_are_document_scoped(tmp_path: Path) -> None:
    document_id = "a" * 64
    record = {
        "source_item_id": "pymupdf:p000001:word:w000001",
        "document_id": document_id,
        "page_number": 1,
        "kind": "word",
        "coordinate_frame": "detector_page",
        "x0": 1.0,
        "y0": 2.0,
        "x1": 3.0,
        "y1": 4.0,
        "text": "value",
        "content_sha256": hashlib.sha256(b"value").hexdigest(),
    }
    legacy_hash = _legacy_catalog_sha256(document_id, [record])
    schema = _LEGACY_SOURCE_CATALOG_SCHEMA.with_metadata({
        **(_LEGACY_SOURCE_CATALOG_SCHEMA.metadata or {}),
        b"pdf2md.document_id": document_id.encode(),
        b"pdf2md.source_catalog_sha256": legacy_hash.encode(),
    })
    path = tmp_path / "legacy.source-items.parquet"
    pq.write_table(pa.Table.from_pylist([record], schema=schema), path)

    migrated = read_source_catalog(path)
    generated = make_source_item(
        document_id=document_id,
        page_number=1,
        kind="word",
        coordinate_frame="detector_page",
        coordinates=(1, 2, 3, 4),
        canonical_coordinates=(1, 2, 3, 4),
        text="value",
    )

    assert migrated.schema_version == "2.0.0"
    assert migrated.migrated_from_schema_version == "1.0.0"
    assert migrated.items[0].id_scheme == "legacy-ordinal-v1"
    assert generated.id_scheme == "native-content-v1"
    assert generated.source_item_id.startswith(f"pymupdf:d{document_id}:p000001:word:")
    assert "extractor upgrades or rewrites" in migrated.id_stability_scope


def test_parquet_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "document-1.parquet"
    expected = [table_element()]

    write_document_elements(expected, output_path)

    assert read_document_elements(output_path) == expected


def test_parquet_refuses_mixed_documents(tmp_path: Path) -> None:
    first = table_element()
    second = first.model_copy(update={"document_id": "document-2", "element_id": "table-2", "order": 1})

    with pytest.raises(ValueError, match="exactly one document_id"):
        write_document_elements([first, second], tmp_path / "mixed.parquet")


def _write_raw_records(
    path: Path,
    records: list[dict[str, object]],
    *,
    schema: pa.Schema = DOCUMENT_SCHEMA,
) -> None:
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path, compression="zstd")


def test_rich_document_roundtrip_is_lossless_render_equivalent_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    first_page = fragment().model_copy(
        update={
            "page_width": 612.0,
            "page_height": 792.0,
            "source_item_ids": ["native:p1:α"],
        }
    )
    second_page = PageFragment(
        page_number=2,
        page_width=792,
        page_height=612,
        bbox=BoundingBox(x0=30, y0=40, x1=300, y1=80),
        source_item_ids=["native:p2:旋転"],
    )
    paragraph = DocumentElement(
        document_id="document-κ",
        element_id="paragraph-多頁",
        order=0,
        element_type="paragraph",
        content="Café naïve — Ελληνικά — 日本語\nsecond page",
        format="text",
        fragments=[first_page, second_page],
        structure=ElementStructure(
            paragraph=ParagraphStructure(role="body"),
            properties=[],
        ),
        annotation=AnnotationMetadata(
            stage="candidate",
            revision=2,
            parent_revision=1,
            annotator="auditor-Δ",
            confidence=0.875,
            adjudication_status="corrected",
        ),
    )
    table = table_element().model_copy(
        update={
            "document_id": paragraph.document_id,
            "element_id": "table-跨頁",
            "order": 1,
            "fragments": [first_page, second_page],
            "structure": ElementStructure(
                table=markdown_table().model_copy(
                    update={
                        "cells": [
                            cell.model_copy(
                                update={
                                    "text": f"{cell.text} Ω" if cell.row_index else cell.text,
                                    "fragments": [first_page, second_page]
                                    if (cell.row_index, cell.column_index) == (1, 0)
                                    else [first_page],
                                }
                            )
                            for cell in markdown_table().cells
                        ]
                    }
                )
            ),
            "annotation": paragraph.annotation,
        }
    )
    figure = DocumentElement(
        document_id=paragraph.document_id,
        element_id="figure-rotated-90",
        order=2,
        element_type="figure",
        content="",
        format="text",
        fragments=[second_page],
        structure=ElementStructure(
            figure=FigureStructure(
                asset_path="assets/図-90.png",
                sha256="a" * 64,
                width=270,
                height=40,
                caption_element_id=None,
            ),
            linked_element_ids=[],
            properties=[],
        ),
        annotation=paragraph.annotation,
    )
    expected = [paragraph, table, figure]
    first_path = tmp_path / "rich-first.parquet"
    second_path = tmp_path / "rich-second.parquet"

    write_document_elements(list(reversed(expected)), first_path)
    write_document_elements(expected, second_path)

    actual = read_document_elements(first_path)
    assert actual == expected
    assert render_document(actual) == render_document(expected)
    assert first_path.read_bytes() == second_path.read_bytes()


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_geometry_fails_fast(coordinate: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        BoundingBox(x0=coordinate, y0=0, x1=10, y1=10)


@pytest.mark.parametrize(
    ("revision", "parent_revision"),
    [(1, 1), (2, 2), (2, 3)],
)
def test_annotation_revision_parent_must_be_an_earlier_revision(revision: int, parent_revision: int) -> None:
    with pytest.raises(ValidationError, match="parent_revision"):
        AnnotationMetadata(
            stage="candidate",
            revision=revision,
            parent_revision=parent_revision,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        )


def test_fragment_null_dimensions_and_source_ids_fail_fast() -> None:
    with pytest.raises(ValidationError, match="both be present or both be null"):
        PageFragment(
            page_number=1,
            page_width=612,
            bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
        )
    with pytest.raises(ValidationError, match="empty values"):
        PageFragment(
            page_number=1,
            bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            source_item_ids=[""],
        )
    with pytest.raises(ValidationError, match="unique"):
        PageFragment(
            page_number=1,
            bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            source_item_ids=["native:1", "native:1"],
        )


def test_read_rejects_bad_metadata_physical_schema_and_row_version(tmp_path: Path) -> None:
    record = table_element().model_dump(mode="json")

    wrong_metadata = DOCUMENT_SCHEMA.with_metadata({b"pdf2md.schema_version": b"999.0.0"})
    metadata_path = tmp_path / "wrong-metadata.parquet"
    _write_raw_records(metadata_path, [record], schema=wrong_metadata)
    with pytest.raises(ValueError, match="schema version"):
        read_document_elements(metadata_path)

    wrong_physical = DOCUMENT_SCHEMA.append(pa.field("unexpected", pa.string()))
    physical_path = tmp_path / "wrong-physical.parquet"
    _write_raw_records(physical_path, [{**record, "unexpected": "value"}], schema=wrong_physical)
    with pytest.raises(ValueError, match="physical schema"):
        read_document_elements(physical_path)

    version_path = tmp_path / "wrong-row-version.parquet"
    _write_raw_records(version_path, [{**record, "schema_version": "999.0.0"}])
    with pytest.raises(ValidationError, match="schema_version"):
        read_document_elements(version_path)


def test_read_rejects_duplicate_ids_orders_noncanonical_rows_and_wrong_stages(tmp_path: Path) -> None:
    first = table_element().model_dump(mode="json")
    second = table_element().model_copy(update={"element_id": "table-2", "order": 1}).model_dump(mode="json")

    duplicate_id_path = tmp_path / "duplicate-id.parquet"
    _write_raw_records(duplicate_id_path, [first, {**second, "element_id": first["element_id"]}])
    with pytest.raises(ValueError, match="element_id values must be unique"):
        read_document_elements(duplicate_id_path)

    duplicate_order_path = tmp_path / "duplicate-order.parquet"
    _write_raw_records(duplicate_order_path, [first, {**second, "order": 0}])
    with pytest.raises(ValueError, match="order must be unique"):
        read_document_elements(duplicate_order_path)

    noncanonical_path = tmp_path / "noncanonical.parquet"
    _write_raw_records(noncanonical_path, [second, first])
    with pytest.raises(ValueError, match="canonical element order"):
        read_document_elements(noncanonical_path)

    wrong_stage_path = tmp_path / "wrong-stage.parquet"
    bad_annotation = {**first["annotation"], "stage": "bronze"}
    _write_raw_records(wrong_stage_path, [{**first, "annotation": bad_annotation}])
    with pytest.raises(ValidationError, match="stage"):
        read_document_elements(wrong_stage_path)


def test_document_collection_rejects_mixed_annotation_stages_and_empty_parquet(tmp_path: Path) -> None:
    first = table_element()
    second = first.model_copy(
        update={
            "element_id": "table-2",
            "order": 1,
            "annotation": first.annotation.model_copy(update={"stage": "candidate"}),
        }
    )
    with pytest.raises(ValueError, match="exactly one annotation stage"):
        write_document_elements([first, second], tmp_path / "mixed-stages.parquet")

    empty_path = tmp_path / "empty.parquet"
    _write_raw_records(empty_path, [])
    with pytest.raises(ValueError, match="at least one element"):
        read_document_elements(empty_path)


def test_read_rejects_malformed_nested_table_structure(tmp_path: Path) -> None:
    record = table_element().model_dump(mode="json")
    structure = dict(record["structure"])
    table = dict(structure["table"])
    table["cells"] = []
    structure["table"] = table
    malformed_path = tmp_path / "malformed-nested.parquet"
    _write_raw_records(malformed_path, [{**record, "structure": structure}])

    with pytest.raises(ValidationError, match="cells"):
        read_document_elements(malformed_path)


def test_physical_schema_rejects_metadata_loss_addition_reordering_and_field_drift(
    tmp_path: Path,
) -> None:
    record = table_element().model_dump(mode="json")
    fields = list(DOCUMENT_SCHEMA)
    variants = {
        "missing-metadata": DOCUMENT_SCHEMA.remove_metadata(),
        "extra-metadata": DOCUMENT_SCHEMA.with_metadata({
            **(DOCUMENT_SCHEMA.metadata or {}),
            b"unexpected": b"duplicate-or-foreign-metadata",
        }),
        "reordered": pa.schema([fields[1], fields[0], *fields[2:]], metadata=DOCUMENT_SCHEMA.metadata),
        "missing-column": pa.schema(fields[:-1], metadata=DOCUMENT_SCHEMA.metadata),
        "type-drift": pa.schema(
            [
                pa.field("order", pa.int64(), nullable=False) if field.name == "order" else field
                for field in fields
            ],
            metadata=DOCUMENT_SCHEMA.metadata,
        ),
        "nullability-drift": pa.schema(
            [
                pa.field("content", pa.string(), nullable=True) if field.name == "content" else field
                for field in fields
            ],
            metadata=DOCUMENT_SCHEMA.metadata,
        ),
    }

    for name, schema in variants.items():
        path = tmp_path / f"{name}.parquet"
        _write_raw_records(path, [record], schema=schema)
        with pytest.raises(ValueError, match="schema version|physical schema|metadata"):
            read_document_elements(path)


def test_geometry_dimensions_and_confidence_are_finite_and_ranged() -> None:
    with pytest.raises(ValidationError, match="positive width and height"):
        BoundingBox(x0=0, y0=0, x1=0, y1=1)

    for dimension in (0, -1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            PageFragment(
                page_number=1,
                page_width=dimension,
                page_height=10,
                bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            )

    for confidence in (-0.01, 1.01, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            AnnotationMetadata(
                stage="candidate",
                revision=1,
                annotator="test",
                confidence=confidence,
                adjudication_status="unreviewed",
            )


def test_nested_table_fragments_must_be_local_to_the_element() -> None:
    element = table_element()
    payload = element.model_dump(mode="json")
    payload["fragments"][0]["page_width"] = 612
    payload["fragments"][0]["page_height"] = 792
    payload["structure"]["table"]["cells"][0]["fragments"][0].update({"page_width": 612, "page_height": 792})
    assert DocumentElement.model_validate(payload)

    wrong_page = element.model_dump(mode="json")
    wrong_page["structure"]["table"]["cells"][0]["fragments"][0]["page_number"] = 2
    with pytest.raises(ValidationError, match="page represented"):
        DocumentElement.model_validate(wrong_page)

    duplicate_source = element.model_dump(mode="json")
    duplicate_source["structure"]["table"]["cells"][0]["fragments"][0]["source_item_ids"] = [
        "liteparse:1:2",
        "liteparse:1:2",
    ]
    with pytest.raises(ValidationError, match="source_item_ids must be unique"):
        DocumentElement.model_validate(duplicate_source)

    wrong_dimensions = payload
    wrong_dimensions["structure"]["table"]["cells"][0]["fragments"][0]["page_width"] = 613
    with pytest.raises(ValidationError, match="page dimensions"):
        DocumentElement.model_validate(wrong_dimensions)


def test_collection_rejects_empty_write_order_gaps_and_mixed_stages_on_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one element"):
        write_document_elements([], tmp_path / "empty-write.parquet")

    first = table_element().model_dump(mode="json")
    gap = table_element().model_copy(update={"element_id": "table-2", "order": 2}).model_dump(mode="json")
    gap_path = tmp_path / "order-gap.parquet"
    _write_raw_records(gap_path, [first, gap])
    with pytest.raises(ValueError, match="contiguous from zero"):
        read_document_elements(gap_path)

    candidate_annotation = {**gap["annotation"], "stage": "candidate"}
    mixed_path = tmp_path / "mixed-stage-read.parquet"
    _write_raw_records(
        mixed_path,
        [first, {**gap, "order": 1, "annotation": candidate_annotation}],
    )
    with pytest.raises(ValueError, match="exactly one annotation stage"):
        read_document_elements(mixed_path)


def _inline_marker_element(
    value: str = '[{"start":5,"end":9,"label":"1"}]',
) -> DocumentElement:
    return DocumentElement(
        document_id="document-markers",
        element_id="paragraph-1",
        order=0,
        element_type="paragraph",
        content="Word [^1] tail",
        format="text",
        fragments=[fragment()],
        structure=ElementStructure(
            paragraph=ParagraphStructure(role="body"),
            properties=[StructureProperty(key="inline_footnote_markers", value=value)],
        ),
        annotation=annotation(),
    )


def test_document_element_construction_validates_exact_inline_footnote_marker_ranges() -> None:
    element = _inline_marker_element()

    assert inline_footnote_marker_ranges(element) == (InlineFootnoteMarkerRange(start=5, end=9, label="1"),)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-json", "valid JSON"),
        ('{"start":5,"end":9,"label":"1"}', "JSON array"),
        ("[1]", "range objects"),
        ('[{"start":5,"end":9}]', "label"),
        ('[{"start":5,"end":9,"label":"1","extra":true}]', "extra"),
        (
            '[{"start":5,"end":9,"label":"1"},{"start":4,"end":8,"label":"1"}]',
            "source order",
        ),
        (
            '[{"start":5,"end":9,"label":"1"},{"start":8,"end":12,"label":"1"}]',
            "overlap",
        ),
        ('[{"start":20,"end":24,"label":"1"}]', "exceeds element content"),
        ('[{"start":5,"end":9,"label":"2"}]', "does not match"),
        ('["1",{"start":5,"end":9,"label":"1"}]', "range objects"),
    ],
)
def test_document_element_construction_rejects_invalid_modern_marker_metadata(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _inline_marker_element(value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start", True),
        ("end", False),
        ("start", 5.0),
        ("end", 9.0),
        ("start", "5"),
        ("end", "9"),
    ],
)
def test_inline_marker_offsets_require_exact_integers(field: str, value: object) -> None:
    marker: dict[str, object] = {"start": 5, "end": 9, "label": "1"}
    marker[field] = value

    with pytest.raises(ValidationError, match=field):
        _inline_marker_element(json.dumps([marker]))


@pytest.mark.parametrize(
    "value",
    [
        '[{"start":5,"start":6,"end":9,"label":"1"}]',
        '[{"start":5,"end":9,"label":{"nested":{"x":1,"x":2}}}]',
    ],
)
def test_inline_marker_json_rejects_duplicate_object_keys_at_every_depth(value: str) -> None:
    with pytest.raises(ValidationError, match="duplicate JSON object key"):
        _inline_marker_element(value)


def test_document_element_rejects_duplicate_inline_marker_properties() -> None:
    payload = _inline_marker_element().model_dump(mode="json")
    payload["structure"]["properties"].append({"key": "inline_footnote_markers", "value": '["1"]'})

    with pytest.raises(ValidationError, match="multiple inline_footnote_markers"):
        DocumentElement.model_validate(payload)


def test_model_validate_json_enforces_inline_marker_schema_boundary() -> None:
    payload = _inline_marker_element().model_dump(mode="json")
    payload["structure"]["properties"][0]["value"] = '[{"start":5,"end":9,"label":"2"}]'

    with pytest.raises(ValidationError, match="does not match"):
        DocumentElement.model_validate_json(json.dumps(payload))


def test_model_validate_json_rejects_duplicate_marker_range_keys() -> None:
    payload = _inline_marker_element().model_dump(mode="json")
    payload["structure"]["properties"][0]["value"] = '[{"start":5,"start":6,"end":9,"label":"1"}]'

    with pytest.raises(ValidationError, match="duplicate JSON object key"):
        DocumentElement.model_validate_json(json.dumps(payload))


def test_inline_marker_parquet_roundtrip_and_write_refusal(tmp_path: Path) -> None:
    element = _inline_marker_element()
    path = tmp_path / "inline-markers.parquet"
    write_document_elements([element], path)
    assert read_document_elements(path) == [element]

    invalid_structure = element.structure.model_copy(
        update={
            "properties": [
                StructureProperty(
                    key="inline_footnote_markers",
                    value='[{"start":5,"end":9,"label":"2"}]',
                )
            ]
        }
    )
    invalid = element.model_copy(update={"structure": invalid_structure})

    with pytest.raises(ValidationError, match="does not match"):
        write_document_elements([invalid], tmp_path / "invalid-inline-markers.parquet")

    invalid_record = element.model_dump(mode="json")
    invalid_record["structure"]["properties"][0]["value"] = '[{"start":5,"end":9,"label":"2"}]'
    invalid_path = tmp_path / "invalid-inline-markers-read.parquet"
    _write_raw_records(invalid_path, [invalid_record])
    with pytest.raises(ValidationError, match="does not match"):
        read_document_elements(invalid_path)

    for name, value, message in (
        ("float-offset", '[{"start":5.0,"end":9,"label":"1"}]', "start"),
        (
            "duplicate-key",
            '[{"start":5,"start":6,"end":9,"label":"1"}]',
            "duplicate JSON object key",
        ),
    ):
        record = element.model_dump(mode="json")
        record["structure"]["properties"][0]["value"] = value
        read_path = tmp_path / f"{name}.parquet"
        _write_raw_records(read_path, [record])
        with pytest.raises(ValidationError, match=message):
            read_document_elements(read_path)


def test_parquet_write_rejects_unchecked_extra_model_attributes(tmp_path: Path) -> None:
    element = _inline_marker_element()
    extra_element = element.model_copy(update={"unexpected": "silently dropped before"})
    extra_structure = element.structure.model_copy(update={"unexpected_nested": True})
    nested_extra_element = element.model_copy(update={"structure": extra_structure})

    for name, invalid in (("element", extra_element), ("nested", nested_extra_element)):
        with pytest.raises(ValueError, match="undeclared model attributes"):
            write_document_elements([invalid], tmp_path / f"extra-{name}.parquet")


def test_legacy_label_only_inline_marker_arrays_remain_conservative_valid_claims(
    tmp_path: Path,
) -> None:
    element = _inline_marker_element('["1","1"]')
    path = tmp_path / "legacy-label-only-markers.parquet"

    assert inline_footnote_marker_ranges(element) == ()
    assert DocumentElement.model_validate_json(element.model_dump_json()) == element
    write_document_elements([element], path)
    assert read_document_elements(path) == [element]


def _legacy_inline_marker_record(value: str, version: str) -> dict[str, object]:
    record = _inline_marker_element().model_dump(mode="json")
    record["schema_version"] = version
    record["structure"]["properties"] = [
        {"key": "source", "value": "preserved"},
        {"key": "inline_footnote_markers", "value": value},
    ]
    if version == "1.0.0":
        record["structure"].pop("style_runs")
        for key in ("title_evidence", "heading_evidence", "list_evidence"):
            record["structure"]["paragraph"].pop(key)
    return record


@pytest.mark.parametrize(
    ("version", "schema"),
    [("1.0.0", LEGACY_DOCUMENT_SCHEMA), ("1.1.0", PREVIOUS_DOCUMENT_SCHEMA)],
)
@pytest.mark.parametrize(
    "value",
    [
        "not-json",
        "['1']",
        '[{"label":"1"}]',
        '{"labels":["1"]}',
        '[{"start":5,"end":9,"label":"2"}]',
    ],
)
def test_legacy_parquet_migrates_invalid_inline_marker_shapes_to_no_range_claim(
    tmp_path: Path,
    version: str,
    schema: pa.Schema,
    value: str,
) -> None:
    record = _legacy_inline_marker_record(value, version)
    first_path = tmp_path / f"legacy-{version}-first.parquet"
    second_path = tmp_path / f"legacy-{version}-second.parquet"
    _write_raw_records(first_path, [record], schema=schema)
    _write_raw_records(second_path, [record], schema=schema)

    first = read_document_elements(first_path)
    second = read_document_elements(second_path)

    assert first == second
    assert inline_footnote_marker_ranges(first[0]) == ()
    assert first[0].structure.properties == [StructureProperty(key="source", value="preserved")]


@pytest.mark.parametrize(
    ("version", "schema"),
    [("1.0.0", LEGACY_DOCUMENT_SCHEMA), ("1.1.0", PREVIOUS_DOCUMENT_SCHEMA)],
)
def test_legacy_parquet_preserves_valid_label_only_marker_arrays(
    tmp_path: Path,
    version: str,
    schema: pa.Schema,
) -> None:
    record = _legacy_inline_marker_record('["脚注","1"]', version)
    path = tmp_path / f"legacy-{version}-labels.parquet"
    _write_raw_records(path, [record], schema=schema)

    migrated = read_document_elements(path)[0]

    assert inline_footnote_marker_ranges(migrated) == ()
    assert migrated.structure.properties[-1] == StructureProperty(
        key="inline_footnote_markers",
        value='["脚注","1"]',
    )


@pytest.mark.parametrize("value", ["not-json", "['1']", '[{"label":"1"}]'])
def test_current_parquet_never_migrates_invalid_legacy_marker_shapes(
    tmp_path: Path,
    value: str,
) -> None:
    record = _inline_marker_element().model_dump(mode="json")
    record["structure"]["properties"][0]["value"] = value
    path = tmp_path / "current-invalid-marker.parquet"
    _write_raw_records(path, [record])

    with pytest.raises(ValidationError):
        read_document_elements(path)
