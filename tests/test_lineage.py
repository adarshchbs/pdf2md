import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from app.pdf2md.benchmark_batch import BenchmarkBatchManifest
from app.pdf2md.evaluation import evaluate_reference_churn
from app.pdf2md.lineage import (
    ChurnArtifactStatus,
    LineageArtifactIdentity,
    LineageTransitionClaim,
    LineageValidationError,
    audit_cycle_lineage,
    audit_lineage_chain,
    serialize_cycle_lineage_report,
)
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
    read_document_elements,
    write_document_elements,
)
from app.pdf2md.source_catalog import (
    SourceItem,
    SourceItemKind,
    build_source_catalog,
    make_source_item,
    source_catalog_path,
    write_source_catalog,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _element(
    document_id: str,
    *,
    stage: str = "silver",
    source_id: str | list[str] = "liteparse:p1:item0",
) -> DocumentElement:
    return DocumentElement(
        document_id=document_id,
        element_id="sample-0000",
        order=0,
        element_type="paragraph",
        content="Grounded text",
        format="text",
        fragments=[
            PageFragment(
                page_number=1,
                bbox=BoundingBox(x0=1, y0=2, x1=20, y1=10),
                source_item_ids=[source_id] if isinstance(source_id, str) else source_id,
            )
        ],
        structure=ElementStructure(paragraph=ParagraphStructure(role="body")),
        annotation=AnnotationMetadata(
            stage=stage,  # type: ignore[arg-type]
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="accepted",
        ),
    )


def _fixture(
    root: Path,
    *,
    churn_current_count: int = 1,
    source_id: str = "liteparse:p1:item0",
    pymupdf_item_count: int = 0,
    pymupdf_kind: SourceItemKind = "word",
) -> None:
    source = root / "data/corpus/dev/sample.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable source")
    source_sha = _sha256(source)

    bronze_dir = root / "data/bronze/sample"
    bronze_dir.mkdir(parents=True)
    liteparse = bronze_dir / "liteparse.json"
    liteparse.write_text(
        json.dumps({"pages": [{"page": 1, "text_items": [{"text": "Grounded text"}]}]}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "source_name": "sample.pdf",
        "source_path": "data/corpus/dev/sample.pdf",
        "source_sha256": source_sha,
        "source_size_bytes": source.stat().st_size,
        "source_page_count": 1,
        "selection": {"requested_pages": [1], "annotated_pages": [1], "context_pages": []},
        "config": {
            "render_dpi": 144,
            "liteparse_dpi": 150,
            "min_native_characters_per_page": 20,
            "liteparse_executable": "lit",
        },
        "pymupdf_version": "test",
        "liteparse_version": "test",
        "artifacts": [
            {
                "path": "liteparse.json",
                "sha256": _sha256(liteparse),
                "size_bytes": liteparse.stat().st_size,
            }
        ],
    }
    (bronze_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    batch = {
        "batch_id": "test-cycle",
        "selection_count": 1,
        "selections": [
            {
                "id": "sample",
                "source_path": "data/corpus/dev/sample.pdf",
                "source_sha256": source_sha,
                "pages": {"annotated": [1], "context": [], "requested": [1]},
                "bundle_path": "data/bronze/sample",
                "verification": {"status": "verified", "artifact_count": 1},
            }
        ],
        "verification_summary": {
            "verified_bundle_count": 1,
            "failed_bundle_count": 0,
            "artifact_count": 1,
        },
    }
    (root / "data").mkdir(exist_ok=True)
    (root / "data/batch.json").write_text(json.dumps(batch), encoding="utf-8")

    source_items = [
        make_source_item(
            document_id=source_sha,
            page_number=1,
            kind=pymupdf_kind,
            coordinate_frame="source_page",
            coordinates=(
                (float(index), 2, float(index + 1), 2)
                if pymupdf_kind == "rule"
                else (float(index), 2, float(index + 1), 10)
            ),
            text=None if pymupdf_kind == "rule" else f"{pymupdf_kind}-{index}",
        )
        for index in range(pymupdf_item_count)
    ]
    source_ids: str | list[str] = (
        [item.source_item_id for item in source_items] if source_items else source_id
    )
    silver = [_element(source_sha, source_id=source_ids)]
    silver_path = root / "data/silver/sample.parquet"
    write_document_elements(silver, silver_path)
    if source_items:
        write_source_catalog(build_source_catalog(source_sha, source_items), source_catalog_path(silver_path))
    candidate = [_element(source_sha, stage="candidate", source_id=source_ids)]
    write_document_elements(candidate, root / "data/candidate-existing/sample.parquet")
    churn = evaluate_reference_churn(silver, silver, candidate=candidate).model_dump(mode="json")
    if churn_current_count != 1:
        churn.update(
            previous_count=churn_current_count,
            current_count=churn_current_count,
            unchanged_count=churn_current_count,
        )
    churn_dir = root / "data/churn"
    churn_dir.mkdir()
    (churn_dir / "sample.json").write_text(json.dumps(churn), encoding="utf-8")


def _replace_fixture_with_native_table(
    root: Path,
    cell_source_ids: list[str],
    cell_texts: list[str],
    *,
    cell_permutation: list[int] | None = None,
) -> None:
    if len(cell_source_ids) != len(cell_texts) or not cell_source_ids:
        raise ValueError("table test cells require matching non-empty source and text lists")
    for path in (
        root / "data/silver/sample.parquet",
        root / "data/candidate-existing/sample.parquet",
    ):
        element = read_document_elements(path)[0]
        cells = [
            TableCell(
                row_index=0,
                column_index=index,
                role="header",
                text=text,
                fragments=[element.fragments[0].model_copy(update={"source_item_ids": [source_id]})],
            )
            for index, (source_id, text) in enumerate(zip(cell_source_ids, cell_texts, strict=True))
        ]
        if cell_permutation is not None:
            cells = [cells[index] for index in cell_permutation]
        table = element.model_copy(
            update={
                "element_type": "table",
                "content": "| " + " | ".join(cell_texts) + " |",
                "format": "markdown",
                "structure": ElementStructure(
                    table=TableStructure(
                        row_count=1,
                        column_count=len(cells),
                        header_row_count=1,
                        representation="markdown",
                        cells=cells,
                    )
                ),
            }
        )
        write_document_elements([table], path, overwrite=True)


def _replace_fixture_with_ledger_only_table(
    root: Path,
    *,
    ledger_value: str,
    source_items: list[SourceItem],
    structural_source_ids: list[str] | None = None,
) -> None:
    silver_path = root / "data/silver/sample.parquet"
    document_id = read_document_elements(silver_path)[0].document_id
    properties = [StructureProperty(key="table_span_suppression_v1", value=ledger_value)]
    if structural_source_ids is not None:
        properties.append(
            StructureProperty(
                key="table_structural_source_item_ids",
                value=json.dumps(structural_source_ids),
            )
        )
    for path in (silver_path, root / "data/candidate-existing/sample.parquet"):
        element = read_document_elements(path)[0]
        fragment = element.fragments[0].model_copy(update={"source_item_ids": []})
        table = element.model_copy(
            update={
                "element_type": "table",
                "content": "",
                "format": "markdown",
                "fragments": [fragment],
                "structure": ElementStructure(
                    table=TableStructure(
                        row_count=1,
                        column_count=1,
                        header_row_count=1,
                        representation="markdown",
                        cells=[
                            TableCell(
                                row_index=0,
                                column_index=0,
                                role="header",
                                text="",
                                fragments=[fragment],
                            )
                        ],
                    ),
                    properties=properties,
                ),
            }
        )
        write_document_elements([table], path, overwrite=True)
    write_source_catalog(
        build_source_catalog(document_id, source_items),
        source_catalog_path(silver_path),
    )


def _suppression_ledger(table_id: str, span_ids: list[str]) -> str:
    return json.dumps({
        "counts": {"retained_overlap": 0, "supported": 0, "suppressed": len(span_ids)},
        "disposition": "suppressed_supported_spans" if span_ids else "no_overlap",
        "page_number": 1,
        "retained_overlap_span_ids": [],
        "supported_span_ids": [],
        "suppressed_span_ids": span_ids,
        "table_id": table_id,
    })


def test_audit_classifies_grounded_legacy_churn_as_current_snapshot_only(tmp_path: Path) -> None:
    _fixture(tmp_path)

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    document = report.documents[0]
    assert report.churn_status_counts == {
        "replayable": 0,
        "current-snapshot-only": 1,
        "stale": 0,
        "invalid": 0,
    }
    assert document.silver.snapshot_parent_links_valid
    assert not document.silver.historical_ancestors_available
    assert document.churn.status == "current-snapshot-only"
    assert document.churn.reason_code == "historical_inputs_unavailable"
    assert report.schema_version == "1.2.0"
    assert [match.path for match in document.churn.candidate_identity_matches] == [
        "data/candidate-existing/sample.parquet"
    ]


def test_audit_preserves_and_classifies_stale_churn(tmp_path: Path) -> None:
    _fixture(tmp_path, churn_current_count=2)

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].churn.status == "stale"
    assert report.documents[0].churn.reason_code == "current_count_mismatch"
    assert "claimed current_count 2" in report.documents[0].churn.reason


def test_audit_validates_existing_catalog_companion_for_liteparse_only_silver(tmp_path: Path) -> None:
    _fixture(tmp_path)
    silver_path = tmp_path / "data/silver/sample.parquet"
    document_id = _sha256(tmp_path / "data/corpus/dev/sample.pdf")
    write_source_catalog(build_source_catalog(document_id, []), source_catalog_path(silver_path))

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].silver.document_id == document_id
    assert report.documents[0].silver.catalog_reference_counts.total_reference_count == 1


@pytest.mark.parametrize("companion_kind", ["malformed", "unrelated"])
def test_audit_rejects_invalid_existing_catalog_companion_for_liteparse_only_silver(
    tmp_path: Path, companion_kind: str
) -> None:
    _fixture(tmp_path)
    catalog_path = source_catalog_path(tmp_path / "data/silver/sample.parquet")
    if companion_kind == "malformed":
        catalog_path.write_bytes(b"not parquet")
    else:
        write_source_catalog(build_source_catalog("f" * 64, []), catalog_path)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == (
        "source_catalog_invalid" if companion_kind == "malformed" else "source_catalog_document_id_mismatch"
    )
    assert caught.value.path == catalog_path


def test_audit_accepts_document_scoped_pymupdf_ids_with_matching_catalog(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=2)

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].silver.source_item_reference_count == 2
    assert report.documents[0].silver.unique_source_item_count == 2
    assert report.documents[0].churn.status == "current-snapshot-only"


def test_audit_accepts_valid_catalog_backed_ledger_only_table(tmp_path: Path) -> None:
    _fixture(tmp_path)
    document_id = _sha256(tmp_path / "data/corpus/dev/sample.pdf")
    span = make_source_item(
        document_id=document_id,
        page_number=1,
        kind="span",
        coordinate_frame="source_page",
        coordinates=(1, 2, 20, 10),
        text="overlap",
    )
    rule = make_source_item(
        document_id=document_id,
        page_number=1,
        kind="rule",
        coordinate_frame="source_page",
        coordinates=(1, 2, 20, 2),
        text=None,
    )
    _replace_fixture_with_ledger_only_table(
        tmp_path,
        ledger_value=_suppression_ledger("sample-0000", [span.source_item_id]),
        source_items=[span, rule],
        structural_source_ids=[rule.source_item_id],
    )

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    silver = report.documents[0].silver
    assert silver.source_item_reference_count == 0
    assert silver.unique_source_item_count == 0
    assert silver.catalog_reference_counts.model_dump() == {
        "fragment_reference_count": 0,
        "unique_fragment_source_item_count": 0,
        "table_structural_reference_count": 1,
        "unique_table_structural_source_item_count": 1,
        "table_suppression_ledger_count": 1,
        "table_supported_span_reference_count": 0,
        "unique_table_supported_span_count": 0,
        "table_suppressed_span_reference_count": 1,
        "unique_table_suppressed_span_count": 1,
        "table_retained_overlap_span_reference_count": 0,
        "unique_table_retained_overlap_span_count": 0,
        "table_suppression_span_reference_count": 1,
        "unique_table_suppression_span_count": 1,
        "total_reference_count": 2,
        "total_unique_source_item_count": 2,
    }


@pytest.mark.parametrize(
    ("ledger_value", "detail"),
    [
        ("not-json", "Expecting value"),
        ("[]", "must be a JSON object"),
        (
            json.dumps({
                "counts": {"retained_overlap": 0, "supported": 0, "suppressed": 1},
                "disposition": "suppressed_supported_spans",
                "page_number": 1,
                "retained_overlap_span_ids": [],
                "supported_span_ids": [],
                "suppressed_span_ids": [],
                "table_id": "sample-0000",
            }),
            "counts do not match ID lists",
        ),
        (
            json.dumps({
                "counts": {"retained_overlap": 0, "supported": 0, "suppressed": 0},
                "disposition": "suppressed_supported_spans",
                "page_number": 1,
                "retained_overlap_span_ids": [],
                "supported_span_ids": [],
                "suppressed_span_ids": [],
                "table_id": "sample-0000",
            }),
            "disposition does not match ID lists",
        ),
    ],
)
def test_audit_rejects_malformed_table_suppression_ledgers(
    tmp_path: Path, ledger_value: str, detail: str
) -> None:
    _fixture(tmp_path)
    _replace_fixture_with_ledger_only_table(
        tmp_path,
        ledger_value=ledger_value,
        source_items=[],
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert detail in caught.value.detail
    assert caught.value.path == source_catalog_path(tmp_path / "data/silver/sample.parquet")
    assert caught.value.__cause__ is not None


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
def test_audit_rejects_noncanonical_table_suppression_counts(tmp_path: Path, counts: object) -> None:
    _fixture(tmp_path)
    ledger = json.loads(_suppression_ledger("sample-0000", []))
    ledger["counts"] = counts
    _replace_fixture_with_ledger_only_table(
        tmp_path,
        ledger_value=json.dumps(ledger),
        source_items=[],
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert caught.value.detail == (
        "table span suppression counts must contain exactly non-negative integer values"
    )
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize(
    ("property_kind", "detail"),
    [
        ("rule", "dangling structural source_item_id"),
        ("span", "dangling table suppression span source_item_id"),
    ],
)
def test_audit_rejects_dangling_catalog_backed_table_property_ids(
    tmp_path: Path, property_kind: str, detail: str
) -> None:
    _fixture(tmp_path)
    missing = f"pymupdf:d{'0' * 64}:p000001:{property_kind}:h{'1' * 64}:d000001"
    _replace_fixture_with_ledger_only_table(
        tmp_path,
        ledger_value=_suppression_ledger("sample-0000", [missing] if property_kind == "span" else []),
        source_items=[],
        structural_source_ids=[missing] if property_kind == "rule" else None,
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert detail in caught.value.detail
    assert missing in caught.value.detail


@pytest.mark.parametrize(
    ("kind", "frame", "page_number", "detail"),
    [
        ("rule", "source_page", 1, "must reference spans"),
        ("span", "detector_page", 1, "coordinate frame mismatch"),
        ("span", "source_page", 2, "document/page mismatch"),
    ],
)
def test_audit_rejects_wrong_suppression_span_kind_frame_or_page(
    tmp_path: Path,
    kind: SourceItemKind,
    frame: str,
    page_number: int,
    detail: str,
) -> None:
    _fixture(tmp_path)
    document_id = _sha256(tmp_path / "data/corpus/dev/sample.pdf")
    item = make_source_item(
        document_id=document_id,
        page_number=page_number,
        kind=kind,
        coordinate_frame=frame,  # type: ignore[arg-type]
        coordinates=(1, 2, 20, 2) if kind == "rule" else (1, 2, 20, 10),
        text=None if kind == "rule" else "overlap",
    )
    _replace_fixture_with_ledger_only_table(
        tmp_path,
        ledger_value=_suppression_ledger("sample-0000", [item.source_item_id]),
        source_items=[item],
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert detail in caught.value.detail
    assert item.source_item_id in caught.value.detail


def test_audit_rejects_suppression_ledger_wrong_consumer_id(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _replace_fixture_with_ledger_only_table(
        tmp_path,
        ledger_value=_suppression_ledger("other-table", []),
        source_items=[],
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert caught.value.detail == "table span suppression consumer table mismatch"


def test_audit_rejects_ledger_catalog_document_mismatch(tmp_path: Path) -> None:
    _fixture(tmp_path)
    other_document = "f" * 64
    span = make_source_item(
        document_id=other_document,
        page_number=1,
        kind="span",
        coordinate_frame="source_page",
        coordinates=(1, 2, 20, 10),
        text="overlap",
    )
    _replace_fixture_with_ledger_only_table(
        tmp_path,
        ledger_value=_suppression_ledger("sample-0000", [span.source_item_id]),
        source_items=[],
    )
    write_source_catalog(
        build_source_catalog(other_document, [span]),
        source_catalog_path(tmp_path / "data/silver/sample.parquet"),
        overwrite=True,
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_document_id_mismatch"
    assert caught.value.detail.endswith(f"!= {_sha256(tmp_path / 'data/corpus/dev/sample.pdf')}")


@pytest.mark.parametrize(
    "source_id",
    [
        "pymupdf:not-document-scoped",
        f"pymupdf:d{'A' * 64}:p000001:word:h{'b' * 64}:d000001",
        f"pymupdf:d{'a' * 64}:p00001:word:h{'b' * 64}:d000001",
        f"pymupdf:d{'a' * 64}:p000001:glyph:h{'b' * 64}:d000001",
        f"pymupdf:d{'a' * 64}:p000001:word:hshort:d000001",
        f"pymupdf:d{'a' * 64}:p000001:word:h{'b' * 64}:d000000",
    ],
)
def test_audit_rejects_malformed_pymupdf_source_ids(tmp_path: Path, source_id: str) -> None:
    _fixture(tmp_path, source_id=source_id)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_item_id_invalid"


def test_audit_accepts_native_word_table_cell_provenance(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    source_id = (
        read_document_elements(tmp_path / "data/silver/sample.parquet")[0].fragments[0].source_item_ids[0]
    )
    _replace_fixture_with_native_table(tmp_path, [source_id], ["word-0"])

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].silver.source_item_reference_count == 2
    assert report.documents[0].silver.unique_source_item_count == 1


def test_audit_rejects_partial_native_word_table_cell_provenance(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    source_id = (
        read_document_elements(tmp_path / "data/silver/sample.parquet")[0].fragments[0].source_item_ids[0]
    )
    _replace_fixture_with_native_table(
        tmp_path,
        [source_id],
        ["word-0 FABRICATED_NONCE"],
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert "do not reconstruct complete normalized cell text" in caught.value.detail


@pytest.mark.parametrize("kind", ["span", "rule"])
def test_audit_rejects_non_word_native_table_cell_provenance(tmp_path: Path, kind: SourceItemKind) -> None:
    _fixture(tmp_path, pymupdf_item_count=1, pymupdf_kind=kind)
    source_id = (
        read_document_elements(tmp_path / "data/silver/sample.parquet")[0].fragments[0].source_item_ids[0]
    )
    _replace_fixture_with_native_table(tmp_path, [source_id], ["Grounded text"])

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert caught.value.detail == (f"table cell content provenance must reference native words: {source_id}")


def test_audit_rejects_native_provenance_on_empty_table_cell(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    source_id = (
        read_document_elements(tmp_path / "data/silver/sample.parquet")[0].fragments[0].source_item_ids[0]
    )
    _replace_fixture_with_native_table(tmp_path, [source_id], [" "])

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert caught.value.detail == "empty table cells cannot cite native source items"


def test_audit_rejects_native_word_cited_by_multiple_table_cells(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    source_id = (
        read_document_elements(tmp_path / "data/silver/sample.parquet")[0].fragments[0].source_item_ids[0]
    )
    _replace_fixture_with_native_table(tmp_path, [source_id, source_id], ["word-0", "word-0"])

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert caught.value.detail == "a native word cannot be cited by more than one logical table cell"


def test_native_table_cell_validation_is_invariant_to_cell_permutation(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=2)
    source_ids = (
        read_document_elements(tmp_path / "data/silver/sample.parquet")[0].fragments[0].source_item_ids
    )
    _replace_fixture_with_native_table(tmp_path, source_ids, ["word-0", "word-1"])
    original = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    _replace_fixture_with_native_table(tmp_path, source_ids, ["word-0", "word-1"], cell_permutation=[1, 0])
    permuted = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert original.documents[0].silver.source_item_reference_count == 4
    assert permuted.documents[0].silver.source_item_reference_count == 4
    assert original.documents[0].silver.unique_source_item_count == 2
    assert permuted.documents[0].silver.unique_source_item_count == 2


def test_audit_requires_catalog_for_document_scoped_pymupdf_ids(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    source_catalog_path(tmp_path / "data/silver/sample.parquet").unlink()

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_missing"


def test_audit_rejects_pymupdf_id_missing_from_catalog(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    silver_path = tmp_path / "data/silver/sample.parquet"
    element = read_document_elements(silver_path)[0]
    missing = make_source_item(
        document_id=element.document_id,
        page_number=1,
        kind="word",
        coordinate_frame="source_page",
        coordinates=(20, 2, 21, 10),
        text="missing",
    )
    changed = element.model_copy(
        update={
            "fragments": [
                element.fragments[0].model_copy(update={"source_item_ids": [missing.source_item_id]})
            ]
        }
    )
    write_document_elements([changed], silver_path, overwrite=True)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_item_id_unknown"


def test_audit_rejects_catalog_document_mismatch(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    silver_path = tmp_path / "data/silver/sample.parquet"
    other_document = "f" * 64
    other_item = make_source_item(
        document_id=other_document,
        page_number=1,
        kind="word",
        coordinate_frame="source_page",
        coordinates=(1, 2, 3, 4),
        text="other",
    )
    write_source_catalog(
        build_source_catalog(other_document, [other_item]),
        source_catalog_path(silver_path),
        overwrite=True,
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_document_id_mismatch"


def test_audit_rejects_pymupdf_document_and_page_mismatches(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    silver_path = tmp_path / "data/silver/sample.parquet"
    element = read_document_elements(silver_path)[0]
    source_id = element.fragments[0].source_item_ids[0]
    wrong_document_id = source_id.replace(element.document_id, "f" * 64, 1)
    wrong_document = element.model_copy(
        update={
            "fragments": [element.fragments[0].model_copy(update={"source_item_ids": [wrong_document_id]})]
        }
    )
    write_document_elements([wrong_document], silver_path, overwrite=True)

    with pytest.raises(LineageValidationError) as document_error:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))
    assert document_error.value.code == "source_item_document_mismatch"

    wrong_page = element.model_copy(
        update={"fragments": [element.fragments[0].model_copy(update={"page_number": 2})]}
    )
    write_document_elements([wrong_page], silver_path, overwrite=True)
    with pytest.raises(LineageValidationError) as page_error:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))
    assert page_error.value.code == "source_item_page_mismatch"


@pytest.mark.parametrize("indices", [[0, 0], [1, 0]])
def test_audit_rejects_duplicate_or_permuted_catalog_items(tmp_path: Path, indices: list[int]) -> None:
    _fixture(tmp_path, pymupdf_item_count=2)
    catalog_path = source_catalog_path(tmp_path / "data/silver/sample.parquet")
    table = pq.read_table(catalog_path)
    pq.write_table(table.take(pa.array(indices)), catalog_path)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"


def test_audit_rejects_catalog_item_type_mismatch(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=1)
    catalog_path = source_catalog_path(tmp_path / "data/silver/sample.parquet")
    table = pq.read_table(catalog_path)
    records = table.to_pylist()
    records[0]["kind"] = "span"
    pq.write_table(pa.Table.from_pylist(records, schema=table.schema), catalog_path)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_catalog_invalid"
    assert "native identity" in caught.value.detail


def test_audit_is_invariant_to_source_reference_permutation(tmp_path: Path) -> None:
    _fixture(tmp_path, pymupdf_item_count=2)
    silver_path = tmp_path / "data/silver/sample.parquet"
    candidate_path = tmp_path / "data/candidate-existing/sample.parquet"
    for path in (silver_path, candidate_path):
        element = read_document_elements(path)[0]
        reversed_ids = list(reversed(element.fragments[0].source_item_ids))
        changed = element.model_copy(
            update={"fragments": [element.fragments[0].model_copy(update={"source_item_ids": reversed_ids})]}
        )
        write_document_elements([changed], path, overwrite=True)

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].silver.source_item_reference_count == 2
    assert report.documents[0].silver.unique_source_item_count == 2


def test_audit_fails_fast_with_machine_readable_source_id_error(tmp_path: Path) -> None:
    _fixture(tmp_path, source_id="liteparse:p1:item9")

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.as_dict() == {
        "code": "source_item_id_unknown",
        "path": str(tmp_path / "data/silver/sample.parquet"),
        "detail": "liteparse:p1:item9",
    }


def test_audit_rejects_bronze_artifact_path_traversal_before_verification(tmp_path: Path) -> None:
    _fixture(tmp_path)
    manifest_path = tmp_path / "data/bronze/sample/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "../liteparse.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "path_traversal_forbidden"


def test_audit_rejects_symlinked_current_silver(tmp_path: Path) -> None:
    _fixture(tmp_path)
    silver_path = tmp_path / "data/silver/sample.parquet"
    target = tmp_path / "data/sample-target.parquet"
    silver_path.replace(target)
    silver_path.symlink_to(target)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "symlink_forbidden"


def test_audit_validates_source_item_page_and_table_cell_provenance(tmp_path: Path) -> None:
    _fixture(tmp_path)
    source_sha = _sha256(tmp_path / "data/corpus/dev/sample.pdf")
    cell_fragment = PageFragment(
        page_number=1,
        bbox=BoundingBox(x0=1, y0=2, x1=20, y1=10),
        source_item_ids=["liteparse:p1:item9"],
    )
    table = _element(source_sha).model_copy(
        update={
            "element_type": "table",
            "format": "markdown",
            "structure": ElementStructure(
                table=TableStructure(
                    row_count=1,
                    column_count=1,
                    header_row_count=1,
                    representation="markdown",
                    cells=[
                        TableCell(
                            row_index=0,
                            column_index=0,
                            role="header",
                            text="Grounded text",
                            fragments=[cell_fragment],
                        )
                    ],
                )
            ),
        }
    )
    write_document_elements([table], tmp_path / "data/silver/sample.parquet", overwrite=True)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_item_id_unknown"


def test_audit_rejects_source_item_attached_to_wrong_fragment_page(tmp_path: Path) -> None:
    _fixture(tmp_path)
    source_sha = _sha256(tmp_path / "data/corpus/dev/sample.pdf")
    element = _element(source_sha).model_copy(
        update={
            "fragments": [
                PageFragment(
                    page_number=2,
                    bbox=BoundingBox(x0=1, y0=2, x1=20, y1=10),
                    source_item_ids=["liteparse:p1:item0"],
                )
            ]
        }
    )
    write_document_elements([element], tmp_path / "data/silver/sample.parquet", overwrite=True)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "source_item_page_mismatch"


def test_audit_rejects_invalid_parent_revision_semantics(tmp_path: Path) -> None:
    _fixture(tmp_path)
    source_sha = _sha256(tmp_path / "data/corpus/dev/sample.pdf")
    element = _element(source_sha).model_copy(
        update={"annotation": _element(source_sha).annotation.model_copy(update={"revision": 2})}
    )
    write_document_elements([element], tmp_path / "data/silver/sample.parquet", overwrite=True)

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "silver_parent_link_invalid"


def test_audit_classifies_partial_candidate_identity_claim_as_invalid(tmp_path: Path) -> None:
    _fixture(tmp_path)
    churn_path = tmp_path / "data/churn/sample.json"
    churn = json.loads(churn_path.read_text(encoding="utf-8"))
    churn["candidate_content_identity_rate"] = None
    churn_path.write_text(json.dumps(churn), encoding="utf-8")

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].churn.status == "invalid"
    assert report.documents[0].churn.reason_code == "candidate_identity_fields_partial"
    assert "both be present or both be null" in report.documents[0].churn.reason


def test_lineage_accepts_canonical_benchmark_manifest_json(tmp_path: Path) -> None:
    _fixture(tmp_path)
    manifest_path = tmp_path / "data/batch.json"

    canonical = BenchmarkBatchManifest.model_validate_json(manifest_path.read_bytes())
    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.batch_id == canonical.batch_id
    assert report.documents[0].bronze_source_sha256 == canonical.selections[0].source_sha256


@pytest.mark.parametrize("source_hash", [None, "not-a-sha256", "A" * 64])
def test_lineage_rejects_missing_or_invalid_selection_source_hash(
    tmp_path: Path, source_hash: str | None
) -> None:
    _fixture(tmp_path)
    batch_path = tmp_path / "data/batch.json"
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    if source_hash is None:
        del payload["selections"][0]["source_sha256"]
    else:
        payload["selections"][0]["source_sha256"] = source_hash
    batch_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "artifact_invalid"
    assert "source_sha256" in caught.value.detail


def test_lineage_rejects_selection_source_hash_not_bound_to_bronze(tmp_path: Path) -> None:
    _fixture(tmp_path)
    batch_path = tmp_path / "data/batch.json"
    payload = json.loads(batch_path.read_text(encoding="utf-8"))
    payload["selections"][0]["source_sha256"] = "f" * 64
    batch_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == "selection_source_hash_mismatch"
    assert caught.value.path == tmp_path / "data/bronze/sample/manifest.json"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("source_size_bytes", 999, "bronze_source_size_mismatch"),
        ("source_sha256", "0" * 64, "bronze_source_hash_mismatch"),
    ],
)
def test_audit_checks_manifest_source_size_and_hash(
    tmp_path: Path, field: str, value: object, code: str
) -> None:
    _fixture(tmp_path)
    manifest_path = tmp_path / "data/bronze/sample/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LineageValidationError) as caught:
        audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert caught.value.code == code


def test_audit_classifies_impossible_churn_arithmetic_as_invalid(tmp_path: Path) -> None:
    _fixture(tmp_path)
    churn_path = tmp_path / "data/churn/sample.json"
    churn = json.loads(churn_path.read_text(encoding="utf-8"))
    churn["added_count"] = 1
    churn_path.write_text(json.dumps(churn), encoding="utf-8")

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].churn.status == "invalid"
    assert report.documents[0].churn.reason_code == "churn_common_count_mismatch"
    assert "common element counts" in report.documents[0].churn.reason


def test_audit_classifies_unmatched_candidate_identity_claim_as_invalid(tmp_path: Path) -> None:
    _fixture(tmp_path)
    churn_path = tmp_path / "data/churn/sample.json"
    churn = json.loads(churn_path.read_text(encoding="utf-8"))
    churn["candidate_identity_rate"] = 0.123
    churn["candidate_content_identity_rate"] = 0.456
    churn_path.write_text(json.dumps(churn), encoding="utf-8")

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].churn.status == "invalid"
    assert report.documents[0].churn.reason_code == "candidate_revision_not_found"
    assert report.documents[0].churn.candidate_identity_matches == []


def test_audit_exposes_invalid_artifact_reason_code(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "data/churn/sample.json").write_text("not json", encoding="utf-8")

    report = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert report.documents[0].churn.status == "invalid"
    assert report.documents[0].churn.reason_code == "churn_artifact_invalid"


def test_churn_status_schema_requires_status_appropriate_reason_codes() -> None:
    with pytest.raises(ValidationError, match="reason_code None.*invalid"):
        ChurnArtifactStatus(path="churn.json", status="invalid", reason="prose is not a code")

    with pytest.raises(ValidationError, match="invalid for churn status stale"):
        ChurnArtifactStatus(
            path="churn.json",
            status="stale",
            reason_code="candidate_revision_not_found",
            reason="wrong status-code pairing",
        )


def test_stale_churn_is_not_rewritten_or_laundered(tmp_path: Path) -> None:
    _fixture(tmp_path, churn_current_count=2)
    churn_path = tmp_path / "data/churn/sample.json"
    before = churn_path.read_bytes()

    first = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))
    second = audit_cycle_lineage(tmp_path, Path("data/batch.json"), Path("data/silver"), Path("data/churn"))

    assert first.documents[0].churn.status == second.documents[0].churn.status == "stale"
    assert churn_path.read_bytes() == before
    assert serialize_cycle_lineage_report(first) == serialize_cycle_lineage_report(second)
    assert serialize_cycle_lineage_report(first).endswith("\n")


def _chain_artifact(
    root: Path,
    directory: str,
    document_id: str,
    content: str,
    *,
    stage: str = "silver",
) -> Path:
    path = root / directory / "sample.parquet"
    element = _element(document_id, stage=stage).model_copy(update={"content": content})
    write_document_elements([element], path)
    return path


def _identity(path: Path, root: Path) -> LineageArtifactIdentity:
    return LineageArtifactIdentity(path=path.relative_to(root).as_posix(), sha256=_sha256(path))


def _claim(
    root: Path,
    parent: Path,
    output: Path,
    *,
    no_change: bool = False,
    superseded: bool = False,
) -> LineageTransitionClaim:
    previous = read_document_elements(parent)
    current = read_document_elements(output)
    return LineageTransitionClaim(
        selection_id="sample",
        parent=_identity(parent, root),
        output=_identity(output, root),
        expected_churn=evaluate_reference_churn(previous, current),
        no_change=no_change,
        superseded=superseded,
    )


def test_lineage_chain_replays_exact_machine_readable_churn(tmp_path: Path) -> None:
    parent = _chain_artifact(tmp_path, "data/parent", "doc", "Before")
    output = _chain_artifact(tmp_path, "data/output", "doc", "After")

    report = audit_lineage_chain(tmp_path, "sample", [_claim(tmp_path, parent, output)])

    assert report.replayable
    assert report.latest == _identity(output, tmp_path)
    assert report.transitions[0].replayed_churn.changed_count == 1


def test_lineage_chain_requires_adjacent_parent_hashes(tmp_path: Path) -> None:
    parent = _chain_artifact(tmp_path, "data/parent", "doc", "Before")
    middle = _chain_artifact(tmp_path, "data/middle", "doc", "Middle")
    unrelated = _chain_artifact(tmp_path, "data/unrelated", "doc", "Unrelated")
    latest = _chain_artifact(tmp_path, "data/latest", "doc", "Latest")
    claims = [
        _claim(tmp_path, parent, middle, superseded=True),
        _claim(tmp_path, unrelated, latest),
    ]

    with pytest.raises(LineageValidationError) as caught:
        audit_lineage_chain(tmp_path, "sample", claims)

    assert caught.value.code == "lineage_parent_chain_mismatch"


def test_lineage_chain_preserves_superseded_revisions(tmp_path: Path) -> None:
    parent = _chain_artifact(tmp_path, "data/parent", "doc", "Before")
    middle = _chain_artifact(tmp_path, "data/middle", "doc", "Middle")
    latest = _chain_artifact(tmp_path, "data/latest", "doc", "Latest")

    report = audit_lineage_chain(
        tmp_path,
        "sample",
        [
            _claim(tmp_path, parent, middle, superseded=True),
            _claim(tmp_path, middle, latest),
        ],
    )

    assert [transition.superseded for transition in report.transitions] == [True, False]
    assert [transition.output.sha256 for transition in report.transitions] == [
        _sha256(middle),
        _sha256(latest),
    ]


def test_lineage_chain_verifies_explicit_no_change_record(tmp_path: Path) -> None:
    parent = _chain_artifact(tmp_path, "data/parent", "doc", "Stable")
    output = tmp_path / "data/output/sample.parquet"
    output.parent.mkdir(parents=True)
    output.write_bytes(parent.read_bytes())

    report = audit_lineage_chain(
        tmp_path,
        "sample",
        [_claim(tmp_path, parent, output, no_change=True)],
    )

    transition = report.transitions[0]
    assert transition.no_change
    assert transition.replayed_churn.churn_rate == 0
    assert transition.replayed_churn.unchanged_count == 1


def test_lineage_chain_fails_fast_on_claimed_artifact_identity_mismatch(tmp_path: Path) -> None:
    parent = _chain_artifact(tmp_path, "data/parent", "doc", "Before")
    output = _chain_artifact(tmp_path, "data/output", "doc", "After")
    claim = _claim(tmp_path, parent, output).model_copy(
        update={"output": LineageArtifactIdentity(path="data/output/sample.parquet", sha256="0" * 64)}
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_lineage_chain(tmp_path, "sample", [claim])

    assert caught.value.code == "lineage_artifact_hash_mismatch"


def test_lineage_chain_fails_fast_on_candidate_document_identity_mismatch(tmp_path: Path) -> None:
    parent = _chain_artifact(tmp_path, "data/parent", "doc", "Before")
    output = _chain_artifact(tmp_path, "data/output", "doc", "After")
    candidate = _chain_artifact(
        tmp_path,
        "data/candidate",
        "other-doc",
        "Candidate",
        stage="candidate",
    )
    valid_candidate = _chain_artifact(
        tmp_path,
        "data/valid-candidate",
        "doc",
        "Candidate",
        stage="candidate",
    )
    expected = evaluate_reference_churn(
        read_document_elements(parent),
        read_document_elements(output),
        candidate=read_document_elements(valid_candidate),
    )
    claim = LineageTransitionClaim(
        selection_id="sample",
        parent=_identity(parent, tmp_path),
        output=_identity(output, tmp_path),
        candidate=_identity(candidate, tmp_path),
        expected_churn=expected,
    )

    with pytest.raises(LineageValidationError) as caught:
        audit_lineage_chain(tmp_path, "sample", [claim])

    assert caught.value.code == "lineage_candidate_document_id_mismatch"
