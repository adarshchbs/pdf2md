import json
from pathlib import Path

import pytest

from app.pdf2md.schema import DocumentElement, read_document_elements
from app.pdf2md.silver_coherence import PARAGRAPH_ROLES, SOURCE_HASHES, migrate, validate

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data/silver-task36-ipcc"
BRONZE = ROOT / "data/bronze"
COHERENCE = ROOT / "data/silver-coherence-v1-corrected"


def _by_id(document: str) -> dict[str, DocumentElement]:
    return {
        element.element_id: element for element in read_document_elements(COHERENCE / f"{document}.parquet")
    }


def test_normalized_corpus_passes_structural_validation() -> None:
    report = validate(COHERENCE, BRONZE)

    documents = report["documents"]
    checks = report["checks"]
    warnings = report["provenance_geometry_warnings"]
    assert isinstance(documents, dict)
    assert isinstance(checks, dict)
    assert isinstance(warnings, list)
    assert set(documents) == set(SOURCE_HASHES)
    assert checks == {
        "document_id": "passed",
        "element_id_order": "passed",
        "links": "passed",
        "parquet_roundtrip": "passed",
        "source_id_resolution_and_sibling_exclusivity": "passed",
        "source_item_bbox_containment": "blocked_inherited",
        "table_grid_and_render": "passed",
        "vocabulary_and_paragraph_structure": "passed",
    }
    assert len(warnings) == 38


def test_ambiguous_bronze_grounded_decisions_are_explicit() -> None:
    eu = _by_id("eu-ai-act")
    continuation = eu["eu-ai-act-listitem-003"]
    continuation_paragraph = continuation.structure.paragraph
    assert continuation_paragraph is not None
    assert continuation_paragraph.role == "body"
    assert continuation.element_id == "eu-ai-act-listitem-003"

    ipcc = _by_id("ipcc-ar6-syr")
    banner = ipcc["ipcc-ar6-syr-heading-001"].structure.paragraph
    subtitle = ipcc["ipcc-ar6-syr-panelsub-001"].structure.paragraph
    assert banner is not None and banner.role == "heading"
    assert subtitle is not None and subtitle.role == "figure_text"

    tabularray = _by_id("latex-tabularray")
    for index in range(1, 5):
        note = tabularray[f"latex-tabularray-footnote-{index:03d}"]
        paragraph = note.structure.paragraph
        assert paragraph is not None and paragraph.role == "footnote"
    marker_ids = [
        "latex-tabularray-continuation-001",
        "latex-tabularray-caption-002",
        "latex-tabularray-continuation-002",
        "latex-tabularray-caption-003",
    ]
    for marker_id in marker_ids:
        marker = tabularray[marker_id]
        paragraph = marker.structure.paragraph
        assert marker.element_type == "note"
        assert paragraph is not None and paragraph.role == "table_continuation_marker"
        assert marker.include_in_output is False


def test_every_non_table_non_figure_has_closed_paragraph_role() -> None:
    for document in SOURCE_HASHES:
        for element in read_document_elements(COHERENCE / f"{document}.parquet"):
            if element.element_type in {"table", "figure"}:
                continue
            assert element.structure.paragraph is not None
            assert element.structure.paragraph.role in PARAGRAPH_ROLES


def test_sparse_tables_and_churn_use_current_exact_vocabulary() -> None:
    rp2040 = _by_id("rp2040-datasheet")
    sparse = rp2040["rp2040-datasheet-t328"].structure.table
    assert sparse is not None
    assert sparse.representation == "html"
    assert sparse.classification_reasons == ["sparse_section"]

    audit = json.loads((COHERENCE / "migration-audit.json").read_text())
    tabularray = next(
        document for document in audit["documents"] if document["document"] == "latex-tabularray"
    )
    marker_changes = [
        change
        for change in tabularray["elements"]
        if change["element_id"]
        in {
            "latex-tabularray-continuation-001",
            "latex-tabularray-caption-002",
            "latex-tabularray-continuation-002",
            "latex-tabularray-caption-003",
        }
    ]
    assert len(marker_changes) == 4
    assert all(
        not any(field.startswith("include_in_output:") for field in change["fields"])
        for change in marker_changes
    )


def test_migration_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "silver-coherence-v1-corrected"
    migrate(SOURCE, BRONZE, output)

    for document in SOURCE_HASHES:
        generated = read_document_elements(output / f"{document}.parquet")
        authoritative = read_document_elements(COHERENCE / f"{document}.parquet")
        assert generated == authoritative
    assert json.loads((output / "migration-audit.json").read_text()) == json.loads(
        (COHERENCE / "migration-audit.json").read_text()
    )
    assert json.loads((output / "validation-report.json").read_text()) == json.loads(
        (COHERENCE / "validation-report.json").read_text()
    )

    with pytest.raises(FileExistsError, match="immutable output"):
        migrate(SOURCE, BRONZE, output)
