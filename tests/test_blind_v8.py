from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

from app.pdf2md.benchmark_batch import (
    BatchPages,
    BatchSelection,
    BatchVerification,
    BenchmarkBatchManifest,
    VerificationSummary,
)
from app.pdf2md.blind_v8 import build_blind_v8, reviewer_alternative_payload, verify_reviewer_bundle
from app.pdf2md.bronze import BronzeArtifact, BronzeConfig, BronzeManifest, PageSelection
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FigureStructure,
    FootnoteStructure,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
    TableCell,
    TableStructure,
    TextStyleRun,
    write_document_elements,
)

SEED = b"blind-v8-test-seed".ljust(32, b"-")
ORIENTATION_TOKENS = {
    "candidate-secret-id",
    "reference-secret-id",
    "candidate-source-item",
    "reference-source-item",
    "candidate-pipeline",
    "reference-pipeline",
    "candidate-annotator",
    "reference-annotator",
    "finder_grid_v1",
    "/private/candidate/asset.png",
    "/private/reference/asset.png",
    "/private/candidate/source.pdf",
    "/private/reference/source.pdf",
    "a" * 64,
    "b" * 64,
}


def _fragment(source: str, *, x0: float = 1) -> PageFragment:
    return PageFragment(
        page_number=1,
        page_width=612,
        page_height=792,
        bbox=BoundingBox(x0=x0, y0=2, x1=x0 + 100, y1=22),
        source_item_ids=[f"{source}-source-item"],
    )


def _annotation(source: str) -> AnnotationMetadata:
    return AnnotationMetadata(
        stage="candidate" if source == "candidate" else "silver",
        revision=1,
        annotator=f"{source}-annotator",
        confidence=0.51 if source == "candidate" else 0.99,
        adjudication_status="unreviewed",
    )


def _paragraph(source: str, content: str = "Visible meaning") -> DocumentElement:
    return DocumentElement(
        document_id=f"{source}-secret-document",
        element_id=f"{source}-secret-id",
        order=0,
        element_type="paragraph",
        content=content,
        format="text",
        fragments=[_fragment(source, x0=1 if source == "candidate" else 88)],
        structure=ElementStructure(
            paragraph=ParagraphStructure(
                role="list_item",
                list_depth=1,
                list_label="•",
                list_evidence=[f"{source}-pipeline"],
            ),
            style_runs=[
                TextStyleRun(
                    start=0,
                    end=7,
                    font_family=f"{source}-pipeline",
                    font_size=9 if source == "candidate" else 12,
                    bold=True,
                )
            ],
            linked_element_ids=[f"{source}-secret-id"],
            properties=[
                StructureProperty(
                    key="inline_footnote_markers",
                    value=f'["{source}-secret-id"]',
                ),
                StructureProperty(
                    key="inline_marker_ranges",
                    value=json.dumps([{"start": 0, "end": 4, "source": source}]),
                ),
                StructureProperty(key="pipeline", value=f"{source}-pipeline"),
                StructureProperty(key="source_path", value=f"/private/{source}/source.pdf"),
            ],
        ),
        annotation=_annotation(source),
    )


def _table(source: str) -> DocumentElement:
    first = TableCell(
        row_index=0,
        column_index=0,
        role="header",
        text="Heading",
        fragments=[_fragment(source, x0=20)],
    )
    second = TableCell(
        row_index=1,
        column_index=0,
        role="body",
        text="Value",
        fragments=[_fragment(source, x0=30)],
    )
    cells = [first, second] if source == "candidate" else [second, first]
    diagnostic = json.dumps({
        "finder": "finder_grid_v1",
        "annotator": f"{source}-annotator",
        "source_path": f"/private/{source}/source.pdf",
        "sha256": "a" * 64 if source == "candidate" else "b" * 64,
        "nested": {"element_id": f"{source}-secret-id"},
    })
    return DocumentElement(
        document_id=f"{source}-secret-document",
        element_id=f"{source}-secret-id",
        order=0,
        element_type="table",
        content="| Heading |\n| --- |\n| Value |",
        format="markdown",
        fragments=[_fragment(source)],
        structure=ElementStructure(
            table=TableStructure(
                row_count=2,
                column_count=1,
                header_row_count=1,
                representation="markdown",
                cells=cells,
            ),
            linked_element_ids=[f"{source}-secret-id"],
            properties=[
                StructureProperty(key="table_diagnostic_v1", value=diagnostic),
                StructureProperty(key="table_span_suppression_v1", value=diagnostic),
                StructureProperty(key="table_structural_source_item_ids", value=diagnostic),
            ],
        ),
        annotation=_annotation(source),
    )


def _nested_strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [str(key) for key in value] + [
            item for nested in value.values() for item in _nested_strings(nested)
        ]
    if isinstance(value, list):
        return [item for nested in value for item in _nested_strings(nested)]
    return [value] if isinstance(value, str) else []


def test_adversarial_nested_provenance_cannot_recover_orientation() -> None:
    candidate = reviewer_alternative_payload(_table("candidate"))
    reference = reviewer_alternative_payload(_table("reference"))
    candidate_paragraph = reviewer_alternative_payload(_paragraph("candidate"))
    reference_paragraph = reviewer_alternative_payload(_paragraph("reference"))
    candidate_figure = _paragraph("candidate").model_copy(
        update={
            "element_type": "figure",
            "structure": ElementStructure(
                figure=FigureStructure(
                    asset_path="/private/candidate/asset.png",
                    sha256="a" * 64,
                    width=640,
                    height=480,
                    caption_element_id="candidate-secret-id",
                )
            ),
        }
    )
    reference_figure = _paragraph("reference").model_copy(
        update={
            "element_type": "figure",
            "structure": ElementStructure(
                figure=FigureStructure(
                    asset_path="/private/reference/asset.png",
                    sha256="b" * 64,
                    width=1200,
                    height=900,
                    caption_element_id="reference-secret-id",
                )
            ),
        }
    )

    assert candidate == reference
    assert candidate_paragraph == reference_paragraph
    assert reviewer_alternative_payload(candidate_figure) == reviewer_alternative_payload(reference_figure)
    alternatives = [
        candidate,
        reference,
        candidate_paragraph,
        reference_paragraph,
        reviewer_alternative_payload(candidate_figure),
        reviewer_alternative_payload(reference_figure),
    ]
    serialized = json.dumps(alternatives, sort_keys=True)
    assert all(token not in serialized for token in ORIENTATION_TOKENS)
    assert not {
        "document_id",
        "element_id",
        "source_item_ids",
        "fragments",
        "properties",
        "classification_reasons",
        "annotation",
    }.intersection(_nested_strings([candidate, reference]))


def test_reviewer_alternative_uses_fixed_neutral_schema() -> None:
    paragraph = reviewer_alternative_payload(_paragraph("candidate"))
    table = reviewer_alternative_payload(_table("candidate"))
    figure_element = _paragraph("candidate").model_copy(
        update={
            "element_type": "figure",
            "structure": ElementStructure(
                figure=FigureStructure(
                    asset_path="/private/candidate/asset.png",
                    sha256="a" * 64,
                    width=640,
                    height=480,
                    caption_element_id="candidate-secret-id",
                )
            ),
        }
    )
    figure = reviewer_alternative_payload(figure_element)
    footnote_element = _paragraph("candidate").model_copy(
        update={
            "element_type": "footnote",
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="footnote"),
                footnote=FootnoteStructure(
                    label="1",
                    reference_element_ids=["candidate-secret-id"],
                    association_confident=True,
                ),
            ),
        }
    )
    footnote = reviewer_alternative_payload(footnote_element)

    top_level = {"element_type", "content", "format", "include_in_output", "structure"}
    structure_keys = {"paragraph", "table", "figure", "footnote", "style_runs"}
    for payload in (paragraph, table, figure, footnote):
        assert set(payload) == top_level
        assert set(cast(dict[str, object], payload["structure"])) == structure_keys
    paragraph_structure = cast(dict[str, object], paragraph["structure"])
    assert paragraph_structure["paragraph"] == {
        "role": "list_item",
        "heading_level": None,
        "list_depth": 1,
        "list_label": "•",
    }
    assert paragraph_structure["style_runs"] == [
        {"start": 0, "end": 7, "bold": True, "italic": False, "underline": False, "strikeout": False}
    ]
    table_structure = cast(dict[str, object], table["structure"])
    assert table_structure["table"] == {
        "row_count": 2,
        "column_count": 1,
        "header_row_count": 1,
        "representation": "markdown",
        "cells": [
            {
                "row_index": 0,
                "column_index": 0,
                "rowspan": 1,
                "colspan": 1,
                "role": "header",
                "text": "Heading",
            },
            {"row_index": 1, "column_index": 0, "rowspan": 1, "colspan": 1, "role": "body", "text": "Value"},
        ],
    }
    assert cast(dict[str, object], figure["structure"])["figure"] == {}
    assert cast(dict[str, object], footnote["structure"])["footnote"] == {"label": "1"}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_synthetic_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    candidate_dir = root / "inputs/candidates"
    reference_dir = root / "inputs/references"
    bronze_dir = root / "inputs/bronze"
    for path in (candidate_dir, reference_dir, bronze_dir):
        path.mkdir(parents=True)
    environment = root / "uv.lock"
    environment.write_text("synthetic-lock\n", encoding="utf-8")
    write_document_elements([_paragraph("candidate", "Candidate wording")], candidate_dir / "doc.parquet")
    reference = _paragraph("reference", "Reference wording").model_copy(
        update={"document_id": "candidate-secret-document"}
    )
    write_document_elements([reference], reference_dir / "doc.parquet")

    bundle = bronze_dir / "doc"
    (bundle / "pages").mkdir(parents=True)
    image = b"synthetic-png"
    (bundle / "pages/page-0001.png").write_bytes(image)
    bronze_manifest = BronzeManifest(
        source_name="synthetic.pdf",
        source_path="inputs/synthetic.pdf",
        source_sha256="c" * 64,
        source_size_bytes=1,
        source_page_count=1,
        selection=PageSelection(requested_pages=[1], annotated_pages=[1]),
        config=BronzeConfig(),
        pymupdf_version="synthetic",
        liteparse_version="synthetic",
        artifacts=[
            BronzeArtifact(
                path="pages/page-0001.png",
                sha256=_sha256_bytes(image),
                size_bytes=len(image),
            )
        ],
    )
    (bundle / "manifest.json").write_text(
        json.dumps(bronze_manifest.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    manifest = BenchmarkBatchManifest(
        batch_id="synthetic",
        selection_count=1,
        selections=[
            BatchSelection(
                id="doc",
                source_path="inputs/synthetic.pdf",
                source_sha256="c" * 64,
                pages=BatchPages(annotated=[1], context=[], requested=[1]),
                bundle_path="inputs/bronze/doc",
                verification=BatchVerification(status="verified", artifact_count=1),
            )
        ],
        verification_summary=VerificationSummary(
            verified_bundle_count=1, failed_bundle_count=0, artifact_count=1
        ),
    )
    manifest_path = root / "inputs/manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    return manifest_path, candidate_dir, reference_dir, bronze_dir, environment


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_seeded_export_is_deterministic_and_verifier_compatible(tmp_path: Path) -> None:
    manifest, candidates, references, bronze, environment = _write_synthetic_inputs(tmp_path)
    first_reviewer = tmp_path / "blind-v8-reviewer-one"
    first_protected = tmp_path / "blind-v8-protected-one"
    second_reviewer = tmp_path / "blind-v8-reviewer-two"
    second_protected = tmp_path / "blind-v8-protected-two"
    for reviewer, protected in (
        (first_reviewer, first_protected),
        (second_reviewer, second_protected),
    ):
        build_blind_v8(
            manifest,
            candidates,
            references,
            bronze,
            reviewer,
            protected,
            root_dir=tmp_path,
            environment_kind="lock",
            environment_path=environment,
            seed=SEED,
        )
        verify_reviewer_bundle(reviewer)
        subprocess.run([sys.executable, str(reviewer / "verify_bundle.py"), str(reviewer)], check=True)

    assert _file_bytes(first_reviewer) == _file_bytes(second_reviewer)
    first_map = json.loads((first_protected / "protected-map.json").read_text(encoding="utf-8"))
    assert first_map == json.loads((second_protected / "protected-map.json").read_text(encoding="utf-8"))
    assert {mapping["A"] for mapping in first_map["mappings"]} <= {"candidate", "reference"}
