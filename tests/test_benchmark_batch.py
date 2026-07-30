from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TypedDict

import pytest
from click.testing import CliRunner

from app.pdf2md import benchmark_batch
from app.pdf2md.benchmark_batch import (
    BenchmarkBatchManifest,
    aggregate_evaluations,
    build_regression_gate,
    run_benchmark_batch,
    verify_benchmark_run,
)
from app.pdf2md.cli import cli
from app.pdf2md.evaluation import EVALUATOR_SEMANTICS_VERSION, EvaluationReport, evaluate_document
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    PageFragment,
    ParagraphStructure,
    write_document_elements,
)
from app.pdf2md.tables import RENDERER_SEMANTICS_VERSION


def _element(document_id: str, stage: str, *, page: int = 1, content: str = "Stable text") -> DocumentElement:
    return DocumentElement(
        document_id=document_id,
        element_id=f"{stage}-{page}",
        order=0,
        element_type="paragraph",
        content=content,
        format="text",
        fragments=[
            PageFragment(
                page_number=page,
                page_width=100,
                page_height=100,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=10),
            )
        ],
        structure=ElementStructure(paragraph=ParagraphStructure(role="body")),
        annotation=AnnotationMetadata(
            stage=stage,  # pyright: ignore[reportArgumentType]
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        ),
    )


def _report() -> EvaluationReport:
    reference = [_element("doc", "silver")]
    candidate = [_element("doc", "candidate")]
    return evaluate_document(candidate, reference)


def test_aggregate_evaluations_excludes_nulls_and_reports_denominators() -> None:
    first = _report()
    second = first.model_copy(
        update={
            "metrics": first.metrics.model_copy(
                update={"reading_order_accuracy": 0.5, "heading_detection_precision": 0.25}
            )
        }
    )

    aggregate = aggregate_evaluations([first, second])

    assert list(aggregate.metrics) == sorted(aggregate.metrics)
    assert aggregate.metrics["element_f1"].value == 1
    assert aggregate.metrics["element_f1"].contributing_document_count == 2
    assert aggregate.metrics["reading_order_accuracy"].value == 0.5
    assert aggregate.metrics["reading_order_accuracy"].contributing_document_count == 1
    assert aggregate.metrics["reading_order_accuracy"].null_document_count == 1
    assert aggregate.metrics["figure_detection_precision"].value is None
    assert aggregate.metrics["figure_detection_precision"].contributing_document_count == 0


def test_regression_gate_applies_thresholds_strict_controls_and_null_policy() -> None:
    baseline = _report()
    current = baseline.model_copy(
        update={
            "metrics": baseline.metrics.model_copy(
                update={
                    "element_f1": baseline.metrics.element_f1 - 0.021,
                    "normalized_character_error_rate": 0.021,
                    "fragmentation_rate": 0.1,
                    "table_cell_token_f1": None,
                    "heading_detection_precision": 0.0,
                }
            )
        }
    )

    gate = build_regression_gate(["sample"], [baseline], [current])

    assert gate.status == "blocked"
    assert [failure.metric for failure in gate.documents[0].failures] == [
        "element_f1",
        "normalized_character_error_rate",
        "fragmentation_rate",
    ]
    assert "table_cell_token_f1" not in {failure.metric for failure in gate.documents[0].failures}
    assert "heading_detection_precision" not in {failure.metric for failure in gate.documents[0].failures}


def test_regression_gate_threshold_is_inclusive_and_document_ids_are_unique() -> None:
    baseline = _report()
    current = baseline.model_copy(
        update={
            "metrics": baseline.metrics.model_copy(
                update={
                    "element_f1": baseline.metrics.element_f1 - 0.02,
                    "normalized_character_error_rate": (
                        baseline.metrics.normalized_character_error_rate + 0.02
                    ),
                }
            )
        }
    )

    assert build_regression_gate(["sample"], [baseline], [current]).status == "passed"
    with pytest.raises(ValueError, match="document ids must be unique"):
        build_regression_gate(["sample", "sample"], [baseline, baseline], [current, current])


def test_batch_manifest_rejects_holdout_layoutlm_and_inconsistent_counts() -> None:
    payload = _batch_payload("sample")
    selections = payload["selections"]
    assert isinstance(selections, list)
    first_selection = selections[0]
    assert isinstance(first_selection, dict)
    first_selection["source_path"] = "data/holdout/sample.pdf"
    with pytest.raises(ValueError, match="holdout selection is forbidden"):
        BenchmarkBatchManifest.model_validate(payload)

    payload = _batch_payload("academic-layoutlm")
    with pytest.raises(ValueError, match="LayoutLM selection is forbidden"):
        BenchmarkBatchManifest.model_validate(payload)

    payload = _batch_payload("sample")
    payload["selection_count"] = 2
    with pytest.raises(ValueError, match="selection_count"):
        BenchmarkBatchManifest.model_validate(payload)

    payload = _batch_payload("sample")
    payload["batch_id"] = "../unsafe"
    with pytest.raises(ValueError, match="batch_id is not a safe artifact name"):
        BenchmarkBatchManifest.model_validate(payload)


def test_batch_run_writes_deterministic_artifacts_and_requires_fresh_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [_element(setup["digest"], "candidate", page=pages[0])],
    )
    output_dir = tmp_path / "output"

    result = run_benchmark_batch(
        setup["manifest"],
        setup["references"],
        setup["baseline"],
        output_dir,
        root_dir=tmp_path,
    )

    assert result.regression_gate.status == "passed"
    assert sorted(
        path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()
    ) == [
        "aggregate.json",
        "anchor.json",
        "baseline-aggregate.json",
        "baseline-evaluations/sample.json",
        "candidates/sample.md",
        "candidates/sample.parquet",
        "candidates/sample.source-items.parquet",
        "evaluations/sample.json",
        "provenance.json",
        "regression-gate.json",
        "report.md",
        "run.json",
    ]
    assert (output_dir / "candidates/sample.md").read_text(encoding="utf-8") == "Stable text\n"
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Null metrics are excluded from macro means, never coerced to zero." in report
    assert "| sample | passed | — |" in report

    second_output_dir = tmp_path / "second-output"
    run_benchmark_batch(
        setup["manifest"],
        setup["references"],
        setup["baseline"],
        second_output_dir,
        root_dir=tmp_path,
    )
    first_artifacts = {
        path.relative_to(output_dir): path.read_bytes() for path in output_dir.rglob("*") if path.is_file()
    }
    second_artifacts = {
        path.relative_to(second_output_dir): path.read_bytes()
        for path in second_output_dir.rglob("*")
        if path.is_file()
    }
    assert second_artifacts == first_artifacts

    with pytest.raises(FileExistsError, match="already exists"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            output_dir,
            root_dir=tmp_path,
        )


def test_archived_provenance_is_portable_complete_and_fails_fast_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [_element(setup["digest"], "candidate", page=pages[0])],
    )
    output_dir = tmp_path / "output"
    run_benchmark_batch(
        setup["manifest"], setup["references"], setup["baseline"], output_dir, root_dir=tmp_path
    )

    provenance = verify_benchmark_run(output_dir)
    payload = provenance.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "created_at" not in serialized
    assert provenance.run_id.startswith("sha256-")
    assert provenance.runner_version == "4.0.0"
    assert provenance.report_schema_version == "4.0.0"
    assert set(provenance.inputs.model_dump()) == {
        "manifest",
        "source",
        "bronze",
        "reference",
        "baseline",
    }
    assert {item.path for item in provenance.candidate.files} == {
        "sample.md",
        "sample.parquet",
        "sample.source-items.parquet",
    }
    assert "aggregate.json" in {item.path for item in provenance.output_payload.files}
    assert "report.md" not in {item.path for item in provenance.output_payload.files}
    assert provenance.runtime.python.count(".") == 2
    assert "pymupdf" in provenance.runtime.installed_distributions
    assert provenance.runtime.os_name
    assert provenance.runtime.os_release
    assert provenance.runtime.architecture
    assert provenance.runtime.native_build.mupdf
    assert provenance.code_fingerprints.code.files
    evaluator_paths = {item.path for item in provenance.code_fingerprints.evaluator.files}
    renderer_paths = {item.path for item in provenance.code_fingerprints.renderer.files}
    assert "app/pdf2md/evaluation.py" in evaluator_paths
    assert "app/pdf2md/engine.py" in evaluator_paths
    assert "app/pdf2md/cli.py" in evaluator_paths
    assert "app/pdf2md/blind_v6.py" in evaluator_paths
    assert "app/pdf2md/blind_v7.py" in evaluator_paths
    assert "app/upload_endpoint.py" in evaluator_paths
    assert evaluator_paths == renderer_paths
    assert provenance.code_fingerprints.schema_fingerprint.files[0].path == "app/pdf2md/schema.py"
    assert [item.path for item in provenance.code_fingerprints.curation_guidance.files] == [
        ".claude/skills/pdf-structure-curation/SKILL.md"
    ]
    assert provenance.curation_compatibility.model_dump() == {
        "element_schema_version": "1.0.0",
        "source_catalog_schema_version": "2.0.0",
        "renderer_semantics_version": RENDERER_SEMANTICS_VERSION,
        "evaluator_semantics_version": EVALUATOR_SEMANTICS_VERSION,
    }
    assert provenance.commitment.curation_guidance_root_sha256 == (
        provenance.code_fingerprints.curation_guidance.root_sha256
    )

    copied_dir = tmp_path / "copied-output"
    shutil.copytree(output_dir, copied_dir)
    assert verify_benchmark_run(copied_dir).run_id == provenance.run_id
    cli_result = CliRunner().invoke(
        cli,
        [
            "verify-benchmark-run",
            str(copied_dir),
            "--expected-run-id",
            provenance.run_id,
            "--expected-anchor",
            str(output_dir / "anchor.json"),
        ],
    )
    assert cli_result.exit_code == 0
    assert provenance.run_id in cli_result.output
    (copied_dir / "candidates/sample.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate fingerprint mismatch"):
        verify_benchmark_run(copied_dir)

    copied_report_dir = tmp_path / "copied-report-output"
    shutil.copytree(output_dir, copied_report_dir)
    (copied_report_dir / "aggregate.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output payload fingerprint mismatch"):
        verify_benchmark_run(copied_report_dir)


def test_archived_provenance_rejects_run_id_and_aggregate_root_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [_element(setup["digest"], "candidate", page=pages[0])],
    )
    output_dir = tmp_path / "output"
    run_benchmark_batch(
        setup["manifest"], setup["references"], setup["baseline"], output_dir, root_dir=tmp_path
    )
    provenance_path = output_dir / "provenance.json"
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    mismatched_id = dict(payload)
    mismatched_id["run_id"] = "sha256-" + "0" * 64
    provenance_path.write_text(json.dumps(mismatched_id), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic run_id mismatch"):
        verify_benchmark_run(output_dir)

    payload["inputs"]["source"]["root_sha256"] = "0" * 64
    provenance_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate root mismatch"):
        verify_benchmark_run(output_dir)


def test_trusted_anchor_rejects_coordinated_artifact_and_provenance_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [_element(setup["digest"], "candidate", page=pages[0])],
    )
    original_dir = tmp_path / "original"
    run_benchmark_batch(
        setup["manifest"], setup["references"], setup["baseline"], original_dir, root_dir=tmp_path
    )
    original = verify_benchmark_run(original_dir)

    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [
            _element(setup["digest"], "candidate", page=pages[0], content="Coordinated rewrite")
        ],
    )
    rewritten_dir = tmp_path / "rewritten"
    run_benchmark_batch(
        setup["manifest"], setup["references"], setup["baseline"], rewritten_dir, root_dir=tmp_path
    )
    rewritten = verify_benchmark_run(rewritten_dir)
    assert rewritten.run_id != original.run_id

    with pytest.raises(ValueError, match="trusted expected run_id mismatch"):
        verify_benchmark_run(rewritten_dir, expected_run_id=original.run_id)
    with pytest.raises(ValueError, match="trusted expected anchor mismatch"):
        verify_benchmark_run(rewritten_dir, expected_anchor=original_dir / "anchor.json")
    anchor = json.loads((original_dir / "anchor.json").read_text(encoding="utf-8"))
    assert "coordinated rewrite" in anchor["trust_model"]


def test_curation_guidance_change_alters_future_run_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [_element(setup["digest"], "candidate", page=pages[0])],
    )
    first_dir = tmp_path / "first-output"
    run_benchmark_batch(
        setup["manifest"], setup["references"], setup["baseline"], first_dir, root_dir=tmp_path
    )
    first = verify_benchmark_run(first_dir)

    guidance = tmp_path / ".claude/skills/pdf-structure-curation/SKILL.md"
    guidance.write_text(guidance.read_text(encoding="utf-8") + "Additional invariant.\n", encoding="utf-8")
    second_dir = tmp_path / "second-output"
    run_benchmark_batch(
        setup["manifest"], setup["references"], setup["baseline"], second_dir, root_dir=tmp_path
    )
    second = verify_benchmark_run(second_dir)

    assert second.code_fingerprints.curation_guidance.root_sha256 != (
        first.code_fingerprints.curation_guidance.root_sha256
    )
    assert second.run_id != first.run_id


@pytest.mark.parametrize("mode", ["missing", "mismatch"])
def test_curation_guidance_compatibility_fails_before_output(tmp_path: Path, mode: str) -> None:
    setup = _setup_batch(tmp_path)
    guidance = tmp_path / ".claude/skills/pdf-structure-curation/SKILL.md"
    if mode == "missing":
        guidance.unlink()
        expected = "SKILL.md"
    else:
        guidance.write_text(
            guidance.read_text(encoding="utf-8").replace(
                f'renderer-semantics-version: "{RENDERER_SEMANTICS_VERSION}"',
                'renderer-semantics-version: "9.0.0"',
            ),
            encoding="utf-8",
        )
        expected = "compatibility mismatch"
    output_dir = tmp_path / "output"

    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        run_benchmark_batch(
            setup["manifest"], setup["references"], setup["baseline"], output_dir, root_dir=tmp_path
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".output-*"))


def test_batch_run_does_not_replace_destination_created_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    output_dir = tmp_path / "output"

    def extract_and_create_destination(_path: Path, *, pages: list[int]) -> list[DocumentElement]:
        output_dir.mkdir()
        (output_dir / "intruder.txt").write_text("preserve", encoding="utf-8")
        return [_element(setup["digest"], "candidate", page=pages[0])]

    monkeypatch.setattr(benchmark_batch, "extract_document_elements", extract_and_create_destination)

    with pytest.raises(FileExistsError, match="appeared during the run"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            output_dir,
            root_dir=tmp_path,
        )

    assert (output_dir / "intruder.txt").read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(".output-*"))


def test_fingerprint_inventory_distinguishes_added_removed_and_changed_files(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("FIRST = 1\n", encoding="utf-8")
    baseline = benchmark_batch._fingerprint_paths([("app/first.py", first)])  # pyright: ignore[reportPrivateUsage]

    second.write_text("SECOND = 2\n", encoding="utf-8")
    added = benchmark_batch._fingerprint_paths(  # pyright: ignore[reportPrivateUsage]
        [("app/first.py", first), ("app/second.py", second)]
    )
    removed = benchmark_batch._fingerprint_paths([  # pyright: ignore[reportPrivateUsage]
        ("app/second.py", second)
    ])
    first.write_text("FIRST = 3\n", encoding="utf-8")
    changed = benchmark_batch._fingerprint_paths([  # pyright: ignore[reportPrivateUsage]
        ("app/first.py", first)
    ])

    assert len({baseline.root_sha256, added.root_sha256, removed.root_sha256, changed.root_sha256}) == 4


def test_legacy_extraction_seam_fails_closed_on_unresolved_source_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    element = _element(setup["digest"], "candidate")
    fragment = element.fragments[0].model_copy(update={"source_item_ids": ["missing"]})
    element = element.model_copy(update={"fragments": [fragment]})
    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [element],
    )
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="dangling source_item_id: missing"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            output_dir,
            root_dir=tmp_path,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".output-*"))


def test_batch_preflight_rejects_source_hash_page_and_stage_mismatches_without_output(
    tmp_path: Path,
) -> None:
    setup = _setup_batch(tmp_path)
    source = tmp_path / "data/source.pdf"
    source.write_bytes(b"changed")
    output_dir = tmp_path / "hash-output"
    with pytest.raises(ValueError, match="source size mismatch|source hash mismatch"):
        run_benchmark_batch(
            setup["manifest"], setup["references"], setup["baseline"], output_dir, root_dir=tmp_path
        )
    assert not output_dir.exists()

    setup = _setup_batch(tmp_path / "page-case", baseline_page=2)
    output_dir = tmp_path / "page-case/output"
    with pytest.raises(ValueError, match="outside the requested selection"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            output_dir,
            root_dir=tmp_path / "page-case",
        )
    assert not output_dir.exists()

    setup = _setup_batch(tmp_path / "stage-case", reference_stage="candidate")
    output_dir = tmp_path / "stage-case/output"
    with pytest.raises(ValueError, match="invalid reference stages"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            output_dir,
            root_dir=tmp_path / "stage-case",
        )
    assert not output_dir.exists()

    coverage_root = tmp_path / "coverage-case"
    setup = _setup_batch(coverage_root)
    batch_payload = json.loads(setup["manifest"].read_text(encoding="utf-8"))
    batch_payload["selections"][0]["pages"] = {
        "annotated": [1, 2],
        "context": [],
        "requested": [1, 2],
    }
    setup["manifest"].write_text(json.dumps(batch_payload), encoding="utf-8")
    bronze_path = coverage_root / "data/bronze/sample/manifest.json"
    bronze_payload = json.loads(bronze_path.read_text(encoding="utf-8"))
    bronze_payload["source_page_count"] = 2
    bronze_payload["selection"] = {
        "annotated_pages": [1, 2],
        "context_pages": [],
        "requested_pages": [1, 2],
    }
    bronze_path.write_text(json.dumps(bronze_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="do not cover requested pages"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            coverage_root / "output",
            root_dir=coverage_root,
        )


def test_batch_preflight_rejects_external_inputs_outputs_and_bronze_artifact_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    setup = _setup_batch(root)
    external = tmp_path / "external"
    external.mkdir()

    with pytest.raises(ValueError, match="reference directory escapes"):
        run_benchmark_batch(setup["manifest"], external, setup["baseline"], root / "output", root_dir=root)
    with pytest.raises(ValueError, match="output directory escapes"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            external / "output",
            root_dir=root,
        )

    bronze_path = root / "data/bronze/sample/manifest.json"
    bronze_payload = json.loads(bronze_path.read_text(encoding="utf-8"))
    bronze_payload["artifacts"] = [
        {"path": "../../../source.pdf", "sha256": setup["digest"], "size_bytes": 13}
    ]
    bronze_path.write_text(json.dumps(bronze_payload), encoding="utf-8")
    batch_payload = json.loads(setup["manifest"].read_text(encoding="utf-8"))
    batch_payload["selections"][0]["verification"]["artifact_count"] = 1
    batch_payload["verification_summary"]["artifact_count"] = 1
    setup["manifest"].write_text(json.dumps(batch_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bronze artifact.*escapes"):
        run_benchmark_batch(
            setup["manifest"],
            setup["references"],
            setup["baseline"],
            root / "artifact-output",
            root_dir=root,
        )


def test_benchmark_batch_cli_returns_failure_when_gate_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _setup_batch(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        benchmark_batch,
        "extract_document_elements",
        lambda _path, *, pages: [
            _element(setup["digest"], "candidate", page=pages[0], content="Regressed text")
        ],
    )

    result = CliRunner().invoke(
        cli,
        [
            "benchmark-batch",
            str(setup["manifest"]),
            str(setup["references"]),
            str(setup["baseline"]),
            str(output_dir),
            "--root-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "regression gate blocked" in result.output
    assert (output_dir / "regression-gate.json").is_file()


class _BatchSetup(TypedDict):
    manifest: Path
    references: Path
    baseline: Path
    digest: str


def _setup_batch(
    root: Path,
    *,
    baseline_page: int = 1,
    reference_stage: str = "silver",
) -> _BatchSetup:
    data_dir = root / "data"
    bundle_dir = data_dir / "bronze/sample"
    references = data_dir / "references"
    baseline = data_dir / "baseline"
    bundle_dir.mkdir(parents=True)
    references.mkdir(parents=True)
    baseline.mkdir(parents=True)
    guidance = root / ".claude/skills/pdf-structure-curation/SKILL.md"
    guidance.parent.mkdir(parents=True)
    guidance.write_text(
        "---\n"
        "name: pdf-structure-curation\n"
        'element-schema-version: "1.0.0"\n'
        'source-catalog-schema-version: "2.0.0"\n'
        f'renderer-semantics-version: "{RENDERER_SEMANTICS_VERSION}"\n'
        f'evaluator-semantics-version: "{EVALUATOR_SEMANTICS_VERSION}"\n'
        "---\n"
        "Synthetic benchmark guidance.\n",
        encoding="utf-8",
    )
    source = data_dir / "source.pdf"
    source.write_bytes(b"synthetic-pdf")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    bronze_manifest = {
        "schema_version": "1.0.0",
        "source_name": "source.pdf",
        "source_path": "data/source.pdf",
        "source_sha256": digest,
        "source_size_bytes": source.stat().st_size,
        "source_page_count": 1,
        "selection": {
            "requested_pages": [1],
            "annotated_pages": [1],
            "context_pages": [],
        },
        "config": {
            "render_dpi": 144,
            "liteparse_dpi": 150,
            "min_native_characters_per_page": 20,
            "liteparse_executable": "lit",
        },
        "pymupdf_version": "test",
        "liteparse_version": "test",
        "artifacts": [],
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(bronze_manifest, sort_keys=True), encoding="utf-8")
    write_document_elements([_element(digest, reference_stage)], references / "sample.parquet")
    write_document_elements([_element(digest, "candidate", page=baseline_page)], baseline / "sample.parquet")
    manifest = data_dir / "benchmark.json"
    manifest.write_text(json.dumps(_batch_payload("sample"), sort_keys=True), encoding="utf-8")
    return {
        "manifest": manifest,
        "references": references,
        "baseline": baseline,
        "digest": digest,
    }


def _batch_payload(document_id: str) -> dict[str, object]:
    return {
        "batch_id": "test",
        "selection_count": 1,
        "selections": [
            {
                "id": document_id,
                "source_path": "data/source.pdf",
                "pages": {"annotated": [1], "context": [], "requested": [1]},
                "bundle_path": "data/bronze/sample",
                "verification": {"status": "verified", "artifact_count": 0},
            }
        ],
        "verification_summary": {
            "verified_bundle_count": 1,
            "failed_bundle_count": 0,
            "artifact_count": 0,
        },
    }
