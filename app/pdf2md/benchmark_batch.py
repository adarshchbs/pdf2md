from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from app.pdf2md.bronze import BronzeManifest, PageSelection, verify_bronze_bundle
from app.pdf2md.engine import ExtractedDocument, extract_document_with_catalog, render_document
from app.pdf2md.evaluation import (
    EVALUATOR_SEMANTICS_VERSION,
    EvaluationMetrics,
    EvaluationReport,
    evaluate_document,
)
from app.pdf2md.schema import (
    SCHEMA_VERSION,
    DocumentElement,
    SchemaModel,
    read_document_elements,
)
from app.pdf2md.source_catalog import (
    SOURCE_CATALOG_SCHEMA_VERSION,
    build_source_catalog,
    read_document_with_source_catalog,
    validate_source_catalog,
    write_document_with_source_catalog,
)
from app.pdf2md.tables import RENDERER_SEMANTICS_VERSION

_RUNNER_VERSION = "5.0.0"
_REPORT_SCHEMA_VERSION = "5.0.0"
_PROVENANCE_SCHEMA_VERSION = "4.0.0"
_ANCHOR_SCHEMA_VERSION = "3.0.0"
_CURATION_GUIDANCE_PATH = Path(".claude/skills/pdf-structure-curation/SKILL.md")
_ANCHOR_TRUST_MODEL = (
    "Self-contained archive verification detects inconsistent changes, not a coordinated rewrite; "
    "publish this anchor through an externally trusted channel."
)
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LAYOUTLM_PATTERN = re.compile(r"layout\s*lm(?:\s*v?2)?\b", re.IGNORECASE)


def _contains_layoutlm(value: str) -> bool:
    return _LAYOUTLM_PATTERN.search(value) is not None


def _metadata_contains_layoutlm(value: object) -> bool:
    if isinstance(value, str):
        return _contains_layoutlm(value)
    if isinstance(value, dict):
        return any(
            _contains_layoutlm(str(key)) or _metadata_contains_layoutlm(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_metadata_contains_layoutlm(item) for item in value)
    return False


def _validated_portable_lexical_path(value: str, label: str) -> str:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in value
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} is not an exact portable lexical path: {value}")
    return value


type GateCategory = Literal["element_f1", "normalized_character_error_rate", "topology", "content", "control"]

_DECREASE_GATE_METRICS: tuple[tuple[GateCategory, str], ...] = (
    ("topology", "table_topology_accuracy"),
    ("topology", "table_grid_topology_accuracy"),
    ("topology", "table_representation_accuracy"),
    ("content", "table_cell_content_f1"),
    ("content", "table_cell_token_f1"),
    ("content", "table_cell_content_assignment_accuracy"),
    ("control", "reading_order_accuracy"),
    ("control", "semantic_role_accuracy"),
    ("control", "semantic_role_conditional_accuracy"),
    ("control", "heading_detection_precision"),
    ("control", "heading_detection_recall"),
    ("control", "caption_detection_precision"),
    ("control", "caption_detection_recall"),
    ("control", "figure_detection_precision"),
    ("control", "figure_detection_recall"),
    ("control", "footnote_detection_precision"),
    ("control", "footnote_detection_recall"),
    ("control", "table_detection_precision"),
    ("control", "table_detection_recall"),
    ("control", "cross_page_continuity_accuracy"),
)
_INCREASE_GATE_METRICS: tuple[tuple[GateCategory, str], ...] = (
    ("control", "fragmentation_rate"),
    ("control", "merge_rate"),
    ("control", "content_fragmentation_rate"),
    ("control", "content_merge_rate"),
)


class BatchPages(SchemaModel):
    annotated: list[int]
    context: list[int]
    requested: list[int]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        PageSelection(
            annotated_pages=self.annotated,
            context_pages=self.context,
            requested_pages=self.requested,
        )
        if not self.requested:
            raise ValueError("batch selections require at least one requested page")
        return self

    def as_page_selection(self) -> PageSelection:
        return PageSelection(
            annotated_pages=self.annotated,
            context_pages=self.context,
            requested_pages=self.requested,
        )


class BatchVerification(SchemaModel):
    status: Literal["verified"]
    artifact_count: int = Field(ge=0)


class BatchSelection(SchemaModel):
    id: str
    source_path: str
    source_sha256: str
    pages: BatchPages
    bundle_path: str
    verification: BatchVerification

    @model_validator(mode="after")
    def validate_non_holdout_selection(self) -> Self:
        if _ID_PATTERN.fullmatch(self.id) is None:
            raise ValueError(f"selection id is not a safe artifact name: {self.id}")
        if _SHA256_PATTERN.fullmatch(self.source_sha256) is None:
            raise ValueError(f"selection has invalid source_sha256: {self.id}")
        _validated_portable_lexical_path(self.source_path, f"source path for {self.id}")
        searchable = "/".join((self.id, self.source_path, self.bundle_path))
        if "holdout" in searchable.casefold():
            raise ValueError(f"holdout selection is forbidden: {self.id}")
        if _contains_layoutlm(searchable):
            raise ValueError(f"LayoutLM selection is forbidden: {self.id}")
        return self


class VerificationSummary(SchemaModel):
    verified_bundle_count: int = Field(ge=0)
    failed_bundle_count: int = Field(ge=0)
    artifact_count: int = Field(ge=0)


class BenchmarkBatchManifest(SchemaModel):
    batch_id: str
    selection_count: int = Field(ge=1)
    selections: list[BatchSelection] = Field(min_length=1)
    verification_summary: VerificationSummary

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if _ID_PATTERN.fullmatch(self.batch_id) is None:
            raise ValueError(f"batch_id is not a safe artifact name: {self.batch_id}")
        if "holdout" in self.batch_id.casefold():
            raise ValueError("holdout batches are forbidden")
        if _contains_layoutlm(self.batch_id):
            raise ValueError("LayoutLM batches are forbidden")
        ids = [selection.id for selection in self.selections]
        if len(ids) != len(set(ids)):
            raise ValueError("batch selection ids must be unique")
        if self.selection_count != len(self.selections):
            raise ValueError("selection_count does not match selections")
        summary = self.verification_summary
        if summary.verified_bundle_count != len(self.selections) or summary.failed_bundle_count != 0:
            raise ValueError("verification summary does not describe a fully verified batch")
        artifact_count = sum(selection.verification.artifact_count for selection in self.selections)
        if summary.artifact_count != artifact_count:
            raise ValueError("verification summary artifact_count does not match selections")
        return self


class AggregateMetric(SchemaModel):
    value: float | None
    contributing_document_count: int = Field(ge=0)
    null_document_count: int = Field(ge=0)


class AggregateMetricsReport(SchemaModel):
    definition: str = "Unweighted arithmetic mean over non-null per-document values."
    document_count: int = Field(ge=1)
    metrics: dict[str, AggregateMetric]


class GateFailure(SchemaModel):
    category: GateCategory
    metric: str
    previous: float
    current: float
    delta: float
    rule: str


class DocumentGate(SchemaModel):
    document_id: str
    status: Literal["passed", "blocked"]
    failures: list[GateFailure]


class RegressionGateReport(SchemaModel):
    status: Literal["passed", "blocked"]
    null_policy: str = "Metrics are compared only when both baseline and current values are non-null."
    blocked_document_count: int = Field(ge=0)
    documents: list[DocumentGate]


class FileHash(SchemaModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str

    @model_validator(mode="after")
    def validate_portable_hash(self) -> Self:
        path = Path(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts or "\\" in self.path:
            raise ValueError(f"file hash path is not portable: {self.path}")
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError(f"invalid sha256 for {self.path}")
        return self


class ArtifactFingerprint(SchemaModel):
    algorithm: Literal["sha256-tree-v1"] = "sha256-tree-v1"
    root_sha256: str
    files: list[FileHash]

    @model_validator(mode="after")
    def validate_root(self) -> Self:
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("fingerprint file paths must be sorted and unique")
        expected = _hash_file_records(self.files)
        if self.root_sha256 != expected:
            raise ValueError("fingerprint aggregate root mismatch")
        return self


class NormalizedInvocation(SchemaModel):
    command: Literal["benchmark-batch"] = "benchmark-batch"
    manifest: str
    corpus_partition: str
    expected_corpus_partition_sha256: str
    reference_dir: str
    baseline_candidate_dir: str
    output_location: Literal["excluded-from-deterministic-provenance"] = (
        "excluded-from-deterministic-provenance"
    )
    selections: dict[str, list[int]]
    extraction_annotator: Literal["pdf2md-pymupdf"] = "pdf2md-pymupdf"

    @model_validator(mode="after")
    def validate_partition_hash(self) -> Self:
        if _SHA256_PATTERN.fullmatch(self.expected_corpus_partition_sha256) is None:
            raise ValueError("invalid expected corpus partition SHA-256")
        return self


class NativeBuildMetadata(SchemaModel):
    pymupdf: str
    mupdf: str
    pymupdf_bind: str
    mupdf_bind: str


class RuntimeVersions(SchemaModel):
    python: str
    implementation: str
    os_name: str
    os_release: str
    platform: str
    architecture: str
    installed_distributions: dict[str, str]
    lock_sha256: str
    native_build: NativeBuildMetadata

    @model_validator(mode="after")
    def validate_runtime_hash(self) -> Self:
        if _SHA256_PATTERN.fullmatch(self.lock_sha256) is None:
            raise ValueError("invalid runtime lock sha256")
        return self


class CurationCompatibility(SchemaModel):
    element_schema_version: str
    source_catalog_schema_version: str
    renderer_semantics_version: str
    evaluator_semantics_version: str


_EXPECTED_CURATION_COMPATIBILITY = CurationCompatibility(
    element_schema_version=SCHEMA_VERSION,
    source_catalog_schema_version=SOURCE_CATALOG_SCHEMA_VERSION,
    renderer_semantics_version=RENDERER_SEMANTICS_VERSION,
    evaluator_semantics_version=EVALUATOR_SEMANTICS_VERSION,
)


class CodeFingerprints(SchemaModel):
    code: ArtifactFingerprint
    evaluator: ArtifactFingerprint
    renderer: ArtifactFingerprint
    schema_fingerprint: ArtifactFingerprint
    curation_guidance: ArtifactFingerprint


class InputFingerprints(SchemaModel):
    manifest: ArtifactFingerprint
    corpus_partition: ArtifactFingerprint
    source: ArtifactFingerprint
    bronze: ArtifactFingerprint
    reference: ArtifactFingerprint
    baseline: ArtifactFingerprint


class _ProvenanceFoundation(SchemaModel):
    invocation: NormalizedInvocation
    runtime: RuntimeVersions
    curation_compatibility: CurationCompatibility
    code_fingerprints: CodeFingerprints
    inputs: InputFingerprints


class RunCommitment(SchemaModel):
    invocation_sha256: str
    runtime_sha256: str
    code_root_sha256: str
    evaluator_root_sha256: str
    renderer_root_sha256: str
    schema_root_sha256: str
    curation_guidance_root_sha256: str
    curation_compatibility_sha256: str
    manifest_root_sha256: str
    corpus_partition_root_sha256: str
    source_root_sha256: str
    bronze_root_sha256: str
    reference_root_sha256: str
    baseline_root_sha256: str
    candidate_root_sha256: str
    output_payload_root_sha256: str

    @model_validator(mode="after")
    def validate_hashes(self) -> Self:
        for name, value in self.model_dump().items():
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"invalid commitment hash for {name}")
        return self


class RunAnchor(SchemaModel):
    schema_version: Literal["3.0.0"] = _ANCHOR_SCHEMA_VERSION
    run_id: str
    commitment: RunCommitment
    trust_model: Literal[
        "Self-contained archive verification detects inconsistent changes, not a coordinated rewrite; "
        "publish this anchor through an externally trusted channel."
    ] = _ANCHOR_TRUST_MODEL

    @model_validator(mode="after")
    def validate_anchor_id(self) -> Self:
        if self.run_id != _derive_run_id(self.commitment):
            raise ValueError("deterministic anchor run_id mismatch")
        return self


class RunProvenance(SchemaModel):
    schema_version: Literal["4.0.0"] = _PROVENANCE_SCHEMA_VERSION
    runner_version: Literal["5.0.0"] = _RUNNER_VERSION
    report_schema_version: Literal["5.0.0"] = _REPORT_SCHEMA_VERSION
    run_id: str
    commitment: RunCommitment
    invocation: NormalizedInvocation
    runtime: RuntimeVersions
    curation_compatibility: CurationCompatibility
    code_fingerprints: CodeFingerprints
    inputs: InputFingerprints
    candidate: ArtifactFingerprint
    output_payload: ArtifactFingerprint

    @model_validator(mode="after")
    def validate_run_id(self) -> Self:
        expected = _build_commitment(
            _ProvenanceFoundation(
                invocation=self.invocation,
                runtime=self.runtime,
                curation_compatibility=self.curation_compatibility,
                code_fingerprints=self.code_fingerprints,
                inputs=self.inputs,
            ),
            self.candidate,
            self.output_payload,
        )
        if self.commitment != expected:
            raise ValueError("provenance commitment mismatch")
        if self.run_id != _derive_run_id(expected):
            raise ValueError("deterministic run_id mismatch")
        return self


class BatchRunReport(SchemaModel):
    schema_version: Literal["5.0.0"] = _REPORT_SCHEMA_VERSION
    runner_version: Literal["5.0.0"] = _RUNNER_VERSION
    provenance_schema_version: Literal["4.0.0"] = _PROVENANCE_SCHEMA_VERSION
    run_id: str
    batch_id: str
    manifest_sha256: str
    document_count: int = Field(ge=1)
    document_ids: list[str]
    regression_gate_status: Literal["passed", "blocked"]


class _PreparedSelection(SchemaModel):
    selection: BatchSelection
    source_path: Path
    source_sha256: str
    reference_sha256: str
    baseline_sha256: str
    reference: list[DocumentElement]
    baseline: list[DocumentElement]


class BatchResult(SchemaModel):
    output_dir: Path
    aggregate: AggregateMetricsReport
    baseline_aggregate: AggregateMetricsReport
    regression_gate: RegressionGateReport


def aggregate_evaluations(reports: list[EvaluationReport]) -> AggregateMetricsReport:
    if not reports:
        raise ValueError("cannot aggregate an empty evaluation batch")
    metrics: dict[str, AggregateMetric] = {}
    for name in sorted(EvaluationMetrics.model_fields):
        values = [value for report in reports if (value := getattr(report.metrics, name)) is not None]
        metrics[name] = AggregateMetric(
            value=sum(values) / len(values) if values else None,
            contributing_document_count=len(values),
            null_document_count=len(reports) - len(values),
        )
    return AggregateMetricsReport(document_count=len(reports), metrics=metrics)


def build_regression_gate(
    document_ids: list[str],
    baseline_reports: list[EvaluationReport],
    current_reports: list[EvaluationReport],
) -> RegressionGateReport:
    if (
        not document_ids
        or len(document_ids) != len(baseline_reports)
        or len(document_ids) != len(current_reports)
    ):
        raise ValueError("regression gate inputs must have the same non-zero document count")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("regression gate document ids must be unique")
    documents: list[DocumentGate] = []
    for label, baseline, current in zip(document_ids, baseline_reports, current_reports, strict=True):
        if baseline.document_id != current.document_id:
            raise ValueError(f"baseline/current document mismatch for {label}")
        failures: list[GateFailure] = []
        f1_loss = baseline.metrics.element_f1 - current.metrics.element_f1
        if f1_loss > 0.02 and not math.isclose(f1_loss, 0.02, abs_tol=1e-12):
            failures.append(
                GateFailure(
                    category="element_f1",
                    metric="element_f1",
                    previous=baseline.metrics.element_f1,
                    current=current.metrics.element_f1,
                    delta=current.metrics.element_f1 - baseline.metrics.element_f1,
                    rule="decrease greater than 0.02",
                )
            )
        cer_increase = (
            current.metrics.normalized_character_error_rate - baseline.metrics.normalized_character_error_rate
        )
        if cer_increase > 0.02 and not math.isclose(cer_increase, 0.02, abs_tol=1e-12):
            failures.append(
                GateFailure(
                    category="normalized_character_error_rate",
                    metric="normalized_character_error_rate",
                    previous=baseline.metrics.normalized_character_error_rate,
                    current=current.metrics.normalized_character_error_rate,
                    delta=cer_increase,
                    rule="increase greater than 0.02",
                )
            )
        for category, metric in _DECREASE_GATE_METRICS:
            previous = getattr(baseline.metrics, metric)
            value = getattr(current.metrics, metric)
            if previous is not None and value is not None and value < previous:
                failures.append(
                    GateFailure(
                        category=category,
                        metric=metric,
                        previous=previous,
                        current=value,
                        delta=value - previous,
                        rule="strict decrease",
                    )
                )
        for category, metric in _INCREASE_GATE_METRICS:
            previous = getattr(baseline.metrics, metric)
            value = getattr(current.metrics, metric)
            if previous is not None and value is not None and value > previous:
                failures.append(
                    GateFailure(
                        category=category,
                        metric=metric,
                        previous=previous,
                        current=value,
                        delta=value - previous,
                        rule="strict increase",
                    )
                )
        documents.append(
            DocumentGate(
                document_id=label,
                status="blocked" if failures else "passed",
                failures=failures,
            )
        )
    blocked = sum(document.status == "blocked" for document in documents)
    return RegressionGateReport(
        status="blocked" if blocked else "passed",
        blocked_document_count=blocked,
        documents=documents,
    )


def extract_document_elements(
    pdf_path: Path,
    *,
    pages: list[int],
) -> ExtractedDocument | list[DocumentElement]:
    """Compatibility seam for deterministic candidate extraction tests."""
    return extract_document_with_catalog(pdf_path, pages=pages)


def _normalize_candidate_extraction(
    extracted: ExtractedDocument | list[DocumentElement],
    document_id: str,
) -> ExtractedDocument:
    if isinstance(extracted, ExtractedDocument):
        return extracted
    return ExtractedDocument(
        elements=tuple(extracted),
        source_catalog=build_source_catalog(document_id, []),
    )


def _validate_train_partition(
    partition_bytes: bytes,
    selections: list[BatchSelection],
    root: Path,
) -> None:
    raw_partition = json.loads(partition_bytes)
    if not isinstance(raw_partition, dict) or not isinstance(raw_partition.get("documents"), list):
        raise ValueError("corpus partition must contain a documents array")
    members: dict[str, tuple[str, str, dict[str, object]]] = {}
    for raw_document in raw_partition["documents"]:
        if not isinstance(raw_document, dict):
            raise ValueError("corpus partition documents must be objects")
        path = raw_document.get("local_path")
        sha256 = raw_document.get("sha256")
        split = raw_document.get("split")
        if not isinstance(path, str) or not path:
            raise ValueError("corpus partition member has no local_path")
        if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError(f"corpus partition member has invalid SHA-256: {path}")
        if split not in {"train", "validation", "holdout"}:
            raise ValueError(f"corpus partition member has invalid split: {path}")
        portable = _validated_portable_lexical_path(path, "corpus partition local_path")
        if portable in members:
            raise ValueError(f"duplicate corpus partition source path: {portable}")
        members[portable] = (sha256, split, cast(dict[str, object], raw_document))

    for selection in selections:
        portable = _validated_portable_lexical_path(selection.source_path, f"source path for {selection.id}")
        member = members.get(portable)
        if member is None:
            raise ValueError(f"benchmark selection is missing from corpus partition: {selection.id}")
        member_sha256, split, metadata = member
        if member_sha256 != selection.source_sha256:
            raise ValueError(f"benchmark selection SHA-256 differs from corpus partition: {selection.id}")
        if split != "train":
            raise ValueError(f"benchmark selection is not assigned to TRAIN: {selection.id}")
        if _metadata_contains_layoutlm(metadata):
            raise ValueError(f"LayoutLM corpus partition member is forbidden: {selection.id}")
        lexical_source = root / portable
        resolved_source = lexical_source.resolve()
        _require_within_root(resolved_source, root, f"source for {selection.id}")
        if resolved_source != lexical_source:
            raise ValueError(f"source path must not use a symlink alias: {selection.id}")


def _require_trusted_partition_snapshot(path: Path, trusted_bytes: bytes) -> None:
    if path.read_bytes() != trusted_bytes:
        raise ValueError("corpus partition changed after trusted TRAIN preflight")


def _static_provenance_inputs(
    root: Path,
) -> tuple[RuntimeVersions, CurationCompatibility, CodeFingerprints]:
    compatibility, guidance = _curation_guidance(root)
    return _runtime_versions(), compatibility, _code_fingerprints(guidance)


def run_benchmark_batch(
    manifest_path: Path,
    reference_dir: Path,
    baseline_candidate_dir: Path,
    output_dir: Path,
    *,
    root_dir: Path,
    corpus_partition_path: Path,
    expected_corpus_partition_sha256: str,
) -> BatchResult:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"batch output already exists: {output_dir}")
    root = root_dir.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    corpus_partition_path = corpus_partition_path.resolve(strict=True)
    reference_dir = reference_dir.resolve(strict=True)
    baseline_candidate_dir = baseline_candidate_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    for path, label in (
        (manifest_path, "manifest"),
        (corpus_partition_path, "corpus partition"),
        (reference_dir, "reference directory"),
        (baseline_candidate_dir, "baseline candidate directory"),
        (output_dir, "output directory"),
    ):
        _require_within_root(path, root, label)
    for path, label in (
        (manifest_path, "manifest"),
        (reference_dir, "reference directory"),
        (baseline_candidate_dir, "baseline candidate directory"),
        (output_dir, "output directory"),
    ):
        _require_non_holdout_path(path, label)
    manifest_bytes = manifest_path.read_bytes()
    manifest = BenchmarkBatchManifest.model_validate_json(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    selections = sorted(manifest.selections, key=lambda selection: selection.id)
    partition_bytes = corpus_partition_path.read_bytes()
    partition_sha256 = hashlib.sha256(partition_bytes).hexdigest()
    if _SHA256_PATTERN.fullmatch(expected_corpus_partition_sha256) is None:
        raise ValueError("invalid expected corpus partition SHA-256")
    if partition_sha256 != expected_corpus_partition_sha256:
        raise ValueError("corpus partition SHA-256 does not match trusted expected value")
    _validate_train_partition(partition_bytes, selections, root)
    _require_trusted_partition_snapshot(corpus_partition_path, partition_bytes)
    runtime, curation_compatibility, code_fingerprints = _static_provenance_inputs(root)
    prepared = [
        _prepare_selection(
            selection,
            root,
            reference_dir,
            baseline_candidate_dir,
            trusted_partition_path=corpus_partition_path,
            trusted_partition_bytes=partition_bytes,
        )
        for selection in selections
    ]
    provenance_foundation = _build_provenance_foundation(
        root,
        manifest_path,
        corpus_partition_path,
        partition_bytes,
        expected_corpus_partition_sha256,
        reference_dir,
        baseline_candidate_dir,
        selections,
        runtime=runtime,
        curation_compatibility=curation_compatibility,
        code_fingerprints=code_fingerprints,
    )
    if provenance_foundation.inputs.manifest.files[0].sha256 != manifest_sha256:
        raise ValueError("batch manifest changed during preflight")
    source_hashes = {item.path: item.sha256 for item in provenance_foundation.inputs.source.files}
    reference_hashes = {item.path: item.sha256 for item in provenance_foundation.inputs.reference.files}
    baseline_hashes = {item.path: item.sha256 for item in provenance_foundation.inputs.baseline.files}
    for item in prepared:
        source_name = f"{item.selection.id}/{item.source_path.name}"
        if source_hashes[source_name] != item.source_sha256:
            raise ValueError(f"source changed during preflight for {item.selection.id}")
        name = f"{item.selection.id}.parquet"
        if reference_hashes[name] != item.reference_sha256:
            raise ValueError(f"reference changed during preflight for {item.selection.id}")
        if baseline_hashes[name] != item.baseline_sha256:
            raise ValueError(f"baseline candidate changed during preflight for {item.selection.id}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    completed = False
    try:
        candidate_dir = staging_dir / "candidates"
        evaluation_dir = staging_dir / "evaluations"
        baseline_evaluation_dir = staging_dir / "baseline-evaluations"
        current_reports: list[EvaluationReport] = []
        baseline_reports: list[EvaluationReport] = []
        for item in prepared:
            selection = item.selection
            _require_trusted_partition_snapshot(corpus_partition_path, partition_bytes)
            raw_extracted = extract_document_elements(item.source_path, pages=selection.pages.requested)
            _require_trusted_partition_snapshot(corpus_partition_path, partition_bytes)
            extracted = _normalize_candidate_extraction(raw_extracted, item.source_sha256)
            elements = list(extracted.elements)
            validate_source_catalog(elements, extracted.source_catalog)
            _require_trusted_partition_snapshot(corpus_partition_path, partition_bytes)
            current_source_sha256 = _sha256(item.source_path)
            _require_trusted_partition_snapshot(corpus_partition_path, partition_bytes)
            if current_source_sha256 != item.source_sha256:
                raise ValueError(f"source changed during extraction for {selection.id}")
            _validate_elements(
                elements, "candidate", item.source_sha256, selection.pages.requested, selection.id
            )
            candidate_dir.mkdir(exist_ok=True)
            markdown_path = candidate_dir / f"{selection.id}.md"
            markdown_path.write_text(render_document(elements) + "\n", encoding="utf-8")
            write_document_with_source_catalog(
                elements,
                extracted.source_catalog,
                candidate_dir / f"{selection.id}.parquet",
            )

            current_report = evaluate_document(elements, item.reference)
            baseline_report = evaluate_document(item.baseline, item.reference)
            current_reports.append(current_report)
            baseline_reports.append(baseline_report)
            _write_json(evaluation_dir / f"{selection.id}.json", current_report.model_dump(mode="json"))
            _write_json(
                baseline_evaluation_dir / f"{selection.id}.json",
                baseline_report.model_dump(mode="json"),
            )

        aggregate = aggregate_evaluations(current_reports)
        baseline_aggregate = aggregate_evaluations(baseline_reports)
        gate = build_regression_gate(
            [selection.id for selection in selections],
            baseline_reports,
            current_reports,
        )
        _write_json(staging_dir / "aggregate.json", aggregate.model_dump(mode="json"))
        _write_json(
            staging_dir / "baseline-aggregate.json",
            baseline_aggregate.model_dump(mode="json"),
        )
        _write_json(staging_dir / "regression-gate.json", gate.model_dump(mode="json"))
        current_runtime, current_compatibility, current_code = _static_provenance_inputs(root)
        current_foundation = _build_provenance_foundation(
            root,
            manifest_path,
            corpus_partition_path,
            partition_bytes,
            expected_corpus_partition_sha256,
            reference_dir,
            baseline_candidate_dir,
            selections,
            runtime=current_runtime,
            curation_compatibility=current_compatibility,
            code_fingerprints=current_code,
        )
        if current_foundation != provenance_foundation:
            raise ValueError("benchmark provenance inputs or code changed during the run")
        candidate_fingerprint = _fingerprint_directory(candidate_dir, logical_prefix="")
        output_payload_fingerprint = _fingerprint_output_payload(staging_dir)
        commitment = _build_commitment(
            provenance_foundation,
            candidate_fingerprint,
            output_payload_fingerprint,
        )
        run_id = _derive_run_id(commitment)
        run_report = BatchRunReport(
            run_id=run_id,
            batch_id=manifest.batch_id,
            manifest_sha256=manifest_sha256,
            document_count=len(prepared),
            document_ids=[selection.id for selection in selections],
            regression_gate_status=gate.status,
        )
        _write_json(staging_dir / "run.json", run_report.model_dump(mode="json"))
        (staging_dir / "report.md").write_text(
            _render_report(manifest.batch_id, aggregate, baseline_aggregate, gate, run_id),
            encoding="utf-8",
        )
        provenance = RunProvenance(
            run_id=run_id,
            commitment=commitment,
            invocation=provenance_foundation.invocation,
            runtime=provenance_foundation.runtime,
            curation_compatibility=provenance_foundation.curation_compatibility,
            code_fingerprints=provenance_foundation.code_fingerprints,
            inputs=provenance_foundation.inputs,
            candidate=candidate_fingerprint,
            output_payload=output_payload_fingerprint,
        )
        anchor = RunAnchor(run_id=run_id, commitment=commitment)
        _write_json(staging_dir / "provenance.json", provenance.model_dump(mode="json"))
        _write_json(staging_dir / "anchor.json", anchor.model_dump(mode="json"))
        verify_benchmark_run(staging_dir)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"batch output appeared during the run: {output_dir}")
        os.replace(staging_dir, output_dir)
        completed = True
    finally:
        if not completed and staging_dir.exists():
            shutil.rmtree(staging_dir)

    return BatchResult(
        output_dir=output_dir,
        aggregate=aggregate,
        baseline_aggregate=baseline_aggregate,
        regression_gate=gate,
    )


def _prepare_selection(
    selection: BatchSelection,
    root: Path,
    reference_dir: Path,
    baseline_candidate_dir: Path,
    *,
    trusted_partition_path: Path,
    trusted_partition_bytes: bytes,
) -> _PreparedSelection:
    _require_trusted_partition_snapshot(trusted_partition_path, trusted_partition_bytes)
    source_path = _resolve_input(root, selection.source_path, f"source for {selection.id}")
    bundle_path = _resolve_input(root, selection.bundle_path, f"bronze bundle for {selection.id}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not bundle_path.is_dir():
        raise FileNotFoundError(bundle_path)
    _require_trusted_partition_snapshot(trusted_partition_path, trusted_partition_bytes)
    bronze_manifest_path = bundle_path / "manifest.json"
    bronze = BronzeManifest.model_validate_json(bronze_manifest_path.read_text(encoding="utf-8"))
    _validate_bronze_artifact_paths(bundle_path, bronze, selection.id)
    verified_bronze = verify_bronze_bundle(bundle_path)
    if verified_bronze != bronze:
        raise ValueError(f"bronze manifest changed during verification for {selection.id}")
    _validate_bronze(selection, source_path, bronze, root)
    _require_trusted_partition_snapshot(trusted_partition_path, trusted_partition_bytes)
    source_sha256 = _sha256(source_path)
    if source_sha256 != selection.source_sha256:
        raise ValueError(f"source hash does not match TRAIN-approved selection for {selection.id}")
    if source_sha256 != bronze.source_sha256:
        raise ValueError(f"source hash mismatch for {selection.id}")

    _require_trusted_partition_snapshot(trusted_partition_path, trusted_partition_bytes)
    reference_path = _resolve_artifact(
        reference_dir, f"{selection.id}.parquet", f"reference for {selection.id}"
    )
    _require_trusted_partition_snapshot(trusted_partition_path, trusted_partition_bytes)
    baseline_path = _resolve_artifact(
        baseline_candidate_dir,
        f"{selection.id}.parquet",
        f"baseline candidate for {selection.id}",
    )
    reference_sha256 = _sha256(reference_path)
    baseline_sha256 = _sha256(baseline_path)
    reference = read_document_elements(reference_path)
    baseline = read_document_elements(baseline_path)
    if _sha256(reference_path) != reference_sha256:
        raise ValueError(f"reference changed while reading for {selection.id}")
    if _sha256(baseline_path) != baseline_sha256:
        raise ValueError(f"baseline candidate changed while reading for {selection.id}")
    _validate_elements(reference, "reference", source_sha256, selection.pages.requested, selection.id)
    _validate_elements(baseline, "candidate", source_sha256, selection.pages.requested, selection.id)
    return _PreparedSelection(
        selection=selection,
        source_path=source_path,
        source_sha256=source_sha256,
        reference_sha256=reference_sha256,
        baseline_sha256=baseline_sha256,
        reference=reference,
        baseline=baseline,
    )


def _validate_bronze_artifact_paths(bundle_path: Path, bronze: BronzeManifest, label: str) -> None:
    resolved_paths: list[Path] = []
    for artifact in bronze.artifacts:
        path = (bundle_path / artifact.path).resolve()
        _require_within_root(path, bundle_path, f"bronze artifact for {label}")
        _require_non_holdout_path(path, f"bronze artifact for {label}")
        resolved_paths.append(path)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError(f"bronze artifact paths must be unique for {label}")


def _resolve_artifact(directory: Path, name: str, label: str) -> Path:
    path = (directory / name).resolve()
    _require_within_root(path, directory, label)
    _require_non_holdout_path(path, label)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_bronze(
    selection: BatchSelection,
    source_path: Path,
    bronze: BronzeManifest,
    root: Path,
) -> None:
    bronze_source = _resolve_input(root, bronze.source_path, f"bronze source for {selection.id}")
    if bronze_source != source_path:
        raise ValueError(f"bronze source path mismatch for {selection.id}")
    if bronze.source_name != source_path.name:
        raise ValueError(f"bronze source name mismatch for {selection.id}")
    if bronze.source_size_bytes != source_path.stat().st_size:
        raise ValueError(f"bronze source size mismatch for {selection.id}")
    if bronze.selection != selection.pages.as_page_selection():
        raise ValueError(f"bronze page selection mismatch for {selection.id}")
    if max(selection.pages.requested) > bronze.source_page_count:
        raise ValueError(f"requested pages exceed bronze source page count for {selection.id}")
    if len(bronze.artifacts) != selection.verification.artifact_count:
        raise ValueError(f"bronze artifact count mismatch for {selection.id}")


def _validate_elements(
    elements: list[DocumentElement],
    expected_stage: Literal["candidate", "reference"],
    source_sha256: str,
    requested_pages: list[int],
    label: str,
) -> None:
    if not elements:
        raise ValueError(f"{expected_stage} elements are empty for {label}")
    stages = {element.annotation.stage for element in elements}
    valid_stages = (
        stages == {"candidate"} if expected_stage == "candidate" else stages in ({"silver"}, {"golden"})
    )
    if not valid_stages:
        raise ValueError(f"invalid {expected_stage} stages for {label}: {sorted(stages)}")
    document_ids = {element.document_id for element in elements}
    if document_ids != {source_sha256}:
        raise ValueError(f"{expected_stage} document_id does not match source hash for {label}")
    element_ids = [element.element_id for element in elements]
    if len(element_ids) != len(set(element_ids)):
        raise ValueError(f"{expected_stage} element_id values are not unique for {label}")
    orders = [element.order for element in elements]
    if orders != list(range(len(elements))):
        raise ValueError(f"{expected_stage} elements are not in contiguous order for {label}")
    actual_pages = {fragment.page_number for element in elements for fragment in element.fragments}
    nested_pages = {
        fragment.page_number
        for element in elements
        if element.structure.table is not None
        for cell in element.structure.table.cells
        for fragment in cell.fragments
    }
    requested_page_set = set(requested_pages)
    unexpected_pages = sorted((actual_pages | nested_pages) - requested_page_set)
    if unexpected_pages:
        raise ValueError(
            f"{expected_stage} elements use pages outside the requested selection for {label}: "
            f"{unexpected_pages}"
        )
    missing_pages = sorted(requested_page_set - actual_pages)
    if missing_pages:
        raise ValueError(
            f"{expected_stage} elements do not cover requested pages for {label}: {missing_pages}"
        )


def verify_benchmark_run(
    run_dir: Path,
    *,
    expected_run_id: str | None = None,
    expected_anchor: Path | RunAnchor | None = None,
) -> RunProvenance:
    """Validate an archive, optionally against a separately trusted run ID or anchor."""
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    _require_non_holdout_path(run_dir, "benchmark run directory")
    provenance = RunProvenance.model_validate_json((run_dir / "provenance.json").read_bytes())
    archive_anchor = RunAnchor.model_validate_json((run_dir / "anchor.json").read_bytes())
    report = BatchRunReport.model_validate_json((run_dir / "run.json").read_bytes())
    if archive_anchor != RunAnchor(run_id=provenance.run_id, commitment=provenance.commitment):
        raise ValueError("archived anchor/provenance mismatch")
    if expected_run_id is not None and provenance.run_id != expected_run_id:
        raise ValueError("trusted expected run_id mismatch")
    if expected_anchor is not None:
        trusted_anchor = (
            RunAnchor.model_validate_json(expected_anchor.read_bytes())
            if isinstance(expected_anchor, Path)
            else expected_anchor
        )
        if archive_anchor != trusted_anchor:
            raise ValueError("trusted expected anchor mismatch")
    if report.run_id != provenance.run_id:
        raise ValueError("run report/provenance run_id mismatch")
    manifest_files = provenance.inputs.manifest.files
    if len(manifest_files) != 1 or report.manifest_sha256 != manifest_files[0].sha256:
        raise ValueError("run report/provenance manifest hash mismatch")
    partition_files = provenance.inputs.corpus_partition.files
    if (
        len(partition_files) != 1
        or partition_files[0].sha256 != provenance.invocation.expected_corpus_partition_sha256
    ):
        raise ValueError("provenance corpus partition hash mismatch")
    actual_candidate = _fingerprint_directory(run_dir / "candidates", logical_prefix="")
    if actual_candidate != provenance.candidate:
        raise ValueError("archived candidate fingerprint mismatch")
    actual_payload = _fingerprint_output_payload(run_dir)
    if actual_payload != provenance.output_payload:
        raise ValueError("archived output payload fingerprint mismatch")
    expected_files = {
        "anchor.json",
        "provenance.json",
        "report.md",
        "run.json",
        *(f"candidates/{item.path}" for item in provenance.candidate.files),
        *(item.path for item in provenance.output_payload.files),
    }
    actual_files = {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("archived file set mismatch")
    candidate_parquets = sorted(
        path
        for path in (run_dir / "candidates").glob("*.parquet")
        if not path.name.endswith(".source-items.parquet")
    )
    if len(candidate_parquets) != report.document_count:
        raise ValueError("archived candidate document count mismatch")
    for candidate_path in candidate_parquets:
        read_document_with_source_catalog(candidate_path)
    aggregate = AggregateMetricsReport.model_validate_json((run_dir / "aggregate.json").read_bytes())
    baseline = AggregateMetricsReport.model_validate_json((run_dir / "baseline-aggregate.json").read_bytes())
    gate = RegressionGateReport.model_validate_json((run_dir / "regression-gate.json").read_bytes())
    expected_report = _render_report(
        report.batch_id,
        aggregate,
        baseline,
        gate,
        provenance.run_id,
    )
    if (run_dir / "report.md").read_text(encoding="utf-8") != expected_report:
        raise ValueError("Markdown report content mismatch")
    return provenance


def _build_provenance_foundation(
    root: Path,
    manifest_path: Path,
    corpus_partition_path: Path,
    trusted_partition_bytes: bytes,
    expected_corpus_partition_sha256: str,
    reference_dir: Path,
    baseline_candidate_dir: Path,
    selections: list[BatchSelection],
    *,
    runtime: RuntimeVersions,
    curation_compatibility: CurationCompatibility,
    code_fingerprints: CodeFingerprints,
) -> _ProvenanceFoundation:
    invocation = NormalizedInvocation(
        manifest=_portable_relative(manifest_path, root),
        corpus_partition=_portable_relative(corpus_partition_path, root),
        expected_corpus_partition_sha256=expected_corpus_partition_sha256,
        reference_dir=_portable_relative(reference_dir, root),
        baseline_candidate_dir=_portable_relative(baseline_candidate_dir, root),
        selections={selection.id: selection.pages.requested for selection in selections},
    )
    source_files: list[tuple[str, Path]] = []
    bronze_files: list[tuple[str, Path]] = []
    reference_files: list[tuple[str, Path]] = []
    baseline_files: list[tuple[str, Path]] = []
    for selection in selections:
        _require_trusted_partition_snapshot(corpus_partition_path, trusted_partition_bytes)
        source_path = _resolve_input(root, selection.source_path, f"source for {selection.id}")
        bundle_path = _resolve_input(root, selection.bundle_path, f"bronze bundle for {selection.id}")
        _require_trusted_partition_snapshot(corpus_partition_path, trusted_partition_bytes)
        bronze_manifest_path = bundle_path / "manifest.json"
        bronze = BronzeManifest.model_validate_json(bronze_manifest_path.read_bytes())
        if verify_bronze_bundle(bundle_path) != bronze:
            raise ValueError(f"bronze manifest changed during provenance snapshot for {selection.id}")
        source_files.append((f"{selection.id}/{source_path.name}", source_path))
        bronze_files.append((f"{selection.id}/manifest.json", bronze_manifest_path))
        for artifact in bronze.artifacts:
            artifact_path = (bundle_path / artifact.path).resolve(strict=True)
            _require_within_root(artifact_path, bundle_path, f"bronze artifact for {selection.id}")
            bronze_files.append((f"{selection.id}/{artifact.path}", artifact_path))
        reference_files.append((
            f"{selection.id}.parquet",
            _resolve_artifact(reference_dir, f"{selection.id}.parquet", f"reference for {selection.id}"),
        ))
        baseline_files.append((
            f"{selection.id}.parquet",
            _resolve_artifact(
                baseline_candidate_dir,
                f"{selection.id}.parquet",
                f"baseline candidate for {selection.id}",
            ),
        ))
    _require_trusted_partition_snapshot(corpus_partition_path, trusted_partition_bytes)
    manifest_fingerprint = _fingerprint_paths([("manifest.json", manifest_path)])
    partition_fingerprint = _fingerprint_bytes("corpus-partition.json", trusted_partition_bytes)
    source_fingerprint = _fingerprint_paths_with_partition_guard(
        source_files,
        trusted_partition_path=corpus_partition_path,
        trusted_partition_bytes=trusted_partition_bytes,
    )
    _require_trusted_partition_snapshot(corpus_partition_path, trusted_partition_bytes)
    bronze_fingerprint = _fingerprint_paths(bronze_files)
    _require_trusted_partition_snapshot(corpus_partition_path, trusted_partition_bytes)
    reference_fingerprint = _fingerprint_paths(reference_files)
    _require_trusted_partition_snapshot(corpus_partition_path, trusted_partition_bytes)
    baseline_fingerprint = _fingerprint_paths(baseline_files)
    return _ProvenanceFoundation(
        invocation=invocation,
        runtime=runtime,
        curation_compatibility=curation_compatibility,
        code_fingerprints=code_fingerprints,
        inputs=InputFingerprints(
            manifest=manifest_fingerprint,
            corpus_partition=partition_fingerprint,
            source=source_fingerprint,
            bronze=bronze_fingerprint,
            reference=reference_fingerprint,
            baseline=baseline_fingerprint,
        ),
    )


def _local_imported_module_paths(project_root: Path) -> set[Path]:
    app_root = (project_root / "app").resolve()
    imported: set[Path] = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.suffix in {".pyc", ".pyo"}:
            path = Path(importlib.util.source_from_cache(str(path)))
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(app_root)
        except (FileNotFoundError, ValueError):
            continue
        if resolved.suffix == ".py":
            imported.add(resolved)
    return imported


def _code_inventory(project_root: Path) -> list[tuple[str, Path]]:
    app_root = project_root / "app"
    paths = {path.resolve(strict=True) for path in app_root.rglob("*.py") if path.is_file()}
    imported = _local_imported_module_paths(project_root)
    omitted = imported - paths
    if omitted:
        rendered = ", ".join(sorted(path.as_posix() for path in omitted))
        raise ValueError(f"imported local module omitted from code fingerprint inventory: {rendered}")
    code_paths = [(path.relative_to(project_root).as_posix(), path) for path in paths]
    for name in ("pyproject.toml", "uv.lock"):
        path = project_root / name
        if path.is_file():
            code_paths.append((name, path.resolve(strict=True)))
    return sorted(code_paths)


def project_code_fingerprint() -> ArtifactFingerprint:
    """Return the runner-v3 fingerprint for the complete local project closure."""
    project_root = Path(__file__).resolve().parents[2]
    return _fingerprint_paths(_code_inventory(project_root))


def runtime_versions() -> RuntimeVersions:
    """Return runner-v3 runtime, platform, lock, distribution, and native-build pins."""
    return _runtime_versions()


def _curation_guidance(root: Path) -> tuple[CurationCompatibility, ArtifactFingerprint]:
    path = (root / _CURATION_GUIDANCE_PATH).resolve()
    _require_within_root(path, root, "curation guidance")
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("curation guidance requires YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("curation guidance frontmatter is not terminated")
    declarations: dict[str, str] = {}
    required = {
        "element-schema-version": "element_schema_version",
        "source-catalog-schema-version": "source_catalog_schema_version",
        "renderer-semantics-version": "renderer_semantics_version",
        "evaluator-semantics-version": "evaluator_semantics_version",
    }
    for line in text[4:end].splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator or key not in required:
            continue
        if key in declarations:
            raise ValueError(f"duplicate curation guidance compatibility declaration: {key}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise ValueError(f"empty curation guidance compatibility declaration: {key}")
        declarations[key] = value
    missing = sorted(set(required) - set(declarations))
    if missing:
        raise ValueError(f"missing curation guidance compatibility declarations: {', '.join(missing)}")
    compatibility = CurationCompatibility.model_validate(
        {required[key]: declarations[key] for key in required},
        strict=True,
    )
    if compatibility != _EXPECTED_CURATION_COMPATIBILITY:
        raise ValueError(
            "curation guidance compatibility mismatch: "
            f"declared={compatibility.model_dump(mode='json')} "
            f"expected={_EXPECTED_CURATION_COMPATIBILITY.model_dump(mode='json')}"
        )
    fingerprint = _fingerprint_paths([(_CURATION_GUIDANCE_PATH.as_posix(), path)])
    return compatibility, fingerprint


def _code_fingerprints(curation_guidance: ArtifactFingerprint) -> CodeFingerprints:
    project_root = Path(__file__).resolve().parents[2]
    package_dir = project_root / "app/pdf2md"
    code_paths = _code_inventory(project_root)
    # The executable package closure includes every local production module. This
    # captures modules imported by the CLI (including active blind builders), and
    # catches files imported lazily after the initial snapshot on the end snapshot.
    package_paths = [item for item in code_paths if item[0].startswith("app/")]
    package_fingerprint = _fingerprint_paths(package_paths)
    return CodeFingerprints(
        code=_fingerprint_paths(code_paths),
        evaluator=package_fingerprint,
        renderer=package_fingerprint,
        schema_fingerprint=_fingerprint_paths([("app/pdf2md/schema.py", package_dir / "schema.py")]),
        curation_guidance=curation_guidance,
    )


def _runtime_versions() -> RuntimeVersions:
    import pymupdf

    project_root = Path(__file__).resolve().parents[2]
    distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise ValueError("installed distribution is missing its Name metadata")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        previous = distributions.setdefault(name, distribution.version)
        if previous != distribution.version:
            raise ValueError(f"conflicting installed distribution versions for {name}")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return RuntimeVersions(
        python=version,
        implementation=platform.python_implementation(),
        os_name=platform.system(),
        os_release=platform.release(),
        platform=sys.platform,
        architecture=platform.machine(),
        installed_distributions=dict(sorted(distributions.items())),
        lock_sha256=_sha256(project_root / "uv.lock"),
        native_build=NativeBuildMetadata(
            pymupdf=importlib.metadata.version("pymupdf"),
            mupdf=pymupdf.mupdf_version,
            pymupdf_bind=pymupdf.VersionBind,
            mupdf_bind=pymupdf.VersionFitz,
        ),
    )


def _build_commitment(
    foundation: _ProvenanceFoundation,
    candidate: ArtifactFingerprint,
    output_payload: ArtifactFingerprint,
) -> RunCommitment:
    code = foundation.code_fingerprints
    inputs = foundation.inputs
    return RunCommitment(
        invocation_sha256=hashlib.sha256(
            _canonical_json(foundation.invocation.model_dump(mode="json"))
        ).hexdigest(),
        runtime_sha256=hashlib.sha256(
            _canonical_json(foundation.runtime.model_dump(mode="json"))
        ).hexdigest(),
        code_root_sha256=code.code.root_sha256,
        evaluator_root_sha256=code.evaluator.root_sha256,
        renderer_root_sha256=code.renderer.root_sha256,
        schema_root_sha256=code.schema_fingerprint.root_sha256,
        curation_guidance_root_sha256=code.curation_guidance.root_sha256,
        curation_compatibility_sha256=hashlib.sha256(
            _canonical_json(foundation.curation_compatibility.model_dump(mode="json"))
        ).hexdigest(),
        manifest_root_sha256=inputs.manifest.root_sha256,
        corpus_partition_root_sha256=inputs.corpus_partition.root_sha256,
        source_root_sha256=inputs.source.root_sha256,
        bronze_root_sha256=inputs.bronze.root_sha256,
        reference_root_sha256=inputs.reference.root_sha256,
        baseline_root_sha256=inputs.baseline.root_sha256,
        candidate_root_sha256=candidate.root_sha256,
        output_payload_root_sha256=output_payload.root_sha256,
    )


def _derive_run_id(commitment: RunCommitment) -> str:
    payload = {
        "anchor_schema_version": _ANCHOR_SCHEMA_VERSION,
        "provenance_schema_version": _PROVENANCE_SCHEMA_VERSION,
        "runner_version": _RUNNER_VERSION,
        "report_schema_version": _REPORT_SCHEMA_VERSION,
        "commitment": commitment.model_dump(mode="json"),
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"sha256-{digest}"


def _fingerprint_output_payload(run_dir: Path) -> ArtifactFingerprint:
    """Fingerprint run outputs that do not contain or depend on the run ID."""
    paths = [
        (name, run_dir / name)
        for name in ("aggregate.json", "baseline-aggregate.json", "regression-gate.json")
    ]
    for directory_name in ("baseline-evaluations", "evaluations"):
        directory = run_dir / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(
            (path.relative_to(run_dir).as_posix(), path) for path in directory.rglob("*") if path.is_file()
        )
    return _fingerprint_paths(paths)


def _fingerprint_directory(
    directory: Path,
    *,
    logical_prefix: str,
    excluded_paths: set[str] | None = None,
) -> ArtifactFingerprint:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    excluded = excluded_paths or set()
    paths = [
        (
            f"{logical_prefix}{path.relative_to(directory).as_posix()}",
            path,
        )
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() not in excluded
    ]
    return _fingerprint_paths(paths)


def _fingerprint_bytes(logical: str, payload: bytes) -> ArtifactFingerprint:
    record = FileHash(path=logical, size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return ArtifactFingerprint(root_sha256=_hash_file_records([record]), files=[record])


def _fingerprint_paths(paths: list[tuple[str, Path]]) -> ArtifactFingerprint:
    logical_paths = [logical for logical, _ in paths]
    if len(logical_paths) != len(set(logical_paths)):
        raise ValueError("fingerprint input paths must be unique")
    records = sorted(
        (_fingerprint_file(logical, path) for logical, path in paths),
        key=lambda item: item.path,
    )
    return ArtifactFingerprint(root_sha256=_hash_file_records(records), files=records)


def _fingerprint_paths_with_partition_guard(
    paths: list[tuple[str, Path]],
    *,
    trusted_partition_path: Path,
    trusted_partition_bytes: bytes,
) -> ArtifactFingerprint:
    logical_paths = [logical for logical, _ in paths]
    if len(logical_paths) != len(set(logical_paths)):
        raise ValueError("fingerprint input paths must be unique")
    records: list[FileHash] = []
    for logical, path in paths:
        _require_trusted_partition_snapshot(trusted_partition_path, trusted_partition_bytes)
        records.append(_fingerprint_file(logical, path))
    _require_trusted_partition_snapshot(trusted_partition_path, trusted_partition_bytes)
    records.sort(key=lambda item: item.path)
    return ArtifactFingerprint(root_sha256=_hash_file_records(records), files=records)


def _fingerprint_file(logical: str, path: Path) -> FileHash:
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    after = path.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if before_signature != after_signature or size != after.st_size:
        raise ValueError(f"file changed while fingerprinting: {logical}")
    return FileHash(path=logical, size_bytes=size, sha256=digest.hexdigest())


def _hash_file_records(records: list[FileHash]) -> str:
    payload = [record.model_dump(mode="json") for record in records]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _portable_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError(f"path is not portable within benchmark root: {path}")
    return relative


def _resolve_input(root: Path, value: str, label: str) -> Path:
    path = (root / value).resolve()
    _require_within_root(path, root, label)
    _require_non_holdout_path(path, label)
    return path


def _require_within_root(path: Path, root: Path, label: str) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the benchmark root: {path}")


def _require_non_holdout_path(path: Path, label: str) -> None:
    if any("holdout" in part.casefold() for part in path.parts):
        raise ValueError(f"{label} must not reference holdout data")
    if any(_contains_layoutlm(part) for part in path.parts):
        raise ValueError(f"{label} must not reference LayoutLM data")


def _render_report(
    batch_id: str,
    current: AggregateMetricsReport,
    baseline: AggregateMetricsReport,
    gate: RegressionGateReport,
    run_id: str,
) -> str:
    lines = [
        f"# Benchmark batch {batch_id}",
        "",
        f"Run ID: `{run_id}`",
        f"Documents: {current.document_count}",
        f"Regression gate: **{gate.status}** ({gate.blocked_document_count} blocked)",
        "",
        "Null metrics are excluded from macro means, never coerced to zero. `n` is the contributing document count.",
        "",
        "| Metric | Baseline macro | Current macro | n (baseline/current) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(current.metrics):
        previous = baseline.metrics[name]
        value = current.metrics[name]
        lines.append(
            f"| {name} | {_format_metric(previous.value)} | {_format_metric(value.value)} | "
            f"{previous.contributing_document_count}/{value.contributing_document_count} |"
        )
    lines.extend([
        "",
        "## Per-document regression gate",
        "",
        "| Document | Status | Failures |",
        "| --- | --- | --- |",
    ])
    for document in gate.documents:
        failures = ", ".join(failure.metric for failure in document.failures) or "—"
        lines.append(f"| {document.document_id} | {document.status} | {failures} |")
    return "\n".join(lines) + "\n"


def _format_metric(value: float | None) -> str:
    return "null" if value is None else f"{value:.6f}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
