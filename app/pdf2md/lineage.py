from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from app.pdf2md.benchmark_batch import BatchSelection, BenchmarkBatchManifest
from app.pdf2md.bronze import BronzeManifest, verify_bronze_bundle
from app.pdf2md.evaluation import ReferenceChurnReport, evaluate_reference_churn
from app.pdf2md.schema import DocumentElement, SchemaModel, read_document_elements
from app.pdf2md.source_catalog import (
    SourceCatalogReferenceCounts,
    read_source_catalog,
    source_catalog_path,
    source_catalog_reference_counts,
    validate_source_catalog,
)

ChurnStatus = Literal["replayable", "current-snapshot-only", "stale", "invalid"]
ChurnReasonCode = Literal[
    "churn_artifact_invalid",
    "candidate_identity_fields_partial",
    "churn_rates_invalid",
    "churn_counts_negative",
    "churn_common_count_mismatch",
    "churn_unchanged_count_mismatch",
    "churn_rate_mismatch",
    "document_id_mismatch",
    "current_count_mismatch",
    "candidate_revision_not_found",
    "historical_inputs_unavailable",
]
_SELECTION_ID = re.compile(r"[a-z0-9][a-z0-9._-]*$")
_LITEPARSE_SOURCE_ITEM_ID = re.compile(r"liteparse:p(?P<page>\d+):(?:item(?P<item>\d+)|text-(?P<text>\d+))$")
_PYMUPDF_SOURCE_ITEM_ID = re.compile(
    r"pymupdf:d(?P<document>[0-9a-f]{64}):p(?P<page>\d{6}):"
    r"(?P<kind>span|word|rule):h(?P<identity>[0-9a-f]{64}):d(?P<duplicate>\d{6})$"
)


class RevisionSummary(SchemaModel):
    revision: int
    parent_revision: int | None
    element_count: int


class CandidateIdentityMatch(SchemaModel):
    path: str
    sha256: str


class LineageArtifactIdentity(SchemaModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LineageTransitionClaim(SchemaModel):
    selection_id: str
    parent: LineageArtifactIdentity
    output: LineageArtifactIdentity
    expected_churn: ReferenceChurnReport
    candidate: LineageArtifactIdentity | None = None
    no_change: bool = False
    superseded: bool = False
    evidence_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        if _SELECTION_ID.fullmatch(self.selection_id) is None:
            raise ValueError("lineage transition selection_id is not a safe artifact name")
        if self.parent.path == self.output.path:
            raise ValueError("lineage transition parent and output paths must differ")
        identity_rates = (
            self.expected_churn.candidate_identity_rate,
            self.expected_churn.candidate_content_identity_rate,
        )
        if (identity_rates[0] is None) != (identity_rates[1] is None):
            raise ValueError("candidate identity fields must both be present or both be null")
        if (self.candidate is None) != (identity_rates[0] is None):
            raise ValueError("candidate artifact presence must match expected candidate identity fields")
        return self


class VerifiedLineageTransition(SchemaModel):
    selection_id: str
    parent: LineageArtifactIdentity
    output: LineageArtifactIdentity
    candidate: LineageArtifactIdentity | None = None
    replayed_churn: ReferenceChurnReport
    no_change: bool
    superseded: bool
    evidence_paths: list[str]


class VerifiedLineageChain(SchemaModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    selection_id: str
    document_id: str
    transitions: list[VerifiedLineageTransition]
    latest: LineageArtifactIdentity
    replayable: Literal[True] = True


class SilverLineageStatus(SchemaModel):
    path: str
    sha256: str
    element_count: int
    document_id: str
    revisions: list[RevisionSummary]
    snapshot_parent_links_valid: bool
    historical_ancestors_available: bool = False
    source_item_reference_count: int
    unique_source_item_count: int
    catalog_reference_counts: SourceCatalogReferenceCounts


class ChurnArtifactStatus(SchemaModel):
    path: str
    status: ChurnStatus
    reason_code: ChurnReasonCode | None = None
    reason: str
    claimed_previous_count: int | None = None
    claimed_current_count: int | None = None
    candidate_identity_matches: list[CandidateIdentityMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reason_code(self) -> ChurnArtifactStatus:
        allowed_codes: dict[ChurnStatus, set[ChurnReasonCode | None]] = {
            "replayable": {None},
            "current-snapshot-only": {"historical_inputs_unavailable"},
            "stale": {"current_count_mismatch"},
            "invalid": {
                "churn_artifact_invalid",
                "candidate_identity_fields_partial",
                "churn_rates_invalid",
                "churn_counts_negative",
                "churn_common_count_mismatch",
                "churn_unchanged_count_mismatch",
                "churn_rate_mismatch",
                "document_id_mismatch",
                "candidate_revision_not_found",
            },
        }
        if self.reason_code not in allowed_codes[self.status]:
            raise ValueError(f"reason_code {self.reason_code!r} is invalid for churn status {self.status}")
        return self


class DocumentLineageStatus(SchemaModel):
    selection_id: str
    bronze_manifest_path: str
    bronze_source_sha256: str
    bronze_artifact_count: int
    silver: SilverLineageStatus
    churn: ChurnArtifactStatus


class CycleLineageReport(SchemaModel):
    schema_version: Literal["1.2.0"] = "1.2.0"
    batch_id: str
    selection_count: int
    holdout_inspected: bool = False
    layoutlm_used: bool = False
    documents: list[DocumentLineageStatus]
    churn_status_counts: dict[str, int]


def serialize_cycle_lineage_report(report: CycleLineageReport) -> str:
    """Return canonical JSON suitable for a stable, reviewable lineage artifact."""
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def audit_lineage_chain(
    repository_root: Path,
    selection_id: str,
    claims: list[LineageTransitionClaim],
) -> VerifiedLineageChain:
    """Replay an explicitly claimed immutable parent chain without inferring history.

    Every path and digest must be supplied by the caller. The function validates
    identities, re-evaluates exact churn from the archived Parquets, and requires
    adjacent output/parent hashes to agree. Merely finding similar artifacts does
    not establish a lineage link.
    """
    if not claims:
        raise ValueError("lineage chain requires at least one transition")
    if _SELECTION_ID.fullmatch(selection_id) is None:
        raise ValueError("lineage chain selection_id is not a safe artifact name")
    root = repository_root.resolve(strict=True)
    if not root.is_dir():
        raise LineageValidationError("repository_root_invalid", root, "repository root must be a directory")
    verified: list[VerifiedLineageTransition] = []
    document_id: str | None = None
    previous_output: LineageArtifactIdentity | None = None
    for index, claim in enumerate(claims):
        if claim.selection_id != selection_id:
            raise LineageValidationError(
                "lineage_selection_mismatch", root, f"{claim.selection_id} != {selection_id}"
            )
        if claim.superseded != (index < len(claims) - 1):
            raise LineageValidationError(
                "lineage_superseded_invalid",
                root,
                f"transition {index} superseded={claim.superseded}",
            )
        parent_path = _verify_claimed_artifact(root, claim.parent, selection_id, "parent")
        output_path = _verify_claimed_artifact(root, claim.output, selection_id, "output")
        if previous_output is not None and previous_output.sha256 != claim.parent.sha256:
            raise LineageValidationError(
                "lineage_parent_chain_mismatch",
                parent_path,
                f"{claim.parent.sha256} != {previous_output.sha256}",
            )
        parent = _read_lineage_elements(parent_path, "parent")
        output = _read_lineage_elements(output_path, "output")
        transition_document_ids = {element.document_id for element in (*parent, *output)}
        if len(transition_document_ids) != 1:
            raise LineageValidationError(
                "lineage_document_id_mismatch",
                output_path,
                str(sorted(transition_document_ids)),
            )
        actual_document_id = next(iter(transition_document_ids))
        if document_id is None:
            document_id = actual_document_id
        elif document_id != actual_document_id:
            raise LineageValidationError(
                "lineage_document_id_mismatch",
                output_path,
                f"{actual_document_id} != {document_id}",
            )
        candidate = None
        if claim.candidate is not None:
            candidate_path = _verify_claimed_artifact(root, claim.candidate, selection_id, "candidate")
            candidate = _read_lineage_elements(candidate_path, "candidate")
            candidate_document_ids = {element.document_id for element in candidate}
            if candidate_document_ids != {actual_document_id}:
                raise LineageValidationError(
                    "lineage_candidate_document_id_mismatch",
                    candidate_path,
                    str(sorted(candidate_document_ids)),
                )
        replayed = evaluate_reference_churn(parent, output, candidate=candidate)
        if replayed != claim.expected_churn:
            raise LineageValidationError(
                "lineage_churn_replay_mismatch",
                output_path,
                "replayed churn does not exactly match the claimed machine-readable churn",
            )
        actual_no_change = claim.parent.sha256 == claim.output.sha256
        if claim.no_change != actual_no_change:
            raise LineageValidationError(
                "lineage_no_change_mismatch",
                output_path,
                f"claimed no_change={claim.no_change}, byte-identical={actual_no_change}",
            )
        verified.append(
            VerifiedLineageTransition(
                selection_id=selection_id,
                parent=claim.parent,
                output=claim.output,
                candidate=claim.candidate,
                replayed_churn=replayed,
                no_change=claim.no_change,
                superseded=claim.superseded,
                evidence_paths=claim.evidence_paths,
            )
        )
        previous_output = claim.output
    if document_id is None or previous_output is None:
        raise AssertionError("validated non-empty chain did not produce an identity")
    return VerifiedLineageChain(
        selection_id=selection_id,
        document_id=document_id,
        transitions=verified,
        latest=previous_output,
    )


class LineageValidationError(ValueError):
    """A fail-fast lineage failure with a stable machine-readable code."""

    def __init__(self, code: str, path: Path, detail: str) -> None:
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(f"{code}: {path}: {detail}")

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": str(self.path), "detail": self.detail}


def audit_cycle_lineage(
    repository_root: Path,
    batch_manifest_path: Path,
    silver_dir: Path,
    churn_dir: Path,
) -> CycleLineageReport:
    """Validate one DEV cycle and classify its legacy churn artifacts.

    Bronze and current-silver defects fail immediately. Legacy aggregate churn
    claims are retained and classified rather than rewritten. ``replayable``
    requires identified, present historical inputs. ``current-snapshot-only``
    reproduces the current aggregate and any candidate identity claims but lacks
    the previous silver input. ``stale`` is internally valid but its claimed
    current count identifies an older snapshot. ``invalid`` is malformed,
    internally inconsistent, or makes a candidate identity claim that matches no
    extant candidate revision.
    """
    unresolved_root = repository_root.absolute()
    root_symlink = _first_symlink_component(unresolved_root)
    if root_symlink is not None:
        raise LineageValidationError("symlink_forbidden", root_symlink, "repository root uses a symlink")
    root = repository_root.resolve()
    if not root.is_dir():
        raise LineageValidationError("repository_root_invalid", root, "repository root must be a directory")
    batch_path = _resolve_within(root, batch_manifest_path)
    silver_root = _resolve_within(root, silver_dir)
    churn_root = _resolve_within(root, churn_dir)
    if any(_forbidden(value) for value in (str(batch_path), str(silver_root), str(churn_root))):
        raise LineageValidationError("forbidden_scope", batch_path, "holdout and LayoutLM are excluded")
    batch = _load_model(batch_path, BenchmarkBatchManifest)

    documents: list[DocumentLineageStatus] = []
    for selection in batch.selections:
        if _forbidden(" ".join((selection.id, selection.source_path, selection.bundle_path))):
            raise LineageValidationError(
                "forbidden_selection", batch_path, f"selection is outside DEV scope: {selection.id}"
            )
        bronze_dir = _resolve_within(root, Path(selection.bundle_path))
        bronze = _verify_bronze(selection, bronze_dir, root)
        silver_path = _resolve_within(root, silver_root / f"{selection.id}.parquet")
        silver, silver_status = _verify_silver(selection, bronze, bronze_dir, silver_path, root)
        churn_path = _resolve_within(root, churn_root / f"{selection.id}.json")
        churn_status = _classify_churn(
            selection.id,
            bronze.source_sha256,
            silver,
            silver_path,
            churn_path,
            root,
        )
        documents.append(
            DocumentLineageStatus(
                selection_id=selection.id,
                bronze_manifest_path=_relative(bronze_dir / "manifest.json", root),
                bronze_source_sha256=bronze.source_sha256,
                bronze_artifact_count=len(bronze.artifacts),
                silver=silver_status,
                churn=churn_status,
            )
        )

    expected_silver = {f"{selection.id}.parquet" for selection in batch.selections}
    actual_silver = {
        path.name for path in silver_root.glob("*.parquet") if not path.name.endswith(".source-items.parquet")
    }
    companion_silver = {path.name for path in silver_root.glob("*.source-items.parquet")}
    allowed_companions = {f"{selection.id}.source-items.parquet" for selection in batch.selections}
    if actual_silver != expected_silver or not companion_silver.issubset(allowed_companions):
        raise LineageValidationError(
            "silver_set_mismatch",
            silver_root,
            f"missing={sorted(expected_silver - actual_silver)}, "
            f"extra={sorted(actual_silver - expected_silver)}, "
            f"unexpected_companions={sorted(companion_silver - allowed_companions)}",
        )
    expected_churn = {f"{selection.id}.json" for selection in batch.selections}
    actual_churn = {path.name for path in churn_root.glob("*.json")}
    if actual_churn != expected_churn:
        raise LineageValidationError(
            "churn_set_mismatch",
            churn_root,
            f"missing={sorted(expected_churn - actual_churn)}, extra={sorted(actual_churn - expected_churn)}",
        )

    counts = Counter(document.churn.status for document in documents)
    return CycleLineageReport(
        batch_id=batch.batch_id,
        selection_count=len(documents),
        documents=documents,
        churn_status_counts={
            status: counts.get(status, 0)
            for status in cast(tuple[str, ...], ("replayable", "current-snapshot-only", "stale", "invalid"))
        },
    )


def _verify_bronze(selection: BatchSelection, bronze_dir: Path, root: Path) -> BronzeManifest:
    manifest_path = _resolve_within(root, bronze_dir / "manifest.json")
    manifest = _load_model(manifest_path, BronzeManifest)
    artifact_paths: list[Path] = []
    for artifact in manifest.artifacts:
        if _forbidden(artifact.path):
            raise LineageValidationError(
                "forbidden_bronze_artifact", manifest_path, f"excluded artifact path: {artifact.path}"
            )
        relative_path = Path(artifact.path)
        if relative_path.is_absolute():
            raise LineageValidationError(
                "bronze_artifact_path_invalid", manifest_path, f"absolute artifact path: {artifact.path}"
            )
        artifact_path = _resolve_within(bronze_dir, relative_path)
        artifact_paths.append(artifact_path)
    if len(artifact_paths) != len(set(artifact_paths)):
        raise LineageValidationError(
            "bronze_artifact_path_invalid", manifest_path, "duplicate artifact paths"
        )
    bundle_entries = list(bronze_dir.rglob("*"))
    symlinks = [path for path in bundle_entries if path.is_symlink()]
    if symlinks:
        raise LineageValidationError(
            "symlink_forbidden", symlinks[0], "bronze bundles cannot contain symlinks"
        )
    actual_files = {
        path.resolve() for path in bundle_entries if path.is_file() and path.resolve() != manifest_path
    }
    if actual_files != set(artifact_paths):
        raise LineageValidationError(
            "bronze_artifact_set_mismatch",
            bronze_dir,
            f"missing={sorted(str(path) for path in set(artifact_paths) - actual_files)}, "
            f"extra={sorted(str(path) for path in actual_files - set(artifact_paths))}",
        )
    try:
        verified_manifest = verify_bronze_bundle(bronze_dir)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise LineageValidationError("bronze_verification_failed", bronze_dir, str(error)) from error
    if verified_manifest != manifest:
        raise LineageValidationError(
            "bronze_manifest_changed", manifest_path, "manifest changed during verification"
        )
    if len(manifest.artifacts) != selection.verification.artifact_count:
        raise LineageValidationError(
            "bronze_artifact_count_mismatch",
            manifest_path,
            f"{len(manifest.artifacts)} != {selection.verification.artifact_count}",
        )
    if manifest.source_path != selection.source_path:
        raise LineageValidationError(
            "bronze_source_path_mismatch", bronze_dir, f"{manifest.source_path} != {selection.source_path}"
        )
    expected_pages = selection.pages
    actual_pages = manifest.selection
    if (
        actual_pages.annotated_pages != expected_pages.annotated
        or actual_pages.context_pages != expected_pages.context
        or actual_pages.requested_pages != expected_pages.requested
    ):
        raise LineageValidationError("bronze_page_selection_mismatch", bronze_dir, selection.id)
    source_path = _resolve_within(root, Path(manifest.source_path))
    if not source_path.is_file():
        raise LineageValidationError("bronze_source_missing", source_path, selection.id)
    if manifest.source_name != source_path.name:
        raise LineageValidationError(
            "bronze_source_name_mismatch", source_path, f"{manifest.source_name} != {source_path.name}"
        )
    if max(actual_pages.rendered_pages, default=0) > manifest.source_page_count:
        raise LineageValidationError("bronze_page_count_mismatch", manifest_path, selection.id)
    if source_path.stat().st_size != manifest.source_size_bytes:
        raise LineageValidationError("bronze_source_size_mismatch", source_path, selection.id)
    if _sha256(source_path) != manifest.source_sha256:
        raise LineageValidationError("bronze_source_hash_mismatch", source_path, selection.id)
    if manifest.source_sha256 != selection.source_sha256:
        raise LineageValidationError(
            "selection_source_hash_mismatch",
            manifest_path,
            f"{selection.source_sha256} != {manifest.source_sha256}",
        )
    return manifest


def _verify_silver(
    selection: BatchSelection,
    bronze: BronzeManifest,
    bronze_dir: Path,
    silver_path: Path,
    root: Path,
) -> tuple[list[DocumentElement], SilverLineageStatus]:
    if not silver_path.is_file():
        raise LineageValidationError("silver_missing", silver_path, selection.id)
    try:
        elements = read_document_elements(silver_path)
    except (ValueError, OSError) as error:
        raise LineageValidationError("silver_invalid", silver_path, str(error)) from error
    if not elements:
        raise LineageValidationError("silver_empty", silver_path, selection.id)
    element_ids = [element.element_id for element in elements]
    if len(element_ids) != len(set(element_ids)):
        raise LineageValidationError("silver_element_ids_invalid", silver_path, "element IDs must be unique")
    orders = [element.order for element in elements]
    if sorted(orders) != list(range(len(elements))):
        raise LineageValidationError(
            "silver_order_invalid", silver_path, "element order must be unique and contiguous from zero"
        )
    document_ids = {element.document_id for element in elements}
    if document_ids != {bronze.source_sha256}:
        raise LineageValidationError(
            "silver_document_id_mismatch", silver_path, f"{sorted(document_ids)} != {bronze.source_sha256}"
        )
    invalid_stages = sorted({
        element.annotation.stage for element in elements if element.annotation.stage != "silver"
    })
    if invalid_stages:
        raise LineageValidationError("silver_stage_invalid", silver_path, str(invalid_stages))

    revision_counts = Counter(
        (element.annotation.revision, element.annotation.parent_revision) for element in elements
    )
    for revision, parent in revision_counts:
        expected_parent = None if revision == 1 else revision - 1
        if parent != expected_parent:
            raise LineageValidationError(
                "silver_parent_link_invalid",
                silver_path,
                f"revision {revision} declares parent {parent}, expected {expected_parent}",
            )

    source_references = _source_item_references(elements)
    _verify_source_ids(
        elements,
        source_references,
        document_id=bronze.source_sha256,
        liteparse_path=bronze_dir / "liteparse.json",
        silver_path=silver_path,
        root=root,
    )
    source_ids = [source_id for source_id, _ in source_references]
    catalog_counts = source_catalog_reference_counts(elements)
    revisions = [
        RevisionSummary(revision=revision, parent_revision=parent, element_count=count)
        for (revision, parent), count in sorted(revision_counts.items())
    ]
    return elements, SilverLineageStatus(
        path=_relative(silver_path, root),
        sha256=_sha256(silver_path),
        element_count=len(elements),
        document_id=bronze.source_sha256,
        revisions=revisions,
        snapshot_parent_links_valid=True,
        source_item_reference_count=len(source_ids),
        unique_source_item_count=len(set(source_ids)),
        catalog_reference_counts=catalog_counts,
    )


def _verify_source_ids(
    elements: list[DocumentElement],
    source_references: list[tuple[str, int]],
    *,
    document_id: str,
    liteparse_path: Path,
    silver_path: Path,
    root: Path,
) -> None:
    liteparse_references: list[tuple[str, int, re.Match[str]]] = []
    pymupdf_references: list[tuple[str, int, re.Match[str]]] = []
    for source_id, fragment_page in source_references:
        if match := _LITEPARSE_SOURCE_ITEM_ID.fullmatch(source_id):
            liteparse_references.append((source_id, fragment_page, match))
        elif match := _PYMUPDF_SOURCE_ITEM_ID.fullmatch(source_id):
            if int(match.group("page")) < 1 or int(match.group("duplicate")) < 1:
                raise LineageValidationError("source_item_id_invalid", silver_path, source_id)
            pymupdf_references.append((source_id, fragment_page, match))
        else:
            raise LineageValidationError("source_item_id_invalid", silver_path, source_id)

    if liteparse_references:
        item_counts = _read_liteparse_item_counts(liteparse_path)
        for source_id, fragment_page, match in liteparse_references:
            page = int(match.group("page"))
            if page != fragment_page:
                raise LineageValidationError(
                    "source_item_page_mismatch",
                    silver_path,
                    f"{source_id} is attached to page {fragment_page}",
                )
            index = int(match.group("item") or match.group("text"))
            if page not in item_counts or index >= item_counts[page]:
                raise LineageValidationError("source_item_id_unknown", silver_path, source_id)

    catalog_property_keys = {
        "table_span_suppression_v1",
        "table_structural_source_item_ids",
    }
    has_catalog_properties = any(
        prop.key in catalog_property_keys for element in elements for prop in element.structure.properties
    )
    catalog_path = _resolve_within(root, source_catalog_path(silver_path))
    requires_catalog = bool(pymupdf_references or has_catalog_properties)
    if not catalog_path.is_file():
        if not requires_catalog:
            return
        raise LineageValidationError(
            "source_catalog_missing",
            catalog_path,
            "document-scoped PyMuPDF provenance requires a source catalog",
        )
    try:
        catalog = read_source_catalog(catalog_path)
    except (OSError, ValueError) as error:
        raise LineageValidationError("source_catalog_invalid", catalog_path, str(error)) from error
    if catalog.document_id != document_id:
        raise LineageValidationError(
            "source_catalog_document_id_mismatch",
            catalog_path,
            f"{catalog.document_id} != {document_id}",
        )
    if not requires_catalog:
        return
    items_by_id = {item.source_item_id: item for item in catalog.items}
    for source_id, fragment_page, match in pymupdf_references:
        if match.group("document") != document_id:
            raise LineageValidationError("source_item_document_mismatch", silver_path, source_id)
        page = int(match.group("page"))
        if page != fragment_page:
            raise LineageValidationError(
                "source_item_page_mismatch",
                silver_path,
                f"{source_id} is attached to page {fragment_page}",
            )
        item = items_by_id.get(source_id)
        if item is None:
            raise LineageValidationError("source_item_id_unknown", silver_path, source_id)
        if item.document_id != document_id:
            raise LineageValidationError("source_item_document_mismatch", silver_path, source_id)
        if item.page_number != page:
            raise LineageValidationError("source_item_page_mismatch", silver_path, source_id)
        if item.kind != match.group("kind"):
            raise LineageValidationError("source_item_type_mismatch", silver_path, source_id)
    try:
        validate_source_catalog(elements, catalog)
    except ValueError as error:
        raise LineageValidationError("source_catalog_invalid", catalog_path, str(error)) from error


def _read_liteparse_item_counts(liteparse_path: Path) -> dict[int, int]:
    try:
        payload = json.loads(liteparse_path.read_text(encoding="utf-8"))
        pages = payload["pages"]
        if not isinstance(pages, list):
            raise TypeError("pages must be a list")
        page_entries: list[tuple[int, int]] = []
        for page_payload in pages:
            if not isinstance(page_payload, dict):
                raise TypeError("page entries must be objects")
            page = page_payload["page"]
            text_items = page_payload["text_items"]
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise TypeError("page must be a positive integer")
            if not isinstance(text_items, list):
                raise TypeError("text_items must be a list")
            page_entries.append((page, len(text_items)))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise LineageValidationError("liteparse_invalid", liteparse_path, str(error)) from error
    item_counts = dict(page_entries)
    if len(item_counts) != len(page_entries):
        raise LineageValidationError("liteparse_invalid", liteparse_path, "page numbers must be unique")
    return item_counts


def _classify_churn(
    selection_id: str,
    document_id: str,
    silver: list[DocumentElement],
    silver_path: Path,
    churn_path: Path,
    root: Path,
) -> ChurnArtifactStatus:
    if not churn_path.is_file():
        raise LineageValidationError("churn_missing", churn_path, selection_id)
    try:
        churn = _load_model(churn_path, ReferenceChurnReport)
    except LineageValidationError as error:
        return ChurnArtifactStatus(
            path=_relative(churn_path, root),
            status="invalid",
            reason_code="churn_artifact_invalid",
            reason=error.detail,
        )

    arithmetic_error = _churn_arithmetic_error(churn)
    identity_fields_partial = (churn.candidate_identity_rate is None) != (
        churn.candidate_content_identity_rate is None
    )
    if identity_fields_partial:
        arithmetic_error = (
            "candidate_identity_fields_partial",
            "candidate identity fields must both be present or both be null",
        )
    if arithmetic_error is not None or churn.document_id != document_id:
        reason_code, reason = arithmetic_error or (
            "document_id_mismatch",
            f"document_id {churn.document_id} != {document_id}",
        )
        return ChurnArtifactStatus(
            path=_relative(churn_path, root),
            status="invalid",
            reason_code=reason_code,
            reason=reason,
            claimed_previous_count=churn.previous_count,
            claimed_current_count=churn.current_count,
        )

    matches = _candidate_matches(churn, silver, selection_id, root)
    if churn.current_count != len(silver):
        return ChurnArtifactStatus(
            path=_relative(churn_path, root),
            status="stale",
            reason_code="current_count_mismatch",
            reason=(
                f"claimed current_count {churn.current_count} does not match current silver count "
                f"{len(silver)} ({_relative(silver_path, root)}); historical inputs are not identified"
            ),
            claimed_previous_count=churn.previous_count,
            claimed_current_count=churn.current_count,
            candidate_identity_matches=matches,
        )
    if churn.candidate_identity_rate is not None and not matches:
        return ChurnArtifactStatus(
            path=_relative(churn_path, root),
            status="invalid",
            reason_code="candidate_revision_not_found",
            reason="candidate identity fields match no candidate revision that exists",
            claimed_previous_count=churn.previous_count,
            claimed_current_count=churn.current_count,
        )
    return ChurnArtifactStatus(
        path=_relative(churn_path, root),
        status="current-snapshot-only",
        reason_code="historical_inputs_unavailable",
        reason="current aggregate and candidate identity fields are reproducible, but no previous silver artifact is identified or present",
        claimed_previous_count=churn.previous_count,
        claimed_current_count=churn.current_count,
        candidate_identity_matches=matches,
    )


def _candidate_matches(
    churn: ReferenceChurnReport,
    silver: list[DocumentElement],
    selection_id: str,
    root: Path,
) -> list[CandidateIdentityMatch]:
    if churn.candidate_identity_rate is None and churn.candidate_content_identity_rate is None:
        return []
    if churn.candidate_identity_rate is None or churn.candidate_content_identity_rate is None:
        return []
    matches: list[CandidateIdentityMatch] = []
    seen_hashes: set[str] = set()
    candidate_dirs = [
        path
        for path in root.glob("data/candidate*")
        if not _forbidden(str(path)) and "semantic9" not in str(path).casefold()
    ]
    for candidate_dir in sorted(candidate_dirs):
        candidate_dir = _resolve_within(root, candidate_dir)
        candidate_path = candidate_dir / f"{selection_id}.parquet"
        if not candidate_path.is_file():
            continue
        candidate_path = _resolve_within(root, candidate_path)
        try:
            candidate = read_document_elements(candidate_path)
            report = evaluate_reference_churn(silver, silver, candidate=candidate)
        except (OSError, ValueError) as error:
            raise LineageValidationError("candidate_invalid", candidate_path, str(error)) from error
        if not (
            math.isclose(
                cast(float, report.candidate_identity_rate),
                churn.candidate_identity_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                cast(float, report.candidate_content_identity_rate),
                churn.candidate_content_identity_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            continue
        sha256 = _sha256(candidate_path)
        if sha256 in seen_hashes:
            continue
        seen_hashes.add(sha256)
        matches.append(CandidateIdentityMatch(path=_relative(candidate_path, root), sha256=sha256))
    return matches


def _churn_arithmetic_error(
    churn: ReferenceChurnReport,
) -> tuple[ChurnReasonCode, str] | None:
    numeric_rates = [churn.churn_rate]
    numeric_rates.extend(
        rate
        for rate in (churn.candidate_identity_rate, churn.candidate_content_identity_rate)
        if rate is not None
    )
    if any(not math.isfinite(rate) or not 0 <= rate <= 1 for rate in numeric_rates):
        return (
            "churn_rates_invalid",
            "churn and candidate identity rates must be finite values from zero to one",
        )
    previous_common = churn.previous_count - churn.removed_count
    current_common = churn.current_count - churn.added_count
    if (
        min(
            churn.previous_count,
            churn.current_count,
            churn.added_count,
            churn.removed_count,
            churn.changed_count,
            churn.unchanged_count,
        )
        < 0
    ):
        return "churn_counts_negative", "churn counts must be non-negative"
    if previous_common != current_common:
        return (
            "churn_common_count_mismatch",
            "added/removed counts imply different common element counts",
        )
    if churn.changed_count + churn.unchanged_count != previous_common:
        return (
            "churn_unchanged_count_mismatch",
            "changed/unchanged counts do not equal the common element count",
        )
    denominator = max(churn.previous_count, churn.current_count)
    expected_rate = (
        (churn.added_count + churn.removed_count + churn.changed_count) / denominator if denominator else 0.0
    )
    if not math.isclose(churn.churn_rate, expected_rate):
        return "churn_rate_mismatch", f"churn_rate {churn.churn_rate} != {expected_rate}"
    return None


def _source_item_references(elements: list[DocumentElement]) -> list[tuple[str, int]]:
    references = [
        (source_id, fragment.page_number)
        for element in elements
        for fragment in element.fragments
        for source_id in fragment.source_item_ids
    ]
    references.extend(
        (source_id, fragment.page_number)
        for element in elements
        if element.structure.table is not None
        for cell in element.structure.table.cells
        for fragment in cell.fragments
        for source_id in fragment.source_item_ids
    )
    return references


def _verify_claimed_artifact(
    root: Path,
    identity: LineageArtifactIdentity,
    selection_id: str,
    role: str,
) -> Path:
    if _forbidden(identity.path):
        raise LineageValidationError("forbidden_scope", root / identity.path, identity.path)
    path = _resolve_within(root, Path(identity.path))
    if not path.is_file():
        raise LineageValidationError("lineage_artifact_missing", path, role)
    if path.name != f"{selection_id}.parquet":
        raise LineageValidationError(
            "lineage_artifact_identity_mismatch",
            path,
            f"expected {selection_id}.parquet for {role}",
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != identity.sha256:
        raise LineageValidationError(
            "lineage_artifact_hash_mismatch",
            path,
            f"{actual_sha256} != {identity.sha256}",
        )
    return path


def _read_lineage_elements(path: Path, role: str) -> list[DocumentElement]:
    try:
        elements = read_document_elements(path)
    except (OSError, ValueError) as error:
        raise LineageValidationError("lineage_artifact_invalid", path, f"{role}: {error}") from error
    if not elements:
        raise LineageValidationError("lineage_artifact_empty", path, role)
    return elements


def _load_model[T: SchemaModel](path: Path, model: type[T]) -> T:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LineageValidationError("artifact_invalid", path, str(error)) from error


def _resolve_within(root: Path, path: Path) -> Path:
    if ".." in path.parts:
        raise LineageValidationError("path_traversal_forbidden", path, "parent traversal is forbidden")
    unresolved = path if path.is_absolute() else root / path
    if not unresolved.is_relative_to(root):
        raise LineageValidationError("path_outside_repository", unresolved, str(root))
    symlink = _first_symlink_component(unresolved, start=root)
    if symlink is not None:
        raise LineageValidationError("symlink_forbidden", symlink, "audited paths cannot use symlinks")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(root):
        raise LineageValidationError("path_outside_repository", resolved, str(root))
    return resolved


def _first_symlink_component(path: Path, *, start: Path | None = None) -> Path | None:
    current = Path(path.anchor) if start is None else start
    parts = path.parts[1:] if start is None else path.relative_to(start).parts
    for part in parts:
        current /= part
        if current.is_symlink():
            return current
    return None


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root))


def _forbidden(value: str) -> bool:
    lowered = value.casefold()
    return "holdout" in lowered or "layoutlm" in lowered


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
