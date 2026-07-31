from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.metadata
import json
import math
import os
import re
import secrets
import stat
import unicodedata
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import JsonValue
from rapidfuzz.distance import Levenshtein
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import LeaveOneGroupOut

from app.pdf2md.benchmark_batch import project_code_fingerprint, runtime_versions
from app.pdf2md.evaluation import ElementAlignment, evaluate_document
from app.pdf2md.schema import SCHEMA_VERSION, DocumentElement, read_document_elements

BUNDLE_SCHEMA_VERSION = "blind-review-v7.0.0"
ALGORITHM_VERSION = "independent-semantic-materiality-v7.0.0"
NORMALIZATION_VERSION = "standalone-neutral-evidence-v7.0.0"
ATTACK_VERSION = "isolated-reconstruction-attack-v7.0.0"
DESIGN_VERSION = "isolated-reviewer-invariants-v7.0.0"
REVIEWER_ROOT_FILES = frozenset({"packets.json", "audit-sample-manifest.json", "manifest.json"})
PROTECTED_FILES = frozenset({"source-map.json", "seed.json", "full-ledger.json", "provenance.json"})

MATERIALITY_THRESHOLDS: dict[str, JsonValue] = {
    "presence_minimum_non_whitespace_characters": 2,
    "character_normalized_edit_distance": 0.01,
    "geometry_region_change": "different cell in a neutral 3_by_3 page grid",
    "qualifying_categories": [
        "element_presence",
        "declared_type",
        "semantic_role",
        "semantic_structure",
        "table_content_topology",
        "text_content",
        "reading_order_segmentation",
        "material_geometry",
    ],
    "category_union_rule": "any qualifying category is material; text equality cannot veto another category",
}
CONFIDENCE_LEVEL = 0.95
FORBIDDEN_REVIEWER_KEYS = frozenset({
    "candidate_index",
    "reference_index",
    "candidate_element_id",
    "reference_element_id",
    "presence_origin",
    "source_a",
    "source_b",
    "neutral_key",
    "document_id",
    "element_id",
    "source_item_ids",
    "annotation",
    "properties",
    "bbox",
    "document",
    "page_numbers",
    "sha256",
    "hash",
    "source_path",
    "source_name",
})


def build_blind_v7(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    *,
    root_dir: Path,
    seed: bytes | None = None,
) -> None:
    """Build a fresh, standalone v7 reviewer export and separated protected closure."""
    inputs = (candidate_dir, deterministic_candidate_dir, reference_dir, bronze_dir, root_dir)
    for path in inputs:
        validate_blind_path(path, must_exist=True)
    for path in (reviewer_dir, protected_dir):
        validate_blind_path(path, must_exist=False)
    _validate_disjoint_outputs(reviewer_dir, protected_dir)
    if reviewer_dir.exists() or protected_dir.exists():
        existing = reviewer_dir if reviewer_dir.exists() else protected_dir
        raise FileExistsError(f"output already exists: {existing}")

    secret_seed = seed if seed is not None else secrets.token_bytes(32)
    if len(secret_seed) < 32:
        raise ValueError("protected assignment seed must contain at least 256 bits")

    candidate_paths = sorted(candidate_dir.glob("*.parquet"))
    if not candidate_paths:
        raise ValueError("candidate directory contains no Parquet documents")
    names = {path.name for path in candidate_paths}
    if names != {path.name for path in deterministic_candidate_dir.glob("*.parquet")}:
        raise ValueError("determinism candidate file sets differ")
    if names != {path.name for path in reference_dir.glob("*.parquet")}:
        raise ValueError("candidate and reference file sets differ")

    input_hashes: dict[str, str] = {}
    raw: list[dict[str, object]] = []
    known_source_paths: list[Path] = []
    for candidate_path in candidate_paths:
        rerun_path = deterministic_candidate_dir / candidate_path.name
        reference_path = reference_dir / candidate_path.name
        known_source_paths.extend((candidate_path, rerun_path, reference_path))
        for path in (candidate_path, rerun_path, reference_path):
            validate_blind_path(path, must_exist=True)
            input_hashes[_relative_path(path, root_dir)] = _sha256(path.read_bytes())
        if candidate_path.read_bytes() != rerun_path.read_bytes():
            raise ValueError(f"candidate is not byte-equal to determinism rerun: {candidate_path.name}")
        raw.extend(
            _recompute_document_disagreements(
                candidate_path.stem,
                read_document_elements(candidate_path),
                read_document_elements(reference_path),
                bronze_dir,
                root_dir,
            )
        )

    material = [record for record in raw if record["material"] is True]
    nonmaterial = [record for record in raw if record["material"] is False]
    if len(raw) != len(material) + len(nonmaterial):
        raise RuntimeError("every recomputed record must be classified exactly once")
    if not material:
        raise ValueError("semantic thresholds selected no material disagreements")

    evidence_paths = sorted({path for record in raw for path in _record_evidence_paths(record)})
    for relative in evidence_paths:
        evidence_path = root_dir / relative
        validate_blind_path(evidence_path, must_exist=True)
        known_source_paths.append(evidence_path)
        input_hashes[relative] = _sha256(evidence_path.read_bytes())

    full_packets, mappings = _build_packets(material, secret_seed)
    sample_packets, sample_mappings = _reviewer_sample(full_packets, mappings, secret_seed)
    reviewer_packets, evidence_files = _standalone_reviewer_packets(sample_packets, bronze_dir, root_dir)
    adjudicability = _adjudicability_checks(reviewer_packets)
    design_acceptance = _deterministic_design_acceptance(reviewer_packets, sample_mappings, secret_seed)
    if adjudicability["result"] != "passed" or design_acceptance["result"] != "passed":
        raise ValueError(
            "v7 standalone evidence or adjudicability gate failed: "
            + json.dumps(
                {"adjudicability": adjudicability, "design_acceptance": design_acceptance},
                sort_keys=True,
            )
        )

    inventory = [
        {"path": name, "media_type": _media_type(name)}
        for name in sorted((*REVIEWER_ROOT_FILES, *evidence_files))
    ]
    audit = _public_audit_manifest(full_packets, reviewer_packets, mappings)
    manifest: dict[str, JsonValue] = {
        "artifact_id": "isolated-human-audit-v7",
        "bundle_schema": BUNDLE_SCHEMA_VERSION,
        "state": "pending_human_review",
        "review_performed": False,
        "publication_stage": "reviewer_material_only",
        "packet_count": len(reviewer_packets),
        "required_execution_environment": (
            "Copy this reviewer bundle alone to an isolated package and review it without access "
            "to any repository, extraction output, comparison corpus, protected files, or prior adjudication material. "
            "Content matching is inherently possible whenever those materials are accessible."
        ),
        "standalone_evidence": (
            "Every packet points only to bundle-local neutral page images and sanitized native "
            "text/geometry excerpts; no external lookup is required."
        ),
        "bundle_inventory": cast(JsonValue, inventory),
        "adjudication": {
            "paired_ab": "choose prefer_a, prefer_b, equivalent, or reconstruct",
            "presence": "choose keep, remove, or reconstruct",
        },
        "sampling_disclosure": (
            "A secret keyed 1-2% stratified sample balances presence and A-position assignments. "
            "The sample is underpowered; no inferential representativeness or leakage claim is made."
        ),
        "validation": {
            "standalone_evidence": "passed",
            "category_adjudicability": "passed",
            "reconstruction_metadata_attack": "pending_final_scan",
            "review_state": "pending_human_review",
        },
    }
    reviewer_files = {
        "packets.json": _pretty_bytes(reviewer_packets),
        "audit-sample-manifest.json": _pretty_bytes(audit),
        "manifest.json": _pretty_bytes(manifest),
        **evidence_files,
    }
    reconstruction = _reconstruction_attack(
        reviewer_files,
        known_source_paths,
        forbidden_names={path.stem for path in candidate_paths},
    )
    if reconstruction["result"] != "passed":
        raise ValueError(f"v7 reconstruction attack failed: {json.dumps(reconstruction, sort_keys=True)}")
    validation = cast(dict[str, JsonValue], manifest["validation"])
    validation["reconstruction_metadata_attack"] = "passed"
    reviewer_files["manifest.json"] = _pretty_bytes(manifest)
    reconstruction = _reconstruction_attack(
        reviewer_files,
        known_source_paths,
        forbidden_names={path.stem for path in candidate_paths},
    )
    if reconstruction["result"] != "passed":
        raise ValueError("final reviewer export introduced lookup metadata")

    reviewer_hashes = {name: _sha256(content) for name, content in sorted(reviewer_files.items())}
    source_map = {
        "artifact_id": "semantic13-protected-source-map-v7",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "reviewer_sha256": reviewer_hashes,
        "sample_mappings": sample_mappings,
        "full_material_mappings": mappings,
    }
    seed_record = {
        "artifact_id": "semantic13-protected-seed-v7",
        "encoding": "hex",
        "bits": len(secret_seed) * 8,
        "value": secret_seed.hex(),
        "hmac": "HMAC-SHA256",
        "domains": ["packet-id", "paired-option-position", "sample-order"],
    }
    full_ledger = {
        "artifact_id": "semantic13-protected-material-exclusion-ledger-v7",
        "policy": "every independently recomputed aligned and one-sided record",
        "thresholds": MATERIALITY_THRESHOLDS,
        "counts": {"raw": len(raw), "material": len(material), "nonmaterial": len(nonmaterial)},
        "records": [
            _ledger_record(record)
            for record in sorted(raw, key=lambda value: cast(str, value["neutral_key"]))
        ],
    }
    provenance = _protected_provenance(
        root_dir=root_dir,
        input_directories={
            "candidate": candidate_dir,
            "determinism": deterministic_candidate_dir,
            "reference": reference_dir,
            "bronze": bronze_dir,
        },
        input_hashes=input_hashes,
        full_packets=full_packets,
        reviewer_packets=reviewer_packets,
        mappings=mappings,
        raw=raw,
        reconstruction=reconstruction,
        adjudicability=adjudicability,
        design_acceptance=design_acceptance,
        reviewer_hashes=reviewer_hashes,
    )
    _stage_and_publish_best_effort(
        reviewer_dir,
        protected_dir,
        reviewer_files,
        {
            "source-map.json": _pretty_bytes(source_map),
            "seed.json": _pretty_bytes(seed_record),
            "full-ledger.json": _pretty_bytes(full_ledger),
            "provenance.json": _pretty_bytes(provenance),
        },
    )


def read_protected_seed(path: Path) -> bytes:
    validate_blind_path(path, must_exist=True)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not isinstance(value.get("value"), str):
        raise TypeError("protected seed file requires a hexadecimal value")
    seed = bytes.fromhex(value["value"])
    if len(seed) < 32:
        raise ValueError("protected seed must contain at least 256 bits")
    return seed


def canonical_review_option(element: DocumentElement | None) -> dict[str, JsonValue] | None:
    return _canonical_option(element)


def material_categories(
    candidate: DocumentElement | None, reference: DocumentElement | None, *, order_issue: bool = False
) -> list[str]:
    """Return every independently qualifying semantic materiality category."""
    return _material_categories(candidate, reference, order_issue)


def validate_blind_path(path: Path, *, must_exist: bool) -> None:
    resolved = path.resolve(strict=must_exist)
    for inspected in (path.absolute(), resolved):
        lowered = [part.lower() for part in inspected.parts]
        if any("holdout" in part for part in lowered):
            raise ValueError(f"holdout path is forbidden: {path}")
        if any("layoutlm" in part for part in lowered):
            raise ValueError(f"LayoutLM path is forbidden: {path}")


def _recompute_document_disagreements(
    document: str,
    candidates: list[DocumentElement],
    references: list[DocumentElement],
    bronze_dir: Path,
    root_dir: Path,
) -> list[dict[str, object]]:
    report = evaluate_document(candidates, references)
    inversions = _inversion_participants(report.alignments)
    values: list[dict[str, object]] = []
    for alignment in report.alignments:
        values.append(
            _with_neighbor_context(
                _disagreement(
                    document,
                    candidates[alignment.candidate_index],
                    references[alignment.reference_index],
                    alignment.candidate_index,
                    alignment.reference_index,
                    alignment.score,
                    (alignment.candidate_index, alignment.reference_index) in inversions,
                    bronze_dir,
                    root_dir,
                ),
                candidates,
                references,
                alignment.candidate_index,
                alignment.reference_index,
            )
        )
    for index in report.unmatched_candidate_indices:
        values.append(
            _with_neighbor_context(
                _disagreement(
                    document, candidates[index], None, index, None, 0.0, False, bronze_dir, root_dir
                ),
                candidates,
                references,
                index,
                None,
            )
        )
    for index in report.unmatched_reference_indices:
        values.append(
            _with_neighbor_context(
                _disagreement(
                    document, None, references[index], None, index, 0.0, False, bronze_dir, root_dir
                ),
                candidates,
                references,
                None,
                index,
            )
        )
    return values


def _with_neighbor_context(
    record: dict[str, object],
    candidates: list[DocumentElement],
    references: list[DocumentElement],
    candidate_index: int | None,
    reference_index: int | None,
) -> dict[str, object]:
    record["candidate_context"] = _local_sequence(candidates, candidate_index)
    record["reference_context"] = _local_sequence(references, reference_index)
    return record


def _local_sequence(elements: list[DocumentElement], index: int | None) -> list[dict[str, JsonValue]] | None:
    if index is None:
        return None
    sequence: list[dict[str, JsonValue]] = []
    for offset in range(-3, 4):
        position = index + offset
        if position < 0 or position >= len(elements):
            continue
        element = elements[position]
        sequence.append({
            "relative_position": "focus"
            if offset == 0
            else f"before_{-offset}"
            if offset < 0
            else f"after_{offset}",
            "text": _text(element.content)[:1000],
            "declared_type": _review_element_class(element),
            "role": _role_payload(element),
            "relative_geometry": _relative_sequence_geometry(element, elements[index]),
        })
    return sequence


def _relative_sequence_geometry(element: DocumentElement, focus: DocumentElement) -> JsonValue:
    focus_pages = sorted(_fragment_pages(focus))
    first_focus_page = focus_pages[0] if focus_pages else 0
    return cast(
        JsonValue,
        [
            {
                "page_offset": cast(int, region["page"]) - first_focus_page,
                "horizontal": region["horizontal"],
                "vertical": region["vertical"],
            }
            for region in cast(list[dict[str, JsonValue]], _region_payload(element))
        ],
    )


def _disagreement(
    document: str,
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
    candidate_index: int | None,
    reference_index: int | None,
    alignment_score: float,
    order_issue: bool,
    bronze_dir: Path,
    root_dir: Path,
) -> dict[str, object]:
    if candidate is None and reference is None:
        raise ValueError("disagreement requires at least one source element")
    categories = _material_categories(candidate, reference, order_issue)
    reason = _exclusion_reason(candidate, reference, categories)
    element = candidate or reference
    assert element is not None
    pages = sorted({
        fragment.page_number for item in (candidate, reference) if item for fragment in item.fragments
    })
    neutral_key = _neutral_key(document, candidate_index, reference_index, candidate, reference)
    bronze_relative = _relative_path(bronze_dir, root_dir)
    evidence = {
        "bronze_manifest": f"{bronze_relative}/{document}/manifest.json",
        "bronze_native_json": f"{bronze_relative}/{document}/liteparse.json",
        "bronze_native_text": f"{bronze_relative}/{document}/liteparse.txt",
        "page_images": [f"{bronze_relative}/{document}/pages/page-{page:04d}.png" for page in pages],
    }
    return {
        "neutral_key": neutral_key,
        "document": document,
        "document_id": element.document_id,
        "mode": "presence" if candidate is None or reference is None else "paired_ab",
        "candidate_index": candidate_index,
        "reference_index": reference_index,
        "candidate_element_id": candidate.element_id if candidate else None,
        "reference_element_id": reference.element_id if reference else None,
        "candidate_option": _canonical_option(candidate),
        "reference_option": _canonical_option(reference),
        "page_numbers": pages,
        "evidence": evidence,
        "categories": categories,
        "severity": _severity(categories),
        "alignment_score": alignment_score,
        "material": reason is None,
        "exclusion_reason": reason,
        "presence_origin": "candidate" if reference is None else "reference" if candidate is None else None,
    }


def _canonical_option(element: DocumentElement | None) -> dict[str, JsonValue] | None:
    if element is None:
        return None
    paragraph = element.structure.paragraph
    role: dict[str, JsonValue] = {
        "paragraph_role": _normalized_label(paragraph.role) if paragraph else None,
        "heading_level": paragraph.heading_level if paragraph else None,
        "list_depth": paragraph.list_depth if paragraph else None,
        "footnote": element.structure.footnote is not None,
    }
    structure: dict[str, JsonValue] = {
        "line_patterns": cast(JsonValue, _visible_structure(element.content)),
        "table": _table_summary(element),
        "fragment_count": len(element.fragments),
        "page_count": len(_fragment_pages(element)),
    }
    return cast(
        dict[str, JsonValue],
        {
            "kind": "document_element",
            "text": _text(element.content),
            "declared_type": _review_element_class(element),
            "role": role,
            "geometry": _region_payload(element),
            "structure": structure,
        },
    )


def _table_summary(element: DocumentElement) -> dict[str, JsonValue] | None:
    table = element.structure.table
    if table is None:
        return None
    roles = Counter(cell.role for cell in table.cells)
    spans = Counter(f"{cell.rowspan}x{cell.colspan}" for cell in table.cells)
    return {
        "row_count": table.row_count,
        "column_count": table.column_count,
        "header_row_count": table.header_row_count,
        "representation": table.representation,
        "cell_role_counts": dict(sorted(roles.items())),
        "span_counts": dict(sorted(spans.items())),
    }


def _review_element_class(element: DocumentElement) -> str:
    value = _normalized_label(element.element_type)
    aliases = {
        "image": "figure",
        "picture": "figure",
        "header": "running_matter",
        "footer": "running_matter",
        "text": "paragraph",
        "body": "paragraph",
    }
    return aliases.get(value, value)


def _normalized_label(value: str) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _page_region(fragment: object) -> dict[str, JsonValue]:
    bbox = getattr(fragment, "bbox")
    width = getattr(fragment, "page_width") or max(float(getattr(bbox, "x1")), 1.0)
    height = getattr(fragment, "page_height") or max(float(getattr(bbox, "y1")), 1.0)
    center_x = (float(getattr(bbox, "x0")) + float(getattr(bbox, "x1"))) / (2 * width)
    center_y = (float(getattr(bbox, "y0")) + float(getattr(bbox, "y1"))) / (2 * height)
    horizontal = ("left", "center", "right")[min(2, max(0, int(center_x * 3)))]
    vertical = ("top", "middle", "bottom")[min(2, max(0, int(center_y * 3)))]
    return {"page": getattr(fragment, "page_number"), "horizontal": horizontal, "vertical": vertical}


def _material_categories(
    candidate: DocumentElement | None, reference: DocumentElement | None, order_issue: bool
) -> list[str]:
    if candidate is None or reference is None:
        element = candidate or reference
        if element is None:
            return []
        if len(_text(element.content).strip()) >= 2 or element.element_type in {"table", "figure"}:
            return ["element_presence"]
        return []
    categories: list[str] = []
    if _review_element_class(candidate) != _review_element_class(reference):
        categories.append("declared_type")
    if _role_payload(candidate) != _role_payload(reference):
        categories.append("semantic_role")
    if _visible_structure(candidate.content) != _visible_structure(reference.content):
        categories.append("semantic_structure")
    if _table_summary(candidate) != _table_summary(reference):
        categories.append("table_content_topology")
    if _text_material(candidate.content, reference.content):
        categories.append("text_content")
    if (
        order_issue
        or candidate.order != reference.order
        or _fragment_pages(candidate) != _fragment_pages(reference)
    ):
        categories.append("reading_order_segmentation")
    if _region_payload(candidate) != _region_payload(reference):
        categories.append("material_geometry")
    return sorted(set(categories))


def _role_payload(element: DocumentElement) -> JsonValue:
    paragraph = element.structure.paragraph
    return cast(
        JsonValue,
        {
            "paragraph_role": _normalized_label(paragraph.role) if paragraph else None,
            "heading_level": paragraph.heading_level if paragraph else None,
            "list_depth": paragraph.list_depth if paragraph else None,
            "footnote": element.structure.footnote is not None,
        },
    )


def _visible_structure(value: str) -> list[str]:
    signatures: list[str] = []
    for line in _text(value).splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            signatures.append("heading_marker")
        elif stripped.startswith(("- ", "* ", "+ ")):
            signatures.append("bullet_marker")
        elif stripped[:1].isdigit() and ". " in stripped[:6]:
            signatures.append("numbered_marker")
        elif stripped.startswith("|") and stripped.endswith("|"):
            signatures.append("table_row_marker")
        else:
            signatures.append("plain_line")
    return signatures


def _region_payload(element: DocumentElement) -> JsonValue:
    return cast(JsonValue, [_page_region(fragment) for fragment in element.fragments])


def _text_material(left: str, right: str) -> bool:
    left_text, right_text = _text(left), _text(right)
    if left_text == right_text:
        return False
    distance = Levenshtein.distance(left_text, right_text)
    return distance / max(len(left_text), len(right_text), 1) >= 0.01


def _exclusion_reason(
    candidate: DocumentElement | None, reference: DocumentElement | None, categories: list[str]
) -> str | None:
    if categories:
        return None
    if candidate is None or reference is None:
        return "nonmaterial_presence_below_minimum_content"
    return "below_all_semantic_thresholds"


def _severity(categories: list[str]) -> str:
    values = set(categories)
    if values & {"element_presence", "table_content_topology"}:
        return "critical"
    if values & {"declared_type", "semantic_role", "semantic_structure", "reading_order_segmentation"}:
        return "high"
    if "text_content" in values:
        return "medium"
    return "low"


def _build_packets(
    material: list[dict[str, object]], seed: bytes
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    packets: list[dict[str, JsonValue]] = []
    mappings: list[dict[str, JsonValue]] = []
    for record in material:
        neutral_key = cast(str, record["neutral_key"])
        packet_id = _secret_public_id(seed, neutral_key)
        common: dict[str, JsonValue] = {
            "packet_id": packet_id,
            "document": cast(str, record["document"]),
            "mode": cast(str, record["mode"]),
            "page_numbers": cast(JsonValue, record["page_numbers"]),
            "evidence": cast(JsonValue, record["evidence"]),
            "categories": cast(JsonValue, record["categories"]),
            "severity": cast(str, record["severity"]),
            "review_state": "pending_human_review",
        }
        if record["mode"] == "paired_ab":
            candidate_first = _secret_bit(seed, "paired-option-position", neutral_key) == 0
            candidate_option = cast(JsonValue, record["candidate_option"])
            reference_option = cast(JsonValue, record["reference_option"])
            option_a, option_b = (
                (candidate_option, reference_option)
                if candidate_first
                else (reference_option, candidate_option)
            )
            packet = {
                **common,
                "prompt": "Which option should define the adjudicated element?",
                "option_a": option_a,
                "option_b": option_b,
                "allowed_decisions": ["prefer_a", "prefer_b", "equivalent", "reconstruct"],
            }
            if "reading_order_segmentation" in cast(list[str], record["categories"]):
                candidate_context = cast(JsonValue, record["candidate_context"])
                reference_context = cast(JsonValue, record["reference_context"])
                packet["order_adjudication"] = {
                    "prompt": "Which complete local sequence has the correct reading order and segmentation?",
                    "sequence_a": candidate_context if candidate_first else reference_context,
                    "sequence_b": reference_context if candidate_first else candidate_context,
                    "context_window": "focus plus up to three neutral elements before and after",
                }
            mapping = {
                "packet_id": packet_id,
                "neutral_key": neutral_key,
                "mode": "paired_ab",
                "source_a": "candidate" if candidate_first else "reference",
                "source_b": "reference" if candidate_first else "candidate",
            }
        else:
            observed = record["candidate_option"] or record["reference_option"]
            packet = {
                **common,
                "prompt": "How should this observed element be handled?",
                "observed_element": cast(JsonValue, observed),
                "allowed_decisions": ["keep", "remove", "reconstruct"],
            }
            mapping = {
                "packet_id": packet_id,
                "neutral_key": neutral_key,
                "mode": "presence",
                "presence_origin": cast(str, record["presence_origin"]),
            }
        mappings.append({
            **mapping,
            "document": cast(str, record["document"]),
            "candidate_index": cast(JsonValue, record["candidate_index"]),
            "reference_index": cast(JsonValue, record["reference_index"]),
            "candidate_element_id": cast(JsonValue, record["candidate_element_id"]),
            "reference_element_id": cast(JsonValue, record["reference_element_id"]),
        })
        packets.append(cast(dict[str, JsonValue], packet))
    packets.sort(key=lambda value: cast(str, value["packet_id"]))
    mappings.sort(key=lambda value: cast(str, value["packet_id"]))
    return packets, mappings


def _reviewer_sample(
    packets: list[dict[str, JsonValue]], mappings: list[dict[str, JsonValue]], seed: bytes
) -> tuple[list[dict[str, JsonValue]], list[dict[str, JsonValue]]]:
    target = max(math.ceil(len(packets) * 0.01), math.floor(len(packets) * 0.02))
    if not (0.01 <= target / len(packets) <= 0.02):
        raise ValueError("population cannot support an integer 1-2% reviewer sample")
    if target < 4:
        raise ValueError("1-2% sample cannot support paired plus balanced presence auditing")
    mapping_by_id = {cast(str, mapping["packet_id"]): mapping for mapping in mappings}

    def ordered(values: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
        eligible = [packet for packet in values if _internally_adjudicable_for_sampling(packet)]
        return sorted(
            eligible,
            key=lambda packet: _secret_token(seed, "sample-order", cast(str, packet["packet_id"]), length=64),
        )

    candidate_presence = ordered([
        packet
        for packet in packets
        if packet["mode"] == "presence"
        and mapping_by_id[cast(str, packet["packet_id"])]["presence_origin"] == "candidate"
    ])
    reference_presence = ordered([
        packet
        for packet in packets
        if packet["mode"] == "presence"
        and mapping_by_id[cast(str, packet["packet_id"])]["presence_origin"] == "reference"
    ])
    paired_candidate_a = ordered([
        packet
        for packet in packets
        if packet["mode"] == "paired_ab"
        and mapping_by_id[cast(str, packet["packet_id"])]["source_a"] == "candidate"
    ])
    paired_reference_a = ordered([
        packet
        for packet in packets
        if packet["mode"] == "paired_ab"
        and mapping_by_id[cast(str, packet["packet_id"])]["source_a"] == "reference"
    ])
    presence_each = min(len(candidate_presence), len(reference_presence), 1)
    paired_count = target - 2 * presence_each
    if presence_each == 0 or paired_count < 2 or paired_count % 2:
        raise ValueError("sample cannot include paired records with exactly balanced presence origins")
    paired_each = paired_count // 2
    if len(paired_candidate_a) < paired_each or len(paired_reference_a) < paired_each:
        raise ValueError("sample cannot balance secret paired HMAC assignments")
    strata = {
        "presence_candidate": (candidate_presence, presence_each),
        "presence_reference": (reference_presence, presence_each),
        "paired_candidate_in_a": (paired_candidate_a, paired_each),
        "paired_reference_in_a": (paired_reference_a, paired_each),
    }
    selected = [packet for population, quota in strata.values() for packet in population[:quota]]
    stratum_by_id = {
        cast(str, packet["packet_id"]): (name, len(population), quota)
        for name, (population, quota) in strata.items()
        for packet in population
    }
    for mapping in mappings:
        values = stratum_by_id.get(cast(str, mapping["packet_id"]))
        if values is None:
            mapping["sampling_eligibility"] = "excluded_unadjudicable"
            mapping["inclusion_probability"] = 0.0
            continue
        name, population_count, quota = values
        probability = quota / population_count
        mapping["sampling_eligibility"] = "eligible"
        mapping["inclusion_stratum"] = name
        mapping["inclusion_probability"] = probability
        mapping["sampling_weight"] = 1.0 / probability
    selected.sort(key=lambda packet: cast(str, packet["packet_id"]))
    selected_ids = {cast(str, packet["packet_id"]) for packet in selected}
    sample_mappings = [mapping for mapping in mappings if mapping["packet_id"] in selected_ids]
    origins = Counter(
        cast(str, mapping["presence_origin"]) for mapping in sample_mappings if mapping["mode"] == "presence"
    )
    if origins["candidate"] != origins["reference"] or not origins["candidate"]:
        raise RuntimeError("sampled presence origins are not exactly balanced")
    return selected, sample_mappings


def _standalone_reviewer_packets(
    packets: list[dict[str, JsonValue]], bronze_dir: Path, root_dir: Path
) -> tuple[list[dict[str, JsonValue]], dict[str, bytes]]:
    reviewer_packets: list[dict[str, JsonValue]] = []
    evidence_files: dict[str, bytes] = {}
    for packet in packets:
        public = cast(dict[str, JsonValue], json.loads(json.dumps(packet)))
        document = cast(str, public.pop("document"))
        source_pages = cast(list[int], public.pop("page_numbers"))
        page_labels = {page: f"page_{position}" for position, page in enumerate(source_pages, start=1)}
        packet_id = cast(str, public["packet_id"])
        evidence_root = f"evidence/{packet_id}"
        image_paths: list[str] = []
        source_evidence = cast(dict[str, JsonValue], public["evidence"])
        source_images = cast(list[str], source_evidence["page_images"])
        if len(source_images) != len(source_pages):
            raise ValueError("sampled packet image/page evidence count mismatch")
        for position, relative in enumerate(source_images, start=1):
            destination = f"{evidence_root}/page-{position:02d}.png"
            source = (root_dir / relative).resolve(strict=True)
            _require_within(source, bronze_dir, "reviewer page evidence")
            evidence_files[destination] = source.read_bytes()
            image_paths.append(destination)
        excerpt_path = f"{evidence_root}/native-evidence.json"
        evidence_files[excerpt_path] = _pretty_bytes(
            _sanitized_native_excerpt(bronze_dir / document / "liteparse.json", source_pages)
        )
        public["evidence"] = cast(
            JsonValue,
            {
                "page_images": image_paths,
                "native_text_geometry_excerpt": excerpt_path,
            },
        )
        _localize_page_references(public, page_labels)
        reviewer_packets.append(public)
    return reviewer_packets, evidence_files


def _sanitized_native_excerpt(path: Path, source_pages: list[int]) -> dict[str, JsonValue]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        raise TypeError("native evidence requires a pages array")
    by_page = {
        page.get("page"): page
        for page in cast(list[object], payload["pages"])
        if isinstance(page, dict) and isinstance(page.get("page"), int)
    }
    result: list[dict[str, JsonValue]] = []
    for position, page_number in enumerate(source_pages, start=1):
        page = by_page.get(page_number)
        if page is None:
            raise ValueError(f"native evidence does not contain sampled page {page_number}")
        width = float(page.get("width", 0.0))
        height = float(page.get("height", 0.0))
        if width <= 0 or height <= 0:
            raise ValueError("native evidence page dimensions must be positive")
        items: list[dict[str, JsonValue]] = []
        raw_items = page.get("text_items")
        if not isinstance(raw_items, list):
            raise TypeError("native evidence page requires text_items")
        for item in raw_items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            items.append({
                "text": _text(item["text"]),
                "left": round(float(item.get("x", 0.0)) / width, 6),
                "top": round(float(item.get("y", 0.0)) / height, 6),
                "width": round(float(item.get("width", 0.0)) / width, 6),
                "height": round(float(item.get("height", 0.0)) / height, 6),
            })
        result.append({
            "local_page": f"page_{position}",
            "text": _text(cast(str, page.get("text", ""))),
            "items": cast(JsonValue, items),
        })
    return {"schema": "sanitized-native-text-geometry-v1", "pages": cast(JsonValue, result)}


def _localize_page_references(value: object, page_labels: dict[int, str]) -> None:
    if isinstance(value, dict):
        if "page" in value and isinstance(value["page"], int):
            page = value["page"]
            if page not in page_labels:
                raise ValueError("option geometry references evidence outside sampled pages")
            value["page"] = page_labels[page]
        for child in value.values():
            _localize_page_references(child, page_labels)
    elif isinstance(value, list):
        for child in value:
            _localize_page_references(child, page_labels)


def _require_within(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes its supplied directory") from error


def _internally_adjudicable_for_sampling(packet: dict[str, JsonValue]) -> bool:
    categories = cast(list[str], packet["categories"])
    if "reading_order_segmentation" not in categories:
        return True
    order = packet.get("order_adjudication")
    if not isinstance(order, dict):
        return False
    left = order.get("sequence_a")
    right = order.get("sequence_b")
    return isinstance(left, list) and isinstance(right, list) and left != right


def _adjudicability_checks(packets: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    checks: list[dict[str, JsonValue]] = []
    for packet in packets:
        options = [
            cast(dict[str, JsonValue], value)
            for key in ("option_a", "option_b", "observed_element")
            if isinstance((value := packet.get(key)), dict)
        ]
        for category in cast(list[str], packet["categories"]):
            passed = _category_is_adjudicable(category, packet, options)
            checks.append({"packet_id": packet["packet_id"], "category": category, "passed": passed})
    if not checks:
        raise ValueError("sample has no category-specific adjudicability checks")
    return cast(
        dict[str, JsonValue],
        {
            "method": "category-specific deterministic checks for every declared category on every sampled packet",
            "packet_count": len(packets),
            "check_count": len(checks),
            "checks": checks,
            "result": "passed" if all(check["passed"] is True for check in checks) else "failed",
        },
    )


def _category_is_adjudicable(
    category: str,
    packet: dict[str, JsonValue],
    options: list[dict[str, JsonValue]],
) -> bool:
    evidence = packet.get("evidence")
    if not isinstance(evidence, dict) or not _evidence_paths(packet):
        return False
    if category == "element_presence":
        return packet["mode"] == "presence" and len(options) == 1
    if category == "declared_type":
        return len(options) == 2 and options[0].get("declared_type") != options[1].get("declared_type")
    if category == "semantic_role":
        return len(options) == 2 and options[0].get("role") != options[1].get("role")
    if category == "semantic_structure":
        return len(options) == 2 and options[0].get("structure") != options[1].get("structure")
    if category == "table_content_topology":
        return len(options) == 2 and any(
            isinstance(option.get("structure"), dict)
            and cast(dict[str, JsonValue], option["structure"]).get("table") is not None
            for option in options
        )
    if category == "text_content":
        return len(options) == 2 and options[0].get("text") != options[1].get("text")
    if category == "reading_order_segmentation":
        order = packet.get("order_adjudication")
        if packet["mode"] != "paired_ab" or not isinstance(order, dict):
            return False
        left, right = order.get("sequence_a"), order.get("sequence_b")
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and left != right
            and all(
                isinstance(item, dict)
                and "relative_position" in item
                and "text" in item
                and "declared_type" in item
                for sequence in (left, right)
                for item in sequence
            )
            and "index" not in json.dumps(order).casefold()
        )
    if category == "material_geometry":
        return len(options) == 2 and options[0].get("geometry") != options[1].get("geometry")
    return False


def _deterministic_design_acceptance(
    packets: list[dict[str, JsonValue]], mappings: list[dict[str, JsonValue]], seed: bytes
) -> dict[str, JsonValue]:
    forbidden_hits = sorted(_recursive_forbidden_keys(packets))
    presence_origins = Counter(
        cast(str, mapping["presence_origin"]) for mapping in mappings if mapping["mode"] == "presence"
    )
    paired_a = Counter(
        cast(str, mapping["source_a"]) for mapping in mappings if mapping["mode"] == "paired_ab"
    )
    domain_keys = {
        domain: hmac.new(seed, f"blind-v7-key\0{domain}".encode(), hashlib.sha256).hexdigest()
        for domain in ("packet-id", "paired-option-position", "sample-order")
    }
    checks = {
        "reviewer_origin_fields_absent": not forbidden_hits,
        "hmac_domain_keys_distinct": len(set(domain_keys.values())) == len(domain_keys),
        "sampled_presence_origins_balanced": presence_origins["candidate"]
        == presence_origins["reference"]
        > 0,
        "sampled_paired_a_positions_balanced": paired_a["candidate"] == paired_a["reference"] > 0,
    }
    return cast(
        dict[str, JsonValue],
        {
            "method": "deterministic design-invariant acceptance; descriptive attacks are excluded",
            "checks": checks,
            "forbidden_reviewer_keys_found": forbidden_hits,
            "hmac_independence": "distinct HMAC-SHA256 domain-derived keys; secret key digests are not exported",
            "result": "passed" if all(checks.values()) else "failed",
        },
    )


def _recursive_forbidden_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value} & set(FORBIDDEN_REVIEWER_KEYS)
        return keys | {hit for child in value.values() for hit in _recursive_forbidden_keys(child)}
    if isinstance(value, list):
        return {hit for child in value for hit in _recursive_forbidden_keys(child)}
    return set()


def _run_descriptive_attacks(
    packets: list[dict[str, JsonValue]], mappings: list[dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    mapping_by_id = {cast(str, mapping["packet_id"]): mapping for mapping in mappings}
    expected = np.asarray([_mapping_label(packet, mapping_by_id) for packet in packets], dtype=bool)
    groups = np.asarray([cast(str, packet["document"]) for packet in packets])
    if len(set(groups.tolist())) < 2 or len(set(expected.tolist())) < 2:
        raise ValueError("descriptive attacks require multiple documents and both protected labels")
    feature_sets = {
        "public_id_nibbles": [_public_id_nibble_features(packet) for packet in packets],
        "public_hash_nibbles": [_public_hash_nibble_features(packet) for packet in packets],
        "normalized_structure_content": [_structural_content_features(packet) for packet in packets],
    }
    model_factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=6106, n_jobs=1
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=6106, n_jobs=1
        ),
    }
    results: dict[str, JsonValue] = {}
    for feature_name, features in feature_sets.items():
        matrix = cast(np.ndarray, DictVectorizer(sparse=False).fit_transform(features))
        for model_name, factory in model_factories.items():
            predictions = _lodo_predictions(matrix, expected, groups, factory())
            results[f"{feature_name}_{model_name}"] = cast(
                JsonValue, _attack_metrics(packets, expected, predictions)
            )
    positives = int(expected.sum())
    return cast(
        dict[str, JsonValue],
        {
            "method": "descriptive LODO RF/ExtraTrees over public ID/hash and normalized packet features",
            "population": {
                "packets": len(packets),
                "paired": sum(packet["mode"] == "paired_ab" for packet in packets),
                "presence": sum(packet["mode"] == "presence" for packet in packets),
                "mapping_label_true": positives,
                "mapping_label_false": len(expected) - positives,
                "documents": len(set(groups.tolist())),
                "excluded": 0,
            },
            "models": results,
            "confidence_intervals": "two-sided Clopper-Pearson exact 95% binomial intervals",
            "interpretation": "descriptive diagnostic only; no threshold, pass/fail, or leakage acceptance",
            "statistical_power": (
                "insufficient for inferential leakage acceptance"
                if len(packets) <= 6
                else "larger full-population diagnostic, still not an acceptance test"
            ),
            "imbalance_disclosure": "raw mode and mapping-label counts are reported above without balancing",
        },
    )


def _lodo_predictions(
    matrix: np.ndarray,
    expected: np.ndarray,
    groups: np.ndarray,
    model: RandomForestClassifier | ExtraTreesClassifier,
) -> np.ndarray:
    predictions = np.zeros(len(expected), dtype=bool)
    for train, test in LeaveOneGroupOut().split(matrix, expected, groups):
        train_labels = expected[train]
        if len(set(train_labels.tolist())) == 1:
            predictions[test] = train_labels[0]
        else:
            model.fit(matrix[train], train_labels)
            predictions[test] = cast(np.ndarray, model.predict(matrix[test]))
    return predictions


def _attack_metrics(
    packets: list[dict[str, JsonValue]], expected: np.ndarray, predicted: np.ndarray
) -> dict[str, JsonValue]:
    paired = np.asarray([packet["mode"] == "paired_ab" for packet in packets])
    presence = ~paired
    if not paired.any() or not presence.any():
        raise ValueError("attacks require paired and presence records")

    def score(mask: np.ndarray) -> dict[str, JsonValue]:
        successes = int(np.sum(expected[mask] == predicted[mask]))
        trials = int(mask.sum())
        lower, upper = _clopper_pearson(successes, trials)
        return {
            "successes": successes,
            "trials": trials,
            "estimate": successes / trials,
            "exact_95_ci": [lower, upper],
        }

    return {
        "raw_mapping_accuracy": score(np.ones(len(packets), dtype=bool)),
        "raw_paired_mapping_accuracy": score(paired),
        "raw_presence_origin_accuracy": score(presence),
    }


def _clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("exact interval requires 0 <= successes <= positive trials")
    alpha = 1.0 - CONFIDENCE_LEVEL
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def _mapping_label(packet: dict[str, JsonValue], mapping_by_id: dict[str, dict[str, JsonValue]]) -> bool:
    mapping = mapping_by_id[cast(str, packet["packet_id"])]
    if packet["mode"] == "paired_ab":
        return mapping["source_a"] == "candidate"
    return mapping["presence_origin"] == "candidate"


def _public_id_nibble_features(packet: dict[str, JsonValue]) -> dict[str, float]:
    packet_id = cast(str, packet["packet_id"])
    return {f"character_{index}": float(ord(value)) for index, value in enumerate(packet_id)}


def _public_hash_nibble_features(packet: dict[str, JsonValue]) -> dict[str, float]:
    packet_hash = hashlib.sha256(_canonical_bytes(packet)).hexdigest()
    return {f"nibble_{index}": float(int(value, 16)) for index, value in enumerate(packet_hash)}


def _structural_content_features(packet: dict[str, JsonValue]) -> dict[str, float | str | bool]:
    options = [
        value
        for key in ("option_a", "option_b", "observed_element")
        if isinstance((value := packet.get(key)), dict)
    ]
    texts = [cast(str, option["text"]) for option in options]
    return {
        "mode": cast(str, packet["mode"]),
        "severity": cast(str, packet["severity"]),
        "categories": "|".join(sorted(cast(list[str], packet["categories"]))),
        "types": "|".join(cast(str, option["declared_type"]) for option in options),
        "text_lengths": "|".join(str(len(text)) for text in texts),
        "digit_counts": "|".join(str(sum(character.isdigit() for character in text)) for text in texts),
        "structure": "|".join(
            hashlib.sha256(_canonical_bytes(option["structure"])).hexdigest()[:8] for option in options
        ),
        "geometry": "|".join(
            hashlib.sha256(_canonical_bytes(option["geometry"])).hexdigest()[:8] for option in options
        ),
    }


def _public_audit_manifest(
    population: list[dict[str, JsonValue]],
    selected: list[dict[str, JsonValue]],
    mappings: list[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    selected_ids = {cast(str, packet["packet_id"]) for packet in selected}
    selected_mappings = [mapping for mapping in mappings if mapping["packet_id"] in selected_ids]
    internal_strata = sorted({
        cast(str, mapping["inclusion_stratum"])
        for mapping in mappings
        if mapping.get("sampling_eligibility") == "eligible"
    })
    aliases = {name: f"balanced_stratum_{position}" for position, name in enumerate(internal_strata, 1)}
    strata: list[dict[str, JsonValue]] = []
    for name in internal_strata:
        members = [mapping for mapping in mappings if mapping.get("inclusion_stratum") == name]
        chosen = [mapping for mapping in selected_mappings if mapping.get("inclusion_stratum") == name]
        probability = cast(float, members[0]["inclusion_probability"])
        strata.append({
            "stratum": aliases[name],
            "population": len(members),
            "selected": len(chosen),
            "inclusion_probability": probability,
            "inclusion_weight": 1.0 / probability,
        })
    return {
        "artifact_id": "isolated-secret-sample-v7",
        "state": "pending_human_review",
        "sampling": {
            "method": "secret keyed stratified sampling without replacement",
            "population": len(population),
            "selected": len(selected),
            "realized_rate": len(selected) / len(population),
            "target_rate": "1-2%",
            "balance": "presence-origin and paired A-position assignments are exactly balanced",
            "strata": cast(JsonValue, strata),
            "statistical_scope": (
                "Underpowered descriptive audit; exact intervals must be reported for any later binary rate, "
                "and no population-level precision claim is authorized."
            ),
        },
        "sample": [
            {"packet_id": packet["packet_id"], "review_state": "pending_human_review"} for packet in selected
        ],
    }


def _audit_manifest(
    population: list[dict[str, JsonValue]],
    selected: list[dict[str, JsonValue]],
    mappings: list[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    mapping_by_id = {cast(str, mapping["packet_id"]): mapping for mapping in mappings}
    selected_ids = {cast(str, packet["packet_id"]) for packet in selected}
    selected_mappings = [mapping for mapping in mappings if mapping["packet_id"] in selected_ids]
    origins = Counter(
        cast(str, mapping["presence_origin"])
        for mapping in selected_mappings
        if mapping["mode"] == "presence"
    )
    return cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic13-secret-human-audit-sample-v7",
            "state": "pending_human_review",
            "review_performed": False,
            "promotion_performed": False,
            "sampling": {
                "method": "forced stratified secret-HMAC sampling without replacement; not uniform",
                "design": (
                    "fixed quotas force both protected presence-origin strata and both protected paired-A "
                    "assignment strata; secret HMAC orders records within each stratum"
                ),
                "inclusion_probabilities": [
                    {
                        "protected_stratum": stratum,
                        "population": len([m for m in mappings if m.get("inclusion_stratum") == stratum]),
                        "selected": len([
                            m for m in selected_mappings if m.get("inclusion_stratum") == stratum
                        ]),
                        "probability": cast(
                            float,
                            next(
                                m["inclusion_probability"]
                                for m in mappings
                                if m.get("inclusion_stratum") == stratum
                            ),
                        ),
                        "weight": cast(
                            float,
                            next(
                                m["sampling_weight"]
                                for m in mappings
                                if m.get("inclusion_stratum") == stratum
                            ),
                        ),
                    }
                    for stratum in sorted({
                        cast(str, m["inclusion_stratum"])
                        for m in mappings
                        if m.get("sampling_eligibility") == "eligible"
                    })
                ],
                "population": len(population),
                "selected_unique_disagreements": len(selected),
                "realized_rate": len(selected) / len(population),
                "actual_coverage": {
                    "documents": sorted({cast(str, packet["document"]) for packet in selected}),
                    "severities": sorted({cast(str, packet["severity"]) for packet in selected}),
                    "categories": sorted({
                        category for packet in selected for category in cast(list[str], packet["categories"])
                    }),
                    "modes": sorted({cast(str, packet["mode"]) for packet in selected}),
                    "presence_count": sum(packet["mode"] == "presence" for packet in selected),
                    "paired_count": sum(packet["mode"] == "paired_ab" for packet in selected),
                },
                "coverage_declaration": "only listed actual coverage is claimed",
                "protected_balance_assertion": origins["candidate"] == origins["reference"] > 0,
            },
            "sample": [
                {
                    "packet_id": packet["packet_id"],
                    "packet_sha256": _sha256(_canonical_bytes(packet)),
                    "review_state": "pending_human_review",
                }
                for packet in selected
                if cast(str, packet["packet_id"]) in mapping_by_id
            ],
        },
    )


def _ledger_record(value: dict[str, object]) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "record_id": _sha256(cast(str, value["neutral_key"]).encode())[:24],
            "neutral_key": value["neutral_key"],
            "document": value["document"],
            "mode": value["mode"],
            "candidate_index": value["candidate_index"],
            "reference_index": value["reference_index"],
            "candidate_element_id": value["candidate_element_id"],
            "reference_element_id": value["reference_element_id"],
            "presence_origin": value["presence_origin"],
            "categories": value["categories"],
            "material": value["material"],
            "exclusion_reason": value["exclusion_reason"],
            "alignment_score": value["alignment_score"],
            "candidate_option": value["candidate_option"],
            "reference_option": value["reference_option"],
            "evidence": value["evidence"],
        },
    )


def _protected_provenance(
    *,
    root_dir: Path,
    input_directories: dict[str, Path],
    input_hashes: dict[str, str],
    full_packets: list[dict[str, JsonValue]],
    reviewer_packets: list[dict[str, JsonValue]],
    mappings: list[dict[str, JsonValue]],
    raw: list[dict[str, object]],
    reconstruction: dict[str, JsonValue],
    adjudicability: dict[str, JsonValue],
    design_acceptance: dict[str, JsonValue],
    reviewer_hashes: dict[str, str],
) -> dict[str, JsonValue]:
    code_fingerprint = project_code_fingerprint().model_dump(mode="json")
    selected_ids = {cast(str, packet["packet_id"]) for packet in reviewer_packets}
    selected_internal = [packet for packet in full_packets if packet["packet_id"] in selected_ids]
    selected_mappings = [mapping for mapping in mappings if mapping["packet_id"] in selected_ids]
    material_count = sum(record["material"] is True for record in raw)
    nonmaterial = [record for record in raw if record["material"] is False]
    category_counts = Counter(
        category for packet in full_packets for category in cast(list[str], packet["categories"])
    )
    return cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic13-protected-provenance-v7",
            "schema_versions": {
                "document": SCHEMA_VERSION,
                "bundle": BUNDLE_SCHEMA_VERSION,
                "normalization": NORMALIZATION_VERSION,
            },
            "algorithm_versions": {
                "selection": ALGORITHM_VERSION,
                "reconstruction_attack": ATTACK_VERSION,
                "design_acceptance": DESIGN_VERSION,
            },
            "inputs": {
                "directories": {
                    name: _relative_path(path, root_dir) for name, path in sorted(input_directories.items())
                },
                "sha256": dict(sorted(input_hashes.items())),
                "determinism": "byte_equal",
                "supplied_bronze_directory": _relative_path(input_directories["bronze"], root_dir),
            },
            "selection": {
                "materiality_thresholds": MATERIALITY_THRESHOLDS,
                "raw_record_count": len(raw),
                "material_record_count": material_count,
                "nonmaterial_record_count": len(nonmaterial),
                "exclusion_reason_counts": dict(
                    sorted(Counter(cast(str, record["exclusion_reason"]) for record in nonmaterial).items())
                ),
                "category_counts": dict(sorted(category_counts.items())),
                "ledger": "full-ledger.json",
            },
            "sampling": _audit_manifest(full_packets, selected_internal, mappings)["sampling"],
            "adjudicability": adjudicability,
            "design_acceptance": design_acceptance,
            "reconstruction_attack": reconstruction,
            "descriptive_mapping_diagnostics": {
                "sample": _run_descriptive_attacks(selected_internal, selected_mappings),
                "full_material_population": _run_descriptive_attacks(full_packets, mappings),
                "acceptance_role": "diagnostic_only",
                "confidence_intervals": "two-sided Clopper-Pearson exact 95% binomial intervals",
                "power_disclosure": "the 1-2% reviewer sample is underpowered for leakage inference",
            },
            "reviewer_export_sha256": reviewer_hashes,
            "runner_v3_provenance_machinery": {
                "runtime": runtime_versions().model_dump(mode="json"),
                "full_project_package_closure": code_fingerprint,
                "closure_rule": "every project app Python file plus pyproject.toml and uv.lock",
                "installed_distribution_records": _distribution_inventory(),
            },
            "binary_trust_limits": [
                "Installed RECORD metadata and versions are pinned but do not independently attest wheel provenance.",
                "Native shared libraries, the Python executable, kernel, hardware, and OS image are not byte-attested.",
                "PyMuPDF/MuPDF build versions come from runtime metadata; reproducibility is not a supply-chain proof.",
            ],
            "separation_and_reproduction": {
                "protected_permissions": "directory 0700; files 0600",
                "hmac": "HMAC-SHA256 with distinct packet-id, paired-option-position, and sample-order domains",
                "exact_seed_reproduction": "re-run with seed.json and the four pinned supplied input directories",
                "reviewer_isolation_required": True,
                "publication": (
                    "validation completes in staging before best-effort two-destination rename; the two renames "
                    "are not globally atomic"
                ),
                "state": "pending_human_review; no review, promotion, commit, or push performed",
            },
        },
    )


def _distribution_inventory() -> dict[str, JsonValue]:
    inventory: dict[str, JsonValue] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise ValueError("installed distribution is missing its Name metadata")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        record = distribution.read_text("RECORD")
        if record is None:
            raise FileNotFoundError(f"installed distribution has no RECORD metadata: {name}")
        pin: dict[str, JsonValue] = {
            "version": distribution.version,
            "record_sha256": _sha256(record.encode()),
        }
        previous = inventory.setdefault(name, pin)
        if previous != pin:
            raise ValueError(f"conflicting installed distribution metadata for {name}")
    return dict(sorted(inventory.items()))


def _reconstruction_attack(
    reviewer_files: dict[str, bytes], known_source_paths: list[Path], *, forbidden_names: set[str]
) -> dict[str, JsonValue]:
    known_paths = sorted({path.resolve(strict=True) for path in known_source_paths})
    known_hashes = {_sha256(path.read_bytes()) for path in known_paths}
    forbidden_tokens = {
        "candidate",
        "reference",
        "bronze",
        "silver",
        "determinism",
        *(name.casefold() for name in forbidden_names),
        *(path.as_posix().casefold() for path in known_paths),
    }
    metadata_strings: list[tuple[str, str]] = []
    content_fields = {"text", "items"}

    def collect(value: object, location: str, parent_key: str | None = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                metadata_strings.append((location, str(key)))
                if str(key) not in content_fields:
                    collect(child, location, str(key))
        elif isinstance(value, list):
            if parent_key not in content_fields:
                for child in value:
                    collect(child, location, parent_key)
        elif isinstance(value, str):
            metadata_strings.append((location, value))

    exact_known_file_matches: list[str] = []
    for name, content in sorted(reviewer_files.items()):
        metadata_strings.append((name, name))
        if Path(name).suffix == ".json":
            collect(json.loads(content), name)
        if any(content == path.read_bytes() for path in known_paths):
            exact_known_file_matches.append(name)
    hits: list[dict[str, JsonValue]] = []
    hash_pattern = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", re.IGNORECASE)
    for location, value in metadata_strings:
        lowered = value.casefold()
        matched = sorted(token for token in forbidden_tokens if token and token in lowered)
        embedded_hashes = sorted(set(hash_pattern.findall(value)))
        known_embedded = sorted(set(embedded_hashes) & known_hashes)
        if matched or embedded_hashes:
            hits.append({
                "location": location,
                "forbidden_lookup_tokens": cast(JsonValue, matched),
                "embedded_hash_count": len(embedded_hashes),
                "known_input_hash_count": len(known_embedded),
            })
    return {
        "method": (
            "scan every exported path and every non-content JSON metadata scalar; attempt exact whole-file "
            "and hash/path/name lookup against all known input/evidence files"
        ),
        "export_file_count": len(reviewer_files),
        "known_file_count": len(known_paths),
        "metadata_hits": cast(JsonValue, hits),
        "exact_known_file_matches": cast(JsonValue, exact_known_file_matches),
        "content_matching_limit": (
            "Copied page evidence and excerpted text remain inherently matchable by content; isolation is mandatory. "
            "Exact evidence-payload matches are inventoried but are not lookup metadata failures."
        ),
        "result": "passed" if not hits else "failed",
    }


def _media_type(path: str) -> str:
    suffix = Path(path).suffix
    if suffix == ".json":
        return "application/json"
    if suffix == ".png":
        return "image/png"
    raise ValueError(f"unsupported reviewer evidence media type: {path}")


def _stage_and_publish_best_effort(
    reviewer_dir: Path,
    protected_dir: Path,
    reviewer_files: dict[str, bytes],
    protected_files: dict[str, bytes],
) -> None:
    reviewer_dir.parent.mkdir(parents=True, exist_ok=True)
    protected_dir.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(12)
    reviewer_stage = reviewer_dir.with_name(f".{reviewer_dir.name}.v7-stage-{token}")
    protected_stage = protected_dir.with_name(f".{protected_dir.name}.v7-stage-{token}")
    if reviewer_stage.exists() or protected_stage.exists():
        raise FileExistsError("staging path already exists")
    reviewer_stage.mkdir()
    protected_stage.mkdir(mode=0o700)
    os.chmod(protected_stage, 0o700)
    for name, content in reviewer_files.items():
        path = reviewer_stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for name, content in protected_files.items():
        path = protected_stage / name
        path.write_bytes(content)
        os.chmod(path, 0o600)
    _verify_exports(reviewer_stage, protected_stage)
    os.replace(reviewer_stage, reviewer_dir)
    try:
        os.replace(protected_stage, protected_dir)
    except OSError:
        os.replace(reviewer_dir, reviewer_stage)
        raise
    _verify_exports(reviewer_dir, protected_dir)


def _record_evidence_paths(record: dict[str, object]) -> list[str]:
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        raise TypeError("ledger record evidence must be an object")
    values = [value for value in evidence.values() if isinstance(value, str)]
    images = evidence.get("page_images")
    if not isinstance(images, list) or any(not isinstance(value, str) for value in images):
        raise TypeError("ledger page image evidence must be a list of strings")
    return values + cast(list[str], images)


def _evidence_paths(packet: dict[str, JsonValue]) -> list[str]:
    evidence = packet["evidence"]
    if not isinstance(evidence, dict):
        raise TypeError("packet evidence must be an object")
    values = [value for value in evidence.values() if isinstance(value, str)]
    images = evidence.get("page_images")
    if not isinstance(images, list) or any(not isinstance(value, str) for value in images):
        raise TypeError("page image evidence must be a list of strings")
    return values + cast(list[str], images)


def _verify_exports(reviewer_dir: Path, protected_dir: Path) -> None:
    root_files = {path.name for path in reviewer_dir.iterdir() if path.is_file()}
    if root_files != set(REVIEWER_ROOT_FILES) or not (reviewer_dir / "evidence").is_dir():
        raise RuntimeError("reviewer bundle root or standalone evidence directory is incomplete")
    manifest = json.loads((reviewer_dir / "manifest.json").read_bytes())
    inventory = manifest.get("bundle_inventory")
    if not isinstance(inventory, list):
        raise TypeError("reviewer manifest requires a complete bundle inventory")
    declared = {item.get("path") for item in inventory if isinstance(item, dict)}
    actual = {path.relative_to(reviewer_dir).as_posix() for path in reviewer_dir.rglob("*") if path.is_file()}
    if declared != actual:
        raise RuntimeError("reviewer evidence inventory does not match exported files")
    if {path.name for path in protected_dir.iterdir()} != set(PROTECTED_FILES):
        raise RuntimeError("protected directory contains unexpected files")
    if stat.S_IMODE(protected_dir.stat().st_mode) != 0o700:
        raise PermissionError("protected directory mode is not 0700")
    for name in PROTECTED_FILES:
        if stat.S_IMODE((protected_dir / name).stat().st_mode) != 0o600:
            raise PermissionError(f"protected file mode is not 0600: {name}")


def _inversion_participants(alignments: list[ElementAlignment]) -> set[tuple[int, int]]:
    participants: set[tuple[int, int]] = set()
    for position, left in enumerate(alignments):
        for right in alignments[position + 1 :]:
            if left.candidate_index > right.candidate_index:
                participants.add((left.candidate_index, left.reference_index))
                participants.add((right.candidate_index, right.reference_index))
    return participants


def _fragment_pages(element: DocumentElement) -> set[int]:
    return {fragment.page_number for fragment in element.fragments}


def _neutral_key(
    document: str,
    candidate_index: int | None,
    reference_index: int | None,
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
) -> str:
    return _sha256(
        _canonical_bytes({
            "document": document,
            "candidate_index": candidate_index,
            "reference_index": reference_index,
            "candidate_digest": _sha256(_canonical_bytes(_canonical_option(candidate))),
            "reference_digest": _sha256(_canonical_bytes(_canonical_option(reference))),
        })
    )


def _secret_public_id(seed: bytes, value: str) -> str:
    raw = bytes.fromhex(_secret_token(seed, "packet-id", value, length=32))
    return "pkt-" + base64.b32encode(raw).decode().rstrip("=").lower()


def _secret_token(seed: bytes, domain: str, value: str, *, length: int) -> str:
    key = hmac.new(seed, f"blind-v7-key\0{domain}".encode(), hashlib.sha256).digest()
    return hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:length]


def _secret_bit(seed: bytes, domain: str, value: str) -> int:
    return int(_secret_token(seed, domain, value, length=2), 16) & 1


def _text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def _validate_disjoint_outputs(reviewer_dir: Path, protected_dir: Path) -> None:
    reviewer = reviewer_dir.resolve(strict=False)
    protected = protected_dir.resolve(strict=False)
    if reviewer == protected or reviewer in protected.parents or protected in reviewer.parents:
        raise ValueError("protected directory must be outside and disjoint from reviewer bundle")


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
