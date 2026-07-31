from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import unicodedata
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import JsonValue
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from app.pdf2md.evaluation import ElementAlignment, evaluate_document
from app.pdf2md.schema import SCHEMA_VERSION, DocumentElement, read_document_elements

BUNDLE_SCHEMA_VERSION = "blind-review-v4.0.0"
ALGORITHM_VERSION = "semantic-materiality-v4.0.0"
NORMALIZATION_VERSION = "neutral-summary-v4.0.0"
ATTACK_VERSION = "pinned-adversarial-suite-v4.0.0"
REVIEWER_FILES = frozenset({"packets.json", "audit-sample-manifest.json", "manifest.json"})
PROTECTED_FILES = frozenset({"source-map.json", "seed.json", "exclusion-ledger.json"})

# These semantic thresholds are protocol constants, not population ranks or quotas.
MATERIALITY_THRESHOLDS: dict[str, JsonValue] = {
    "presence_minimum_non_whitespace_characters": 2,
    "character_normalized_edit_distance": 0.01,
    "geometry_region_change": "different cell in a neutral 3_by_3 page grid",
    "qualifying_categories": [
        "element_presence",
        "semantic_structure",
        "table_content_topology",
        "text_content",
        "reading_order_segmentation",
        "material_geometry",
    ],
}
ATTACK_THRESHOLDS: dict[str, float] = {
    "option_balanced_accuracy": 0.65,
    "paired_mapping_recovery": 0.70,
    "one_sided_mapping_recovery": 0.70,
    "full_packet_mapping_recovery": 0.70,
    "public_id_position_recovery": 0.60,
    "public_hash_position_recovery": 0.60,
}


def build_blind_v4(
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
    """Build v4 from pinned byte-identical candidates and current silver.

    Reviewer and protected outputs are disjoint. Generation is transactional in the
    sense that no output directory is created until all pinned attacks have passed.
    """
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
    if {path.name for path in candidate_paths} != {
        path.name for path in deterministic_candidate_dir.glob("*.parquet")
    }:
        raise ValueError("determinism candidate file sets differ")

    input_hashes: dict[str, str] = {}
    raw: list[dict[str, object]] = []
    for candidate_path in candidate_paths:
        rerun_path = deterministic_candidate_dir / candidate_path.name
        reference_path = reference_dir / candidate_path.name
        for path in (candidate_path, rerun_path, reference_path):
            validate_blind_path(path, must_exist=True)
            input_hashes[_relative_path(path, root_dir)] = _sha256(path.read_bytes())
        if candidate_path.read_bytes() != rerun_path.read_bytes():
            raise ValueError(
                f"semantic13 candidate is not byte-equal to determinism rerun: {candidate_path.name}"
            )
        raw.extend(
            _recompute_document_disagreements(
                candidate_path.stem,
                read_document_elements(candidate_path),
                read_document_elements(reference_path),
            )
        )

    material = [value for value in raw if value["material"] is True]
    excluded = [value for value in raw if value["material"] is False]
    if len(material) + len(excluded) != len(raw):
        raise RuntimeError("every recomputed disagreement must be material or ledgered")
    if not material:
        raise ValueError("semantic materiality thresholds selected no disagreements")

    packets: list[dict[str, JsonValue]] = []
    mappings: list[dict[str, JsonValue]] = []
    for disagreement in material:
        neutral_key = cast(str, disagreement["neutral_key"])
        packet_id = _secret_token(secret_seed, "packet-id", neutral_key, length=24)
        first = _secret_bit(secret_seed, "option-position", neutral_key) == 0
        candidate_option = cast(JsonValue, disagreement["candidate_option"])
        reference_option = cast(JsonValue, disagreement["reference_option"])
        option_a, option_b = (
            (candidate_option, reference_option) if first else (reference_option, candidate_option)
        )
        source_a, source_b = ("candidate", "reference") if first else ("reference", "candidate")
        if option_a is None:
            source_a = "no_element"
        if option_b is None:
            source_b = "no_element"
        mode = cast(str, disagreement["mode"])
        packets.append(
            cast(
                dict[str, JsonValue],
                {
                    "packet_id": packet_id,
                    "document": disagreement["document"],
                    "document_id": disagreement["document_id"],
                    "mode": mode,
                    "page_numbers": disagreement["page_numbers"],
                    "evidence": disagreement["evidence"],
                    "option_a": option_a,
                    "option_b": option_b,
                    "categories": disagreement["categories"],
                    "severity": disagreement["severity"],
                    "review_state": "pending_human_review",
                },
            )
        )
        mappings.append(
            cast(
                dict[str, JsonValue],
                {
                    "packet_id": packet_id,
                    "neutral_key": neutral_key,
                    "document": disagreement["document"],
                    "mode": mode,
                    "candidate_index": disagreement["candidate_index"],
                    "reference_index": disagreement["reference_index"],
                    "candidate_element_id": disagreement["candidate_element_id"],
                    "reference_element_id": disagreement["reference_element_id"],
                    "review_element_classes": disagreement["review_element_classes"],
                    "source_a": source_a,
                    "source_b": source_b,
                },
            )
        )

    packets.sort(key=lambda value: cast(str, value["packet_id"]))
    mappings.sort(key=lambda value: cast(str, value["packet_id"]))
    for packet in packets:
        for relative in _evidence_paths(packet):
            evidence_path = root_dir / relative
            validate_blind_path(evidence_path, must_exist=True)
            input_hashes.setdefault(relative, _sha256(evidence_path.read_bytes()))

    attacks = _run_adversarial_acceptance(packets, mappings)
    if attacks["result"] != "passed":
        raise ValueError(f"pinned adversarial acceptance failed: {json.dumps(attacks, sort_keys=True)}")

    sample = _reviewer_sample(packets, mappings, secret_seed)
    packet_bytes = _pretty_bytes(packets)
    audit = _audit_manifest(packets, mappings, sample)
    audit_bytes = _pretty_bytes(audit)
    category_counts = Counter(
        category for packet in packets for category in cast(list[str], packet["categories"])
    )
    mode_counts = Counter(cast(str, packet["mode"]) for packet in packets)
    exclusion_counts = Counter(cast(str, value["exclusion_reason"]) for value in excluded)
    manifest = cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic13-non-holdout-blind-adjudication-v4",
            "scope": "non_holdout",
            "state": "pending_human_review",
            "human_review_performed": False,
            "golden_promotion_performed": False,
            "packet_count": len(packets),
            "schema_versions": {
                "document": SCHEMA_VERSION,
                "bundle": BUNDLE_SCHEMA_VERSION,
                "normalization": NORMALIZATION_VERSION,
            },
            "algorithm_versions": {
                "selection": ALGORITHM_VERSION,
                "adversarial_acceptance": ATTACK_VERSION,
            },
            "inputs": {
                "candidate_directory": _relative_path(candidate_dir, root_dir),
                "determinism_candidate_directory": _relative_path(deterministic_candidate_dir, root_dir),
                "reference_directory": _relative_path(reference_dir, root_dir),
                "bronze_directory": _relative_path(bronze_dir, root_dir),
                "sha256": dict(sorted(input_hashes.items())),
                "candidate_determinism": "byte_equal",
            },
            "selection": {
                "source": "independent current evaluator recomputation from semantic13 and current silver",
                "policy": "all records satisfying one or more semantic category thresholds; no rank or top-N",
                "materiality_thresholds": MATERIALITY_THRESHOLDS,
                "raw_disagreement_count": len(raw),
                "material_disagreement_count": len(material),
                "excluded_record_count": len(excluded),
                "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
                "protected_external_exclusion_ledger": "exclusion-ledger.json",
            },
            "adjudication_modes": {
                "paired_ab": "two present elements are compared",
                "presence_absence": "one present element is compared with a secret-randomized no-element position",
                "counts": dict(sorted(mode_counts.items())),
            },
            "normalization": {
                "option_schema": ["kind", "summary", "text"],
                "geometry": "exact coordinates and source boxes omitted from options",
                "roles": "source roles and element types omitted from options",
                "table": "representation, dimensions, cells, and cell roles omitted from options",
                "provenance_identifiers_properties": "omitted recursively",
                "adjudicability": "visible text, text-derived neutral counts, packet-level pages, and external bronze evidence",
            },
            "stratification": {
                "attack_population": "all material packets including every presence_absence packet",
                "training_balance": "inverse-frequency fold weights; one-sided source classes and adjudication modes receive equal scoring weight",
                "raw_accuracy_reporting": "raw one-sided and paired accuracies are retained beside balanced acceptance scores",
                "category_counts": dict(sorted(category_counts.items())),
            },
            "adversarial_acceptance": attacks,
            "reviewer_export_files": sorted(REVIEWER_FILES),
            "protected_files": sorted(PROTECTED_FILES),
            "separation": {
                "filesystem_permissions": "protected directory 0700 and files 0600 where supported",
                "acl_note": "filesystem modes and ACLs are defense-in-depth, not the sole identity separation",
                "identity_design": "mapping and seed are separate from the exactly-three-file reviewer bundle",
                "hmac_domains": ["packet-id", "option-position", "sample-order"],
            },
            "sha256": {
                "packets.json": _sha256(packet_bytes),
                "audit-sample-manifest.json": _sha256(audit_bytes),
            },
            "validation": {
                "reviewer_source_map_absent": "passed",
                "holdout": "not_inspected",
                "layout_model": "not_used",
                "human_review": "not_performed",
                "promotion": "not_performed",
            },
        },
    )
    manifest_bytes = _pretty_bytes(manifest)
    source_map = {
        "artifact_id": "semantic13-protected-source-map-v4",
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "reviewer_sha256": {
            "packets.json": _sha256(packet_bytes),
            "audit-sample-manifest.json": _sha256(audit_bytes),
            "manifest.json": _sha256(manifest_bytes),
        },
        "mappings": mappings,
    }
    seed_record = {
        "artifact_id": "semantic13-protected-seed-v4",
        "encoding": "hex",
        "bits": len(secret_seed) * 8,
        "value": secret_seed.hex(),
        "hmac": "HMAC-SHA256",
        "domains": ["packet-id", "option-position", "sample-order"],
    }
    exclusion_ledger = {
        "artifact_id": "semantic13-protected-exclusion-ledger-v4",
        "policy": "all independently recomputed nonmaterial or uncategorized records",
        "thresholds": MATERIALITY_THRESHOLDS,
        "record_count": len(excluded),
        "records": [
            _ledger_record(value) for value in sorted(excluded, key=lambda x: cast(str, x["neutral_key"]))
        ],
    }

    reviewer_dir.mkdir(parents=True)
    protected_dir.mkdir(parents=True, mode=0o700)
    os.chmod(protected_dir, 0o700)
    (reviewer_dir / "packets.json").write_bytes(packet_bytes)
    (reviewer_dir / "audit-sample-manifest.json").write_bytes(audit_bytes)
    (reviewer_dir / "manifest.json").write_bytes(manifest_bytes)
    protected_values = {
        "source-map.json": source_map,
        "seed.json": seed_record,
        "exclusion-ledger.json": exclusion_ledger,
    }
    for name, value in protected_values.items():
        path = protected_dir / name
        path.write_bytes(_pretty_bytes(value))
        os.chmod(path, 0o600)
    _verify_exports(reviewer_dir, protected_dir)


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


def validate_blind_path(path: Path, *, must_exist: bool) -> None:
    resolved = path.resolve(strict=must_exist)
    for inspected in (path.absolute(), resolved):
        lowered = [part.lower() for part in inspected.parts]
        if any("holdout" in part for part in lowered):
            raise ValueError(f"holdout path is forbidden: {path}")
        if any("layoutlm" in part for part in lowered):
            raise ValueError(f"LayoutLM path is forbidden: {path}")


def _recompute_document_disagreements(
    document: str, candidates: list[DocumentElement], references: list[DocumentElement]
) -> list[dict[str, object]]:
    report = evaluate_document(candidates, references)
    inversions = _inversion_participants(report.alignments)
    values: list[dict[str, object]] = []
    for alignment in report.alignments:
        candidate = candidates[alignment.candidate_index]
        reference = references[alignment.reference_index]
        if _canonical_option(candidate) == _canonical_option(reference):
            # Evaluator alignments that are equal under the reviewer protocol are still ledgered.
            values.append(
                _disagreement(
                    document,
                    candidate,
                    reference,
                    alignment.candidate_index,
                    alignment.reference_index,
                    alignment.score,
                    (alignment.candidate_index, alignment.reference_index) in inversions,
                    force_exclusion="equivalent_after_neutral_canonicalization",
                )
            )
            continue
        values.append(
            _disagreement(
                document,
                candidate,
                reference,
                alignment.candidate_index,
                alignment.reference_index,
                alignment.score,
                (alignment.candidate_index, alignment.reference_index) in inversions,
            )
        )
    for index in report.unmatched_candidate_indices:
        values.append(_disagreement(document, candidates[index], None, index, None, 0.0, True))
    for index in report.unmatched_reference_indices:
        values.append(_disagreement(document, None, references[index], None, index, 0.0, True))
    return values


def _disagreement(
    document: str,
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
    candidate_index: int | None,
    reference_index: int | None,
    alignment_score: float,
    order_issue: bool,
    *,
    force_exclusion: str | None = None,
) -> dict[str, object]:
    if candidate is None and reference is None:
        raise ValueError("disagreement requires at least one source element")
    categories = _material_categories(candidate, reference, order_issue)
    reason = force_exclusion or _exclusion_reason(candidate, reference, categories)
    material = reason is None
    element = candidate or reference
    assert element is not None
    pages = sorted({
        fragment.page_number for item in (candidate, reference) if item for fragment in item.fragments
    })
    neutral_key = _neutral_key(document, candidate_index, reference_index, candidate, reference)
    return {
        "neutral_key": neutral_key,
        "document": document,
        "document_id": element.document_id,
        "mode": "presence_absence" if candidate is None or reference is None else "paired_ab",
        "candidate_index": candidate_index,
        "reference_index": reference_index,
        "candidate_element_id": candidate.element_id if candidate else None,
        "reference_element_id": reference.element_id if reference else None,
        "candidate_option": _canonical_option(candidate),
        "reference_option": _canonical_option(reference),
        "page_numbers": pages,
        "evidence": {
            "bronze_manifest": f"data/bronze/{document}/manifest.json",
            "bronze_native_json": f"data/bronze/{document}/liteparse.json",
            "bronze_native_text": f"data/bronze/{document}/liteparse.txt",
            "page_images": [f"data/bronze/{document}/pages/page-{page:04d}.png" for page in pages],
        },
        "categories": categories,
        "severity": _severity(categories, candidate, reference),
        "review_element_classes": sorted({
            _review_element_class(item) for item in (candidate, reference) if item is not None
        }),
        "alignment_score": alignment_score,
        "material": material,
        "exclusion_reason": reason,
    }


def _canonical_option(element: DocumentElement | None) -> dict[str, JsonValue] | None:
    if element is None:
        return None
    text = _text(element.content)
    # Every field is derived from visible text or page membership. Source role,
    # element type, exact geometry, and table-cell schema are intentionally absent.
    summary: dict[str, JsonValue] = {
        "line_count": len(text.splitlines()) if text else 0,
        "non_whitespace_characters": sum(not character.isspace() for character in text),
        "token_count": len(text.split()),
    }
    return cast(
        dict[str, JsonValue],
        {
            "kind": "visible_content",
            "text": text,
            "summary": summary,
        },
    )


def _review_element_class(element: DocumentElement) -> str:
    value = _text(element.element_type).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "image": "figure",
        "picture": "figure",
        "header": "running_matter",
        "footer": "running_matter",
        "text": "paragraph",
        "body": "paragraph",
    }
    return aliases.get(value, value)


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
    text_differs = _text_material(candidate.content, reference.content)
    if _visible_structure(candidate.content) != _visible_structure(reference.content):
        categories.append("semantic_structure")
    if text_differs and (candidate.structure.table is not None or reference.structure.table is not None):
        categories.append("table_content_topology")
    if text_differs:
        categories.append("text_content")
    if order_issue or _fragment_pages(candidate) != _fragment_pages(reference):
        categories.append("reading_order_segmentation")
    # Geometry is qualifying only when it changes the neutral 3x3 page region.
    # Exact source boxes never enter an option or a materiality decision.
    if _region_payload(candidate) != _region_payload(reference):
        categories.append("material_geometry")
    return sorted(set(categories))


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
    return "uncategorized_or_below_semantic_thresholds"


def _severity(
    categories: list[str], candidate: DocumentElement | None, reference: DocumentElement | None
) -> str:
    category_set = set(categories)
    if "table_content_topology" in category_set and "text_content" in category_set:
        return "critical"
    if "element_presence" in category_set or category_set & {
        "table_content_topology",
        "semantic_structure",
        "reading_order_segmentation",
    }:
        return "high"
    if "text_content" in category_set:
        return "medium"
    if "material_geometry" in category_set:
        return "low"
    return "low"


def _run_adversarial_acceptance(
    packets: list[dict[str, JsonValue]], mappings: list[dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    mapping_by_id = {cast(str, value["packet_id"]): value for value in mappings}
    features: list[dict[str, float | str | bool]] = []
    labels: list[bool] = []
    groups: list[str] = []
    references: list[tuple[str, str]] = []
    for packet in packets:
        mapping = mapping_by_id[cast(str, packet["packet_id"])]
        for option_name, source_name in (("option_a", "source_a"), ("option_b", "source_b")):
            option = packet[option_name]
            source = mapping[source_name]
            if not isinstance(option, dict) or source == "no_element":
                continue
            features.append(_no_content_features(cast(dict[str, JsonValue], option), packet))
            labels.append(source == "candidate")
            groups.append(cast(str, packet["document"]))
            references.append((cast(str, packet["packet_id"]), option_name))
    if len(set(groups)) < 2 or len(set(labels)) < 2:
        raise ValueError("adversarial acceptance requires both origins across multiple documents")

    vectorizer = DictVectorizer(sparse=False)
    matrix = cast(np.ndarray, vectorizer.fit_transform(features))
    logo = LeaveOneGroupOut()
    model_results: dict[str, JsonValue] = {}
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=4104, n_jobs=1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=4104, n_jobs=1
        ),
    }
    labels_array = np.asarray(labels, dtype=bool)
    groups_array = np.asarray(groups)
    for name, model in models.items():
        probabilities = np.zeros(len(labels), dtype=float)
        for train, test in logo.split(matrix, labels_array, groups_array):
            train_labels = labels_array[train]
            candidate_count = int(train_labels.sum())
            reference_count = len(train_labels) - candidate_count
            if candidate_count == 0 or reference_count == 0:
                raise ValueError("every LODO training fold must contain both origins")
            weights = np.where(
                train_labels,
                len(train_labels) / (2 * candidate_count),
                len(train_labels) / (2 * reference_count),
            )
            model.fit(matrix[train], train_labels, sample_weight=weights)
            class_index = list(model.classes_).index(True)
            fold_probabilities = cast(np.ndarray, np.asarray(model.predict_proba(matrix[test])))
            probabilities[test] = fold_probabilities[:, class_index]
        predictions = probabilities >= 0.5
        raw_option_balanced = _mode_balanced_option_accuracy(packets, references, labels_array, predictions)
        option_balanced = max(raw_option_balanced, 1.0 - raw_option_balanced)
        mapping_scores = _mapping_scores(packets, mapping_by_id, references, probabilities)
        model_results[name] = cast(
            JsonValue,
            {
                "option_balanced_accuracy": option_balanced,
                **mapping_scores,
            },
        )

    public = _public_position_attacks(packets, mapping_by_id)
    maxima = {
        metric: max(
            cast(float, cast(dict[str, JsonValue], value)[metric]) for value in model_results.values()
        )
        for metric in (
            "option_balanced_accuracy",
            "paired_mapping_recovery",
            "one_sided_mapping_recovery",
            "full_packet_mapping_recovery",
        )
    }
    passed = all(maxima[key] <= ATTACK_THRESHOLDS[key] for key in maxima) and all(
        public[key] <= ATTACK_THRESHOLDS[key] for key in public
    )
    return cast(
        dict[str, JsonValue],
        {
            "method": "LODO no-content structural RF/ExtraTrees; balanced fold weights and mode/class-balanced full mapping",
            "population": {
                "packets": len(packets),
                "paired": sum(packet["mode"] == "paired_ab" for packet in packets),
                "one_sided": sum(packet["mode"] == "presence_absence" for packet in packets),
                "present_options": len(labels),
                "excluded_from_attacks": 0,
            },
            "models": model_results,
            "public_attacks": public,
            "worst_model": maxima,
            "thresholds": ATTACK_THRESHOLDS,
            "result": "passed" if passed else "failed",
        },
    )


def _no_content_features(
    option: dict[str, JsonValue], packet: dict[str, JsonValue]
) -> dict[str, float | str | bool]:
    # Content and every content-derived summary value are excluded. All remaining
    # public packet structure is represented, including mode and population strata.
    categories = cast(list[str], packet["categories"])
    evidence = cast(dict[str, JsonValue], packet["evidence"])
    return {
        "option_kind": cast(str, option["kind"]),
        "summary_schema": "|".join(sorted(cast(dict[str, JsonValue], option["summary"]))),
        "mode": cast(str, packet["mode"]),
        "severity": cast(str, packet["severity"]),
        "categories": "|".join(sorted(categories)),
        "packet_page_count": float(len(cast(list[int], packet["page_numbers"]))),
        "evidence_image_count": float(len(cast(list[str], evidence["page_images"]))),
    }


def _mode_balanced_option_accuracy(
    packets: list[dict[str, JsonValue]],
    references: list[tuple[str, str]],
    expected: np.ndarray,
    predicted: np.ndarray,
) -> float:
    mode_by_id = {cast(str, packet["packet_id"]): cast(str, packet["mode"]) for packet in packets}
    scores: list[float] = []
    for mode in ("paired_ab", "presence_absence"):
        indices = [index for index, (packet_id, _) in enumerate(references) if mode_by_id[packet_id] == mode]
        if not indices:
            raise ValueError(f"option acceptance requires mode: {mode}")
        scores.append(float(balanced_accuracy_score(expected[indices], predicted[indices])))
    return sum(scores) / len(scores)


def _mapping_scores(
    packets: list[dict[str, JsonValue]],
    mapping_by_id: dict[str, dict[str, JsonValue]],
    references: list[tuple[str, str]],
    probabilities: np.ndarray,
) -> dict[str, float]:
    probability_by_option = dict(zip(references, probabilities.tolist(), strict=True))
    paired_hits: list[bool] = []
    one_sided_expected: list[bool] = []
    one_sided_predicted: list[bool] = []
    for packet in packets:
        packet_id = cast(str, packet["packet_id"])
        mapping = mapping_by_id[packet_id]
        if packet["mode"] == "paired_ab":
            candidate_is_a = (
                probability_by_option[(packet_id, "option_a")]
                > probability_by_option[(packet_id, "option_b")]
            )
            paired_hits.append(candidate_is_a == (mapping["source_a"] == "candidate"))
        else:
            present_name = "option_a" if isinstance(packet["option_a"], dict) else "option_b"
            one_sided_predicted.append(probability_by_option[(packet_id, present_name)] >= 0.5)
            one_sided_expected.append(mapping[f"source_{present_name[-1]}"] == "candidate")
    if (
        not paired_hits
        or not one_sided_expected
        or len(paired_hits) + len(one_sided_expected) != len(packets)
    ):
        raise ValueError("full mapping acceptance requires every paired and one-sided packet")
    raw_paired = sum(paired_hits) / len(paired_hits)
    paired_recovery = max(raw_paired, 1.0 - raw_paired)
    raw_one_sided = sum(
        predicted == expected
        for predicted, expected in zip(one_sided_predicted, one_sided_expected, strict=True)
    ) / len(one_sided_expected)
    balanced_one_sided = float(
        balanced_accuracy_score(np.asarray(one_sided_expected), np.asarray(one_sided_predicted))
    )
    one_sided_recovery = max(balanced_one_sided, 1.0 - balanced_one_sided)
    # Equal mode weight prevents the much larger one-sided stratum from leaking its
    # population majority while every packet still contributes within its stratum.
    full_recovery = (paired_recovery + one_sided_recovery) / 2
    return {
        "paired_mapping_recovery": paired_recovery,
        "one_sided_mapping_recovery": one_sided_recovery,
        "full_packet_mapping_recovery": full_recovery,
        "raw_paired_mapping_accuracy": raw_paired,
        "raw_one_sided_mapping_accuracy": raw_one_sided,
    }


def _public_position_attacks(
    packets: list[dict[str, JsonValue]], mapping_by_id: dict[str, dict[str, JsonValue]]
) -> dict[str, float]:
    parity_hits = 0
    hash_hits = 0
    for packet in packets:
        packet_id = cast(str, packet["packet_id"])
        element_is_a = (
            isinstance(packet["option_a"], dict)
            if packet["mode"] == "presence_absence"
            else (mapping_by_id[packet_id]["source_a"] == "candidate")
        )
        parity_hits += (int(packet_id[-1], 16) % 2 == 0) == element_is_a
        public = _canonical_bytes(packet)
        hash_hits += (hashlib.sha256(public).digest()[0] % 2 == 0) == element_is_a
    parity = parity_hits / len(packets)
    public_hash = hash_hits / len(packets)
    return {
        "public_id_position_recovery": max(parity, 1.0 - parity),
        "public_hash_position_recovery": max(public_hash, 1.0 - public_hash),
    }


def _reviewer_sample(
    packets: list[dict[str, JsonValue]], mappings: list[dict[str, JsonValue]], seed: bytes
) -> list[dict[str, JsonValue]]:
    minimum = math.ceil(len(packets) * 0.01)
    target = max(minimum, math.floor(len(packets) * 0.02))
    if not (0.01 <= target / len(packets) <= 0.02):
        raise ValueError("population cannot support an integer 1-2% reviewer sample")
    mapping_by_id = {cast(str, value["packet_id"]): value for value in mappings}
    requirements = [
        ("severity:medium", lambda packet: packet["severity"] == "medium"),
        (
            "type:figure",
            lambda packet: "figure" in _packet_kinds(packet, mapping_by_id),
        ),
        (
            "type:footnote",
            lambda packet: "footnote" in _packet_kinds(packet, mapping_by_id),
        ),
        ("mode:paired_ab", lambda packet: packet["mode"] == "paired_ab"),
        ("mode:presence_absence", lambda packet: packet["mode"] == "presence_absence"),
    ]
    active = [(name, predicate) for name, predicate in requirements if any(predicate(p) for p in packets)]
    selected: list[dict[str, JsonValue]] = []
    covered: set[str] = set()
    remaining = list(packets)
    while len(selected) < target:

        def rank(packet: dict[str, JsonValue]) -> tuple[int, str]:
            gains = sum(name not in covered and predicate(packet) for name, predicate in active)
            token = _secret_token(seed, "sample-order", cast(str, packet["packet_id"]), length=64)
            return (-gains, token)

        chosen = min(remaining, key=rank)
        selected.append(chosen)
        for name, predicate in active:
            if predicate(chosen):
                covered.add(name)
        remaining.remove(chosen)
    missing = {name for name, _ in active} - covered
    if missing:
        raise ValueError(f"1-2% reviewer sample cannot cover required dimensions: {sorted(missing)}")
    return [
        {
            "packet_id": packet["packet_id"],
            "packet_sha256": _sha256(_canonical_bytes(packet)),
            "review_state": "pending_human_review",
        }
        for packet in sorted(selected, key=lambda value: cast(str, value["packet_id"]))
    ]


def _audit_manifest(
    packets: list[dict[str, JsonValue]],
    mappings: list[dict[str, JsonValue]],
    sample: list[dict[str, JsonValue]],
) -> dict[str, JsonValue]:
    ids = {cast(str, value["packet_id"]) for value in sample}
    selected = [packet for packet in packets if packet["packet_id"] in ids]
    mapping_by_id = {cast(str, value["packet_id"]): value for value in mappings}
    return cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic13-non-holdout-human-audit-sample-v4",
            "state": "pending_human_review",
            "review_performed": False,
            "promotion_performed": False,
            "sampling": {
                "method": "secret-seeded requirement-covering greedy without replacement",
                "population": len(packets),
                "selected_unique_disagreements": len(sample),
                "realized_rate": len(sample) / len(packets),
                "required_when_qualifying": [
                    "medium severity",
                    "figure",
                    "footnote",
                    "each adjudication mode",
                ],
                "actual_coverage": {
                    "documents": sorted({cast(str, packet["document"]) for packet in selected}),
                    "severities": sorted({cast(str, packet["severity"]) for packet in selected}),
                    "categories": sorted({
                        category for packet in selected for category in cast(list[str], packet["categories"])
                    }),
                    "element_kinds": sorted({
                        kind for packet in selected for kind in _packet_kinds(packet, mapping_by_id)
                    }),
                    "modes": sorted({cast(str, packet["mode"]) for packet in selected}),
                },
                "coverage_declaration": "only actual_coverage is claimed; unlisted population values are not claimed covered",
            },
            "sample": sample,
        },
    )


def _packet_kinds(packet: dict[str, JsonValue], mapping_by_id: dict[str, dict[str, JsonValue]]) -> set[str]:
    mapping = mapping_by_id[cast(str, packet["packet_id"])]
    return set(cast(list[str], mapping["review_element_classes"]))


def _ledger_record(value: dict[str, object]) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "record_id": _sha256(cast(str, value["neutral_key"]).encode())[:24],
            "neutral_key": value["neutral_key"],
            "document": value["document"],
            "candidate_index": value["candidate_index"],
            "reference_index": value["reference_index"],
            "candidate_element_id": value["candidate_element_id"],
            "reference_element_id": value["reference_element_id"],
            "categories": value["categories"],
            "exclusion_reason": value["exclusion_reason"],
        },
    )


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
    if {path.name for path in reviewer_dir.iterdir()} != set(REVIEWER_FILES):
        raise RuntimeError("reviewer bundle must contain exactly three declared files")
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


def _secret_token(seed: bytes, domain: str, value: str, *, length: int) -> str:
    key = hmac.new(seed, f"blind-v4-key\0{domain}".encode(), hashlib.sha256).digest()
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
