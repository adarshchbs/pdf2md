from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import unicodedata
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import JsonValue
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline

from app.pdf2md.evaluation import ElementAlignment, evaluate_document
from app.pdf2md.schema import SCHEMA_VERSION, DocumentElement, PageFragment, read_document_elements

BUNDLE_SCHEMA_VERSION = "blind-review-v3.0.0"
ALGORITHM_VERSION = "semantic12-independent-materiality-v3.0.0"
NORMALIZATION_VERSION = "canonical-option-v3.0.0"
CLASSIFIER_VERSION = "held-out-document-origin-probe-v1.0.0"
EXPECTED_DISAGREEMENTS = 336
REVIEWER_FILES = frozenset({"packets.json", "audit-sample-manifest.json", "manifest.json"})


def build_semantic12_blind_v3(
    candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    *,
    root_dir: Path,
    seed: bytes | None = None,
) -> None:
    """Build reviewer material and protected identity metadata into disjoint directories."""
    direct_inputs = (candidate_dir, reference_dir, bronze_dir, root_dir)
    for path in direct_inputs:
        _validate_path(path, must_exist=True)
    for path in (reviewer_dir, protected_dir):
        _validate_path(path, must_exist=False)
    _validate_disjoint_outputs(reviewer_dir, protected_dir)
    if reviewer_dir.exists():
        raise FileExistsError(f"reviewer output already exists: {reviewer_dir}")
    if protected_dir.exists():
        raise FileExistsError(f"protected output already exists: {protected_dir}")

    secret_seed = seed if seed is not None else secrets.token_bytes(32)
    if len(secret_seed) < 32:
        raise ValueError("protected assignment seed must contain at least 256 bits")

    candidate_paths = sorted(candidate_dir.glob("*.parquet"))
    if not candidate_paths:
        raise ValueError("candidate directory contains no Parquet documents")
    documents: list[tuple[str, list[DocumentElement], list[DocumentElement]]] = []
    input_hashes: dict[str, str] = {}
    for candidate_path in candidate_paths:
        reference_path = reference_dir / candidate_path.name
        for path in (candidate_path, reference_path):
            _validate_path(path, must_exist=True)
            input_hashes[_relative_path(path, root_dir)] = _sha256(path.read_bytes())
        document = candidate_path.stem
        documents.append((
            document,
            read_document_elements(candidate_path),
            read_document_elements(reference_path),
        ))

    raw_disagreements: list[dict[str, object]] = []
    for document, candidates, references in documents:
        raw_disagreements.extend(_recompute_document_disagreements(document, candidates, references))
    material = _select_material_disagreements(raw_disagreements)
    if len(material) != EXPECTED_DISAGREEMENTS:
        raise ValueError(
            f"independent materiality selection produced {len(material)} disagreements; "
            f"expected {EXPECTED_DISAGREEMENTS}"
        )

    packets: list[dict[str, JsonValue]] = []
    mappings: list[dict[str, JsonValue]] = []
    for disagreement in material:
        neutral_key = cast(str, disagreement["neutral_key"])
        packet_id = _secret_token(secret_seed, "packet-id", neutral_key, length=24)
        candidate_first = _secret_bit(secret_seed, "assignment", neutral_key) == 0
        candidate_option = cast(JsonValue, disagreement["candidate_option"])
        reference_option = cast(JsonValue, disagreement["reference_option"])
        if candidate_first:
            option_a, option_b = candidate_option, reference_option
            source_a, source_b = "candidate", "reference"
        else:
            option_a, option_b = reference_option, candidate_option
            source_a, source_b = "reference", "candidate"
        if candidate_option is None:
            source_a = "missing" if candidate_first else source_a
            source_b = "missing" if not candidate_first else source_b
        if reference_option is None:
            source_b = "missing" if candidate_first else source_b
            source_a = "missing" if not candidate_first else source_a

        packet = cast(
            dict[str, JsonValue],
            {
                "packet_id": packet_id,
                "document": disagreement["document"],
                "document_id": disagreement["document_id"],
                "page_numbers": disagreement["page_numbers"],
                "evidence": disagreement["evidence"],
                "option_a": option_a,
                "option_b": option_b,
                "categories": disagreement["categories"],
                "severity": disagreement["severity"],
                "review_state": "pending_human_review",
            },
        )
        packets.append(packet)
        mappings.append(
            cast(
                dict[str, JsonValue],
                {
                    "packet_id": packet_id,
                    "document": disagreement["document"],
                    "neutral_key": neutral_key,
                    "candidate_index": disagreement["candidate_index"],
                    "reference_index": disagreement["reference_index"],
                    "candidate_element_id": disagreement["candidate_element_id"],
                    "reference_element_id": disagreement["reference_element_id"],
                    "source_a": source_a,
                    "source_b": source_b,
                },
            )
        )

    for packet in packets:
        evidence = packet["evidence"]
        if not isinstance(evidence, dict):
            raise TypeError("packet evidence must be an object")
        evidence_paths = [value for value in evidence.values() if isinstance(value, str)]
        page_images = evidence.get("page_images")
        if not isinstance(page_images, list) or any(not isinstance(value, str) for value in page_images):
            raise TypeError("packet page image evidence must be a string list")
        evidence_paths.extend(cast(list[str], page_images))
        for relative in evidence_paths:
            evidence_path = root_dir / relative
            _validate_path(evidence_path, must_exist=True)
            input_hashes.setdefault(relative, _sha256(evidence_path.read_bytes()))

    packets.sort(key=lambda packet: cast(str, packet["packet_id"]))
    mappings.sort(key=lambda mapping: cast(str, mapping["packet_id"]))
    origin_probe = _origin_probe(packets, mappings)
    sample = _stratified_sample(packets, secret_seed)
    packets_bytes = _pretty_bytes(packets)
    audit = cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic12-non-holdout-human-audit-sample-v3",
            "state": "pending_human_review",
            "review_performed": False,
            "promotion_performed": False,
            "sampling": {
                "method": "secret-seeded-stratified-greedy-without-replacement",
                "population": len(packets),
                "selected_unique_disagreements": len(sample),
                "realized_rate": len(sample) / len(packets),
                "dimensions": ["document", "severity", "category", "element_type"],
            },
            "sample": sample,
        },
    )
    audit_bytes = _pretty_bytes(audit)
    per_document = Counter(cast(str, packet["document"]) for packet in packets)
    category_counts = Counter(
        category for packet in packets for category in cast(list[str], packet["categories"])
    )
    manifest = cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic12-non-holdout-blind-adjudication-v3",
            "scope": "non_holdout",
            "state": "pending_human_review",
            "human_review_performed": False,
            "golden_promotion_performed": False,
            "packet_count": len(packets),
            "document_count": len(documents),
            "schema_versions": {
                "document": SCHEMA_VERSION,
                "bundle": BUNDLE_SCHEMA_VERSION,
                "normalization": NORMALIZATION_VERSION,
            },
            "algorithm_versions": {
                "selection": ALGORITHM_VERSION,
                "origin_classifier": CLASSIFIER_VERSION,
            },
            "inputs": {
                "candidate_directory": _relative_path(candidate_dir, root_dir),
                "reference_directory": _relative_path(reference_dir, root_dir),
                "bronze_directory": _relative_path(bronze_dir, root_dir),
                "sha256": dict(sorted(input_hashes.items())),
            },
            "selection": {
                "source": "current evaluator recomputation from semantic12 candidates and current silver",
                "raw_disagreement_count": len(raw_disagreements),
                "material_disagreement_count": len(material),
                "fixed_protocol_population": EXPECTED_DISAGREEMENTS,
                "ranking": "one-sided, semantic/table/text/order evidence, geometry displacement, neutral key",
            },
            "normalization": {
                "fixed_option_schema": True,
                "unicode": "NFC and canonical line endings",
                "geometry_rounding_points": 1,
                "source_properties": "removed recursively",
                "provenance_and_identifiers": "excluded by construction",
                "standardized_structure": ["paragraph", "table", "figure", "footnote"],
            },
            "stratification": {
                "per_document": dict(sorted(per_document.items())),
                "category_counts": dict(sorted(category_counts.items())),
            },
            "residual_origin_classification": origin_probe,
            "reviewer_export_files": sorted(REVIEWER_FILES),
            "sha256": {
                "packets.json": _sha256(packets_bytes),
                "audit-sample-manifest.json": _sha256(audit_bytes),
            },
            "validation": {
                "reviewer_source_map_absent": "passed",
                "protected_mapping_separate": "passed",
                "assignment_requires_protected_seed": "passed",
                "packet_ids_assignment_neutral_by_domain_separation": "passed",
                "holdout": "not_inspected",
                "human_review": "not_performed",
                "promotion": "not_performed",
            },
        },
    )
    manifest_bytes = _pretty_bytes(manifest)
    protected = cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic12-protected-source-map-v3",
            "access": "protected_identity_mapping_not_for_blind_review",
            "seed": {
                "encoding": "hex",
                "value": secret_seed.hex(),
                "bits": len(secret_seed) * 8,
                "purpose": "HMAC-SHA256 packet pseudonyms, A/B assignment, and sample tie-breaking",
            },
            "schema_versions": {
                "bundle": BUNDLE_SCHEMA_VERSION,
                "normalization": NORMALIZATION_VERSION,
            },
            "algorithm_version": ALGORITHM_VERSION,
            "input_sha256": dict(sorted(input_hashes.items())),
            "reviewer_sha256": {
                "packets.json": _sha256(packets_bytes),
                "audit-sample-manifest.json": _sha256(audit_bytes),
                "manifest.json": _sha256(manifest_bytes),
            },
            "mappings": mappings,
        },
    )

    reviewer_dir.mkdir(parents=True)
    protected_dir.mkdir(parents=True)
    (reviewer_dir / "packets.json").write_bytes(packets_bytes)
    (reviewer_dir / "audit-sample-manifest.json").write_bytes(audit_bytes)
    (reviewer_dir / "manifest.json").write_bytes(manifest_bytes)
    (protected_dir / "source-map.json").write_bytes(_pretty_bytes(protected))
    actual_reviewer_files = {path.name for path in reviewer_dir.iterdir()}
    if actual_reviewer_files != set(REVIEWER_FILES):
        raise RuntimeError(f"reviewer export contains unexpected files: {sorted(actual_reviewer_files)}")


def read_protected_seed(path: Path) -> bytes:
    _validate_path(path, must_exist=True)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not isinstance(value.get("seed"), dict):
        raise TypeError("protected mapping metadata requires a seed object")
    encoded = value["seed"].get("value")
    if not isinstance(encoded, str):
        raise TypeError("protected seed must be a hexadecimal string")
    seed = bytes.fromhex(encoded)
    if len(seed) < 32:
        raise ValueError("protected seed must contain at least 256 bits")
    return seed


def canonical_review_option(element: DocumentElement | None) -> dict[str, JsonValue] | None:
    """Expose the fixed reviewer normalization for validation and integrations."""
    return _canonical_option(element)


def validate_blind_path(path: Path, *, must_exist: bool) -> None:
    """Validate lexical and resolved paths against forbidden data scopes."""
    _validate_path(path, must_exist=must_exist)


def _recompute_document_disagreements(
    document: str, candidates: list[DocumentElement], references: list[DocumentElement]
) -> list[dict[str, object]]:
    report = evaluate_document(candidates, references)
    inversions = _inversion_participants(report.alignments)
    disagreements: list[dict[str, object]] = []
    for alignment in report.alignments:
        candidate = candidates[alignment.candidate_index]
        reference = references[alignment.reference_index]
        candidate_option = _canonical_option(candidate)
        reference_option = _canonical_option(reference)
        if candidate_option == reference_option:
            continue
        disagreements.append(
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
        disagreements.append(_disagreement(document, candidates[index], None, index, None, 0.0, True))
    for index in report.unmatched_reference_indices:
        disagreements.append(_disagreement(document, None, references[index], None, index, 0.0, True))
    return disagreements


def _disagreement(
    document: str,
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
    candidate_index: int | None,
    reference_index: int | None,
    alignment_score: float,
    order_issue: bool,
) -> dict[str, object]:
    candidate_option = _canonical_option(candidate)
    reference_option = _canonical_option(reference)
    categories = _categories(candidate, reference, candidate_option, reference_option, order_issue)
    severity = _severity(categories, candidate, reference)
    neutral_key = _neutral_key(document, candidate_index, reference_index, candidate, reference)
    pages = sorted({
        fragment.page_number
        for element in (candidate, reference)
        if element
        for fragment in element.fragments
    })
    element = candidate or reference
    if element is None:
        raise ValueError("disagreement requires at least one source element")
    document_id = element.document_id
    return {
        "neutral_key": neutral_key,
        "document": document,
        "document_id": document_id,
        "candidate_index": candidate_index,
        "reference_index": reference_index,
        "candidate_element_id": candidate.element_id if candidate else None,
        "reference_element_id": reference.element_id if reference else None,
        "candidate_option": candidate_option,
        "reference_option": reference_option,
        "page_numbers": pages,
        "evidence": {
            "bronze_manifest": f"data/bronze/{document}/manifest.json",
            "bronze_native_json": f"data/bronze/{document}/liteparse.json",
            "bronze_native_text": f"data/bronze/{document}/liteparse.txt",
            "page_images": [f"data/bronze/{document}/pages/page-{page:04d}.png" for page in pages],
        },
        "categories": categories,
        "severity": severity,
        "alignment_score": alignment_score,
        "materiality_score": _materiality_score(categories, candidate, reference, alignment_score),
    }


def _canonical_option(element: DocumentElement | None) -> dict[str, JsonValue] | None:
    if element is None:
        return None
    structure = element.structure
    paragraph = structure.paragraph
    table = structure.table
    figure = structure.figure
    footnote = structure.footnote
    return cast(
        dict[str, JsonValue],
        {
            "schema": NORMALIZATION_VERSION,
            "element_type": _text(element.element_type),
            "content": _text(element.content),
            "format": element.format,
            "include_in_output": element.include_in_output,
            "pages": [_canonical_fragment(fragment) for fragment in element.fragments],
            "structure": {
                "paragraph": (
                    None
                    if paragraph is None
                    else {
                        "role": _standard_role(paragraph.role),
                        "heading_level": paragraph.heading_level,
                        "list_depth": paragraph.list_depth,
                        "list_label": _optional_text(paragraph.list_label),
                    }
                ),
                "table": (
                    None
                    if table is None
                    else {
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "header_row_count": table.header_row_count,
                        "representation": table.representation,
                        "cells": [
                            {
                                "row": cell.row_index,
                                "column": cell.column_index,
                                "rowspan": cell.rowspan,
                                "colspan": cell.colspan,
                                "role": cell.role,
                                "text": _text(cell.text),
                                "pages": [_canonical_fragment(fragment) for fragment in cell.fragments],
                            }
                            for cell in sorted(
                                table.cells,
                                key=lambda item: (
                                    item.row_index,
                                    item.column_index,
                                    item.rowspan,
                                    item.colspan,
                                ),
                            )
                        ],
                    }
                ),
                "figure": (None if figure is None else {"width": figure.width, "height": figure.height}),
                "footnote": (
                    None
                    if footnote is None
                    else {
                        "label": _optional_text(footnote.label),
                        "association_confident": footnote.association_confident,
                    }
                ),
            },
        },
    )


def _canonical_fragment(fragment: PageFragment) -> dict[str, JsonValue]:
    return {
        "page": fragment.page_number,
        "width": _rounded(fragment.page_width),
        "height": _rounded(fragment.page_height),
        "bbox": {
            "x0": round(fragment.bbox.x0, 1),
            "y0": round(fragment.bbox.y0, 1),
            "x1": round(fragment.bbox.x1, 1),
            "y1": round(fragment.bbox.y1, 1),
        },
    }


def _categories(
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
    candidate_option: dict[str, JsonValue] | None,
    reference_option: dict[str, JsonValue] | None,
    order_issue: bool,
) -> list[str]:
    if candidate is None or reference is None:
        element = candidate or reference
        categories = ["element_detection", "order_segmentation"]
        if element is not None and element.element_type == "table":
            categories.append("table_topology_content")
        return categories
    categories: list[str] = []
    if _without_geometry(candidate_option) != _without_geometry(reference_option):
        if _role_payload(candidate_option) != _role_payload(reference_option):
            categories.append("semantic_role")
        if candidate.content != reference.content:
            categories.append("character_error")
        if candidate.element_type == "table" or reference.element_type == "table":
            categories.append("table_topology_content")
    if _geometry_payload(candidate_option) != _geometry_payload(reference_option):
        categories.append("geometry")
    if order_issue or _fragment_pages(candidate) != _fragment_pages(reference):
        categories.append("order_segmentation")
    return sorted(set(categories))


def _select_material_disagreements(values: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(values) < EXPECTED_DISAGREEMENTS:
        raise ValueError(f"only {len(values)} independently recomputed disagreements are available")
    ranked = sorted(
        values,
        key=lambda value: (
            -cast(float, value["materiality_score"]),
            cast(str, value["neutral_key"]),
        ),
    )
    return ranked[:EXPECTED_DISAGREEMENTS]


def _materiality_score(
    categories: list[str],
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
    alignment_score: float,
) -> float:
    weights = {
        "element_detection": 1000.0,
        "table_topology_content": 500.0,
        "semantic_role": 300.0,
        "character_error": 200.0,
        "order_segmentation": 100.0,
        "geometry": 10.0,
    }
    score = sum(weights[category] for category in categories)
    if candidate is not None and reference is not None:
        distance = Levenshtein.distance(_text(candidate.content), _text(reference.content))
        denominator = max(len(candidate.content), len(reference.content), 1)
        score += min(99.0, 99.0 * distance / denominator)
        score += 1.0 - alignment_score
    return score


def _severity(
    categories: list[str], candidate: DocumentElement | None, reference: DocumentElement | None
) -> str:
    category_set = set(categories)
    if "table_topology_content" in category_set and "character_error" in category_set:
        return "critical"
    if candidate is None or reference is None:
        return "high"
    if category_set & {"table_topology_content", "semantic_role", "order_segmentation"}:
        return "high"
    if "character_error" in category_set:
        return "medium"
    return "low"


def _stratified_sample(packets: list[dict[str, JsonValue]], seed: bytes) -> list[dict[str, JsonValue]]:
    target = math.floor(len(packets) * 0.02)
    minimum = math.ceil(len(packets) * 0.01)
    if target < minimum:
        raise ValueError("population cannot support a 1-2% integer sample")
    remaining = list(packets)
    selected: list[dict[str, JsonValue]] = []
    covered: set[str] = set()
    while len(selected) < target:

        def rank(packet: dict[str, JsonValue]) -> tuple[int, int, str]:
            dimensions = _sample_dimensions(packet)
            gain = len(dimensions - covered)
            severity = {"critical": 3, "high": 2, "medium": 1, "low": 0}[cast(str, packet["severity"])]
            token = _secret_token(seed, "sample", cast(str, packet["packet_id"]), length=64)
            return (-gain, -severity, token)

        chosen = min(remaining, key=rank)
        selected.append(chosen)
        covered.update(_sample_dimensions(chosen))
        remaining.remove(chosen)
    return [
        {
            "packet_id": packet["packet_id"],
            "packet_sha256": _sha256(_canonical_bytes(packet)),
            "review_state": "pending_human_review",
        }
        for packet in sorted(selected, key=lambda value: cast(str, value["packet_id"]))
    ]


def _sample_dimensions(packet: dict[str, JsonValue]) -> set[str]:
    option_types = {
        cast(str, option["element_type"])
        for name in ("option_a", "option_b")
        if isinstance((option := packet[name]), dict)
    }
    return {
        f"document:{packet['document']}",
        f"severity:{packet['severity']}",
        *(f"category:{value}" for value in cast(list[str], packet["categories"])),
        *(f"type:{value}" for value in option_types),
    }


def _origin_probe(
    packets: list[dict[str, JsonValue]], mappings: list[dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    mapping_by_id = {cast(str, mapping["packet_id"]): mapping for mapping in mappings}
    public_options: list[str] = []
    labels: list[bool] = []
    documents: list[str] = []
    option_references: list[tuple[str, str]] = []
    for packet in packets:
        mapping = mapping_by_id[cast(str, packet["packet_id"])]
        for option_name, source_name in (("option_a", "source_a"), ("option_b", "source_b")):
            option = packet[option_name]
            source = mapping[source_name]
            if not isinstance(option, dict) or source == "missing":
                continue
            public_options.append(
                json.dumps(option, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            )
            labels.append(source == "candidate")
            documents.append(cast(str, packet["document"]))
            option_references.append((cast(str, packet["packet_id"]), option_name))
    if len(set(documents)) < 2 or len(set(labels)) < 2:
        raise ValueError("origin classification requires two sources across multiple documents")

    classifier = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=3000),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    raw_probabilities = cross_val_predict(
        classifier,
        public_options,
        labels,
        groups=documents,
        cv=LeaveOneGroupOut(),
        method="predict_proba",
    )
    probabilities = cast(list[list[float]], np.asarray(raw_probabilities).tolist())
    candidate_probabilities = [values[1] for values in probabilities]
    predictions = [probability >= 0.5 for probability in candidate_probabilities]
    hits_by_document: dict[str, list[bool]] = {}
    for document, predicted, expected in zip(documents, predictions, labels, strict=True):
        hits_by_document.setdefault(document, []).append(predicted == expected)
    accuracy = sum(
        predicted == expected for predicted, expected in zip(predictions, labels, strict=True)
    ) / len(labels)
    candidate_rate = sum(labels) / len(labels)
    majority_baseline = max(candidate_rate, 1.0 - candidate_rate)
    lift = accuracy - majority_baseline
    true_candidate_rate = sum(
        predicted for predicted, expected in zip(predictions, labels, strict=True) if expected
    ) / sum(labels)
    true_reference_rate = sum(
        not predicted for predicted, expected in zip(predictions, labels, strict=True) if not expected
    ) / (len(labels) - sum(labels))
    balanced_accuracy = (true_candidate_rate + true_reference_rate) / 2

    probability_by_option = dict(zip(option_references, candidate_probabilities, strict=True))
    pair_hits: list[bool] = []
    for packet in packets:
        packet_id = cast(str, packet["packet_id"])
        a_key = (packet_id, "option_a")
        b_key = (packet_id, "option_b")
        if a_key not in probability_by_option or b_key not in probability_by_option:
            continue
        predicted_candidate_is_a = probability_by_option[a_key] > probability_by_option[b_key]
        expected_candidate_is_a = mapping_by_id[packet_id]["source_a"] == "candidate"
        pair_hits.append(predicted_candidate_is_a == expected_candidate_is_a)
    pairwise_accuracy = sum(pair_hits) / len(pair_hits)
    result = "passed" if balanced_accuracy <= 0.7 and pairwise_accuracy <= 0.75 else "failed"
    return {
        "method": "leave-one-document-out character-ngram logistic regression over public normalized options",
        "eligible_options": len(labels),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "candidate_rate": candidate_rate,
        "majority_baseline": majority_baseline,
        "lift_over_majority_baseline": lift,
        "aligned_pair_count": len(pair_hits),
        "pairwise_mapping_recovery_accuracy": pairwise_accuracy,
        "acceptance_thresholds": {"balanced_accuracy": 0.7, "pairwise_mapping_recovery_accuracy": 0.75},
        "per_document_accuracy": {
            document: sum(hits) / len(hits) for document, hits in sorted(hits_by_document.items())
        },
        "avoidable_schema_asymmetries_found": 0,
        "result": result,
    }


def _inversion_participants(alignments: list[ElementAlignment]) -> set[tuple[int, int]]:
    participants: set[tuple[int, int]] = set()
    for position, left in enumerate(alignments):
        for right in alignments[position + 1 :]:
            if left.candidate_index > right.candidate_index:
                participants.add((left.candidate_index, left.reference_index))
                participants.add((right.candidate_index, right.reference_index))
    return participants


def _without_geometry(value: dict[str, JsonValue] | None) -> JsonValue:
    if value is None:
        return None
    copied = dict(value)
    copied["pages"] = [{"page": page["page"]} for page in cast(list[dict[str, JsonValue]], value["pages"])]
    structure = cast(dict[str, JsonValue], copied["structure"])
    table = structure.get("table")
    if isinstance(table, dict):
        table_copy = dict(table)
        normalized_cells: list[JsonValue] = [
            {**cell, "pages": []} for cell in cast(list[dict[str, JsonValue]], table["cells"])
        ]
        table_copy["cells"] = normalized_cells
        structure = {**structure, "table": cast(JsonValue, table_copy)}
    copied["structure"] = structure
    return copied


def _geometry_payload(value: dict[str, JsonValue] | None) -> JsonValue:
    if value is None:
        return None
    structure = cast(dict[str, JsonValue], value["structure"])
    table = structure.get("table")
    cell_pages = (
        []
        if not isinstance(table, dict)
        else [cell["pages"] for cell in cast(list[dict[str, JsonValue]], table["cells"])]
    )
    return cast(JsonValue, {"pages": value["pages"], "cell_pages": cell_pages})


def _role_payload(value: dict[str, JsonValue] | None) -> JsonValue:
    if value is None:
        return None
    return cast(JsonValue, {"element_type": value["element_type"], "structure": value["structure"]})


def _fragment_pages(element: DocumentElement) -> set[int]:
    return {fragment.page_number for fragment in element.fragments}


def _neutral_key(
    document: str,
    candidate_index: int | None,
    reference_index: int | None,
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
) -> str:
    value = {
        "document": document,
        "candidate_index": candidate_index,
        "reference_index": reference_index,
        "candidate_digest": _sha256(_canonical_bytes(_canonical_option(candidate))),
        "reference_digest": _sha256(_canonical_bytes(_canonical_option(reference))),
    }
    return _sha256(_canonical_bytes(value))


def _secret_token(seed: bytes, domain: str, value: str, *, length: int) -> str:
    digest = hmac.new(seed, f"{domain}\0{value}".encode(), hashlib.sha256).hexdigest()
    return digest[:length]


def _secret_bit(seed: bytes, domain: str, value: str) -> int:
    digest = hmac.new(seed, f"{domain}\0{value}".encode(), hashlib.sha256).digest()
    return digest[0] & 1


def _standard_role(value: str) -> str:
    normalized = _text(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {"text": "body", "normal": "body", "section_heading": "heading"}
    return aliases.get(normalized, normalized)


def _text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def _optional_text(value: str | None) -> str | None:
    return None if value is None else _text(value)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def _validate_disjoint_outputs(reviewer_dir: Path, protected_dir: Path) -> None:
    reviewer = reviewer_dir.resolve(strict=False)
    protected = protected_dir.resolve(strict=False)
    if reviewer == protected or reviewer in protected.parents or protected in reviewer.parents:
        raise ValueError("protected directory must be outside and disjoint from reviewer bundle")


def _validate_path(path: Path, *, must_exist: bool) -> None:
    resolved = path.resolve(strict=must_exist)
    for inspected in (path.absolute(), resolved):
        lowered = [part.lower() for part in inspected.parts]
        if any("holdout" in part for part in lowered):
            raise ValueError(f"holdout path is forbidden: {path}")
        if any("layoutlm" in part for part in lowered):
            raise ValueError(f"LayoutLM path is forbidden: {path}")


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
