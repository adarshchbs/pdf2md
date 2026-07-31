from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from app.pdf2md.adjudication import BlindElement, blind_element_payload
from app.pdf2md.schema import DocumentElement, read_document_elements

_BUNDLE_FILES = ("packets.json", "source-map.json", "audit-sample-manifest.json", "manifest.json")


def reproduce_semantic11_blind_material(
    source_dir: Path,
    semantic12_candidate_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    *,
    root_dir: Path,
) -> None:
    """Reproduce the fixed bundle from v1 selection metadata and semantic12-identical candidates."""
    for path in (source_dir, semantic12_candidate_dir, reference_dir, output_dir, root_dir):
        _reject_forbidden_path(path)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    for filename in _BUNDLE_FILES:
        if not (source_dir / filename).is_file():
            raise FileNotFoundError(source_dir / filename)

    old_manifest = _read_object(source_dir / "manifest.json")
    old_packets = _read_object_list(source_dir / "packets.json")
    old_source_map = _read_object_list(source_dir / "source-map.json")
    old_audit = _read_object(source_dir / "audit-sample-manifest.json")
    if old_manifest.get("scope") != "non_holdout":
        raise ValueError("source bundle must be explicitly non-holdout")
    if old_manifest.get("state") != "pending_human_review":
        raise ValueError("source bundle must remain pending human review")
    if old_manifest.get("golden_promotion_performed") is not False:
        raise ValueError("source bundle must not have promoted golden data")

    packet_by_id = _unique_by_string_key(old_packets, "packet_id")
    map_by_id = _unique_by_string_key(old_source_map, "packet_id")
    if set(packet_by_id) != set(map_by_id):
        raise ValueError("packet/source-map packet_id values are not bijective")
    if len(old_packets) != 336:
        raise ValueError(f"expected 336 material disagreements, found {len(old_packets)}")

    documents = sorted({cast(str, mapping["document"]) for mapping in old_source_map})
    manifest_source = old_manifest.get("source")
    if not isinstance(manifest_source, dict):
        raise TypeError("manifest source must be an object")
    old_candidate_directory = manifest_source.get("candidate_directory")
    if not isinstance(old_candidate_directory, str):
        raise TypeError("manifest candidate_directory must be a string")
    semantic11_candidate_dir = root_dir / old_candidate_directory

    candidates: dict[str, list[DocumentElement]] = {}
    references: dict[str, list[DocumentElement]] = {}
    for document in documents:
        old_candidate_path = semantic11_candidate_dir / f"{document}.parquet"
        candidate_path = semantic12_candidate_dir / f"{document}.parquet"
        reference_path = reference_dir / f"{document}.parquet"
        if _sha256(old_candidate_path.read_bytes()) != _sha256(candidate_path.read_bytes()):
            raise ValueError(f"semantic12 candidate is not byte-identical to semantic11: {document}")
        candidates[document] = read_document_elements(candidate_path)
        references[document] = read_document_elements(reference_path)

    rebuilt_packets: list[dict[str, JsonValue]] = []
    protected_mappings: list[dict[str, JsonValue]] = []
    for old_packet in old_packets:
        packet_id = cast(str, old_packet["packet_id"])
        mapping = map_by_id[packet_id]
        document = cast(str, mapping["document"])
        candidate = _mapped_element(candidates[document], mapping, "candidate")
        reference = _mapped_element(references[document], mapping, "reference")
        option_by_source = {
            "candidate": blind_element_payload(candidate),
            "reference": blind_element_payload(reference),
            "missing": None,
        }
        packet = dict(old_packet)
        packet["option_a"] = _dump_blind(option_by_source[_source(mapping, "source_a")])
        packet["option_b"] = _dump_blind(option_by_source[_source(mapping, "source_b")])
        rebuilt_packets.append(packet)
        protected_mappings.append(dict(mapping))

    audit = dict(old_audit)
    audit["artifact_id"] = "semantic11-non-holdout-human-audit-manifest-v2"
    audit_sample = audit.get("sample")
    if not isinstance(audit_sample, list):
        raise TypeError("audit sample must be a list")
    selected_ids: set[str] = set()
    for entry in audit_sample:
        if not isinstance(entry, dict) or not isinstance(entry.get("packet_id"), str):
            raise TypeError("audit sample entries require packet_id")
        packet_id = entry["packet_id"]
        if not isinstance(packet_id, str):
            raise TypeError("audit sample packet_id must be a string")
        if packet_id in selected_ids:
            raise ValueError("audit sample packet_id values must be unique")
        selected_ids.add(packet_id)
        entry["packet_sha256"] = _sha256(_canonical_bytes(packet_by_id_from(rebuilt_packets, packet_id)))
    sampling = audit.get("sampling_metadata")
    if not isinstance(sampling, dict):
        raise TypeError("audit sampling_metadata must be an object")
    realized = len(selected_ids) / len(rebuilt_packets)
    if not 0.01 <= realized <= 0.02:
        raise ValueError(f"audit sampling rate is outside 1-2%: {realized}")
    sampling["population_element_count"] = len(rebuilt_packets)
    sampling["selected_unique_disagreements"] = len(selected_ids)
    sampling["realized_sampling_rate"] = realized

    source_map = cast(
        dict[str, JsonValue],
        {
            "artifact_id": "semantic11-non-holdout-protected-source-map-v2",
            "access": "protected_identity_mapping_not_for_blind_review",
            "mappings": protected_mappings,
        },
    )
    packets_bytes = _pretty_bytes(rebuilt_packets)
    source_map_bytes = _pretty_bytes(source_map)
    audit_bytes = _pretty_bytes(audit)

    manifest = dict(old_manifest)
    manifest["artifact_id"] = "semantic11-non-holdout-blind-adjudication-v2"
    manifest["semantic_batch"] = "semantic12-identical-candidates"
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise TypeError("manifest source must be an object")
    source["candidate_directory"] = _relative_path(semantic12_candidate_dir, root_dir)
    source["selection_metadata_directory"] = _relative_path(source_dir, root_dir)
    manifest["sha256"] = {
        "audit-sample-manifest.json": _sha256(audit_bytes),
        "packets.json": _sha256(packets_bytes),
        "source-map.json": _sha256(source_map_bytes),
    }
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise TypeError("manifest validation must be an object")
    validation.update({
        "candidate_identity_absent_from_packets": "passed",
        "deterministic_regeneration": "passed",
        "golden_promotion": "not_performed",
        "holdout": "not_inspected",
        "human_review_status": "not_performed",
        "nested_identity_and_provenance_absent_from_options": "passed",
        "option_asset_identity_removed": "passed",
        "option_identifier_fields_removed": "passed",
        "option_source_notes_removed": "passed",
        "review_state": "pending_human_review",
        "semantic12_candidate_compatibility": "byte_identical_to_semantic11",
    })
    manifest_bytes = _pretty_bytes(manifest)

    output_dir.mkdir(parents=True)
    (output_dir / "packets.json").write_bytes(packets_bytes)
    (output_dir / "source-map.json").write_bytes(source_map_bytes)
    (output_dir / "audit-sample-manifest.json").write_bytes(audit_bytes)
    (output_dir / "manifest.json").write_bytes(manifest_bytes)


def _mapped_element(
    elements: list[DocumentElement], mapping: dict[str, JsonValue], source: str
) -> DocumentElement | None:
    index = mapping.get(f"{source}_index")
    expected_id = mapping.get(f"{source}_element_id")
    if index is None:
        if expected_id is not None:
            raise ValueError(f"{source} mapping has an ID without an index")
        return None
    if not isinstance(index, int) or isinstance(index, bool) or not isinstance(expected_id, str):
        raise TypeError(f"{source} mapping index/ID has invalid types")
    element = elements[index]
    if element.element_id != expected_id:
        raise ValueError(f"{source} mapping does not match source element at index {index}")
    return element


def _source(mapping: dict[str, JsonValue], key: str) -> str:
    value = mapping.get(key)
    if value not in {"candidate", "reference", "missing"}:
        raise ValueError(f"invalid source-map value for {key}: {value!r}")
    return value


def _dump_blind(value: BlindElement | None) -> JsonValue:
    if value is None:
        return None
    return cast(JsonValue, value.model_dump(mode="json"))


def packet_by_id_from(packets: list[dict[str, JsonValue]], packet_id: str) -> dict[str, JsonValue]:
    for packet in packets:
        if packet.get("packet_id") == packet_id:
            return packet
    raise ValueError(f"audit sample references unknown packet_id: {packet_id}")


def _read_object(path: Path) -> dict[str, JsonValue]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return cast(dict[str, JsonValue], value)


def _read_object_list(path: Path) -> list[dict[str, JsonValue]]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TypeError(f"expected list of JSON objects: {path}")
    return cast(list[dict[str, JsonValue]], value)


def _unique_by_string_key(values: list[dict[str, JsonValue]], key: str) -> dict[str, dict[str, JsonValue]]:
    result: dict[str, dict[str, JsonValue]] = {}
    for value in values:
        identifier = value.get(key)
        if not isinstance(identifier, str):
            raise TypeError(f"{key} must be a string")
        if identifier in result:
            raise ValueError(f"duplicate {key}: {identifier}")
        result[identifier] = value
    return result


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _reject_forbidden_path(path: Path) -> None:
    lowered_parts = {part.lower() for part in path.parts}
    if any("holdout" in part for part in lowered_parts):
        raise ValueError(f"holdout path is forbidden: {path}")
    if any("layoutlm" in part for part in lowered_parts):
        raise ValueError(f"LayoutLM path is forbidden: {path}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
