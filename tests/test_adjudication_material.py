from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from app.pdf2md.adjudication_material import reproduce_semantic11_blind_material

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "data/adjudication-semantic11-blind-material-v1"
BUNDLE = ROOT / "data/adjudication-semantic11-blind-material-v2"
SEMANTIC11 = ROOT / "data/benchmark-batch-semantic11/candidates"
SEMANTIC12 = ROOT / "data/benchmark-batch-semantic12/candidates"
REFERENCES = ROOT / "data/silver-cycle2"


def _load(path: Path) -> object:
    return json.loads(path.read_bytes())


def test_checked_in_bundle_is_exactly_reproducible(tmp_path: Path) -> None:
    reproduced = tmp_path / "reproduced"
    reproduce_semantic11_blind_material(
        SOURCE,
        SEMANTIC12,
        REFERENCES,
        reproduced,
        root_dir=ROOT,
    )

    expected = {path.name: path.read_bytes() for path in BUNDLE.iterdir() if path.is_file()}
    actual = {path.name: path.read_bytes() for path in reproduced.iterdir() if path.is_file()}
    assert actual == expected


def test_bundle_retains_material_sample_state_and_semantic12_compatibility() -> None:
    manifest = cast(dict[str, object], _load(BUNDLE / "manifest.json"))
    packets = cast(list[dict[str, object]], _load(BUNDLE / "packets.json"))
    audit = cast(dict[str, object], _load(BUNDLE / "audit-sample-manifest.json"))
    source_map = cast(dict[str, object], _load(BUNDLE / "source-map.json"))
    mappings = cast(list[dict[str, object]], source_map["mappings"])
    sample = cast(list[dict[str, object]], audit["sample"])

    assert len(packets) == len(mappings) == 336
    assert len({cast(str, packet["packet_id"]) for packet in packets}) == 336
    assert len({cast(str, mapping["packet_id"]) for mapping in mappings}) == 336
    assert 0.01 <= len(sample) / len(packets) <= 0.02
    assert len({cast(str, entry["packet_id"]) for entry in sample}) == len(sample) == 6
    assert manifest["scope"] == "non_holdout"
    assert manifest["state"] == "pending_human_review"
    assert manifest["human_review_performed"] is False
    assert manifest["golden_promotion_performed"] is False
    assert manifest["semantic_batch"] == "semantic12-identical-candidates"
    assert source_map["access"] == "protected_identity_mapping_not_for_blind_review"
    packet_by_id = {cast(str, packet["packet_id"]): packet for packet in packets}
    for entry in sample:
        packet = packet_by_id[cast(str, entry["packet_id"])]
        canonical = json.dumps(packet, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        assert hashlib.sha256(canonical).hexdigest() == entry["packet_sha256"]

    hashes = cast(dict[str, str], manifest["sha256"])
    for filename, expected in hashes.items():
        assert hashlib.sha256((BUNDLE / filename).read_bytes()).hexdigest() == expected
    for semantic11_path in sorted(SEMANTIC11.glob("*.parquet")):
        semantic12_path = SEMANTIC12 / semantic11_path.name
        assert semantic11_path.read_bytes() == semantic12_path.read_bytes()


def test_every_nested_option_field_resists_identity_and_provenance_classification() -> None:
    packets = cast(list[dict[str, object]], _load(BUNDLE / "packets.json"))
    source_map = cast(dict[str, object], _load(BUNDLE / "source-map.json"))
    mappings = cast(list[dict[str, object]], source_map["mappings"])
    mapping_by_packet = {cast(str, mapping["packet_id"]): mapping for mapping in mappings}
    forbidden_exact_fields = {
        "asset_path",
        "caption_element_id",
        "linked_element_ids",
        "reference_element_ids",
        "sha256",
        "source_item_ids",
        "source_note",
    }
    provenance_markers = ("bronze", "liteparse", "native pdf", "pymupdf")

    def attack(value: object, identity_tokens: set[str], path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                lowered = key.lower()
                assert lowered not in forbidden_exact_fields, ".".join((*path, key))
                assert not lowered.endswith(("_element_id", "_element_ids")), ".".join((*path, key))
                attack(nested, identity_tokens, (*path, key))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                attack(nested, identity_tokens, (*path, str(index)))
        elif isinstance(value, str):
            lowered = value.lower()
            assert not any(marker in lowered for marker in provenance_markers), ".".join(path)
            assert not any(token and token in value for token in identity_tokens), ".".join(path)
            assert not (len(value) == 64 and all(char in "0123456789abcdef" for char in value)), ".".join(
                path
            )
            assert not value.startswith(("data/", "pdf://", "file://")), ".".join(path)

    for packet in packets:
        mapping = mapping_by_packet[cast(str, packet["packet_id"])]
        identity_tokens = {
            value for key, value in mapping.items() if key.endswith("_element_id") and isinstance(value, str)
        }
        for option_name in ("option_a", "option_b"):
            option = packet[option_name]
            if option is not None:
                attack(option, identity_tokens, (option_name,))


def test_reproduction_rejects_holdout_and_layoutlm_paths(tmp_path: Path) -> None:
    for forbidden in (tmp_path / "secret-holdout", tmp_path / "academic-layoutlm"):
        with pytest.raises(ValueError, match="forbidden"):
            reproduce_semantic11_blind_material(
                forbidden,
                SEMANTIC12,
                REFERENCES,
                tmp_path / "output",
                root_dir=ROOT,
            )
