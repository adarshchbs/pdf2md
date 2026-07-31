from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from app.pdf2md.blind_v3 import (
    REVIEWER_FILES,
    build_semantic12_blind_v3,
    canonical_review_option,
    read_protected_seed,
    validate_blind_path,
)
from app.pdf2md.schema import read_document_elements

ROOT = Path(__file__).parents[1]
CANDIDATES = ROOT / "data/benchmark-batch-semantic12/candidates"
REFERENCES = ROOT / "data/silver-cycle2"
BRONZE = ROOT / "data/bronze"
FIXED_SEED = bytes.fromhex("94" * 32)


def _load(path: Path) -> object:
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    parent = tmp_path_factory.mktemp("blind-v3")
    reviewer = parent / "reviewer"
    protected = parent / "protected"
    build_semantic12_blind_v3(
        CANDIDATES,
        REFERENCES,
        BRONZE,
        reviewer,
        protected,
        root_dir=ROOT,
        seed=FIXED_SEED,
    )
    return reviewer, protected


def test_reviewer_export_is_minimal_and_protected_map_is_separate(
    generated: tuple[Path, Path],
) -> None:
    reviewer, protected = generated
    assert {path.name for path in reviewer.iterdir()} == REVIEWER_FILES
    assert {path.name for path in protected.iterdir()} == {"source-map.json"}
    assert not any("source-map" in path.name for path in reviewer.rglob("*"))

    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    packets = cast(list[dict[str, object]], _load(reviewer / "packets.json"))
    audit = cast(dict[str, object], _load(reviewer / "audit-sample-manifest.json"))
    sample = cast(list[dict[str, object]], audit["sample"])
    assert len(packets) == 336
    assert len({packet["packet_id"] for packet in packets}) == 336
    assert len(sample) == len({entry["packet_id"] for entry in sample}) == 6
    assert 0.01 <= len(sample) / len(packets) <= 0.02
    assert manifest["state"] == "pending_human_review"
    assert manifest["human_review_performed"] is False
    assert manifest["golden_promotion_performed"] is False
    probe = cast(dict[str, object], manifest["residual_origin_classification"])
    assert probe["result"] == "passed"
    assert cast(float, probe["lift_over_majority_baseline"]) <= 0.1
    assert cast(float, probe["balanced_accuracy"]) <= 0.7
    assert cast(float, probe["pairwise_mapping_recovery_accuracy"]) <= 0.75


def test_exact_regeneration_requires_separately_supplied_seed(
    generated: tuple[Path, Path], tmp_path: Path
) -> None:
    reviewer, protected = generated
    reproduced_reviewer = tmp_path / "reviewer"
    reproduced_protected = tmp_path / "protected"
    build_semantic12_blind_v3(
        CANDIDATES,
        REFERENCES,
        BRONZE,
        reproduced_reviewer,
        reproduced_protected,
        root_dir=ROOT,
        seed=read_protected_seed(protected / "source-map.json"),
    )
    expected = {path.name: path.read_bytes() for path in reviewer.iterdir()}
    actual = {path.name: path.read_bytes() for path in reproduced_reviewer.iterdir()}
    assert actual == expected
    assert (reproduced_protected / "source-map.json").read_bytes() == (
        protected / "source-map.json"
    ).read_bytes()


def test_packet_id_parity_and_public_fields_do_not_recover_assignment(
    generated: tuple[Path, Path],
) -> None:
    reviewer, protected = generated
    packets = cast(list[dict[str, object]], _load(reviewer / "packets.json"))
    source_map = cast(dict[str, object], _load(protected / "source-map.json"))
    mappings = cast(list[dict[str, object]], source_map["mappings"])
    mapping_by_id = {cast(str, mapping["packet_id"]): mapping for mapping in mappings}

    parity_hits = 0
    public_hash_hits = 0
    eligible = 0
    for packet in packets:
        packet_id = cast(str, packet["packet_id"])
        mapping = mapping_by_id[packet_id]
        if mapping["source_a"] == "missing" or mapping["source_b"] == "missing":
            continue
        candidate_is_a = mapping["source_a"] == "candidate"
        parity_hits += (int(packet_id[-1], 16) % 2 == 0) == candidate_is_a
        public = json.dumps(packet, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        public_hash_hits += (hashlib.sha256(public).digest()[0] % 2 == 0) == candidate_is_a
        eligible += 1
    assert eligible > 100
    assert abs(parity_hits / eligible - 0.5) <= 0.1
    assert abs(public_hash_hits / eligible - 0.5) <= 0.1


def test_option_schema_is_fixed_and_recursive_provenance_is_absent(
    generated: tuple[Path, Path],
) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, object]], _load(reviewer / "packets.json"))
    option_keys: set[tuple[str, ...]] = set()
    forbidden_fields = {
        "annotation",
        "asset_path",
        "caption_element_id",
        "classification_reasons",
        "element_id",
        "linked_element_ids",
        "properties",
        "reference_element_ids",
        "sha256",
        "source_item_ids",
    }

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_fields.isdisjoint(value)
            for nested in value.values():
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)

    for packet in packets:
        for name in ("option_a", "option_b"):
            option = packet[name]
            if isinstance(option, dict):
                option_keys.add(tuple(sorted(option)))
                inspect(option)
    assert option_keys == {
        ("content", "element_type", "format", "include_in_output", "pages", "schema", "structure")
    }


def test_normalization_removes_source_conventions_and_rounds_geometry() -> None:
    element = read_document_elements(next(iter(sorted(CANDIDATES.glob("*.parquet")))))[0]
    option = cast(dict[str, JsonValue], canonical_review_option(element))
    pages = cast(list[dict[str, JsonValue]], option["pages"])
    for page in pages:
        bbox = cast(dict[str, float], page["bbox"])
        assert all(value == round(value, 1) for value in bbox.values())
    structure = cast(dict[str, JsonValue], option["structure"])
    assert set(structure) == {"paragraph", "table", "figure", "footnote"}


def test_path_validation_uses_resolved_target_without_creating_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ordinary = tmp_path / "ordinary"
    original_resolve = Path.resolve

    def mocked_resolve(path: Path, *, strict: bool = False) -> Path:
        if path == ordinary:
            return tmp_path / "secret-holdout" / "resolved"
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", mocked_resolve)
    with pytest.raises(ValueError, match="holdout path is forbidden"):
        validate_blind_path(ordinary, must_exist=False)


def test_direct_forbidden_paths_and_nested_outputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holdout path is forbidden"):
        validate_blind_path(tmp_path / "holdout-data", must_exist=False)
    with pytest.raises(ValueError, match="LayoutLM path is forbidden"):
        validate_blind_path(tmp_path / "layoutlm-data", must_exist=False)
    with pytest.raises(ValueError, match="outside and disjoint"):
        build_semantic12_blind_v3(
            CANDIDATES,
            REFERENCES,
            BRONZE,
            tmp_path / "bundle",
            tmp_path / "bundle" / "protected",
            root_dir=ROOT,
            seed=FIXED_SEED,
        )
