from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from app.pdf2md import blind_v5
from app.pdf2md.blind_v5 import (
    ATTACK_THRESHOLDS,
    PROTECTED_FILES,
    REVIEWER_FILES,
    build_blind_v5,
    canonical_review_option,
    material_categories,
    read_protected_seed,
    validate_blind_path,
)
from app.pdf2md.schema import read_document_elements

ROOT = Path(__file__).parents[1]
CANDIDATES = ROOT / "data/benchmark-batch-semantic13/candidates"
DETERMINISM = ROOT / "data/benchmark-batch-semantic13-determinism-rerun/candidates"
REFERENCES = ROOT / "data/silver-cycle2"
BRONZE = ROOT / "data/bronze"
FIXED_SEED = bytes.fromhex("a5" * 32)


def _load(path: Path) -> object:
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    parent = tmp_path_factory.mktemp("blind-v5")
    reviewer = parent / "reviewer"
    protected = parent / "protected"
    build_blind_v5(
        CANDIDATES,
        DETERMINISM,
        REFERENCES,
        BRONZE,
        reviewer,
        protected,
        root_dir=ROOT,
        seed=FIXED_SEED,
    )
    return reviewer, protected


def test_reviewer_is_only_secret_sample_and_protected_is_complete(
    generated: tuple[Path, Path],
) -> None:
    reviewer, protected = generated
    assert {path.name for path in reviewer.iterdir()} == REVIEWER_FILES
    assert {path.name for path in protected.iterdir()} == PROTECTED_FILES
    assert stat.S_IMODE(protected.stat().st_mode) == 0o700
    assert all(stat.S_IMODE((protected / name).stat().st_mode) == 0o600 for name in PROTECTED_FILES)

    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    ledger = cast(dict[str, object], _load(protected / "full-ledger.json"))
    counts = cast(dict[str, int], ledger["counts"])
    records = cast(list[dict[str, object]], ledger["records"])
    population = cast(int, manifest["protected_material_population"])
    assert len(packets) < population
    assert 0.01 <= len(packets) / population <= 0.02
    assert counts["raw"] == counts["material"] + counts["nonmaterial"] == len(records)
    assert counts["material"] == population
    assert {record["material"] for record in records} == {False, True}
    assert all(record["categories"] or record["exclusion_reason"] for record in records)


def test_all_semantic_categories_survive_text_equality() -> None:
    candidate = read_document_elements(next(iter(sorted(CANDIDATES.glob("*.parquet")))))[0]
    fragment = candidate.fragments[0]
    moved_box = fragment.bbox.model_copy(
        update={
            "x0": fragment.bbox.x0 + (fragment.page_width or fragment.bbox.x1) * 0.7,
            "x1": fragment.bbox.x1 + (fragment.page_width or fragment.bbox.x1) * 0.7,
        }
    )
    moved = candidate.model_copy(update={"fragments": [fragment.model_copy(update={"bbox": moved_box})]})
    categories = material_categories(candidate, moved)
    assert candidate.content == moved.content
    assert "material_geometry" in categories

    changed_type = candidate.model_copy(update={"element_type": "custom_role_type"})
    assert "declared_type" in material_categories(candidate, changed_type)


def test_presence_is_neutral_and_sample_origins_are_exactly_balanced(
    generated: tuple[Path, Path],
) -> None:
    reviewer, protected = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    source_map = cast(dict[str, object], _load(protected / "source-map.json"))
    mappings = cast(list[dict[str, JsonValue]], source_map["sample_mappings"])
    presence = [packet for packet in packets if packet["mode"] == "presence"]
    paired = [packet for packet in packets if packet["mode"] == "paired_ab"]
    assert presence and paired
    assert all("observed_element" in packet for packet in presence)
    assert all("option_a" not in packet and "option_b" not in packet for packet in presence)
    assert all(packet["allowed_decisions"] == ["keep", "remove", "reconstruct"] for packet in presence)
    origins = [mapping["presence_origin"] for mapping in mappings if mapping["mode"] == "presence"]
    assert origins.count("candidate") == origins.count("reference") > 0
    assert all(
        packet["allowed_decisions"] == ["prefer_a", "prefer_b", "equivalent", "reconstruct"]
        for packet in paired
    )


def test_standardized_options_expose_adjudicable_semantics_without_identity(
    generated: tuple[Path, Path],
) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    forbidden = {"element_id", "document_id", "annotation", "source_item_ids", "bbox", "properties"}
    for packet in packets:
        options = [
            value
            for key in ("option_a", "option_b", "observed_element")
            if isinstance((value := packet.get(key)), dict)
        ]
        for option in options:
            assert set(option) == {"kind", "text", "declared_type", "role", "geometry", "structure"}
            assert forbidden.isdisjoint(option)
            assert isinstance(option["geometry"], list)
            assert isinstance(option["role"], dict)
            assert isinstance(option["structure"], dict)

    element = read_document_elements(next(iter(sorted(CANDIDATES.glob("*.parquet")))))[0]
    option = canonical_review_option(element)
    assert option is not None
    assert {"declared_type", "role", "geometry", "structure"}.issubset(option)


def test_attacks_use_raw_recovery_and_all_sample_packets(generated: tuple[Path, Path]) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    attacks = cast(dict[str, object], manifest["adversarial_acceptance"])
    population = cast(dict[str, int], attacks["population"])
    models = cast(dict[str, dict[str, float]], attacks["models"])
    assert attacks["result"] == "passed"
    assert population["reviewer_packets"] == len(packets)
    assert population["excluded"] == 0
    assert {
        "public_id_nibbles_random_forest",
        "public_id_nibbles_extra_trees",
        "public_hash_nibbles_random_forest",
        "public_hash_nibbles_extra_trees",
        "structural_content_random_forest",
        "structural_content_extra_trees",
    } == set(models)
    assert all("raw_mapping_accuracy" in result for result in models.values())
    worst = cast(dict[str, float], attacks["worst_recovery"])
    public = cast(dict[str, float], attacks["public_attacks"])
    assert all(worst[name] <= ATTACK_THRESHOLDS[name] for name in worst)
    assert all(public[name] <= ATTACK_THRESHOLDS[name] for name in public)
    assert "no balanced substitute" in cast(str, attacks["raw_recovery_policy"])


def test_honest_coverage_bronze_path_pins_and_reproduction(
    generated: tuple[Path, Path], tmp_path: Path
) -> None:
    reviewer, protected = generated
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    inputs = cast(dict[str, object], manifest["inputs"])
    pins = cast(dict[str, object], manifest["implementation_pins"])
    sampling = cast(dict[str, object], manifest["sampling"])
    coverage = cast(dict[str, object], sampling["actual_coverage"])
    assert inputs["bronze_directory"] == "data/bronze"
    assert pins["evaluation_function_sha256"]
    assert pins["selection_function_sha256"]
    assert cast(dict[str, str], pins["packages"])["scikit-learn"]
    assert set(cast(list[str], coverage["modes"])) == {"paired_ab", "presence"}
    assert "only listed actual coverage" in cast(str, sampling["coverage_declaration"])

    reproduced_reviewer = tmp_path / "reviewer"
    reproduced_protected = tmp_path / "protected"
    build_blind_v5(
        CANDIDATES,
        DETERMINISM,
        REFERENCES,
        BRONZE,
        reproduced_reviewer,
        reproduced_protected,
        root_dir=ROOT,
        seed=read_protected_seed(protected / "seed.json"),
    )
    assert {path.name: path.read_bytes() for path in reproduced_reviewer.iterdir()} == {
        path.name: path.read_bytes() for path in reviewer.iterdir()
    }
    assert {path.name: path.read_bytes() for path in reproduced_protected.iterdir()} == {
        path.name: path.read_bytes() for path in protected.iterdir()
    }


def test_failed_gate_leaves_no_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reviewer = tmp_path / "reviewer"
    protected = tmp_path / "protected"
    monkeypatch.setattr(
        blind_v5, "_run_adversarial_acceptance", lambda _packets, _mappings: {"result": "failed"}
    )
    with pytest.raises(ValueError, match="pinned adversarial acceptance failed"):
        build_blind_v5(
            CANDIDATES,
            DETERMINISM,
            REFERENCES,
            BRONZE,
            reviewer,
            protected,
            root_dir=ROOT,
            seed=FIXED_SEED,
        )
    assert not reviewer.exists()
    assert not protected.exists()


def test_resolved_path_mocks_are_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    original_resolve = Path.resolve

    def mocked_resolve(path: Path, *, strict: bool = False) -> Path:
        if path == ordinary:
            return tmp_path / "secret-holdout" / "resolved"
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", mocked_resolve)
    with pytest.raises(ValueError, match="holdout path is forbidden"):
        validate_blind_path(ordinary, must_exist=False)
    with pytest.raises(ValueError, match="outside and disjoint"):
        build_blind_v5(
            CANDIDATES,
            DETERMINISM,
            REFERENCES,
            BRONZE,
            tmp_path / "bundle",
            tmp_path / "bundle" / "protected",
            root_dir=ROOT,
            seed=FIXED_SEED,
        )
