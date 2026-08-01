from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from app.pdf2md import blind_v6
from app.pdf2md.blind_v6 import (
    PROTECTED_FILES,
    REVIEWER_FILES,
    build_blind_v6,
    read_protected_seed,
    validate_blind_path,
)

ROOT = Path(__file__).parents[1]
CANDIDATES = ROOT / "data/benchmark-batch-semantic13/candidates"
DETERMINISM = ROOT / "data/benchmark-batch-semantic13-determinism-rerun/candidates"
REFERENCES = ROOT / "data/silver-cycle2"
BRONZE = ROOT / "data/bronze"
FIXED_SEED = bytes.fromhex("b6" * 32)


def _load(path: Path) -> object:
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    parent = tmp_path_factory.mktemp("blind-v6")
    reviewer, protected = parent / "reviewer", parent / "protected"
    build_blind_v6(
        CANDIDATES, DETERMINISM, REFERENCES, BRONZE, reviewer, protected, root_dir=ROOT, seed=FIXED_SEED
    )
    return reviewer, protected


def test_exact_exports_full_ledger_and_protected_modes(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    assert {path.name for path in reviewer.iterdir()} == REVIEWER_FILES
    assert {path.name for path in protected.iterdir()} == PROTECTED_FILES
    assert stat.S_IMODE(protected.stat().st_mode) == 0o700
    assert all(stat.S_IMODE((protected / name).stat().st_mode) == 0o600 for name in PROTECTED_FILES)
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    ledger = cast(dict[str, object], _load(protected / "full-ledger.json"))
    counts = cast(dict[str, int], ledger["counts"])
    assert counts == {"raw": 425, "material": 346, "nonmaterial": 79}
    assert len(cast(list[object], ledger["records"])) == 425
    assert len(packets) == 6
    assert 0.01 <= len(packets) / cast(int, manifest["protected_material_population"]) <= 0.02


def test_every_sampled_category_has_passing_adjudicability_check(generated: tuple[Path, Path]) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    adjudicability = cast(dict[str, object], manifest["adjudicability_checks"])
    checks = cast(list[dict[str, object]], adjudicability["checks"])
    expected = {
        (packet["packet_id"], category)
        for packet in packets
        for category in cast(list[str], packet["categories"])
    }
    assert adjudicability["result"] == "passed"
    assert {(check["packet_id"], check["category"]) for check in checks} == expected
    assert all(check["passed"] is True for check in checks)


def test_reading_order_packets_have_neutral_alternates_and_context(generated: tuple[Path, Path]) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    order_packets = [
        packet for packet in packets if "reading_order_segmentation" in cast(list[str], packet["categories"])
    ]
    assert order_packets
    forbidden = {"candidate_index", "reference_index", "source_a", "source_b", "presence_origin"}
    for packet in order_packets:
        order = cast(dict[str, object], packet["order_adjudication"])
        assert set(order) == {"assignment_a", "assignment_b", "context_window", "prompt", "provenance"}
        assert order["provenance"] == "omitted"
        for name in ("assignment_a", "assignment_b"):
            assignment = cast(dict[str, object], order[name])
            assert {"position", "previous", "next"} == set(assignment)
            assert assignment["previous"] is not None or assignment["next"] is not None
        assert forbidden.isdisjoint(json.dumps(packet))


def test_design_invariants_are_acceptance_and_attacks_are_honest_diagnostics(
    generated: tuple[Path, Path],
) -> None:
    reviewer, _ = generated
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    design = cast(dict[str, object], manifest["deterministic_design_acceptance"])
    assert design["result"] == "passed"
    assert all(cast(dict[str, bool], design["checks"]).values())
    attacks = cast(dict[str, object], manifest["descriptive_attacks"])
    assert "diagnostic_only" in cast(str, attacks["acceptance_role"])
    assert "insufficient" in cast(str, attacks["statistical_power"])
    sample = cast(dict[str, object], attacks["reviewer_sample"])
    assert "result" not in sample and "thresholds" not in sample
    assert "insufficient" in cast(str, sample["statistical_power"])
    for result in cast(dict[str, dict[str, object]], sample["models"]).values():
        for metric in result.values():
            interval = cast(list[float], cast(dict[str, object], metric)["exact_95_ci"])
            assert 0 <= interval[0] <= interval[1] <= 1


def test_forced_stratified_sampling_records_probabilities_and_weights(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    audit = cast(dict[str, object], _load(reviewer / "audit-sample-manifest.json"))
    sampling = cast(dict[str, object], audit["sampling"])
    assert "forced stratified" in cast(str, sampling["method"])
    assert "not uniform" in cast(str, sampling["method"])
    strata = cast(list[dict[str, object]], sampling["inclusion_probabilities"])
    assert len(strata) == 4
    assert sum(cast(int, stratum["selected"]) for stratum in strata) == 6
    for stratum in strata:
        probability = cast(float, stratum["probability"])
        assert probability == cast(int, stratum["selected"]) / cast(int, stratum["population"])
        assert cast(float, stratum["weight"]) == pytest.approx(1 / probability)
    source_map = cast(dict[str, object], _load(protected / "source-map.json"))
    mappings = cast(list[dict[str, object]], source_map["full_material_mappings"])
    assert all("inclusion_probability" in mapping and "sampling_weight" in mapping for mapping in mappings)


def test_all_69_ledger_evidence_paths_and_dependency_closure_are_pinned(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    inputs = cast(dict[str, object], manifest["inputs"])
    assert inputs["bronze_directory"] == "data/bronze"
    assert inputs["distinct_ledger_evidence_path_count"] == 69
    ledger = cast(dict[str, object], _load(protected / "full-ledger.json"))
    paths: set[str] = set()
    for record in cast(list[dict[str, object]], ledger["records"]):
        evidence = cast(dict[str, object], record["evidence"])
        paths.update(value for value in evidence.values() if isinstance(value, str))
        paths.update(cast(list[str], evidence["page_images"]))
    hashes = cast(dict[str, str], inputs["sha256"])
    assert len(paths) == 69 and paths <= set(hashes)
    pins = cast(dict[str, object], manifest["implementation_pins"])
    assert set(cast(dict[str, str], pins["full_module_sha256"])) == {
        "app.pdf2md.blind_v6",
        "app.pdf2md.evaluation",
        "app.pdf2md.schema",
        "app.pdf2md.tables",
    }
    assert cast(dict[str, object], pins["runtime"])["python_version"]
    assert {"pyproject.toml", "uv.lock"} == set(cast(dict[str, str], pins["config_sha256"]))
    assert len(cast(dict[str, object], pins["package_distribution_pins"])) == 6


def test_exact_reproduction_and_seed_map_colocation_disclosure(
    generated: tuple[Path, Path], tmp_path: Path
) -> None:
    reviewer, protected = generated
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    separation = cast(dict[str, object], manifest["separation"])
    assert "co-located" in cast(str, separation["seed_map_colocation_limitation"])
    assert "not globally atomic" in cast(str, separation["publication_guarantee"])
    reproduced_reviewer, reproduced_protected = tmp_path / "reviewer", tmp_path / "protected"
    build_blind_v6(
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


def test_failed_design_gate_leaves_no_final_destinations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reviewer, protected = tmp_path / "reviewer", tmp_path / "protected"
    monkeypatch.setattr(blind_v6, "_deterministic_design_acceptance", lambda _p, _m, _s: {"result": "failed"})
    with pytest.raises(ValueError, match="deterministic generation gates failed"):
        build_blind_v6(
            CANDIDATES, DETERMINISM, REFERENCES, BRONZE, reviewer, protected, root_dir=ROOT, seed=FIXED_SEED
        )
    assert not reviewer.exists() and not protected.exists()


def test_forbidden_paths_remain_blocked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holdout path is forbidden"):
        validate_blind_path(tmp_path / "holdout", must_exist=False)
    with pytest.raises(ValueError, match="LayoutLM path is forbidden"):
        validate_blind_path(tmp_path / "LayoutLM", must_exist=False)
