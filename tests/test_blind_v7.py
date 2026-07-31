from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from app.pdf2md import blind_v7
from app.pdf2md.blind_v7 import PROTECTED_FILES, REVIEWER_ROOT_FILES, build_blind_v7, read_protected_seed

ROOT = Path(__file__).parents[1]
CANDIDATES = ROOT / "data/benchmark-batch-semantic13/candidates"
DETERMINISM = ROOT / "data/benchmark-batch-semantic13-determinism-rerun/candidates"
REFERENCES = ROOT / "data/silver-cycle2"
BRONZE = ROOT / "data/bronze"
FIXED_SEED = bytes.fromhex("b7" * 32)


def _load(path: Path) -> object:
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    parent = tmp_path_factory.mktemp("blind-v7")
    reviewer, protected = parent / "reviewer", parent / "protected"
    build_blind_v7(
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


def test_reviewer_export_is_standalone_and_accurately_inventoried(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    assert {path.name for path in reviewer.iterdir() if path.is_file()} == REVIEWER_ROOT_FILES
    assert (reviewer / "evidence").is_dir()
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    inventory = cast(list[dict[str, str]], manifest["bundle_inventory"])
    declared = {item["path"] for item in inventory}
    actual = {path.relative_to(reviewer).as_posix() for path in reviewer.rglob("*") if path.is_file()}
    assert declared == actual
    assert len(actual) > 3
    assert "isolated package" in cast(str, manifest["required_execution_environment"])
    assert "without access" in cast(str, manifest["required_execution_environment"])
    assert {path.name for path in protected.iterdir()} == PROTECTED_FILES
    assert stat.S_IMODE(protected.stat().st_mode) == 0o700
    assert all(stat.S_IMODE((protected / name).stat().st_mode) == 0o600 for name in PROTECTED_FILES)


def test_packets_use_only_neutral_bundle_local_evidence(generated: tuple[Path, Path]) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    assert len(packets) == 6
    for packet in packets:
        assert "document" not in packet and "page_numbers" not in packet
        assert cast(str, packet["packet_id"]).startswith("pkt-")
        evidence = cast(dict[str, JsonValue], packet["evidence"])
        paths = [
            cast(str, evidence["native_text_geometry_excerpt"]),
            *cast(list[str], evidence["page_images"]),
        ]
        assert all(path.startswith(f"evidence/{packet['packet_id']}/") for path in paths)
        assert all((reviewer / path).is_file() for path in paths)
        excerpt = cast(dict[str, object], _load(reviewer / paths[0]))
        assert excerpt["schema"] == "sanitized-native-text-geometry-v1"
        for page in cast(list[dict[str, object]], excerpt["pages"]):
            assert set(page) == {"items", "local_page", "text"}
            for item in cast(list[dict[str, object]], page["items"]):
                assert set(item) == {"height", "left", "text", "top", "width"}


def test_reading_order_sequences_are_distinct_semantic_local_alternatives(
    generated: tuple[Path, Path],
) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    order_packets = [
        packet for packet in packets if "reading_order_segmentation" in cast(list[str], packet["categories"])
    ]
    assert order_packets
    for packet in order_packets:
        order = cast(dict[str, object], packet["order_adjudication"])
        left = cast(list[dict[str, object]], order["sequence_a"])
        right = cast(list[dict[str, object]], order["sequence_b"])
        assert left != right
        assert left and right
        for item in [*left, *right]:
            assert {"relative_position", "text", "declared_type", "role", "relative_geometry"} == set(item)
        assert "index" not in json.dumps(order).casefold()


def test_reconstruction_attack_and_all_generation_gates_pass(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    validation = cast(dict[str, object], manifest["validation"])
    assert validation == {
        "standalone_evidence": "passed",
        "category_adjudicability": "passed",
        "reconstruction_metadata_attack": "passed",
        "review_state": "pending_human_review",
    }
    provenance = cast(dict[str, object], _load(protected / "provenance.json"))
    attack = cast(dict[str, object], provenance["reconstruction_attack"])
    assert attack["result"] == "passed"
    assert attack["metadata_hits"] == []
    assert cast(list[str], attack["exact_known_file_matches"])
    assert cast(dict[str, object], provenance["adjudicability"])["result"] == "passed"
    assert cast(dict[str, object], provenance["design_acceptance"])["result"] == "passed"


def test_sampling_ledger_and_provenance_are_protected_and_honest(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    audit = cast(dict[str, object], _load(reviewer / "audit-sample-manifest.json"))
    sampling = cast(dict[str, object], audit["sampling"])
    assert 0.01 <= cast(float, sampling["realized_rate"]) <= 0.02
    strata = cast(list[dict[str, object]], sampling["strata"])
    assert len(strata) == 4
    for stratum in strata:
        probability = cast(float, stratum["inclusion_probability"])
        assert probability == cast(int, stratum["selected"]) / cast(int, stratum["population"])
        assert cast(float, stratum["inclusion_weight"]) == pytest.approx(1 / probability)
    ledger = cast(dict[str, object], _load(protected / "full-ledger.json"))
    assert cast(dict[str, int], ledger["counts"]) == {"raw": 425, "material": 346, "nonmaterial": 79}
    assert len(cast(list[object], ledger["records"])) == 425
    provenance = cast(dict[str, object], _load(protected / "provenance.json"))
    runner = cast(dict[str, object], provenance["runner_v3_provenance_machinery"])
    closure = cast(dict[str, object], runner["full_project_package_closure"])
    closure_paths = {item["path"] for item in cast(list[dict[str, object]], closure["files"])}
    assert "app/pdf2md/blind_v7.py" in closure_paths
    assert {"pyproject.toml", "uv.lock"} <= closure_paths
    assert cast(dict[str, object], runner["installed_distribution_records"])
    assert cast(dict[str, object], runner["runtime"])["installed_distributions"]
    assert len(cast(list[str], provenance["binary_trust_limits"])) >= 3


def test_exact_seed_reproduction(generated: tuple[Path, Path], tmp_path: Path) -> None:
    reviewer, protected = generated
    reproduced_reviewer = tmp_path / "reviewer"
    reproduced_protected = tmp_path / "protected"
    build_blind_v7(
        CANDIDATES,
        DETERMINISM,
        REFERENCES,
        BRONZE,
        reproduced_reviewer,
        reproduced_protected,
        root_dir=ROOT,
        seed=read_protected_seed(protected / "seed.json"),
    )
    original_files = {
        path.relative_to(reviewer).as_posix(): path.read_bytes()
        for path in reviewer.rglob("*")
        if path.is_file()
    }
    reproduced_files = {
        path.relative_to(reproduced_reviewer).as_posix(): path.read_bytes()
        for path in reproduced_reviewer.rglob("*")
        if path.is_file()
    }
    assert reproduced_files == original_files
    assert {path.name: path.read_bytes() for path in reproduced_protected.iterdir()} == {
        path.name: path.read_bytes() for path in protected.iterdir()
    }


def test_failed_gate_publishes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reviewer, protected = tmp_path / "reviewer", tmp_path / "protected"
    monkeypatch.setattr(blind_v7, "_adjudicability_checks", lambda _packets: {"result": "failed"})
    with pytest.raises(ValueError, match="standalone evidence or adjudicability gate failed"):
        build_blind_v7(
            CANDIDATES,
            DETERMINISM,
            REFERENCES,
            BRONZE,
            reviewer,
            protected,
            root_dir=ROOT,
            seed=FIXED_SEED,
        )
    assert not reviewer.exists() and not protected.exists()
