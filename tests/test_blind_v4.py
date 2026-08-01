from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue

from app.pdf2md import blind_v4
from app.pdf2md.blind_v4 import (
    ATTACK_THRESHOLDS,
    PROTECTED_FILES,
    REVIEWER_FILES,
    build_blind_v4,
    canonical_review_option,
    read_protected_seed,
    validate_blind_path,
)
from app.pdf2md.schema import read_document_elements

ROOT = Path(__file__).parents[1]
CANDIDATES = ROOT / "data/benchmark-batch-semantic13/candidates"
DETERMINISM = ROOT / "data/benchmark-batch-semantic13-determinism-rerun/candidates"
REFERENCES = ROOT / "data/silver-cycle2"
BRONZE = ROOT / "data/bronze"
FIXED_SEED = bytes.fromhex("a4" * 32)


def _load(path: Path) -> object:
    return json.loads(path.read_bytes())


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    parent = tmp_path_factory.mktemp("blind-v4")
    reviewer = parent / "reviewer"
    protected = parent / "protected"
    build_blind_v4(
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


def test_v4_exports_are_separate_complete_and_permissioned(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    assert {path.name for path in reviewer.iterdir()} == REVIEWER_FILES
    assert {path.name for path in protected.iterdir()} == PROTECTED_FILES
    assert stat.S_IMODE(protected.stat().st_mode) == 0o700
    assert all(stat.S_IMODE((protected / name).stat().st_mode) == 0o600 for name in PROTECTED_FILES)
    assert not any(path.name in PROTECTED_FILES for path in reviewer.rglob("*"))


def test_materiality_is_threshold_based_and_all_other_records_are_ledgered(
    generated: tuple[Path, Path],
) -> None:
    reviewer, protected = generated
    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    selection = cast(dict[str, object], manifest["selection"])
    ledger = cast(dict[str, object], _load(protected / "exclusion-ledger.json"))
    records = cast(list[dict[str, object]], ledger["records"])
    assert "no rank or top-N" in cast(str, selection["policy"])
    assert "fixed_protocol_population" not in selection
    assert cast(int, selection["raw_disagreement_count"]) == cast(
        int, selection["material_disagreement_count"]
    ) + len(records)
    assert ledger["record_count"] == len(records) > 0
    assert len({record["neutral_key"] for record in records}) == len(records)
    assert all(record["exclusion_reason"] for record in records)


def test_modes_randomization_and_full_attack_population(generated: tuple[Path, Path]) -> None:
    reviewer, protected = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    source_map = cast(dict[str, object], _load(protected / "source-map.json"))
    mappings = cast(list[dict[str, JsonValue]], source_map["mappings"])
    by_id = {cast(str, mapping["packet_id"]): mapping for mapping in mappings}
    one_sided = [packet for packet in packets if packet["mode"] == "presence_absence"]
    paired = [packet for packet in packets if packet["mode"] == "paired_ab"]
    assert one_sided and paired
    assert {packet["option_a"] is None for packet in one_sided} == {False, True}
    assert all((packet["option_a"] is None) != (packet["option_b"] is None) for packet in one_sided)
    assert all(
        {
            cast(str, by_id[cast(str, packet["packet_id"])]["source_a"]),
            cast(str, by_id[cast(str, packet["packet_id"])]["source_b"]),
        }
        == {"candidate", "reference"}
        for packet in paired
    )
    assert len(mappings) == len(packets)

    manifest = cast(dict[str, object], _load(reviewer / "manifest.json"))
    attacks = cast(dict[str, object], manifest["adversarial_acceptance"])
    population = cast(dict[str, object], attacks["population"])
    assert attacks["result"] == "passed"
    assert population["packets"] == len(packets)
    assert population["one_sided"] == len(one_sided)
    assert population["excluded_from_attacks"] == 0
    worst = cast(dict[str, float], attacks["worst_model"])
    public = cast(dict[str, float], attacks["public_attacks"])
    assert all(worst[name] <= ATTACK_THRESHOLDS[name] for name in worst)
    assert all(public[name] <= ATTACK_THRESHOLDS[name] for name in public)


def test_options_remove_role_geometry_and_table_schema(generated: tuple[Path, Path]) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    forbidden = {
        "annotation",
        "bbox",
        "cells",
        "column_count",
        "element_type",
        "format",
        "heading_level",
        "include_in_output",
        "properties",
        "representation",
        "role",
        "row_count",
    }
    schemas: set[tuple[str, ...]] = set()
    for packet in packets:
        for name in ("option_a", "option_b"):
            option = packet[name]
            if not isinstance(option, dict):
                continue
            schemas.add(tuple(sorted(option)))
            assert forbidden.isdisjoint(option)
            summary = cast(dict[str, JsonValue], option["summary"])
            assert set(summary) == {"line_count", "non_whitespace_characters", "token_count"}
    assert schemas == {("kind", "summary", "text")}

    element = read_document_elements(next(iter(sorted(CANDIDATES.glob("*.parquet")))))[0]
    option = canonical_review_option(element)
    assert option is not None and set(option) == {"kind", "summary", "text"}


def test_reviewer_sample_is_honest_and_covers_required_dimensions(
    generated: tuple[Path, Path],
) -> None:
    reviewer, _ = generated
    packets = cast(list[dict[str, JsonValue]], _load(reviewer / "packets.json"))
    audit = cast(dict[str, object], _load(reviewer / "audit-sample-manifest.json"))
    sample = cast(list[dict[str, object]], audit["sample"])
    sampling = cast(dict[str, object], audit["sampling"])
    coverage = cast(dict[str, list[str]], sampling["actual_coverage"])
    assert len(sample) == len({entry["packet_id"] for entry in sample})
    assert 0.01 <= len(sample) / len(packets) <= 0.02
    assert "medium" in coverage["severities"]
    assert {"figure", "footnote"}.issubset(coverage["element_kinds"])
    assert set(coverage["modes"]) == {"paired_ab", "presence_absence"}
    assert "only actual_coverage is claimed" in cast(str, sampling["coverage_declaration"])


def test_exact_reproduction_uses_separate_seed_file(generated: tuple[Path, Path], tmp_path: Path) -> None:
    reviewer, protected = generated
    reproduced_reviewer = tmp_path / "reviewer"
    reproduced_protected = tmp_path / "protected"
    build_blind_v4(
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


def test_failed_acceptance_creates_no_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reviewer = tmp_path / "reviewer"
    protected = tmp_path / "protected"
    monkeypatch.setattr(
        blind_v4, "_run_adversarial_acceptance", lambda _packets, _mappings: {"result": "failed"}
    )
    with pytest.raises(ValueError, match="pinned adversarial acceptance failed"):
        build_blind_v4(
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


def test_resolved_path_validation_and_disjoint_outputs(
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
    with pytest.raises(ValueError, match="outside and disjoint"):
        build_blind_v4(
            CANDIDATES,
            DETERMINISM,
            REFERENCES,
            BRONZE,
            tmp_path / "bundle",
            tmp_path / "bundle" / "protected",
            root_dir=ROOT,
            seed=FIXED_SEED,
        )
