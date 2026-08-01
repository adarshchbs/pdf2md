from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl
import pymupdf
import pytest

from benchmarks.adapters import get_adapter
from benchmarks.runner import run_adapter_attempts
from benchmarks.smoke import acceptance_failures, create_synthetic_pdf, run_smoke


def test_synthetic_pdf_contains_required_born_digital_content(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.pdf"

    create_synthetic_pdf(path)

    with pymupdf.open(path) as document:
        assert document.page_count == 1
        page = document[0]
        text = str(page.get_text("text"))
        drawings = page.get_drawings()
    assert "Offline Extraction Smoke" in text
    assert "Café résumé" in text
    assert "1,234.50" in text
    assert {"Item", "Units", "Price", "Alpha", "Beta"}.issubset(text.split())
    assert drawings


def test_acceptance_rejects_deterministic_output_missing_expected_text(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.pdf"
    create_synthetic_pdf(source)
    parser_results = run_adapter_attempts(get_adapter("our_parser"), source, tmp_path / "parser", attempts=2)
    baseline_results = run_adapter_attempts(
        get_adapter("pymupdf_text"), source, tmp_path / "baseline", attempts=2
    )
    first = parser_results[0]
    bad_page = first.pages[0].model_copy(update={"markdown": "deterministic but wrong"})
    parser_results[0] = replace(first, pages=(bad_page,))

    failures = acceptance_failures({
        "our_parser": parser_results,
        "pymupdf_text": baseline_results,
    })

    assert any("omitted required synthetic text" in failure for failure in failures)


def test_smoke_writes_complete_development_evidence(tmp_path: Path) -> None:
    output = tmp_path / "smoke"

    manifest, failures = run_smoke(output)

    assert failures == []
    assert manifest["reproduction_status"] == "NOT_COMPARABLE"
    assert manifest["same_harness"] is False
    assert manifest["network"] == {"policy": "denied", "use_observed": "not_verified"}
    assert manifest["models"] == {
        "configured": False,
        "downloads_policy": "denied",
        "downloads_observed": "not_verified",
    }
    assert manifest["isolation"]["network_enforcement"] == "not_enforced"  # type: ignore[index]
    for filename in (
        "manifest.json",
        "benchmark_summary.json",
        "metrics.parquet",
        "benchmark_report.md",
    ):
        assert (output / filename).is_file()
    for tool_id in ("our_parser", "pymupdf_text"):
        attempts = manifest["attempts"][tool_id]  # type: ignore[index]
        assert len(attempts) == 3
        assert {attempt["status"] for attempt in attempts} == {"success"}
        for attempt in attempts:
            assert (output / attempt["raw_path"]).is_file()
            assert (output / attempt["canonical_path"]).is_file()
    assert [attempt["table_count"] for attempt in manifest["attempts"]["our_parser"]] == [1, 1, 1]  # type: ignore[index]
    assert [attempt["table_count"] for attempt in manifest["attempts"]["pymupdf_text"]] == [0, 0, 0]  # type: ignore[index]

    metrics = pl.read_parquet(output / "metrics.parquet")
    assert set(metrics.get_column("tool_id")) == {"our_parser", "pymupdf_text"}
    summary = json.loads((output / "benchmark_summary.json").read_text())
    assert summary["passed"] is True
    assert all(item["status"].startswith("SKIPPED_") for item in summary["external_tools"])

    with pytest.raises(FileExistsError):
        run_smoke(output)
