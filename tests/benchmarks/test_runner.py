from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.canonical import (
    AdapterRun,
    CanonicalPage,
    CanonicalProvenance,
    PageSize,
    RuntimeStats,
)
from benchmarks.runner import attempts_are_deterministic, run_adapter_attempts

_SHA = "a" * 64


class _SuccessfulAdapter:
    adapter_id = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def process(self, pdf_path: Path) -> AdapterRun:
        del pdf_path
        self.calls += 1
        runtime = RuntimeStats(wall_seconds=float(self.calls))
        page = CanonicalPage(
            document_id="fixture",
            page_index=0,
            page_size=PageSize(width=100, height=100, unit="point"),
            markdown="fixture",
            elements=[],
            tables=[],
            figures=[],
            runtime=runtime,
            provenance=CanonicalProvenance(
                tool_name="fake",
                tool_version="1",
                mode="test",
                input_sha256=_SHA,
                config_sha256="b" * 64,
            ),
        )
        return AdapterRun(
            status="success",
            pages=[page],
            raw_records=[{"call": self.calls}],
            runtime=runtime,
        )


class _CrashingAdapter:
    adapter_id = "crasher"

    def process(self, pdf_path: Path) -> AdapterRun:
        del pdf_path
        raise RuntimeError("intentional test crash")


class _MutatingCrashingAdapter:
    adapter_id = "mutating-crasher"

    def process(self, pdf_path: Path) -> AdapterRun:
        pdf_path.write_bytes(b"mutated")
        raise RuntimeError("crashed after mutation")


class _MalformedFailureAdapter:
    adapter_id = "malformed-failure"

    def process(self, pdf_path: Path) -> AdapterRun:
        del pdf_path
        runtime = RuntimeStats(wall_seconds=0)
        page = CanonicalPage(
            document_id="fixture",
            page_index=0,
            page_size=PageSize(width=100, height=100, unit="point"),
            markdown="fixture",
            elements=[],
            tables=[],
            figures=[],
            runtime=runtime,
            provenance=CanonicalProvenance(
                tool_name="fake",
                tool_version="1",
                mode="test",
                input_sha256=_SHA,
                config_sha256="b" * 64,
            ),
        )
        return AdapterRun(
            status="failed",
            pages=[page, page],
            raw_records=[{"malformed": True}],
            runtime=runtime,
            error="tool failed with malformed pages",
        )


def test_runner_retains_every_attempt_and_ignores_runtime_for_determinism(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source.write_bytes(b"immutable bytes")

    results = run_adapter_attempts(
        _SuccessfulAdapter(),
        source,
        tmp_path / "runs",
        attempts=3,
        timeout_seconds=10,
    )

    assert len(results) == 3
    assert attempts_are_deterministic(results)
    assert {result.runtime_seconds for result in results} == {1.0, 2.0, 3.0}
    assert len({result.normalized_sha256 for result in results}) == 1
    assert all(result.raw_path.is_file() for result in results)
    assert all(result.canonical_path.is_file() for result in results)


def test_runner_keeps_crashes_in_denominator_with_empty_canonical_artifact(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source.write_bytes(b"immutable bytes")

    results = run_adapter_attempts(
        _CrashingAdapter(),
        source,
        tmp_path / "runs",
        attempts=3,
        timeout_seconds=10,
    )

    assert [result.status for result in results] == ["crashed"] * 3
    assert not attempts_are_deterministic(results)
    assert all(result.canonical_path.read_bytes() == b"" for result in results)
    assert all("intentional test crash" in result.raw_path.read_text() for result in results)


def test_runner_restores_input_after_mutating_crash(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    original = b"immutable bytes"
    source.write_bytes(original)

    results = run_adapter_attempts(
        _MutatingCrashingAdapter(),
        source,
        tmp_path / "runs",
        attempts=2,
        timeout_seconds=10,
    )

    assert [result.status for result in results] == ["invalid", "invalid"]
    assert all("mutated the immutable input PDF" in (result.error or "") for result in results)
    assert source.read_bytes() == original


def test_runner_retains_malformed_failed_pages_as_invalid_attempts(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source.write_bytes(b"immutable bytes")

    results = run_adapter_attempts(
        _MalformedFailureAdapter(),
        source,
        tmp_path / "runs",
        attempts=2,
        timeout_seconds=10,
    )

    assert [result.status for result in results] == ["invalid", "invalid"]
    assert all(result.canonical_path.read_bytes() == b"" for result in results)
    assert all("canonical serialization failed" in (result.error or "") for result in results)


def test_runner_requires_a_new_destination(tmp_path: Path) -> None:
    source = tmp_path / "input.pdf"
    source.write_bytes(b"immutable bytes")
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        run_adapter_attempts(_SuccessfulAdapter(), source, destination)
