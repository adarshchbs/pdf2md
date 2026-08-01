from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import app.pdf2md.candidate_batch as candidate_batch
from app.pdf2md.candidate_batch import (
    _artifacts_match,  # pyright: ignore[reportPrivateUsage]
    _load_latest,  # pyright: ignore[reportPrivateUsage]
    _worker,  # pyright: ignore[reportPrivateUsage]
    sha256_file,
)


def test_load_latest_uses_final_outcome_per_document(tmp_path: Path) -> None:
    status = tmp_path / "status.jsonl"
    records = [
        {"id": "a", "status": "error"},
        {"id": "b", "status": "success"},
        {"id": "a", "status": "success"},
    ]
    status.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")

    latest = _load_latest(status)

    assert latest["a"]["status"] == "success"
    assert latest["b"]["status"] == "success"


def test_artifacts_match_verifies_names_sizes_and_hashes(tmp_path: Path) -> None:
    artifacts: dict[str, dict[str, str | int]] = {}
    for name, content in (
        ("doc.parquet", b"elements"),
        ("doc.source-items.parquet", b"sources"),
        ("doc.md", b"text"),
    ):
        path = tmp_path / name
        path.write_bytes(content)
        artifacts[name] = {"sha256": sha256_file(path), "size_bytes": len(content)}
    record = {"artifacts": artifacts}

    assert _artifacts_match(tmp_path, record)
    (tmp_path / "doc.md").write_bytes(b"changed")
    assert not _artifacts_match(tmp_path, record)


def test_artifacts_match_rejects_unsafe_names(tmp_path: Path) -> None:
    record = {
        "artifacts": {
            "../outside": {"sha256": "0" * 64, "size_bytes": 0},
            "doc.parquet": {"sha256": "0" * 64, "size_bytes": 0},
            "doc.md": {"sha256": "0" * 64, "size_bytes": 0},
        }
    }
    assert not _artifacts_match(tmp_path, record)


def test_worker_preserves_preexisting_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    output = tmp_path / "output"
    output.mkdir()
    parquet = output / "doc.parquet"
    parquet.write_bytes(b"existing")

    record = _worker({
        "id": "doc",
        "split": "train",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "output_dir": str(output),
    })

    assert record["status"] == "error"
    assert record["error_type"] == "FileExistsError"
    assert parquet.read_bytes() == b"existing"


def test_worker_times_out_stalled_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")

    def stall(_source: Path) -> None:
        time.sleep(1)

    monkeypatch.setattr(candidate_batch, "extract_document_with_catalog", stall)
    record = _worker({
        "id": "doc",
        "split": "train",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "output_dir": str(tmp_path / "output"),
        "timeout_seconds": 0.01,
    })

    assert record["status"] == "error"
    assert record["error_type"] == "TimeoutError"
