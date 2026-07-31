import gc
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import app.pdf2md.corpus_partition as corpus_partition_module
from app.pdf2md.cli import cli
from app.pdf2md.corpus_partition import (
    RATIOS,
    _adapt_checkpoint_document,  # pyright: ignore[reportPrivateUsage]
    _atomic_components,  # pyright: ignore[reportPrivateUsage]
    _atomic_groups,  # pyright: ignore[reportPrivateUsage]
    _atomic_keys,  # pyright: ignore[reportPrivateUsage]
    _checkpoint_kind,  # pyright: ignore[reportPrivateUsage]
    _enrich,  # pyright: ignore[reportPrivateUsage]
    _fixed_parent_output,  # pyright: ignore[reportPrivateUsage]
    _normalize_url_value,  # pyright: ignore[reportPrivateUsage]
    _partition_policy,  # pyright: ignore[reportPrivateUsage]
    _read_json,  # pyright: ignore[reportPrivateUsage]
    _read_records,  # pyright: ignore[reportPrivateUsage]
    _SnapshotStore,  # pyright: ignore[reportPrivateUsage]
    _strata_summary,  # pyright: ignore[reportPrivateUsage]
    _summary,  # pyright: ignore[reportPrivateUsage]
    _validate_checkpoint,  # pyright: ignore[reportPrivateUsage]
    _validated_revision_sidecars,  # pyright: ignore[reportPrivateUsage]
    _verify,  # pyright: ignore[reportPrivateUsage]
    build_corpus_partition,
    document_targets,
    verify_corpus_partition_descriptor,
)


def _sha(index: int) -> str:
    return hashlib.sha256(f"document-{index}".encode()).hexdigest()


def _normalize(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def _refresh_deduplication_keys(document: dict[str, object]) -> None:
    issuer = document.get("issuer")
    producer = str(issuer or document.get("producer"))
    document["producer"] = producer
    normalized_title = _normalize(document["title"])
    title_family = _normalize(re.sub(r"\b(?:19|20)\d{2}\b", "", normalized_title))
    template_family = f"publisher-template:{_normalize(producer)}|{title_family}"
    document["template_family"] = template_family
    document["deduplication_keys"] = {
        "sha256": document["sha256"],
        "normalized_url": _normalize_url_value(str(document["source_url"])),
        "normalized_title": normalized_title,
        "normalized_publisher": _normalize(producer),
        "template_family": template_family,
    }


def _document(index: int, *, title: str | None = None, issuer: str | None = None) -> dict[str, object]:
    document: dict[str, object] = {
        "id": f"doc-{index}",
        "title": title or f"Document {index}",
        "local_path": f"staging/doc-{index}.pdf",
        "sha256": _sha(index),
        "page_count": index + 1,
        "size_bytes": 1_000 + index,
        "category": "report" if index % 2 else "form",
        "status": "accepted",
        "provenance": "Synthetic partition-test fixture.",
        "producer": issuer or f"producer {index} example",
        "region": "test-region",
        "year": 2026,
        "language": "en",
        "layout_signals": {"synthetic": True},
        "source_url": f"https://producer-{index}.example/doc.pdf",
        "accuracy_inspection_status": "uninspected",
    }
    if issuer is not None:
        document["issuer"] = issuer
    _refresh_deduplication_keys(document)
    return document


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    corpus_path = root / "data/corpus/manifest.json"
    partition_path = root / "data/corpus/partition-v1.json"
    staging_path = root / "staging/accepted.json"
    corpus_documents = [_document(index) for index in range(8)]
    corpus_documents.extend([
        _document(100, title="LayoutLM: Pre-training of Text and Layout"),
        _document(101, title="LayoutLMv2: Multi-modal Pre-training"),
    ])
    _write_json(corpus_path, {"documents": corpus_documents})
    _write_json(
        partition_path,
        {
            "documents": [
                {**corpus_documents[0], "split": "dev"},
                {**corpus_documents[1], "split": "validation"},
                {**corpus_documents[2], "split": "holdout"},
            ]
        },
    )
    new_documents = [_document(index, issuer=f"Issuer {index}") for index in range(20, 32)]
    _write_json(staging_path, {"documents": new_documents})
    return corpus_path, partition_path, staging_path


def _difficult_checkpoint(root: Path, staging_path: Path) -> Path:
    stage = root / "data/corpus-expansion/v3-difficult"
    source_path = stage / "source_manifest.jsonl"
    accepted_path = stage / "accepted_manifest.jsonl"
    rejected_path = stage / "rejected_manifest.jsonl"
    summary_path = stage / "summary.json"
    documents = json.loads(staging_path.read_text(encoding="utf-8"))["documents"]
    sources: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for document in documents:
        source = {
            "slug": document["id"],
            "title": document["title"],
            "producer": document["producer"],
            "edition": document["id"],
            "source_url": document["source_url"],
            "source_kind": "test",
            "stratum": document["category"],
            "template_family": document["template_family"],
            "assignment": "unassigned",
        }
        sources.append(source)
        accepted.append({
            **source,
            "status": "accepted",
            "local_path": document["local_path"],
            "sha256": document["sha256"],
            "size_bytes": document["size_bytes"],
            "page_count": document["page_count"],
            "native_text_reliable": True,
            "useful_structural_signals": True,
            "validation_note": "Metadata-only uninspected checkpoint fixture.",
            "sampled_detected_table_count": 0,
            "rotated_page_count": 0,
            "landscape_page_count": 0,
            "form_field_count": 0,
        })
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
    accepted_path.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    rejected_path.write_text("", encoding="utf-8")
    _write_json(
        summary_path,
        {
            "task": 43,
            "source_count": len(sources),
            "accepted_count": len(accepted),
            "rejected_count": 0,
            "accepted_page_count": sum(int(row["page_count"]) for row in accepted),
            "accepted_size_bytes": sum(int(row["size_bytes"]) for row in accepted),
            "scope_guard": "No accuracy, bronze, silver, validation, or holdout artifacts were inspected.",
        },
    )
    return accepted_path


def _financial_checkpoint(root: Path) -> Path:
    stage = root / "v3-financial-r3"
    source_path = stage / "source_manifest.json"
    accepted_path = stage / "accepted_manifest.jsonl"
    rejected_path = stage / "rejected_manifest.jsonl"
    summary_path = stage / "summary.json"
    sources = [
        {
            "id": f"financial-{index}",
            "source_url": f"https://financial.example/download?document={index}",
            "canonical_url": f"https://financial.example/report?document={index}",
            "title": f"Financial {index}",
            "issuer": "Issuer",
            "year": 2026,
        }
        for index in range(2)
    ]
    _write_json(
        source_path,
        {"candidate_status": "unassigned", "accuracy_inspected": False, "sources": sources},
    )
    accepted = [
        {
            **sources[0],
            "status": "accepted",
            "candidate_status": "unassigned",
            "page_count": 2,
            "size_bytes": 20,
            "sha256": _sha(9_000),
            "outcome_note": "documented accepted-only field",
        }
    ]
    rejected = [
        {
            **sources[1],
            "status": "rejected",
            "rejection_reason": "documented rejected-only field",
        }
    ]
    accepted_path.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    rejected_path.write_text("".join(json.dumps(row) + "\n" for row in rejected), encoding="utf-8")
    _write_json(
        summary_path,
        {
            "source_count": 2,
            "accepted": 1,
            "rejected": 1,
            "pages": 2,
            "size_bytes": 20,
            "sha256_manifest": hashlib.sha256(_sha(9_000).encode()).hexdigest(),
        },
    )
    return accepted_path


def _checkpoint_sidecar(root: Path, accepted_path: Path) -> tuple[Path, str]:
    kind = "financial" if "financial" in accepted_path.parent.name else "difficult"
    source_path = accepted_path.with_name(
        "source_manifest.json" if kind == "financial" else "source_manifest.jsonl"
    )
    rejected_path = accepted_path.with_name("rejected_manifest.jsonl")
    summary_path = accepted_path.with_name("summary.json")

    def descriptor(path: Path, record_count: int) -> dict[str, object]:
        return {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "record_count": record_count,
        }

    source_count = (
        len(_read_records(source_path))
        if source_path.suffix == ".jsonl"
        else len(json.loads(source_path.read_text(encoding="utf-8"))["sources"])
    )
    sidecar = {
        "schema_version": "1.0",
        "revision_id": f"fixture-{accepted_path.parent.name}",
        "immutable": True,
        "revision": descriptor(accepted_path, len(_read_records(accepted_path))),
        "parent": descriptor(source_path, source_count),
        "companions": {
            "rejected": descriptor(rejected_path, len(_read_records(rejected_path))),
            "summary": descriptor(summary_path, 1),
        },
    }
    sidecar_path = accepted_path.with_name("accepted_manifest.revision.json")
    _write_json(sidecar_path, sidecar)
    return sidecar_path, hashlib.sha256(sidecar_path.read_bytes()).hexdigest()


def _track_snapshot_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[set[int], set[Path]]:
    descriptors: set[int] = set()
    locked_paths: set[Path] = set()
    original_read = _SnapshotStore.read

    def tracked_read(store: _SnapshotStore, path: Path) -> object:
        snapshot = original_read(store, path)
        descriptors.add(snapshot.descriptor)
        locked_paths.add(snapshot.path)
        return snapshot

    monkeypatch.setattr(_SnapshotStore, "read", tracked_read)
    return descriptors, locked_paths


def _assert_snapshot_resources_closed(descriptors: set[int], locked_paths: set[Path]) -> None:
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            corpus_partition_module.os.fstat(descriptor)
    for path in locked_paths:
        descriptor = corpus_partition_module.os.open(path, corpus_partition_module.os.O_RDONLY)
        try:
            corpus_partition_module.fcntl.flock(
                descriptor,
                corpus_partition_module.fcntl.LOCK_EX | corpus_partition_module.fcntl.LOCK_NB,
            )
        finally:
            corpus_partition_module.fcntl.flock(descriptor, corpus_partition_module.fcntl.LOCK_UN)
            corpus_partition_module.os.close(descriptor)


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_snapshot_store_close_attempts_every_descriptor_once_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_index: int
) -> None:
    paths = [tmp_path / f"snapshot-{index}.json" for index in range(3)]
    for index, path in enumerate(paths):
        _write_json(path, {"index": index})
    snapshots = _SnapshotStore()
    descriptors = [snapshots.read(path).descriptor for path in paths]
    original_close = corpus_partition_module.os.close
    attempts: list[int] = []

    def close_with_one_reported_failure(descriptor: int) -> None:
        attempts.append(descriptor)
        original_close(descriptor)
        if descriptor == descriptors[failed_index]:
            raise OSError(f"injected close failure {failed_index}")

    monkeypatch.setattr(corpus_partition_module.os, "close", close_with_one_reported_failure)
    with pytest.raises(ExceptionGroup, match="snapshot descriptor close failures") as raised:
        snapshots.close()
    assert attempts == descriptors
    assert f"injected close failure {failed_index}" in str(raised.value.exceptions[0])
    snapshots.close()
    assert attempts == descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            corpus_partition_module.os.fstat(descriptor)


def test_snapshot_store_close_collects_failures_in_descriptor_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / f"snapshot-{index}.json" for index in range(3)]
    for index, path in enumerate(paths):
        _write_json(path, {"index": index})
    snapshots = _SnapshotStore()
    descriptors = [snapshots.read(path).descriptor for path in paths]
    original_close = corpus_partition_module.os.close

    def report_every_close_failure(descriptor: int) -> None:
        original_close(descriptor)
        raise OSError(f"close-{descriptor}")

    monkeypatch.setattr(corpus_partition_module.os, "close", report_every_close_failure)
    with pytest.raises(ExceptionGroup) as raised:
        snapshots.close()
    assert [str(error) for error in raised.value.exceptions] == [
        f"close-{descriptor}" for descriptor in descriptors
    ]


def test_snapshot_store_destructor_uses_direct_non_raising_idempotent_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "snapshot.json"
    _write_json(path, {"ok": True})
    snapshots = _SnapshotStore()
    descriptor = snapshots.read(path).descriptor
    original_close = corpus_partition_module.os.close
    attempts: list[int] = []

    def close_then_report_failure(value: int) -> None:
        attempts.append(value)
        original_close(value)
        raise OSError("destructor close failure")

    monkeypatch.setattr(corpus_partition_module.os, "close", close_then_report_failure)
    snapshots.__del__()
    snapshots.__del__()
    assert attempts == []
    assert snapshots._snapshots == {}  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(OSError):
        corpus_partition_module.os.fstat(descriptor)


def test_explicitly_closed_store_finalization_does_not_consume_later_close_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older_path = tmp_path / "older.json"
    later_path = tmp_path / "later.json"
    _write_json(older_path, {"store": "older"})
    _write_json(later_path, {"store": "later"})

    older = _SnapshotStore()
    older.read(older_path)
    older.close()
    older._retained_cycle = older  # type: ignore[attr-defined]
    del older

    later = _SnapshotStore()
    later_descriptor = later.read(later_path).descriptor
    original_store_close = _SnapshotStore.close
    original_os_close = corpus_partition_module.os.close
    store_close_calls: list[_SnapshotStore] = []
    descriptor_close_calls: list[int] = []

    def tracked_store_close(store: _SnapshotStore) -> None:
        store_close_calls.append(store)
        original_store_close(store)

    def close_then_report_injected_failure(descriptor: int) -> None:
        descriptor_close_calls.append(descriptor)
        original_os_close(descriptor)
        raise OSError("injected later-store descriptor close failure")

    monkeypatch.setattr(_SnapshotStore, "close", tracked_store_close)
    monkeypatch.setattr(corpus_partition_module.os, "close", close_then_report_injected_failure)

    gc.collect()
    assert store_close_calls == []
    assert descriptor_close_calls == []

    with pytest.raises(ExceptionGroup, match="snapshot descriptor close failures") as raised:
        later.close()
    assert store_close_calls == [later]
    assert descriptor_close_calls == [later_descriptor]
    assert "injected later-store descriptor close failure" in str(raised.value.exceptions[0])


def test_snapshot_acquisition_preserves_primary_when_descriptor_close_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "snapshot.json"
    _write_json(path, {"ok": True})
    opened_descriptor: int | None = None
    original_flock = corpus_partition_module.fcntl.flock
    original_close = corpus_partition_module.os.close

    def fail_lock(descriptor: int, operation: int) -> None:
        nonlocal opened_descriptor
        opened_descriptor = descriptor
        if operation == corpus_partition_module.fcntl.LOCK_SH:
            raise RuntimeError("injected acquisition failure")
        original_flock(descriptor, operation)

    def close_then_report_failure(descriptor: int) -> None:
        original_close(descriptor)
        raise OSError("injected acquisition close failure")

    monkeypatch.setattr(corpus_partition_module.fcntl, "flock", fail_lock)
    monkeypatch.setattr(corpus_partition_module.os, "close", close_then_report_failure)
    with pytest.raises(RuntimeError, match="injected acquisition failure") as raised:
        _SnapshotStore().read(path)
    assert isinstance(raised.value.__cause__, OSError)
    assert "injected acquisition close failure" in str(raised.value.__cause__)
    assert opened_descriptor is not None
    with pytest.raises(OSError):
        corpus_partition_module.os.fstat(opened_descriptor)


@pytest.mark.parametrize("reader", [_read_json, _read_records])
def test_standalone_read_helpers_close_snapshots_on_decode_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: Callable[[Path], object],
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("not-json", encoding="utf-8")
    descriptors, locked_paths = _track_snapshot_resources(monkeypatch)
    with pytest.raises(ValueError, match="invalid JSON"):
        reader(path)
    _assert_snapshot_resources_closed(descriptors, locked_paths)


@pytest.mark.parametrize("reader", [_read_json, _read_records])
def test_standalone_read_helpers_preserve_decode_error_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: Callable[[Path], object],
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("not-json", encoding="utf-8")
    original_close = _SnapshotStore.close

    def close_then_report_failure(store: _SnapshotStore) -> None:
        original_close(store)
        raise OSError("injected helper close failure")

    monkeypatch.setattr(_SnapshotStore, "close", close_then_report_failure)
    with pytest.raises(ValueError, match="invalid JSON") as raised:
        reader(path)
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert "injected helper close failure" in str(raised.value.__cause__)
    assert "Expecting value" in str(raised.value.__cause__)


def test_owned_checkpoint_validator_closes_snapshots_on_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus_path, _partition_path, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    records = _read_records(checkpoint)
    summary_path = checkpoint.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["task"] = 999
    _write_json(summary_path, summary)
    descriptors, locked_paths = _track_snapshot_resources(monkeypatch)
    with pytest.raises(ValueError, match="does not identify completed task"):
        _validate_checkpoint(checkpoint, records)
    _assert_snapshot_resources_closed(descriptors, locked_paths)


def test_owned_checkpoint_validator_preserves_primary_when_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _corpus_path, _partition_path, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    records = _read_records(checkpoint)
    summary_path = checkpoint.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["task"] = 999
    _write_json(summary_path, summary)
    original_close = _SnapshotStore.close

    def close_then_report_failure(store: _SnapshotStore) -> None:
        original_close(store)
        raise OSError("injected checkpoint helper close failure")

    monkeypatch.setattr(_SnapshotStore, "close", close_then_report_failure)
    with pytest.raises(ValueError, match="does not identify completed task") as raised:
        _validate_checkpoint(checkpoint, records)
    assert isinstance(raised.value.__cause__, OSError)
    assert "injected checkpoint helper close failure" in str(raised.value.__cause__)


def test_builder_is_exact_family_safe_and_preserves_v1(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    partition_before = partition_v1.read_bytes()
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "data/corpus/manifest-v3.json",
        output_partition_path=tmp_path / "data/corpus/partition-v3.json",
        status="provisional",
    )

    assert result.document_counts == {"train": 12, "validation": 2, "holdout": 6}
    assert partition_v1.read_bytes() == partition_before
    output = json.loads(result.partition_path.read_text(encoding="utf-8"))
    by_id = {document["id"]: document for document in output["documents"]}
    assert by_id["doc-0"]["split"] == "train"
    assert by_id["doc-1"]["split"] == "validation"
    assert by_id["doc-2"]["split"] == "holdout"
    assert "doc-100" not in by_id
    assert "doc-101" not in by_id
    assert output["ideal_targets_family_reachable"] is True
    assert len(output["excluded_documents"]) == 2
    assert set(output["strata_summary"]) >= {"category", "page_count_band", "source_producer"}

    repeated = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "repeat/manifest-v3.json",
        output_partition_path=tmp_path / "repeat/partition-v3.json",
        status="provisional",
    )
    repeated_output = json.loads(repeated.partition_path.read_text(encoding="utf-8"))
    assert repeated_output["documents"] == output["documents"]
    assert repeated_output["summary"] == output["summary"]
    assert repeated_output["selected_document_targets"] == output["selected_document_targets"]


def test_builder_keeps_publisher_family_atomic(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0].update({
        "issuer": "Shared Publisher",
        "title": "Shared Publisher Annual Report 2023",
    })
    staging["documents"][1].update({
        "issuer": "Shared Publisher",
        "title": "Shared Publisher Annual Report 2024",
    })
    _refresh_deduplication_keys(staging["documents"][0])
    _refresh_deduplication_keys(staging["documents"][1])
    _write_json(staging_path, staging)
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        status="provisional",
    )
    output = json.loads(result.partition_path.read_text(encoding="utf-8"))
    shared = [
        document for document in output["documents"] if document["source_producer"] == "shared publisher"
    ]
    assert len(shared) == 2
    assert len({document["split"] for document in shared}) == 1


def test_builder_forces_new_member_of_existing_train_family_to_train(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0].update({"issuer": "producer 0 example", "title": "Document 0 2026"})
    _refresh_deduplication_keys(staging["documents"][0])
    _write_json(staging_path, staging)
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        status="provisional",
    )
    output = json.loads(result.partition_path.read_text(encoding="utf-8"))
    member = next(document for document in output["documents"] if document["id"] == "doc-20")
    assert member["split"] == "train"
    assert member["assignment_origin"] == "new-v3-existing-family"
    assert output["new_documents_forced_by_existing_family"] == 1


def test_builder_rejects_inspected_new_candidate(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0]["accuracy_inspected"] = True
    _write_json(staging_path, staging)
    with pytest.raises(ValueError, match="forbidden evidence flags must be boolean false"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_never_overwrites_versioned_outputs(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output = tmp_path / "partition-v3.json"
    output.write_text("sealed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must not already exist"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=output,
            status="provisional",
        )


def test_builder_rejects_stale_checkpoint(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    checkpoint.with_name("source_manifest.jsonl").touch()

    with pytest.raises(ValueError, match="stale relative to source"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[checkpoint],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_duplicate_difficult_source_outcome_identity(tmp_path: Path) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    source_path = checkpoint.with_name("source_manifest.jsonl")
    sources = _read_records(source_path)
    accepted = _read_records(checkpoint)
    sources[1]["slug"] = sources[0]["slug"]
    sources[1]["source_url"] = sources[0]["source_url"]
    accepted[1]["slug"] = accepted[0]["slug"]
    accepted[1]["source_url"] = accepted[0]["source_url"]
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    checkpoint.with_name("summary.json").touch()

    with pytest.raises(ValueError, match="source manifest contains duplicate slugs"):
        _validate_checkpoint(checkpoint, _read_records(checkpoint))


def test_builder_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    summary_path = checkpoint.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["accepted_count"] += 1
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="completeness counts"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[checkpoint],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_final_requires_expanded_inputs(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="final status requires"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="final",
        )


def test_eur_lex_semantic_uri_queries_remain_distinct() -> None:
    base = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF"
    urls = [
        f"{base}?uri=CELEX:32024R1689",  # AI Act
        f"{base}?uri=CELEX:32022R2065",  # Digital Services Act
        f"{base}?uri=CELEX:32016R0679",  # GDPR
    ]
    assert len({_normalize_url_value(url) for url in urls}) == 3
    legacy_commitment = _normalize_url_value(base)
    documents = [
        {
            "sha256": _sha(7_000 + index),
            "source_url": url,
            "canonical_url": legacy_commitment,
            "title": f"EUR-Lex document {index}",
            "producer": "EUR-Lex",
            "deduplication_keys": {"normalized_url": legacy_commitment},
        }
        for index, url in enumerate(urls)
    ]
    atomic_urls = [
        next(key for key in _atomic_keys(document) if key.startswith("url:")) for document in documents
    ]
    assert len(set(atomic_urls)) == 3


def test_parent_prefers_semantic_source_url_over_stale_canonical_url(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    base = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF"
    urls = [
        f"{base}?uri=CELEX:32024R1689",
        f"{base}?uri=CELEX:32022R2065",
        f"{base}?uri=CELEX:32016R0679",
    ]
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_v1.read_text(encoding="utf-8"))
    for index, source_url in enumerate(urls):
        updates = {
            "source_url": source_url,
            "canonical_url": base,
            "title": f"EUR-Lex document {index}",
            "producer": "EUR-Lex",
        }
        corpus["documents"][index].update(updates)
        corpus["documents"][index]["deduplication_keys"]["normalized_url"] = base
        partition["documents"][index].update(updates)
        partition["documents"][index]["deduplication_keys"]["normalized_url"] = base
    _write_json(corpus_path, corpus)
    _write_json(partition_v1, partition)
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v7.json",
        output_partition_path=tmp_path / "partition-v7.json",
        revision="v5",
        status="provisional",
    )
    assert sum(result.document_counts.values()) == 20


def test_url_query_order_encoding_and_tracking_are_canonicalized() -> None:
    ordered = "https://EUR-LEX.europa.eu/legal-content/EN/TXT/PDF?lang=EN&uri=CELEX%3A32024R1689"
    reordered = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF?uri=CELEX:32024R1689&lang=EN"
    tracked = f"{reordered}&utm_source=test&fbclid=tracker#section"
    assert _normalize_url_value(ordered) == _normalize_url_value(reordered)
    assert _normalize_url_value(tracked) == _normalize_url_value(reordered)


def test_adapted_checkpoint_dedup_commitment_uses_semantic_query_canonicalizer(tmp_path: Path) -> None:
    source_url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF?uri=CELEX:32024R1689&lang=EN"
    adapted = _adapt_checkpoint_document(
        {
            "slug": "ai-act",
            "source_url": source_url,
            "title": "AI Act",
            "producer": "EUR-Lex",
            "validation_note": "metadata only",
            "stratum": "legal",
            "sha256": _sha(8_000),
            "template_family": "eu-law",
        },
        tmp_path / "v3-difficult/accepted_manifest.jsonl",
    )
    assert adapted["deduplication_keys"]["normalized_url"] == _normalize_url_value(source_url)


def test_tracking_fragments_trailing_slashes_and_path_identity() -> None:
    canonical = "https://example.com/legal/content"
    tracking_only = "HTTPS://EXAMPLE.COM/legal/content/?utm_medium=email&sessionid=abc#page-2"
    assert _normalize_url_value(tracking_only) == _normalize_url_value(canonical)
    assert _normalize_url_value(f"{canonical}/#fragment") == _normalize_url_value(canonical)
    assert _normalize_url_value("https://example.com/a%2Fb") != _normalize_url_value(
        "https://example.com/a/b"
    )
    assert _normalize_url_value("https://example.com/%7euser") == _normalize_url_value(
        "https://example.com/~user"
    )


def test_target_ratios_are_60_10_30_at_816_documents() -> None:
    assert RATIOS == {"train": 0.60, "validation": 0.10, "holdout": 0.30}
    assert document_targets(816) == {"train": 490, "validation": 82, "holdout": 244}
    assert document_targets(734) == {"train": 440, "validation": 73, "holdout": 221}


def test_builder_rejects_incomplete_new_metadata(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    del staging["documents"][0]["provenance"]
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="missing provenance"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_duplicate_normalized_url(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][1]["source_url"] = staging["documents"][0]["source_url"] + "#copy"
    _refresh_deduplication_keys(staging["documents"][1])
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="duplicate URL"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_duplicate_title_and_publisher(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][1].update({
        "title": staging["documents"][0]["title"],
        "issuer": staging["documents"][0]["issuer"],
    })
    _refresh_deduplication_keys(staging["documents"][1])
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="duplicate normalized title and publisher"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_string_inspection_flag(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0]["accuracy_inspected"] = "true"
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="forbidden evidence flags must be boolean false"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_final_v3_artifacts_are_fresh_complete_exact_and_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/corpus/manifest-v3.json"
    partition_path = root / "data/corpus/partition-v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    previous = json.loads((root / "data/corpus/partition-v2.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "21a6e8d797ae748beb67ea50c896c983d1b07d504e11f17ccf5d70bbb66c8cf7"
    )
    assert hashlib.sha256(partition_path.read_bytes()).hexdigest() == (
        "998f26f8e03ce6769a389cdfb13b8968f672958c8d93f262ce4c4d616497cce7"
    )
    assert manifest["status"] == partition["status"] == "final"
    assert manifest["documents"] == partition["documents"]
    assert len(partition["documents"]) == 787
    assert (
        partition["selected_document_targets"]
        == document_targets(787)
        == {
            "train": 472,
            "validation": 79,
            "holdout": 236,
        }
    )
    assert manifest["selection"] == {
        "previously_active_documents": 408,
        "new_accuracy_uninspected_documents": 379,
        "retained_unique_documents": 787,
    }

    documents = partition["documents"]
    assert len({document["id"] for document in documents}) == len(documents)
    assert len({document["sha256"] for document in documents}) == len(documents)
    assert not any(re.search(r"layout\s*lm(?:v?2)?", json.dumps(document), re.I) for document in documents)
    by_sha = {document["sha256"]: document for document in documents}
    assert all(by_sha[document["sha256"]]["split"] == document["split"] for document in previous["documents"])

    new_documents = [
        document for document in documents if str(document["assignment_origin"]).startswith("new-v3")
    ]
    family_splits: dict[str, set[str]] = {}
    for document in new_documents:
        family_splits.setdefault(str(document["family_id"]), set()).add(str(document["split"]))
    assert all(len(splits) == 1 for splits in family_splits.values())


def test_final_v4_artifacts_consolidate_all_finalized_inputs_without_leakage() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/corpus/manifest-v4.json"
    partition_path = root / "data/corpus/partition-v4.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    previous = json.loads((root / "data/corpus/partition-v2.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "5b4bad443480cf12cc6362bb96256965cba0823fad6d7dc838d03c493c184de2"
    )
    assert hashlib.sha256(partition_path.read_bytes()).hexdigest() == (
        "af7b196bdeb61cfdf8613baba8c64b63a31bed57c02f27d3d2de8897148e6f54"
    )
    assert manifest["schema_version"] == partition["schema_version"] == "4.0"
    assert manifest["status"] == partition["status"] == "final"
    assert manifest["documents"] == partition["documents"]
    assert (
        partition["selected_document_targets"]
        == document_targets(851)
        == {
            "train": 511,
            "validation": 85,
            "holdout": 255,
        }
    )
    assert manifest["selection"] == {
        "previously_active_documents": 408,
        "new_accuracy_uninspected_documents": 443,
        "retained_unique_documents": 851,
    }
    assert {descriptor["path"]: descriptor["sha256"] for descriptor in partition["inputs"]} == {
        "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl": (
            "ef775c528961ee1a95dbc8de84cbaa04e5a2f3e921cfba4eb45512b89cd707a8"
        ),
        "data/corpus-expansion/v3-financial/accepted_manifest.jsonl": (
            "4a258db265198b63b5668be1d54e7155ebfbd5921a0fba8c3da28e68cb8d9f8f"
        ),
        "data/corpus-expansion/v3-gap/accepted_manifest.jsonl": (
            "7ded8c2545fb4932e66d7e0152ae25d7f7da3ed6c374e643c7c0a557854f6372"
        ),
    }

    documents = partition["documents"]
    assert len(documents) == 851 >= 816
    assert len({document["id"] for document in documents}) == len(documents)
    assert len({document["sha256"] for document in documents}) == len(documents)
    assert sum(int(document["page_count"]) for document in documents) == 101_187
    assert not any(re.search(r"layout\s*lm(?:v?2)?", json.dumps(document), re.I) for document in documents)
    by_sha = {document["sha256"]: document for document in documents}
    assert all(by_sha[document["sha256"]]["split"] == document["split"] for document in previous["documents"])

    new_documents = [
        document for document in documents if str(document["assignment_origin"]).startswith("new-v4")
    ]
    assert len(new_documents) == 443
    assert len({document["deduplication_keys"]["normalized_url"] for document in new_documents}) == 443
    assert len({document["deduplication_keys"]["normalized_title"] for document in new_documents}) == 443
    family_splits: dict[str, set[str]] = {}
    for document in new_documents:
        family_splits.setdefault(str(document["family_id"]), set()).add(str(document["split"]))
    assert all(len(splits) == 1 for splits in family_splits.values())

    assert hashlib.sha256((root / "data/corpus/manifest-v3.json").read_bytes()).hexdigest() == (
        "21a6e8d797ae748beb67ea50c896c983d1b07d504e11f17ccf5d70bbb66c8cf7"
    )
    assert hashlib.sha256((root / "data/corpus/partition-v3.json").read_bytes()).hexdigest() == (
        "998f26f8e03ce6769a389cdfb13b8968f672958c8d93f262ce4c4d616497cce7"
    )


def test_final_v5_artifacts_use_corrected_financial_ledger_without_leakage() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/corpus/manifest-v5.json"
    partition_path = root / "data/corpus/partition-v5.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    previous = json.loads((root / "data/corpus/partition-v2.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "658f5591674b1ba50c6aa5a261ee9cb8ece1a17c8bc39ccb3fd30f99f980f3f2"
    )
    assert hashlib.sha256(partition_path.read_bytes()).hexdigest() == (
        "b576123750f6f02de53e14633012e337111ed05f75b16af6c6f7acf221d4b9eb"
    )
    assert manifest["schema_version"] == partition["schema_version"] == "5.0"
    assert manifest["status"] == partition["status"] == "final"
    assert manifest["documents"] == partition["documents"]
    assert len(partition["documents"]) == 851 >= 816
    assert (
        partition["selected_document_targets"]
        == document_targets(851)
        == {
            "train": 511,
            "validation": 85,
            "holdout": 255,
        }
    )
    assert manifest["selection"] == {
        "previously_active_documents": 408,
        "new_accuracy_uninspected_documents": 443,
        "retained_unique_documents": 851,
    }
    assert {descriptor["path"]: descriptor["sha256"] for descriptor in partition["inputs"]} == {
        "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl": (
            "ef775c528961ee1a95dbc8de84cbaa04e5a2f3e921cfba4eb45512b89cd707a8"
        ),
        "data/corpus-expansion/v3-financial-r2/accepted_manifest.jsonl": (
            "4a258db265198b63b5668be1d54e7155ebfbd5921a0fba8c3da28e68cb8d9f8f"
        ),
        "data/corpus-expansion/v3-gap/accepted_manifest.jsonl": (
            "7ded8c2545fb4932e66d7e0152ae25d7f7da3ed6c374e643c7c0a557854f6372"
        ),
    }

    documents = partition["documents"]
    by_sha = {document["sha256"]: document for document in documents}
    assert all(by_sha[document["sha256"]]["split"] == document["split"] for document in previous["documents"])
    new_documents = [
        document for document in documents if str(document["assignment_origin"]).startswith("new-v5")
    ]
    family_splits: dict[str, set[str]] = {}
    for document in new_documents:
        family_splits.setdefault(str(document["family_id"]), set()).add(str(document["split"]))
    assert len(new_documents) == 443
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert hashlib.sha256((root / "data/corpus/manifest-v4.json").read_bytes()).hexdigest() == (
        "5b4bad443480cf12cc6362bb96256965cba0823fad6d7dc838d03c493c184de2"
    )
    assert hashlib.sha256((root / "data/corpus/partition-v4.json").read_bytes()).hexdigest() == (
        "af7b196bdeb61cfdf8613baba8c64b63a31bed57c02f27d3d2de8897148e6f54"
    )


def test_builder_rejects_repeated_sha_even_when_path_matches(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    duplicate = dict(staging["documents"][0])
    duplicate["id"] = "duplicate-id"
    staging["documents"].append(duplicate)
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="duplicate SHA-256"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_unknown_output_revision(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="unsupported corpus revision"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v6.json",
            output_partition_path=tmp_path / "partition-v6.json",
            status="provisional",
        )


def test_builder_rejects_ambiguous_inferred_output_revisions(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="imply conflicting revisions"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v4.json",
            output_partition_path=tmp_path / "partition-v5.json",
            status="provisional",
        )


def test_json_array_records_must_be_objects(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_json(path, {"documents": [{"id": "valid"}, "not-an-object"]})
    with pytest.raises(ValueError, match=r"documents\[1\]"):
        _read_records(path)


@pytest.mark.parametrize("invalid_false", [0, "false", None])
def test_forbidden_evidence_flags_require_boolean_false(tmp_path: Path, invalid_false: object) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0]["accuracy_inspected"] = invalid_false
    _write_json(staging_path, staging)
    with pytest.raises(ValueError, match="forbidden evidence flags must be boolean false"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_existing_financial_checkpoint_fails_closed_on_omitted_source_metadata() -> None:
    root = Path(__file__).resolve().parents[1]
    defective = root / "data/corpus-expansion/v3-financial/accepted_manifest.jsonl"
    corrected = root / "data/corpus-expansion/v3-financial-r2/accepted_manifest.jsonl"

    with pytest.raises(ValueError, match="completeness counts do not match"):
        _validate_checkpoint(defective, _read_records(defective))

    with pytest.raises(ValueError, match="outcomes do not exactly cover source canonical URLs"):
        _validate_checkpoint(corrected, _read_records(corrected))
    assert corrected.read_bytes() == defective.read_bytes()


def test_checkpoint_kind_recognizes_only_named_financial_revisions(tmp_path: Path) -> None:
    assert _checkpoint_kind(tmp_path / "v3-financial-r3/accepted_manifest.jsonl") == "financial"
    assert _checkpoint_kind(tmp_path / "v3-financial-r4/accepted_manifest.jsonl") is None
    assert _checkpoint_kind(tmp_path / "v3-financial-r3/other_manifest.jsonl") is None


def test_synthetic_financial_checkpoint_binds_all_source_fields_and_allows_outcome_fields(
    tmp_path: Path,
) -> None:
    accepted_path = _financial_checkpoint(tmp_path)
    descriptor = _validate_checkpoint(accepted_path, _read_records(accepted_path))
    assert descriptor is not None
    assert descriptor["kind"] == "financial"


def test_financial_checkpoint_binds_accepted_id_and_source_metadata(tmp_path: Path) -> None:
    accepted_path = _financial_checkpoint(tmp_path)
    accepted = _read_records(accepted_path)
    accepted[0]["title"] = "mutated title"
    accepted_path.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    accepted_path.with_name("summary.json").touch()
    with pytest.raises(ValueError, match="accepted outcome alters source metadata"):
        _validate_checkpoint(accepted_path, _read_records(accepted_path))


def test_difficult_checkpoint_binds_url_and_source_metadata(tmp_path: Path) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    accepted = _read_records(checkpoint)
    accepted[0]["source_url"] = "https://different.example/document.pdf"
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    checkpoint.with_name("summary.json").touch()
    with pytest.raises(ValueError, match="accepted outcome alters source metadata"):
        _validate_checkpoint(checkpoint, _read_records(checkpoint))


def test_difficult_checkpoint_binds_source_attributes(tmp_path: Path) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    accepted = _read_records(checkpoint)
    accepted[0]["producer"] = "different producer"
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    checkpoint.with_name("summary.json").touch()
    with pytest.raises(ValueError, match="accepted outcome alters source metadata"):
        _validate_checkpoint(checkpoint, _read_records(checkpoint))


def test_difficult_checkpoint_rejects_omitted_source_attribute(tmp_path: Path) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    accepted = _read_records(checkpoint)
    del accepted[0]["source_kind"]
    checkpoint.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    checkpoint.with_name("summary.json").touch()
    with pytest.raises(ValueError, match="accepted outcome omits source metadata"):
        _validate_checkpoint(checkpoint, _read_records(checkpoint))


def test_parent_manifest_rejects_duplicate_publisher_title(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["documents"][1]["title"] = corpus["documents"][0]["title"]
    corpus["documents"][1]["producer"] = corpus["documents"][0]["producer"]
    _write_json(corpus_path, corpus)
    with pytest.raises(ValueError, match="parent manifest contains duplicate normalized title and publisher"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


@pytest.mark.parametrize("duplicate_in", ["manifest", "partition"])
def test_parent_rejects_duplicate_canonical_urls(tmp_path: Path, duplicate_in: str) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    target_path = corpus_path if duplicate_in == "manifest" else partition_v1
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    first_url = payload["documents"][0]["deduplication_keys"]["normalized_url"]
    payload["documents"][1]["source_url"] = f"{first_url}#duplicate"
    payload["documents"][1]["deduplication_keys"]["normalized_url"] = first_url
    _write_json(target_path, payload)
    with pytest.raises(ValueError, match=f"parent {duplicate_in} contains duplicate canonical URLs"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_parent_manifest_partition_relationship_and_uniqueness_are_validated(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    partition = json.loads(partition_v1.read_text(encoding="utf-8"))
    partition["documents"][1]["id"] = partition["documents"][0]["id"]
    _write_json(partition_v1, partition)
    with pytest.raises(ValueError, match="duplicate document IDs"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_parent_manifest_partition_metadata_relationship_is_validated(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    partition = json.loads(partition_v1.read_text(encoding="utf-8"))
    partition["documents"][0]["title"] = "Mutated parent title"
    _write_json(partition_v1, partition)
    with pytest.raises(ValueError, match="manifest/partition metadata mismatch"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_parent_sha_uniqueness_is_validated(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["documents"][1]["sha256"] = corpus["documents"][0]["sha256"]
    _write_json(corpus_path, corpus)
    with pytest.raises(ValueError, match="duplicate SHA-256"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_title_deduplication_is_publisher_scoped(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][1]["title"] = staging["documents"][0]["title"]
    _refresh_deduplication_keys(staging["documents"][1])
    _write_json(staging_path, staging)

    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        status="provisional",
    )
    assert sum(result.document_counts.values()) == 20


def test_dedup_commitment_and_admission_share_canonical_publisher(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0]["producer"] = "IRS.gov"
    staging["documents"][0]["issuer"] = "IRS.gov"
    _refresh_deduplication_keys(staging["documents"][0])
    staging["documents"][0]["deduplication_keys"]["normalized_publisher"] = "irs"
    _write_json(staging_path, staging)
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        status="provisional",
    )
    assert sum(result.document_counts.values()) == 20


def test_template_series_and_edition_family_keys_are_publisher_scoped() -> None:
    left = _document(1_000, issuer="Publisher One")
    right = _document(1_001, issuer="Publisher Two")
    for document in (left, right):
        document["template_family"] = "shared-template"
        document["report_series"] = "quarterly"
        document["edition"] = "first"
    assert set(_atomic_keys(left)).isdisjoint(_atomic_keys(right))


def test_arbitrary_output_basenames_are_supported_with_explicit_revision(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v7.json",
        output_partition_path=tmp_path / "partition-v7.json",
        revision="v5",
        status="provisional",
    )
    assert result.manifest_path.name == "manifest-v7.json"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "5.0"


def test_revision_sidecar_is_versioned_unambiguous_and_authenticates_companions(
    tmp_path: Path,
) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    sidecar_path, sidecar_sha = _checkpoint_sidecar(tmp_path, checkpoint)
    snapshots = _SnapshotStore()
    validated = _validated_revision_sidecars(
        root_dir=tmp_path,
        new_manifest_paths=[checkpoint],
        revision_sidecar_paths=[sidecar_path],
        expected_sidecar_sha256=[sidecar_sha],
        snapshots=snapshots,
    )
    snapshots.close()
    assert validated[0]["companions_authenticated"] == ["rejected", "summary"]

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["output"] = sidecar["revision"]
    _write_json(sidecar_path, sidecar)
    ambiguous_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="exactly one of revision or output"):
        _validated_revision_sidecars(
            root_dir=tmp_path,
            new_manifest_paths=[checkpoint],
            revision_sidecar_paths=[sidecar_path],
            expected_sidecar_sha256=[ambiguous_sha],
            snapshots=_SnapshotStore(),
        )


def test_revision_sidecar_validates_parent_count_and_complete_companions(tmp_path: Path) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    sidecar_path, _sidecar_sha = _checkpoint_sidecar(tmp_path, checkpoint)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["parent"]["record_count"] += 1
    _write_json(sidecar_path, sidecar)
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="parent record count mismatch"):
        _validated_revision_sidecars(
            root_dir=tmp_path,
            new_manifest_paths=[checkpoint],
            revision_sidecar_paths=[sidecar_path],
            expected_sidecar_sha256=[sidecar_sha],
            snapshots=_SnapshotStore(),
        )

    sidecar["parent"]["record_count"] -= 1
    del sidecar["companions"]["summary"]
    _write_json(sidecar_path, sidecar)
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="authenticate exactly rejected and summary"):
        _validated_revision_sidecars(
            root_dir=tmp_path,
            new_manifest_paths=[checkpoint],
            revision_sidecar_paths=[sidecar_path],
            expected_sidecar_sha256=[sidecar_sha],
            snapshots=_SnapshotStore(),
        )


def test_revision_sidecar_requires_version_and_expected_source_parent(tmp_path: Path) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    sidecar_path, _sidecar_sha = _checkpoint_sidecar(tmp_path, checkpoint)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    del sidecar["revision_id"]
    _write_json(sidecar_path, sidecar)
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="supported schema_version 1.0"):
        _validated_revision_sidecars(
            root_dir=tmp_path,
            new_manifest_paths=[checkpoint],
            revision_sidecar_paths=[sidecar_path],
            expected_sidecar_sha256=[sidecar_sha],
            snapshots=_SnapshotStore(),
        )

    sidecar_path, _sidecar_sha = _checkpoint_sidecar(tmp_path, checkpoint)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["parent"] = sidecar["revision"]
    _write_json(sidecar_path, sidecar)
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="parent does not describe the expected checkpoint file"):
        _validated_revision_sidecars(
            root_dir=tmp_path,
            new_manifest_paths=[checkpoint],
            revision_sidecar_paths=[sidecar_path],
            expected_sidecar_sha256=[sidecar_sha],
            snapshots=_SnapshotStore(),
        )


@pytest.mark.parametrize("invalid_version", [True, "2.0", "", None])
def test_revision_sidecar_requires_exact_supported_schema_version(
    tmp_path: Path, invalid_version: object
) -> None:
    _corpus_path, _partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    sidecar_path, _sidecar_sha = _checkpoint_sidecar(tmp_path, checkpoint)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["schema_version"] = invalid_version
    _write_json(sidecar_path, sidecar)
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="supported schema_version 1.0"):
        _validated_revision_sidecars(
            root_dir=tmp_path,
            new_manifest_paths=[checkpoint],
            revision_sidecar_paths=[sidecar_path],
            expected_sidecar_sha256=[sidecar_sha],
            snapshots=_SnapshotStore(),
        )


def test_final_floor_is_checked_before_atomic_grouping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    monkeypatch.setattr(corpus_partition_module, "_validate_final_inputs", lambda *_args: None)
    monkeypatch.setattr(corpus_partition_module, "_validated_revision_sidecars", lambda **_kwargs: [])

    def grouping_must_not_run(*_args: object) -> object:
        raise AssertionError("atomic grouping ran before final floor")

    monkeypatch.setattr(corpus_partition_module, "_atomic_groups", grouping_must_not_run)
    with pytest.raises(ValueError, match="below the 816-document floor"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            output_descriptor_path=tmp_path / "partition-v3.commit.json",
            revision_sidecar_paths=[staging_path],
            expected_revision_sidecar_sha256=["0" * 64],
            status="final",
        )


def test_builder_rejects_unsupported_runtime_status_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    outputs = [tmp_path / "manifest-v3.json", tmp_path / "partition-v3.json"]

    def publication_must_not_start(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("publication started for an unsupported status")

    monkeypatch.setattr(corpus_partition_module, "_stage_output", publication_must_not_start)
    with pytest.raises(ValueError, match="status must be exactly provisional or final"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=outputs[0],
            output_partition_path=outputs[1],
            status="draft",  # type: ignore[arg-type]
        )
    assert not any(path.exists() for path in outputs)


@pytest.mark.parametrize("phase", ["checkpoint", "solver", "staging", "publication"])
def test_builder_closes_snapshot_resources_after_injected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    descriptors, locked_paths = _track_snapshot_resources(monkeypatch)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    output_descriptor = tmp_path / "partition-v3.commit.json"
    outputs = [output_manifest, output_partition, output_descriptor]
    inputs_before = {path: path.read_bytes() for path in (corpus_path, partition_v1, staging_path)}

    def fail_phase(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"injected {phase} failure")

    if phase == "checkpoint":
        monkeypatch.setattr(corpus_partition_module, "_validate_checkpoint", fail_phase)
    elif phase == "solver":
        monkeypatch.setattr(corpus_partition_module, "_optimize", fail_phase)
    elif phase == "staging":
        monkeypatch.setattr(corpus_partition_module, "_stage_output", fail_phase)
    else:
        original_link = corpus_partition_module.os.link

        def fail_partition_publication(source: Path, destination: Path) -> None:
            if Path(destination) == output_partition:
                raise RuntimeError("injected publication failure")
            original_link(source, destination)

        monkeypatch.setattr(corpus_partition_module.os, "link", fail_partition_publication)

    with pytest.raises(RuntimeError, match=f"injected {phase} failure"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=output_manifest,
            output_partition_path=output_partition,
            output_descriptor_path=output_descriptor,
            status="provisional",
        )

    _assert_snapshot_resources_closed(descriptors, locked_paths)
    assert not any(path.exists() for path in outputs)
    assert not list(tmp_path.glob(".*.staged"))
    assert {path: path.read_bytes() for path in inputs_before} == inputs_before


def test_close_failure_is_chained_without_masking_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    original_close = _SnapshotStore.close
    close_calls = 0

    def fail_first_close(store: _SnapshotStore) -> None:
        nonlocal close_calls
        original_close(store)
        close_calls += 1
        if close_calls == 1:
            raise OSError("injected snapshot close failure")

    def fail_solver(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected solver failure")

    monkeypatch.setattr(_SnapshotStore, "close", fail_first_close)
    monkeypatch.setattr(corpus_partition_module, "_optimize", fail_solver)
    with pytest.raises(RuntimeError, match="injected solver failure") as raised:
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )
    assert isinstance(raised.value.__cause__, OSError)
    assert "injected snapshot close failure" in str(raised.value.__cause__)
    assert close_calls >= 1
    assert not (tmp_path / "manifest-v3.json").exists()
    assert not (tmp_path / "partition-v3.json").exists()


def test_close_failure_after_publication_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    output_descriptor = tmp_path / "partition-v3.commit.json"
    original_close = _SnapshotStore.close
    close_calls = 0

    def fail_first_close(store: _SnapshotStore) -> None:
        nonlocal close_calls
        original_close(store)
        close_calls += 1
        if close_calls == 1:
            raise OSError("injected snapshot close failure")

    monkeypatch.setattr(_SnapshotStore, "close", fail_first_close)
    with pytest.raises(OSError, match="injected snapshot close failure"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=output_manifest,
            output_partition_path=output_partition,
            output_descriptor_path=output_descriptor,
            status="provisional",
        )
    assert close_calls >= 1
    assert output_manifest.exists()
    assert output_partition.exists()
    assert output_descriptor.exists()
    monkeypatch.setattr(_SnapshotStore, "close", original_close)
    descriptor_sha = hashlib.sha256(output_descriptor.read_bytes()).hexdigest()
    verified = verify_corpus_partition_descriptor(
        output_descriptor,
        root_dir=tmp_path,
        expected_descriptor_sha256=descriptor_sha,
    )
    assert verified["revision_id"] == "corpus-partition-v3"


def test_verifier_preserves_primary_error_when_snapshot_close_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    descriptor_path = tmp_path / "partition-v3.commit.json"
    build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        output_descriptor_path=descriptor_path,
        status="provisional",
    )
    original_close = _SnapshotStore.close
    close_failed_once = False

    def fail_first_close(store: _SnapshotStore) -> None:
        nonlocal close_failed_once
        original_close(store)
        if not close_failed_once:
            close_failed_once = True
            raise OSError("injected verifier close failure")

    monkeypatch.setattr(_SnapshotStore, "close", fail_first_close)
    with pytest.raises(ValueError, match="output descriptor SHA-256 mismatch") as raised:
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256="0" * 64,
        )
    assert isinstance(raised.value.__cause__, OSError)
    assert "injected verifier close failure" in str(raised.value.__cause__)


def test_metadata_change_before_publication_fails_without_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    original_revalidate = corpus_partition_module._SnapshotStore.revalidate  # pyright: ignore[reportPrivateUsage]

    def mutate_then_revalidate(store: object) -> None:
        staging_path.write_bytes(staging_path.read_bytes() + b"\n")
        original_revalidate(store)  # type: ignore[arg-type]

    monkeypatch.setattr(
        corpus_partition_module._SnapshotStore,  # pyright: ignore[reportPrivateUsage]
        "revalidate",
        mutate_then_revalidate,
    )
    with pytest.raises(RuntimeError, match="changed before publication"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=output_manifest,
            output_partition_path=output_partition,
            status="provisional",
        )
    assert not output_manifest.exists()
    assert not output_partition.exists()


def test_descriptor_publication_failure_rolls_back_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    output_descriptor = tmp_path / "partition-v3.commit.json"
    original_link = corpus_partition_module.os.link

    def fail_descriptor(source: Path, destination: Path) -> None:
        if Path(destination) == output_descriptor:
            raise OSError("injected descriptor failure")
        original_link(source, destination)

    monkeypatch.setattr(corpus_partition_module.os, "link", fail_descriptor)
    with pytest.raises(OSError, match="injected descriptor failure"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=output_manifest,
            output_partition_path=output_partition,
            output_descriptor_path=output_descriptor,
            status="provisional",
        )
    assert not output_manifest.exists()
    assert not output_partition.exists()
    assert not output_descriptor.exists()


def test_publication_rollback_continues_after_cleanup_failure_and_preserves_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    descriptors, locked_paths = _track_snapshot_resources(monkeypatch)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    output_descriptor = tmp_path / "partition-v3.commit.json"
    outputs = [output_manifest, output_partition, output_descriptor]
    original_link = corpus_partition_module.os.link
    original_unlink = Path.unlink
    original_close = _SnapshotStore.close
    cleanup_attempts: list[Path] = []
    failed_once = False
    close_failed_once = False

    def fail_publication(source: Path, destination: Path) -> None:
        if Path(destination) == output_descriptor:
            raise OSError("primary publication failure")
        original_link(source, destination)

    def fail_one_cleanup(path: Path, missing_ok: bool = False) -> None:
        nonlocal failed_once
        if path in outputs:
            cleanup_attempts.append(path)
        if path == output_partition and not failed_once:
            failed_once = True
            raise OSError("transient rollback failure")
        original_unlink(path, missing_ok=missing_ok)

    def fail_first_close(store: _SnapshotStore) -> None:
        nonlocal close_failed_once
        owned_snapshots = bool(store._snapshots)  # pyright: ignore[reportPrivateUsage]
        original_close(store)
        if owned_snapshots and not close_failed_once:
            close_failed_once = True
            raise OSError("snapshot close failure after rollback")

    monkeypatch.setattr(corpus_partition_module.os, "link", fail_publication)
    monkeypatch.setattr(Path, "unlink", fail_one_cleanup)
    monkeypatch.setattr(_SnapshotStore, "close", fail_first_close)
    with pytest.raises(OSError, match="primary publication failure") as raised:
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=output_manifest,
            output_partition_path=output_partition,
            output_descriptor_path=output_descriptor,
            status="provisional",
        )
    assert output_manifest in cleanup_attempts
    assert output_partition in cleanup_attempts
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert "transient rollback failure" in str(raised.value.__cause__)
    assert "snapshot close failure after rollback" in str(raised.value.__cause__)
    _assert_snapshot_resources_closed(descriptors, locked_paths)
    assert not any(path.exists() for path in outputs)
    assert not list(tmp_path.glob(".*.staged"))


def test_rollback_existence_error_does_not_abort_later_cleanup_or_replace_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    output_descriptor = tmp_path / "partition-v3.commit.json"
    original_link = corpus_partition_module.os.link
    original_exists = Path.exists
    rollback_started = False
    failed_once = False

    def fail_publication(source: Path, destination: Path) -> None:
        nonlocal rollback_started
        if Path(destination) == output_descriptor:
            rollback_started = True
            raise OSError("primary publication failure")
        original_link(source, destination)

    def fail_one_existence_check(path: Path) -> bool:
        nonlocal failed_once
        if rollback_started and path == output_partition and not failed_once:
            failed_once = True
            raise OSError("rollback stat failure")
        return original_exists(path)

    monkeypatch.setattr(corpus_partition_module.os, "link", fail_publication)
    monkeypatch.setattr(Path, "exists", fail_one_existence_check)
    with pytest.raises(OSError, match="primary publication failure") as raised:
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=output_manifest,
            output_partition_path=output_partition,
            output_descriptor_path=output_descriptor,
            status="provisional",
        )
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert "rollback stat failure" in str(raised.value.__cause__)
    assert not original_exists(output_manifest)
    assert not original_exists(output_partition)


def test_stage_cleanup_failure_preserves_staging_error_and_removes_transient_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    original_fsync_directory = corpus_partition_module._fsync_directory  # pyright: ignore[reportPrivateUsage]
    original_unlink = Path.unlink
    staged_paths: list[Path] = []
    failed_cleanup_once = False

    def fail_staging_directory_sync(path: Path) -> None:
        if path == tmp_path:
            raise OSError("primary staging failure")
        original_fsync_directory(path)

    def fail_first_staged_cleanup(path: Path, missing_ok: bool = False) -> None:
        nonlocal failed_cleanup_once
        if path.name.endswith(".staged"):
            staged_paths.append(path)
            if not failed_cleanup_once:
                failed_cleanup_once = True
                raise OSError("staged cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(corpus_partition_module, "_fsync_directory", fail_staging_directory_sync)
    monkeypatch.setattr(Path, "unlink", fail_first_staged_cleanup)
    with pytest.raises(OSError, match="primary staging failure") as raised:
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )
    assert isinstance(raised.value.__cause__, ExceptionGroup)
    assert "staged cleanup failure" in str(raised.value.__cause__)
    assert staged_paths
    assert not any(path.exists() for path in staged_paths)


def test_post_commit_input_mutation_rolls_back_all_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    output_descriptor = tmp_path / "partition-v3.commit.json"
    original_revalidate = corpus_partition_module._SnapshotStore.revalidate  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def mutate_after_commit(store: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            staging_path.write_bytes(staging_path.read_bytes() + b"\n")
        original_revalidate(store)  # type: ignore[arg-type]

    monkeypatch.setattr(
        corpus_partition_module._SnapshotStore,  # pyright: ignore[reportPrivateUsage]
        "revalidate",
        mutate_after_commit,
    )
    with pytest.raises(RuntimeError, match="changed before publication"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=output_manifest,
            output_partition_path=output_partition,
            output_descriptor_path=output_descriptor,
            status="provisional",
        )
    assert not output_manifest.exists()
    assert not output_partition.exists()
    assert not output_descriptor.exists()


def test_cli_final_build_requires_output_descriptor(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "build-corpus-partition",
            "--root-dir",
            str(tmp_path),
            "--corpus-manifest",
            str(corpus_path),
            "--previous-partition",
            str(partition_v1),
            "--new-manifest",
            str(staging_path),
            "--output-manifest",
            str(tmp_path / "manifest-v3.json"),
            "--output-partition",
            str(tmp_path / "partition-v3.json"),
            "--status",
            "final",
        ],
    )
    assert result.exit_code != 0
    assert "requires an immutable output descriptor" in result.output


def test_cli_final_build_requires_pinned_checkpoint_sidecars(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "build-corpus-partition",
            "--root-dir",
            str(tmp_path),
            "--corpus-manifest",
            str(corpus_path),
            "--previous-partition",
            str(partition_v1),
            "--new-manifest",
            str(staging_path),
            "--output-manifest",
            str(tmp_path / "manifest-v3.json"),
            "--output-partition",
            str(tmp_path / "partition-v3.json"),
            "--output-descriptor",
            str(tmp_path / "partition-v3.commit.json"),
            "--status",
            "final",
        ],
    )
    assert result.exit_code != 0
    assert "externally SHA-pinned immutable checkpoint sidecars" in result.output


def test_cli_explicit_revision_allows_v7_output_basenames(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output_manifest = tmp_path / "manifest-v7.json"
    output_partition = tmp_path / "partition-v7.json"
    result = CliRunner().invoke(
        cli,
        [
            "build-corpus-partition",
            "--root-dir",
            str(tmp_path),
            "--corpus-manifest",
            str(corpus_path),
            "--previous-partition",
            str(partition_v1),
            "--new-manifest",
            str(staging_path),
            "--output-manifest",
            str(output_manifest),
            "--output-partition",
            str(output_partition),
            "--revision",
            "v5",
            "--status",
            "provisional",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output_manifest.read_text(encoding="utf-8"))["schema_version"] == "5.0"


def test_cli_build_corpus_partition_wires_provisional_build(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output_manifest = tmp_path / "manifest-v3.json"
    output_partition = tmp_path / "partition-v3.json"
    output_descriptor = tmp_path / "partition-v3.commit.json"
    result = CliRunner().invoke(
        cli,
        [
            "build-corpus-partition",
            "--root-dir",
            str(tmp_path),
            "--corpus-manifest",
            str(corpus_path),
            "--previous-partition",
            str(partition_v1),
            "--new-manifest",
            str(staging_path),
            "--output-manifest",
            str(output_manifest),
            "--output-partition",
            str(output_partition),
            "--output-descriptor",
            str(output_descriptor),
            "--status",
            "provisional",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "wrote provisional partition: train=12, validation=2, holdout=6" in result.output
    partition = json.loads(output_partition.read_text(encoding="utf-8"))
    descriptor = json.loads(output_descriptor.read_text(encoding="utf-8"))
    assert descriptor["schema_version"] == "2.0"
    assert descriptor["immutable"] is True
    assert descriptor["outputs"]["manifest"]["record_count"] == 20
    assert descriptor["outputs"]["partition"]["record_count"] == 20
    assert descriptor["parents"]["manifest"]["record_count"] == 10
    assert descriptor["parents"]["partition"]["record_count"] == 3
    assert [item["role"] for item in descriptor["authenticated_inputs"]] == ["accepted"]
    assert re.fullmatch(r"[0-9a-f]{64}", descriptor["input_commitment_sha256"])
    assert (
        descriptor["outputs"]["partition"]["sha256"]
        == hashlib.sha256(output_partition.read_bytes()).hexdigest()
    )
    assert partition["manifest"]["sha256"] == hashlib.sha256(output_manifest.read_bytes()).hexdigest()
    assert partition["parent_manifest"]["sha256"] == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert partition["parent_partition"]["sha256"] == hashlib.sha256(partition_v1.read_bytes()).hexdigest()
    descriptor_sha = hashlib.sha256(output_descriptor.read_bytes()).hexdigest()
    verification = CliRunner().invoke(
        cli,
        [
            "verify-corpus-partition",
            str(output_descriptor),
            "--root-dir",
            str(tmp_path),
            "--expected-descriptor-sha256",
            descriptor_sha,
        ],
    )
    assert verification.exit_code == 0, verification.output
    assert "verified corpus partition corpus-partition-v3" in verification.output


def _output_descriptor_fixture(tmp_path: Path) -> tuple[Path, Path]:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    descriptor_path = tmp_path / "partition-v8.commit.json"
    build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v8.json",
        output_partition_path=tmp_path / "partition-v8.json",
        output_descriptor_path=descriptor_path,
        revision="v5",
        status="provisional",
    )
    return descriptor_path, staging_path


@pytest.mark.parametrize("tamper", ["missing", "mismatched"])
def test_output_descriptor_rejects_missing_or_mismatched_input_commitment(
    tmp_path: Path, tamper: str
) -> None:
    descriptor_path, _staging_path = _output_descriptor_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if tamper == "missing":
        del descriptor["input_commitment_sha256"]
    else:
        descriptor["input_commitment_sha256"] = "0" * 64
    _write_json(descriptor_path, descriptor)
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="input commitment SHA-256 mismatch"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_output_descriptor_rejects_input_and_record_count_tampering(tmp_path: Path) -> None:
    descriptor_path, staging_path = _output_descriptor_fixture(tmp_path)
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    staging_path.write_bytes(staging_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match=r"authenticated_inputs\[0\] SHA-256 mismatch"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_descriptor, _clean_staging = _output_descriptor_fixture(clean_root)
    descriptor = json.loads(clean_descriptor.read_text(encoding="utf-8"))
    descriptor["outputs"]["manifest"]["record_count"] += 1
    _write_json(clean_descriptor, descriptor)
    tampered_sha = hashlib.sha256(clean_descriptor.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="outputs.manifest record count mismatch"):
        verify_corpus_partition_descriptor(
            clean_descriptor,
            root_dir=clean_root,
            expected_descriptor_sha256=tampered_sha,
        )


def test_cli_verifier_rejects_tampered_authenticated_input(tmp_path: Path) -> None:
    descriptor_path, staging_path = _output_descriptor_fixture(tmp_path)
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    staging_path.write_bytes(staging_path.read_bytes() + b"\n")
    result = CliRunner().invoke(
        cli,
        [
            "verify-corpus-partition",
            str(descriptor_path),
            "--root-dir",
            str(tmp_path),
            "--expected-descriptor-sha256",
            descriptor_sha,
        ],
    )
    assert result.exit_code != 0
    assert "authenticated_inputs[0] SHA-256 mismatch" in result.output


def _rewrite_descriptor(path: Path, descriptor: dict[str, Any]) -> str:
    _write_json(path, descriptor)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_descriptor_outputs(
    descriptor_path: Path,
    root: Path,
    *,
    status: str,
    input_paths: list[str],
    document_count: int,
    revision: str = "v5",
) -> str:
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    manifest_path = root / descriptor["outputs"]["manifest"]["path"]
    partition_path = root / descriptor["outputs"]["partition"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    documents = [{"id": f"fixture-{index}"} for index in range(document_count)]
    inputs = [
        {
            "path": path,
            "sha256": "0" * 64,
            "size_bytes": 0,
            "accepted_unique_documents": 0,
            "checkpoint_validation": {},
        }
        for path in input_paths
    ]
    schema_version = {"v3": "3.0", "v4": "4.0", "v5": "5.0"}[revision]
    for payload in (manifest, partition):
        payload["schema_version"] = schema_version
        payload["status"] = status
        payload["inputs"] = inputs
        payload["documents"] = documents
    _write_json(manifest_path, manifest)
    partition["manifest"] = {
        "path": str(manifest_path.relative_to(root)),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "size_bytes": manifest_path.stat().st_size,
        "record_count": document_count,
    }
    _write_json(partition_path, partition)
    descriptor["revision_id"] = f"corpus-partition-{revision}"
    descriptor["status"] = status
    for label, path in (("manifest", manifest_path), ("partition", partition_path)):
        descriptor["outputs"][label].update({
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "record_count": document_count,
        })
    return _rewrite_descriptor(descriptor_path, descriptor)


def _mutate_authenticated_outputs(
    descriptor_path: Path,
    root: Path,
    mutate: Any,
) -> str:
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    manifest_path = root / descriptor["outputs"]["manifest"]["path"]
    partition_path = root / descriptor["outputs"]["partition"]["path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    mutate(manifest, partition)
    _write_json(manifest_path, manifest)
    partition["manifest"] = {
        "path": str(manifest_path.relative_to(root)),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "size_bytes": manifest_path.stat().st_size,
        "record_count": len(manifest["documents"]),
    }
    _write_json(partition_path, partition)
    for label, path in (("manifest", manifest_path), ("partition", partition_path)):
        descriptor["outputs"][label].update({
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "record_count": len(manifest["documents"]),
        })
    return _rewrite_descriptor(descriptor_path, descriptor)


def test_final_descriptor_rejects_empty_corpus_partition_v5(tmp_path: Path) -> None:
    descriptor_path, _staging_path = _output_descriptor_fixture(tmp_path)
    descriptor_sha = _rewrite_descriptor_outputs(
        descriptor_path,
        tmp_path,
        status="final",
        input_paths=[],
        document_count=0,
    )
    with pytest.raises(ValueError, match="below the 816-document floor"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


@pytest.mark.parametrize(
    ("revision", "input_paths", "message"),
    [
        (
            "v3",
            [
                "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl",
                "v3-financial-r3/accepted_manifest.jsonl",
                "data/corpus-expansion/v3-gap/accepted_manifest.jsonl",
            ],
            "requires exactly these v3 checkpoints",
        ),
        (
            "v3",
            [
                "data/corpus-expansion/v3-difficult/substituted.jsonl",
                "v3-financial-r3/accepted_manifest.jsonl",
            ],
            "requires exactly these v3 checkpoints",
        ),
        (
            "v4",
            [
                "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl",
                "v3-financial-r3/accepted_manifest.jsonl",
            ],
            "requires exactly these v4 checkpoints",
        ),
        (
            "v5",
            [
                "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl",
                "v3-financial-r3/accepted_manifest.jsonl",
            ],
            "requires exactly these v5 checkpoints",
        ),
        (
            "v5",
            [
                "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl",
                "v3-financial/accepted_manifest.jsonl",
                "data/corpus-expansion/v3-gap/accepted_manifest.jsonl",
            ],
            "requires a corrected v3-financial-r2/r3 checkpoint",
        ),
    ],
)
def test_final_descriptor_enforces_builder_checkpoint_contract_without_external_state(
    tmp_path: Path, revision: str, input_paths: list[str], message: str
) -> None:
    descriptor_path, _staging_path = _output_descriptor_fixture(tmp_path)
    descriptor_sha = _rewrite_descriptor_outputs(
        descriptor_path,
        tmp_path,
        status="final",
        input_paths=input_paths,
        document_count=816,
        revision=revision,
    )
    with pytest.raises(ValueError, match=message):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_descriptor_rejects_self_consistent_input_identity_tamper(tmp_path: Path) -> None:
    descriptor_path, _staging_path = _output_descriptor_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    parent = descriptor["parents"]["manifest"]
    descriptor["authenticated_inputs"][0].update({
        "path": parent["path"],
        "sha256": parent["sha256"],
        "size_bytes": parent["size_bytes"],
        "record_count": parent["record_count"],
    })
    canonical = json.dumps(
        descriptor["authenticated_inputs"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    descriptor["input_commitment_sha256"] = hashlib.sha256(canonical).hexdigest()
    descriptor_sha = _rewrite_descriptor(descriptor_path, descriptor)
    with pytest.raises(ValueError, match="authenticated input identity mismatch at index 0"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "bogus", "status must be exactly provisional or final"),
        ("revision_id", "corpus-partition-v99", "revision_id is missing or unsupported"),
    ],
)
def test_descriptor_rejects_bogus_status_and_revision(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    descriptor_path, _staging_path = _output_descriptor_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor[field] = value
    descriptor_sha = _rewrite_descriptor(descriptor_path, descriptor)
    with pytest.raises(ValueError, match=message):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_descriptor_requires_parent_roles_and_embedded_parent_equality(tmp_path: Path) -> None:
    descriptor_path, _staging_path = _output_descriptor_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["parents"]["manifest"]["role"] = "wrong-parent-role"
    descriptor_sha = _rewrite_descriptor(descriptor_path, descriptor)
    with pytest.raises(ValueError, match="does not equal embedded parent commitment"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


@pytest.mark.parametrize("tamper", ["shared-policy", "embedded-manifest-count"])
def test_descriptor_rejects_output_contract_tampering(tmp_path: Path, tamper: str) -> None:
    descriptor_path, _staging_path = _output_descriptor_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    partition_path = tmp_path / descriptor["outputs"]["partition"]["path"]
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    if tamper == "shared-policy":
        partition["policy"] = {"tampered": True}
        expected_message = "manifest/partition shared fields differ"
    else:
        partition["manifest"]["record_count"] += 1
        expected_message = "partition embedded manifest commitment mismatch"
    _write_json(partition_path, partition)
    descriptor["outputs"]["partition"]["sha256"] = hashlib.sha256(partition_path.read_bytes()).hexdigest()
    descriptor["outputs"]["partition"]["size_bytes"] = partition_path.stat().st_size
    descriptor_sha = _rewrite_descriptor(descriptor_path, descriptor)
    with pytest.raises(ValueError, match=expected_message):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def _authenticated_descriptor_fixture(
    tmp_path: Path,
    *,
    tamper_checkpoint_semantics: bool = False,
    grandfathered_conflict: bool = False,
    current_like_legacy_parents: bool = False,
) -> Path:
    corpus_path, partition_path, staging_path = _fixture(tmp_path)
    if current_like_legacy_parents:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        corpus["documents"] = corpus["documents"][:8] + [_document(index) for index in range(30_000, 30_400)]
        partition = json.loads(partition_path.read_text(encoding="utf-8"))
        for payload in (corpus, partition):
            for document in payload["documents"]:
                for field in (
                    "template_family",
                    "deduplication_keys",
                    "accuracy_inspection_status",
                ):
                    document.pop(field, None)
        _write_json(corpus_path, corpus)
        _write_json(partition_path, partition)
    if grandfathered_conflict:
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        partition = json.loads(partition_path.read_text(encoding="utf-8"))
        for payload in (corpus, partition):
            for document in payload["documents"][:2]:
                document["producer"] = "Shared Parent Publisher"
                document["report_series"] = "Shared Parent Series"
                _refresh_deduplication_keys(document)
        _write_json(corpus_path, corpus)
        _write_json(partition_path, partition)
    accepted_paths = [
        _difficult_checkpoint(tmp_path, staging_path),
        _financial_checkpoint(tmp_path),
    ]
    checkpoints: list[dict[str, Any]] = []
    for accepted_path in accepted_paths:
        checkpoint = _validate_checkpoint(accepted_path, _read_records(accepted_path))
        assert checkpoint is not None
        checkpoints.append({
            key: (str(Path(str(value)).relative_to(tmp_path)) if key.endswith("_path") else value)
            for key, value in checkpoint.items()
        })
    if tamper_checkpoint_semantics:
        summary_path = accepted_paths[0].with_name("summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["task"] = 999
        _write_json(summary_path, summary)
        checkpoints[0]["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    sidecar_pairs = [_checkpoint_sidecar(tmp_path, path) for path in accepted_paths]
    snapshots = _SnapshotStore()
    sidecars = _validated_revision_sidecars(
        root_dir=tmp_path,
        new_manifest_paths=accepted_paths,
        revision_sidecar_paths=[pair[0] for pair in sidecar_pairs],
        expected_sidecar_sha256=[pair[1] for pair in sidecar_pairs],
        snapshots=snapshots,
    )
    snapshots.close()

    def file_descriptor(role: str, path: Path, count: int) -> dict[str, object]:
        return {
            "role": role,
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "record_count": count,
        }

    parent_record_count = len(json.loads(corpus_path.read_text(encoding="utf-8"))["documents"])
    parent_manifest = file_descriptor("parent_manifest", corpus_path, parent_record_count)
    parent_partition = {
        **file_descriptor("parent_partition", partition_path, 3),
        "unchanged": True,
    }
    input_rows = [
        {
            "path": str(path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
            "accepted_unique_documents": len(_read_records(path)),
            "checkpoint_validation": checkpoint,
        }
        for path, checkpoint in zip(accepted_paths, checkpoints, strict=True)
    ]
    corpus_documents = json.loads(corpus_path.read_text(encoding="utf-8"))["documents"]
    if not current_like_legacy_parents:
        corpus_documents = corpus_documents[:8]
    previous_documents = json.loads(partition_path.read_text(encoding="utf-8"))["documents"]
    previous_splits = {row["sha256"]: row["split"] for row in previous_documents}
    documents: list[dict[str, Any]] = []
    for raw in corpus_documents:
        prior_split = previous_splits.get(raw["sha256"])
        documents.append(
            _fixed_parent_output(
                raw,
                source_manifest=str(corpus_path.relative_to(tmp_path)),
                split="train" if prior_split in (None, "dev") else prior_split,
                previous_split=prior_split,
            )
        )
    required_new_counts = (
        {"train": 105, "validation": 84, "holdout": 254}
        if current_like_legacy_parents
        else {"train": 484, "validation": 81, "holdout": 243}
    )
    next_index = 20_000
    for split, count in required_new_counts.items():
        for _offset in range(count):
            raw = _document(next_index)
            document = _enrich(raw, "authenticated-fixture")
            document.update({
                "split": split,
                "assignment_origin": "new-v3",
                "assignment_rationale": "Synthetic authenticated output fixture.",
            })
            documents.append(document)
            next_index += 1
    fixed_count = len(corpus_documents)
    _atomic_groups(documents[:fixed_count], documents[fixed_count:], "v3")
    forced_count = 0
    conflicts: list[dict[str, Any]] = []
    for component_id, indices in _atomic_components(documents):
        fixed_members = [documents[index] for index in indices if index < fixed_count]
        new_members = [documents[index] for index in indices if index >= fixed_count]
        fixed_splits = {str(document["split"]) for document in fixed_members}
        if len(fixed_splits) > 1:
            conflicts.append({
                "family_id": component_id,
                "fixed_splits": sorted(fixed_splits, key=("train", "validation", "holdout").index),
            })
        elif fixed_members:
            forced_count += len(new_members)
    targets = document_targets(len(documents))
    checks = _verify(
        documents,
        json.loads(corpus_path.read_text(encoding="utf-8")),
        json.loads(partition_path.read_text(encoding="utf-8")),
        targets,
        parent_source_manifest=str(corpus_path.relative_to(tmp_path)),
    )
    common = {
        "schema_version": "3.0",
        "status": "final",
        "generated_date": "2026-07-31",
        "policy": _partition_policy("v3"),
        "parent_manifest": parent_manifest,
        "parent_partition": parent_partition,
        "inputs": input_rows,
        "authenticated_revision_sidecars": sidecars,
        "excluded_documents": [],
        "ideal_document_targets": targets,
        "selected_document_targets": targets,
        "ideal_targets_family_reachable": True,
        "new_documents_forced_by_existing_family": forced_count,
        "grandfathered_fixed_family_conflicts": conflicts,
        "summary": _summary(documents),
        "integrity_checks": checks,
        "selection": {
            "previously_active_documents": fixed_count,
            "new_accuracy_uninspected_documents": len(documents) - fixed_count,
            "retained_unique_documents": len(documents),
        },
        "documents": documents,
    }
    manifest_path = tmp_path / "manifest-v3.json"
    partition_output_path = tmp_path / "partition-v3.json"
    descriptor_path = tmp_path / "partition-v3.publication.json"
    _write_json(manifest_path, common)
    partition = {
        **common,
        "manifest": {
            "path": str(manifest_path.relative_to(tmp_path)),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "size_bytes": manifest_path.stat().st_size,
            "record_count": len(documents),
        },
        "output_descriptor": str(descriptor_path.relative_to(tmp_path)),
        "strata_summary": _strata_summary(documents),
    }
    _write_json(partition_output_path, partition)
    companions: list[tuple[str, Path]] = []
    for accepted_path, (sidecar_path, _sidecar_sha) in zip(accepted_paths, sidecar_pairs, strict=True):
        source_name = (
            "source_manifest.json"
            if _checkpoint_kind(accepted_path) == "financial"
            else "source_manifest.jsonl"
        )
        companions.extend([
            ("accepted", accepted_path),
            ("source", accepted_path.with_name(source_name)),
            ("rejected", accepted_path.with_name("rejected_manifest.jsonl")),
            ("summary", accepted_path.with_name("summary.json")),
            ("sidecar", sidecar_path),
        ])

    def committed_record_count(path: Path) -> int:
        if path.suffix == ".jsonl":
            return len(_read_records(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        return len(payload["sources"]) if path.name == "source_manifest.json" else 1

    authenticated_inputs = [
        file_descriptor(role, path, committed_record_count(path)) for role, path in companions
    ]
    canonical = json.dumps(
        authenticated_inputs, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode()
    descriptor: dict[str, Any] = {
        "schema_version": "2.0",
        "revision_id": "corpus-partition-v3",
        "immutable": True,
        "status": "final",
        "parents": {"manifest": parent_manifest, "partition": parent_partition},
        "input_commitment_sha256": hashlib.sha256(canonical).hexdigest(),
        "authenticated_inputs": authenticated_inputs,
        "outputs": {
            "manifest": file_descriptor("unused", manifest_path, len(documents)),
            "partition": file_descriptor("unused", partition_output_path, len(documents)),
        },
    }
    for output in descriptor["outputs"].values():
        del output["role"]
    _write_json(descriptor_path, descriptor)
    return descriptor_path


def test_final_descriptor_accepts_408_legacy_parents_and_443_strict_new_documents(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(
        tmp_path,
        current_like_legacy_parents=True,
    )
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    verified = verify_corpus_partition_descriptor(
        descriptor_path,
        root_dir=tmp_path,
        expected_descriptor_sha256=descriptor_sha,
    )
    partition_path = tmp_path / verified["outputs"]["partition"]["path"]
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    assert partition["selection"] == {
        "previously_active_documents": 408,
        "new_accuracy_uninspected_documents": 443,
        "retained_unique_documents": 851,
    }
    assert partition["output_descriptor"] == str(descriptor_path.relative_to(tmp_path))
    assert verified["schema_version"] == "2.0"


@pytest.mark.parametrize(
    "missing_field",
    ["template_family", "deduplication_keys", "accuracy_inspection_status"],
)
def test_final_descriptor_rejects_missing_new_document_schema_field(
    tmp_path: Path, missing_field: str
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def remove_new_field(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        for payload in (manifest, partition):
            payload["documents"][8].pop(missing_field)

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, remove_new_field)
    with pytest.raises(ValueError, match=f"new document missing {missing_field}"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_final_descriptor_rejects_forged_parent_origin_on_invalid_new_document(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def forge_origin_and_remove_schema(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        for payload in (manifest, partition):
            document = payload["documents"][8]
            document["assignment_origin"] = "fixed-previous"
            document.pop("template_family")

    descriptor_sha = _mutate_authenticated_outputs(
        descriptor_path,
        tmp_path,
        forge_origin_and_remove_schema,
    )
    with pytest.raises(ValueError, match="new document missing template_family"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


@pytest.mark.parametrize("variant", ["absolute", "traversal", "dot-alias", "forged"])
def test_descriptor_rejects_self_consistently_rehashed_noncanonical_embedded_path(
    tmp_path: Path, variant: str
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)
    if variant == "absolute":
        embedded_path = str(descriptor_path.resolve())
    elif variant == "traversal":
        embedded_path = f"alias/../{descriptor_path.name}"
    elif variant == "dot-alias":
        embedded_path = f"./{descriptor_path.name}"
    else:
        embedded_path = "forged/partition-v3.publication.json"

    def forge_descriptor_path(_manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        partition["output_descriptor"] = embedded_path

    descriptor_sha = _mutate_authenticated_outputs(
        descriptor_path,
        tmp_path,
        forge_descriptor_path,
    )
    with pytest.raises(ValueError, match="canonical verified descriptor path"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_descriptor_rejects_manifest_partition_embedded_descriptor_disagreement(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def forge_manifest_descriptor(manifest: dict[str, Any], _partition: dict[str, Any]) -> None:
        manifest["output_descriptor"] = "forged/descriptor.json"

    descriptor_sha = _mutate_authenticated_outputs(
        descriptor_path,
        tmp_path,
        forge_manifest_descriptor,
    )
    with pytest.raises(ValueError, match="embedded output_descriptor paths differ"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_builder_emits_canonical_descriptor_path_from_alias_output_path(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    descriptor_alias = alias_parent / ".." / "partition-v3.commit.json"
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        output_descriptor_path=descriptor_alias,
        status="provisional",
    )
    partition = json.loads(result.partition_path.read_text(encoding="utf-8"))
    assert partition["output_descriptor"] == "partition-v3.commit.json"
    descriptor_path = descriptor_alias.resolve()
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    verified = verify_corpus_partition_descriptor(
        alias_parent / ".." / descriptor_path.name,
        root_dir=alias_parent / "..",
        expected_descriptor_sha256=descriptor_sha,
    )
    assert verified["revision_id"] == "corpus-partition-v3"


def test_verifier_rejects_descriptor_outside_repository_root(tmp_path: Path) -> None:
    descriptor_path = tmp_path.parent / f"{tmp_path.name}-outside.commit.json"
    descriptor_path.write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="escapes repository root"):
            verify_corpus_partition_descriptor(
                descriptor_path,
                root_dir=tmp_path,
                expected_descriptor_sha256=hashlib.sha256(descriptor_path.read_bytes()).hexdigest(),
            )
    finally:
        descriptor_path.unlink()


def test_final_descriptor_preserves_complete_builder_parent_projection(tmp_path: Path) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    verified = verify_corpus_partition_descriptor(
        descriptor_path,
        root_dir=tmp_path,
        expected_descriptor_sha256=descriptor_sha,
    )
    assert verified["status"] == "final"


@pytest.mark.parametrize(
    "mutation",
    ["page-count", "category", "provenance", "region", "language", "nested-layout", "added"],
)
def test_final_descriptor_rejects_arbitrary_authenticated_parent_metadata_mutation(
    tmp_path: Path, mutation: str
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def mutate_parent(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        for payload in (manifest, partition):
            document = payload["documents"][3]
            if mutation == "page-count":
                document["page_count"] += 777
            elif mutation == "nested-layout":
                document["layout_signals"]["synthetic"] = False
            elif mutation == "added":
                document["unauthenticated_extra"] = {"nested": [1, 2, 3]}
            else:
                document[mutation] = f"mutated-{mutation}"
        summary = _summary(manifest["documents"])
        manifest["summary"] = summary
        partition["summary"] = summary
        partition["strata_summary"] = _strata_summary(partition["documents"])

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, mutate_parent)
    with pytest.raises(ValueError, match="authenticated parent metadata changed"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_final_descriptor_preserves_authenticated_grandfathered_family_conflict(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path, grandfathered_conflict=True)
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    verified = verify_corpus_partition_descriptor(
        descriptor_path,
        root_dir=tmp_path,
        expected_descriptor_sha256=descriptor_sha,
    )
    assert verified["revision_id"] == "corpus-partition-v3"


@pytest.mark.parametrize("tamper", ["sparse-split", "split-swap", "relabel"])
def test_final_descriptor_uses_parent_burned_membership_not_assignment_origin(
    tmp_path: Path, tamper: str
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def escape_parent(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        for payload in (manifest, partition):
            document = payload["documents"][3]
            document["assignment_origin"] = "new-v3-forged"
            if tamper in ("sparse-split", "split-swap"):
                document["split"] = "validation"
                if tamper == "split-swap":
                    payload["documents"][1]["split"] = "train"
                    payload["documents"][1]["assignment_origin"] = "new-v3-forged"
            else:
                document["id"] = "relabelled-parent-document"

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, escape_parent)
    message = "member relabeled" if tamper == "relabel" else "membership changed"
    with pytest.raises(ValueError, match=message):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_final_descriptor_recomputes_transitive_atomic_families_and_family_ids(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def forge_family_ids(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        train = next(row for row in manifest["documents"][8:] if row["split"] == "train")
        validation = next(row for row in manifest["documents"][8:] if row["split"] == "validation")
        target_ids = {train["id"], validation["id"]}
        for payload in (manifest, partition):
            for row in payload["documents"]:
                if row["id"] in target_ids:
                    row["source_producer"] = "shared report publisher"
                    row["report_series"] = "shared transitive series"
                    row["deduplication_keys"]["normalized_publisher"] = _normalize("shared report publisher")

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, forge_family_ids)
    with pytest.raises(ValueError, match="family_id does not match atomic component"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "new_documents_forced_by_existing_family",
            7,
            "forced-by-existing-family count does not match",
        ),
        (
            "grandfathered_fixed_family_conflicts",
            [{"family_id": "fabricated", "fixed_splits": ["train", "holdout"]}],
            "grandfathered fixed-family conflicts do not match",
        ),
    ],
)
def test_final_descriptor_recomputes_family_declarations(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def fabricate_declaration(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        manifest[field] = value
        partition[field] = value

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, fabricate_declaration)
    with pytest.raises(ValueError, match=message):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_final_descriptor_rejects_816_self_consistently_committed_duplicate_documents(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def duplicate_documents(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        duplicates = [manifest["documents"][0]] * 816
        manifest["documents"] = duplicates
        partition["documents"] = duplicates

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, duplicate_documents)
    with pytest.raises(ValueError, match="duplicate document IDs after consolidation"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_final_descriptor_rejects_self_consistently_committed_empty_policy(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def empty_policy(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        manifest["policy"] = {}
        partition["policy"] = {}

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, empty_policy)
    with pytest.raises(ValueError, match="output policy does not match the revision policy"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_descriptor_reconciles_all_manifest_revision_sidecar_commitments(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)

    def disagree_with_sidecar(manifest: dict[str, Any], partition: dict[str, Any]) -> None:
        for payload in (manifest, partition):
            payload["authenticated_revision_sidecars"][0]["parent_sha256"] = "0" * 64

    descriptor_sha = _mutate_authenticated_outputs(descriptor_path, tmp_path, disagree_with_sidecar)
    with pytest.raises(ValueError, match="does not match authenticated sidecar"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


def test_final_descriptor_semantically_revalidates_exact_committed_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path, tamper_checkpoint_semantics=True)
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="does not identify completed task 43"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )


@pytest.mark.parametrize("index", [1, 4])
def test_final_descriptor_reconciles_checkpoint_and_sidecar_identities(tmp_path: Path, index: int) -> None:
    descriptor_path = _authenticated_descriptor_fixture(tmp_path)
    descriptor_sha = hashlib.sha256(descriptor_path.read_bytes()).hexdigest()
    verify_corpus_partition_descriptor(
        descriptor_path,
        root_dir=tmp_path,
        expected_descriptor_sha256=descriptor_sha,
    )
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    replacement = descriptor["authenticated_inputs"][2]
    descriptor["authenticated_inputs"][index].update({
        key: replacement[key] for key in ("path", "sha256", "size_bytes", "record_count")
    })
    canonical = json.dumps(
        descriptor["authenticated_inputs"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    descriptor["input_commitment_sha256"] = hashlib.sha256(canonical).hexdigest()
    descriptor_sha = _rewrite_descriptor(descriptor_path, descriptor)
    with pytest.raises(ValueError, match=f"authenticated input identity mismatch at index {index}"):
        verify_corpus_partition_descriptor(
            descriptor_path,
            root_dir=tmp_path,
            expected_descriptor_sha256=descriptor_sha,
        )
