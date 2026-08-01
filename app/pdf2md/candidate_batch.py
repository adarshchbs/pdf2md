from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.pdf2md.engine import extract_document_with_catalog, render_document
from app.pdf2md.source_catalog import source_catalog_path, write_document_with_source_catalog

RUNNER_VERSION = "1.0.0"
SPLITS = ("train", "validation", "holdout")
JOB_TIMEOUT_SECONDS = 30 * 60


def _raise_timeout(_signum: int, _frame: object) -> None:
    raise TimeoutError("candidate extraction exceeded its runtime limit")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _versions() -> dict[str, str]:
    names = ("pdf2md", "PyMuPDF", "pymupdf", "pydantic", "pyarrow")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _worker(job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    source = Path(job["source_path"])
    split = str(job["split"])
    document_id = str(job["id"])
    output = Path(job["output_dir"])
    parquet = output / f"{document_id}.parquet"
    source_items = output / f"{document_id}.source-items.parquet"
    markdown = output / f"{document_id}.md"
    timeout_seconds = float(job.get("timeout_seconds", JOB_TIMEOUT_SECONDS))
    if timeout_seconds <= 0:
        raise ValueError("candidate extraction timeout must be positive")
    previous_handler: Any = signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    owns_outputs = False
    try:
        output.mkdir(parents=True, exist_ok=True)
        existing = next((path for path in (parquet, source_items, markdown) if path.exists()), None)
        if existing is not None:
            raise FileExistsError(existing)
        owns_outputs = True
        actual = sha256_file(source)
        if actual != job["source_sha256"]:
            raise ValueError(f"source SHA-256 changed before extraction for {document_id}")
        extracted = extract_document_with_catalog(source)
        elements = list(extracted.elements)
        temporary = parquet.with_name(f".{document_id}.{os.getpid()}.parquet")
        temporary_sidecar = source_catalog_path(temporary)
        try:
            write_document_with_source_catalog(elements, extracted.source_catalog, temporary)
            os.replace(temporary_sidecar, source_items)
            os.replace(temporary, parquet)
        except BaseException:
            temporary.unlink(missing_ok=True)
            temporary_sidecar.unlink(missing_ok=True)
            raise
        _atomic_bytes((output / f"{document_id}.md"), (render_document(elements) + "\n").encode())
        artifacts = {}
        for path in (parquet, source_items, markdown):
            artifacts[path.name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        return {
            "id": document_id,
            "split": split,
            "status": "success",
            "source_sha256": job["source_sha256"],
            "artifacts": artifacts,
            "pages": len({f.page_number for e in elements for f in e.fragments}),
            "element_count": len(elements),
            "runtime_seconds": time.monotonic() - started,
        }
    except BaseException as exc:
        if owns_outputs:
            for path in (parquet, source_items, markdown):
                path.unlink(missing_ok=True)
        return {
            "id": document_id,
            "split": split,
            "status": "error",
            "source_sha256": job["source_sha256"],
            "runtime_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def _validate_partition(root: Path, partition_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(partition_path.read_text(encoding="utf-8"))
    documents = raw.get("documents") if isinstance(raw, dict) else None
    if not isinstance(documents, list) or len(documents) != 851:
        raise ValueError(
            f"corpus-v8 must contain exactly 851 documents, got {len(documents) if isinstance(documents, list) else 'invalid'}"
        )
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in documents:
        if not isinstance(item, dict):
            raise ValueError("partition document must be an object")
        identifier, split, local, expected = (
            item.get("id"),
            item.get("split"),
            item.get("local_path"),
            item.get("sha256"),
        )
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in seen
            or Path(identifier).name != identifier
        ):
            raise ValueError(f"invalid or duplicate deterministic catalog id: {identifier}")
        if split not in SPLITS:
            raise ValueError(f"invalid split for {identifier}: {split}")
        if not isinstance(local, str) or Path(local).is_absolute() or ".." in Path(local).parts:
            raise ValueError(f"invalid local_path for {identifier}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid SHA-256 for {identifier}")
        source = (root / local).resolve()
        if source != root.joinpath(local) or not source.is_file():
            raise FileNotFoundError(f"PDF path missing or symlink alias for {identifier}: {local}")
        actual = sha256_file(source)
        if actual != expected:
            raise ValueError(f"SHA-256 differs from corpus partition for {identifier}")
        seen.add(identifier)
        jobs.append({"id": identifier, "split": split, "source_path": str(source), "source_sha256": expected})
    return sorted(jobs, key=lambda value: value["id"])


def _load_latest(status_path: Path) -> dict[str, dict[str, Any]]:
    if not status_path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[str(record["id"])] = record
    return latest


def _artifacts_match(output: Path, record: dict[str, Any]) -> bool:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != 3:
        return False
    for name, metadata in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(metadata, dict):
            return False
        expected_sha = metadata.get("sha256")
        expected_size = metadata.get("size_bytes")
        path = output / name
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or not isinstance(expected_size, int)
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected_sha
        ):
            return False
    return True


def run_candidate_batch(
    root: Path, output_root: Path, *, workers: int = 8, resume: bool = True
) -> dict[str, int]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    partition = root / "data/corpus/partition-v8.json"
    if not partition.is_file():
        raise FileNotFoundError(partition)
    jobs = _validate_partition(root.resolve(), partition.resolve())
    status_path = output_root / "status.jsonl"
    latest = _load_latest(status_path) if resume else {}
    successes = {
        identifier: record for identifier, record in latest.items() if record.get("status") == "success"
    }
    pending: list[dict[str, Any]] = []
    for job in jobs:
        job["output_dir"] = str(output_root / str(job["split"]))
        prior = successes.get(job["id"])
        if prior and _artifacts_match(output_root / str(job["split"]), prior):
            continue
        pending.append(job)
    manifest = {
        "schema_version": "candidate-batch-v1",
        "runner_version": RUNNER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "partition": "data/corpus/partition-v8.json",
        "partition_sha256": sha256_file(partition),
        "document_count": len(jobs),
        "splits": {split: sum(j["split"] == split for j in jobs) for split in SPLITS},
        "workers": workers,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": _versions(),
        "config": {"extractor": "extract_document_with_catalog", "pages": "all", "process_based": True},
    }
    _atomic_bytes(
        output_root / "manifest.json", json.dumps(manifest, sort_keys=True, indent=2).encode()
    ) if not (output_root / "manifest.json").exists() else None
    with status_path.open("a", encoding="utf-8") as status:
        context = __import__("multiprocessing").get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [pool.submit(_worker, job) for job in pending]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                status.write(json.dumps(record, sort_keys=True) + "\n")
                status.flush()
    final = _load_latest(status_path)
    return {
        "total": len(jobs),
        "resumed": len(jobs) - len(pending),
        "submitted": len(pending),
        "success": sum(record.get("status") == "success" for record in final.values()),
        "errors": sum(record.get("status") == "error" for record in final.values()),
    }
