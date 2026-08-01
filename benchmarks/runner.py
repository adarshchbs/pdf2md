from __future__ import annotations

import hashlib
import signal
import threading
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from benchmarks.adapters.base import BenchmarkAdapter
from benchmarks.canonical import AdapterRun, CanonicalPage, canonical_json, semantic_hash, write_pages_jsonl


@dataclass(frozen=True, slots=True)
class AttemptResult:
    adapter_id: str
    attempt: int
    status: str
    runtime_seconds: float
    raw_path: Path
    canonical_path: Path
    raw_sha256: str
    canonical_sha256: str
    normalized_sha256: str | None
    page_count: int
    table_count: int
    error: str | None
    pages: tuple[CanonicalPage, ...]

    def manifest_record(self, root: Path) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
            "raw_path": str(self.raw_path.relative_to(root)),
            "canonical_path": str(self.canonical_path.relative_to(root)),
            "raw_sha256": self.raw_sha256,
            "canonical_sha256": self.canonical_sha256,
            "normalized_sha256": self.normalized_sha256,
            "page_count": self.page_count,
            "table_count": self.table_count,
            "error": self.error,
        }


class AttemptTimeoutError(TimeoutError):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_adapter_attempts(
    adapter: BenchmarkAdapter,
    pdf_path: Path,
    output_dir: Path,
    *,
    attempts: int = 3,
    timeout_seconds: float = 120.0,
) -> list[AttemptResult]:
    """Run an adapter repeatedly and retain complete per-attempt artifacts.

    Exceptions and invalid adapter returns become failed attempt records rather
    than disappearing from the benchmark denominator. The destination is new by
    contract so stale output can never be mistaken for current evidence.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    input_sha256 = sha256_path(pdf_path)
    output_dir.mkdir(parents=True)
    results: list[AttemptResult] = []
    input_bytes = pdf_path.read_bytes()
    for attempt_number in range(1, attempts + 1):
        attempt_dir = output_dir / f"attempt-{attempt_number}"
        attempt_dir.mkdir()
        raw_path = attempt_dir / "raw.jsonl"
        canonical_path = attempt_dir / "canonical.jsonl"
        started = perf_counter()
        run: AdapterRun | None = None
        status = "crashed"
        error: str | None = None
        pages: list[CanonicalPage] = []
        raw_records: list[object] = []
        try:
            with _deadline(timeout_seconds):
                candidate = adapter.process(pdf_path)
            if not isinstance(candidate, AdapterRun):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("adapter process() must return AdapterRun")
            run = candidate
            pages = list(run.pages)
            raw_records = list(run.raw_records)
            status = "success" if run.status == "success" else "failed"
            error = run.error
        except AttemptTimeoutError as caught:
            status = "timeout"
            error = str(caught)
        except (TypeError, ValueError) as caught:
            status = "invalid"
            error = f"{type(caught).__name__}: {caught}"
        except Exception as caught:  # The harness must retain unexpected crashes as evidence.
            status = "crashed"
            error = f"{type(caught).__name__}: {caught}"
        finally:
            if not pdf_path.is_file() or sha256_path(pdf_path) != input_sha256:
                status = "invalid"
                error = _append_error(error, "adapter mutated the immutable input PDF")
                pdf_path.write_bytes(input_bytes)

        try:
            _write_raw_jsonl(raw_path, raw_records, status=status, error=error)
        except (TypeError, ValueError) as caught:
            status = "invalid"
            error = _append_error(error, f"raw serialization failed: {type(caught).__name__}: {caught}")
            _write_raw_jsonl(raw_path, [], status=status, error=error)

        try:
            if pages:
                write_pages_jsonl(pages, canonical_path)
            else:
                canonical_path.touch(exist_ok=False)
            normalized_sha256 = semantic_hash(pages) if status == "success" and pages else None
        except (TypeError, ValueError) as caught:
            status = "invalid"
            error = _append_error(
                error,
                f"canonical serialization failed: {type(caught).__name__}: {caught}",
            )
            pages = []
            canonical_path.unlink(missing_ok=True)
            canonical_path.touch(exist_ok=False)
            normalized_sha256 = None

        runtime_seconds = run.runtime.wall_seconds if run is not None else perf_counter() - started
        results.append(
            AttemptResult(
                adapter_id=adapter.adapter_id,
                attempt=attempt_number,
                status=status,
                runtime_seconds=runtime_seconds,
                raw_path=raw_path,
                canonical_path=canonical_path,
                raw_sha256=sha256_path(raw_path),
                canonical_sha256=sha256_path(canonical_path),
                normalized_sha256=normalized_sha256,
                page_count=len(pages),
                table_count=sum(len(page.tables) for page in pages),
                error=error,
                pages=tuple(pages),
            )
        )
    return results


def attempts_are_deterministic(results: Sequence[AttemptResult]) -> bool:
    return (
        len(results) >= 2
        and all(result.status == "success" for result in results)
        and len({result.normalized_sha256 for result in results}) == 1
    )


def missing_artifacts(results: Sequence[AttemptResult]) -> list[Path]:
    missing: list[Path] = []
    for result in results:
        for path in (result.raw_path, result.canonical_path):
            if not path.is_file():
                missing.append(path)
    return missing


def _append_error(current: str | None, message: str) -> str:
    return message if current is None else f"{current}; {message}"


def _write_raw_jsonl(
    path: Path,
    records: Sequence[object],
    *,
    status: str,
    error: str | None,
) -> None:
    materialized = list(records)
    if not materialized:
        materialized.append({"record_type": "attempt_outcome", "status": status, "error": error})
    lines = [canonical_json(record) for record in materialized]  # type: ignore[arg-type]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")


@contextmanager
def _deadline(seconds: float) -> Generator[None]:
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("in-process timeout enforcement requires POSIX signals on the main thread")

    def expire(_signum: int, _frame: object) -> None:
        raise AttemptTimeoutError(f"adapter exceeded {seconds:g} second timeout")

    previous_handler = signal.signal(signal.SIGALRM, expire)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
