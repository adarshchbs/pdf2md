from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from benchmarks.canonical import AdapterRun, RuntimeStats


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Typed interface shared by local benchmark adapters."""

    @property
    def adapter_id(self) -> str: ...

    def process(self, pdf_path: Path) -> AdapterRun: ...


class AdapterRegistry(Mapping[str, BenchmarkAdapter]):
    """Small, lazy registry that does not import one adapter through another."""

    def __init__(self, factories: Mapping[str, Callable[[], BenchmarkAdapter]]) -> None:
        self._factories = dict(factories)

    def __getitem__(self, adapter_id: str) -> BenchmarkAdapter:
        try:
            factory = self._factories[adapter_id]
        except KeyError:
            available = ", ".join(sorted(self._factories))
            raise KeyError(f"unknown benchmark adapter {adapter_id!r}; available: {available}") from None
        return factory()

    def __iter__(self) -> Iterator[str]:
        return iter(self._factories)

    def __len__(self) -> int:
        return len(self._factories)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: Mapping[str, object]) -> str:
    payload = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def failed_run(error: Exception, wall_seconds: float) -> AdapterRun:
    """Represent an expected, input-level PDF decoding failure."""
    return AdapterRun(
        status="failed",
        pages=[],
        raw_records=[],
        runtime=RuntimeStats(wall_seconds=wall_seconds, peak_rss_bytes=None),
        error=f"{type(error).__name__}: {error}",
    )
