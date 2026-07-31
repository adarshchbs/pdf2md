from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ImmutableFileSnapshot:
    data: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def snapshot_regular_file(path: Path) -> ImmutableFileSnapshot:
    """Read one immutable byte snapshot without following or reopening symlinks."""
    initial_path_stat = path.lstat()
    if stat.S_ISLNK(initial_path_stat.st_mode):
        raise ValueError(f"source path cannot be a symlink: {path}")
    if not stat.S_ISREG(initial_path_stat.st_mode):
        raise ValueError(f"source path must be a regular file: {path}")

    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if path.is_symlink():
            raise ValueError(f"source path cannot be a symlink: {path}") from error
        raise
    try:
        before = os.fstat(descriptor)
        initial_identity = _file_identity(initial_path_stat)
        before_identity = _file_identity(before)
        if not stat.S_ISREG(before.st_mode) or before_identity != initial_identity:
            raise ValueError("source path changed before immutable snapshot read")
        data = _read_fd_bytes(descriptor)
        after_identity = _file_identity(os.fstat(descriptor))
        if after_identity != before_identity or len(data) != before_identity.size:
            raise ValueError("source file mutated during immutable snapshot read")
        final_path_stat = path.lstat()
        if stat.S_ISLNK(final_path_stat.st_mode) or _file_identity(final_path_stat) != before_identity:
            raise ValueError("source path was replaced during immutable snapshot read")
    finally:
        os.close(descriptor)
    return ImmutableFileSnapshot(data=data, sha256=hashlib.sha256(data).hexdigest())


def sha256_file(path: Path) -> str:
    return snapshot_regular_file(path).sha256


def _read_fd_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _file_identity(value: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
    )


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
