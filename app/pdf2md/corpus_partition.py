from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

Split = Literal["train", "validation", "holdout"]
SPLITS: tuple[Split, ...] = ("train", "validation", "holdout")
RATIOS: dict[Split, float] = {"train": 0.60, "validation": 0.10, "holdout": 0.30}
SALTS = {
    "v3": "pdf2md/corpus-partition/v3/2026-07-31",
    "v4": "pdf2md/corpus-partition/v4/2026-07-31",
    "v5": "pdf2md/corpus-partition/v5/2026-07-31",
}
_LAYOUTLM = re.compile(r"layout\s*lm(?:v?2)?", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_PERCENT_ESCAPE = re.compile(r"%([0-9a-fA-F]{2})")
_UNRESERVED_URL_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~")
_TRACKING_QUERY_PARAMETERS = frozenset({
    "_ga",
    "dclid",
    "fbclid",
    "gclid",
    "jsessionid",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "phpsessid",
    "session",
    "sessionid",
})
_FORBIDDEN_EVIDENCE_KEYS = {
    "accuracy_inspected",
    "bronze_inspected",
    "candidate_output_inspected",
    "element_failures_inspected",
    "evaluation_inspected",
    "reference_inspected",
    "silver_inspected",
}


@dataclass(frozen=True)
class BuildResult:
    manifest_path: Path
    partition_path: Path
    descriptor_path: Path | None
    document_counts: dict[Split, int]
    page_counts: dict[Split, int]
    status: str


@dataclass
class _Group:
    family_id: str
    documents: list[dict[str, Any]]

    @property
    def count(self) -> int:
        return len(self.documents)

    @property
    def pages(self) -> int:
        return sum(int(document["page_count"]) for document in self.documents)


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    data: bytes
    sha256: str
    size_bytes: int
    descriptor: int
    identity: tuple[int, int, int, int]


def _cleanup_cause(primary_error: BaseException, cleanup_error: BaseException) -> BaseException:
    existing_cause = primary_error.__cause__
    if existing_cause is None:
        return cleanup_error
    return BaseExceptionGroup(
        f"primary error cleanup failures: {existing_cause}; {cleanup_error}",
        [existing_cause, cleanup_error],
    )


_FINALIZER_CLOSE_DESCRIPTOR = os.close


class _SnapshotStore:
    """Hold stable, shared-locked descriptors for every metadata input."""

    def __init__(self) -> None:
        self._snapshots: dict[Path, _Snapshot] = {}

    @staticmethod
    def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    @staticmethod
    def _read_descriptor(descriptor: int, size: int) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while offset < size:
            chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)

    def read(self, path: Path) -> _Snapshot:
        resolved = path.resolve()
        existing = self._snapshots.get(resolved)
        if existing is not None:
            return existing
        descriptor = os.open(resolved, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            initial = os.fstat(descriptor)
            data = self._read_descriptor(descriptor, initial.st_size)
            opened = os.fstat(descriptor)
            current = resolved.stat()
            identity = self._identity(initial)
            if identity != self._identity(opened) or identity != self._identity(current):
                raise RuntimeError(f"metadata input changed while being read: {resolved}")
            if len(data) != initial.st_size:
                raise RuntimeError(f"metadata input was truncated while being read: {resolved}")
            snapshot = _Snapshot(
                resolved,
                data,
                hashlib.sha256(data).hexdigest(),
                len(data),
                descriptor,
                identity,
            )
            self._snapshots[resolved] = snapshot
            return snapshot
        except BaseException as acquisition_error:
            try:
                os.close(descriptor)
            except BaseException as close_error:
                close_error.add_note(f"unclosed acquisition descriptor {descriptor}: {resolved}")
                raise acquisition_error from _cleanup_cause(acquisition_error, close_error)
            raise

    def revalidate(self) -> None:
        for snapshot in self._snapshots.values():
            try:
                before = os.fstat(snapshot.descriptor)
                current = self._read_descriptor(snapshot.descriptor, before.st_size)
                after = os.fstat(snapshot.descriptor)
                path_stat = snapshot.path.stat()
            except (FileNotFoundError, OSError) as error:
                raise RuntimeError(
                    f"metadata input disappeared before publication: {snapshot.path}"
                ) from error
            if (
                self._identity(before) != snapshot.identity
                or self._identity(after) != snapshot.identity
                or self._identity(path_stat) != snapshot.identity
                or current != snapshot.data
            ):
                raise RuntimeError(f"metadata input changed before publication: {snapshot.path}")

    def close(self) -> None:
        owned_snapshots, self._snapshots = self._snapshots, {}
        failures: list[BaseException] = []
        for snapshot in owned_snapshots.values():
            try:
                os.close(snapshot.descriptor)
            except BaseException as error:
                error.add_note(f"snapshot descriptor {snapshot.descriptor}: {snapshot.path}")
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("snapshot descriptor close failures", failures)

    def __del__(self) -> None:
        try:
            owned_snapshots, self._snapshots = self._snapshots, {}
        except BaseException:
            return
        for snapshot in owned_snapshots.values():
            try:
                _FINALIZER_CLOSE_DESCRIPTOR(snapshot.descriptor)
            except BaseException:
                pass


def _close_snapshot_store(snapshots: _SnapshotStore, primary_error: BaseException | None) -> None:
    try:
        snapshots.close()
    except BaseException as close_error:
        if primary_error is not None:
            raise primary_error from _cleanup_cause(primary_error, close_error)
        raise


def _decode_json(snapshot: _Snapshot) -> Any:
    try:
        return json.loads(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON metadata: {snapshot.path}: {error}") from error


def _json_from_snapshot(snapshot: _Snapshot) -> dict[str, Any]:
    value = _decode_json(snapshot)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {snapshot.path}")
    return cast(dict[str, Any], value)


def _records_from_snapshot(snapshot: _Snapshot) -> list[dict[str, Any]]:
    path = snapshot.path
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        try:
            text = snapshot.data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"invalid UTF-8 metadata: {path}: {error}") from error
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            records.append(cast(dict[str, Any], value))
        return records
    manifest = _json_from_snapshot(snapshot)
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"manifest has no documents array: {path}")
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"expected JSON object at {path}:documents[{index}]")
    return [cast(dict[str, Any], document) for document in documents]


def _read_json(path: Path) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    snapshots = _SnapshotStore()
    primary_error: BaseException | None = None
    try:
        return _json_from_snapshot(snapshots.read(path))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_snapshot_store(snapshots, primary_error)


def _read_records(path: Path) -> list[dict[str, Any]]:  # pyright: ignore[reportUnusedFunction]
    snapshots = _SnapshotStore()
    primary_error: BaseException | None = None
    try:
        return _records_from_snapshot(snapshots.read(path))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_snapshot_store(snapshots, primary_error)


def _normalized(value: str) -> str:
    return _NON_WORD.sub(" ", value.lower()).strip()


def _normalize_url_path(raw_path: str) -> str:
    def normalize_escape(match: re.Match[str]) -> str:
        value = int(match.group(1), 16)
        character = chr(value)
        return character if character in _UNRESERVED_URL_CHARACTERS else f"%{value:02X}"

    normalized_escapes = _PERCENT_ESCAPE.sub(normalize_escape, raw_path)
    return quote(normalized_escapes, safe="/%:@!$&'()*+,;=-._~").rstrip("/")


def _normalize_url_value(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip())
    semantic_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_key = key.casefold()
        if normalized_key.startswith("utm_") or normalized_key in _TRACKING_QUERY_PARAMETERS:
            continue
        semantic_query.append((key, value))
    semantic_query.sort()
    return urlunsplit((
        parsed.scheme.casefold(),
        parsed.netloc.casefold(),
        _normalize_url_path(parsed.path),
        urlencode(semantic_query, doseq=True, quote_via=quote),
        "",
    ))


def _derived_normalized_url(document: dict[str, Any]) -> str:
    raw_url = str(document.get("source_url") or document.get("canonical_url") or "")
    return _normalize_url_value(raw_url) if raw_url.strip() else ""


def _derived_title_publisher_key(document: dict[str, Any]) -> tuple[str, str]:
    title = str(document.get("normalized_title") or document.get("title") or "")
    return _producer(document), _normalized(title)


def _normalized_url(document: dict[str, Any]) -> str:
    deduplication_keys = document.get("deduplication_keys")
    curated_url = deduplication_keys.get("normalized_url") if isinstance(deduplication_keys, dict) else None
    raw_url = str(document.get("source_url") or document.get("canonical_url") or curated_url or "").strip()
    return _normalize_url_value(raw_url) if raw_url else ""


def _is_layoutlm(document: dict[str, Any]) -> bool:
    searchable = json.dumps(document, ensure_ascii=True, sort_keys=True)
    return _LAYOUTLM.search(searchable) is not None


def _canonical_producer(value: str) -> str:
    normalized = _normalized(value).removeprefix("us ")
    aliases = {
        "arxiv org": "arxiv",
        "ctan org": "ctan",
        "fema gov": "fema",
        "ipcc ch": "ipcc",
        "irs gov": "irs",
        "nist gov": "nist",
        "uscis gov": "uscis",
    }
    return aliases.get(normalized, normalized) or "unknown"


def _producer(document: dict[str, Any]) -> str:
    explicit = (
        document.get("source_producer")
        or document.get("producer")
        or document.get("issuer")
        or document.get("jurisdiction_or_publisher")
    )
    if explicit:
        return _canonical_producer(str(explicit).split("/")[0].strip())
    for key in ("canonical_url", "source_url"):
        hostname = urlsplit(str(document.get(key, ""))).hostname
        if hostname:
            parts = hostname.removeprefix("www.").split(".")
            domain = ".".join(parts[-2:]) if len(parts) >= 2 else parts[0]
            return _canonical_producer(domain)
    provenance = _normalized(str(document.get("provenance", "")))
    return _canonical_producer(" ".join(provenance.split()[:4]))


def _title_family(document: dict[str, Any]) -> str:
    title = _YEAR.sub("", str(document.get("normalized_title") or document.get("title") or ""))
    title = re.sub(r"\b(?:appendix|archive document|full report|revision|rev)\b", "", title, flags=re.I)
    return _normalized(title)


def _template_key(document: dict[str, Any]) -> str:
    title = _title_family(document)
    form_match = re.search(r"\bform\s+([a-z]*-?\d+[a-z]*)\b", title)
    if form_match:
        return f"form-{form_match.group(1)}"
    ics_match = re.search(r"\bics\s+(?:form\s+)?(\d+)\b", title)
    if ics_match:
        return f"ics-{ics_match.group(1)}"
    document_type = _normalized(str(document.get("document_type", "")))
    if document.get("issuer") and document_type:
        return document_type
    return title or _normalized(str(document.get("category", "unknown")))


def _family_id(document: dict[str, Any]) -> str:
    curated = document.get("family_id") or document.get("template_family")
    if curated:
        return str(curated)
    return f"publisher-template:{_producer(document)}|{_template_key(document)}"


def _atomic_keys(document: dict[str, Any]) -> tuple[str, ...]:
    producer = _producer(document)
    keys = {
        f"sha256:{document.get('sha256', '')}",
        f"url:{_normalized_url(document)}",
        (
            f"publisher-title:{producer}|"
            f"{_normalized(str(document.get('normalized_title') or document.get('title') or ''))}"
        ),
        (
            f"publisher-template:{producer}|"
            f"{_normalized(str(document.get('template_family') or document.get('family_id') or ''))}"
        ),
    }
    report_series = _normalized(str(document.get("report_series") or ""))
    edition = _normalized(str(document.get("edition") or ""))
    if report_series:
        keys.add(f"publisher-report-series:{producer}|{report_series}")
    if edition:
        keys.add(f"producer-edition:{producer}|{edition}")
    return tuple(sorted(key for key in keys if not key.endswith(":")))


def _atomic_components(documents: list[dict[str, Any]]) -> list[tuple[str, list[int]]]:
    parent = list(range(len(documents)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    key_owner: dict[str, int] = {}
    for index, document in enumerate(documents):
        for key in _atomic_keys(document):
            owner = key_owner.setdefault(key, index)
            union(index, owner)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(documents)):
        components[find(index)].append(index)
    ordered = sorted(components.values(), key=lambda values: min(str(documents[i]["sha256"]) for i in values))
    return [
        (
            "atomic-v3:"
            + hashlib.sha256(
                "|".join(sorted(str(documents[index]["sha256"]) for index in indices)).encode()
            ).hexdigest()[:24],
            indices,
        )
        for indices in ordered
    ]


def _family_declarations(
    documents: list[dict[str, Any]], fixed_hashes: set[str]
) -> tuple[int, list[dict[str, Any]]]:
    forced_count = 0
    conflicts: list[dict[str, Any]] = []
    for component_id, indices in _atomic_components(documents):
        fixed_members = [documents[index] for index in indices if documents[index]["sha256"] in fixed_hashes]
        new_members = [
            documents[index] for index in indices if documents[index]["sha256"] not in fixed_hashes
        ]
        fixed_splits = {str(document["split"]) for document in fixed_members}
        if len(fixed_splits) > 1:
            conflicts.append({
                "family_id": component_id,
                "fixed_splits": sorted(fixed_splits, key=SPLITS.index),
            })
        elif fixed_members:
            forced_count += len(new_members)
    return forced_count, conflicts


def _atomic_groups(
    fixed: list[dict[str, Any]], new_documents: list[dict[str, Any]], revision: str
) -> tuple[list[dict[str, Any]], list[_Group]]:
    documents = fixed + new_documents
    forced_new: list[dict[str, Any]] = []
    variable_groups: list[_Group] = []
    fixed_count = len(fixed)
    for component_id, indices in _atomic_components(documents):
        for index in indices:
            document = documents[index]
            document.setdefault("source_family_id", str(document.get("family_id", "")))
            document["family_id"] = component_id
        new_members = [documents[index] for index in indices if index >= fixed_count]
        if not new_members:
            continue
        fixed_splits = {cast(Split, documents[index]["split"]) for index in indices if index < fixed_count}
        if len(fixed_splits) > 1:
            raise ValueError(
                f"new documents link to a grandfathered family spanning fixed splits: {component_id}"
            )
        if fixed_splits:
            forced_split = next(iter(fixed_splits))
            for document in new_members:
                document.update({
                    "split": forced_split,
                    "assignment_origin": f"new-{revision}-existing-family",
                    "assignment_rationale": (
                        "Accuracy-uninspected document inherited its atomic prior-family split."
                    ),
                })
            forced_new.extend(new_members)
        else:
            variable_groups.append(_Group(component_id, new_members))
    return forced_new, variable_groups


def _page_band(page_count: int) -> str:
    if page_count <= 2:
        return "1-2"
    if page_count <= 10:
        return "3-10"
    if page_count <= 50:
        return "11-50"
    if page_count <= 200:
        return "51-200"
    return "201+"


def _bool_signal(value: object) -> str:
    if value is True or (isinstance(value, int) and value > 0):
        return "present"
    if value is False or value == 0:
        return "absent"
    return "unknown"


def _enrich(document: dict[str, Any], source_manifest: str) -> dict[str, Any]:
    enriched = dict(document)
    indicators = document.get("indicators")
    signal = cast(dict[str, Any], indicators) if isinstance(indicators, dict) else document
    page_count = int(document["page_count"])
    country = str(document.get("country") or "unspecified")
    region = str(document.get("region") or "unspecified")
    tags = [str(tag) for tag in document.get("tags", [])] if isinstance(document.get("tags"), list) else []
    landscape_count = signal.get("landscape_pages", signal.get("landscape_page_count", 0))
    layout = "unknown"
    if signal.get("has_mixed_orientation") is True or int(landscape_count or 0) > 0:
        layout = "mixed-or-landscape"
    elif any(tag.startswith("layout:") for tag in tags):
        layout = next(tag for tag in tags if tag.startswith("layout:")).split(":", 1)[1]
    table_count = signal.get("tables_in_sample", signal.get("sampled_detected_table_count"))
    rotation_count = signal.get("rotated_pages", signal.get("rotated_page_count"))
    form_count = signal.get("form_field_count")
    if form_count is None and "form" in _normalized(str(document.get("document_type", ""))):
        form_count = 1
    document_stratum = str(
        document.get("document_type") or document.get("category") or document.get("stratum") or "unknown"
    )
    enriched.update({
        "source_manifest": source_manifest,
        "source_producer": _producer(document),
        "family_id": _family_id(document),
        "title_family": _title_family(document),
        "country": country,
        "region": region,
        "country_region": f"{country} / {region}",
        "document_stratum": document_stratum,
        "language_mix": str(document.get("language_mix") or document.get("language") or "unspecified"),
        "page_count_band": _page_band(page_count),
        "rotation_signal": _bool_signal(rotation_count),
        "layout_signal": layout,
        "table_signal": _bool_signal(table_count),
        "form_signal": _bool_signal(form_count),
        "difficult_stratum": str(
            document.get("failure_stratum_similarity") or document.get("stratum") or "not-staged-difficult"
        ),
    })
    return enriched


def _checkpoint_kind(path: Path) -> Literal["difficult", "financial", "gap"] | None:
    if path.name != "accepted_manifest.jsonl":
        return None
    if path.parent.name == "v3-difficult":
        return "difficult"
    if path.parent.name in ("v3-financial", "v3-financial-r2", "v3-financial-r3"):
        return "financial"
    if path.parent.name == "v3-gap":
        return "gap"
    return None


def _load_jsonl(path: Path, snapshots: _SnapshotStore) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"checkpoint companion is missing: {path}")
    return _records_from_snapshot(snapshots.read(path))


def _validate_source_binding(
    *,
    kind: str,
    outcome_kind: str,
    identity: object,
    source: dict[str, Any],
    outcome: dict[str, Any],
) -> None:
    missing = sorted(source.keys() - outcome.keys())
    if missing:
        raise ValueError(f"v3 {kind} {outcome_kind} outcome omits source metadata {identity}: {missing}")
    altered = sorted(key for key, value in source.items() if outcome[key] != value)
    if altered:
        raise ValueError(f"v3 {kind} {outcome_kind} outcome alters source metadata {identity}: {altered}")


def _validate_checkpoint_with_snapshots(
    path: Path,
    records: list[dict[str, Any]],
    snapshots: _SnapshotStore,
) -> dict[str, Any] | None:
    kind = _checkpoint_kind(path)
    if kind is None:
        return None
    summary_path = path.with_name("summary.json")
    source_path = path.with_name(
        "source_manifest.jsonl" if kind in ("difficult", "gap") else "source_manifest.json"
    )
    rejected_path = path.with_name("rejected_manifest.jsonl")
    try:
        accepted_snapshot = snapshots.read(path)
        summary_snapshot = snapshots.read(summary_path)
        source_snapshot = snapshots.read(source_path)
        snapshots.read(rejected_path)
    except FileNotFoundError as error:
        raise ValueError(f"checkpoint companion is missing: {error.filename}") from error
    if records != _records_from_snapshot(accepted_snapshot):
        raise ValueError("accepted checkpoint records do not match its stable byte snapshot")
    if summary_snapshot.identity[3] < accepted_snapshot.identity[3]:
        raise ValueError(f"checkpoint summary is stale relative to accepted manifest: {path}")
    if accepted_snapshot.identity[3] < source_snapshot.identity[3]:
        raise ValueError(f"accepted manifest is stale relative to source manifest: {path}")

    summary = _json_from_snapshot(summary_snapshot)
    rejected = _load_jsonl(rejected_path, snapshots)
    if kind in ("difficult", "gap"):
        sources = _load_jsonl(source_path, snapshots)
        expected_task = 43 if kind == "difficult" else 48
        if summary.get("task") != expected_task:
            raise ValueError(f"v3 {kind} checkpoint does not identify completed task {expected_task}")
        expected = (
            int(summary.get("source_count", -1)),
            int(summary.get("accepted_count", -1)),
            int(summary.get("rejected_count", -1)),
        )
        if expected != (len(sources), len(records), len(rejected)):
            raise ValueError(f"v3 {kind} checkpoint completeness counts do not match")
        source_by_slug = {str(row.get("slug", "")).strip(): row for row in sources}
        accepted_by_slug = {str(row.get("slug", "")).strip(): row for row in records}
        rejected_by_slug = {str(row.get("slug", "")).strip(): row for row in rejected}
        for label, rows, by_slug in (
            ("source", sources, source_by_slug),
            ("accepted", records, accepted_by_slug),
            ("rejected", rejected, rejected_by_slug),
        ):
            if "" in by_slug:
                raise ValueError(f"v3 {kind} {label} manifest contains a blank slug")
            if len(by_slug) != len(rows):
                raise ValueError(f"v3 {kind} {label} manifest contains duplicate slugs")
        if accepted_by_slug.keys() & rejected_by_slug.keys():
            raise ValueError(f"v3 {kind} accepted and rejected outcomes are not disjoint")
        if source_by_slug.keys() != accepted_by_slug.keys() | rejected_by_slug.keys():
            raise ValueError(f"v3 {kind} outcomes do not exactly cover source slugs")
        for outcome_kind, outcomes in (
            ("accepted", accepted_by_slug),
            ("rejected", rejected_by_slug),
        ):
            for slug, outcome in outcomes.items():
                _validate_source_binding(
                    kind=kind,
                    outcome_kind=outcome_kind,
                    identity=slug,
                    source=source_by_slug[slug],
                    outcome=outcome,
                )
        if any(row.get("status") != "rejected" for row in rejected):
            raise ValueError(f"v3 {kind} rejected checkpoint contains a non-rejected record")
        if int(summary.get("accepted_page_count", -1)) != sum(int(row["page_count"]) for row in records):
            raise ValueError(f"v3 {kind} checkpoint page total does not match")
        if int(summary.get("accepted_size_bytes", -1)) != sum(int(row["size_bytes"]) for row in records):
            raise ValueError(f"v3 {kind} checkpoint byte total does not match")
        scope_guard = str(summary.get("scope_guard", "")).casefold()
        if "accuracy" not in scope_guard or "validation" not in scope_guard:
            raise ValueError(f"v3 {kind} checkpoint lacks the required blind-scope attestation")
        if any(row.get("assignment") != "unassigned" for row in records):
            raise ValueError(f"v3 {kind} checkpoint contains an assigned candidate")
        if kind == "gap" and any(
            row.get("candidate_status") != "unassigned" or row.get("accuracy_inspected") is not False
            for row in records
        ):
            raise ValueError("v3 gap checkpoint lacks record-level uninspected/unassigned attestations")
    else:
        source = _json_from_snapshot(snapshots.read(source_path))
        sources_value = source.get("sources")
        if not isinstance(sources_value, list):
            raise ValueError("v3 financial source manifest has no sources array")
        for index, row in enumerate(sources_value):
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {source_path}:sources[{index}]")
        sources = [cast(dict[str, Any], row) for row in sources_value]
        if source.get("candidate_status") != "unassigned" or source.get("accuracy_inspected") is not False:
            raise ValueError("v3 financial checkpoint lacks an uninspected/unassigned attestation")
        actual_counts = (len(sources), len(records), len(rejected))
        summary_counts = (
            int(summary.get("source_count", -1)),
            int(summary.get("accepted", -1)),
            int(summary.get("rejected", -1)),
        )
        if summary_counts != actual_counts or len(sources) != len(records) + len(rejected):
            raise ValueError(
                "v3 financial checkpoint completeness counts do not match "
                f"source/accepted/rejected records: {actual_counts}"
            )

        def financial_identity(row: dict[str, Any]) -> tuple[str, str]:
            canonical_url = _normalize_url_value(str(row.get("canonical_url") or row.get("source_url") or ""))
            return canonical_url, str(row.get("id", "")).strip()

        source_by_url = {financial_identity(row): row for row in sources}
        accepted_by_url = {financial_identity(row): row for row in records}
        rejected_by_url = {financial_identity(row): row for row in rejected}
        for label, rows, by_url in (
            ("source", sources, source_by_url),
            ("accepted", records, accepted_by_url),
            ("rejected", rejected, rejected_by_url),
        ):
            if any(not url or not source_id for url, source_id in by_url):
                raise ValueError(f"v3 financial {label} manifest contains a blank canonical identity")
            if len(by_url) != len(rows):
                raise ValueError(f"v3 financial {label} manifest contains duplicate canonical identities")
        if accepted_by_url.keys() & rejected_by_url.keys():
            raise ValueError("v3 financial accepted and rejected outcomes are not disjoint")
        if source_by_url.keys() != accepted_by_url.keys() | rejected_by_url.keys():
            raise ValueError("v3 financial outcomes do not exactly cover source canonical URLs")
        for outcome_kind, outcomes in (
            ("accepted", accepted_by_url),
            ("rejected", rejected_by_url),
        ):
            for canonical_identity, outcome in outcomes.items():
                _validate_source_binding(
                    kind="financial",
                    outcome_kind=outcome_kind,
                    identity=canonical_identity,
                    source=source_by_url[canonical_identity],
                    outcome=outcome,
                )
        if any(row.get("status") != "rejected" for row in rejected):
            raise ValueError("v3 financial rejected checkpoint contains a non-rejected record")
        if any(not str(row.get("rejection_reason", "")).strip() for row in rejected):
            raise ValueError("v3 financial rejected outcome lacks a rejection reason")
        if int(summary.get("pages", -1)) != sum(int(row["page_count"]) for row in records):
            raise ValueError("v3 financial checkpoint page total does not match")
        if int(summary.get("size_bytes", -1)) != sum(int(row["size_bytes"]) for row in records):
            raise ValueError("v3 financial checkpoint byte total does not match")
        ordered_hash_digest = hashlib.sha256(
            "".join(str(row["sha256"]) for row in records).encode()
        ).hexdigest()
        if summary.get("sha256_manifest") != ordered_hash_digest:
            raise ValueError("v3 financial checkpoint ordered hash digest does not match")
        if any(row.get("candidate_status") != "unassigned" for row in records):
            raise ValueError("v3 financial checkpoint contains an assigned candidate")
    if any(row.get("status") != "accepted" for row in records):
        raise ValueError(f"{kind} accepted checkpoint contains a non-accepted record")
    return {
        "kind": kind,
        "summary_path": str(summary_path),
        "summary_sha256": snapshots.read(summary_path).sha256,
        "source_path": str(source_path),
        "source_sha256": snapshots.read(source_path).sha256,
        "rejected_path": str(rejected_path),
        "rejected_sha256": snapshots.read(rejected_path).sha256,
    }


def _validate_checkpoint(
    path: Path,
    records: list[dict[str, Any]],
    snapshots: _SnapshotStore | None = None,
) -> dict[str, Any] | None:
    if snapshots is not None:
        return _validate_checkpoint_with_snapshots(path, records, snapshots)
    owned_snapshots = _SnapshotStore()
    primary_error: BaseException | None = None
    try:
        return _validate_checkpoint_with_snapshots(path, records, owned_snapshots)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_snapshot_store(owned_snapshots, primary_error)


def _adapt_checkpoint_document(document: dict[str, Any], path: Path) -> dict[str, Any]:
    kind = _checkpoint_kind(path)
    if kind is None:
        return document
    adapted = dict(document)
    if kind in ("difficult", "gap"):
        adapted.update({
            "id": str(document["slug"]),
            "canonical_url": _normalize_url_value(str(document["source_url"])),
            "normalized_title": _normalized(str(document["title"])),
            "provenance": str(document["validation_note"]),
            "region": str(document.get("region") or "unspecified"),
            "country": str(document.get("country") or "unspecified"),
            "year": str(document.get("year") or "unspecified"),
            "language": str(document.get("language") or "unspecified"),
            "category": str(document["stratum"]),
            "document_type": str(document["stratum"]),
            "indicators": dict(document),
            "accuracy_inspection_status": "uninspected",
        })
    else:
        adapted.update({
            "producer": str(document["issuer"]),
            "language": "unspecified",
            "category": str(document["document_type"]),
            "accuracy_inspection_status": "uninspected",
        })
    publisher, title = _derived_title_publisher_key(adapted)
    adapted["deduplication_keys"] = {
        "sha256": str(adapted["sha256"]),
        "normalized_url": _derived_normalized_url(adapted),
        "normalized_title": title,
        "normalized_publisher": publisher,
        "template_family": str(adapted["template_family"]),
    }
    return adapted


def _validate_new_document(document: dict[str, Any], source: Path) -> None:
    required = (
        "sha256",
        "local_path",
        "page_count",
        "size_bytes",
        "title",
        "source_url",
        "provenance",
        "region",
        "year",
        "template_family",
        "deduplication_keys",
        "accuracy_inspection_status",
    )
    missing = [key for key in required if document.get(key) in (None, "")]
    if not any(
        document.get(key) not in (None, "")
        for key in ("source_producer", "producer", "issuer", "jurisdiction_or_publisher")
    ):
        missing.append("producer/issuer/jurisdiction_or_publisher")
    if not any(document.get(key) not in (None, "") for key in ("language", "language_mix")):
        missing.append("language/language_mix")
    if not any(
        document.get(key) not in (None, "", [], {}) for key in ("indicators", "layout_signals", "tags")
    ):
        missing.append("indicators/layout_signals/tags")
    if missing:
        raise ValueError(f"{source}: new document missing {', '.join(missing)}")
    if document["accuracy_inspection_status"] != "uninspected":
        raise ValueError(f"{source}: new document must explicitly attest uninspected accuracy status")
    deduplication_keys = document["deduplication_keys"]
    if not isinstance(deduplication_keys, dict):
        raise ValueError(f"{source}: deduplication_keys must be an object")
    derived_publisher, derived_title = _derived_title_publisher_key(document)
    expected_deduplication_keys = {
        "sha256": str(document["sha256"]),
        "normalized_url": _derived_normalized_url(document),
        "normalized_title": derived_title,
        "normalized_publisher": derived_publisher,
        "template_family": str(document["template_family"]),
    }
    if any(deduplication_keys.get(key) != value for key, value in expected_deduplication_keys.items()):
        raise ValueError(f"{source}: deduplication_keys do not match the accepted record")
    sha256 = str(document["sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError(f"{source}: invalid SHA-256 for {document.get('id', document['title'])}")
    if int(document["page_count"]) <= 0:
        raise ValueError(f"{source}: non-positive page count for {document['title']}")
    status = document.get("status", "accepted")
    if status != "accepted":
        raise ValueError(f"{source}: non-accepted document supplied: {document['title']}")
    invalid_evidence_flags = [
        key for key in _FORBIDDEN_EVIDENCE_KEYS if key in document and document[key] is not False
    ]
    if invalid_evidence_flags:
        raise ValueError(
            f"{source}: forbidden evidence flags must be boolean false: {invalid_evidence_flags}"
        )


def document_targets(total: int) -> dict[Split, int]:
    """Round TRAIN and validation half-up, then assign the residual to holdout."""
    if total < 0:
        raise ValueError("total document count cannot be negative")

    def half_up(value: Fraction) -> int:
        quotient, remainder = divmod(value.numerator, value.denominator)
        return quotient + int(remainder * 2 >= value.denominator)

    train = half_up(total * Fraction(3, 5))
    validation = half_up(total * Fraction(1, 10))
    return {"train": train, "validation": validation, "holdout": total - train - validation}


def _reachable_targets(
    fixed_counts: dict[Split, int], groups: list[_Group], ideal: dict[Split, int]
) -> tuple[dict[Split, int], bool]:
    # Dynamic programming proves whether family-safe exact targets are reachable.
    states: set[tuple[int, int]] = {(fixed_counts["train"], fixed_counts["validation"])}
    processed = 0
    for group in groups:
        processed += group.count
        next_states: set[tuple[int, int]] = set()
        for train, validation in states:
            next_states.add((train + group.count, validation))
            next_states.add((train, validation + group.count))
            next_states.add((train, validation))
        states = next_states
    total = sum(fixed_counts.values()) + processed
    ideal_pair = (ideal["train"], ideal["validation"])
    if ideal_pair in states:
        holdout = total - sum(ideal_pair)
        if holdout == ideal["holdout"]:
            return dict(ideal), True

    def score(state: tuple[int, int]) -> tuple[float, int, int]:
        counts = (state[0], state[1], total - state[0] - state[1])
        squared = sum(((counts[index] / total) - RATIOS[split]) ** 2 for index, split in enumerate(SPLITS))
        return squared, state[0], state[1]

    train, validation = min(states, key=score)
    return {"train": train, "validation": validation, "holdout": total - train - validation}, False


def _strata(document: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    dimensions = (
        "region",
        "country",
        "document_stratum",
        "category",
        "difficult_stratum",
        "language_mix",
        "source_producer",
        "source_manifest",
        "page_count_band",
        "rotation_signal",
        "table_signal",
        "form_signal",
        "layout_signal",
    )
    return tuple((dimension, str(document.get(dimension, "unknown"))) for dimension in dimensions)


def _optimize(
    fixed: list[dict[str, Any]], groups: list[_Group], targets: dict[Split, int], salt: str
) -> dict[str, Split]:
    group_count = len(groups)
    x_count = group_count * len(SPLITS)
    forced_new = [document for document in fixed if str(document["assignment_origin"]).startswith("new-v")]
    prior_fixed = [
        document for document in fixed if not str(document["assignment_origin"]).startswith("new-v")
    ]
    fixed_strata: dict[tuple[str, str, Split], int] = defaultdict(int)
    total_strata: dict[tuple[str, str], int] = defaultdict(int)
    for document in forced_new:
        split = cast(Split, document["split"])
        for dimension_value in _strata(document):
            fixed_strata[(*dimension_value, split)] += 1
            total_strata[dimension_value] += 1
    group_strata: list[dict[tuple[str, str], int]] = []
    for group in groups:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for document in group.documents:
            for dimension_value in _strata(document):
                counts[dimension_value] += 1
                total_strata[dimension_value] += 1
        group_strata.append(counts)

    strata_rows: list[tuple[tuple[str, str], Split]] = [
        (key, split) for key in sorted(total_strata) for split in SPLITS
    ]
    page_rows = list(SPLITS)
    slack_start = x_count
    page_slack_start = slack_start + len(strata_rows) * 2
    variable_count = page_slack_start + len(page_rows) * 2
    objective = [0.0] * variable_count
    integrality = [0] * variable_count
    upper = [math.inf] * variable_count
    for group_index, group in enumerate(groups):
        digest_rank = int(hashlib.sha256(f"{salt}|{group.family_id}".encode()).hexdigest()[:12], 16)
        for split_index, _split in enumerate(SPLITS):
            index = group_index * len(SPLITS) + split_index
            objective[index] = digest_rank * 1e-18 + split_index * 1e-12
            integrality[index] = 1
            upper[index] = 1.0
    dimension_cardinality: dict[str, int] = defaultdict(int)
    for dimension, _value in total_strata:
        dimension_cardinality[dimension] += 1
    for row_index, (key, _split) in enumerate(strata_rows):
        weight = 1.0 / (total_strata[key] * dimension_cardinality[key[0]])
        objective[slack_start + row_index * 2] = weight
        objective[slack_start + row_index * 2 + 1] = weight
    new_pages = sum(int(document["page_count"]) for document in forced_new) + sum(
        group.pages for group in groups
    )
    for row_index in range(len(page_rows)):
        objective[page_slack_start + row_index * 2] = 1e-4 / max(new_pages, 1)
        objective[page_slack_start + row_index * 2 + 1] = 1e-4 / max(new_pages, 1)

    row_count = group_count + len(SPLITS) + len(strata_rows) + len(page_rows)
    matrix = lil_matrix((row_count, variable_count), dtype=float)
    lower: list[float] = []
    row = 0
    for group_index in range(group_count):
        for split_index in range(len(SPLITS)):
            matrix[row, group_index * len(SPLITS) + split_index] = 1
        lower.append(1.0)
        row += 1
    fixed_counts: dict[Split, int] = {
        split: sum(document["split"] == split for document in fixed) for split in SPLITS
    }
    for split_index, split in enumerate(SPLITS):
        for group_index, group in enumerate(groups):
            matrix[row, group_index * len(SPLITS) + split_index] = group.count
        lower.append(float(targets[split] - fixed_counts[split]))
        row += 1
    prior_counts: dict[Split, int] = {
        split: sum(document["split"] == split for document in prior_fixed) for split in SPLITS
    }
    new_targets: dict[Split, int] = {split: targets[split] - prior_counts[split] for split in SPLITS}
    if any(target < 0 for target in new_targets.values()):
        raise ValueError(f"fixed prior membership already exceeds a target: {new_targets}")
    new_document_count = sum(new_targets.values())
    for row_index, (key, split) in enumerate(strata_rows):
        split_index = SPLITS.index(split)
        for group_index, counts in enumerate(group_strata):
            if key in counts:
                matrix[row, group_index * len(SPLITS) + split_index] = counts[key]
        matrix[row, slack_start + row_index * 2] = -1
        matrix[row, slack_start + row_index * 2 + 1] = 1
        desired = total_strata[key] * new_targets[split] / new_document_count
        lower.append(desired - fixed_strata[(*key, split)])
        row += 1
    fixed_pages = {
        split: sum(int(document["page_count"]) for document in forced_new if document["split"] == split)
        for split in SPLITS
    }
    for split_index, split in enumerate(SPLITS):
        for group_index, group in enumerate(groups):
            matrix[row, group_index * len(SPLITS) + split_index] = group.pages
        matrix[row, page_slack_start + split_index * 2] = -1
        matrix[row, page_slack_start + split_index * 2 + 1] = 1
        desired_pages = new_pages * new_targets[split] / new_document_count
        lower.append(desired_pages - fixed_pages[split])
        row += 1
    constraint_bounds = np.asarray(lower, dtype=np.float64)
    constraints = LinearConstraint(
        matrix.tocsr(),
        constraint_bounds,  # pyright: ignore[reportArgumentType]
        constraint_bounds,  # pyright: ignore[reportArgumentType]
    )
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(
            np.zeros(variable_count),  # pyright: ignore[reportArgumentType]
            np.asarray(upper, dtype=np.float64),  # pyright: ignore[reportArgumentType]
        ),
        constraints=constraints,
        options={"presolve": True, "time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"partition optimization failed: {result.message}")
    assignments: dict[str, Split] = {}
    for group_index, group in enumerate(groups):
        values = result.x[group_index * len(SPLITS) : (group_index + 1) * len(SPLITS)]
        split = SPLITS[max(range(len(SPLITS)), key=lambda index: values[index])]
        assignments[group.family_id] = split
    return assignments


def _partition_policy(revision: str) -> dict[str, Any]:
    return {
        "ratios": RATIOS,
        "ratio_basis": "document count after SHA-256 consolidation and LayoutLM-family exclusion",
        "rounding": (
            "TRAIN at 3/5 and validation at 1/10 are rounded independently half-up; "
            "holdout receives the exact residual"
        ),
        "partition_salt": SALTS[revision],
        "assignment_priority": (
            "exact half-up/residual document targets; strata balance; page balance; salted tie-break"
        ),
        "family_rule": (
            "transitive atomic components over SHA-256, canonical URL, normalized title, "
            "publisher/report series, producer/edition, and template family; new members of an "
            "existing component inherit its fixed split"
        ),
        "holdout_evidence_rule": (
            "manifest metadata only; no candidate output, bronze, silver, evaluation, element "
            "failures, validation PDFs, or holdout PDFs inspected"
        ),
    }


def _summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pl.DataFrame({
        "split": [str(document["split"]) for document in documents],
        "page_count": [int(document["page_count"]) for document in documents],
        "family_id": [str(document["family_id"]) for document in documents],
    })
    rows = frame.group_by("split").agg(
        pl.len().alias("document_count"),
        pl.col("page_count").sum().alias("page_count"),
        pl.col("family_id").n_unique().alias("family_count"),
    )
    by_split = {str(row["split"]): row for row in rows.to_dicts()}
    total_documents = len(documents)
    total_pages = sum(int(document["page_count"]) for document in documents)
    output: dict[str, Any] = {}
    for split in SPLITS:
        row = by_split.get(split, {"document_count": 0, "page_count": 0, "family_count": 0})
        document_ratio = int(row["document_count"]) / total_documents
        page_ratio = int(row["page_count"]) / total_pages
        output[split] = {
            **row,
            "document_ratio": round(document_ratio, 8),
            "document_ratio_delta": round(document_ratio - RATIOS[split], 8),
            "page_ratio": round(page_ratio, 8),
            "page_ratio_delta": round(page_ratio - RATIOS[split], 8),
        }
    return output


def _strata_summary(documents: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for dimension in (
        "region",
        "country",
        "document_stratum",
        "category",
        "difficult_stratum",
        "language_mix",
        "source_producer",
        "source_manifest",
        "page_count_band",
        "rotation_signal",
        "table_signal",
        "form_signal",
        "layout_signal",
    ):
        frame = pl.DataFrame({
            dimension: [str(document.get(dimension, "unknown")) for document in documents],
            "split": [str(document["split"]) for document in documents],
            "page_count": [int(document["page_count"]) for document in documents],
        })
        output[dimension] = (
            frame
            .group_by(dimension, "split")
            .agg(pl.len().alias("document_count"), pl.col("page_count").sum().alias("page_count"))
            .sort(dimension, "split")
            .to_dicts()
        )
    return output


def _verify(
    documents: list[dict[str, Any]],
    parent_manifest: dict[str, Any],
    previous_partition: dict[str, Any],
    targets: dict[Split, int],
    *,
    parent_source_manifest: str,
    strict_document_schema: bool = True,
) -> list[str]:
    parent_documents = cast(list[dict[str, Any]], parent_manifest["documents"])
    previous_documents = cast(list[dict[str, Any]], previous_partition["documents"])
    expected_parent = _expected_parent_splits(parent_documents, previous_documents)
    fixed_hashes = set(expected_parent)
    for index, document in enumerate(documents):
        if strict_document_schema and str(document.get("sha256", "")) not in fixed_hashes:
            _validate_new_document(document, Path(f"output.documents[{index}]"))
        if document.get("split") not in SPLITS:
            raise ValueError(f"output document has invalid split: {document.get('id', index)}")
        for field in ("id", "family_id", "assignment_origin", "assignment_rationale"):
            if not str(document.get(field, "")).strip():
                raise ValueError(f"output document has blank {field}: {document.get('id', index)}")
    ids = [str(document["id"]).strip() for document in documents]
    hashes = [str(document["sha256"]) for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document IDs after consolidation")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate SHA-256 values after consolidation")
    canonical_urls = [_derived_normalized_url(document) for document in documents]
    canonical_urls = [url for url in canonical_urls if url]
    if len(canonical_urls) != len(set(canonical_urls)):
        raise ValueError("duplicate canonical URLs after consolidation")
    title_publishers = [_derived_title_publisher_key(document) for document in documents]
    if len(title_publishers) != len(set(title_publishers)):
        raise ValueError("duplicate normalized title and publisher after consolidation")
    if any(_is_layoutlm(document) for document in documents):
        raise ValueError("LayoutLM/LayoutLMv2 remained in active corpus")
    parent_by_sha = {str(document["sha256"]): document for document in parent_documents}
    previous_by_sha = {str(document["sha256"]): document for document in previous_documents}
    by_sha = {str(document["sha256"]): document for document in documents}
    for sha256, (expected_id, expected_split) in expected_parent.items():
        current = by_sha.get(sha256)
        if current is None:
            raise ValueError(f"parent-manifest member missing: {expected_id}")
        if current["id"] != expected_id:
            raise ValueError(f"parent-manifest member relabeled: {expected_id}")
        parent_document = parent_by_sha[sha256]
        previous = previous_by_sha.get(sha256)
        if current["split"] != expected_split:
            raise ValueError(f"parent-manifest membership changed for {expected_id}")
        expected_output = _fixed_parent_output(
            parent_document,
            source_manifest=parent_source_manifest,
            split=expected_split,
            previous_split=previous.get("split") if previous is not None else None,
        )
        expected_output["family_id"] = current["family_id"]
        if current != expected_output:
            raise ValueError(f"authenticated parent metadata changed for {expected_id}")

    for component_id, indices in _atomic_components(documents):
        component = [documents[index] for index in indices]
        if any(document["family_id"] != component_id for document in component):
            raise ValueError(f"committed family_id does not match atomic component: {component_id}")
        fixed_members = [document for document in component if document["sha256"] in fixed_hashes]
        new_members = [document for document in component if document["sha256"] not in fixed_hashes]
        fixed_splits = {str(document["split"]) for document in fixed_members}
        new_splits = {str(document["split"]) for document in new_members}
        if len(fixed_splits) > 1 and new_members:
            raise ValueError(f"new documents join a grandfathered conflicting family: {component_id}")
        if len(fixed_splits) == 1 and new_splits and new_splits != fixed_splits:
            raise ValueError(f"new member crossed an existing family boundary: {component_id}")
        if not fixed_members and len(new_splits) > 1:
            raise ValueError(f"new family leakage detected: {component_id}")
    counts = {split: sum(document["split"] == split for document in documents) for split in SPLITS}
    if counts != targets:
        raise ValueError(f"document counts do not match selected targets: {counts} != {targets}")
    return [
        "checkpoint metadata freshness and source/outcome completeness validated",
        "unique document IDs/SHA-256 values and no new canonical-URL/title duplicates",
        "LayoutLM and LayoutLMv2 absent from active corpus",
        "every partition-v2 membership and previously burned TRAIN assignment preserved",
        "new hash/URL/title/report-series/edition/template components assigned atomically",
        "document counts equal exact half-up/residual targets",
        "new candidates are checkpoint-attested accuracy-uninspected and unassigned",
    ]


def _validate_output_contract(
    *,
    manifest: dict[str, Any],
    partition: dict[str, Any],
    parent_manifest: dict[str, Any],
    parent_partition: dict[str, Any],
    revision: str,
) -> None:
    if manifest.get("policy") != _partition_policy(revision):
        raise ValueError("output policy does not match the revision policy")
    documents_value = manifest.get("documents")
    if not isinstance(documents_value, list) or any(
        not isinstance(document, dict) for document in documents_value
    ):
        raise ValueError("output documents must be an array of objects")
    documents = cast(list[dict[str, Any]], documents_value)
    ideal_targets = document_targets(len(documents))
    if manifest.get("ideal_document_targets") != ideal_targets:
        raise ValueError("ideal document targets do not match the output corpus")
    if manifest.get("selected_document_targets") != ideal_targets:
        raise ValueError("selected document targets do not match exact half-up/residual targets")
    if manifest.get("ideal_targets_family_reachable") is not True:
        raise ValueError("final output does not attest reachable exact family targets")
    parent_documents, previous_documents = _validate_parent_relationship(parent_manifest, parent_partition)
    checks = _verify(
        documents,
        parent_manifest,
        parent_partition,
        ideal_targets,
        parent_source_manifest=str(manifest["parent_manifest"]["path"]),
    )
    if manifest.get("integrity_checks") != checks:
        raise ValueError("integrity checks do not match recomputed output checks")
    if manifest.get("summary") != _summary(documents):
        raise ValueError("output summary does not match committed documents")
    expected_parent = _expected_parent_splits(parent_documents, previous_documents)
    fixed_hashes = set(expected_parent)
    forced_count, conflicts = _family_declarations(documents, fixed_hashes)
    if manifest.get("new_documents_forced_by_existing_family") != forced_count:
        raise ValueError("forced-by-existing-family count does not match atomic components")
    if manifest.get("grandfathered_fixed_family_conflicts") != conflicts:
        raise ValueError("grandfathered fixed-family conflicts do not match parent state")
    previously_active = len(expected_parent)
    expected_selection = {
        "previously_active_documents": previously_active,
        "new_accuracy_uninspected_documents": len(documents) - previously_active,
        "retained_unique_documents": len(documents),
    }
    if manifest.get("selection") != expected_selection:
        raise ValueError("output selection counts do not match committed documents")
    if partition.get("strata_summary") != _strata_summary(documents):
        raise ValueError("partition strata summary does not match committed documents")


def _descriptor_record_count(snapshot: _Snapshot) -> int:
    if snapshot.path.suffix == ".jsonl":
        return len(_records_from_snapshot(snapshot))
    payload = _json_from_snapshot(snapshot)
    for key in ("documents", "sources"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return 1


def _commitment_descriptor(
    *, role: str, path: Path, root_dir: Path, snapshots: _SnapshotStore
) -> dict[str, Any]:
    snapshot = snapshots.read(path)
    return {
        "role": role,
        "path": str(path.resolve().relative_to(root_dir.resolve())),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "record_count": _descriptor_record_count(snapshot),
    }


def _validate_sidecar_descriptor(
    *,
    label: str,
    payload: object,
    expected_path: Path,
    root_dir: Path,
    snapshots: _SnapshotStore,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"revision sidecar {label} descriptor must be an object")
    required = {"path", "sha256", "size_bytes", "record_count"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"revision sidecar {label} is missing {', '.join(missing)}")
    described_path = (root_dir / str(payload["path"])).resolve()
    try:
        described_path.relative_to(root_dir.resolve())
    except ValueError as error:
        raise ValueError(f"revision sidecar {label} escapes repository root") from error
    if described_path != expected_path.resolve():
        raise ValueError(f"revision sidecar {label} does not describe the expected checkpoint file")
    snapshot = snapshots.read(described_path)
    if payload["sha256"] != snapshot.sha256 or payload["size_bytes"] != snapshot.size_bytes:
        raise ValueError(f"revision sidecar {label} hash/size mismatch: {described_path}")
    if payload["record_count"] != _descriptor_record_count(snapshot):
        raise ValueError(f"revision sidecar {label} record count mismatch: {described_path}")
    return cast(dict[str, Any], payload)


def _validated_revision_sidecars(
    *,
    root_dir: Path,
    new_manifest_paths: list[Path],
    revision_sidecar_paths: list[Path],
    expected_sidecar_sha256: list[str],
    snapshots: _SnapshotStore,
) -> list[dict[str, Any]]:
    if len(revision_sidecar_paths) != len(new_manifest_paths):
        raise ValueError("every new manifest requires exactly one revision sidecar")
    if len(expected_sidecar_sha256) != len(revision_sidecar_paths):
        raise ValueError("every revision sidecar requires an externally supplied SHA-256")
    manifests_by_path = {path.resolve(): path for path in new_manifest_paths}
    if len(manifests_by_path) != len(new_manifest_paths):
        raise ValueError("new manifest paths must be unique")
    if len({path.resolve() for path in revision_sidecar_paths}) != len(revision_sidecar_paths):
        raise ValueError("revision sidecar paths must be unique")
    validated: list[dict[str, Any]] = []
    seen_revision_ids: set[str] = set()
    for sidecar_path, expected_sha256 in zip(revision_sidecar_paths, expected_sidecar_sha256, strict=True):
        sidecar_path = sidecar_path.resolve()
        try:
            sidecar_path.relative_to(root_dir.resolve())
        except ValueError as error:
            raise ValueError(f"revision sidecar escapes repository root: {sidecar_path}") from error
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"invalid expected revision sidecar SHA-256: {expected_sha256}")
        sidecar_snapshot = snapshots.read(sidecar_path)
        if sidecar_snapshot.sha256 != expected_sha256:
            raise ValueError(f"revision sidecar SHA-256 mismatch: {sidecar_path}")
        sidecar = _json_from_snapshot(sidecar_snapshot)
        revision_id = str(sidecar.get("revision_id", "")).strip()
        if sidecar.get("immutable") is not True or not revision_id or sidecar.get("schema_version") != "1.0":
            raise ValueError(
                f"revision sidecar is not immutable with supported schema_version 1.0: {sidecar_path}"
            )
        if revision_id in seen_revision_ids:
            raise ValueError(f"duplicate revision_id across sidecars: {revision_id}")
        seen_revision_ids.add(revision_id)
        descriptor_keys = [key for key in ("revision", "output") if key in sidecar]
        if len(descriptor_keys) != 1:
            raise ValueError("revision sidecar requires exactly one of revision or output")
        descriptor = cast(dict[str, Any], sidecar[descriptor_keys[0]])
        revision_path = (root_dir / str(descriptor.get("path", ""))).resolve()
        manifest_path = manifests_by_path.get(revision_path)
        if manifest_path is None:
            raise ValueError(f"revision sidecar does not match a supplied manifest: {sidecar_path}")
        kind = _checkpoint_kind(manifest_path)
        if kind is None:
            raise ValueError("authenticated revision sidecars are only valid for completed checkpoints")
        source_path = manifest_path.with_name(
            "source_manifest.jsonl" if kind in ("difficult", "gap") else "source_manifest.json"
        )
        rejected_path = manifest_path.with_name("rejected_manifest.jsonl")
        summary_path = manifest_path.with_name("summary.json")
        companions = sidecar.get("companions")
        if not isinstance(companions, dict) or set(companions) != {"rejected", "summary"}:
            raise ValueError("revision sidecar must authenticate exactly rejected and summary companions")
        _validate_sidecar_descriptor(
            label="revision/output",
            payload=descriptor,
            expected_path=manifest_path,
            root_dir=root_dir,
            snapshots=snapshots,
        )
        parent = _validate_sidecar_descriptor(
            label="parent",
            payload=sidecar.get("parent"),
            expected_path=source_path,
            root_dir=root_dir,
            snapshots=snapshots,
        )
        _validate_sidecar_descriptor(
            label="rejected companion",
            payload=companions["rejected"],
            expected_path=rejected_path,
            root_dir=root_dir,
            snapshots=snapshots,
        )
        _validate_sidecar_descriptor(
            label="summary companion",
            payload=companions["summary"],
            expected_path=summary_path,
            root_dir=root_dir,
            snapshots=snapshots,
        )
        validated.append({
            "path": str(sidecar_path.relative_to(root_dir)),
            "sha256": sidecar_snapshot.sha256,
            "revision_id": revision_id,
            "revision_path": str(manifest_path.relative_to(root_dir)),
            "parent_path": str(source_path.relative_to(root_dir)),
            "parent_sha256": str(parent["sha256"]),
            "companions_authenticated": ["rejected", "summary"],
        })
    if {Path(item["revision_path"]) for item in validated} != {
        path.relative_to(root_dir) for path in new_manifest_paths
    }:
        raise ValueError("revision sidecars do not cover every supplied manifest")
    return validated


def _validate_final_inputs(new_manifest_paths: list[Path], revision: str) -> None:
    kinds = [_checkpoint_kind(path) for path in new_manifest_paths]
    required = ["difficult", "financial"] if revision == "v3" else ["difficult", "financial", "gap"]
    if sorted(kind for kind in kinds if kind is not None) != sorted(required) or len(kinds) != len(required):
        raise ValueError(f"final status requires exactly these {revision} checkpoints: {required}")
    if revision == "v5" and not any(
        path.parent.name in ("v3-financial-r2", "v3-financial-r3") for path in new_manifest_paths
    ):
        raise ValueError("final v5 status requires a corrected v3-financial-r2/r3 checkpoint")


def _validate_parent_relationship(
    corpus_manifest: dict[str, Any], previous_partition: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def documents(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
        value = payload.get("documents")
        if not isinstance(value, list):
            raise ValueError(f"parent {label} has no documents array")
        if any(not isinstance(row, dict) for row in value):
            raise ValueError(f"parent {label} documents must all be objects")
        return cast(list[dict[str, Any]], value)

    manifest_documents = documents(corpus_manifest, "manifest")
    partition_documents = documents(previous_partition, "partition")
    for label, rows in (("manifest", manifest_documents), ("partition", partition_documents)):
        ids = [str(row.get("id", "")).strip() for row in rows]
        hashes = [str(row.get("sha256", "")).strip() for row in rows]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"parent {label} contains blank or duplicate document IDs")
        if any(not value for value in hashes) or len(set(hashes)) != len(hashes):
            raise ValueError(f"parent {label} contains blank or duplicate SHA-256 values")
        canonical_urls = [_derived_normalized_url(row) for row in rows]
        canonical_urls = [url for url in canonical_urls if url]
        if len(canonical_urls) != len(set(canonical_urls)):
            raise ValueError(f"parent {label} contains duplicate canonical URLs")
    title_publishers = [_derived_title_publisher_key(row) for row in manifest_documents]
    active_keys = [
        key for row, key in zip(manifest_documents, title_publishers, strict=True) if not _is_layoutlm(row)
    ]
    if len(active_keys) != len(set(active_keys)):
        raise ValueError("parent manifest contains duplicate normalized title and publisher")
    manifest_by_sha = {str(row["sha256"]): row for row in manifest_documents}
    for row in partition_documents:
        parent = manifest_by_sha.get(str(row["sha256"]))
        if parent is None:
            raise ValueError(f"parent partition member is absent from parent manifest: {row['id']}")
        if parent["id"] != row["id"]:
            raise ValueError(f"parent manifest/partition ID mismatch for SHA-256 {row['sha256']}")
        mismatched = [key for key, value in parent.items() if row.get(key) != value]
        if mismatched:
            raise ValueError(
                f"parent manifest/partition metadata mismatch for {row['id']}: {sorted(mismatched)}"
            )
    return manifest_documents, partition_documents


def _expected_parent_splits(
    parent_manifest_documents: list[dict[str, Any]],
    previous_partition_documents: list[dict[str, Any]],
) -> dict[str, tuple[str, Split]]:
    previous_by_sha = {str(document["sha256"]): document for document in previous_partition_documents}
    expected: dict[str, tuple[str, Split]] = {}
    for document in parent_manifest_documents:
        if _is_layoutlm(document):
            continue
        sha256 = str(document["sha256"])
        previous = previous_by_sha.get(sha256)
        previous_split = previous.get("split") if previous is not None else None
        if previous_split in (None, "dev"):
            split: Split = "train"
        elif previous_split in SPLITS:
            split = previous_split
        else:
            raise ValueError(f"parent partition has unsupported split for {document['id']}")
        expected[sha256] = (str(document["id"]), split)
    return expected


def _fixed_parent_output(
    document: dict[str, Any],
    *,
    source_manifest: str,
    split: Split,
    previous_split: object,
) -> dict[str, Any]:
    output = _enrich(document, source_manifest)
    if previous_split is None:
        rationale = "Previously active and therefore exposed/burned; permanently TRAIN."
    elif previous_split == "dev":
        rationale = "Previous DEV exposure is permanent; mapped to TRAIN."
    else:
        rationale = f"Preserved immutable previous-partition {split} membership."
    output.update({
        "split": split,
        "assignment_origin": "fixed-previous",
        "assignment_rationale": rationale,
    })
    output.setdefault("source_family_id", str(output.get("family_id", "")))
    return output


def _canonical_root_relative_path(path: Path, root_dir: Path, *, label: str) -> str:
    resolved_root = root_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository root: {path}") from error
    if relative == Path("."):
        raise ValueError(f"{label} path must name a file below the repository root")
    return relative.as_posix()


def _serialized_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _input_commitment_sha256(descriptors: list[dict[str, Any]]) -> str:
    canonical = json.dumps(descriptors, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_output(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staged")
    descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
        return staged
    except BaseException as staging_error:
        cleanup_failures = _cleanup_publication_paths([staged])
        if cleanup_failures:
            raise staging_error from _cleanup_failure_group(cleanup_failures)
        raise


def _cleanup_publication_paths(paths: list[Path]) -> list[Exception]:
    """Attempt every cleanup and retry paths left behind by transient failures."""
    failures: list[Exception] = []
    retry: list[Path] = []
    for path in paths:
        try:
            existed = path.exists()
        except Exception as error:
            failures.append(error)
            existed = True
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            failures.append(error)
            retry.append(path)
        if existed:
            try:
                _fsync_directory(path.parent)
            except Exception as error:
                failures.append(error)
    for path in retry:
        try:
            path.unlink(missing_ok=True)
        except Exception as error:
            failures.append(error)
        try:
            _fsync_directory(path.parent)
        except Exception as error:
            failures.append(error)
    return failures


def _cleanup_failure_group(failures: list[Exception]) -> ExceptionGroup:
    details = "; ".join(str(error) for error in failures)
    return ExceptionGroup(f"publication cleanup failures: {details}", failures)


def _publish_bundle(
    outputs: list[tuple[Path, bytes]],
    revalidate_inputs: Callable[[], None],
) -> None:
    """Stage all bytes, then atomically link payloads and the immutable descriptor last."""
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for path, payload in outputs:
            staged.append((path, _stage_output(path, payload)))
        revalidate_inputs()
        for path, staged_path in staged:
            os.link(staged_path, path)
            published.append(path)
            staged_path.unlink()
            _fsync_directory(path.parent)
        revalidate_inputs()
    except BaseException as publication_error:
        cleanup_failures = _cleanup_publication_paths([
            *reversed(published),
            *(staged_path for _path, staged_path in staged),
        ])
        if cleanup_failures:
            raise publication_error from _cleanup_failure_group(cleanup_failures)
        raise
    cleanup_failures = _cleanup_publication_paths([staged_path for _path, staged_path in staged])
    if cleanup_failures:
        raise _cleanup_failure_group(cleanup_failures)


def _verify_committed_file(
    *,
    payload: object,
    label: str,
    root_dir: Path,
    snapshots: _SnapshotStore,
    expected_role: str | None = None,
) -> _Snapshot:
    if not isinstance(payload, dict):
        raise ValueError(f"descriptor {label} must be an object")
    required = {"path", "sha256", "size_bytes", "record_count"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"descriptor {label} is missing {', '.join(missing)}")
    if expected_role is not None and payload.get("role") != expected_role:
        raise ValueError(f"descriptor {label} has mismatched role")
    path = (root_dir / str(payload["path"])).resolve()
    try:
        path.relative_to(root_dir.resolve())
    except ValueError as error:
        raise ValueError(f"descriptor {label} path escapes repository root") from error
    snapshot = snapshots.read(path)
    if payload["sha256"] != snapshot.sha256:
        raise ValueError(f"descriptor {label} SHA-256 mismatch: {path}")
    if payload["size_bytes"] != snapshot.size_bytes:
        raise ValueError(f"descriptor {label} size mismatch: {path}")
    if payload["record_count"] != _descriptor_record_count(snapshot):
        raise ValueError(f"descriptor {label} record count mismatch: {path}")
    return snapshot


def verify_corpus_partition_descriptor(
    descriptor_path: Path,
    *,
    root_dir: Path,
    expected_descriptor_sha256: str,
) -> dict[str, Any]:
    """Verify a schema-2 immutable corpus output descriptor and every commitment."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_descriptor_sha256):
        raise ValueError("invalid expected output descriptor SHA-256")
    expected_descriptor_path = _canonical_root_relative_path(
        descriptor_path,
        root_dir,
        label="output descriptor",
    )
    snapshots = _SnapshotStore()
    primary_error: BaseException | None = None
    try:
        descriptor_snapshot = snapshots.read(descriptor_path)
        if descriptor_snapshot.sha256 != expected_descriptor_sha256:
            raise ValueError("output descriptor SHA-256 mismatch")
        descriptor = _json_from_snapshot(descriptor_snapshot)
        if descriptor.get("schema_version") != "2.0" or descriptor.get("immutable") is not True:
            raise ValueError("unsupported or mutable corpus output descriptor")
        status = descriptor.get("status")
        if status not in ("provisional", "final"):
            raise ValueError("output descriptor status must be exactly provisional or final")
        revision_match = re.fullmatch(r"corpus-partition-(v[345])", str(descriptor.get("revision_id", "")))
        if revision_match is None:
            raise ValueError("output descriptor revision_id is missing or unsupported")
        parents = descriptor.get("parents")
        outputs = descriptor.get("outputs")
        authenticated_inputs = descriptor.get("authenticated_inputs")
        if not isinstance(parents, dict) or set(parents) != {"manifest", "partition"}:
            raise ValueError("output descriptor requires manifest and partition parent commitments")
        if not isinstance(outputs, dict) or set(outputs) != {"manifest", "partition"}:
            raise ValueError("output descriptor requires manifest and partition output commitments")
        if not isinstance(authenticated_inputs, list) or not authenticated_inputs:
            raise ValueError("output descriptor requires ordered authenticated input commitments")
        if any(not isinstance(item, dict) for item in authenticated_inputs):
            raise ValueError("authenticated input descriptors must be objects")
        input_descriptors = cast(list[dict[str, Any]], authenticated_inputs)
        if descriptor.get("input_commitment_sha256") != _input_commitment_sha256(input_descriptors):
            raise ValueError("output descriptor input commitment SHA-256 mismatch")

        manifest_snapshot = _verify_committed_file(
            payload=outputs["manifest"],
            label="outputs.manifest",
            root_dir=root_dir,
            snapshots=snapshots,
        )
        partition_snapshot = _verify_committed_file(
            payload=outputs["partition"],
            label="outputs.partition",
            root_dir=root_dir,
            snapshots=snapshots,
        )
        manifest = _json_from_snapshot(manifest_snapshot)
        partition = _json_from_snapshot(partition_snapshot)
        if partition.get("output_descriptor") != expected_descriptor_path:
            raise ValueError(
                "partition output_descriptor does not match the canonical verified descriptor path"
            )
        if (
            "output_descriptor" in manifest
            and manifest["output_descriptor"] != partition["output_descriptor"]
        ):
            raise ValueError("manifest/partition embedded output_descriptor paths differ")
        expected_schema = {"v3": "3.0", "v4": "4.0", "v5": "5.0"}[revision_match.group(1)]
        if manifest.get("schema_version") != expected_schema:
            raise ValueError("manifest schema_version does not match descriptor revision_id")
        shared_fields = (
            "schema_version",
            "status",
            "generated_date",
            "policy",
            "parent_manifest",
            "parent_partition",
            "inputs",
            "authenticated_revision_sidecars",
            "excluded_documents",
            "ideal_document_targets",
            "selected_document_targets",
            "ideal_targets_family_reachable",
            "new_documents_forced_by_existing_family",
            "grandfathered_fixed_family_conflicts",
            "summary",
            "integrity_checks",
            "selection",
            "documents",
        )
        missing_shared = [field for field in shared_fields if field not in manifest or field not in partition]
        if missing_shared:
            raise ValueError(f"manifest/partition missing required shared fields: {missing_shared}")
        mismatched_shared = [field for field in shared_fields if manifest[field] != partition[field]]
        if mismatched_shared:
            raise ValueError(f"manifest/partition shared fields differ: {mismatched_shared}")
        if manifest.get("status") != status:
            raise ValueError("descriptor status does not match manifest and partition")
        documents = manifest["documents"]
        if not isinstance(documents, list):
            raise ValueError("manifest/partition documents must be an array")
        if status == "final" and len(documents) < 816:
            raise ValueError(f"post-dedup final corpus is below the 816-document floor: {len(documents)}")
        if status == "final":
            manifest_inputs_for_contract = manifest["inputs"]
            if not isinstance(manifest_inputs_for_contract, list) or any(
                not isinstance(item, dict) for item in manifest_inputs_for_contract
            ):
                raise ValueError("manifest inputs must be an ordered descriptor list")
            _validate_final_inputs(
                [root_dir / str(item.get("path", "")) for item in manifest_inputs_for_contract],
                revision_match.group(1),
            )

        parent_snapshots: dict[str, _Snapshot] = {}
        for label, expected_role, embedded_key in (
            ("manifest", "parent_manifest", "parent_manifest"),
            ("partition", "parent_partition", "parent_partition"),
        ):
            parent_payload = parents[label]
            if parent_payload != manifest.get(embedded_key):
                raise ValueError(f"descriptor parent {label} does not equal embedded parent commitment")
            parent_snapshots[label] = _verify_committed_file(
                payload=parent_payload,
                label=f"parents.{label}",
                root_dir=root_dir,
                snapshots=snapshots,
                expected_role=expected_role,
            )

        manifest_commitment = partition.get("manifest")
        expected_manifest_commitment = {
            "path": outputs["manifest"].get("path"),
            "sha256": manifest_snapshot.sha256,
            "size_bytes": manifest_snapshot.size_bytes,
            "record_count": _descriptor_record_count(manifest_snapshot),
        }
        if manifest_commitment != expected_manifest_commitment:
            raise ValueError("partition embedded manifest commitment mismatch")

        manifest_inputs = manifest.get("inputs")
        sidecars = manifest.get("authenticated_revision_sidecars")
        if not isinstance(manifest_inputs, list) or any(
            not isinstance(item, dict) for item in manifest_inputs
        ):
            raise ValueError("manifest inputs must be an ordered descriptor list")
        if not isinstance(sidecars, list) or any(not isinstance(item, dict) for item in sidecars):
            raise ValueError("manifest authenticated sidecars must be an ordered descriptor list")
        input_rows = cast(list[dict[str, Any]], manifest_inputs)
        sidecar_rows = cast(list[dict[str, Any]], sidecars)
        expected_inputs: list[dict[str, Any]] = []
        for input_row in input_rows:
            accepted_path = str(input_row.get("path", ""))
            expected_inputs.append({
                "role": "accepted",
                "path": accepted_path,
                "sha256": str(input_row.get("sha256", "")),
                "size_bytes": input_row.get("size_bytes"),
            })
            accepted_absolute = (root_dir / accepted_path).resolve()
            checkpoint = input_row.get("checkpoint_validation")
            kind = _checkpoint_kind(accepted_absolute)
            source_path: Path | None = None
            if kind is not None:
                if not isinstance(checkpoint, dict):
                    raise ValueError("checkpoint input lacks embedded companion commitments")
                accepted_records = _records_from_snapshot(snapshots.read(accepted_absolute))
                validated_checkpoint = _validate_checkpoint(accepted_absolute, accepted_records, snapshots)
                if validated_checkpoint is None:
                    raise ValueError("final input is not an exact supported checkpoint basename")
                committed_checkpoint = {
                    key: (str(Path(str(value)).relative_to(root_dir)) if key.endswith("_path") else value)
                    for key, value in validated_checkpoint.items()
                }
                if checkpoint != committed_checkpoint:
                    raise ValueError("embedded checkpoint validation does not match exact committed contents")
                source_path = accepted_absolute.with_name(
                    "source_manifest.jsonl" if kind in ("difficult", "gap") else "source_manifest.json"
                )
                companions = (
                    ("source", source_path),
                    ("rejected", accepted_absolute.with_name("rejected_manifest.jsonl")),
                    ("summary", accepted_absolute.with_name("summary.json")),
                )
                for role, companion_path in companions:
                    relative_path = str(companion_path.relative_to(root_dir))
                    if checkpoint.get(f"{role}_path") != relative_path:
                        raise ValueError(f"checkpoint {role} path commitment mismatch")
                    expected_inputs.append({
                        "role": role,
                        "path": relative_path,
                        "sha256": str(checkpoint.get(f"{role}_sha256", "")),
                    })
            matching_sidecars = [row for row in sidecar_rows if row.get("revision_path") == accepted_path]
            if len(matching_sidecars) > 1:
                raise ValueError("multiple sidecars commit the same accepted input")
            if matching_sidecars:
                if kind is None or source_path is None:
                    raise ValueError("authenticated sidecar is attached to a non-checkpoint input")
                sidecar_row = matching_sidecars[0]
                sidecar_relative = str(sidecar_row.get("path", ""))
                sidecar_path = (root_dir / sidecar_relative).resolve()
                try:
                    sidecar_path.relative_to(root_dir.resolve())
                except ValueError as error:
                    raise ValueError("authenticated sidecar path escapes repository root") from error
                sidecar_snapshot = snapshots.read(sidecar_path)
                if sidecar_row.get("sha256") != sidecar_snapshot.sha256:
                    raise ValueError("manifest sidecar SHA-256 commitment mismatch")
                sidecar = _json_from_snapshot(sidecar_snapshot)
                if (
                    sidecar.get("schema_version") != "1.0"
                    or sidecar.get("immutable") is not True
                    or sidecar.get("revision_id") != sidecar_row.get("revision_id")
                ):
                    raise ValueError("authenticated sidecar identity/version mismatch")
                revision_keys = [key for key in ("revision", "output") if key in sidecar]
                if len(revision_keys) != 1:
                    raise ValueError("authenticated sidecar has ambiguous revision/output")
                _validate_sidecar_descriptor(
                    label="revision/output",
                    payload=sidecar[revision_keys[0]],
                    expected_path=accepted_absolute,
                    root_dir=root_dir,
                    snapshots=snapshots,
                )
                _validate_sidecar_descriptor(
                    label="parent",
                    payload=sidecar.get("parent"),
                    expected_path=source_path,
                    root_dir=root_dir,
                    snapshots=snapshots,
                )
                sidecar_companions = sidecar.get("companions")
                if not isinstance(sidecar_companions, dict) or set(sidecar_companions) != {
                    "rejected",
                    "summary",
                }:
                    raise ValueError("authenticated sidecar companion set mismatch")
                _validate_sidecar_descriptor(
                    label="rejected companion",
                    payload=sidecar_companions["rejected"],
                    expected_path=accepted_absolute.with_name("rejected_manifest.jsonl"),
                    root_dir=root_dir,
                    snapshots=snapshots,
                )
                _validate_sidecar_descriptor(
                    label="summary companion",
                    payload=sidecar_companions["summary"],
                    expected_path=accepted_absolute.with_name("summary.json"),
                    root_dir=root_dir,
                    snapshots=snapshots,
                )
                expected_sidecar_row = {
                    "path": sidecar_relative,
                    "sha256": sidecar_snapshot.sha256,
                    "revision_id": sidecar["revision_id"],
                    "revision_path": accepted_path,
                    "parent_path": str(source_path.relative_to(root_dir)),
                    "parent_sha256": sidecar["parent"]["sha256"],
                    "companions_authenticated": ["rejected", "summary"],
                }
                if sidecar_row != expected_sidecar_row:
                    raise ValueError(
                        "manifest revision-sidecar commitment does not match authenticated sidecar"
                    )
                expected_inputs.append({
                    "role": "sidecar",
                    "path": sidecar_relative,
                    "sha256": str(sidecar_row.get("sha256", "")),
                })
            elif status == "final":
                raise ValueError("final checkpoint input lacks an authenticated sidecar")
        if len(sidecar_rows) != sum(item["role"] == "sidecar" for item in expected_inputs):
            raise ValueError("manifest contains unmatched authenticated sidecars")
        if len(input_descriptors) != len(expected_inputs):
            raise ValueError("authenticated input descriptor count does not match manifest inputs")
        for index, (actual, expected) in enumerate(zip(input_descriptors, expected_inputs, strict=True)):
            if any(actual.get(key) != value for key, value in expected.items()):
                raise ValueError(f"authenticated input identity mismatch at index {index}")
            _verify_committed_file(
                payload=actual,
                label=f"authenticated_inputs[{index}]",
                root_dir=root_dir,
                snapshots=snapshots,
                expected_role=expected["role"],
            )
        if status == "final" and [item["role"] for item in expected_inputs] != [
            role
            for _input_row in input_rows
            for role in ("accepted", "source", "rejected", "summary", "sidecar")
        ]:
            raise ValueError("final output descriptor has incomplete or unordered authenticated inputs")
        if status == "final":
            _validate_output_contract(
                manifest=manifest,
                partition=partition,
                parent_manifest=_json_from_snapshot(parent_snapshots["manifest"]),
                parent_partition=_json_from_snapshot(parent_snapshots["partition"]),
                revision=revision_match.group(1),
            )
        snapshots.revalidate()
        return descriptor
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_snapshot_store(snapshots, primary_error)


def _build_corpus_partition(
    *,
    root_dir: Path,
    corpus_manifest_path: Path,
    previous_partition_path: Path,
    new_manifest_paths: list[Path],
    output_manifest_path: Path,
    output_partition_path: Path,
    status: Literal["provisional", "final"],
    output_descriptor_path: Path | None,
    revision: Literal["v3", "v4", "v5"] | None,
    revision_sidecar_paths: list[Path] | None,
    expected_revision_sidecar_sha256: list[str] | None,
    snapshots: _SnapshotStore,
) -> BuildResult:
    """Build an immutable deterministic corpus revision from metadata only.

    This function never opens PDFs or any candidate, bronze, silver, evaluation, or
    element-failure artifact. It consumes corpus/staging manifests exclusively.
    """
    if status not in ("provisional", "final"):
        raise ValueError("status must be exactly provisional or final")
    output_paths = [output_manifest_path, output_partition_path]
    if output_descriptor_path is not None:
        output_paths.append(output_descriptor_path)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("manifest, partition, and descriptor outputs must be distinct paths")
    if any(path.exists() for path in output_paths):
        raise FileExistsError("versioned partition outputs must not already exist")
    if status == "final" and output_descriptor_path is None:
        raise ValueError("final status requires an immutable output descriptor path")
    canonical_output_descriptor = (
        _canonical_root_relative_path(
            output_descriptor_path,
            root_dir,
            label="output descriptor",
        )
        if output_descriptor_path is not None
        else None
    )
    inferred = {
        match.group(1)
        for path in (output_manifest_path, output_partition_path)
        if (match := re.search(r"(?:^|[-_.])(v\d+)(?:[-_.]|$)", path.name)) is not None
    }
    if revision is None:
        unsupported = inferred - SALTS.keys()
        if unsupported:
            raise ValueError(f"unsupported corpus revision in output filename: {sorted(unsupported)}")
        if len(inferred) > 1:
            raise ValueError("output filenames imply conflicting revisions; supply revision explicitly")
        revision = cast(Literal["v3", "v4", "v5"], next(iter(inferred), "v3"))
    salt = SALTS[revision]
    if status == "final":
        _validate_final_inputs(new_manifest_paths, revision)
    sidecar_paths = revision_sidecar_paths or []
    expected_sidecar_hashes = expected_revision_sidecar_sha256 or []
    if status == "final" and (not sidecar_paths or not expected_sidecar_hashes):
        raise ValueError(
            "final status requires one externally SHA-pinned immutable revision sidecar per checkpoint"
        )
    revision_inputs = (
        _validated_revision_sidecars(
            root_dir=root_dir,
            new_manifest_paths=new_manifest_paths,
            revision_sidecar_paths=sidecar_paths,
            expected_sidecar_sha256=expected_sidecar_hashes,
            snapshots=snapshots,
        )
        if sidecar_paths or expected_sidecar_hashes
        else []
    )
    corpus_snapshot = snapshots.read(corpus_manifest_path)
    previous_snapshot = snapshots.read(previous_partition_path)
    corpus_manifest = _json_from_snapshot(corpus_snapshot)
    previous_partition = _json_from_snapshot(previous_snapshot)
    previous_sha = previous_snapshot.sha256
    corpus_documents, previous_documents = _validate_parent_relationship(corpus_manifest, previous_partition)
    previous_by_sha = {str(document["sha256"]): document for document in previous_documents}
    expected_parent_splits = _expected_parent_splits(corpus_documents, previous_documents)
    fixed: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for document in corpus_documents:
        if _is_layoutlm(document):
            excluded.append({"id": str(document["id"]), "reason": "LayoutLM family excluded by policy"})
            continue
        sha256 = str(document["sha256"])
        old = previous_by_sha.get(sha256)
        fixed.append(
            _fixed_parent_output(
                document,
                source_manifest=str(corpus_manifest_path.relative_to(root_dir)),
                split=expected_parent_splits[sha256][1],
                previous_split=old.get("split") if old is not None else None,
            )
        )

    existing_hashes = {str(document["sha256"]) for document in fixed}
    existing_urls = {_derived_normalized_url(document) for document in fixed} - {""}
    existing_titles = {
        (
            _producer(document),
            _normalized(str(document.get("normalized_title") or document.get("title") or "")),
        )
        for document in fixed
    }
    existing_titles = {key for key in existing_titles if key[1]}
    new_by_hash: dict[str, dict[str, Any]] = {}
    new_by_url: dict[str, dict[str, Any]] = {}
    new_by_title: dict[tuple[str, str], dict[str, Any]] = {}
    source_inputs: list[dict[str, Any]] = []
    for path in new_manifest_paths:
        input_snapshot = snapshots.read(path)
        records = _records_from_snapshot(input_snapshot)
        checkpoint = _validate_checkpoint(path, records, snapshots)
        accepted = 0
        for raw_document in records:
            document = _adapt_checkpoint_document(raw_document, path)
            _validate_new_document(document, path)
            if _is_layoutlm(document):
                excluded.append({
                    "id": str(document.get("id", document["title"])),
                    "reason": "LayoutLM family excluded by policy",
                })
                continue
            sha256 = str(document["sha256"])
            if sha256 in existing_hashes:
                raise ValueError(f"new manifest duplicates existing corpus SHA-256: {sha256}")
            enriched = _enrich(document, str(path.relative_to(root_dir)))
            enriched.setdefault("id", f"v3-{sha256[:16]}")
            if sha256 in new_by_hash:
                raise ValueError(f"new manifests contain duplicate SHA-256: {sha256}")
            normalized_url = _normalized_url(enriched)
            if normalized_url in existing_urls:
                raise ValueError(f"new manifest duplicates existing corpus URL: {normalized_url}")
            if normalized_url in new_by_url:
                raise ValueError(f"new manifests contain duplicate URL: {normalized_url}")
            normalized_title = _normalized(
                str(enriched.get("normalized_title") or enriched.get("title") or "")
            )
            title_publisher = (_producer(enriched), normalized_title)
            if title_publisher in existing_titles:
                raise ValueError(
                    f"new manifest duplicates existing normalized title and publisher: {title_publisher}"
                )
            if title_publisher in new_by_title:
                raise ValueError(
                    f"new manifests contain duplicate normalized title and publisher: {title_publisher}"
                )
            new_by_hash[sha256] = enriched
            new_by_url[normalized_url] = enriched
            new_by_title[title_publisher] = enriched
            accepted += 1
        checkpoint_descriptor: dict[str, Any] | None = None
        if checkpoint is not None:
            checkpoint_descriptor = {}
            for key, value in checkpoint.items():
                if key.endswith("_path"):
                    checkpoint_descriptor[key] = str(Path(str(value)).relative_to(root_dir))
                else:
                    checkpoint_descriptor[key] = value
        source_inputs.append({
            "path": str(path.relative_to(root_dir)),
            "sha256": input_snapshot.sha256,
            "size_bytes": input_snapshot.size_bytes,
            "accepted_unique_documents": accepted,
            "checkpoint_validation": checkpoint_descriptor,
        })
    new_documents = sorted(new_by_hash.values(), key=lambda document: str(document["sha256"]))
    retained_count = len(fixed) + len(new_documents)
    if status == "final" and retained_count < 816:
        raise ValueError(f"post-dedup final corpus is below the 816-document floor: {retained_count}")
    forced_new, variable_groups = _atomic_groups(fixed, new_documents, revision)
    optimization_fixed = fixed + forced_new
    ideal_targets = document_targets(len(fixed) + len(new_documents))
    fixed_counts: dict[Split, int] = {
        split: sum(document["split"] == split for document in optimization_fixed) for split in SPLITS
    }
    selected_targets, family_exact = _reachable_targets(fixed_counts, variable_groups, ideal_targets)
    if not family_exact or selected_targets != ideal_targets:
        raise ValueError(
            f"half-up/residual targets are not reachable with atomic families: {selected_targets} != {ideal_targets}"
        )
    assignments = _optimize(optimization_fixed, variable_groups, ideal_targets, salt)
    assigned_new: list[dict[str, Any]] = list(forced_new)
    for group in variable_groups:
        split = assignments[group.family_id]
        for document in group.documents:
            document.update({
                "split": split,
                "assignment_origin": f"new-{revision}",
                "assignment_rationale": (
                    f"Accuracy-uninspected new PDF assigned as part of atomic {document['family_id']} family; "
                    "deterministic optimizer prioritized multi-stratum balance, then page balance."
                ),
            })
            assigned_new.append(document)
    documents = sorted(
        fixed + assigned_new, key=lambda document: (SPLITS.index(document["split"]), document["id"])
    )
    checks = _verify(
        documents,
        corpus_manifest,
        previous_partition,
        selected_targets,
        parent_source_manifest=str(corpus_manifest_path.relative_to(root_dir)),
        strict_document_schema=status == "final",
    )
    summary = _summary(documents)
    forced_count, grandfathered_conflicts = _family_declarations(
        documents, {str(document["sha256"]) for document in fixed}
    )
    common = {
        "schema_version": {"v3": "3.0", "v4": "4.0", "v5": "5.0"}[revision],
        "status": status,
        "generated_date": "2026-07-31",
        "policy": _partition_policy(revision),
        "parent_manifest": {
            "role": "parent_manifest",
            "path": str(corpus_manifest_path.relative_to(root_dir)),
            "sha256": corpus_snapshot.sha256,
            "size_bytes": corpus_snapshot.size_bytes,
            "record_count": len(corpus_documents),
        },
        "parent_partition": {
            "role": "parent_partition",
            "path": str(previous_partition_path.relative_to(root_dir)),
            "sha256": previous_sha,
            "size_bytes": previous_snapshot.size_bytes,
            "record_count": len(previous_documents),
            "unchanged": True,
        },
        "inputs": source_inputs,
        "authenticated_revision_sidecars": revision_inputs,
        "excluded_documents": excluded,
        "ideal_document_targets": ideal_targets,
        "selected_document_targets": selected_targets,
        "ideal_targets_family_reachable": family_exact,
        "new_documents_forced_by_existing_family": forced_count,
        "grandfathered_fixed_family_conflicts": grandfathered_conflicts,
        "summary": summary,
        "integrity_checks": checks,
        "selection": {
            "previously_active_documents": len(fixed),
            "new_accuracy_uninspected_documents": len(assigned_new),
            "retained_unique_documents": len(documents),
        },
    }
    manifest = {**common, "documents": documents}
    manifest_bytes = _serialized_json(manifest)
    partition = {
        **common,
        "manifest": {
            "path": str(output_manifest_path.relative_to(root_dir)),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size_bytes": len(manifest_bytes),
            "record_count": len(documents),
        },
        "output_descriptor": canonical_output_descriptor,
        "strata_summary": _strata_summary(documents),
        "documents": documents,
    }
    if status == "final":
        _validate_output_contract(
            manifest=manifest,
            partition=partition,
            parent_manifest=corpus_manifest,
            parent_partition=previous_partition,
            revision=revision,
        )
    partition_bytes = _serialized_json(partition)
    outputs = [
        (output_manifest_path, manifest_bytes),
        (output_partition_path, partition_bytes),
    ]
    if output_descriptor_path is not None:
        sidecars_by_revision = {
            str(item["revision_path"]): root_dir / str(item["path"]) for item in revision_inputs
        }
        authenticated_inputs: list[dict[str, Any]] = []
        for input_path in new_manifest_paths:
            authenticated_inputs.append(
                _commitment_descriptor(
                    role="accepted", path=input_path, root_dir=root_dir, snapshots=snapshots
                )
            )
            kind = _checkpoint_kind(input_path)
            if kind is not None:
                source_path = input_path.with_name(
                    "source_manifest.jsonl" if kind in ("difficult", "gap") else "source_manifest.json"
                )
                for role, companion_path in (
                    ("source", source_path),
                    ("rejected", input_path.with_name("rejected_manifest.jsonl")),
                    ("summary", input_path.with_name("summary.json")),
                ):
                    authenticated_inputs.append(
                        _commitment_descriptor(
                            role=role,
                            path=companion_path,
                            root_dir=root_dir,
                            snapshots=snapshots,
                        )
                    )
            relative_input = str(input_path.relative_to(root_dir))
            sidecar_path = sidecars_by_revision.get(relative_input)
            if sidecar_path is not None:
                authenticated_inputs.append(
                    _commitment_descriptor(
                        role="sidecar", path=sidecar_path, root_dir=root_dir, snapshots=snapshots
                    )
                )
        if status == "final" and [item["role"] for item in authenticated_inputs] != [
            role
            for _input_path in new_manifest_paths
            for role in ("accepted", "source", "rejected", "summary", "sidecar")
        ]:
            raise ValueError("final output descriptor lacks complete ordered authenticated inputs")
        descriptor = {
            "schema_version": "2.0",
            "revision_id": f"corpus-partition-{revision}",
            "immutable": True,
            "status": status,
            "parents": {
                "manifest": common["parent_manifest"],
                "partition": common["parent_partition"],
            },
            "input_commitment_sha256": _input_commitment_sha256(authenticated_inputs),
            "authenticated_inputs": authenticated_inputs,
            "outputs": {
                "manifest": {
                    "path": str(output_manifest_path.relative_to(root_dir)),
                    "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "size_bytes": len(manifest_bytes),
                    "record_count": len(documents),
                },
                "partition": {
                    "path": str(output_partition_path.relative_to(root_dir)),
                    "sha256": hashlib.sha256(partition_bytes).hexdigest(),
                    "size_bytes": len(partition_bytes),
                    "record_count": len(documents),
                },
            },
        }
        outputs.append((output_descriptor_path, _serialized_json(descriptor)))
    if any(path.exists() for path, _payload in outputs):
        raise FileExistsError("versioned partition outputs must not already exist")
    _publish_bundle(outputs, snapshots.revalidate)
    return BuildResult(
        manifest_path=output_manifest_path,
        partition_path=output_partition_path,
        descriptor_path=output_descriptor_path,
        document_counts={split: int(summary[split]["document_count"]) for split in SPLITS},
        page_counts={split: int(summary[split]["page_count"]) for split in SPLITS},
        status=status,
    )


def build_corpus_partition(
    *,
    root_dir: Path,
    corpus_manifest_path: Path,
    previous_partition_path: Path,
    new_manifest_paths: list[Path],
    output_manifest_path: Path,
    output_partition_path: Path,
    status: Literal["provisional", "final"],
    output_descriptor_path: Path | None = None,
    revision: Literal["v3", "v4", "v5"] | None = None,
    revision_sidecar_paths: list[Path] | None = None,
    expected_revision_sidecar_sha256: list[str] | None = None,
) -> BuildResult:
    """Build an immutable deterministic corpus revision from metadata only.

    This function never opens PDFs or any candidate, bronze, silver, evaluation, or
    element-failure artifact. It consumes corpus/staging manifests exclusively.
    """
    snapshots = _SnapshotStore()
    primary_error: BaseException | None = None
    try:
        return _build_corpus_partition(
            root_dir=root_dir,
            corpus_manifest_path=corpus_manifest_path,
            previous_partition_path=previous_partition_path,
            new_manifest_paths=new_manifest_paths,
            output_manifest_path=output_manifest_path,
            output_partition_path=output_partition_path,
            status=status,
            output_descriptor_path=output_descriptor_path,
            revision=revision,
            revision_sidecar_paths=revision_sidecar_paths,
            expected_revision_sidecar_sha256=expected_revision_sidecar_sha256,
            snapshots=snapshots,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _close_snapshot_store(snapshots, primary_error)
