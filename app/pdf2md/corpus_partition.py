from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit, urlunsplit

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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            records.append(cast(dict[str, Any], value))
        return records
    manifest = _read_json(path)
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"manifest has no documents array: {path}")
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"expected JSON object at {path}:documents[{index}]")
    return [cast(dict[str, Any], document) for document in documents]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(value: str) -> str:
    return _NON_WORD.sub(" ", value.lower()).strip()


def _normalize_url_value(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip())
    return urlunsplit((
        parsed.scheme.casefold(),
        parsed.netloc.casefold(),
        unquote(parsed.path).rstrip("/"),
        "",
        "",
    ))


def _derived_normalized_url(document: dict[str, Any]) -> str:
    raw_url = str(document.get("source_url") or document.get("canonical_url") or "")
    return _normalize_url_value(raw_url) if raw_url.strip() else ""


def _derived_title_publisher_key(document: dict[str, Any]) -> tuple[str, str]:
    explicit = (
        document.get("source_producer")
        or document.get("producer")
        or document.get("issuer")
        or document.get("jurisdiction_or_publisher")
    )
    publisher = str(explicit or "")
    title = str(document.get("normalized_title") or document.get("title") or "")
    return _normalized(publisher), _normalized(title)


def _normalized_url(document: dict[str, Any]) -> str:
    deduplication_keys = document.get("deduplication_keys")
    curated_url = deduplication_keys.get("normalized_url") if isinstance(deduplication_keys, dict) else None
    raw_url = str(curated_url or document.get("canonical_url") or document.get("source_url") or "").strip()
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
        f"title:{_normalized(str(document.get('normalized_title') or document.get('title') or ''))}",
        f"template:{_normalized(str(document.get('template_family') or document.get('family_id') or ''))}",
    }
    report_series = _normalized(str(document.get("report_series") or ""))
    edition = _normalized(str(document.get("edition") or ""))
    if report_series:
        keys.add(f"publisher-report-series:{producer}|{report_series}")
    if edition:
        keys.add(f"producer-edition:{producer}|{edition}")
    return tuple(sorted(key for key in keys if not key.endswith(":")))


def _atomic_groups(
    fixed: list[dict[str, Any]], new_documents: list[dict[str, Any]], revision: str
) -> tuple[list[dict[str, Any]], list[_Group]]:
    documents = fixed + new_documents
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

    forced_new: list[dict[str, Any]] = []
    variable_groups: list[_Group] = []
    fixed_count = len(fixed)
    for indices in sorted(
        components.values(), key=lambda values: min(str(documents[i]["sha256"]) for i in values)
    ):
        component_hashes = sorted(str(documents[index]["sha256"]) for index in indices)
        component_id = "atomic-v3:" + hashlib.sha256("|".join(component_hashes).encode()).hexdigest()[:24]
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
    if path.parent.name in ("v3-financial", "v3-financial-r2"):
        return "financial"
    if path.parent.name == "v3-gap":
        return "gap"
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"checkpoint companion is missing: {path}")
    return _read_records(path)


def _validate_checkpoint(path: Path, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    kind = _checkpoint_kind(path)
    if kind is None:
        return None
    summary_path = path.with_name("summary.json")
    source_path = path.with_name(
        "source_manifest.jsonl" if kind in ("difficult", "gap") else "source_manifest.json"
    )
    rejected_path = path.with_name("rejected_manifest.jsonl")
    for companion in (summary_path, source_path, rejected_path):
        if not companion.is_file():
            raise ValueError(f"checkpoint companion is missing: {companion}")
    if summary_path.stat().st_mtime_ns < path.stat().st_mtime_ns:
        raise ValueError(f"checkpoint summary is stale relative to accepted manifest: {path}")
    if path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
        raise ValueError(f"accepted manifest is stale relative to source manifest: {path}")

    summary = _read_json(summary_path)
    rejected = _load_jsonl(rejected_path)
    accepted_urls = {str(row.get("source_url", "")) for row in records}
    if kind in ("difficult", "gap"):
        sources = _load_jsonl(source_path)
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
        if {str(row.get("slug", "")) for row in sources} != {
            str(row.get("slug", "")) for row in records + rejected
        }:
            raise ValueError(f"v3 {kind} checkpoint outcomes do not cover its sources")
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
        source_urls = {str(row.get("source_url", "")) for row in sources}
    else:
        source = _read_json(source_path)
        sources_value = source.get("sources")
        if not isinstance(sources_value, list):
            raise ValueError("v3 financial source manifest has no sources array")
        for index, row in enumerate(sources_value):
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {source_path}:sources[{index}]")
        sources = [cast(dict[str, Any], row) for row in sources_value]
        if source.get("candidate_status") != "unassigned" or source.get("accuracy_inspected") is not False:
            raise ValueError("v3 financial checkpoint lacks an uninspected/unassigned attestation")
        expected_counts = (540, 171, 369)
        actual_counts = (len(sources), len(records), len(rejected))
        summary_counts = (
            int(summary.get("source_count", -1)),
            int(summary.get("accepted", -1)),
            int(summary.get("rejected", -1)),
        )
        if actual_counts != expected_counts or summary_counts != expected_counts:
            raise ValueError(
                "v3 financial checkpoint completeness counts do not match "
                f"the required source/accepted/rejected counts: {actual_counts}"
            )
        source_identities = [
            (str(row.get("id", "")).strip(), str(row.get("source_url", "")).strip()) for row in sources
        ]
        outcome_identities = [
            (str(row.get("id", "")).strip(), str(row.get("source_url", "")).strip())
            for row in records + rejected
        ]
        if any(not source_id or not source_url for source_id, source_url in source_identities):
            raise ValueError("v3 financial source manifest contains a blank source identity")
        if any(not source_id or not source_url for source_id, source_url in outcome_identities):
            raise ValueError("v3 financial outcome manifest contains a blank source identity")
        if len(set(source_identities)) != len(source_identities):
            raise ValueError("v3 financial source manifest contains duplicate source identities")
        if len(set(outcome_identities)) != len(outcome_identities):
            raise ValueError("v3 financial outcome manifests contain duplicate source identities")
        if set(source_identities) != set(outcome_identities):
            raise ValueError("v3 financial outcomes do not exactly cover source identities")
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
        source_urls = {str(row.get("source_url", "")) for row in sources}
    if not accepted_urls <= source_urls:
        raise ValueError(f"{kind} accepted checkpoint contains a source not in its source manifest")
    if any(row.get("status") != "accepted" for row in records):
        raise ValueError(f"{kind} accepted checkpoint contains a non-accepted record")
    return {
        "kind": kind,
        "summary_path": str(summary_path),
        "summary_sha256": _digest(summary_path),
        "source_path": str(source_path),
        "source_sha256": _digest(source_path),
        "rejected_path": str(rejected_path),
        "rejected_sha256": _digest(rejected_path),
    }


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
    """Return Hamilton largest-remainder targets with split order as the exact tie-break."""
    if total < 0:
        raise ValueError("total document count cannot be negative")
    ratio_fractions: dict[Split, Fraction] = {
        "train": Fraction(3, 5),
        "validation": Fraction(1, 10),
        "holdout": Fraction(3, 10),
    }
    quotas = {split: total * ratio_fractions[split] for split in SPLITS}
    targets: dict[Split, int] = {
        split: quotas[split].numerator // quotas[split].denominator for split in SPLITS
    }
    remainder = total - sum(targets.values())
    ranked = sorted(SPLITS, key=lambda split: (-(quotas[split] - targets[split]), SPLITS.index(split)))
    for split in ranked[:remainder]:
        targets[split] += 1
    return targets


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
    documents: list[dict[str, Any]], previous_partition: dict[str, Any], targets: dict[Split, int]
) -> list[str]:
    ids = [str(document["id"]) for document in documents]
    hashes = [str(document["sha256"]) for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate document IDs after consolidation")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate SHA-256 values after consolidation")
    if any(_is_layoutlm(document) for document in documents):
        raise ValueError("LayoutLM/LayoutLMv2 remained in active corpus")
    by_sha = {str(document["sha256"]): document for document in documents}
    for previous in cast(list[dict[str, Any]], previous_partition["documents"]):
        current = by_sha.get(str(previous["sha256"]))
        if current is None and not _is_layoutlm(previous):
            raise ValueError(f"previous-partition member missing: {previous['id']}")
        if current is not None:
            expected: Split = "train" if previous["split"] == "dev" else cast(Split, previous["split"])
            if current["split"] != expected:
                raise ValueError(f"previous-partition membership changed for {previous['id']}")
    new_family_splits: dict[str, set[str]] = defaultdict(set)
    fixed_family_splits: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        family = str(document["family_id"])
        if str(document["assignment_origin"]).startswith("new-v"):
            new_family_splits[family].add(str(document["split"]))
        else:
            fixed_family_splits[family].add(str(document["split"]))
    leaking = {family: splits for family, splits in new_family_splits.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"new family leakage detected: {leaking}")
    for family, new_splits in new_family_splits.items():
        prior_splits = fixed_family_splits.get(family, set())
        if prior_splits and "train" in prior_splits and new_splits != {"train"}:
            raise ValueError(f"new member of burned TRAIN family escaped TRAIN: {family}")
        if len(prior_splits) == 1 and new_splits != prior_splits:
            raise ValueError(f"new member crossed an existing family boundary: {family}")
    counts = {split: sum(document["split"] == split for document in documents) for split in SPLITS}
    if counts != targets:
        raise ValueError(f"document counts do not match selected targets: {counts} != {targets}")
    return [
        "checkpoint metadata freshness and source/outcome completeness validated",
        "unique document IDs/SHA-256 values and no new canonical-URL/title duplicates",
        "LayoutLM and LayoutLMv2 absent from active corpus",
        "every partition-v2 membership and previously burned TRAIN assignment preserved",
        "new hash/URL/title/report-series/edition/template components assigned atomically",
        "document counts equal exact Hamilton targets",
        "new candidates are checkpoint-attested accuracy-uninspected and unassigned",
    ]


def _validated_revision_sidecars(
    *,
    root_dir: Path,
    new_manifest_paths: list[Path],
    revision_sidecar_paths: list[Path],
    expected_sidecar_sha256: list[str],
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
    for sidecar_path, expected_sha256 in zip(revision_sidecar_paths, expected_sidecar_sha256, strict=True):
        sidecar_path = sidecar_path.resolve()
        try:
            sidecar_path.relative_to(root_dir.resolve())
        except ValueError as error:
            raise ValueError(f"revision sidecar escapes repository root: {sidecar_path}") from error
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"invalid expected revision sidecar SHA-256: {expected_sha256}")
        actual_sidecar_sha256 = _digest(sidecar_path)
        if actual_sidecar_sha256 != expected_sha256:
            raise ValueError(f"revision sidecar SHA-256 mismatch: {sidecar_path}")
        sidecar = _read_json(sidecar_path)
        if sidecar.get("immutable") is not True:
            raise ValueError(f"revision sidecar is not immutable: {sidecar_path}")
        descriptor = sidecar.get("revision") or sidecar.get("output")
        parent = sidecar.get("parent")
        if not isinstance(descriptor, dict) or not isinstance(parent, dict):
            raise ValueError(f"revision sidecar is missing revision/output or parent: {sidecar_path}")
        required_descriptor_fields = {"path", "sha256", "size_bytes", "record_count"}
        for label, payload in (("revision/output", descriptor), ("parent", parent)):
            missing = sorted(required_descriptor_fields - payload.keys())
            if missing:
                raise ValueError(f"revision sidecar {label} is missing {', '.join(missing)}: {sidecar_path}")
        revision_path = (root_dir / str(descriptor["path"])).resolve()
        try:
            revision_path.relative_to(root_dir.resolve())
        except ValueError as error:
            raise ValueError(f"revision path escapes repository root: {revision_path}") from error
        manifest_path = manifests_by_path.get(revision_path)
        if manifest_path is None:
            raise ValueError(f"revision sidecar does not match a supplied manifest: {sidecar_path}")
        if descriptor.get("sha256") != _digest(manifest_path):
            raise ValueError(f"revision manifest SHA-256 mismatch: {manifest_path}")
        if descriptor["size_bytes"] != manifest_path.stat().st_size:
            raise ValueError(f"revision manifest size mismatch: {manifest_path}")
        records = _read_records(manifest_path)
        if descriptor["record_count"] != len(records):
            raise ValueError(f"revision manifest record count mismatch: {manifest_path}")
        parent_path = (root_dir / str(parent.get("path", ""))).resolve()
        try:
            parent_path.relative_to(root_dir.resolve())
        except ValueError as error:
            raise ValueError(f"revision parent escapes repository root: {parent_path}") from error
        if not parent_path.is_file() or parent.get("sha256") != _digest(parent_path):
            raise ValueError(f"revision parent SHA-256 mismatch: {parent_path}")
        if parent.get("size_bytes") not in (None, parent_path.stat().st_size):
            raise ValueError(f"revision parent size mismatch: {parent_path}")
        validated.append({
            "path": str(sidecar_path.relative_to(root_dir)),
            "sha256": actual_sidecar_sha256,
            "revision_path": str(manifest_path.relative_to(root_dir)),
            "parent_path": str(parent_path.relative_to(root_dir)),
            "parent_sha256": str(parent["sha256"]),
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
    if revision == "v5" and not any(path.parent.name == "v3-financial-r2" for path in new_manifest_paths):
        raise ValueError("final v5 status requires the corrected v3-financial-r2 checkpoint")


def build_corpus_partition(
    *,
    root_dir: Path,
    corpus_manifest_path: Path,
    previous_partition_path: Path,
    new_manifest_paths: list[Path],
    output_manifest_path: Path,
    output_partition_path: Path,
    status: Literal["provisional", "final"],
    revision_sidecar_paths: list[Path] | None = None,
    expected_revision_sidecar_sha256: list[str] | None = None,
) -> BuildResult:
    """Build an immutable deterministic corpus revision from metadata only.

    This function never opens PDFs or any candidate, bronze, silver, evaluation, or
    element-failure artifact. It consumes corpus/staging manifests exclusively.
    """
    if output_manifest_path.exists() or output_partition_path.exists():
        raise FileExistsError("versioned partition outputs must not already exist")
    manifest_match = re.fullmatch(r"manifest-(v[345])\.json", output_manifest_path.name)
    partition_match = re.fullmatch(r"partition-(v[345])\.json", output_partition_path.name)
    if manifest_match is None or partition_match is None:
        raise ValueError("output filenames must use an explicitly supported v3, v4, or v5 revision")
    manifest_revision = manifest_match.group(1)
    partition_revision = partition_match.group(1)
    if manifest_revision != partition_revision:
        raise ValueError("manifest and partition output revisions do not match")
    revision = manifest_revision
    salt = SALTS[revision]
    if status == "final":
        _validate_final_inputs(new_manifest_paths, revision)
    sidecar_paths = revision_sidecar_paths or []
    expected_sidecar_hashes = expected_revision_sidecar_sha256 or []
    revision_inputs = (
        _validated_revision_sidecars(
            root_dir=root_dir,
            new_manifest_paths=new_manifest_paths,
            revision_sidecar_paths=sidecar_paths,
            expected_sidecar_sha256=expected_sidecar_hashes,
        )
        if sidecar_paths or expected_sidecar_hashes
        else []
    )
    corpus_manifest = _read_json(corpus_manifest_path)
    previous_partition = _read_json(previous_partition_path)
    previous_sha = _digest(previous_partition_path)
    previous_by_sha = {
        str(document["sha256"]): document
        for document in cast(list[dict[str, Any]], previous_partition["documents"])
    }
    fixed: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    corpus_documents = cast(list[dict[str, Any]], corpus_manifest.get("documents", []))
    for document in corpus_documents:
        if _is_layoutlm(document):
            excluded.append({"id": str(document["id"]), "reason": "LayoutLM family excluded by policy"})
            continue
        enriched = _enrich(document, str(corpus_manifest_path.relative_to(root_dir)))
        old = previous_by_sha.get(str(document["sha256"]))
        if old is None:
            split: Split = "train"
            rationale = "Previously active and therefore exposed/burned; permanently TRAIN."
        elif old["split"] == "dev":
            split = "train"
            rationale = "Previous DEV exposure is permanent; mapped to TRAIN."
        else:
            split = cast(Split, old["split"])
            rationale = f"Preserved immutable previous-partition {split} membership."
        enriched.update({
            "split": split,
            "assignment_origin": "fixed-previous",
            "assignment_rationale": rationale,
        })
        fixed.append(enriched)

    existing_hashes = {str(document["sha256"]) for document in fixed}
    existing_urls = {_normalized_url(document) for document in fixed} - {""}
    existing_titles = {
        _normalized(str(document.get("normalized_title") or document.get("title") or ""))
        for document in fixed
    } - {""}
    new_by_hash: dict[str, dict[str, Any]] = {}
    new_by_url: dict[str, dict[str, Any]] = {}
    new_by_title: dict[str, dict[str, Any]] = {}
    source_inputs: list[dict[str, Any]] = []
    for path in new_manifest_paths:
        records = _read_records(path)
        checkpoint = _validate_checkpoint(path, records)
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
            if normalized_title in existing_titles:
                raise ValueError(
                    f"new manifest duplicates existing normalized title and publisher: {normalized_title}"
                )
            if normalized_title in new_by_title:
                raise ValueError(
                    f"new manifests contain duplicate normalized title and publisher: {normalized_title}"
                )
            new_by_hash[sha256] = enriched
            new_by_url[normalized_url] = enriched
            new_by_title[normalized_title] = enriched
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
            "sha256": _digest(path),
            "size_bytes": path.stat().st_size,
            "accepted_unique_documents": accepted,
            "checkpoint_validation": checkpoint_descriptor,
        })
    new_documents = sorted(new_by_hash.values(), key=lambda document: str(document["sha256"]))
    forced_new, variable_groups = _atomic_groups(fixed, new_documents, revision)
    optimization_fixed = fixed + forced_new
    ideal_targets = document_targets(len(fixed) + len(new_documents))
    fixed_counts: dict[Split, int] = {
        split: sum(document["split"] == split for document in optimization_fixed) for split in SPLITS
    }
    selected_targets, family_exact = _reachable_targets(fixed_counts, variable_groups, ideal_targets)
    if not family_exact or selected_targets != ideal_targets:
        raise ValueError(
            f"Hamilton targets are not reachable with atomic families: {selected_targets} != {ideal_targets}"
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
    if revision in ("v4", "v5") and len(documents) < 816:
        raise ValueError(f"post-dedup {revision} corpus is below the 816-document floor: {len(documents)}")
    checks = _verify(documents, previous_partition, selected_targets)
    summary = _summary(documents)
    fixed_family_membership: dict[str, set[str]] = defaultdict(set)
    for document in fixed:
        fixed_family_membership[str(document["family_id"])].add(str(document["split"]))
    grandfathered_conflicts = [
        {"family_id": family, "fixed_splits": sorted(splits, key=SPLITS.index)}
        for family, splits in sorted(fixed_family_membership.items())
        if len(splits) > 1
    ]
    common = {
        "schema_version": {"v3": "3.0", "v4": "4.0", "v5": "5.0"}[revision],
        "status": status,
        "generated_date": "2026-07-31",
        "policy": {
            "ratios": RATIOS,
            "ratio_basis": "document count after SHA-256 consolidation and LayoutLM-family exclusion",
            "rounding": (
                "Hamilton largest remainder over exact 3/5, 1/10, and 3/10 quotas; "
                "TRAIN, validation, holdout order breaks equal remainders"
            ),
            "partition_salt": salt,
            "assignment_priority": "exact Hamilton document targets; strata balance; page balance; salted tie-break",
            "family_rule": (
                "transitive atomic components over SHA-256, canonical URL, normalized title, "
                "publisher/report series, producer/edition, and template family; new members of an "
                "existing component inherit its fixed split"
            ),
            "holdout_evidence_rule": (
                "manifest metadata only; no candidate output, bronze, silver, evaluation, element "
                "failures, validation PDFs, or holdout PDFs inspected"
            ),
        },
        "parent_partition": {
            "path": str(previous_partition_path.relative_to(root_dir)),
            "sha256": previous_sha,
            "unchanged": _digest(previous_partition_path) == previous_sha,
        },
        "inputs": source_inputs,
        "authenticated_revision_sidecars": revision_inputs,
        "excluded_documents": excluded,
        "ideal_document_targets": ideal_targets,
        "selected_document_targets": selected_targets,
        "ideal_targets_family_reachable": family_exact,
        "new_documents_forced_by_existing_family": len(forced_new),
        "grandfathered_fixed_family_conflicts": grandfathered_conflicts,
        "summary": summary,
        "integrity_checks": checks,
    }
    manifest = {
        **common,
        "selection": {
            "previously_active_documents": len(fixed),
            "new_accuracy_uninspected_documents": len(assigned_new),
            "retained_unique_documents": len(documents),
        },
        "documents": documents,
    }
    partition = {
        **common,
        "manifest": str(output_manifest_path.relative_to(root_dir)),
        "strata_summary": _strata_summary(documents),
        "documents": documents,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_partition_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_partition_path.write_text(json.dumps(partition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BuildResult(
        manifest_path=output_manifest_path,
        partition_path=output_partition_path,
        document_counts={split: int(summary[split]["document_count"]) for split in SPLITS},
        page_counts={split: int(summary[split]["page_count"]) for split in SPLITS},
        status=status,
    )
