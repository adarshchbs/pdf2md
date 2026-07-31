from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import polars as pl

PROTECTED_FIELDS = ("id", "sha256", "split", "family_id")
BACKFILL_FIELDS = ("country", "region", "language_mix")
GENERIC_VALUES = {"", "global", "unknown", "unspecified"}
EXPECTED_INPUT_SHA256 = {
    "data/corpus/manifest-v5.json": "658f5591674b1ba50c6aa5a261ee9cb8ece1a17c8bc39ccb3fd30f99f980f3f2",
    "data/corpus/partition-v5.json": "b576123750f6f02de53e14633012e337111ed05f75b16af6c6f7acf221d4b9eb",
    "data/corpus-expansion/v3-difficult/source_manifest.jsonl": (
        "05cbde681ee8a10968529837872ea998d7353883d7b55211d5b84c6c40ceb6f6"
    ),
    "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl": (
        "ef775c528961ee1a95dbc8de84cbaa04e5a2f3e921cfba4eb45512b89cd707a8"
    ),
    "data/corpus-expansion/v3-financial-r2/source_manifest.json": (
        "5378665cf647f6930598647c960d4b9c1d7f34b744e9351a6dfd434ffcb375a1"
    ),
    "data/corpus-expansion/v3-financial-r2/accepted_manifest.jsonl": (
        "4a258db265198b63b5668be1d54e7155ebfbd5921a0fba8c3da28e68cb8d9f8f"
    ),
    "data/corpus-expansion/v3-gap/source_manifest.jsonl": (
        "c059cf62dfda7d6d1bdd0b44a5db26f25d9488f8d240da358ec2927c532b0fe4"
    ),
    "data/corpus-expansion/v3-gap/accepted_manifest.jsonl": (
        "7ded8c2545fb4932e66d7e0152ae25d7f7da3ed6c374e643c7c0a557854f6372"
    ),
}
SOURCE_PATHS = tuple(path for path in EXPECTED_INPUT_SHA256 if "corpus-expansion" in path)
EXPECTED_PROPOSAL_SHA256 = "bc6e98ef2c8773be1a16fb67a4940ee487a8d8fcb71a1caaa638fc699ef6df38"
EXPECTED_PROTECTED_MEMBERSHIP_SHA256 = "7d3ee30e02ec7041d1dc62210f2e461b944191227a16b859e2f9e97fbaa69b32"
SUPPORTED_RULE_IDS = {
    "brazil-central-bank-domain",
    "declared-source-metadata",
    "eu-institution-domain",
    "explicit-english-title-url-concordance",
    "uk-gov-domain",
    "us-gov-domain",
}
STRATA_DIMENSIONS = (
    "category",
    "country",
    "difficult_stratum",
    "document_stratum",
    "form_signal",
    "language_mix",
    "layout_signal",
    "page_count_band",
    "region",
    "rotation_signal",
    "source_manifest",
    "source_producer",
    "table_signal",
)
SPLITS = ("train", "validation", "holdout")
RATIOS = {"train": 0.60, "validation": 0.10, "holdout": 0.30}


@dataclass(frozen=True)
class Evidence:
    field: str
    value: str
    rule_id: str
    source: str
    locator: str
    evidence_value: str
    confidence: str = "exact"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        output: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            output.append(cast(dict[str, Any], value))
        return output
    value = _read_object(path)
    records = value.get("documents", value.get("sources"))
    if not isinstance(records, list):
        raise ValueError(f"expected documents or sources array: {path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"expected JSON object at {path}[{index}]")
    return [cast(dict[str, Any], record) for record in records]


def _generic(value: object) -> bool:
    return str(value or "").strip().casefold() in GENERIC_VALUES


def _normalized_url(document: dict[str, Any]) -> str:
    raw = str(document.get("source_url") or document.get("canonical_url") or "").strip()
    parsed = urlsplit(raw)
    return (
        parsed
        ._replace(
            scheme=parsed.scheme.casefold(),
            netloc=parsed.netloc.casefold(),
            query="",
            fragment="",
        )
        .geturl()
        .rstrip("/")
    )


def _hostname(document: dict[str, Any]) -> str:
    return (urlsplit(_normalized_url(document)).hostname or "").casefold().removeprefix("www.")


def exact_url_evidence(document: dict[str, Any]) -> list[Evidence]:
    host = _hostname(document)
    if not host:
        return []
    country: str | None = None
    region: str | None = None
    rule_id: str | None = None
    # These namespaces are jurisdiction-restricted. Generic ccTLDs and organization
    # names are deliberately not mapped.
    if host.endswith(".gov"):
        country, region, rule_id = "United States", "North America", "us-gov-domain"
    elif host == "gov.uk" or host.endswith(".gov.uk"):
        country, region, rule_id = "United Kingdom", "Europe", "uk-gov-domain"
    elif host == "bcb.gov.br" or host.endswith(".bcb.gov.br"):
        country, region, rule_id = "Brazil", "Latin America", "brazil-central-bank-domain"
    elif host == "europa.eu" or host.endswith(".europa.eu"):
        region, rule_id = "Europe", "eu-institution-domain"
    if rule_id is None:
        return []
    values = {"country": country, "region": region}
    return [
        Evidence(
            field=field,
            value=value,
            rule_id=rule_id,
            source="source_url",
            locator=host,
            evidence_value=_normalized_url(document),
        )
        for field, value in values.items()
        if value is not None
    ]


def exact_language_evidence(document: dict[str, Any]) -> list[Evidence]:
    title = str(document.get("title") or "").strip()
    url = str(document.get("source_url") or document.get("canonical_url") or "").strip()
    title_folded = title.casefold()
    url_folded = url.casefold()
    english_concordant = ("english" in title_folded and "english" in url_folded) or (
        "eng version" in title_folded and "eng-version" in url_folded
    )
    if not english_concordant:
        return []
    return [
        Evidence(
            field="language_mix",
            value="English",
            rule_id="explicit-english-title-url-concordance",
            source="declared_title+source_url",
            locator=str(document["id"]),
            evidence_value=f"title={title}; url={url}",
        )
    ]


def _source_evidence(root: Path) -> tuple[dict[str, list[Evidence]], dict[str, list[Evidence]]]:
    by_hash: dict[str, list[Evidence]] = defaultdict(list)
    by_url: dict[str, list[Evidence]] = defaultdict(list)
    for relative in SOURCE_PATHS:
        path = root / relative
        for row in _read_records(path):
            values = {
                "country": row.get("country"),
                "region": row.get("region"),
                "language_mix": row.get("language_mix") or row.get("language"),
            }
            for field, raw_value in values.items():
                if _generic(raw_value):
                    continue
                evidence = Evidence(
                    field=field,
                    value=str(raw_value).strip(),
                    rule_id="declared-source-metadata",
                    source=relative,
                    locator=str(row.get("id") or row.get("slug") or row.get("source_url") or ""),
                    evidence_value=str(raw_value).strip(),
                )
                sha256 = str(row.get("sha256") or "").strip()
                if sha256:
                    by_hash[sha256].append(evidence)
                normalized_url = _normalized_url(row)
                if normalized_url:
                    by_url[normalized_url].append(evidence)
    return by_hash, by_url


def _validate_inputs(root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_INPUT_SHA256.items():
        path = root / relative
        digest = _digest(path)
        if digest != expected:
            raise ValueError(f"metadata input SHA-256 mismatch: {relative}: {digest} != {expected}")
        actual[relative] = digest
    return actual


def build_proposal(root: Path) -> dict[str, Any]:
    input_hashes = _validate_inputs(root)
    manifest = _read_object(root / "data/corpus/manifest-v5.json")
    partition = _read_object(root / "data/corpus/partition-v5.json")
    documents_value = manifest.get("documents")
    if not isinstance(documents_value, list) or not all(isinstance(row, dict) for row in documents_value):
        raise ValueError("manifest-v5 documents must be JSON objects")
    documents = [cast(dict[str, Any], row) for row in documents_value]
    if partition.get("documents") != documents:
        raise ValueError("manifest-v5 and partition-v5 documents differ")
    if len(documents) != 851:
        raise ValueError(f"expected 851 v5 documents, found {len(documents)}")
    for field in PROTECTED_FIELDS:
        values = [str(document.get(field, "")) for document in documents]
        if (
            any(not value for value in values)
            or len(values) != len(set(values))
            and field in ("id", "sha256")
        ):
            raise ValueError(f"invalid protected field: {field}")

    source_by_hash, source_by_url = _source_evidence(root)
    proposals: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for document in documents:
        candidates = (
            source_by_hash.get(str(document["sha256"]), [])
            + source_by_url.get(_normalized_url(document), [])
            + exact_url_evidence(document)
            + exact_language_evidence(document)
        )
        for field in BACKFILL_FIELDS:
            if not _generic(document.get(field)):
                continue
            field_candidates = [candidate for candidate in candidates if candidate.field == field]
            distinct_values = sorted({candidate.value for candidate in field_candidates})
            if len(distinct_values) > 1:
                conflicts.append({
                    "id": document["id"],
                    "sha256": document["sha256"],
                    "field": field,
                    "values": distinct_values,
                    "rule_ids": sorted({candidate.rule_id for candidate in field_candidates}),
                })
                continue
            if len(distinct_values) == 1:
                selected = sorted(
                    field_candidates,
                    key=lambda item: (item.rule_id, item.source, item.locator, item.evidence_value),
                )[0]
                proposals.append({
                    **{key: document[key] for key in PROTECTED_FIELDS},
                    "field": field,
                    "old_value": str(document.get(field) or ""),
                    "new_value": selected.value,
                    "confidence": selected.confidence,
                    "rule_id": selected.rule_id,
                    "evidence_source": selected.source,
                    "evidence_locator": selected.locator,
                    "evidence_value": selected.evidence_value,
                })
            else:
                unresolved.append({
                    "id": document["id"],
                    "sha256": document["sha256"],
                    "field": field,
                    "current_value": str(document.get(field) or ""),
                    "reason": "no exact declared-source or restricted-domain evidence",
                })
    if conflicts:
        raise ValueError(f"conflicting exact metadata evidence: {conflicts}")
    proposals.sort(key=lambda row: (str(row["sha256"]), str(row["field"])))
    unresolved.sort(key=lambda row: (str(row["sha256"]), str(row["field"])))

    before_rows = [
        {field: str(document.get(field) or "") for field in BACKFILL_FIELDS} for document in documents
    ]
    before = (
        pl
        .DataFrame(before_rows)
        .select([
            pl.col(field).map_elements(_generic, return_dtype=pl.Boolean).sum().alias(field)
            for field in BACKFILL_FIELDS
        ])
        .to_dicts()[0]
    )
    applied_counts = (
        pl.DataFrame({"field": [str(row["field"]) for row in proposals]}).group_by("field").len().to_dicts()
        if proposals
        else []
    )
    reductions = {field: 0 for field in BACKFILL_FIELDS}
    for row in applied_counts:
        reductions[str(row["field"])] = int(row["len"])
    after = {field: int(before[field]) - reductions[field] for field in BACKFILL_FIELDS}
    protected_digest = hashlib.sha256(
        json.dumps(
            [{key: document[key] for key in PROTECTED_FIELDS} for document in documents],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "status": "proposal-only",
        "scope": "metadata-only; no PDFs or accuracy/holdout evidence inspected",
        "policy": {
            "generic_values": sorted(GENERIC_VALUES),
            "confidence_required": "exact",
            "ambiguous_names_or_generic_cc_tlds": "unresolved",
            "conflicting_exact_evidence": "fail-closed",
            "v5_overwrite": "forbidden",
        },
        "inputs": input_hashes,
        "protected_membership_sha256": protected_digest,
        "counts": {
            "documents": len(documents),
            "proposed_field_updates": len(proposals),
            "proposed_documents": len({str(row["sha256"]) for row in proposals}),
            "generic_before": {key: int(value) for key, value in before.items()},
            "generic_after_proposal": after,
            "generic_reduction": reductions,
            "unresolved_fields": len(unresolved),
        },
        "exact_mappings": proposals,
        "unresolved": unresolved,
        "conflicts": [],
    }


def protected_membership_digest(documents: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            [{key: document[key] for key in PROTECTED_FIELDS} for document in documents],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _evidence_identity(evidence: Evidence) -> tuple[str, str, str, str, str, str]:
    return (
        evidence.field,
        evidence.value,
        evidence.rule_id,
        evidence.source,
        evidence.locator,
        evidence.evidence_value,
    )


def validate_proposal(root: Path, proposal: dict[str, Any]) -> list[dict[str, Any]]:
    if proposal.get("schema_version") != "1.0" or proposal.get("status") != "proposal-only":
        raise ValueError("unsupported metadata proposal schema or status")
    if proposal.get("conflicts") != []:
        raise ValueError("metadata proposal contains unsupported evidence conflicts")
    if proposal.get("inputs") != _validate_inputs(root):
        raise ValueError("metadata proposal inputs are stale or incomplete")

    manifest = _read_object(root / "data/corpus/manifest-v5.json")
    partition = _read_object(root / "data/corpus/partition-v5.json")
    documents_value = manifest.get("documents")
    if not isinstance(documents_value, list) or not all(isinstance(row, dict) for row in documents_value):
        raise ValueError("manifest-v5 documents must be JSON objects")
    documents = [cast(dict[str, Any], row) for row in documents_value]
    if partition.get("documents") != documents:
        raise ValueError("manifest-v5 and partition-v5 documents differ")
    membership_digest = protected_membership_digest(documents)
    if membership_digest != EXPECTED_PROTECTED_MEMBERSHIP_SHA256:
        raise ValueError("protected v5 membership digest does not match the v6 contract")
    if proposal.get("protected_membership_sha256") != membership_digest:
        raise ValueError("metadata proposal protected membership digest is stale")

    mappings_value = proposal.get("exact_mappings")
    if not isinstance(mappings_value, list) or not all(isinstance(row, dict) for row in mappings_value):
        raise ValueError("metadata proposal exact_mappings must be JSON objects")
    mappings = [cast(dict[str, Any], row) for row in mappings_value]
    if len(mappings) != 225:
        raise ValueError(f"v6 requires exactly 225 exact mappings, found {len(mappings)}")
    by_hash = {str(document["sha256"]): document for document in documents}
    source_by_hash, source_by_url = _source_evidence(root)
    seen: set[tuple[str, str]] = set()
    for mapping in mappings:
        field = str(mapping.get("field", ""))
        sha256 = str(mapping.get("sha256", ""))
        identity = (sha256, field)
        if identity in seen:
            raise ValueError(f"duplicate metadata proposal mapping: {identity}")
        seen.add(identity)
        if field not in BACKFILL_FIELDS or mapping.get("confidence") != "exact":
            raise ValueError(f"unsupported metadata proposal field or confidence: {identity}")
        if mapping.get("rule_id") not in SUPPORTED_RULE_IDS:
            raise ValueError(f"unsupported metadata evidence rule: {mapping.get('rule_id')}")
        document = by_hash.get(sha256)
        if document is None:
            raise ValueError(f"metadata proposal references unknown SHA-256: {sha256}")
        if any(mapping.get(key) != document.get(key) for key in PROTECTED_FIELDS):
            raise ValueError(f"metadata proposal protected membership conflict: {identity}")
        if not _generic(document.get(field)) or str(mapping.get("old_value", "")) != str(
            document.get(field) or ""
        ):
            raise ValueError(f"metadata proposal old value is stale or non-generic: {identity}")
        candidates = (
            source_by_hash.get(sha256, [])
            + source_by_url.get(_normalized_url(document), [])
            + exact_url_evidence(document)
            + exact_language_evidence(document)
        )
        field_candidates = [candidate for candidate in candidates if candidate.field == field]
        distinct_values = {candidate.value for candidate in field_candidates}
        if distinct_values != {str(mapping.get("new_value", ""))}:
            raise ValueError(
                f"ambiguous or unsupported exact evidence: {identity}: {sorted(distinct_values)}"
            )
        mapping_evidence = (
            field,
            str(mapping["new_value"]),
            str(mapping["rule_id"]),
            str(mapping["evidence_source"]),
            str(mapping["evidence_locator"]),
            str(mapping["evidence_value"]),
        )
        if mapping_evidence not in {_evidence_identity(candidate) for candidate in field_candidates}:
            raise ValueError(f"metadata proposal evidence locator is unsupported: {identity}")

    rebuilt = build_proposal(root)
    if proposal != rebuilt:
        raise ValueError("metadata proposal is stale or differs from deterministic exact evidence")
    return documents


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
        row = by_split[split]
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
    for dimension in STRATA_DIMENSIONS:
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


def apply_proposal(
    *,
    root: Path,
    proposal_path: Path,
    output_manifest_path: Path,
    output_partition_path: Path,
    revision: Literal["v6"] | str,
) -> tuple[Path, Path]:
    if revision != "v6":
        raise ValueError(f"unsupported metadata backfill revision: {revision}")
    if output_manifest_path.name != "manifest-v6.json" or output_partition_path.name != "partition-v6.json":
        raise ValueError("v6 contract requires manifest-v6.json and partition-v6.json outputs")
    if output_manifest_path.exists() or output_partition_path.exists():
        raise FileExistsError("immutable v6 outputs must not already exist")
    proposal_sha256 = _digest(proposal_path)
    if proposal_sha256 != EXPECTED_PROPOSAL_SHA256:
        raise ValueError("metadata proposal SHA-256 is stale or unauthenticated")
    newest_input = max((root / relative).stat().st_mtime_ns for relative in EXPECTED_INPUT_SHA256)
    if proposal_path.stat().st_mtime_ns < newest_input:
        raise ValueError("metadata proposal is stale relative to an authenticated input")
    proposal = _read_object(proposal_path)
    documents = validate_proposal(root, proposal)
    before_membership = protected_membership_digest(documents)

    mapping_by_identity = {
        (str(mapping["sha256"]), str(mapping["field"])): mapping
        for mapping in cast(list[dict[str, Any]], proposal["exact_mappings"])
    }
    updated_documents = deepcopy(documents)
    applied = 0
    for document in updated_documents:
        for field in BACKFILL_FIELDS:
            mapping = mapping_by_identity.get((str(document["sha256"]), field))
            if mapping is None:
                continue
            document[field] = str(mapping["new_value"])
            applied += 1
    if applied != 225:
        raise ValueError(f"v6 applied update count differs from proposal: {applied}")
    after_membership = protected_membership_digest(updated_documents)
    if after_membership != before_membership:
        raise ValueError("v6 metadata application changed protected membership")

    manifest_v5_path = root / "data/corpus/manifest-v5.json"
    partition_v5_path = root / "data/corpus/partition-v5.json"
    manifest_v5 = _read_object(manifest_v5_path)
    partition_v5 = _read_object(partition_v5_path)
    exact_mappings = cast(list[dict[str, Any]], proposal["exact_mappings"])
    exact_mappings_sha256 = hashlib.sha256(
        json.dumps(exact_mappings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lineage = {
        "parent_manifest": {
            "path": "data/corpus/manifest-v5.json",
            "sha256": _digest(manifest_v5_path),
        },
        "parent_partition": {
            "path": "data/corpus/partition-v5.json",
            "sha256": _digest(partition_v5_path),
        },
        "proposal": {
            "path": str(proposal_path.relative_to(root)),
            "sha256": proposal_sha256,
            "exact_mappings_sha256": exact_mappings_sha256,
        },
        "authenticated_inputs": cast(dict[str, str], proposal["inputs"]),
        "protected_membership_sha256_before": before_membership,
        "protected_membership_sha256_after": after_membership,
    }
    metadata_backfill = {
        "task": 58,
        "mode": "exact-evidence-only",
        "applied_field_updates": applied,
        "applied_documents": len({str(mapping["sha256"]) for mapping in exact_mappings}),
        "fields": list(BACKFILL_FIELDS),
        "proposal_sha256": proposal_sha256,
        "exact_mappings_sha256": exact_mappings_sha256,
    }
    summary = _summary(updated_documents)
    base_integrity = [str(item) for item in cast(list[object], manifest_v5["integrity_checks"])]
    integrity_checks = base_integrity + [
        "225 independently audited exact-evidence metadata updates applied",
        "all 851 IDs, SHA-256 values, splits, and atomic families preserved",
        "protected membership digest unchanged before and after metadata backfill",
        "v5 artifacts unchanged and v6 outputs immutable",
    ]

    manifest = deepcopy(manifest_v5)
    manifest.update({
        "schema_version": "6.0",
        "status": "final",
        "generated_date": "2026-07-31",
        "summary": summary,
        "metadata_backfill": metadata_backfill,
        "lineage": lineage,
        "integrity_checks": integrity_checks,
        "documents": updated_documents,
    })
    partition = deepcopy(partition_v5)
    partition.update({
        "schema_version": "6.0",
        "status": "final",
        "generated_date": "2026-07-31",
        "manifest": str(output_manifest_path.relative_to(root)),
        "summary": summary,
        "metadata_backfill": metadata_backfill,
        "lineage": lineage,
        "integrity_checks": integrity_checks,
        "strata_summary": _strata_summary(updated_documents),
        "documents": updated_documents,
    })
    output_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_partition_path.write_text(json.dumps(partition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_manifest_path, output_partition_path


def render_report(proposal: dict[str, Any]) -> str:
    counts = cast(dict[str, Any], proposal["counts"])
    before = cast(dict[str, int], counts["generic_before"])
    after = cast(dict[str, int], counts["generic_after_proposal"])
    reduction = cast(dict[str, int], counts["generic_reduction"])
    exact = cast(list[dict[str, Any]], proposal["exact_mappings"])
    rules: dict[str, int] = defaultdict(int)
    for row in exact:
        rules[str(row["rule_id"])] += 1
    task56_remaining = {
        "country": max(386 - reduction["country"], 0),
        "region": max(331 - reduction["region"], 0),
        "language_mix": max(371 - reduction["language_mix"], 0),
    }
    lines = [
        "# Deterministic v5 metadata backfill proposal",
        "",
        "**Status:** proposal only; no v5 field, split, family, ID, or hash was changed, and no v6 artifact was created.",
        "",
        "The proposal uses declared source/checkpoint metadata, jurisdiction-restricted government URL namespaces, and explicit language markers that agree in both declared title and source URL. Ambiguous organization names, generic country-code domains, and titles without concordant URL evidence remain unresolved. Any conflicting exact evidence fails the build.",
        "",
        "## Quantified result",
        "",
        "| Field | Generic before | Exact proposals | Generic after | Task56 target still outstanding |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, label in (("country", "Country"), ("region", "Region"), ("language_mix", "Language mix")):
        lines.append(
            f"| {label} | {before[field]} | {reduction[field]} | {after[field]} | {task56_remaining[field]} |"
        )
    lines.extend([
        "",
        f"The {counts['proposed_field_updates']} exact field updates cover {counts['proposed_documents']} documents. The remaining {counts['unresolved_fields']} generic fields are explicitly retained as unresolved.",
        "",
        "## Exact-rule yield",
        "",
        "| Rule | Field updates |",
        "|---|---:|",
    ])
    for rule, count in sorted(rules.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{rule}` | {count} |")
    lines.extend([
        "",
        "## Decision",
        "",
        "No immutable manifest/partition revision is emitted. The proposal is deterministic and exactly evidenced, but most unknowns remain unresolved and the active partition builder has no v6 contract. A future revision should consume the machine-readable proposal only after an independent review, update country/region/language strata summaries without reassigning documents, and preserve the recorded protected-membership digest.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python -m app.pdf2md.corpus_metadata_backfill --root . --proposal data/corpus-metadata-backfill-v5-task57.json --report data/corpus-metadata-backfill-v5-task57.md",
        "```",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, proposal_path: Path, report_path: Path) -> None:
    proposal = build_proposal(root)
    proposal_text = json.dumps(proposal, indent=2, sort_keys=True) + "\n"
    report_text = render_report(proposal)
    proposal_path.write_text(proposal_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed v5 metadata backfill proposal")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    proposal_path = args.proposal if args.proposal.is_absolute() else root / args.proposal
    report_path = args.report if args.report.is_absolute() else root / args.report
    if proposal_path.exists() or report_path.exists():
        raise FileExistsError("proposal outputs must not already exist")
    write_outputs(root, proposal_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
