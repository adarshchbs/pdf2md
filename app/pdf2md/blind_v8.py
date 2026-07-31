from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import json
import math
import os
import secrets
import shutil
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, cast

from app.pdf2md.benchmark_batch import BenchmarkBatchManifest
from app.pdf2md.bronze import verify_bronze_bundle
from app.pdf2md.schema import DocumentElement, read_document_elements

_SCHEMA_VERSION = "8.0.0"
_FORBIDDEN_PATH_PARTS = ("holdout", "layoutlm")
_PROTECTED_NAMES = ("candidate", "silver", "reference", "protected", "source-map", "seed", "key.bin")


@dataclass(frozen=True)
class _Unit:
    page: int
    element_type: str
    content: str
    payload: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class _Task:
    task_id: str
    document_id: str
    page: int
    kind: Literal["presence", "content_structure", "order"]
    prompt: str
    left: object
    right: object
    evidence_id: str
    source_for_left: Literal["candidate", "reference"]
    source_for_right: Literal["candidate", "reference"]


def build_blind_v8(
    manifest_path: Path,
    candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    *,
    root_dir: Path,
    environment_kind: Literal["lock", "container"],
    environment_path: Path,
    seed: bytes | None = None,
) -> None:
    """Build fresh v8 material without consulting any earlier blind bundle."""
    _require_new_output(reviewer_dir)
    _require_new_output(protected_dir)
    root = root_dir.resolve(strict=True)
    environment = environment_path.resolve(strict=True)
    _require_within(environment, root, "declared environment")
    if not environment.is_file():
        raise FileNotFoundError(environment)
    if environment_kind == "lock" and environment.name != "uv.lock":
        raise ValueError("lock reproduction requires the supplied environment file to be named uv.lock")

    manifest_path = manifest_path.resolve(strict=True)
    candidate_dir = candidate_dir.resolve(strict=True)
    reference_dir = reference_dir.resolve(strict=True)
    bronze_dir = bronze_dir.resolve(strict=True)
    for path, label in (
        (manifest_path, "manifest"),
        (candidate_dir, "candidate directory"),
        (reference_dir, "reference directory"),
        (bronze_dir, "bronze directory"),
    ):
        _require_within(path, root, label)
        _reject_forbidden_path(path)

    manifest = BenchmarkBatchManifest.model_validate_json(manifest_path.read_bytes())
    secret = seed or secrets.token_bytes(32)
    if len(secret) < 32:
        raise ValueError("the protected HMAC seed must contain at least 32 bytes")

    tasks: list[_Task] = []
    evidence_sources: dict[str, tuple[Path, dict[str, object]]] = {}
    input_paths: list[tuple[str, Path]] = [("manifest.json", manifest_path), (environment.name, environment)]
    for selection in sorted(manifest.selections, key=lambda item: item.id):
        candidate_path = candidate_dir / f"{selection.id}.parquet"
        reference_path = reference_dir / f"{selection.id}.parquet"
        bundle_path = bronze_dir / selection.id
        for path in (candidate_path, reference_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        bronze = verify_bronze_bundle(bundle_path)
        if bronze.selection.requested_pages != selection.pages.requested:
            raise ValueError(f"bronze page selection mismatch for {selection.id}")
        input_paths.extend([
            (f"candidate/{selection.id}.parquet", candidate_path),
            (f"reference/{selection.id}.parquet", reference_path),
            (f"bronze/{selection.id}/manifest.json", bundle_path / "manifest.json"),
        ])
        candidate = read_document_elements(candidate_path)
        reference = read_document_elements(reference_path)
        tasks.extend(_comparison_tasks(selection.id, candidate, reference))
        for page in selection.pages.requested:
            evidence_id = _evidence_id(selection.id, page)
            image = bundle_path / "pages" / f"page-{page:04d}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            evidence_sources[evidence_id] = (
                image,
                {
                    "evidence_id": evidence_id,
                    "document_id": selection.id,
                    "page_number": page,
                    "source_name": bronze.source_name,
                    "source_path": bronze.source_path,
                    "source_sha256": bronze.source_sha256,
                    "bronze_manifest_sha256": _sha256(bundle_path / "manifest.json"),
                    "image_sha256": _sha256(image),
                },
            )
            input_paths.append((f"bronze/{selection.id}/pages/page-{page:04d}.png", image))

    if not tasks:
        raise ValueError("candidate/reference comparison produced no adjudicable alternatives")
    tasks.sort(key=lambda item: item.task_id)
    sample, design = _stratified_sample(tasks, secret)

    staging_parent = reviewer_dir.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    reviewer_stage = Path(tempfile.mkdtemp(prefix=f".{reviewer_dir.name}-", dir=staging_parent))
    protected_stage = Path(tempfile.mkdtemp(prefix=f".{protected_dir.name}-", dir=protected_dir.parent))
    completed = False
    try:
        packets_dir = reviewer_stage / "packets"
        evidence_dir = reviewer_stage / "evidence"
        packets_dir.mkdir()
        evidence_dir.mkdir()
        protected_map: list[dict[str, object]] = []
        sampled_evidence = {task.evidence_id for task in sample}
        for evidence_id in sorted(sampled_evidence):
            image, metadata = evidence_sources[evidence_id]
            target = evidence_dir / evidence_id
            target.mkdir()
            shutil.copyfile(image, target / "page.png")
            _write_json(target / "metadata.json", metadata)

        design_strata = cast(list[dict[str, object]], design["strata"])
        strata = {cast(str, item["stratum"]): item for item in design_strata}
        target_sample_size = cast(int, design["target_sample_size"])
        for task in sample:
            stratum = _design_stratum(task, target_sample_size)
            orientation = _orientation(task.task_id, secret)
            alternatives = [task.left, task.right]
            sources = [task.source_for_left, task.source_for_right]
            if orientation:
                alternatives.reverse()
                sources.reverse()
            stratum_design = strata[stratum]
            packet = {
                "schema_version": _SCHEMA_VERSION,
                "packet_id": task.task_id,
                "document_id": task.document_id,
                "page_number": task.page,
                "task_kind": task.kind,
                "sampling_stratum": stratum,
                "stratum_universe_size": stratum_design["universe_size"],
                "stratum_sample_size": stratum_design["sample_size"],
                "inclusion_probability": stratum_design["inclusion_probability"],
                "inclusion_weight": stratum_design["inclusion_weight"],
                "evidence_ids": [task.evidence_id],
                "prompt": task.prompt,
                "alternatives": {"A": alternatives[0], "B": alternatives[1]},
                "allowed_decisions": ["A", "B", "both_acceptable", "neither_acceptable"],
                "evidence_sufficiency": "pending_evidence_confirmation",
                "reviewer_decision": None,
                "reviewer_rationale": None,
            }
            _write_json(packets_dir / f"{task.task_id}.json", packet)
            protected_map.append({
                "packet_id": task.task_id,
                "A": sources[0],
                "B": sources[1],
                "orientation_hmac_sha256": hmac.new(
                    secret, task.task_id.encode(), hashlib.sha256
                ).hexdigest(),
            })

        _write_json(reviewer_stage / "sampling-design.json", design)
        (reviewer_stage / "PROTOCOL.md").write_text(_protocol_text(), encoding="utf-8")
        (reviewer_stage / "CHECKLIST.md").write_text(_checklist_text(), encoding="utf-8")
        (reviewer_stage / "verify_bundle.py").write_text(_verifier_source(), encoding="utf-8")
        os.chmod(reviewer_stage / "verify_bundle.py", 0o755)
        _write_integrity_manifest(reviewer_stage)

        os.chmod(protected_stage, 0o700)
        key_path = protected_stage / "hmac-key.bin"
        key_path.write_bytes(secret)
        os.chmod(key_path, 0o600)
        map_path = protected_stage / "protected-map.json"
        _write_json(map_path, {"schema_version": _SCHEMA_VERSION, "mappings": protected_map})
        os.chmod(map_path, 0o600)
        provenance = {
            "schema_version": _SCHEMA_VERSION,
            "root_dir": str(root),
            "root_dir_fingerprint": _fingerprint_tree(
                root,
                excluded=(
                    reviewer_dir,
                    protected_dir,
                    reviewer_stage,
                    protected_stage,
                    reviewer_dir.with_suffix(".tar.gz"),
                    reviewer_dir.with_suffix(".zip"),
                ),
            ),
            "root_dir_fingerprint_scope": {
                "statement": "Fingerprints the supplied root tree, excluding generated outputs, VCS/cache/environment internals, forbidden holdout/LayoutLM paths, and untouched blind v1-v7 material.",
                "excluded_names": [".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"],
                "excluded_prior_blind_versions": "v1-v7",
            },
            "inputs_fingerprint": _fingerprint_paths(input_paths),
            "declared_environment": {
                "kind": environment_kind,
                "path": environment.relative_to(root).as_posix(),
                "sha256": _sha256(environment),
                "statement": "Exact reproduction requires this declared lock/container plus a compatible captured platform; the lock alone does not guarantee bit-identical native behavior.",
            },
            "reproduction": {
                "command": "uv run python -m app.pdf2md.blind_v8 build MANIFEST CANDIDATE_DIR REFERENCE_DIR BRONZE_DIR REVIEWER_DIR PROTECTED_DIR --root-dir ROOT --environment-kind lock --environment-path ROOT/uv.lock --seed-file PROTECTED_DIR/hmac-key.bin",
                "requirements": "Use the full captured input tree matching root_dir_fingerprint and the declared lock/container. Output paths must not exist.",
            },
        }
        provenance_path = protected_stage / "provenance.json"
        _write_json(provenance_path, provenance)
        os.chmod(provenance_path, 0o600)

        verify_reviewer_bundle(reviewer_stage)
        _validate_adjudicability(reviewer_stage)
        _write_json(
            reviewer_stage / "DESIGNATION.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "designation": "blind-v8",
                "status": "designated",
                "gates": {
                    "protocol": "passed",
                    "package": "passed",
                    "integrity": "passed",
                    "adjudicability": "passed",
                },
            },
        )
        _write_integrity_manifest(reviewer_stage)
        verify_reviewer_bundle(reviewer_stage)
        _validate_adjudicability(reviewer_stage)
        os.replace(reviewer_stage, reviewer_dir)
        os.replace(protected_stage, protected_dir)
        _create_archives(reviewer_dir)
        completed = True
    finally:
        if not completed:
            shutil.rmtree(reviewer_stage, ignore_errors=True)
            shutil.rmtree(protected_stage, ignore_errors=True)


def verify_reviewer_bundle(bundle_dir: Path) -> None:
    bundle = bundle_dir.resolve(strict=True)
    manifest_path = bundle / "MANIFEST.sha256"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["path"]: item for item in records["files"]}
    actual = {
        path.relative_to(bundle).as_posix(): path
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    }
    if set(actual) != set(expected):
        raise ValueError("reviewer bundle file set differs from integrity manifest")
    for relative, path in actual.items():
        item = expected[relative]
        if path.stat().st_size != item["size_bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"reviewer bundle integrity failure: {relative}")
        lowered_parts = [part.casefold() for part in Path(relative).parts]
        if any(any(token in part for token in _PROTECTED_NAMES) for part in lowered_parts):
            raise ValueError(f"protected-role filename leaked into reviewer bundle: {relative}")
    for packet_path in (bundle / "packets").glob("*.json"):
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        forbidden_keys = {"candidate", "silver", "reference", "mapping", "seed", "hmac_key"}
        if forbidden_keys.intersection(_all_json_keys(packet)):
            raise ValueError(f"protected mapping key leaked into {packet_path.name}")
        if packet["evidence_sufficiency"] != "pending_evidence_confirmation":
            raise ValueError(f"evidence sufficiency was not left for human confirmation: {packet_path.name}")


def _comparison_tasks(
    document_id: str, candidate: list[DocumentElement], reference: list[DocumentElement]
) -> list[_Task]:
    candidate_pages = _units_by_page(candidate)
    reference_pages = _units_by_page(reference)
    tasks: list[_Task] = []
    for page in sorted(set(candidate_pages) | set(reference_pages)):
        left = candidate_pages.get(page, [])
        right = reference_pages.get(page, [])
        matcher = SequenceMatcher(
            a=[unit.digest for unit in left], b=[unit.digest for unit in right], autojunk=False
        )
        ordinal = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            left_units = left[i1:i2]
            right_units = right[j1:j2]
            paired = min(len(left_units), len(right_units)) if tag == "replace" else 0
            for offset in range(paired):
                ordinal += 1
                tasks.append(
                    _make_task(
                        document_id,
                        page,
                        ordinal,
                        "content_structure",
                        left_units[offset].payload,
                        right_units[offset].payload,
                    )
                )
            unpaired: list[tuple[_Unit, Literal["candidate", "reference"]]] = [
                *((item, "candidate") for item in left_units[paired:]),
                *((item, "reference") for item in right_units[paired:]),
            ]
            for unit, source in unpaired:
                ordinal += 1
                other: Literal["candidate", "reference"] = (
                    "reference" if source == "candidate" else "candidate"
                )
                tasks.append(
                    _make_task(
                        document_id,
                        page,
                        ordinal,
                        "presence",
                        {"action": "include", "unit": unit.payload},
                        {"action": "omit", "unit": unit.payload},
                        left_source=source,
                        right_source=other,
                    )
                )
        left_order = [unit.payload for unit in left]
        right_order = [unit.payload for unit in right]
        if (
            len(left_order) >= 2
            and len(right_order) >= 2
            and [unit.digest for unit in left] != [unit.digest for unit in right]
        ):
            ordinal += 1
            tasks.append(_make_task(document_id, page, ordinal, "order", left_order, right_order))
    return tasks


def _make_task(
    document_id: str,
    page: int,
    ordinal: int,
    kind: Literal["presence", "content_structure", "order"],
    left: object,
    right: object,
    *,
    left_source: Literal["candidate", "reference"] = "candidate",
    right_source: Literal["candidate", "reference"] = "reference",
) -> _Task:
    task_id = hashlib.sha256(f"v8|{document_id}|{page}|{ordinal}|{kind}".encode()).hexdigest()[:20]
    prompts = {
        "presence": "Using only the cited evidence, which neutral include/omit alternative is better supported?",
        "content_structure": "Using only the cited evidence, which neutral content/structure alternative is better supported?",
        "order": "Using only the cited evidence, which complete ordering alternative is better supported?",
    }
    return _Task(
        task_id,
        document_id,
        page,
        kind,
        prompts[kind],
        left,
        right,
        _evidence_id(document_id, page),
        left_source,
        right_source,
    )


def reviewer_alternative_payload(element: DocumentElement) -> dict[str, object]:
    """Project an element onto the fixed, source-neutral reviewer schema."""
    paragraph = element.structure.paragraph
    table = element.structure.table
    footnote = element.structure.footnote
    style_runs = sorted(element.structure.style_runs, key=lambda item: (item.start, item.end))
    table_cells = (
        sorted(
            table.cells,
            key=lambda item: (
                item.row_index,
                item.column_index,
                item.rowspan,
                item.colspan,
                item.role,
                item.text,
            ),
        )
        if table is not None
        else []
    )
    return {
        "element_type": element.element_type,
        "content": element.content,
        "format": element.format,
        "include_in_output": element.include_in_output,
        "structure": {
            "paragraph": (
                None
                if paragraph is None
                else {
                    "role": paragraph.role,
                    "heading_level": paragraph.heading_level,
                    "list_depth": paragraph.list_depth,
                    "list_label": paragraph.list_label,
                }
            ),
            "table": (
                None
                if table is None
                else {
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "header_row_count": table.header_row_count,
                    "representation": table.representation,
                    "cells": [
                        {
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "rowspan": cell.rowspan,
                            "colspan": cell.colspan,
                            "role": cell.role,
                            "text": cell.text,
                        }
                        for cell in table_cells
                    ],
                }
            ),
            "figure": {} if element.structure.figure is not None else None,
            "footnote": None if footnote is None else {"label": footnote.label},
            "style_runs": [
                {
                    "start": run.start,
                    "end": run.end,
                    "bold": run.bold,
                    "italic": run.italic,
                    "underline": run.underline,
                    "strikeout": run.strikeout,
                }
                for run in style_runs
            ],
        },
    }


def _units_by_page(elements: list[DocumentElement]) -> dict[int, list[_Unit]]:
    result: dict[int, list[_Unit]] = defaultdict(list)
    for element in elements:
        page = min(fragment.page_number for fragment in element.fragments)
        payload = reviewer_alternative_payload(element)
        digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
        result[page].append(_Unit(page, element.element_type, element.content, payload, digest))
    return result


def _stratified_sample(tasks: list[_Task], secret: bytes) -> tuple[list[_Task], dict[str, object]]:
    universe_size = len(tasks)
    lower = max(1, math.ceil(universe_size * 0.01))
    upper = max(lower, math.floor(universe_size * 0.02))
    target = min(upper, max(lower, round(universe_size * 0.015)))
    groups: dict[str, list[_Task]] = defaultdict(list)
    for task in tasks:
        groups[_design_stratum(task, target)].append(task)
    if len(groups) > target:
        raise ValueError("forced strata exceed the valid 1-2% sample size; enlarge the task universe")
    allocation = {name: 1 for name in groups}
    remaining = target - len(groups)
    ranked_strata = sorted(groups, key=lambda name: (-len(groups[name]), name))
    while remaining:
        eligible = [name for name in ranked_strata if allocation[name] < len(groups[name])]
        if not eligible:
            raise ValueError("cannot allocate requested sample size")
        name = max(eligible, key=lambda item: (len(groups[item]) / allocation[item], item))
        allocation[name] += 1
        remaining -= 1
    sample: list[_Task] = []
    strata: list[dict[str, object]] = []
    for name in sorted(groups):
        ranked = sorted(
            groups[name],
            key=lambda task: hmac.new(secret, f"sample|{task.task_id}".encode(), hashlib.sha256).digest(),
        )
        n = allocation[name]
        sample.extend(ranked[:n])
        probability = n / len(ranked)
        strata.append({
            "stratum": name,
            "universe_size": len(ranked),
            "sample_size": n,
            "inclusion_probability": probability,
            "inclusion_weight": 1.0 / probability,
        })
    return sorted(sample, key=lambda item: item.task_id), {
        "schema_version": _SCHEMA_VERSION,
        "method": "secret-HMAC-ranked forced-stratified sample without replacement",
        "universe_size": universe_size,
        "target_sample_size": target,
        "realized_fraction": target / universe_size,
        "allowed_fraction": "1-2% (integer rounding may bind for very small universes)",
        "strata": strata,
        "analysis_note": "Packet weights support design-based descriptive estimates. Inferential statistics are descriptive and underpowered at this sample size.",
    }


def _design_stratum(task: _Task, target: int) -> str:
    if target == 1:
        return "all_tasks"
    if target == 2:
        return "presence" if task.kind == "presence" else "alternative"
    return task.kind


def _orientation(task_id: str, secret: bytes) -> bool:
    return bool(hmac.new(secret, f"orientation|{task_id}".encode(), hashlib.sha256).digest()[0] & 1)


def _evidence_id(document_id: str, page: int) -> str:
    safe = "".join(character if character.isalnum() or character == "-" else "-" for character in document_id)
    return f"ev-{safe}-p{page:04d}"


def _validate_adjudicability(bundle: Path) -> None:
    packets = sorted((bundle / "packets").glob("*.json"))
    if not packets:
        raise ValueError("reviewer bundle has no packets")
    evidence_ids: set[str] = set()
    for path in packets:
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet["alternatives"]["A"] == packet["alternatives"]["B"]:
            raise ValueError(f"packet alternatives are not adjudicable: {path.name}")
        if packet["task_kind"] == "order" and not all(
            isinstance(packet["alternatives"][side], list) for side in ("A", "B")
        ):
            raise ValueError(f"order packet does not contain complete sequence alternatives: {path.name}")
        evidence_ids.update(packet["evidence_ids"])
    actual_evidence = {path.name for path in (bundle / "evidence").iterdir() if path.is_dir()}
    if evidence_ids != actual_evidence:
        raise ValueError("evidence is not exactly deduplicated by shared neutral evidence ID")


def _protocol_text() -> str:
    return """# Blind v8 reviewer protocol

## Explicit blindness and threat model

Blindness hides only candidate-versus-reference orientation and the protected A/B mappings. Source-document identity and bronze evidence are intentionally visible. The security assumption is that the reviewer has **no access** to candidate or silver/reference outputs, the protected map or HMAC seed, the comparison repository, or code containing mappings.

This package does not and cannot provide source-document anonymity. It does not claim resistance to source/content matching by a reviewer with corpus access. The preflight verifier cannot prevent a reviewer from independently obtaining comparison outputs; separate-account/environment delivery and access control are required.

## Review procedure

1. Run `python3 verify_bundle.py .` before opening packets.
2. Open each cited shared evidence ID once; duplicate same-page evidence is intentionally deduplicated.
3. For every packet, first change `evidence_sufficiency` from `pending_evidence_confirmation` to either `confirmed_sufficient` or `insufficient`. No automated evidence-pass exists.
4. If sufficient, select exactly one allowed decision: `A`, `B`, `both_acceptable`, or `neither_acceptable`, and record a rationale. If insufficient, leave the substantive decision null and explain what evidence is missing.
5. Do not seek candidate/silver outputs, protected mappings, the comparison repository, or mapping-bearing code.

Presence prompts are neutral include/omit tasks. Order prompts provide complete adjudicable alternatives, including `both_acceptable` and `neither_acceptable` outcomes through the allowed-decision field.

Sampling is a 1-2% secret-HMAC-ranked forced-stratified sample. Packet-level strata, inclusion probabilities, and inverse-probability weights are published for design-based descriptive analysis. Inferential statistics must be labeled descriptive and underpowered.
"""


def _checklist_text() -> str:
    return """# Reviewer checklist

- [ ] I am using a separate account/environment with no candidate or silver/reference outputs.
- [ ] I have no protected map/seed, comparison repository, or mapping-bearing code.
- [ ] `python3 verify_bundle.py .` passed before review.
- [ ] I understand source identity and bronze evidence are visible by design.
- [ ] I understand this protocol does not resist source/content matching under corpus access.
- [ ] For every packet, I explicitly resolved `pending_evidence_confirmation` before any substantive choice.
- [ ] Every sufficient packet has one allowed decision and a rationale.
- [ ] Every insufficient packet identifies missing evidence and has no forced A/B choice.
- [ ] Any weighted summary uses packet inclusion weights and is labeled descriptive/underpowered.
"""


def _verifier_source() -> str:
    return """#!/usr/bin/env python3
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
manifest = json.loads((root / "MANIFEST.sha256").read_text(encoding="utf-8"))
expected = {item["path"]: item for item in manifest["files"]}
actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"}
if set(actual) != set(expected):
    raise SystemExit("FAIL: file set differs from integrity manifest")
for rel, path in actual.items():
    item = expected[rel]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != item["size_bytes"] or digest != item["sha256"]:
        raise SystemExit(f"FAIL: integrity mismatch: {rel}")
    parts = [part.casefold() for part in pathlib.Path(rel).parts]
    forbidden = ("candidate", "silver", "reference", "protected", "source-map", "seed", "key.bin")
    if any(any(token in part for token in forbidden) for part in parts):
        raise SystemExit(f"FAIL: protected-role filename: {rel}")
for path in (root / "packets").glob("*.json"):
    packet = json.loads(path.read_text(encoding="utf-8"))
    keys = set()
    stack = [packet]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            keys.update(str(k).casefold() for k in value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    if keys & {"candidate", "silver", "reference", "mapping", "seed", "hmac_key"}:
        raise SystemExit(f"FAIL: protected mapping key in {path.name}")
    if packet.get("evidence_sufficiency") != "pending_evidence_confirmation":
        raise SystemExit(f"FAIL: non-pending evidence state in {path.name}")
print(f"PASS: {len(actual)} files verified; no candidate/silver/protected filenames or mapping keys found")
print("LIMIT: this verifier cannot prevent independent acquisition of comparison outputs")
"""


def _write_integrity_manifest(root: Path) -> None:
    files = [path for path in root.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256"]
    payload = {
        "algorithm": "sha256",
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        ],
    }
    _write_json(root / "MANIFEST.sha256", payload)


def _create_archives(reviewer_dir: Path) -> None:
    tar_path = reviewer_dir.with_suffix(".tar.gz")
    zip_path = reviewer_dir.with_suffix(".zip")
    for path in (tar_path, zip_path):
        if path.exists():
            raise FileExistsError(path)
    with tar_path.open("wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                archive.add(reviewer_dir, arcname=reviewer_dir.name, filter=_portable_tar_info)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(reviewer_dir.rglob("*")):
            if path.is_file():
                relative = f"{reviewer_dir.name}/{path.relative_to(reviewer_dir).as_posix()}"
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o755 if path.name == "verify_bundle.py" else 0o644) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _portable_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() or info.name.endswith("/verify_bundle.py") else 0o644
    return info


def _fingerprint_tree(root: Path, *, excluded: tuple[Path, ...]) -> dict[str, object]:
    excluded_resolved = tuple(path.resolve() for path in excluded)
    paths: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if any(path == item or path.is_relative_to(item) for item in excluded_resolved):
            continue
        if _excluded_root_path(path, root):
            continue
        if path.is_symlink():
            raise ValueError(f"root_dir fingerprint rejects symlinks: {path}")
        if path.is_file():
            paths.append((path.relative_to(root).as_posix(), path))
    return _fingerprint_paths(paths)


def _excluded_root_path(path: Path, root: Path) -> bool:
    parts = [part.casefold() for part in path.relative_to(root).parts]
    if {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}.intersection(parts):
        return True
    if any(any(token in part for token in _FORBIDDEN_PATH_PARTS) for part in parts):
        return True
    if any("blind-v8-reviewer" in part or "blind-v8-protected" in part for part in parts):
        return True
    return any(
        marker in part
        for part in parts
        for version in range(1, 8)
        for marker in (f"blind-v{version}", f"blind_v{version}")
    )


def _fingerprint_paths(paths: list[tuple[str, Path]]) -> dict[str, object]:
    if len({name for name, _ in paths}) != len(paths):
        raise ValueError("fingerprint logical paths must be unique")
    files = [
        {"path": name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in sorted(paths)
    ]
    return {
        "algorithm": "sha256-tree-v1",
        "root_sha256": hashlib.sha256(_canonical_json(files)).hexdigest(),
        "files": files,
    }


def _all_json_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key).casefold() for key in value} | set().union(
            *(_all_json_keys(item) for item in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_all_json_keys(item) for item in value), set())
    return set()


def _require_new_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)


def _require_within(path: Path, root: Path, label: str) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"{label} must be within supplied root_dir")


def _reject_forbidden_path(path: Path) -> None:
    if any(any(token in part.casefold() for token in _FORBIDDEN_PATH_PARTS) for part in path.parts):
        raise ValueError(f"forbidden holdout/LayoutLM path: {path}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify fresh blind-v8 reviewer material")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    for name in ("manifest", "candidate_dir", "reference_dir", "bronze_dir", "reviewer_dir", "protected_dir"):
        build.add_argument(name, type=Path)
    build.add_argument("--root-dir", required=True, type=Path)
    build.add_argument("--environment-kind", required=True, choices=("lock", "container"))
    build.add_argument("--environment-path", required=True, type=Path)
    build.add_argument("--seed-file", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify_reviewer_bundle(args.bundle_dir)
        print(f"verified reviewer bundle: {args.bundle_dir}")
        return
    protected_seed = args.seed_file.read_bytes() if args.seed_file else None
    build_blind_v8(
        args.manifest,
        args.candidate_dir,
        args.reference_dir,
        args.bronze_dir,
        args.reviewer_dir,
        args.protected_dir,
        root_dir=args.root_dir,
        environment_kind=args.environment_kind,
        environment_path=args.environment_path,
        seed=protected_seed,
    )
    print(f"built reviewer bundle: {args.reviewer_dir}")
    print(f"built protected material: {args.protected_dir}")


if __name__ == "__main__":
    _main()
