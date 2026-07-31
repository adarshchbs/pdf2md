from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import locale
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
import pymupdf

from app.pdf2md.pymupdf_runtime import open_document
from benchmarks.adapters import get_adapter
from benchmarks.canonical import CanonicalTable
from benchmarks.metrics import Metric
from benchmarks.metrics.efficiency import efficiency_metrics
from benchmarks.metrics.tables import table_metrics
from benchmarks.runner import (
    AttemptResult,
    attempts_are_deterministic,
    missing_artifacts,
    run_adapter_attempts,
    sha256_path,
)

_ATTEMPTS: Final = 3
_TIMEOUT_SECONDS: Final = 120.0
_TOOLS: Final = ("our_parser", "pymupdf_text")
_REQUIRED_TEXT: Final = (
    "Offline Extraction Smoke",
    "Café résumé",
    "1,234.50",
    "All names and values on this page are synthetic.",
)
_TABLE_METRIC_NAMES: Final = (
    "numeric_recall",
    "exact_numeric_cell_precision",
    "exact_numeric_cell_recall",
    "exact_numeric_cell_f1",
    "span_structure_precision",
    "span_structure_recall",
    "span_structure_f1",
    "row_exact",
    "column_exact",
)
_REFERENCE_CELLS: Final[list[dict[str, object]]] = [
    {"row": 0, "column": 0, "text": "Item"},
    {"row": 0, "column": 1, "text": "Units"},
    {"row": 0, "column": 2, "text": "Price"},
    {"row": 1, "column": 0, "text": "Alpha"},
    {"row": 1, "column": 1, "text": "12"},
    {"row": 1, "column": 2, "text": "3.50"},
    {"row": 2, "column": 0, "text": "Beta"},
    {"row": 2, "column": 1, "text": "7"},
    {"row": 2, "column": 2, "text": "8.25"},
]
_EXTERNAL_SKIPS: Final[list[dict[str, str]]] = [
    {
        "tool_id": "docling",
        "status": "SKIPPED_NOT_PINNED",
        "reason": "not locally pinned; model/license/resource and download gates are unresolved",
    },
    {
        "tool_id": "paddleocr",
        "status": "SKIPPED_NOT_PINNED",
        "reason": "pipeline and model weights are not pinned; download and accelerator gates are unresolved",
    },
    {
        "tool_id": "mineru",
        "status": "SKIPPED_NOT_PINNED",
        "reason": "release and model bundle are not pinned; network, license, and resource gates are unresolved",
    },
    {
        "tool_id": "marker",
        "status": "SKIPPED_NOT_PINNED",
        "reason": "package and model revisions are not pinned; credential and model gates are unresolved",
    },
    {
        "tool_id": "nougat",
        "status": "SKIPPED_NOT_PINNED",
        "reason": "compatible package and weights are not pinned; model and GPU gates are unresolved",
    },
    {
        "tool_id": "olmocr",
        "status": "SKIPPED_NOT_PINNED",
        "reason": "code and inference weights are not pinned; model, GPU, and network gates are unresolved",
    },
    {
        "tool_id": "adobe_pdf_extract",
        "status": "SKIPPED_CREDENTIALS_OR_PAYMENT",
        "reason": "external credentials, billing approval, and PDF transmission approval are absent",
    },
    {
        "tool_id": "google_document_ai",
        "status": "SKIPPED_CREDENTIALS_OR_PAYMENT",
        "reason": "external credentials, billing approval, and PDF transmission approval are absent",
    },
    {
        "tool_id": "aws_textract",
        "status": "SKIPPED_CREDENTIALS_OR_PAYMENT",
        "reason": "external credentials, billing approval, and PDF transmission approval are absent",
    },
    {
        "tool_id": "azure_document_intelligence",
        "status": "SKIPPED_CREDENTIALS_OR_PAYMENT",
        "reason": "external credentials, billing approval, and PDF transmission approval are absent",
    },
]


def create_synthetic_pdf(path: Path) -> None:
    """Create a one-page, born-digital fixture containing only invented content."""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_document() as document:
        page = document.new_page(width=612, height=792)
        page.insert_text((54, 68), "Offline Extraction Smoke", fontname="hebo", fontsize=18)
        page.insert_text(
            (54, 100),
            "Café résumé totals: 1,234.50 USD; growth 25%.",
            fontname="helv",
            fontsize=11,
        )
        page.insert_text(
            (54, 120),
            "All names and values on this page are synthetic.",
            fontname="helv",
            fontsize=10,
        )

        left, top = 54.0, 160.0
        widths = (210.0, 120.0, 120.0)
        row_height = 34.0
        rows = (("Item", "Units", "Price"), ("Alpha", "12", "3.50"), ("Beta", "7", "8.25"))
        x_positions = [left]
        for width in widths:
            x_positions.append(x_positions[-1] + width)
        for x in x_positions:
            page.draw_line((x, top), (x, top + row_height * len(rows)), color=(0, 0, 0), width=1)
        for row_index in range(len(rows) + 1):
            y = top + row_index * row_height
            page.draw_line((left, y), (x_positions[-1], y), color=(0, 0, 0), width=1)
        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                cell = pymupdf.Rect(
                    x_positions[column_index] + 6,
                    top + row_index * row_height + 4,
                    x_positions[column_index + 1] - 6,
                    top + (row_index + 1) * row_height - 4,
                )
                page.insert_textbox(
                    cell,
                    value,
                    fontname="hebo" if row_index == 0 else "helv",
                    fontsize=10,
                    align=pymupdf.TEXT_ALIGN_LEFT,
                )
        document.set_metadata({
            "title": "Offline Extraction Smoke",
            "author": "pdf2md development smoke harness",
            "subject": "Synthetic fixture; no public or third-party source material",
        })
        document.save(path, garbage=4, deflate=True)


def run_smoke(output: Path) -> tuple[dict[str, object], list[str]]:
    """Execute the credential-free development smoke run and write its evidence."""
    if output.exists():
        raise FileExistsError(f"smoke output destination already exists: {output}")
    output.mkdir(parents=True)
    input_path = output / "input" / "synthetic-smoke.pdf"
    create_synthetic_pdf(input_path)
    input_sha256 = sha256_path(input_path)

    results: dict[str, list[AttemptResult]] = {}
    for tool_id in _TOOLS:
        results[tool_id] = run_adapter_attempts(
            get_adapter(tool_id),
            input_path,
            output / "attempts" / tool_id,
            attempts=_ATTEMPTS,
            timeout_seconds=_TIMEOUT_SECONDS,
        )

    metric_rows = _metric_rows(results)
    metrics_path = output / "metrics.parquet"
    pl.DataFrame(
        metric_rows,
        schema={
            "tool_id": pl.String,
            "attempt": pl.Int64,
            "metric": pl.String,
            "value": pl.Float64,
            "supported": pl.Boolean,
            "n": pl.Int64,
            "eligible": pl.Int64,
            "failures": pl.Int64,
        },
    ).write_parquet(metrics_path)

    failures = acceptance_failures(results)
    summary = _build_summary(results, failures)
    summary_path = output / "benchmark_summary.json"
    _write_json(summary_path, summary)

    manifest = _build_manifest(output, input_path, input_sha256, results, failures)
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    report_path = output / "benchmark_report.md"
    report_path.write_text(_report(summary, manifest), encoding="utf-8", newline="\n")

    required = [manifest_path, summary_path, metrics_path, report_path, input_path]
    absent = [path for path in required if not path.is_file()]
    absent.extend(path for tool_results in results.values() for path in missing_artifacts(tool_results))
    if absent:
        failures.append("missing artifacts: " + ", ".join(str(path) for path in absent))
        summary = _build_summary(results, failures)
        _write_json(summary_path, summary)
        manifest["acceptance"] = {"passed": False, "failures": failures}
        _write_json(manifest_path, manifest)
        report_path.write_text(_report(summary, manifest), encoding="utf-8", newline="\n")
    return manifest, failures


def _metric_rows(results: Mapping[str, Sequence[AttemptResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tool_id, attempts in results.items():
        operational = efficiency_metrics([
            {
                "case_id": "synthetic-smoke",
                "status": item.status,
                **(
                    {"output_digest": item.normalized_sha256}
                    if item.status == "success" and item.normalized_sha256 is not None
                    else {}
                ),
            }
            for item in attempts
        ])
        rows.extend(_metrics_to_rows(tool_id, 0, operational))

        for item in attempts:
            if item.status != "success":
                rows.extend(
                    _metrics_to_rows(
                        tool_id,
                        item.attempt,
                        {
                            name: Metric(
                                value=None,
                                supported=tool_id == "our_parser",
                                n=1,
                                eligible=0,
                                failures=1,
                            )
                            for name in _TABLE_METRIC_NAMES
                        },
                    )
                )
                continue
            if tool_id != "our_parser":
                rows.extend(
                    _metrics_to_rows(
                        tool_id,
                        item.attempt,
                        {name: Metric.unsupported(n=1) for name in _TABLE_METRIC_NAMES},
                    )
                )
                continue
            rows.extend(
                _metrics_to_rows(
                    tool_id,
                    item.attempt,
                    table_metrics(_candidate_cells(item), _REFERENCE_CELLS),
                )
            )
    return rows


def _metrics_to_rows(
    tool_id: str,
    attempt: int,
    metrics: Mapping[str, Metric],
) -> list[dict[str, object]]:
    return [_metric_row(tool_id, attempt, name, metric) for name, metric in metrics.items()]


def _metric_row(tool_id: str, attempt: int, name: str, metric: Metric) -> dict[str, object]:
    return {
        "tool_id": tool_id,
        "attempt": attempt,
        "metric": name,
        "value": metric.value,
        "supported": metric.supported,
        "n": metric.n,
        "eligible": metric.eligible,
        "failures": metric.failures,
    }


def _logical_table_signature(table: CanonicalTable) -> tuple[object, ...]:
    return (
        table.id,
        table.row_count,
        table.column_count,
        table.markdown,
        table.html,
        tuple(
            (cell.id, cell.row_index, cell.column_index, cell.rowspan, cell.colspan, cell.text)
            for cell in table.cells
        ),
    )


def _unique_tables(attempt: AttemptResult) -> dict[str, CanonicalTable]:
    tables: dict[str, CanonicalTable] = {}
    for page in attempt.pages:
        for table in page.tables:
            previous = tables.setdefault(table.id, table)
            if _logical_table_signature(previous) != _logical_table_signature(table):
                raise ValueError(f"logical table {table.id!r} differs across page projections")
    return tables


def _candidate_cells(attempt: AttemptResult) -> list[dict[str, object]]:
    return [
        {
            "row": cell.row_index,
            "column": cell.column_index,
            "text": cell.text,
            "rowspan": cell.rowspan,
            "colspan": cell.colspan,
        }
        for table in _unique_tables(attempt).values()
        for cell in table.cells
    ]


def acceptance_failures(results: Mapping[str, Sequence[AttemptResult]]) -> list[str]:
    failures: list[str] = []
    for tool_id, attempts in results.items():
        for attempt in attempts:
            if attempt.status != "success":
                failures.append(f"{tool_id} attempt {attempt.attempt} was {attempt.status}: {attempt.error}")
        if not attempts_are_deterministic(attempts):
            failures.append(f"{tool_id} normalized output was not deterministic across all attempts")
    for tool_id, attempts in results.items():
        for attempt in attempts:
            if attempt.status != "success":
                continue
            markdown = "\n".join(page.markdown for page in attempt.pages)
            missing_text = [value for value in _REQUIRED_TEXT if value not in markdown]
            if missing_text:
                failures.append(
                    f"{tool_id} attempt {attempt.attempt} omitted required synthetic text: {missing_text}"
                )

    parser_tables = [item.table_count for item in results["our_parser"]]
    if any(count == 0 for count in parser_tables):
        failures.append("our_parser did not emit a structured table on every attempt")
    for attempt in results["our_parser"]:
        if attempt.status != "success":
            continue
        unique_tables = _unique_tables(attempt)
        if len(unique_tables) != 1:
            failures.append(
                f"our_parser attempt {attempt.attempt} emitted {len(unique_tables)} logical tables; expected 1"
            )
            continue
        metrics = table_metrics(_candidate_cells(attempt), _REFERENCE_CELLS)
        required_exact = ("numeric_recall", "exact_numeric_cell_f1", "row_exact", "column_exact")
        failed_metrics = [name for name in required_exact if metrics[name].value != 1.0]
        if failed_metrics:
            failures.append(
                f"our_parser attempt {attempt.attempt} failed synthetic table checks: {failed_metrics}"
            )

    baseline_tables = [item.table_count for item in results["pymupdf_text"]]
    if any(count != 0 for count in baseline_tables):
        failures.append("pymupdf_text baseline claimed structured table output")
    return failures


def _build_summary(
    results: Mapping[str, Sequence[AttemptResult]],
    failures: Sequence[str],
) -> dict[str, object]:
    tools: dict[str, object] = {}
    for tool_id, attempts in results.items():
        counts = Counter(item.status for item in attempts)
        tools[tool_id] = {
            "attempt_count": len(attempts),
            "status_counts": dict(sorted(counts.items())),
            "successful_attempt_count": counts["success"],
            "failed_or_invalid_attempt_count": len(attempts) - counts["success"],
            "all_attempts_in_denominator": True,
            "normalized_deterministic": attempts_are_deterministic(attempts),
            "table_counts": [item.table_count for item in attempts],
        }
    return {
        "track": "development",
        "reproduction_status": "NOT_COMPARABLE",
        "same_harness": False,
        "passed": not failures,
        "failures": list(failures),
        "tools": tools,
        "external_tools": _EXTERNAL_SKIPS,
    }


def _build_manifest(
    output: Path,
    input_path: Path,
    input_sha256: str,
    results: Mapping[str, Sequence[AttemptResult]],
    failures: Sequence[str],
) -> dict[str, object]:
    repo = Path(__file__).resolve().parents[1]
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    policy = {
        "attempts": _ATTEMPTS,
        "timeout_seconds_per_attempt": _TIMEOUT_SECONDS,
        "timeout_mechanism": "POSIX SIGALRM in the main thread; not a process sandbox",
        "retries": 0,
        "rerun_failed_attempts_only": False,
        "failed_and_invalid_attempts_in_denominator": True,
    }
    package_names = ("pdf2md", "pymupdf", "polars", "pydantic")
    package_versions = {name: importlib.metadata.version(name) for name in package_names}
    config_hashes: dict[str, str | None] = {}
    for tool_id, attempts in results.items():
        hashes = {page.provenance.config_sha256 for attempt in attempts for page in attempt.pages}
        config_hashes[tool_id] = next(iter(hashes)) if len(hashes) == 1 else None
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "track": "development",
        "reproduction_status": "NOT_COMPARABLE",
        "same_harness": False,
        "git": {"sha": git_sha, "dirty": git_dirty},
        "input": {
            "path": str(input_path.relative_to(output)),
            "sha256": input_sha256,
            "size_bytes": input_path.stat().st_size,
            "source": "runtime-generated legally safe synthetic PDF",
            "contains": ["heading", "Unicode and numeric text", "ruled simple table"],
        },
        "versions": {
            "python": platform.python_version(),
            "packages": package_versions,
            "uv_lock_sha256": sha256_path(repo / "uv.lock"),
        },
        "environment": {
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "locale": locale.getlocale(),
        },
        "hardware": {
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _physical_memory_bytes(),
            "gpu_required": False,
        },
        "commands": [
            {
                "purpose": "harness",
                "argv": ["uv", "run", "python", "-m", "benchmarks.smoke", "--output", str(output)],
            },
            *[
                {
                    "purpose": f"adapter_attempt_{attempt.attempt}",
                    "tool_id": tool_id,
                    "execution_mode": "in_process_python_api",
                    "callable": "benchmarks.adapters.get_adapter(tool_id).process",
                    "arguments": {"pdf_path": str(input_path)},
                }
                for tool_id, attempts in results.items()
                for attempt in attempts
            ],
        ],
        "config_hashes": {
            "harness_policy": _json_sha256(policy),
            "adapters": config_hashes,
        },
        "policy": policy,
        "isolation": {
            "adapter_execution": "in_process",
            "network_enforcement": "not_enforced",
            "credential_access_enforcement": "not_enforced",
            "memory_limit_enforcement": "not_enforced",
            "disk_limit_enforcement": "not_enforced",
            "leakage_isolation": "not_enforced",
        },
        "network": {"policy": "denied", "use_observed": "not_verified"},
        "credentials": {"required_by_configured_tools": False, "access_observed": "not_verified"},
        "models": {
            "configured": False,
            "downloads_policy": "denied",
            "downloads_observed": "not_verified",
        },
        "attempts": {
            tool_id: [attempt.manifest_record(output) for attempt in attempts]
            for tool_id, attempts in results.items()
        },
        "external_tool_skips": _EXTERNAL_SKIPS,
        "acceptance": {"passed": not failures, "failures": list(failures)},
    }


def _report(summary: Mapping[str, object], manifest: Mapping[str, object]) -> str:
    tools = summary["tools"]
    if not isinstance(tools, dict):
        raise TypeError("summary tools must be a mapping")
    lines = [
        "# Offline development smoke report",
        "",
        "This is synthetic development evidence, not a corpus quality benchmark or an external-tool ranking.",
        "",
        f"- Result: {'PASS' if summary['passed'] else 'FAIL'}",
        "- Track: development",
        "- Comparison status: NOT_COMPARABLE (synthetic harness validation only)",
        f"- Input SHA-256: `{manifest['input']['sha256']}`",  # type: ignore[index]
        f"- Attempts per local tool: {_ATTEMPTS} (every failure remains in the denominator)",
        f"- Timeout: {_TIMEOUT_SECONDS:g}s per attempt; retries: 0",
        "- Network, credential access, models, and downloads: not sandbox-enforced; no external tools configured",
        "",
        "## Local results",
        "",
        "| Tool | Successful / attempted | Normalized deterministic | Tables by attempt |",
        "| --- | ---: | --- | --- |",
    ]
    for tool_id in _TOOLS:
        tool = tools[tool_id]
        if not isinstance(tool, dict):
            raise TypeError("tool summary must be a mapping")
        lines.append(
            f"| {tool_id} | {tool['successful_attempt_count']} / {tool['attempt_count']} | "
            f"{tool['normalized_deterministic']} | {tool['table_counts']} |"
        )
    lines.extend(["", "## External tools", ""])
    lines.extend(f"- `{item['tool_id']}`: {item['status']} — {item['reason']}" for item in _EXTERNAL_SKIPS)
    failures = summary["failures"]
    if failures:
        lines.extend(["", "## Acceptance failures", ""])
        lines.extend(f"- {failure}" for failure in failures)  # type: ignore[union-attr]
    lines.append("")
    return "\n".join(lines)


def _physical_memory_bytes() -> int | None:
    if not hasattr(os, "sysconf"):
        return None
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return page_size * page_count


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline synthetic development smoke benchmark")
    parser.add_argument("--output", type=Path, required=True, help="new destination directory")
    arguments = parser.parse_args(argv)
    _, failures = run_smoke(arguments.output.resolve())
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"PASS: smoke artifacts written to {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
