from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from benchmarks.digital_pdf_slice import DigitalSliceConfig, classify_pdf, parsebench_test_id

EXCLUDED_TASK_TAGS = frozenset({"ocr", "handwritting"})


@dataclass(frozen=True)
class DatasetRecord:
    test_id: str
    source_path: Path
    categories: tuple[str, ...]
    tags: tuple[str, ...]


def score_digital_slice(
    evaluator_dir: Path,
    dataset_dir: Path,
    candidates: dict[str, Path],
    output_dir: Path,
    *,
    config: DigitalSliceConfig | None = None,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not candidates:
        raise ValueError("at least one candidate is required")
    config = config or DigitalSliceConfig()
    dataset_root = dataset_dir.resolve()
    records = load_dataset_records(dataset_root)

    classifications: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    exclusion_counts: Counter[str] = Counter()
    for record in records:
        classification = classify_pdf(record.source_path, config)
        excluded_tags = sorted(EXCLUDED_TASK_TAGS.intersection(record.tags))
        reason = "excluded_task_tag" if excluded_tags else classification.reason
        eligible = classification.eligible and not excluded_tags
        if eligible:
            selected_ids.add(record.test_id)
        else:
            exclusion_counts[reason] += 1
        classifications.append({
            "test_id": record.test_id,
            "source_path": str(record.source_path.relative_to(dataset_root)),
            "categories": list(record.categories),
            "tags": list(record.tags),
            "eligible": eligible,
            "reason": reason,
            "excluded_tags": excluded_tags,
            "page_count": classification.page_count,
            "scanned_page_count": classification.scanned_page_count,
            "pages": [asdict(page) for page in classification.pages],
        })

    evaluator_src = str((evaluator_dir / "src").resolve())
    sys.path.insert(0, evaluator_src)
    try:
        from parse_bench.evaluation.runner import (  # pyright: ignore[reportMissingImports]
            EvaluationRunner,
            _is_infra_failure,
            _is_skipped_result,
        )
        from parse_bench.schemas.evaluation import (  # pyright: ignore[reportMissingImports]
            EvaluationSummary,
        )
    finally:
        sys.path.remove(evaluator_src)

    aggregation_runner = EvaluationRunner.__new__(EvaluationRunner)
    candidate_results: dict[str, object] = {}
    selected_id_list = sorted(selected_ids)
    selected_records = {record.test_id: record for record in records if record.test_id in selected_ids}
    selected_categories = sorted({category for record in selected_records.values() for category in record.categories})

    def aggregate_results(results: list[Any]) -> dict[str, object]:
        aggregate = aggregation_runner._aggregate_metrics(results)
        skipped = sum(_is_skipped_result(result) for result in results)
        infrastructure_failures = sum(_is_infra_failure(result) for result in results)
        successful = sum(result.success for result in results)
        return {
            "total": len(results),
            "successful": successful,
            "failed": len(results) - successful - skipped - infrastructure_failures,
            "skipped": skipped,
            "infrastructure_failures": infrastructure_failures,
            "failure_details": [
                {
                    "test_id": result.test_id,
                    "error": result.error,
                    "skipped": _is_skipped_result(result),
                    "infrastructure_failure": _is_infra_failure(result),
                }
                for result in results
                if not result.success
            ],
            "aggregate_metrics": aggregate,
        }

    for name, candidate_dir in sorted(candidates.items()):
        report_path = candidate_dir / "_evaluation_report.json"
        report = EvaluationSummary.model_validate_json(report_path.read_text(encoding="utf-8"))
        by_id = {result.test_id: result for result in report.per_example_results}
        missing = selected_ids.difference(by_id)
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} selected evaluation results")
        filtered = [by_id[test_id] for test_id in selected_id_list]
        result = aggregate_results(filtered)
        result["categories"] = {
            category: aggregate_results([
                by_id[test_id]
                for test_id in selected_id_list
                if category in selected_records[test_id].categories
            ])
            for category in selected_categories
        }
        result["source_report"] = str(report_path)
        candidate_results[name] = result

    summary: dict[str, object] = {
        "result_label": "DIAGNOSTIC_NOT_COMPARABLE",
        "method": {
            "reference_independent": True,
            "config": asdict(config),
            "excluded_task_tags": sorted(EXCLUDED_TASK_TAGS),
            "classifier_sha256": _sha256(Path(__file__).parents[2] / "digital_pdf_slice.py"),
            "scorer_sha256": _sha256(Path(__file__)),
            "dataset_jsonl_sha256": _bundle_sha256(sorted(dataset_root.glob("*.jsonl"))),
            "evaluator_revision": _git_revision(evaluator_dir),
        },
        "slice": {
            "total_test_ids": len(records),
            "selected_test_ids": len(selected_ids),
            "excluded_test_ids": len(records) - len(selected_ids),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
        },
        "candidates": candidate_results,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "classifications.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in classifications),
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def load_dataset_records(dataset_dir: Path) -> list[DatasetRecord]:
    grouped: dict[str, dict[str, object]] = {}
    jsonl_paths = sorted(dataset_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise ValueError(f"no ParseBench JSONL files found below {dataset_dir}")
    for jsonl_path in jsonl_paths:
        category = jsonl_path.stem
        for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            relative_path = Path(row["pdf"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe PDF path in {jsonl_path}:{line_number}: {relative_path}")
            source_path = (dataset_dir / relative_path).resolve()
            if not source_path.is_relative_to(dataset_dir.resolve()):
                raise ValueError(f"PDF path escapes dataset root: {relative_path}")
            test_id = parsebench_test_id(category, relative_path)
            entry = grouped.setdefault(test_id, {
                "source_path": source_path,
                "categories": set(),
                "tags": set(),
            })
            if entry["source_path"] != source_path:
                raise ValueError(f"test ID maps to multiple source paths: {test_id}")
            cast(set[str], entry["categories"]).add(category)
            cast(set[str], entry["tags"]).update(row.get("tags") or [])
    return [
        DatasetRecord(
            test_id=test_id,
            source_path=cast(Path, entry["source_path"]),
            categories=tuple(sorted(cast(set[str], entry["categories"]))),
            tags=tuple(sorted(cast(set[str], entry["tags"]))),
        )
        for test_id, entry in sorted(grouped.items())
    ]


def _parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must use NAME=PATH")
    return name, Path(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_sha256(paths: list[Path]) -> str:
    if not paths:
        raise ValueError("cannot hash an empty file bundle")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluator_dir", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, required=True)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        raise ValueError("candidate names must be unique")
    summary = score_digital_slice(
        args.evaluator_dir,
        args.dataset_dir,
        candidates,
        args.output_dir,
    )
    print(json.dumps(summary["slice"], sort_keys=True))


if __name__ == "__main__":
    main()
