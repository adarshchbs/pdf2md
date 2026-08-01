from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import TypedDict, cast

import polars as pl


class ScoreRecord(TypedDict):
    sample_id: str
    score: float
    status: str
    error: str | None


def score_predictions(
    scorer_dir: Path,
    dataset_dir: Path,
    predictions_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    sys.path.insert(0, str(scorer_dir.resolve()))
    from convert import html_to_numpy  # pyright: ignore[reportMissingImports]
    from grading import table_similarity  # pyright: ignore[reportMissingImports]

    groundtruth_paths = sorted((dataset_dir / "groundtruth").glob("*.html"))
    if not groundtruth_paths:
        raise ValueError(f"no RD-TableBench ground truth found below {dataset_dir}")
    records: list[ScoreRecord] = []
    for groundtruth_path in groundtruth_paths:
        prediction_path = predictions_dir / groundtruth_path.name
        if not prediction_path.is_file():
            records.append({
                "sample_id": groundtruth_path.stem,
                "score": 0.0,
                "status": "missing",
                "error": "prediction file missing",
            })
            continue
        try:
            groundtruth = html_to_numpy(groundtruth_path.read_text(encoding="utf-8"))
            prediction = html_to_numpy(prediction_path.read_text(encoding="utf-8"))
            if groundtruth.size == 0 or prediction.size == 0:
                raise ValueError("empty parsed table")
            score = float(table_similarity(groundtruth, prediction))
            records.append({
                "sample_id": groundtruth_path.stem,
                "score": score,
                "status": "scored",
                "error": None,
            })
        except (TypeError, ValueError, IndexError) as error:
            records.append({
                "sample_id": groundtruth_path.stem,
                "score": 0.0,
                "status": "invalid",
                "error": f"{type(error).__name__}: {error}",
            })
    scores = [record["score"] for record in records]
    summary: dict[str, object] = {
        "metric": "official_rd_tablebench_mean_similarity",
        "samples": len(records),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "scored": sum(record["status"] == "scored" for record in records),
        "missing": sum(record["status"] == "missing" for record in records),
        "invalid": sum(record["status"] == "invalid" for record in records),
        "failures_scored_as_zero": True,
    }
    output_dir.mkdir(parents=True)
    typed_records = cast(list[dict[str, object]], records)
    pl.DataFrame(typed_records).write_parquet(output_dir / "scores.parquet")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scorer_dir", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("predictions_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(score_predictions(args.scorer_dir, args.dataset_dir, args.predictions_dir, args.output_dir)))


if __name__ == "__main__":
    main()
