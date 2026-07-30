from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from benchmarks.adapters.our_parser import OurParserAdapter
from benchmarks.adapters.rd_tablebench import render_rd_tablebench_html


class Outcome(TypedDict):
    sample_id: str
    status: str
    detected_tables: int
    latency_seconds: float
    error: str | None


def _process_pdf(pdf_path: Path, output_dir: Path) -> Outcome:
    started = time.perf_counter()
    run = OurParserAdapter().process(pdf_path)
    unique_tables = {}
    if run.status == "success":
        for page in run.pages:
            for table in page.tables:
                unique_tables.setdefault(table.id, table)
        rendered = [render_rd_tablebench_html(table) for table in unique_tables.values()]
        prediction = max(rendered, key=len) if rendered else "<table></table>"
        status = "success" if rendered else "empty"
        error = None
    else:
        prediction = "<table></table>"
        status = "failed"
        error = run.error
    (output_dir / f"{pdf_path.stem}.html").write_text(
        prediction,
        encoding="utf-8",
        newline="\n",
    )
    return {
        "sample_id": pdf_path.stem,
        "status": status,
        "detected_tables": len(unique_tables),
        "latency_seconds": time.perf_counter() - started,
        "error": error,
    }


def generate_candidates(dataset_dir: Path, output_dir: Path, *, workers: int) -> list[Outcome]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if workers < 1:
        raise ValueError("workers must be positive")
    pdf_paths = sorted((dataset_dir / "pdfs").glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"no RD-TableBench PDFs found below {dataset_dir}")
    output_dir.mkdir(parents=True)
    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_pdf, path, output_dir): path for path in pdf_paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as error:
                (output_dir / f"{path.stem}.html").write_text(
                    "<table></table>",
                    encoding="utf-8",
                    newline="\n",
                )
                outcomes.append({
                    "sample_id": path.stem,
                    "status": "failed",
                    "detected_tables": 0,
                    "latency_seconds": 0.0,
                    "error": f"{type(error).__name__}: {error}",
                })
    outcomes.sort(key=lambda item: item["sample_id"])
    (output_dir / "_outcomes.json").write_text(
        json.dumps(outcomes, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    outcomes = generate_candidates(args.dataset_dir, args.output_dir, workers=args.workers)
    counts = {status: sum(item["status"] == status for item in outcomes) for status in ("success", "empty", "failed")}
    print(json.dumps({"total": len(outcomes), **counts}))


if __name__ == "__main__":
    main()
