from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from benchmarks.adapters.olmocr_bench import export_olmocr_bench
from benchmarks.adapters.our_parser import OurParserAdapter


class Outcome(TypedDict):
    pdf: str
    status: str
    pages: list[int]
    latency_seconds: float
    error: str | None


def _load_requests(bench_dir: Path) -> dict[Path, set[int]]:
    requests: dict[Path, set[int]] = {}
    for jsonl_path in sorted(bench_dir.glob("*.jsonl")):
        for line_number, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            record = json.loads(line)
            relative_pdf = Path(record["pdf"])
            page = record["page"]
            if relative_pdf.is_absolute() or ".." in relative_pdf.parts:
                raise ValueError(f"invalid PDF path at {jsonl_path}:{line_number}: {relative_pdf}")
            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise ValueError(f"invalid page at {jsonl_path}:{line_number}: {page!r}")
            requests.setdefault(relative_pdf, set()).add(page)
    if not requests:
        raise ValueError(f"no olmOCR test requests found below {bench_dir}")
    return requests


def _empty_candidate_path(output_dir: Path, relative_pdf: Path, page: int) -> Path:
    return output_dir / relative_pdf.parent / f"{relative_pdf.stem}_pg{page}_repeat1.md"


def _process_pdf(
    bench_dir: Path,
    output_dir: Path,
    relative_pdf: Path,
    requested_pages: set[int],
) -> Outcome:
    started = time.perf_counter()
    run = OurParserAdapter().process(bench_dir / "pdfs" / relative_pdf)
    if run.status == "success":
        page_keys = {(page.document_id, page.page_index) for page in run.pages}
        mapping = {
            key: relative_pdf
            for key in page_keys
            if key[1] + 1 in requested_pages
        }
        selected_pages = [page for page in run.pages if page.page_index + 1 in requested_pages]
        missing = requested_pages.difference(page.page_index + 1 for page in selected_pages)
        if missing:
            raise ValueError(f"parser omitted requested pages: {sorted(missing)}")
        export_olmocr_bench(selected_pages, output_dir, file_mapping=mapping)
        status = "success"
        error = None
    else:
        for page in sorted(requested_pages):
            target = _empty_candidate_path(output_dir, relative_pdf, page)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("", encoding="utf-8", newline="\n")
        status = "failed"
        error = run.error
    return {
        "pdf": relative_pdf.as_posix(),
        "status": status,
        "pages": sorted(requested_pages),
        "latency_seconds": time.perf_counter() - started,
        "error": error,
    }


def generate_candidates(bench_dir: Path, output_dir: Path, *, workers: int) -> list[Outcome]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if workers < 1:
        raise ValueError("workers must be positive")
    requests = _load_requests(bench_dir)
    output_dir.mkdir(parents=True)
    outcomes: list[Outcome] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_pdf, bench_dir, output_dir, relative_pdf, pages): relative_pdf
            for relative_pdf, pages in requests.items()
        }
        for future in as_completed(futures):
            relative_pdf = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as error:
                for page in sorted(requests[relative_pdf]):
                    target = _empty_candidate_path(output_dir, relative_pdf, page)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("", encoding="utf-8", newline="\n")
                outcomes.append({
                    "pdf": relative_pdf.as_posix(),
                    "status": "failed",
                    "pages": sorted(requests[relative_pdf]),
                    "latency_seconds": 0.0,
                    "error": f"{type(error).__name__}: {error}",
                })
    outcomes.sort(key=lambda item: item["pdf"])
    (output_dir / "_outcomes.json").write_text(
        json.dumps(outcomes, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    outcomes = generate_candidates(args.bench_dir, args.output_dir, workers=args.workers)
    failures = sum(outcome["status"] != "success" for outcome in outcomes)
    print(json.dumps({"total": len(outcomes), "successful": len(outcomes) - failures, "failed": failures}))


if __name__ == "__main__":
    main()
