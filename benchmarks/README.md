# PDF extraction benchmark checkpoint

This directory contains an **offline development benchmark scaffold**. It includes a canonical page schema, local adapters, product metrics, public-benchmark export boundaries, and a synthetic smoke harness. It does not contain a public benchmark corpus, downloaded models, committed raw outputs, or a frozen quality score. Smoke results validate the harness only and must not be quoted as parser-accuracy evidence.

## Scope

The intended task is born-digital PDF extraction, with document Markdown/text and structured table output evaluated separately. Scanned and image-only PDFs are out of scope for the current parser. The benchmark definition is in `config/benchmarks.yaml`; candidate integrations and their readiness are in `config/tools.yaml`.

Only `our_parser` and the locked PyMuPDF native-text baseline are `READY`. `READY` means that the implementation is already available in this repository and can be invoked without acquiring a new credential, paid service, model, or GPU. It does **not** mean the tool has been benchmarked. Every other entry is `NOT_READY` and has an exact skip reason.

## One-command smoke run

Run the credential-free synthetic comparison from the repository root:

```bash
uv run python -m benchmarks.smoke --output benchmarks/runs/smoke-local
```

The destination must not exist. The command generates its PDF fixture at runtime, runs both local adapters three times on identical bytes, checks normalized determinism, and retains raw/canonical JSONL, a manifest, Polars metrics, a machine-readable summary, and a Markdown report. `benchmarks/scripts/smoke.sh` provides the same entry point with an optional destination argument.

The smoke PDF is independent synthetic development material. It is not copied from ParseBench, OmniDocBench, olmOCR-Bench, RD-TableBench, or the repository corpus.

## Current invocations

Run from the repository root after `uv sync --all-groups`.

Full parser, all pages:

```bash
uv run pdf2md extract INPUT.pdf benchmarks/runs/our_parser/DOCUMENT/output.md \
  --parquet-path benchmarks/runs/our_parser/DOCUMENT/elements.parquet
```

Selected pages are one-based and must be repeated:

```bash
uv run pdf2md extract INPUT.pdf benchmarks/runs/our_parser/DOCUMENT/output.md \
  --parquet-path benchmarks/runs/our_parser/DOCUMENT/elements.parquet \
  --page 7 --page 8
```

The locked PyMuPDF text baseline is the following direct native-text call. It intentionally has no OCR, layout adapter, table structure, or bounding boxes:

```bash
uv run python -c 'from pathlib import Path; import pymupdf,sys; source=Path(sys.argv[1]); target=Path(sys.argv[2]); pages=[int(value) for value in sys.argv[3:]]; document=pymupdf.open(source); selected=pages or list(range(1, document.page_count + 1)); invalid=[page for page in selected if page < 1 or page > document.page_count]; invalid and (_ for _ in ()).throw(ValueError(f"pages outside the document range: {invalid}")); target.parent.mkdir(parents=True, exist_ok=True); target.write_text("\n\f\n".join(document[page - 1].get_text("text", sort=True) for page in selected), encoding="utf-8")' INPUT.pdf benchmarks/runs/pymupdf_text/DOCUMENT/output.txt [PAGE ...]
```

The optional trailing pages are one-based. The version is resolved from this repository's `uv.lock`; do not run the baseline from an unrelated environment.

## Adapter boundary

A tool adapter ends at the tool's raw, immutable output. A separate normalization layer must convert that output into benchmark records; it must not repair text, infer tables absent from the raw output, use references, or call another extraction model. Preserve raw output under `benchmarks/raw/` and normalized run artifacts under `benchmarks/runs/` (both ignored).

The normalized boundary is schema-versioned JSONL with one `CanonicalPage` record per source page. Each record carries document/page identity, page geometry, page Markdown, ordered elements, structured tables and cells, figures, runtime, and provenance. Bounding boxes are nullable: unsupported geometry remains null and is never synthesized from a reference or another parser.

The smoke harness uses the typed in-process product boundary `extract_document_with_catalog()` so it can test the adapter seam without temporary product artifacts. Frozen public comparisons should invoke the public CLI as a subprocess and normalize its schema-versioned Markdown and Parquet outputs. Both routes must produce the same canonical page semantics before they may be compared.

The PyMuPDF text baseline emits native page text only, represented as a nullable-bbox paragraph. It receives no element geometry, table structure, or types from `our_parser`.

Credential-free integration prototypes currently produce:

- a ParseBench-targeted interchange JSONL preserving types, element boxes, tables, and cells;
- OmniDocBench-targeted page Markdown plus optional layout JSON;
- olmOCR-Bench-targeted page Markdown/text retaining compact HTML tables;
- RD-TableBench-targeted exact HTML tables, with Markdown-only tables reported as unsupported.

These are not official upstream adapters and do not establish compatibility with any moving checkout. Promote them only after fixture-based ingestion tests against immutable benchmark commits and dataset revisions.

## Coordinates and page projection

Benchmark geometry uses PDF points (1/72 inch), origin at the **top-left** of the displayed upright cropped page, positive x to the right and positive y downward. Boxes are `[x0, y0, x1, y1]` with positive width and height. `page_width` and `page_height` describe that same upright coordinate frame.

Tool/source pages are numbered from 1; canonical `page_index` values are zero-based. For rotated PDFs, source/tool boxes must be projected through the page's declared rotation into upright display coordinates before scoring; width and height swap for quarter turns. Crop-box translation is applied before rotation so `(0, 0)` is the displayed crop's top-left. Clip only tiny floating-point overshoot to page bounds. Reject a box with an unknown frame, non-finite coordinates, a non-right-angle page rotation, or a materially out-of-page extent; do not guess a transform. Preserve the tool-native box and transform metadata in raw output.

The current parser already emits upright page dimensions and projected boxes. Adapters for other tools must document and test their source frame before they can become `READY`.

## Restrictions

- **Credentials and payment:** do not request, discover, reuse, or commit API keys, cloud credentials, cookies, billing accounts, credits, or paid subscriptions. A hosted tool requiring any of these is skipped unless the owner separately approves and provisions it.
- **Resources:** do not download model weights or containers, install system packages, use a GPU, start a cloud job, or run an unbounded CPU/RAM/disk workload as part of this checkpoint. Record model size, expected accelerator, and resource limits before enabling a model-based tool.
- **Licensing:** before downloading or running a tool/model, record code license, model-weight license, dataset/output restrictions, and redistribution constraints. Unknown or incompatible terms are a skip, not permission. PyMuPDF is available under AGPL/commercial terms; downstream use must be reviewed for the applicable distribution model. This is not legal advice.
- **Leakage:** benchmark PDFs, labels, silver/golden Parquet, expected Markdown, and adjudication material must never enter prompts, fine-tuning, retrieval indexes, few-shot examples, tool configuration, or adapter heuristics. Do not tune on holdouts. Keep reviewer source maps and protected seeds outside tool-visible directories. Network-capable tools require an explicit data-handling review before receiving any benchmark PDF.

## Result labeling

A result may be labeled `same_harness: true` only when all tools used the same immutable source bytes, document/page selection, adapter contract, coordinate policy, normalization version, evaluator version, reference revision, timeout/resource policy, and run environment capture. Failures and skips remain in the denominator according to `config/benchmarks.yaml`; rerunning only failed documents breaks same-harness comparability.

Anything imported from a paper, vendor page, prior run, different corpus, different page subset, or different evaluator is `same_harness: false`, is contextual only, and must not appear in the same ranking table. Never relabel a reported number as an in-harness run.

See `INTEGRATION_NOTES.md` for the guarded path from this checkpoint to pinned integrations and actual runs.
