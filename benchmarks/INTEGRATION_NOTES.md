# Integration notes

## Checkpoint state

As of 2026-07-31, the synthetic development harness remains available and pinned public evaluations have been reproduced locally. The full ParseBench snapshot was scored for pdf2md, PyMuPDF native text, and PyPDF native text. The small ParseBench snapshot was additionally scored with pinned Docling and MinerU 2.5 Pro. pdf2md was also scored with the official olmOCR-Bench and RD-TableBench evaluators. See `results/parsebench-2026-07-31.md` and its machine-readable JSON companion for the public results. See `results/digital-pdf-comparison-2026-07-31.md` for the reference-independent born-digital diagnostic slice, direct native-text comparison, and prioritized weakness summary.

Generated checkouts, datasets, models, and run artifacts remain ignored. They are evidence, not product fixtures, and must not enter parser rules or tests. The public scores do not turn the synthetic smoke into accuracy evidence and do not promote unrelated integration prototypes to official adapters.

## Admission gate for a future integration

A tool may move to `READY` only in a change that supplies all of the following:

1. A reproducible integration pinned by an actual immutable identifier: package version plus lockfile resolution, container digest, and/or model revision/weight digest as applicable. Do not invent a pin in advance, and do not use a floating branch, `latest` image, or mutable model tag.
2. Code-license and model-weight-license records, including commercial-use, redistribution, remote-code, and output restrictions. Record dataset terms when bundled data is involved.
3. Model/resource notes: download size, installed size, CPU architecture, RAM, disk, GPU/accelerator type and memory, expected runtime, worker count, timeout, and whether network access is required.
4. An isolated adapter that invokes the public tool boundary, captures stdout/stderr, exit status, elapsed time, peak resources, raw output, tool/model identity, and effective configuration without exposing references.
5. Tests for page selection, one-based page mapping, top-left point-coordinate projection, rotations (0/90/180/270), crop boxes, missing-field reporting, deterministic naming, and nonzero exits/timeouts.
6. Approval for any credential, payment, model download, container pull, GPU/cloud use, network transmission, or license obligation. Approval must be explicit and out-of-band; it must not be inferred from this file.
7. A leakage review proving that labels, references, holdouts, reviewer artifacts, protected source maps/seeds, prior tool outputs, and expected answers are unavailable to the tool and adapter.

Pinning and admission happen before a benchmark run. A proposed integration with no verified pin remains `NOT_READY`; the current file intentionally names no exact versions for tools that were not installed.

## Exact adapter contract

The current typed seam is `BenchmarkAdapter.process(pdf_path) -> AdapterRun` in `adapters/base.py`. The development smoke invokes it in-process and validates the input hash before and after every attempt. A frozen benchmark runner should pass an adapter only:

- immutable input PDF path and SHA-256;
- ordered, unique, one-based selected pages (or an explicit all-pages value);
- an empty per-document raw-output directory;
- a per-tool timeout/resource envelope;
- tool configuration that contains no reference-derived values.

The current local adapters return an `AdapterRun` containing immutable raw records and canonical pages; the runner persists each stream separately. External-tool adapters must instead return a process outcome plus a manifest of raw files, followed by a separate deterministic normalization step. Evaluators consume normalized records only. They must not invoke tools or mutate records.

For `our_parser`, the public boundary is exactly:

```text
uv run pdf2md extract INPUT.pdf OUTPUT.md --parquet-path ELEMENTS.parquet [--page PAGE]...
```

For `pymupdf_text`, the exact `uv run python -c ...` command is recorded in `README.md`. That baseline exposes page-separated text only. Its adapter must not import `app.pdf2md`, call table detection, or borrow parser geometry.

## Page projection policy

Normalize geometry to the displayed upright crop in PDF points: top-left origin, x rightward, y downward. Map the tool's declared native frame to crop-local coordinates, apply the PDF page rotation, recompute the axis-aligned envelope, and report upright page width/height. Quarter turns swap dimensions. Keep both native coordinates and transform provenance in raw output.

Never infer whether coordinates are pixels, points, normalized fractions, media-box coordinates, crop-box coordinates, bottom-left coordinates, or already rotated. An undocumented or ambiguous frame yields the exact skip/failure reason `coordinate_frame_unverified`. A tool with text but no geometry remains eligible for text-only metrics with geometry fields explicitly missing; it is not assigned zero-valued boxes.

## Missing fields and failures

Normalization must distinguish:

- `not_emitted`: the tool does not expose the field;
- `not_applicable`: the field does not apply to the element;
- `parse_error`: raw output exists but violates the documented format;
- `coordinate_frame_unverified`: geometry cannot be projected safely;
- `tool_error`: nonzero exit or tool-declared failure;
- `timeout`: resource envelope expired;
- `resource_exceeded`: enforced RAM/disk/GPU limit was exceeded;
- `policy_blocked`: credential, payment, network, license, model, or leakage gate blocked execution.

Do not convert any of these to an empty string, zero box, empty table, or successful document. Preserve failed and skipped documents in manifests and aggregate denominators.

## Same-harness reports

Each run manifest must capture source hashes, selected pages, tool and model pins, adapter/normalizer/evaluator revisions, reference revision, environment, command, effective settings, limits, and every document outcome. Only a single orchestrated run satisfying the identity rules in `README.md` may use `same_harness: true`.

External published scores may be cited only in a separate contextual section labeled `same_harness: false`, with their original corpus and metric. They cannot fill missing cells, establish a local rank, or be described as a run by this project.

## Tool-specific future work

- **Docling:** select and lock the package/container and every downloaded model artifact; verify licenses and CPU/GPU behavior; implement native structure/geometry mapping.
- **PaddleOCR / PP-Structure:** select the exact pipeline and language models; lock code and weights; record model licenses and accelerator requirements; verify pixel-to-point projection.
- **MinerU:** select a reproducible release/container and model bundle; review code/weight licenses and remote-code/network behavior; implement its structured-output adapter.
- **Marker:** pin package and model revisions; review code and weight terms; measure RAM/GPU requirements; isolate optional LLM/provider paths so no credential or paid endpoint is used implicitly.
- **Nougat:** establish maintained, reproducible package/weights compatible with the benchmark environment; review weight license and GPU cost before adapter work.
- **olmOCR:** pin code, model, and inference stack; review model/data terms and GPU requirements; ensure no hosted inference or benchmark-data transmission occurs.
- **Cloud document APIs:** remain blocked until the owner explicitly approves provider, credentials, billing cap, region/retention settings, and benchmark-data transmission. Approval would not make vendor-reported scores same-harness evidence.

These are planned gates, not claims that installation or execution succeeded.
