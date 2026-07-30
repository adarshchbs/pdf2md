# ParseBench local-provider integration

These provider modules are the reference-free inference adapters used for the reproduced scores in `../../results/parsebench-2026-07-31.md`. They are copied into an ignored ParseBench checkout pinned at:

```text
1d460294b3b9c57fb3fa944dc17a9c044c24d1e5
```

They are intentionally not importable by product code. ParseBench owns its provider registry, pipeline registry, normalized output models, and evaluator-specific layout adaptation, so the modules must execute inside that pinned isolated environment.

## Checkout wiring

Copy the Python files into:

```text
src/parse_bench/inference/providers/parse/
```

Add `pdf2md_local`, `docling_local`, and (for a precomputed local MinerU run) `mineru2605pro_precomputed` to `_PROVIDER_MODULES` in that directory's `__init__.py`. Register these `PipelineSpec` values in `src/parse_bench/inference/pipelines/parse.py`:

```python
PipelineSpec(
    pipeline_name="pdf2md_local",
    provider_name="pdf2md_local",
    product_type=ProductType.PARSE,
    config={},
)

PipelineSpec(
    pipeline_name="docling_local",
    provider_name="docling_local",
    product_type=ProductType.PARSE,
    config={},
)

PipelineSpec(
    pipeline_name="mineru2605pro_precomputed",
    provider_name="mineru2605pro_precomputed",
    product_type=ProductType.PARSE,
    config={},
)
```

Add `docling_local` to ParseBench's existing Docling layout-adapter registration. Register `pdf2md_local` to a local subclass of `LlamaParseLayoutAdapter` whose `to_layout_output()` passes `raw_output["pages"]` directly to `extract_all_layouts_from_llamaparse_output()`. This avoids importing the hosted `llama_cloud` SDK. It must set `_pages_payload` so the inherited attribution-block conversion sees the same immutable element payload. Add `mineru2605pro_precomputed` to the existing `MinerU25LayoutAdapter` registration; the precomputed provider deliberately reuses the pinned evaluator's native MinerU 2.5 normalization and layout projection.

The run used these isolated installation commands:

```bash
uv sync --project benchmarks/downloads/ParseBench
uv pip install --python benchmarks/downloads/ParseBench/.venv/bin/python pymupdf pypdf
uv pip install --python benchmarks/downloads/ParseBench/.venv/bin/python --no-deps --editable .
uv pip install --python benchmarks/downloads/ParseBench/.venv/bin/python \
  --editable 'benchmarks/downloads/docling[standard]'
```

No dataset labels or expected output are passed to any provider. Each receives only the source document path and pipeline metadata. The pdf2md adapter uses top-left PDF-point geometry exactly as emitted by the product schema. Docling's official ParseBench normalization converts its native geometry. MinerU was run first through its local MLX CLI with references unavailable; `MINERU_PRECOMPUTED_DIR` then points ParseBench at those immutable Markdown and model-layout outputs so scoring does not rerun the model or expose labels.

## Reproduced commands

```bash
uv run --project benchmarks/downloads/ParseBench parse-bench inference run \
  pdf2md_local --input_dir benchmarks/datasets/parsebench-full \
  --output_dir benchmarks/runs/parsebench-full/pdf2md_local \
  --max_concurrent 8 --no_rich --per_file_timeout 120

uv run --project benchmarks/downloads/ParseBench parse-bench evaluation run \
  --output_dir benchmarks/runs/parsebench-full/pdf2md_local/pdf2md_local \
  --test_cases_dir benchmarks/datasets/parsebench-full \
  --pipeline_name pdf2md_local --max_workers 8 --force
```

Use the identical inference/evaluation commands with `pymupdf_text` and `pypdf_baseline` for the native-text baselines. Docling was run on `parsebench-test` with `max_concurrent=1` and a 600-second per-file timeout.

## Known limitations

- pdf2md rejects non-PDF image inputs rather than silently converting them.
- The current pdf2md adapter provides element-level spans only. ParseBench's character/cell attribution metric therefore remains unsupported in substance and scored zero where attempted.
- Docling's default pipeline downloads local model weights on first use. Record those artifacts before any future frozen full-corpus run.
- ParseBench's optional `judge` normalizer was unavailable and skipped. Do not install an API SDK or send benchmark inputs to a hosted model merely to remove that log message.
