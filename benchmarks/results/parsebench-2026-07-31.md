# Public PDF parser benchmark results — 2026-07-31

## Status and scope

- **ParseBench full snapshot:** `llamaindex/ParseBench` revision `2805a1d940f95a203e0ae4b88be9934f7765b3fc` (2,078 unique evaluation files; 2,553 test-case records across five dimensions).
- **ParseBench small snapshot:** revision `68bbab242f749df2e2ef753daabcbbbe291d943e` (12 unique files; three files per dimension, with text files reused across two dimensions).
- **ParseBench evaluator:** code revision `1d460294b3b9c57fb3fa944dc17a9c044c24d1e5`, version `0.2.0`.
- **olmOCR-Bench:** evaluator revision `f7cfe4c22098b154c76b6ec950d1c0a464eecf8d`; dataset revision `54a96a6fb6a2bd3b297e59869491db4d3625b711` (1,403 PDFs, 8,413 scored tests).
- **RD-TableBench:** official scorer revision `1cae108e6395ddc8389af17385f9769519070558`; dataset revision `7748503e2bd5f210d27aa2ef5fdf4b8aa13099bb` (1,000 PDFs and reference tables).
- **Result label:** `NOT_COMPARABLE` across candidates. Source bytes and evaluator revisions match within each snapshot, but concurrency/resource envelopes differed and the environment was not isolated from other local workloads. The full and small snapshots are also separate harnesses and must not be placed in one ranking.
- All scores below are reproduced locally, not vendor-published scores. They are side-by-side diagnostic evidence, not a controlled ranking.
- No public evaluation case was copied into a product rule, fixture, prompt, or regression test.
- Anthropic API cost: **$0**. Gemini API cost: **$0**. No hosted parser or paid API was called.

ParseBench attempted to initialize its optional `judge` normalization strategy, but the Anthropic SDK was intentionally absent. The official evaluator logged the import failure and continued with its deterministic rule-based evaluation. This omission applied identically to every local candidate and is preserved in raw run logs.

### Captured implementation identity

The parser was run from a dirty working tree, so the Git commit alone is insufficient. The run identity is:

- Repository HEAD observed after the run: `c5709ab32b6c826a2f45641cbdcfa3db19c12b74`
- Sorted `app/pdf2md/*.py` content bundle: 28 files, SHA-256 `cc7b203f324a60542573781cb94c6bf4bc6c61998b99f86debb1d000ade51eeb`
- `uv.lock` SHA-256: `178aff8f34037f55b782fe81a15b85657f2996244adf77752910e35d51d89290`
- Runtime: CPython 3.13.13, macOS arm64
- ParseBench environment: pdf2md 0.1.0, PyMuPDF 1.28.0, PyPDF 6.14.2, ParseBench 0.2.0

The HEAD and content bundle were captured after the run while concurrent repository work was present; they bound the retained post-run state but cannot prove that every file stayed unchanged during inference. The result therefore remains `NOT_COMPARABLE` rather than claiming a frozen release.

## Full ParseBench scores

Values are official ParseBench aggregates on a 0–100 scale. `—` means that the candidate did not emit the required capability; it is not treated as a measured zero.

| Candidate | Content faithfulness | Semantic formatting | Tables (GTRM composite) | Charts (data-point pass) | Visual grounding (element pass) | Layout mAP@[.50:.95] |
|---|---:|---:|---:|---:|---:|---:|
| **pdf2md local** | **43.59** | **0.76** | **5.14** | **1.20** | **1.63** | **13.63** |
| PyMuPDF native text | 13.36 | 0.49 | — | — | — | — |
| PyPDF native text | 13.19 | 0.47 | — | — | — | — |

Additional pdf2md full-run metrics:

- Reading-order normalized score: **39.90**
- Text-correctness normalized score: **20.60**
- Table GriTS content score: **7.46**
- Visual-grounding localization pass: **46.19**
- Visual-grounding classification pass: **36.38**
- Visual-grounding attribution pass: **0.00**
- Micro rule pass rate across all eligible rules: **24.22**

### Full-run outcomes and latency

| Candidate | Inference success | Inference failures | Average latency per file | Evaluator success / failed / skipped |
|---|---:|---:|---:|---:|
| **pdf2md local** | 2,003 / 2,078 (**96.39%**) | 75 | 2,707.8 ms | 2,020 / 50 / 8 |
| PyMuPDF native text | 2,036 / 2,078 (**97.98%**) | 42 | 8.86 ms | 1,642 / 436 / 0 |
| PyPDF native text | 2,011 / 2,078 (**96.78%**) | 67 | 66.80 ms | 1,642 / 436 / 0 |

Failure accounting is intentionally retained:

- The corpus contains non-PDF `.png`/`.jpg` layout inputs. The current pdf2md product boundary is born-digital PDF-only, so those attempts failed rather than being silently converted or omitted.
- Some PDFs exposed genuine pdf2md exceptions, including `page elements must share one known canonical page width`; these remain in the failure denominator and raw `_errors.json`.
- ParseBench synthesized blank results for missing inference outputs before evaluation, as recorded in the evaluator log. Evaluator outcome counts are therefore not identical to inference outcome counts.
- PyPDF's upstream `_summary.json` reports `total=2053` even though `successful=2011` and `failed=67` sum to 2,078. The table uses 2,078 as the source-request denominator and does not conceal this upstream accounting defect.
- Text-only baselines cannot produce structured tables or visual grounding. Their layout evaluations fail by capability, so no layout score is reported.

## Local model competitors (small ParseBench snapshot only)

Docling was run locally at code tag/version `v2.117.0`, commit `f2683c0b5aa14a53b74373b0640260891cdbc1b0`, using its default CPU PDF pipeline. Resolved packages included docling-core 2.88.0, docling-ibm-models 3.13.3, and docling-parse 7.8.1. The observed Hugging Face model snapshot was `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8`: layout-heron cache 171,764,371 bytes (bundle SHA-256 `8c3c9a5a112a8523528e124980b89f808f135bef68cb78757f396d531c05d796`) and docling-models cache 358,236,323 bytes (bundle SHA-256 `f03b36f44e7e7ac189f006c696290fdcbc607f76dd16b61b7513db9731f4d02f`). RapidOCR additionally downloaded 40.09 MB of local weights.

MinerU was run locally at code revision `0dfc9460cd9ab693b9af60ae3fbffd7bc111b062` (checkout tag `mineru-3.4.4-released`, package version 3.4.3) with the MLX VLM engine. The model was `opendatalab/MinerU2.5-Pro-2605-1.2B` revision `bff20d4ae2bf202df9f45284b4d43681555a97ed`: 13 files, 2,328,028,720 bytes, bundle SHA-256 `dd52a7a95a5832428d2f6e5629ce6361ed0bca41fc54480d80dc9517702f4b62`. Runtime packages included MLX 0.31.1, mlx-vlm 0.3.9, transformers 4.57.6, torch 2.13.0, and torchvision 0.28.0. The batch took 138.9 seconds for 12 one-page PDFs (about 11.58 seconds per file), including one model load and sequential inference. No hosted API was called.

These model results are observed local runs, but they remain `NOT_COMPARABLE` to the full-snapshot table and to each other because resource controls and inference stacks differ. Docling's first run allowed uncontrolled model downloads, and TableFormerV2's weight license was not declared in the inspected metadata. MinerU's code license adds commercial thresholds and an online-service attribution obligation; deployment use requires a separate license review.

| Candidate | Content normalized text | Semantic formatting | Tables | Charts | Visual element pass | Attribution pass | Layout mAP | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MinerU 2.5 Pro local MLX** | **38.16** | **38.16** | **61.13** | **33.33** | **69.76** | **78.54** | **42.39** | ~11,575 ms |
| Docling local | 35.71 | 35.71 | 0.00 | 0.00 | 69.43 | 70.27 | 28.18 | 7,482.08 ms |
| pdf2md local | 0.00 | 0.00 | 1.32 | 0.00 | 1.15 | 0.00 | 14.33 | 841.00 ms |
| PyMuPDF native text | 0.00 | 0.00 | — | — | — | — | — | 12.75 ms |
| PyPDF native text | 0.00 | 0.00 | — | — | — | — | — | 49.25 ms |

MinerU, Docling, and pdf2md completed 12/12 inference and 12/12 evaluation attempts. The two native-text baselines completed 12/12 inference but failed the three visual-grounding evaluations because they emit no layout structure.

## olmOCR-Bench official score

The official credential-free evaluator scored all 8,413 tests against the 1,403 generated candidate files. Parser inference produced 1,370 successful documents and 33 failures; failed documents were retained as empty Markdown candidates. The official evaluator reported:

- **Headline score: 30.2%** (bootstrap 95% CI **29.3%–31.2%**; average of per-JSONL scores)
- Baseline validity: **84.8%** over 1,403 tests
- Absent-text checks: **42.5%** over 823 tests
- Reading order: **44.9%** over 1,061 tests
- Present-text checks: **25.7%** over 721 tests
- Table checks: **10.4%** over 1,020 tests
- Math checks: **0.0%** over 3,385 tests

Per-file-category results were arXiv math 0.0%, baseline 84.9%, headers/footers 37.6%, long tiny text 41.9%, multi-column 53.8%, old scans 13.3%, old-scans math 0.0%, and tables 10.5%. The evaluator retained 6,106 failed tests in `benchmarks/runs/olmocr-bench/pdf2md_local/failed.jsonl`. This is a metric score, not a pytest pass rate.

## RD-TableBench official score

The official hierarchical table-similarity scorer evaluated all 1,000 samples:

- **Mean table similarity: 0.0000**
- Median similarity: **0.0000**
- Missing predictions: 0
- Empty/invalid parsed tables scored as zero: **1,000**

The scanned/image-table corpus produced no detected canonical tables because the current parser is born-digital PDF-only and has no OCR table path. This is a genuine capability result rather than an adapter crash. RD-TableBench data is CC BY-NC-ND 4.0 and remains evaluation-only.

## Interpretation

1. The observed full-corpus content-faithfulness score was **43.59** for pdf2md and **13.36/13.19** for the native-text baselines. Because the run is `NOT_COMPARABLE`, this is a diagnostic difference rather than a controlled ranking claim.
2. pdf2md's observed structured-table score is nonzero but low (**5.14 GTRM composite**), and chart extraction is effectively absent (**1.20**).
3. Localization is much stronger than element-level visual grounding: **46.19 localization pass** but only **1.63 element pass** and **0 attribution pass**. The adapter has element boxes but does not yet provide text-span/cell attribution in the form ParseBench expects.
4. Semantic formatting is largely lost (**0.76**), despite a higher reading-order score (**39.90**).
5. MinerU and Docling both produced substantially stronger small-snapshot layout/formatting scores than pdf2md; MinerU also produced the strongest table and chart scores on that snapshot. Identical controls and full-snapshot runs are still required before making a defensible ranking.
6. The **30.2% olmOCR** score exposes the clearest quality gaps: math is completely unsupported, table checks are low, and scanned-document extraction is weak.
7. The **0.0000 RD-TableBench** score confirms that scanned table extraction is unsupported without an OCR-capable product path.

## Raw evidence

Generated evidence is intentionally ignored by Git and retained locally under:

- `benchmarks/runs/parsebench-full/pdf2md_local/pdf2md_local/`
- `benchmarks/runs/parsebench-full/pymupdf_text/pymupdf_text/`
- `benchmarks/runs/parsebench-full/pypdf_baseline/pypdf_baseline/`
- `benchmarks/runs/parsebench-test/docling_local/docling_local/`
- `benchmarks/runs/parsebench-test/mineru_vlm/parsebench-v2/mineru2605pro_precomputed/`
- `benchmarks/runs/olmocr-bench/pdf2md_local/`
- `benchmarks/runs/rd-tablebench/pdf2md_local/official-scores/`

ParseBench directories contain normalized inference JSON, raw provider JSON, summaries, errors where applicable, evaluator JSON/CSV, rule-level CSV, Markdown, and HTML reports. The olmOCR directory retains the official failed-test JSONL. The RD directory retains per-sample Parquet scores and the official summary JSON.
