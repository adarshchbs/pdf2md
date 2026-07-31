# Native-text PDF parser comparison — corrected audit, 2026-07-31

## Bottom line

The earlier interpretation of this report was wrong. The values **78.85 / 60.87 / 82.85** for pdf2md were not computed over 361 common text PDFs:

- content faithfulness: **15 PDFs**;
- reading order: **10 PDFs**;
- text correctness: **15 PDFs**; and
- semantic formatting: **318 PDFs** in the corrected slice.

ParseBench emits a metric only when that example has relevant rules. In addition, `text_content` and `text_formatting` share stable test IDs. Treating a category membership count as every metric's denominator made four differently scoped measurements look like one 361-document comparison.

A candidate-blind source audit followed by direct comparison of all 15 content-metric PDFs found that the aggregates are useful as a small diagnostic, but **not reliable enough to support exact percentage or broad corpus-quality claims**. The evaluator sometimes ranks the candidates sensibly, sometimes misses major structural differences, and is outright reversed on `dash`, `notATable`, and `ikea`.

## Corrected source-only slice

Selection uses only source-PDF internals and upstream task tags. It does not inspect references, evaluator rules, candidate outputs, or scores. The revised classifier additionally rejects PDFs with no native text. Predominantly invisible text is treated as an OCR-layer signal only when the same page also has at least 90% raster coverage; this supplements the `GlyphLessFont` fingerprint without classifying invisible text alone as OCR.

| Slice accounting | Unique test IDs |
|---|---:|
| Full ParseBench source set | 2,078 |
| **Selected source-only diagnostic set** | **1,848** |
| Excluded by OCR/handwriting task tag | 132 |
| Non-PDF input | 42 |
| Raster-scan majority | 24 |
| OCR/invisible text layer | 21 |
| No native text | 11 |

Selected source-category memberships are 342 text-content, 327 text-formatting, 497 table, 446 layout, and 563 chart. These are **memberships, not metric denominators**. All 327 selected text-formatting IDs overlap text-content IDs and retain one evaluator result per shared ID.

The classifier remains diagnostic rather than authoritative. Manual inspection found custom-font PDFs whose native text exists but is unusable or partly corrupted; a source-only rule cannot prove semantic extractability in every font encoding.

## Corrected metric table

All three candidates completed the 342 selected text-content memberships, but only the following examples emitted each metric:

| Metric | Emitting PDFs | pdf2md local | PyMuPDF native text | PyPDF native text |
|---|---:|---:|---:|---:|
| Content faithfulness | **15** | 78.85 | **87.29** | 79.13 |
| Reading order | **10** | 60.87 | **76.73** | 65.89 |
| Text correctness | **15** | 82.85 | **89.94** | 82.01 |
| Semantic formatting | **318** | 0.96 | **1.04** | 0.99 |

The values are percentages, rounded to two decimals. They are official ParseBench aggregates over the retained per-example results, but the rows do not describe the same population.

The PyPDF aggregate is also heavily affected by one missing retained output, `text_simple__returns`, which the evaluator records as a successful empty-Markdown example with score zero. Removing that zero would change its 15-case content faithfulness from about **79.13 to 84.78**. That sensitivity is another reason not to treat the aggregate as a stable parser ranking.

## Manual PDF-grounded verification

The 15 content-metric source PDFs were audited before candidate outputs or evaluator references were opened. After the evidence packets were frozen and hashed, pdf2md, PyMuPDF, and PyPDF outputs were compared directly with the rendered PDFs and native PDF structure. References were consulted only afterward to diagnose metric behavior.

### Findings that are real

- **PyMuPDF is the strongest of these three on this small set.** It generally preserves native text, visual lines, and record boundaries better than pdf2md.
- **pdf2md damages recoverable structure.** Observed defects include flattening lists/logs/number sequences, column-major ordering of a row-major image grid, splitting captions into malformed table cells, and breaking a body sentence around a reconstructed form table.
- **pdf2md has severe foreign-content contamination in retained outputs.** `tableofcontent` and `notATable` contain large unrelated tables; `gridofnumbers` contains an unrelated invented table and loses eight codes. The source PDFs prove that content does not belong to those documents.
- **pdf2md sometimes improves page-furniture order.** It places footers more visually in `dash`, `name_list`, and other audited pages, while raw native extractors often follow PDF object order. That improvement does not offset the larger line/list/record damage on those cases.
- **PyPDF has its own defects.** It loses line structure, appends nonexistent text on `concession`, and has no retained output for `returns`.

### Where the evaluator is not sensible

- `notATable`: exact, line-faithful native outputs score below pdf2md even though pdf2md flattens all 20 records and appends unrelated content. All candidates receive order zero because pipe normalization differs between order and sentence rules.
- `dash`: pdf2md ranks first despite destroying nested-list structure. Artificial bracketed dash anchors make all order scores implausibly low.
- `ikea`: PyMuPDF ranks first even though pdf2md is visibly closer to page reading order. No order metric is emitted, and digit rules are contaminated by digits from reference image descriptions rather than visible PDF text.
- `gridofimages`: exact row-major caption extraction scores only about 0.66, no order metric is emitted, and all 50 photographs are outside the scalar's coverage.
- `japan2`, `unknonw`, and `clilogs`: byte-exact native extraction is penalized by line-wrap-sensitive anchors, malformed Thai tokenization, false occurrence limits, and an `O365` → `0365` reference error.
- `name_list`: Turkish dotted-I and sentence segmentation produce many false missing rules; the order metric ignores the misplaced page number and lost item boundaries.
- `numbers`: all candidates receive perfect content scores although pdf2md collapses most of a vertical list into one line.
- `legalRef`: all candidates are penalized for banner text that is genuinely image-only and absent from the native text layer.

The strict overall manual ranking conflicts with the evaluator's top ranking on three cases and several additional scalar ties hide meaningful line, list, grid, or paragraph damage. Absolute scores therefore are not interpretable as “percent of visible PDF recovered.”

## What can guide parser work

Use the audit as defect evidence, not as a leaderboard:

1. **Stop cross-document contamination first.** Unrelated tables appearing in several outputs are correctness failures more serious than the aggregate gap.
2. **Preserve native text before adding structure.** Assert document-local content conservation through grouping, table detection, joining, normalization, and rendering.
3. **Measure line/list/record structure separately.** The current content and order rules often ignore the structural damage pdf2md introduces.
4. **Keep page-furniture ordering as a distinct metric.** pdf2md sometimes improves headers/footers even when it worsens body structure.
5. **Do not tune parser heuristics to these public examples.** Reproduce defects on private or synthetic evidence before changing product rules.

## Structured-capability diagnostics

These are separate ParseBench capabilities and should not be compared directly with plain-text baselines.

| Area | Metric | Emitting / aggregate denominator | pdf2md |
|---|---|---:|---:|
| Tables | GriTS/TRM composite | 497 / 497 | 5.72 |
| Tables | GriTS cell content | 497 / 497 | 8.30 |
| Tables | Table record match | 497 / 497 | 2.26 |
| Layout | Content/order faithfulness | 62 / 70 | 66.20 |
| Layout | Localization pass | 376 / 376 | 46.13 |
| Layout | Classification pass | 376 / 376 | 36.44 |
| Layout | mAP@[.50:.95] | 376 / 376 | 13.69 |
| Layout | Attribution pass | 376 / 376 | 0.00 |
| Layout | Combined element-rule pass | 376 / 376 | 1.64 |
| Formatting | Text styling | 253 / 253 | 0.80 |
| Formatting | Title accuracy | 282 / 282 | 0.09 |
| Formatting | Semantic formatting | 318 / 318 | 0.96 |
| Charts, outside target | Data-point rule pass | 563 / 563 | 1.32 |

The layout content/order aggregate includes eight genuine-failure zeroes in addition to 62 emitted values. This is why its official denominator is 70.

The zero attribution value is partly an adapter/schema coverage gap: pdf2md does not project source spans and table-cell provenance into ParseBench's expected representation. Table reconstruction and semantic Markdown remain independently weak.

## Reproducibility and limitations

- Result label: `DIAGNOSTIC_NOT_COMPARABLE`.
- Dataset revision: `2805a1d940f95a203e0ae4b88be9934f7765b3fc`.
- Evaluator revision: `1d460294b3b9c57fb3fa944dc17a9c044c24d1e5`.
- Dataset JSONL bundle SHA-256: `4139192f2507c61f02d3169de750bea56af80c921cb78609f19f42cc931f98a4`.
- Classifier SHA-256: `42062b8e987796f1a3225e3f1b853a3d425f81ba140b4c508f070f8c189b87c3`.
- Scorer SHA-256: `8d02b7289f74b9306c43f195dcda71303c7a031b2db84a8f303b06e6b720d016`.
- Generated raw-summary SHA-256: `1b50cd76ce9b0140b5173ee3c2e15e82549d4028240725d63056a9d0114bb2fc`.
- Generated classifications SHA-256: `2dc31a0e01bacb098827fb541519bc03923d50b0a03e3461f2d4be40b8b80874`.
- Metrics were re-aggregated from retained per-example results using the pinned official implementation. `macro_metric_aggregation_counts` records emitted examples, genuine-failure zero padding, and the denominator of each emitted metric's official `avg_*` macro aggregate. It does not describe micro, count-weighted, or synthetic `_predicted` aggregates.
- Durable manual evidence is retained under `benchmarks/evidence/digital-pdf-manual-audit-2026-07-31/`.
- Candidate-blind source audits: `exact-metric-audit-a.md` (`c942e531f71cfb1a442da3f4b927f37a006a29311411fbe17096d84cb5b6836a`), `exact-metric-audit-b.md` (`4e192cbaa89fd5642d828ae6b3f15e639b9f87ed7d74f3db6ca898887a257c0f`), and `exact-metric-audit-c.md` (`a2d53b82a6d4e2eab568fc4ec9d7ea7219838589aa3d881fbd39ca543aa93dd4`).
- Candidate verification reports: `exact-metric-verification-a.md` (`be90d0cf358c78cff4e037dc433302345eaa581ed914e8a94bb0f3c17a027b75`), `exact-metric-verification-b.md` (`114828a7d334b508a19a74605c06160b301bde0020ecda079e0c8e3ec12cb06f`), and `exact-metric-verification-c.md` (`3f8cad2cb980e5d8808ba192d0ea82f3e2da149e2a3beb55d6536d7c73ebe7d9`). `legalRef` was additionally verified in `digital-comparison-b.md` (`b92e7d561d0fa695bd16f8d47673159fbc20b03e950feff0a47abb4292899968`).
- Candidate resource envelopes differed, and the original pdf2md run came from a mutable working tree.
- Public cases remain evaluation-only and were not copied into tests, prompts, retrieval systems, or parser rules.
- Anthropic API cost: $0. Gemini API cost: $0.

Machine-readable results are in `digital-pdf-comparison-2026-07-31.json`. Generated evidence remains ignored under `benchmarks/runs/parsebench-digital-pdf-2026-07-31-v7/`.
