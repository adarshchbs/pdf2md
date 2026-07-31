# Born-digital PDF parser comparison — 2026-07-31

## Scope

This diagnostic removes OCR/scanned-document work from the retained full ParseBench results and compares pdf2md with PyMuPDF native text and PyPDF native text on an exact common set.

The selection is reference-independent. It does not inspect expected Markdown, rules, labels, or candidate output. A test ID is excluded when:

- the source is not a PDF;
- the task is tagged `ocr` or `handwritting` upstream;
- any page uses the `GlyphLessFont` OCR-layer fingerprint; or
- at least half of its pages have fewer than 20 native characters while raster images cover at least 90% of the page.

A low-text page is not rejected by itself. This keeps sparse born-digital pages and chart-heavy pages whose raster coverage does not indicate a full-page scan.

| Slice accounting | Test IDs |
|---|---:|
| Full ParseBench source set | 2,078 |
| **Selected born-digital set** | **1,879** |
| OCR/handwriting-tagged tasks | 132 |
| Non-PDF images | 42 |
| Raster-scan majority | 24 |
| OCR text-layer fingerprint | 1 |

Selected task counts are 361 text-content, 346 text-formatting, 499 table, 451 layout, and 568 chart test IDs. Some text test IDs occur in both text categories.

## Fair common text comparison

These are the 361 selected `text_content` test IDs. All three candidates completed every one, and all three emit the required text capability. This is the cleanest direct comparison of the problem all three parsers actually solve.

| Candidate | Content faithfulness | Reading order | Text correctness | Semantic formatting |
|---|---:|---:|---:|---:|
| **pdf2md local** | 78.85 | 60.87 | 82.85 | 0.91 |
| PyMuPDF native text | **87.29** | **76.73** | **89.94** | **0.99** |
| PyPDF native text | 79.13 | 65.89 | 82.01 | 0.94 |

### What this means

1. **pdf2md does not beat raw PyMuPDF on clean digital text.** It trails by 8.44 points in content faithfulness, 15.86 in reading order, and 7.09 in text correctness.
2. **pdf2md is approximately level with PyPDF on content correctness**, but still trails it by 5.02 points in reading order.
3. The parser's grouping and normalization heuristics add structure, but they also damage otherwise recoverable native text through joining, ordering, filtering, or normalization decisions.
4. Semantic formatting is effectively absent for every candidate. pdf2md is not currently converting its richer font/geometry evidence into useful Markdown semantics.

The much larger full-slice content difference—pdf2md 68.70 versus PyMuPDF 14.46 and PyPDF 14.26—is not a fair text-only ranking. It includes 451 layout tasks where the native-text baselines cannot emit the required structured layout and retain 387 capability failures each. The table above avoids that distortion.

## pdf2md structured capabilities

PyMuPDF and PyPDF native-text baselines do not emit structured tables or layout boxes. Their corresponding cells are therefore unsupported (`—`), not measured zeroes.

### Digital tables — 499 test IDs

| Metric | pdf2md |
|---|---:|
| GriTS/TRM composite | **5.70** |
| GriTS cell content | **8.26** |
| Table record match | **2.25** |

This is the largest weakness inside the parser's intended scope. Detection occasionally succeeds, but row/column recovery, cell assignment, merged spans, headers, and complete content reconstruction are not reliable.

### Digital layout — 451 test IDs

| Metric | pdf2md |
|---|---:|
| Layout-task content faithfulness | **66.59** |
| Layout-task reading order | **66.59** |
| Localization pass | **46.14** |
| Classification pass | **36.48** |
| Layout mAP@[.50:.95] | **13.67** |
| Attribution pass | **0.00** |
| Combined element-rule pass | **1.63** |

The parser often finds approximately relevant regions, but exact boxes and element classes remain weak. The zero attribution score is partly an integration/schema gap: pdf2md does not project source spans and table-cell provenance into the representation ParseBench expects. Since the combined element rule requires attribution, it collapses to 1.63 even though localization is 46.14.

### Semantic formatting — 346 test IDs

| Metric | pdf2md |
|---|---:|
| Semantic formatting | **0.91** |
| Text styling | **0.76** |
| Title accuracy | **0.08** |
| Title hierarchy | **0.00** |
| Underline | **0.00** |
| Strikeout | **0.00** |

The current Markdown renderer loses nearly all font-derived semantics. Heading hierarchy, bold/italic evidence, decorations, and title roles need to survive extraction, normalization, and rendering.

### Charts — outside the current target

The retained born-digital slice contains 568 chart test IDs. pdf2md's data-point score is **1.31**. Chart interpretation remains an explicitly unsupported capability and is not included in the work priorities below.

## Robustness on the digital slice

pdf2md evaluation outcomes:

- 1,869 successful
- 8 genuine failures
- 2 layout skips
- 1,879 total selected test IDs

The eight failures are all relevant digital-PDF robustness defects:

| Failure type | Count |
|---|---:|
| Legacy source-item ID not page/kind scoped | 4 |
| Unknown canonical page width | 2 |
| Degenerate zero-width drawing rectangle | 2 |

These should fail at a narrow element boundary or be normalized correctly rather than aborting the entire document.

## Prioritized work summary

### Priority 0 — stop degrading native text

Use PyMuPDF native text as a per-document lower-bound diagnostic. Identify where pdf2md loses the 8.44 faithfulness and 15.86 reading-order points through:

- span and line joining;
- dehyphenation and whitespace normalization;
- duplicate suppression;
- header/footer removal;
- column ordering; and
- element filtering.

The target is not to copy raw extraction blindly; it is to ensure each structural heuristic demonstrates an improvement rather than silently reducing recoverable text.

### Priority 0 — rebuild table evaluation by stage

Split table quality into boundary detection, row clustering, column clustering, cell assignment, span reconstruction, and rendering. The 5.70 composite is too low for aggregate heuristic tuning without stage-level evidence.

### Priority 0 — preserve semantic formatting

Carry font weight, font size, style, decoration, title evidence, and list evidence through the canonical schema and renderer. Formatting below 1% means this is effectively unimplemented.

### Priority 1 — improve reading order

Replace predominantly coordinate-based ordering with explicit page regions, column segmentation, containment, and a reading-order graph. pdf2md currently trails raw PyMuPDF by 15.86 points on the clean common text set.

### Priority 1 — complete layout attribution

Project canonical source fragments and table cells to ParseBench-compatible text/cell attribution. This separates a real parser-quality problem from the current adapter coverage gap and makes the combined visual metric meaningful.

### Priority 1 — remove whole-document robustness failures

Fix source-item IDs, page-width normalization, and degenerate drawing handling. These are deterministic correctness defects on supported digital inputs.

### Priority 2 — improve element classification and exact geometry

After attribution is measurable, focus on the 36.48 classification and 13.67 mAP scores. Localization at 46.14 shows there is usable signal, but boxes and labels need substantial refinement.

## Reproducibility and limitations

- Result label: `DIAGNOSTIC_NOT_COMPARABLE`.
- Dataset revision: `2805a1d940f95a203e0ae4b88be9934f7765b3fc`.
- Evaluator revision: `1d460294b3b9c57fb3fa944dc17a9c044c24d1e5`.
- Dataset JSONL bundle SHA-256: `4139192f2507c61f02d3169de750bea56af80c921cb78609f19f42cc931f98a4`.
- Classifier SHA-256: `b52a44577cb843706f43ca205f44244e937734cea891ef069c6c1461fc130214`.
- Scorer SHA-256: `62e7bc41dccaa975b4ea2d889d64041acfe0e5cc7108074ece3676770c4c65f0`.
- Generated raw-summary SHA-256: `d0742673b547a9af1a9793fac1759d5bebc533ee5231b8f730f816dd7183a814`.
- The committed Markdown and compact JSON are manually curated from that generated raw summary; numeric scores are rounded for presentation.
- Metrics were re-aggregated from retained per-example evaluator results using the pinned official aggregation logic. Eligible failures remain in denominators.
- The classifier is a conservative diagnostic rule, not authoritative document ground truth.
- Candidate resource envelopes differed, and the original pdf2md run came from a mutable working tree. Values are diagnostic evidence, not a controlled production ranking.
- Public evaluation cases remain evaluation-only and were not copied into tests, fixtures, prompts, retrieval systems, or parser rules.
- Anthropic API cost: $0. Gemini API cost: $0.

Machine-readable results are in `digital-pdf-comparison-2026-07-31.json`. Generated classifications and full aggregates remain ignored under `benchmarks/runs/parsebench-digital-pdf-2026-07-31-v4/`.
