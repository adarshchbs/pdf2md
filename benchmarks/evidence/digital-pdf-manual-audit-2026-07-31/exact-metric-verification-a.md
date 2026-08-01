# Exact-metric verification A

## Scope and evidence discipline

This is a verification-only comparison of retained outputs from `pdf2md_local`, `pymupdf_text`, and `pypdf_baseline` for four one-page ParseBench PDFs. No parser was rerun and no repository file was edited.

The frozen candidate-blind source audit was read first and never modified:

- `/private/tmp/claude-501/-Users-adarsh-project-pdf2md/1c16afff-29c0-4394-8e54-7ccb74cc0ca2/scratchpad/exact-metric-audit-a.md`
- recorded SHA-256: `c942e531f71cfb1a442da3f4b927f37a006a29311411fbe17096d84cb5b6836a`

Candidate judgments below are grounded in that audit and direct inspection of its frozen 3x PDF renders, not in expected Markdown. The evaluator reference/rules were opened only afterward to audit whether their scores agree with the PDF evidence.

Retained candidate files are under:

- `/Users/adarsh/project/pdf2md/benchmarks/runs/parsebench-full/pdf2md_local/pdf2md_local/text/`
- `/Users/adarsh/project/pdf2md/benchmarks/runs/parsebench-full/pymupdf_text/pymupdf_text/text/`
- `/Users/adarsh/project/pdf2md/benchmarks/runs/parsebench-full/pypdf_baseline/pypdf_baseline/text/`

Evaluator evidence came from each pipeline's `_evaluation_results.csv` and `_evaluation_report_detailed.html`. The nominal `_evaluation_rule_results.csv` files are all zero-byte files; therefore the retained per-rule pass/fail details had to be read from the HTML report's embedded `DATA.examples[].ruleResults` and `ruleDetails` objects. Reference rules are in `/Users/adarsh/project/pdf2md/benchmarks/datasets/parsebench-full/text_content.jsonl`.

Metric labels used below:

- **score** = `normalized_text_score` (the evaluator's headline scalar)
- **faith** = `content_faithfulness`
- **order** = `normalized_order`
- **correct** = `normalized_text_correctness`
- **rules** = aggregate passed/total exact rules

## Score overview and manual ranking

| PDF | candidate | score | faith | order | correct | rules | evaluator rank by score | manual PDF-grounded rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tableofcontent | pdf2md_local | 0.372149 | 0.447018 | 0.147541 | 0.596757 | 91/506 | 3 | 3 |
| tableofcontent | pymupdf_text | 0.980903 | 0.974537 | 1.000000 | 0.961806 | 481/506 | =1 | =1 |
| tableofcontent | pypdf_baseline | 0.980903 | 0.974537 | 1.000000 | 0.961806 | 481/506 | =1 | =1 |
| dash | pdf2md_local | 0.537143 | 0.662857 | 0.160000 | 0.914286 | 80/105 | 1 | 3 |
| dash | pymupdf_text | 0.532449 | 0.669932 | 0.120000 | 0.944898 | 79/105 | =2 | =1 |
| dash | pypdf_baseline | 0.532449 | 0.669932 | 0.120000 | 0.944898 | 79/105 | =2 | =1 |
| gridofimages | pdf2md_local | 0.647947 | 0.647947 | N/A | 0.647947 | 40/157 | 3 | 3 |
| gridofimages | pymupdf_text | 0.659550 | 0.659550 | N/A | 0.659550 | 42/157 | =1 | =1 |
| gridofimages | pypdf_baseline | 0.659550 | 0.659550 | N/A | 0.659550 | 42/157 | =1 | =1 |
| notATable | pdf2md_local | 0.428571 | 0.571429 | 0.000000 | 0.857143 | 87/127 | 1 | 3 |
| notATable | pymupdf_text | 0.416667 | 0.555556 | 0.000000 | 0.833333 | 106/127 | =2 | =1 |
| notATable | pypdf_baseline | 0.416667 | 0.555556 | 0.000000 | 0.833333 | 106/127 | =2 | =1 |

Manual ranking is overall extraction quality against the visible PDF. On `gridofimages`, the tie for first is only among these three candidates: all three omit the photographs themselves and therefore none is content-complete for the page as a visual document.

## 1. `text_dense__tableofcontent`

### Direct candidate-to-PDF findings

The visible PDF is a two-column directory, read down the complete left column and then down the complete right column. It contains page number `4`, header `Fourth Quarter 2002`, and 134 name/extension records. Accents and apostrophes render correctly, while the embedded font's broken ToUnicode map yields systematic native-extraction substitutions.

- **`pymupdf_text` and `pypdf_baseline` are tied first.** Both contain all directory records in the correct column order. Their substantive defect is the same source-driven character decoding failure: examples include `AndrÈ` for `André`, `Líhon` for `L'hon`, `LÈger` for `Léger`, `AurËle` for `Aurèle`, `OíBrien` for `O'Brien`, and other bad accented characters. These are real candidate text errors against the rendered PDF, although they are explainable by the malformed source CMap. `pymupdf_text` puts `4` and `Fourth Quarter 2002` on separate lines; `pypdf_baseline` joins them as `4 Fourth Quarter 2002`. This is not a meaningful overall-quality difference.
- **`pdf2md_local` is decisively third.** Its output begins with a large unrelated HTML table containing fragments about the Companies Act 2006, wage deciles, Ireland's tax regime, and a skilled-talent pool. None of that exists in this PDF. It then retains only two genuine directory fragments (roughly `Makonnen` through `Matte`, and `Robson` through `Singh`) while losing the large majority of the page. This is severe cross-document content contamination plus massive omission, not merely imperfect table recognition.

### Per-rule detail

| candidate | sentence % | unexpected sentence % | excess sentence % | word % | unexpected word % | excess word % | digits % | specific sentences | specific words | order | header |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pdf2md_local | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 0/1 | 0/1 | 22/148 | 59/289 | 9/61 | 0/1 |
| pymupdf_text | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1/1 | 133/148 | 284/289 | 61/61 | 0/1 |
| pypdf_baseline | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1/1 | 133/148 | 284/289 | 61/61 | 0/1 |

For both native baselines, the 15 failed specific-sentence rules are exactly accent/apostrophe-sensitive fragments such as `Serré, Benoît`, `O'Reilly`, `O'Brien`, `O'Connor`, `François`, and several `L'hon` forms. The five failed specific-word rules are `benoît`, `brien`, `connor`, `françois`, and `reilly`. This supports the PDF-grounded diagnosis rather than suggesting missing names or bad order.

### Are the scalars and ranking sensible?

- **Relative rank: sensible.** The two complete native baselines tie far above the contaminated, mostly missing local output.
- **Order: sensible.** `1.0/1.0/0.147541` correctly distinguishes the complete directory order from the local corruption.
- **Correctness/faithfulness: directionally sensible but too charitable to `pdf2md_local`.** A candidate dominated by unrelated content and retaining only small page fragments still receives `correct=0.596757` and `faith=0.447018`.
- **Native-baseline scores below 1 are defensible against the rendered PDF**, because the Unicode characters are genuinely wrong in their outputs. They should, however, be interpreted as source-CMap susceptibility rather than ordinary parser omission/order failure.

### Reference/rule anomalies

- The reference correctly uses the visually rendered Unicode (`André`, `L'hon`, `François`, etc.), which is preferable to copying the broken text layer. This is not a reference error.
- Sentence rules are mechanically fragmented around initials and punctuation, producing odd cross-record expectations such as `3278 Gervais, Jean-Pierre 3163` and `3540 Graham, Alan R`. This makes the exact-rule layer brittle and hard to interpret, though it does not reverse the ranking here.
- The `is_header` rule requires semantic header markup and all three fail it; this says little about text extraction fidelity.

## 2. `text_misc__dash`

### Direct candidate-to-PDF findings

The visible source genuinely consists mostly of literal dash runs arranged as a nested bulleted list, followed by two prose/dash paragraphs and a bottom footer. The dashes are source redactions, not OCR loss.

- **`pymupdf_text` and `pypdf_baseline` tie first.** Both preserve all legible prose, literal dash runs, list symbols, and sequence. They retain line wrapping and indentation far better than `pdf2md_local`. Their main visual-order defect is that the bottom footer (`USAble Mutual Insurance Company`, page `11`, filing title) is emitted first because of PDF object order rather than last where it appears visually. `pymupdf_text` often puts a bullet marker on its own line; `pypdf_baseline` generally keeps it with the dash run, a minor presentation advantage that is insufficient to break the overall tie.
- **`pdf2md_local` is third.** It retains the substantive prose and dash characters and correctly leaves the footer at the end, but it collapses nested bullets, continuations, and several adjacent dash runs into long concatenated lines (`...o ...o...•...`). The visual hierarchy and line boundaries become substantially less usable. This is a structure/order defect, not missing redacted information.

### Per-rule detail

| candidate | sentence % | unexpected sentence % | excess sentence % | word % | unexpected word % | excess word % | digits % | specific sentences | specific words | order | footer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pdf2md_local | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 9/10 | 62/62 | 4/25 | 0/1 |
| pymupdf_text | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 9/10 | 62/62 | 3/25 | 0/1 |
| pypdf_baseline | 0/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 9/10 | 62/62 | 3/25 | 0/1 |

All three fail the same specific sentence, `2021 Off-Exchange Small Group Rate Filing 11`, because the PDF/footer layout has the title and company at lower left and page `11` at lower right rather than that literal inline ordering. All 62 specific words pass.

### Are the scalars and ranking sensible?

- **Correctness rank is sensible:** the two line-faithful baselines score `0.944898`, ahead of local at `0.914286`.
- **Headline score/order rank is not sensible:** local is ranked first solely because order is `0.16` versus `0.12`, despite visibly destroying much more of the nested list structure. The 0.04 difference comes from brittle exact matching of artificial dash tokens, not a credible reading-order judgment.
- **Absolute order values are implausible for all three.** The page sequence is overwhelmingly preserved, yet 21 or 22 of 25 order rules fail.

### Reference/rule anomalies

- Order rules wrap every expected dash run in square brackets, e.g. `[------------------]`, even though the PDF contains literal dashes with no brackets. Candidate outputs faithfully containing the source dashes cannot match those order anchors exactly.
- Several order anchors combine prose with bracketed runs across visual line wraps. These artificial strings, rather than actual PDF text spans, explain the near-zero order scores.
- The footer-specific sentence and `is_footer` rule require a synthetic linear/semantic representation (`<page_number>11</page_number>`). All candidates fail despite extracting all three footer components. That is a representation mismatch, not missing content.

## 3. `text_misc__gridofimages`

### Direct candidate-to-PDF findings

The source is a 5-column by 10-row contact sheet: 50 raster photographs plus captions `053_20090718白山祭り042` through `102_20090718白山祭り091`. Correct caption order is row-major.

- **`pymupdf_text` and `pypdf_baseline` tie first among these candidates.** Both recover all 50 captions exactly and in correct row-major order. `pypdf_baseline` adds harmless leading spaces to columns 2–5. Neither represents or describes any of the 50 photographs, so both are text-correct but visually content-incomplete.
- **`pdf2md_local` is third.** It emits captions in column-major chunks (`053, 058, 063, ...`, then `054, 059, ...`) rather than row-major order, invents two malformed HTML tables around captions, and corrupts captions 061 and 066 into split cell fragments (`12009` / `0718...050`, `62009` / `0718...055`). Its later captions 072 and 077 are likewise split across cells, although their components remain recognizable. It also omits any representation of the photographs.

### Per-rule detail

| candidate | sentence % | unexpected sentence % | excess sentence % | word % | unexpected word % | excess word % | digits % | specific sentences | specific words | order rules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pdf2md_local | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 1/1 | 0/50 | 37/100 | none | 
| pymupdf_text | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0/50 | 39/100 | none |
| pypdf_baseline | 0/1 | 1/1 | 1/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0/50 | 39/100 | none |

### Are the scalars and ranking sensible?

- **Relative score rank is weakly sensible:** the complete row-major baselines narrowly beat local.
- **The absolute scores are not sensible:** exact, complete caption extraction receives only `0.659550`, 0/50 specific-sentence passes, and 39/100 specific-word passes.
- **There is no order scalar at all.** Consequently the evaluator has no direct signal for the largest text-extraction defect: local's column-major ordering versus the baselines' correct row-major order.
- **All scores ignore the photographs.** The metric can only judge captions, so `0.659550` must not be read as 66% page-content recovery; all candidates omit the page's dominant information content.

### Reference/rule anomalies

- Reference sentence strings replace the literal underscore with a space (`053 20090718白山祭り042`), whereas candidates faithfully output the visible caption `053_20090718白山祭り042`. The exact sentence matcher then fails all 50 correct baseline captions. This is an asymmetric normalization/rule-generation defect.
- The same tokenization makes the 100 word rules and digit rule unreliable for Japanese filename-style captions.
- No order rules were generated for a page whose sequence is explicit and visually important. This omission hides local's major ordering failure.
- The reference covers only captions; it has no rule for retaining images or producing image placeholders/alt content.

## 4. `text_misc__notATable`

### Direct candidate-to-PDF findings

The PDF visibly contains exactly 20 plain pipe-delimited records, one per line, in a single top-to-bottom sequence. It is not a ruled or aligned table.

- **`pymupdf_text` and `pypdf_baseline` tie first.** Both recover all 20 records, every pipe and numeric field, one record per line, in exact visual order. Their outputs differ only by inconsequential trailing whitespace.
- **`pdf2md_local` is decisively third.** It initially contains all 20 records and their fields but flattens them into one giant line, losing record boundaries. It then appends a large unrelated malformed HTML table containing M&A/interest-rate prose and veterinary adverse-reaction text. None of that appears in the PDF. This is another severe cross-document contamination event.

### Per-rule detail

| candidate | sentence % | unexpected sentence % | excess sentence % | word % | unexpected word % | excess word % | digits % | specific sentences | specific words | order |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pdf2md_local | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/20 | 81/81 | 0/19 |
| pymupdf_text | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 | 20/20 | 81/81 | 0/19 |
| pypdf_baseline | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 | 0/1 | 1/1 | 20/20 | 81/81 | 0/19 |

### Are the scalars and ranking sensible?

- **No. The headline ranking is reversed.** The contaminated, line-destroying local output scores `0.428571`, above both exact baselines at `0.416667`.
- **Correctness is also reversed:** local `0.857143` versus exact baselines `0.833333`.
- **Order is nonsensical:** all three receive `0.0`, even though both baselines preserve all 20 records in exact source order. Local's 0 is appropriate, but for flattening—not because the record sequence itself was permuted.
- The broad unexpected-sentence and unexpected-word rules pass even for local's large unrelated HTML table, showing that the evaluator is effectively blind to major foreign-content injection on this example.

### Reference/rule anomalies

- Reference sentence/order strings replace every literal pipe with a space (`T TJEFF 202508 ...`). Baseline candidates correctly preserve `T|TJEFF|202508|...`. Specific-sentence rules apparently normalize enough to pass all 20, but order matching does not: the same exact records fail all 19 adjacency rules. This is an internally inconsistent normalization path.
- The broad `too_many_*` rules penalize the exact baselines but not the contaminated local candidate, directly producing a perverse correctness advantage for the worse output.
- Treating a literal delimited record as an ordinary sentence/word bag loses the one structure that matters here: exact line boundaries plus exact delimiter retention.

## Cross-case conclusions

1. **Manual overall ranking:**
   - `tableofcontent`: `pymupdf_text = pypdf_baseline` > `pdf2md_local`.
   - `dash`: `pymupdf_text = pypdf_baseline` > `pdf2md_local` (local alone places the footer correctly at the end, but its list flattening is materially worse).
   - `gridofimages`: `pymupdf_text = pypdf_baseline` > `pdf2md_local`; all three remain image-content-incomplete.
   - `notATable`: `pymupdf_text = pypdf_baseline` > `pdf2md_local`.
2. **`pdf2md_local` shows two severe foreign-content contamination failures** (`tableofcontent`, `notATable`) and two major structural/order failures (`dash`, `gridofimages`). Its headline score nevertheless wins `dash` and `notATable`, so the scalar ranking is not dependable.
3. **The native baselines are highly reliable on these four text layers.** Their only substantive content defect is the known broken-CMap Unicode rendering on `tableofcontent`; they otherwise preserve source text and order. They do not recover photographic semantics on `gridofimages`.
4. **Exact-rule normalization is inconsistent.** Underscores and pipes are changed to spaces in references without equivalent matching in all rule paths; dash runs acquire nonexistent square brackets; sentence segmentation crosses record boundaries; semantic header/footer markup is required from plain-text candidates.
5. **Order coverage is especially defective:** correct outputs score 0.12 on the simple dash page, 0.0 on the exact 20-line record dump, and no order score exists at all for the explicitly ordered image grid.
6. **Absolute correctness/content scores should not be interpreted as proportions of visible PDF content.** The grid metric ignores all 50 photos, and the notATable metric gives a contaminated output a higher correctness score than exact outputs.
7. **The retained per-rule CSV artifact is missing in practice:** all three `_evaluation_rule_results.csv` files are empty. The HTML-embedded details are sufficient for this audit but are a retention/reporting anomaly that makes automated downstream verification harder.
