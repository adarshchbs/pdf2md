# Exact-metric verification B

## Scope, provenance, and method

I first read the frozen source audit at
`/private/tmp/claude-501/-Users-adarsh-project-pdf2md/1c16afff-29c0-4394-8e54-7ccb74cc0ca2/scratchpad/exact-metric-audit-b.md` and recorded its SHA-256 as:

`4e192cbaa89fd5642d828ae6b3f15e639b9f87ed7d74f3db6ca898887a257c0f`

I did not modify it. I then independently opened each frozen PDF at
`/Users/adarsh/project/pdf2md/benchmarks/datasets/parsebench-full/docs/text/` with PyMuPDF, extracted its native page text, and compared that text and its visual line order with each retained candidate's `output.markdown` in
`/Users/adarsh/project/pdf2md/benchmarks/runs/parsebench-full/{pipeline}/{pipeline}/text/*.result.json`. I inspected per-example metric metadata and every failed rule in each pipeline's `_evaluation_report.json`. Expected Markdown was consulted only after PDF-grounded comparison, to diagnose reference/rule anomalies; it was not treated as truth.

All five PDFs independently agree with the frozen audit: they are one-page, born-digital files whose visible text is faithfully and unambiguously returned by native PyMuPDF extraction. For all five, `pymupdf_text` is byte-for-byte identical to direct native PDF text extraction.

Score columns below are `normalized_text_score / normalized_order / normalized_text_correctness`; `RPR` is `rule_pass_rate`. A dash means the evaluator emitted no order scalar.

## Score overview and score-produced ranking

| Document | pdf2md_local | pymupdf_text | pypdf_baseline | Scalar ranking |
|---|---:|---:|---:|---|
| gridofnumbers | 0.925854 / 0.914286 / 0.937422; RPR 0.960978 | 1.000000 / 1.000000 / 1.000000; RPR 0.997297 | 1.000000 / 1.000000 / 1.000000; RPR 0.997297 | pymupdf = pypdf > local |
| japan2 | 0.484564 / 0.312500 / 0.656629; RPR 0.525863 | 0.893172 / 1.000000 / 0.786344; RPR 0.764797 | 0.763495 / 0.750000 / 0.776990; RPR 0.726782 | pymupdf > pypdf > local |
| unknonw | 0.757277 / 1.000000 / 0.514553; RPR 0.121436 | 0.857805 / 1.000000 / 0.715611; RPR 0.150830 | 0.857805 / 1.000000 / 0.715611; RPR 0.150830 | pymupdf = pypdf > local |
| clilogs | 0.957842 / 1.000000 / 0.915684; RPR 0.979416 | 0.968968 / 1.000000 / 0.937937; RPR 0.980115 | 0.968333 / 1.000000 / 0.936665; RPR 0.980075 | pymupdf > pypdf > local |
| concession | 1.000000 / — / 1.000000; RPR 1.000000 | 1.000000 / — / 1.000000; RPR 1.000000 | 0.935265 / — / 0.935265; RPR 0.989211 | pymupdf = local > pypdf |

Where order exists, the combined text score is the mean of order and correctness. The absence of an order score for `concession` is an evaluator/rule-coverage omission, not evidence that order is unknowable.

## 1. text_multicolumns__gridofnumbers

### PDF-grounded candidate defects

- **pymupdf_text:** exact. Its 3,180 characters and 289 extracted lines are byte-identical to direct native PDF extraction. It preserves all 288 numeric codes in visible row-major order and the final `Page 3 of 8` footer.
- **pypdf_baseline:** all codes and footer are correct and remain in the correct token order, but all 289 visual/native lines are collapsed into one line. It therefore loses the 8-column by approximately 36-row grid structure. This is a presentation/structure defect, not a scalar-content or token-order defect.
- **pdf2md_local:** serious corruption in the middle of an otherwise ordered code stream. It omits exactly eight visible codes: `20121579`, `20121581`, `20121587`, `20121588`, `20121599`, `20121603`, `20121604`, and `20121605`. In their location it inserts an unrelated, nonexistent Markdown table headed `Nationwide | Texas Pre-RS`, with rows `3034/3035 Limited | 2,168 | 1,458` and `3034/3035 Lifetime | 3,035 | 2,278`. No such table or values exist anywhere in the PDF. It also flattens the actual grid into a few long paragraphs rather than preserving rows.

### Rule details and anomalies

- `pdf2md_local` fails 19/370 rules: the aggregate missing-sentence rule scores 0.945946, unexpected-sentence 0.700000, unexpected-word 0.952703, missing-word 0.972414, and bag-of-digit 0.990893; eight `missing_specific_word`, two `missing_specific_sentence`, three `order`, and one `is_footer` rule also fail. These failures correctly expose the eight omissions and hallucinated table. The three order failures are anchor-not-found consequences of the omissions, rather than independent evidence of global reordering.
- Both exact-content candidates fail only the same `is_footer` rule, which demands literal annotation markup `Page <page_number>3</page_number> of 8` inside `<page_footer>...</page_footer>`. The visible footer text is correctly extracted by both. This is a structural-markup requirement inappropriate to plain-text pipelines; it explains RPR 0.997297 rather than 1.0 and is not a content defect.

### Judgment and manual rank

The scalar content/order ranking is sensible for its narrow remit: pymupdf and pypdf tie because both have complete, correct token sequences; local is lower. The local score of 0.925854 is somewhat generous for an eight-code omission plus an invented table, but the defect affects a small fraction of 288 codes. The scalar tie does not capture pypdf's total loss of grid row structure.

**Manual overall rank:** 1 pymupdf_text; 2 pypdf_baseline; 3 pdf2md_local. **No tie** once visible structure is considered. For content and token order alone, pymupdf_text and pypdf_baseline legitimately tie.

## 2. text_multilang__japan2

### PDF-grounded candidate defects

- **pymupdf_text:** exact, byte-for-byte (1,224 characters, 38 lines), including Japanese characters, full-width article/form numbers, punctuation, spaces, and visual line wraps.
- **pypdf_baseline:** complete and correctly ordered. Its only differences from native PDF text are 12 inserted spaces, principally after clause markers and around parentheses; no substantive character is added, omitted, or changed.
- **pdf2md_local:** complete and correctly ordered at the semantic character level, but normalizes full-width digits and parentheses to ASCII (for example `第１１条` to `第11条` and `（実績報告）` to `(実績報告)`), removes some source spaces, and turns visual wraps/paragraph boundaries into a 69-line Markdown layout with extra blank lines. After Unicode NFKC normalization and removal of whitespace/punctuation, all three candidates are exactly identical to the PDF content. Thus local's defects are typography and segmentation, not missing Japanese prose or reordered clauses.

### Rule/reference anomalies

An exact native extraction fails 35/147 rules and receives correctness only 0.786344. Its failed aggregate rules are: missing-sentence 0.653061, unexpected-sentence 0.946429, unexpected-word 0.530435, too-many-word-occurrence 0.704082, and missing-word 0.884058, plus 13 `missing_specific_word` and 17 `missing_specific_sentence` failures.

These are demonstrably anomalous against the PDF:

- Rules label genuine visible tokens such as `条`, `条第`, and full-width `１０`, `１１`, `１２` as unexpected.
- Occurrence limits are false for the source: for example, `間接補助事業者` is allowed once although it visibly occurs ten times; `事務局に提出しなければならない`, `交付すべき補助金の額を確定し`, and `実績報告` are likewise assigned limits below their authored occurrences.
- Several supposedly missing phrases are present but cross a native visual wrap, such as `事務局は期\n限について猶予することができる`; the evaluator then reports the wrapped form as unexpected while demanding an unwrapped anchor. Similar punctuation/space normalization defects affect parenthesized clauses.
- `pypdf_baseline` receives order 0.75 from four anchor-not-found rules around article 13 and clause (4), despite preserving the same sequence as the PDF; its inserted spaces break overly strict anchors. `pdf2md_local` receives order 0.3125 from eleven anchor failures caused by ASCII/full-width and punctuation normalization, not actual reordering.

### Judgment and manual rank

The score ranking direction is sensible—exact native extraction first, pypdf's minor spacing changes second, local's broader typographic/layout normalization third—but the magnitudes are not. In particular, 0.786 correctness for byte-exact text and 0.75 order for text with only inserted spaces are unsound. Local's 0.656629 correctness also greatly overstates semantic loss: its normalized substantive character stream is exact.

**Manual overall rank:** 1 pymupdf_text; 2 pypdf_baseline; 3 pdf2md_local. **No tie.**

## 3. text_multilang__unknonw

### PDF-grounded candidate defects

- **pymupdf_text:** exact, byte-for-byte (2,085 characters, 41 lines), including Thai combining marks, numerals, Latin tokens, and wraps.
- **pypdf_baseline:** complete and correctly ordered; it differs only by deleting three newline characters (two blank separators before numbered items and the final newline). All substantive characters are exact.
- **pdf2md_local:** all substantive characters are exact and in order, but it collapses the entire 41-line report to one line. This destroys headings, numbered-activity separation, bullets, and paragraph boundaries. Under Unicode normalization with whitespace/punctuation removed, it is exactly equal to the PDF.

### Rule/reference anomalies

The exact native extraction fails 216 rules (210 `missing_specific_word`, two `missing_specific_sentence`, and four aggregate rules), producing an implausible RPR of 0.150830 and correctness of 0.715611. Aggregate scores are missing-sentence 0.818182, unexpected-sentence 0.923077, unexpected-word 0.214286, and missing-word 0.053731.

The failures reveal broken Thai rule construction/tokenization rather than extraction defects:

- The evaluator demands fragments with vowels/diacritics stripped or split (for example missing fragments `กศ`, `กษา`, `จกรรม`, `ทยาล`, `ยราชภ`) while marking the correctly rendered Thai words as unexpected.
- A reference glyph disagrees with the visible PDF: the rule demands `กิจกรรม วันราชภัฏ` (normalized explanation `กจกรรม วนราชภฏ`), while the PDF and exact native extraction contain `กิจกรรม วันราชภัฎ` (`...ภฎ`).
- Long sentence anchors fail at source line wraps and combining-mark boundaries. The same correctly authored text is consequently counted on both missing and unexpected sides.

`pdf2md_local` fails 222 rules and has correctness 0.514553. Its lower score relative to the exact candidates is directionally appropriate because total line flattening damages structure, but most of the absolute penalty shared by all three comes from malformed Thai rules. All candidates receive order 1.0; that is correct for token sequence but fails to measure local's erased line/list structure. Pymupdf and pypdf receive identical scalars despite pypdf's three newline deletions; that is acceptable for content/order but not exact layout fidelity.

**Manual overall rank:** 1 pymupdf_text; 2 pypdf_baseline; 3 pdf2md_local. **No tie** under exact visual-text fidelity; pymupdf_text and pypdf_baseline are effectively tied for substantive content and token order.

## 4. text_simple__clilogs

### PDF-grounded candidate defects

- **pymupdf_text:** exact, byte-for-byte (2,118 characters, 88 lines). It preserves each `COMMENT`, `FOLDER`, and `FILE` line and all paths.
- **pypdf_baseline:** all substantive characters and their order are exact, but it removes 31 line breaks, mostly joining a standalone `COMMENT` marker to the following line. It retains 58 lines, so much of the log remains usable, but record/comment formatting is degraded.
- **pdf2md_local:** all substantive characters and order are exact, but all 88 native lines are flattened into one line. This destroys the defining line-record structure of the CLI/log document.

### Rule/reference anomalies

Even the byte-exact extraction fails ten rules and scores only 0.937937 correctness. Its aggregate failures are missing-sentence 0.923077, unexpected-sentence 0.975000, too-many-sentence-occurrence 0.693333, unexpected-word 0.993443, missing-word 0.993443, and bag-of-digit 0.987261, plus one `missing_specific_word` and three `missing_specific_sentence` failures.

The clearest source/reference error is `O365` versus `0365`: the PDF visibly and natively contains `C:\Users\deand\Indiana University\O365-[Sec] IN-ULIB-` twice (capital letter O), while rules require `0365` (zero), report correct `o365` as unexpected, and claim two missing zeros. Occurrence rules are also false: they cap `COMMENT` at four although the PDF contains 26, and cap `karenware` at one although it appears twice. Sentence rules demand line-joined/header forms inconsistent with the visible monospaced line layout.

The scalar ranking direction is sensible but compressed: exact pymupdf narrowly beats partially flattened pypdf, which narrowly beats fully flattened local. A 0.011126 gap between exact 88-line output and a single flattened line substantially underweights this document's core record structure. Order 1.0 is defensible for token order but misleading as a complete reading-order/structure judgment.

**Manual overall rank:** 1 pymupdf_text; 2 pypdf_baseline; 3 pdf2md_local. **No tie.**

## 5. text_simple__concession

### PDF-grounded candidate defects

- **pymupdf_text:** exact, byte-for-byte (219 characters, three visible wrapped lines). No text is missing from the photographic page.
- **pdf2md_local:** all caption words and their order are exact. It removes trailing spaces/final newline and places blank lines between the three visual wraps, which incorrectly represents one wrapped caption as three Markdown paragraphs. This is minor compared with a content error but is not exact structural fidelity.
- **pypdf_baseline:** correctly extracts the entire caption, then hallucinates/appends `Chapter 1: Getting the foundation right`, which does not occur anywhere in the PDF image or native text.

### Rule details and anomalies

- pymupdf and local pass every available rule and both score 1.0. Treating local's whitespace/paragraph segmentation as content-equivalent is reasonable for a content scalar, though the evaluator has no order/paragraph rule to distinguish it from exact output.
- pypdf correctly fails unexpected-sentence (0.750000), unexpected-word (0.891892), and too-many-word-occurrence (0.969697, because invented text adds another `the`). Its RPR is 0.989211 and correctness/text score 0.935265. These failures correctly identify the hallucinated chapter. The final score is somewhat generous for appending a wholly nonexistent sentence to a caption this short, but its direction and diagnosis are sound.
- The source's grammatical `A concessions` is authored in the PDF and correctly retained by all candidates; it must not be “corrected” or scored as an extraction defect.

**Manual overall rank:** 1 pymupdf_text; 2 pdf2md_local; 3 pypdf_baseline. **No tie** under exact fidelity. For content words/order alone, pymupdf_text and pdf2md_local legitimately tie.

## Final assessment

### Manual ranks across all five

| Document | Manual rank (overall PDF fidelity) | Manual content/order ties |
|---|---|---|
| gridofnumbers | pymupdf_text > pypdf_baseline > pdf2md_local | pymupdf_text = pypdf_baseline for complete token content/order |
| japan2 | pymupdf_text > pypdf_baseline > pdf2md_local | none |
| unknonw | pymupdf_text > pypdf_baseline > pdf2md_local | pymupdf_text ≈ pypdf_baseline for substantive content/order |
| clilogs | pymupdf_text > pypdf_baseline > pdf2md_local | none |
| concession | pymupdf_text > pdf2md_local > pypdf_baseline | pymupdf_text = pdf2md_local for content words/order |

The scalar rankings are mostly directionally sensible, but exact-score validity is poor on three references: Japanese, Thai, and CLI logs. Byte-exact native extraction scores only 0.786344, 0.715611, and 0.937937 correctness respectively because of malformed references, strict wrap/spacing anchors, broken Thai tokenization/diacritic handling, false occurrence limits, and `O365`→`0365` reference corruption. Grid content/order scalars work, apart from a footer-markup rule and omission of grid structure. Concession rules correctly catch pypdf's hallucination but omit paragraph/order fidelity. These cases should not be used to claim exact absolute correctness without repairing or disabling the anomalous rules.