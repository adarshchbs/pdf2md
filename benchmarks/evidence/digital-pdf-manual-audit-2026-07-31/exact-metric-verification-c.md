# Exact-metric verification C

## Scope and integrity

Verification only; no repository files were changed. I inspected the retained outputs and per-rule evaluator records under `/Users/adarsh/project/pdf2md/benchmarks/runs/parsebench-full/`, the five PDFs under `/Users/adarsh/project/pdf2md/benchmarks/datasets/parsebench-full/docs/text/`, native text/geometry and the frozen page renders. Expected Markdown was inspected only to diagnose reference/rule anomalies; it was not treated as truth.

Frozen source audit (read before any candidate/evaluator artifact and never modified):

- Path: `/private/tmp/claude-501/-Users-adarsh-project-pdf2md/1c16afff-29c0-4394-8e54-7ccb74cc0ca2/scratchpad/exact-metric-audit-c.md`
- SHA-256 recorded immediately after reading: `a2d53b82a6d4e2eab568fc4ec9d7ea7219838589aa3d881fbd39ca543aa93dd4`

Score notation below: **C** = `content_faithfulness`, **O** = `normalized_order`, **K** = `normalized_text_correctness`, **T** = `normalized_text_score`, **R** = `rule_pass_rate`. `N/A` means the evaluator emitted no order scalar because the reference supplied no order rules.

## Score summary and manual ranks

| PDF | Candidate | C | O | K | T | R | Manual rank |
|---|---|---:|---:|---:|---:|---:|---|
| `text_simple__name_list` | pdf2md_local | 0.804782 | 0.969231 | 0.722558 | 0.845894 | 0.693538 | 2 |
|  | pymupdf_text | 0.867097 | 0.969231 | 0.816030 | 0.892630 | 0.695212 | =1 |
|  | pypdf_baseline | 0.867097 | 0.969231 | 0.816030 | 0.892630 | 0.695212 | =1 |
| `text_simple__returns` | pdf2md_local | 0.912708 | 0.833333 | 0.952395 | 0.892864 | 0.968175 | 2 |
|  | pymupdf_text | 0.923965 | 0.833333 | 0.969280 | 0.901307 | 0.958718 | 1 |
|  | pypdf_baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3 |
| `text_sparse__blank` | pdf2md_local | 1.000000 | N/A | 1.000000 | 1.000000 | 0.888889 | =1 |
|  | pymupdf_text | 1.000000 | N/A | 1.000000 | 1.000000 | 0.888889 | =1 |
|  | pypdf_baseline | 1.000000 | N/A | 1.000000 | 1.000000 | 0.888889 | =1 |
| `text_sparse__ikea` | pdf2md_local | 0.892857 | N/A | 0.892857 | 0.892857 | 0.923913 | 1 |
|  | pymupdf_text | 0.964286 | N/A | 0.964286 | 0.964286 | 0.945652 | 2 |
|  | pypdf_baseline | 0.819841 | N/A | 0.819841 | 0.819841 | 0.814734 | 3 |
| `text_sparse__numbers` | pdf2md_local | 1.000000 | N/A | 1.000000 | 1.000000 | 1.000000 | 2 |
|  | pymupdf_text | 1.000000 | N/A | 1.000000 | 1.000000 | 1.000000 | =1 |
|  | pypdf_baseline | 1.000000 | N/A | 1.000000 | 1.000000 | 1.000000 | =1 |

Manual ranks assess visible scalar text, its reading order, and preservation of meaningful line boundaries. They do not reward invented image descriptions.

## 1. `text_simple__name_list`

### Frozen PDF evidence

The render shows 66 centered list entries in top-to-bottom order and page number `VIII` at bottom right. One entry visibly ends in English `Turkey` (Demirali Y. ERGİN); the others use `Türkiye`.

There is an important inconsistency in the frozen audit itself: it correctly locates `VIII` at bottom right but also says native extraction is exactly visual top-to-bottom. Directly inspecting the PDF and retained native output shows that PyMuPDF emits `VIII` **first**, before the list. The render is decisive: visual order places it last.

### Candidate defects

- **pdf2md_local:** Contains every visible lexical token and puts `VIII` at the end, which is the correct visual position. Its defect is structural: it frequently joins multiple distinct list entries onto one Markdown line/paragraph. This damages the page's central one-entry-per-line list structure even though the entries remain in sequence.
- **pymupdf_text:** Preserves every list entry and every line exactly, including Turkish diacritics and the visible `Turkey`. Its sole substantive order defect is placing bottom-right `VIII` first.
- **pypdf_baseline:** Effectively identical to PyMuPDF here (apart from terminal whitespace): complete one-entry-per-line list, with `VIII` incorrectly first.

Manual rank is therefore **pymupdf_text = pypdf_baseline > pdf2md_local**: preserving all 66 item boundaries outweighs the one misplaced footer token. If scoring only lexical token sequence and treating line boundaries as irrelevant, pdf2md_local would instead have the best visual order because it correctly moves `VIII` to the end.

### Are the metrics/ranking sensible?

Only partly.

- The evaluator rank `pymupdf_text = pypdf_baseline > pdf2md_local` agrees with the manual rank, and the lower pdf2md_local correctness score reacts to its line/paragraph joining.
- **O = 0.969231 for all three is not discriminative.** It checks 65 adjacent-name anchors, not the footer's relation to the list or item boundaries. The two lost order rules are also contaminated by the `Turkey`/`Türkiye` mismatch. Thus it neither rewards pdf2md_local's correct footer placement nor penalizes native baselines for prepending `VIII`.
- The absolute correctness values are badly undercalibrated. An output exactly matching all native visible text receives only **K = 0.816030** and **R = 0.695212**. pdf2md_local receives **K = 0.722558**, much lower than its near-perfect lexical content warrants.

### Reference/rule anomalies

1. The reference changes the visibly printed `Turkey` to `Türkiye`. Exact native candidates are consequently charged for an unexpected `Turkey`, a missing `Türkiye`, and associated order/anchor failures.
2. Sentence rules are pathological for the repeated abbreviation `Yrd. Doç. Dr.`. The reference bag contains `Yrd Doç` 66 times plus separate `Dr <name>...` fragments, yet the exact native output gets `missing_sentence_percent = 0.007463` and 68 failed `missing_specific_sentence` rules. This is an evaluator sentence-boundary/normalization failure, not missing PDF text.
3. Turkish uppercase/dotted-I normalization generates invalid word anchors such as `ali`, `aydemi`, `ayteki`, `brahim`, `buyuki`, `demi`, `efi`, `ei`, `ergi`, etc. Exact native output fails 30 `missing_specific_word` rules, while word aggregates report false missing and unexpected fragments. This is a tokenizer/case-fold anomaly.
4. All candidates fail `is_footer` solely because none emits the benchmark-specific `<page_footer>` wrapper; that structural failure is reasonable as a markup check but says nothing about whether `VIII` was extracted or positioned correctly.

## 2. `text_simple__returns`

### Frozen PDF evidence

The render is an IRS Schedule O page. The visual sequence is: top header (`Schedule O ... Page 2`, organization/name and EIN fields), narrative body, state list, then bottom footer (`132212 11-11-21` at left and `Schedule O ... 2021` at right). All body text is born-digital and unambiguous. Native PyMuPDF contains every token but emits compact header/footer cells in nonvisual object order; its body is exact and top-to-bottom.

### Candidate defects

- **pdf2md_local:** Extracts all visible content and reconstructs the organization/EIN pair as a table. However, it emits `RECORDS ... MOST CLASSES ARE`, then inserts that header table, then resumes `TAUGHT BY ...`. This puts header fields inside the first substantive sentence and breaks that sentence. The footer is correctly moved after the body, but it is not marked as a footer.
- **pymupdf_text:** The entire narrative body and state list are verbatim and in correct order. The initial cell sequence is nonvisual (`132212 11-11-21`, `2`, `Employer identification number`, two Schedule O strings, `Page`, `Name...`, `NEADS, INC.`, EIN). Thus the defect is limited to header/footer field ordering, not substantive body content.
- **pypdf_baseline:** There is **no retained `.raw.json` or `.result.json` for this PDF**. The evaluator nevertheless records a successful example with “No markdown content provided” for every rule and assigns all five scalars/R as zero. It is a complete output failure for this case.

Manual rank is **pymupdf_text > pdf2md_local > pypdf_baseline**. Keeping the substantive narrative intact is more important than reconstructing some header geometry while splitting a body sentence around it.

### Are the metrics/ranking sensible?

Mostly at the ranking level, but not fully at the rule level.

- The scalar rank (`T`: 0.901307 > 0.892864 > 0) matches manual judgment.
- Tying both nonempty candidates at **O = 0.833333** is arithmetically explainable (each fails 3 of 18 order links) but hides severity. pdf2md_local fails the header-to-body link and both links around the split first sentence; PyMuPDF fails three compact header/footer-cell links while preserving every body link. The same scalar therefore represents qualitatively different damage.
- The slight PyMuPDF content advantage is sensible because its body is pristine.
- pypdf_baseline's zero score is sensible for an absent output, but reporting `success: true` and silently evaluating empty Markdown is not. This should be a parse/retention failure, not an ordinary successful candidate result.
- pdf2md_local's rule details claim the organization/EIN header words occur three times (`23`, `7281887`, `Employer`, `Identification`, `Inc`, etc.) although the retained Markdown contains the header table once. That points to duplicate internal table-text accounting or another evaluator representation mismatch; it should not be interpreted as candidate hallucination without tracing the evaluator's text view.

The expected reference's visual header/body/footer order is well grounded in the PDF here. The failures caused by PyMuPDF's nonvisual cell order are real. The mandatory `<page_header>`/`<page_footer>` failures are markup-specific rather than extraction-content defects.

## 3. `text_sparse__blank`

### Frozen PDF evidence

The page truly has no body. It contains only top header `vhdl-style-guide Documentation, Release 3.0.0` and bottom footer with `8` at left and `Chapter 3. Installation` at right.

### Candidate defects

- **pdf2md_local:** Complete and ordered: header, then `8 Chapter 3. Installation` on one line.
- **pymupdf_text:** Complete and ordered: header, then `8`, then `Chapter 3. Installation` on separate lines.
- **pypdf_baseline:** Complete and ordered, with the two footer fields on one line.

All are defensible Markdown renderings of the same sparse page. Manual rank is a **three-way tie**.

### Are the metrics/ranking sensible?

Yes for scalar text content: **C = K = T = 1** for all three. `O = N/A` is expected because no order rules were authored, although it means order is assumed rather than measured. **R = 0.888889** for all because each fails `is_header` and `is_footer`; those failures only indicate absence of benchmark-specific wrappers. The reference text and visible digit bag are accurate.

## 4. `text_sparse__ikea`

### Frozen PDF evidence

Visual reading order is top-left `Everyday IKEA`, top-right `Extra images – Living with children`, then photograph codes left-to-right `PH134503`, `PH134740`, `PH134501`. The page's dominant content is three photographs; the five strings above are all of its actual text. Native extraction is complete but orders codes first and then the two headers in reverse (`Extra...`, `Everyday IKEA`).

### Candidate defects

- **pdf2md_local:** Has all five visible strings. It sensibly places the header text before the image codes, but reverses the two header fields (`Extra... Everyday IKEA`). Codes are correctly left-to-right.
- **pymupdf_text:** Has all five strings, but emits all codes before the header and reverses the header fields. It therefore has worse visual reading order than pdf2md_local despite a higher scalar.
- **pypdf_baseline:** Has the three codes first, then concatenates `childrenEveryday` without whitespace. It shares PyMuPDF's broad order defect and additionally corrupts the boundary between two visible words.

Manual rank is **pdf2md_local > pymupdf_text > pypdf_baseline**.

### Are the metrics/ranking sensible?

No.

- The evaluator ranks **pymupdf_text (T 0.964286) > pdf2md_local (0.892857) > pypdf_baseline (0.819841)**. It gives no order scalar and therefore rewards native extraction even though pdf2md_local is visibly closer to reading order.
- pypdf_baseline being last is sensible because `childrenEveryday` is a real merge defect.
- Every candidate with all visible codes receives `bag_of_digit_percent = 0.75`. The visible codes contain digit counts `{0:3, 1:4, 3:3, 4:4, 5:2, 7:1}`, but the rule requires `{0:4, 1:5, 3:5, 4:6, 5:2, 7:2}`. The excess digits come from code-like text repeated inside reference image descriptions, not additional visible labels. This falsely lowers every content/correctness scalar, including exact native text.
- The reference contains roughly 1,300 characters of prose image alt descriptions not present as text in the PDF. Those descriptions are non-authoritative generated interpretations (and at least the people description is not safely grounded as exact scalar text). They should not contaminate scalar text rules. The current evaluator mostly excludes their words/sentences but leaks their digits into `bag_of_digit_percent`, an internally inconsistent treatment.
- `is_header` correctly observes that candidates do not emit the custom wrapper, but no rule checks the actual visible order among header fields and codes.

## 5. `text_sparse__numbers`

### Frozen PDF evidence

The render is a vertical one-number-per-line list from 182 through 210, with no omissions, duplicates, or reordering.

### Candidate defects

- **pdf2md_local:** Contains all integers in exact sequence but collapses 187–210 onto one Markdown line. It therefore loses most of the visible list structure.
- **pymupdf_text:** Exact sequence and one integer per line.
- **pypdf_baseline:** Exact sequence and one integer per line (only terminal-newline whitespace differs).

Manual rank is **pymupdf_text = pypdf_baseline > pdf2md_local**.

### Are the metrics/ranking sensible?

Content scores of 1 are sensible if the metric is explicitly token-only. The three-way scalar tie and **O = N/A** are not sufficient for faithful parsing: no order rules exist, and none of the rules checks one-item-per-line structure. Thus **C = K = T = R = 1** incorrectly implies equivalence for a structure-aware Markdown task. The expected reference itself is grounded in the PDF; the anomaly is missing rule coverage, not bad reference content.

## Consolidated findings

1. **Manual ranks:**
   - name list: `pymupdf_text = pypdf_baseline > pdf2md_local`
   - returns: `pymupdf_text > pdf2md_local > pypdf_baseline`
   - blank: three-way tie
   - IKEA: `pdf2md_local > pymupdf_text > pypdf_baseline`
   - numbers: `pymupdf_text = pypdf_baseline > pdf2md_local`
2. The scalar rank is sensible for name list and returns, fully sensible for blank, backward for IKEA because order is unmeasured, and falsely tied for numbers because line/list structure is unmeasured.
3. The most serious rule defects are the name-list Turkish/sentence normalization failures and IKEA's digit bag contaminated by non-PDF image descriptions.
4. The most serious retained-output anomaly is pypdf_baseline/returns: no retained output, yet the evaluator labels it successful and scores empty Markdown as an ordinary zero result.
5. Header/footer wrapper failures should be reported separately from lexical extraction quality. They depress `rule_pass_rate` but generally do not enter the text scalars, which is appropriate only if the distinction is made explicit.
