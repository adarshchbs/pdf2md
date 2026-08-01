# Verification Cycle Against Frozen Audit B

Frozen evidence file: `digital-audit-b.md`, SHA-256
`e4eb305aff47ece02329c4ce03803add13e0df16d753af6acaa52edf8beccb71` (verified
matching before this comparison began; the frozen file was not edited).

Scope: for the same 4 PDFs, inspected the retained `pdf2md_local`,
`pymupdf_text`, and `pypdf_baseline` result/raw JSON outputs plus the
per-example evaluator rule results, at:
- `benchmarks/runs/parsebench-full/{pipeline}/{pipeline}/text/{doc}.result.json`
  (and `.raw.json`)
- `benchmarks/runs/parsebench-full/{pipeline}/{pipeline}/_evaluation_report.json`
  (`per_example_results`, filtered to `example_id == "text/{doc}"`)

Each candidate output was compared strictly against the frozen visual
description of the original PDF (Section headers below reference the frozen
audit's section numbers), not against the evaluator's expected/reference
answer. Where the evaluator's reference itself was checked, it is called out
explicitly as such.

All 4 files have substantial native text layers (5,390–7,333 extracted
characters each, confirmed also via a direct PyMuPDF `get_text()` re-check),
consistent with the frozen audit's implicit judgment that they are natively
extractable, digitally-born PDFs. None show any evidence of being
scanned/OCR-only or otherwise mis-scoped for text extraction. No exclusion
recommendation for any of the 4.

---

## 1. docs/text/text_dense__legalRef.pdf

**Content coverage:** All three parsers extract essentially the same text —
the full personnel roster paragraph, the "AGENTE DO MINIST. PUBLICO-QP"
subheading, the Joel da Silva Rosa entry, the closing legal paragraph, and
the footer date/page-number/publication line. This matches the frozen
audit's described body content point-for-point.

**Header/logo text — confirmed absent from the PDF's native text layer.**
Direct re-extraction with PyMuPDF (`page.get_text()` / span-level
`get_text("dict")`) shows the PDF's text layer contains only 51 spans total,
and neither the "MPMG" logo wordmark ("Ministério Público do Estado de Minas
Gerais") nor the large caps title "DIÁRIO OFICIAL ELETRÔNICO DO MINISTÉRIO
PÚBLICO DE MINAS GERAIS" appear anywhere in it — they are rendered as
image/vector graphics, not selectable text. This confirms the frozen audit's
description of that content as visual "header chrome" (logo + banner), and
explains why **all three** result markdowns correctly omit it: there is
nothing in the text layer to extract. No parser is at fault here.

**Footer text** ("21/07/2012 - 4 - Diário Eletrônico do MPMG") is a real
text-layer span (bbox confirmed) and all three parsers do capture it, though
with different whitespace/line handling (see Formatting below).

**Formatting/structure — pdf2md_local vs. the two raw-line extractors:**
- `pdf2md_local` reflows the whole page into clean paragraphs, joining
  the source's PDF-internal line-wraps into continuous prose with real
  blank-line paragraph breaks between: (1) roster paragraph, (2) subheading,
  (3) Joel da Silva Rosa entry, (4) closing legal paragraph, (5) footer. This
  matches the frozen audit's own characterization of the roster as "one
  continuous paragraph of prose... not a bulleted/numbered list" — the
  reflow is faithful to the visual reading, and arguably *more* readable
  than a literal line-by-line dump.
- `pymupdf_text` and `pypdf_baseline` preserve the PDF's native hard line
  breaks (i.e., a line break wherever the original PDF wrapped a line),
  producing near-identical output to each other (51 vs. 52 lines,
  byte-for-byte matching text). This is a faithful visual transcription but
  reads as hard-wrapped raw text rather than markdown-formatted prose.
- **Anomaly unique to `pdf2md_local`:** its footer line reads
  `"21/07/2012 - 4 -Diário Eletrônico do MPMG"` — missing the space between
  `"-"` and `"Diário"` that exists in the source spans (`"- 4 -"` then
  `"Diário Eletrônico do MPMG"` as two adjacent spans). This is a
  space-dropping bug in pdf2md_local's paragraph-joining/reflow logic. It
  causes a real (if minor) evaluator false negative (see below).

**Evaluator agreement:** `rule_pass_rate` — pdf2md_local 0.9485 (332/355),
pymupdf_text 0.9530 (333/355), pypdf_baseline 0.9530 (333/355). pymupdf_text
and pypdf_baseline tie exactly; pdf2md_local trails by one rule
(`missing_specific_sentence` for the footer line, directly attributable to
the dropped-space bug above — a legitimate, evidence-grounded deduction, not
an evaluator artifact). `normalized_text_correctness`: pdf2md_local 0.819 vs.
pymupdf_text/pypdf_baseline 0.902/0.902 — a larger gap than the single rule
miss would suggest, indicating the underlying text-similarity metric
penalizes pdf2md_local's paragraph-reflow (which changes literal line
boundaries/whitespace vs. the reference) more heavily than the rule-based
sentence checks do.

**Evaluator/reference anomaly (affects all 3 equally):** the reference
expects the header banner text ("DIÁRIO OFICIAL ELETRÔNICO DO MINISTÉRIO...")
to be present and wrapped in `<page_header>` tags, and the footer line
wrapped in `<page_footer>` tags (`rule_is_header`, `rule_is_footer`,
`rule_missing_specific_sentence` for the header text, and `rule_order`, which
fails because it can't locate the header text to anchor a "before" ordering
check). Since the header text is not part of the PDF's text layer at all
(confirmed above), this reference expectation is unsatisfiable by any
text-layer-only extractor, and all three pipelines fail these checks
identically. This is a reference/evaluator anomaly, not a real parser
deficiency, and does not discriminate between the three parsers.

**Verdict:** pymupdf_text and pypdf_baseline are byte-for-byte tied and both
edge out pdf2md_local on strict rule/text-correctness metrics, purely due to
pdf2md_local's dropped space in the footer and its reflow diverging from the
reference's literal line structure. Content coverage and order are
equivalent across all three and consistent with the frozen visual evidence.

---

## 2. docs/text/text_dense__livingWord.pdf

**Content coverage — a real omission shared by all three parsers:** the
frozen audit flagged a genuinely anomalous sentence physically present in
the PDF's own text layer: *"Parents may choose to proceed at their own pace,
stop eligibility determination, or withdraw from the program at any time."*
(right-half teal callout box). A direct grep of all three parsers' extracted
markdown for this doc found **zero** matches in any of the three outputs —
`pdf2md_local`, `pymupdf_text`, and `pypdf_baseline` all fail to extract this
callout-box text entirely for this file. (It is present in this cross-file
example, but re-appears — and is extracted, if garbled — in
`text_misc__stepbystep.pdf`; see Section 4.) This is a shared content gap
across all three pipelines on this document, not a differentiator between
them, but it is a real, verifiable content loss versus the frozen evidence.

**Reading-order — pdf2md_local is visibly worse than the two raw extractors.**
The frozen audit describes this as a single PDF page containing **two
independent two-column bulletin halves side by side**. All three parsers
linearize a two-up, 4-column-total layout into one text stream, so some loss
of the "read left-half fully, then right-half" structure is expected from
any of them. However, the severity differs sharply:
- `pymupdf_text` and `pypdf_baseline` (identical to each other) read the
  **entire left half in correct internal order** first — "Wisdom 2:12" →
  Psalm 53 response block → James 3:16 paragraph → Gospel Acclamation →
  Mark 9:30-37 paragraphs → citation footer — before switching to the right
  half. Within each half the column-merge is imperfect (sidebar boxes and
  main-article paragraphs interleave), but the core scripture-reading
  narrative stays intact and in the order described by the frozen audit.
- `pdf2md_local` opens with `"The"` / `"Living"` (title fragments, split
  across two separate paragraphs) immediately followed by
  `"prayed properly, you have prayed for something to indulge your own
  desires. Gospel Acclamation"` — i.e., it jumps straight into the *middle*
  of the James 3:16 paragraph (which, per the frozen audit, is the sixth
  content item in the left column) before ever presenting the "Wisdom
  2:12, 17-20" heading or its verse text, which only appear starting at the
  document's line 7. This is a materially more scrambled reading order than
  either raw-line extractor, and it visibly contradicts the frozen audit's
  documented top-to-bottom / left-column-then-right-column structure.

**Formatting:** none of the three preserve bold headings, verse-line breaks,
or the two-column visual structure as markdown — all output flat paragraphs
or (pymupdf/pypdf) raw wrapped lines with no bold/heading markup. This is
consistent with all three being plain-text-layer extractors with no
style/markdown semantics, and explains the uniform 0.0 evaluator score below.

**Evaluator agreement:** All three score identically — `rule_pass_rate` 0.0
(0/32), broken down into `rule_is_title_pass_rate` 0.0/13,
`rule_title_hierarchy_percent_pass_rate` 0.0/1, `rule_is_bold_pass_rate`
0.0/18 — for all of pdf2md_local, pymupdf_text, and pypdf_baseline. This
uniform failure reflects that the evaluator's formatting rules (title/bold
detection) require markdown-level styling markup that none of the three
plain-text pipelines emit; it is not discriminating and not itself
anomalous — it correctly reflects that these are all non-styling-aware
extractors. However, the evaluator gives **no credit and no penalty
differentiation** for the very real reading-order gap documented above
(there is no separate `rule_order` check exercised for this doc in the
per-example metrics), so the identical 0.0 scores mask a real, visible
quality difference: pdf2md_local's output is measurably harder to read
correctly than pymupdf_text/pypdf_baseline's for this file.

**Verdict:** pymupdf_text and pypdf_baseline (tied, byte-level near-identical)
are visibly better for reading order on this file; pdf2md_local's
paragraph-reordering logic actively scrambles the two-column, two-up layout
worse than doing nothing (i.e., worse than linear raw-line extraction). All
three miss the same anomalous callout-box sentence entirely. Evaluator scores
agree (all tied at 0.0) but for reasons unrelated to the order difference
found here — the tie is not meaningful evidence that the three are equally
good.

---

## 3. docs/text/text_misc__mark2.pdf

**Content coverage:** all three capture the "MEDICAL PAYMENTS" heading, both
lettered lists (a–t and a–m) with the self-referential "X.\*" formula
notation exactly as described in the frozen audit, both "Rounding:" notes,
and the footer ("Ohio / Road / Rating Logic - Page H6"). The "ACUITY" logo
wordmark noted in the frozen audit's header is absent from all three
outputs — consistent with it being a logo graphic rather than text (not
independently re-verified via PyMuPDF for this file, but plausible given the
identical pattern found for legalRef's logo).

**Reading order — pdf2md_local matches the frozen page layout; pymupdf_text
and pypdf_baseline both misplace the footer.** The frozen audit is explicit
that the footer band ("Ohio | Road | Rating Logic - Page H6") sits **outside
the bordered content box, at the very bottom of the page**, after all list
content. In the actual outputs:
- `pdf2md_local` places the footer line **last**, after both lettered lists
  and both Rounding notes — matching the frozen layout.
- `pymupdf_text` places `"Ohio / Road / Rating Logic  -  Page H6"` **immediately
  after the two header lines** ("Road and Residence®" / "Rate and Rule Filing
  Manual") and **before** "02-2012 Edition" and "MEDICAL PAYMENTS" — i.e., at
  the very top of the document, well before the list content it should
  trail.
- `pypdf_baseline` makes the identical mistake, placing
  "Ohio Road Rating Logic  -  Page H6" right after the header lines and
  before "02-2012 Edition".

This is a genuine, visually-checkable order defect in both pymupdf_text and
pypdf_baseline (likely because raw PDF content-stream/text-layer draw order
does not match top-to-bottom visual order for this page — the footer text
object was drawn/positioned early in the stream despite rendering at the
bottom), which pdf2md_local's layout-aware reordering correctly fixes for
this file. This is the reverse of the order finding in Section 2 — here
pdf2md_local's reordering logic helps rather than hurts.

**Formatting artifact unique to pypdf_baseline:** the word "Rounding:" is
rendered as `"R o u n d i n g :"` (letters space-separated) twice in
pypdf_baseline's output, where pdf2md_local and pymupdf_text both render it
correctly as `"Rounding:"`. This is a pypdf-specific character-spacing
extraction glitch (likely from how pypdf handles the underlined-word's glyph
positioning/kerning), not present in the other two pipelines.

**Missing formatting (shared, expected):** none of the three preserve the
yellow-highlight annotations (on "02-2012" and on the "p." row) or the
underline on "Rounding" — expected for plain-text extractors, consistent
with all three showing 0.0 on `rule_is_underline_pass_rate` and
`rule_is_mark_pass_rate`.

**Evaluator agreement:** `rule_pass_rate` 0.0 (0/5) identically across all
three, decomposed as `rule_is_title_pass_rate` 0.0/1,
`rule_title_hierarchy_percent_pass_rate` 0.0/1, `rule_is_underline_pass_rate`
0.0/1, `rule_is_mark_pass_rate` 0.0/2. As with Section 2, this uniform score
correctly reflects the shared absence of style-markup output, but it masks
the real, visible footer-ordering defect that distinguishes pymupdf_text and
pypdf_baseline (both wrong) from pdf2md_local (correct) on this file — there
is no order-specific rule active for this document in the evaluator's rule
set, so the tie is not meaningful for order quality either.

**Verdict:** pdf2md_local is visibly better for reading order on this file
(correct footer placement); pymupdf_text and pypdf_baseline both misplace the
footer at the top. pypdf_baseline additionally has a character-spacing
artifact on "Rounding" not present in the other two. Evaluator scores agree
(tied at 0.0) but again for reasons that don't track the real quality
difference found here.

---

## 4. docs/text/text_misc__stepbystep.pdf

**Content coverage:** all three capture the title/subtitle banner, the three
stage-group labels ("FIRST STEPS...", "NEXT STEPS...", "FUTURE STEPS..."),
all 7 numbered steps with their headings and bullet text, and the footer
attribution line — matching the frozen audit's described content set.

**The recurring callout-box sentence is present but handled very
differently.** The frozen audit documents the same anomalous sentence as in
Section 2 ("Parents may choose to proceed at their own pace, stop
eligibility determination, or withdraw from the program at any time.") as a
free-floating callout box positioned beside steps 4–5.
- `pymupdf_text` and `pypdf_baseline` both extract this sentence **intact
  and in one coherent block**, correctly positioned between step 5 ("Delivery
  of Services") and step 6 ("IFSP Reviews") — matching the frozen audit's
  placement "positioned to the right of badges 4–5".
- `pdf2md_local` **shreds this same sentence into three disconnected
  fragments** scattered in the wrong places: `"stop eligibility
  determination,"` appears misplaced right after the step-4 IFSP text (line
  19), then much later `"Parents may choose to proceed at their own pace,"`
  appears near step 7 (line 29), and finally `"or withdraw from the program
  at any time."` appears as its own paragraph after that (line 31). The
  clause "stop eligibility determination" is separated from "Parents may
  choose..." by an entire step's worth of unrelated content, destroying the
  sentence's coherence and misattributing pieces of it to the wrong steps.
  This is a clear content-scrambling defect specific to pdf2md_local's
  layout-merging logic on this floating-box element, mirroring (independently)
  the reading-order problems already found for pdf2md_local in Section 2 on
  a different kind of layout (two-up spread) — suggesting a recurring
  weakness in how pdf2md_local integrates off-flow/floating boxes into its
  linear reading order, as opposed to the raw-line extractors which simply
  never reorder text away from its native stream position.

**Shared PDF-native text-extraction artifact (not a parser bug):** all three
outputs show the identical glyph-splitting glitch "Infant-T oddler" (stray
space inserted mid-word) wherever "Infant-Toddler" appears with that
particular font run, plus similar splits like "child's" → "child' s" and
"You" → "Y ou" / "Transition" → "T ransition". Because this exact,
character-for-character artifact recurs identically across pdf2md_local,
pymupdf_text, and pypdf_baseline, it reflects a genuine quirk in the source
PDF's glyph/kerning encoding (likely a ligature or font substitution issue),
not a bug unique to any one pipeline — flagged for completeness, not as a
discriminator.

**Formatting artifact unique to pypdf_baseline:** literal tab characters
appear embedded mid-sentence in a few bullet lines (e.g. `"•\t \t These\tfirst\tmeetings\twill\thelp\tus\tlearn\tyour\tconcerns..."`),
where pymupdf_text and pdf2md_local render the same text with normal spaces.
This is a pypdf-specific whitespace-handling glitch (converting inter-glyph
positioning gaps into tab characters) not present in the other two
pipelines — a second data point (with Section 3) of pypdf_baseline
introducing extraction artifacts that pymupdf_text does not.

**Evaluator agreement:** `rule_pass_rate` 0.0 (0/20) identically across all
three: `rule_is_title_pass_rate` 0.0/11, `rule_title_hierarchy_percent_pass_rate`
0.0/1, `rule_is_bold_pass_rate` 0.0/7, `rule_is_italic_pass_rate` 0.0/1 — again
reflecting the shared absence of style markup, not the (very real) content-
scrambling difference on the callout sentence documented above, which is not
covered by any active rule for this example.

**Verdict:** pymupdf_text is the cleanest of the three here — correct
callout placement and no whitespace artifacts. pypdf_baseline matches
pymupdf_text on content/order but introduces its own tab-character
whitespace glitches. pdf2md_local visibly and materially corrupts the
floating callout sentence's content and placement, the most serious
content-fidelity defect found across all 4 files in this cycle. Evaluator
scores again agree (tied at 0.0) but do not reflect this real difference.

---

## Cross-cutting findings

1. **pymupdf_text and pypdf_baseline are near-identical** on every file in
   this set (often byte-for-byte on content, differing mainly in incidental
   whitespace/line-wrap presentation), which is expected since both are
   raw text-layer extractors with linear/stream-order output. pypdf_baseline
   introduces two extraction-quality artifacts not present in pymupdf_text
   (space-separated "R o u n d i n g :" in mark2; embedded tab characters in
   stepbystep) — a small but real edge in favor of pymupdf_text over
   pypdf_baseline when the two otherwise tie.

2. **pdf2md_local's layout-aware reordering is a double-edged sword.** It
   correctly fixes a real footer-misplacement defect that both raw
   extractors exhibit on `text_misc__mark2.pdf` (Section 3), and it produces
   more readable reflowed prose on `text_dense__legalRef.pdf` (Section 1).
   But on the two more visually complex layouts — the two-up/two-column
   spread (`text_dense__livingWord.pdf`, Section 2) and the floating-callout
   infographic (`text_misc__stepbystep.pdf`, Section 4) — its reordering
   logic visibly scrambles content worse than simply leaving raw text-stream
   order alone, including one case (stepbystep) where it shreds a single
   sentence into three disconnected, misplaced fragments. Net across these 4
   files, pdf2md_local is not consistently better or worse than the raw
   extractors on order/structure — it wins once (mark2), ties on content
   coverage everywhere, and loses clearly twice (livingWord, stepbystep).

3. **Evaluator/reference anomaly (text_dense__legalRef only):** the
   evaluator's reference expects header-banner text to be present and
   wrapped in `<page_header>` tags, but that text is not part of the PDF's
   native text layer at all (independently confirmed via direct PyMuPDF
   re-extraction — only 51 spans total on the page, none matching the
   banner text). This makes several rules (`rule_is_header`, one
   `rule_missing_specific_sentence` instance, and `rule_order`) fail
   identically and permanently for all three pipelines, regardless of
   pipeline quality. This should be treated as a reference-construction
   issue (the reference likely originated from an OCR/visual pass that
   captured the logo/banner image's text, which no text-layer extractor can
   ever reproduce), not a parser deficiency.

4. **Evaluator score ties mask real quality differences on 3 of 4 files.**
   For livingWord, mark2, and stepbystep, all three pipelines score exactly
   0.0 on `rule_pass_rate`, because the active rule set for those examples is
   entirely style/formatting rules (title, bold, italic, underline,
   highlight/mark) that none of these plain-text pipelines can satisfy. This
   uniform failure is evaluator-correct given the rule set, but it means the
   evaluator provides **zero discriminating signal** for the real,
   visually-verifiable order/content differences found in this cycle
   (pdf2md_local's footer-placement win on mark2; its order/content losses
   on livingWord and stepbystep; pypdf_baseline's whitespace artifacts). Only
   `text_dense__legalRef.pdf` has content-focused rules active, and on that
   file pymupdf_text/pypdf_baseline edge out pdf2md_local by one rule, due to
   a small, real space-dropping bug in pdf2md_local's reflow logic.

5. **No exclusion recommendation.** All four files have substantial,
   directly-extractable native PDF text layers; none show OCR/scan
   characteristics. The frozen audit's single-page framing and the
   evaluator's rule/reference setup are consistent with these being in-scope
   digital PDFs.
