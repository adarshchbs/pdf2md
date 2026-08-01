# Candidate-blind audit — exact-metric source PDFs (audit B)

Inspection method: opened each source PDF directly with PyMuPDF (fitz) — no
candidate outputs, benchmark JSONL/references, evaluator reports, parser
code, or run artifacts were consulted. For each PDF I extracted native text
(`page.get_text("text")` and `"dict"`), inspected embedded font info
(`page.get_fonts`), inspected images (`page.get_images`), and rendered the
page to a 200-dpi PNG for visual comparison against the extracted text/blocks.

All five files are single-page PDFs.

## 1. docs/text/text_multicolumns__gridofnumbers.pdf

- Content: a dense grid of 8-digit numeric codes (e.g. 20121234, 20121235,
  ...) arranged in an 8-column, ~37-row grid, plus a footer "Page 3 of 8"
  (this file is a single extracted page from a larger multi-page source
  document; the footer numbering is inherited from the original doc and does
  not imply extra pages are present in this file — page_count == 1).
- Font: embedded TrueType subset `AAAAAC+Cambria+1` — genuine embedded font,
  not a raster/scanned page.
- Images: 0. Text blocks: 37 (one per row).
- Native text extraction (`get_text("text")`) returns the numbers in
  left-to-right, top-to-bottom row-major order, which exactly matches the
  visual grid layout in the rendered PNG (verified by direct visual
  comparison of the first ~35 rows against the extracted string).
- Verdict: genuinely born-digital / native-text-solvable. Visible content is
  simple (numeric grid), reading order native extraction gives is correct
  (row-major, matching visual layout), and native extraction is reliable —
  no images, no encoding anomalies, straightforward monotonic per-row blocks.

## 2. docs/text/text_multilang__japan2.pdf

- Content: Japanese legal/regulatory text (grant/subsidy regulations,
  articles 11–14, "実績報告", "額の確定等", "支払", "交付決定の取消し等" etc.),
  single column, standard paragraph/clause layout.
- Fonts: embedded `MS-Mincho` in both a Type0/Identity-H CID form and a
  TrueType form — standard embedding pattern for CJK text in
  Word-generated PDFs.
- Images: 0. Text blocks: 38.
- Native text extraction (`get_text("text")`) reproduces the Japanese
  characters correctly and in the correct top-to-bottom, single-column
  reading order; direct visual comparison of the rendered PNG against the
  extracted string confirms character-for-character match (headings,
  numbered clauses, half-width parentheses around headings, full-width
  punctuation) with no mojibake/cmap corruption.
- Verdict: genuinely born-digital / native-text-solvable. Single-column
  reading order is unambiguous and native extraction is reliable (CID→glyph
  mapping resolves correctly, verified visually).

## 3. docs/text/text_multilang__unknonw.pdf

- Content: Thai-language university activity report (numbered activities
  2–7, e.g. "กิจกรรม ปัจฉิมนิเทศ", "กิจกรรม พัฒนาบุคลิกภาพ", body paragraphs
  describing student-development and primary-health-service activities,
  including embedded Latin-script tokens/numbers such as "YRU-ERP",
  "2,634,800 บาท", "Hosp", "A-med"). Single column, boxed page (visible
  border rectangle in the render).
- Fonts: embedded `THSarabunPSK` and `THSarabunPSK-Bold`, each present in
  both TrueType and Type0/Identity-H CID form — standard Thai font embedding.
- Images: 0. `get_text("dict")` reports 1 top-level block for the whole page
  (Thai script line-breaking makes MuPDF merge it into one flow block); this
  is a text-extraction/segmentation quirk, not evidence of a raster page.
- Native text extraction reproduces the Thai text (including Thai numerals
  context, mixed Latin acronyms/numbers, and tone marks/vowel diacritics)
  and the rendered PNG visually matches the extracted string exactly,
  including line breaks at the same wrap points — no cmap corruption or
  garbled glyphs observed.
- Verdict: genuinely born-digital / native-text-solvable. Reading order is a
  single linear column (unambiguous). Native extraction is reliable — Thai
  diacritic/tone-mark stacking, which is a common failure mode for naive
  text extractors, renders correctly and matches the visual page.

## 4. docs/text/text_simple__clilogs.pdf

- Content: plain-text-style CLI/log output from "Karen's Directory Printer
  v5.4.4" — COMMENT lines explaining format, then FOLDER/FILE inventory
  lines (monospace, one entry per line: e.g. "FILE 001 title page",
  "FOLDER C:\Users\deand\...\ ---A--- 0 59 4,661,248 4,661,248").
- Font: embedded `CourierNewPSMT` (monospace), in both Type0/Identity-H and
  TrueType form.
- Images: 0. `get_text("dict")` reports 1 block (again a single flowed text
  block, consistent with a continuous monospaced log dump rather than
  multiple discrete text frames).
- Native text extraction reproduces every line faithfully in strict
  top-to-bottom order, including embedded literal tab-separated column
  headers ("<TAB>") and Windows file paths with backslashes; visual
  comparison of the rendered PNG confirms an exact line-for-line match with
  no reordering or dropped tokens.
- Verdict: genuinely born-digital / native-text-solvable. This is single-
  column monospaced log text — the simplest possible case for reading order
  and native extraction reliability, and no issues were observed.

## 5. docs/text/text_simple__concession.pdf

- Content: full-bleed color photograph of a black helicopter in flight over
  a New Zealand back-country landscape (mountains, forest, paddock), with a
  short native-text caption block overlaid near the top of the page:
  "A concessions for these helicopters allows the operator, James Scott to
  provide value transport services to hunters, climbers and trampers
  wanting to quickly access the amazing West Coast back country in New
  Zealand." (Note: the caption itself contains a grammatical error — "A
  concessions" — but this is a defect of the source document's authored
  text, not of extraction.)
- Font: embedded `Arial-BoldMT` (MacRomanEncoding) for the caption text only.
- Images: 1 (the full-page background photo — a genuine decorative/content
  photograph, not a text raster: confirmed visually, no textual content is
  baked into the pixels beyond what a viewer would recognize as a photo of a
  helicopter/landscape).
- `get_text("dict")` reports 4 blocks total: 1 image block (bbox spans
  virtually the whole page, 0,0 to ~595,842) plus 3 text-line blocks for the
  three visually-wrapped lines of the caption, all confirmed via bbox
  y-coordinates (60.5, 71.3, 82.1 — i.e. top-to-bottom, non-overlapping,
  matching the visual line order in the render).
- Total native text length is short (219 characters) simply because the
  page's only textual content is a 3-line caption — this is not a sign of
  failed/partial extraction; there is no other visible text on the page to
  miss (verified by direct visual inspection of the full-page render, which
  shows only the caption text and the photograph, nothing else).
- Verdict: genuinely born-digital / native-text-solvable for the caption
  text (the only actual "content" text on the page). The image is
  non-textual decorative/subject-matter photography and does not carry
  additional content that native text extraction would need to capture.
  Reading order of the 3 caption lines is unambiguous (simple top-to-bottom
  stack, confirmed by bbox ordering) and native extraction is reliable.

## Summary table

| File | Pages | Embedded fonts (native, not raster) | Images | Reading order | Native extraction reliability |
|---|---|---|---|---|---|
| gridofnumbers | 1 | Cambria (TT) | 0 | row-major grid, matches visual | reliable |
| japan2 | 1 | MS-Mincho (CID+TT) | 0 | single column | reliable, no mojibake |
| unknonw (Thai) | 1 | THSarabunPSK ×2 (CID+TT) | 0 | single column | reliable, diacritics correct |
| clilogs | 1 | CourierNewPSMT (CID+TT) | 0 | single flowed block, line order correct | reliable |
| concession | 1 | Arial-BoldMT | 1 (decorative photo, no baked-in text) | 3-line caption, top-to-bottom | reliable; short text is genuine (not a truncation artifact) |

All five documents are genuinely born-digital and native-text-solvable.
None showed evidence of scanned/rasterized text, cmap/encoding corruption,
or ambiguous/incorrect reading order relative to their visual layout.
