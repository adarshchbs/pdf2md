# Candidate-blind audit — exact-metric source PDFs (set A)

Method: inspected ONLY the source PDFs directly (native text layer via `page.get_text('text'/'words')`,
embedded font/encoding info, and 3x-scale page-image renders). Did not open any candidate output,
benchmark JSONL/reference, evaluator report, parser code, or run artifact. All 4 PDFs are single-page.

Renders saved to: `/private/tmp/claude-501/-Users-adarsh-project-pdf2md/1c16afff-29c0-4394-8e54-7ccb74cc0ca2/scratchpad/renders/`
(`text_dense__tableofcontent.png`, `text_misc__dash.png`, `text_misc__gridofimages.png`, `text_misc__notATable.png`)

---

## 1. docs/text/text_dense__tableofcontent.pdf

- **Page**: 1 page, 595x842pt (A4). 0 images, 1 vector drawing (a rule line under the header).
- **Content**: A two-column alphabetical name/phone-extension directory ("Fourth Quarter 2002"),
  e.g. "Geldart, Sharon .......... 3164". Despite the filename, this is an index-style directory
  page, not a hierarchical table of contents — but structurally it is the same two-column,
  dot-leader "label ... page/ext-number" pattern that a genuine TOC would have.
- **Reading order**: Native text extraction (`get_text('text')`) reads column 1 fully top-to-bottom
  ("Geldart, Sharon" → ... → "Matte, Francine"), then column 2 top-to-bottom ("Matthews, Bill" → ...
  → "Singh, Mohindar"). This matches the visual layout exactly — no column interleaving errors.
  Verdict: reading order is **correct and reliable** for this page.
- **Extraction reliability defect (real, not a render artifact)**: the embedded font is a subsetted
  TrueType (`AAAAAC+TimesNewRoman`) with a broken/non-standard ToUnicode CMap. Accented characters
  and typographic apostrophes are visually rendered correctly in the page image (e.g. "Gosselin,
  André", "Harvey, André", "The Hon./L'hon.") but the native text layer decodes them to the wrong
  Unicode code points: é → È (e.g. "AndrÈ" instead of "André"), ' → í (e.g. "Líhon" instead of
  "L'hon"). This is systematic (every À/é and every apostrophe in the sampled words is affected),
  confirmed via `page.get_text('words')` returning literal `'AndrÈ'` tokens tied to glyph positions
  that visually show "André". Any text-based scoring against this page's raw extracted string will
  be penalized on every accented name/apostrophe even though the source is genuinely born-digital
  and the layout/reading-order is fine.
- **Verdict**: Genuinely born-digital, native-text-solvable for reading order and plain-ASCII
  content; NOT reliable at the character level for accented Latin letters or apostrophes due to a
  broken embedded-font ToUnicode map. A native-extraction pipeline that doesn't special-case/repair
  this font's cmap will systematically mis-render such names.

## 2. docs/text/text_misc__dash.pdf

- **Page**: 1 page, 612x792pt (US Letter), page 11 of "USAble Mutual Insurance Company — 2021
  Off-Exchange Small Group Rate Filing". 0 images, 0 drawings.
- **Content**: A nested bulleted/numbered list plus prose paragraphs about risk-adjustment-transfer
  variables. The overwhelming majority of the "content" — every bullet label and most sentence
  continuations — consists of literal run-of-dash placeholder strings (e.g.
  `----------------------`) of varying lengths, standing in for text that was redacted at the
  source before the PDF was produced. This was confirmed by cross-checking: the rendered page image
  shows the exact same dash runs at the exact same positions as the native text layer (bullets,
  indentation, and dash-run lengths match 1:1 between image and extracted text) — i.e. this is not
  an OCR/extraction failure or a font-substitution artifact, the PDF's actual text content is these
  dash characters.
  Two labeled sub-bullets and a short prose lead-in/close remain genuinely legible
  ("When estimating the risk adjustment transfer...", "Other variables ... include the following:",
  "Finally, the HCRP was estimated by ------...", "The following exhibit below demonstrates...").
  The "following exhibit" it refers to is not present on this page (page ends in blank space) —
  consistent with this being one page excerpted from a larger multi-page filing.
- **Reading order**: Single-column, top-to-bottom, nested list — unambiguous, correct.
- **Verdict**: Genuinely born-digital / native-text-extractable, and extraction is faithful to the
  visible page — but the underlying document is a **source-redacted filing**: most of the
  substantive information a metric would need to check was replaced with dashes by the filer, not
  lost by any parsing step. Any ground-truth/candidate comparison keyed to "content" for this page
  should expect dash placeholders as the correct answer for those spans, not real numbers/labels.

## 3. docs/text/text_misc__gridofimages.pdf

- **Page**: 1 page, 595x842pt (A4). 50 embedded raster images, 51 vector drawings (thin border/frame
  rects around each photo). 0 substantive body text — only 50 short Japanese filename-style
  captions (e.g. "053_20090718白山祭り042" … "102_20090718白山祭り091") printed directly under each
  thumbnail.
- **Content**: A pure photo contact-sheet: 50 small event photographs (a Japanese summer festival,
  "白山まつり" = "Hakusan Matsuri") arranged in a 5-column x 10-row grid, each with its own
  filename-derived caption underneath. Photos progress from a lit stage performance (top rows) to
  a nighttime street procession/lantern scene (bottom rows), i.e. the sequence is chronological/
  narrative, not arbitrary.
- **Reading order**: Native text extraction returns the 50 captions in row-major order (left→right
  within each row, then down: 053,054,055,056,057, 058,059,...,102), which exactly matches the
  visual grid position (row 1 = 053–057, row 2 = 058–062, ... row 10 = 098–102). Reading order is
  **correct**.
- **Verdict**: Genuinely born-digital for the caption text (correct, complete, correctly ordered),
  but the actual page "content" that a viewer would care about (the 50 photographs themselves) is
  raster image data with no text representation — native text extraction alone recovers only the
  filenames, not any depiction of what is in the photos. This is a case where native-text-solvable
  ≠ content-complete: text extraction is reliable for what little text exists, but the page is
  overwhelmingly image content by area and by information content.

## 4. docs/text/text_misc__notATable.pdf

- **Page**: 1 page, 792x612pt (US Letter, landscape). 0 images, 0 drawings.
- **Content**: 20 lines of raw pipe-delimited (`|`) data records, e.g.
  `T|TJEFF|202508|BINC|11|22|2|1400|0|0|1400|1400|0|0|0|0|0.0035|0.0035|1400|0.0033|0.0|0|0.0|0|0.0000|0.0`.
  This is literally a flat delimited text dump (ticker/date/code + numeric fields), rendered as
  plain left-aligned monospace-ish text — there are no table borders, no ruling lines, no grid
  structure, and no column alignment beyond what whitespace/tabs happen to produce visually. The
  filename ("notATable") accurately describes it: it is not a table at all, just literal delimited
  text that superficially resembles tabular data because of the repeated `|` separators.
- **Reading order**: Single column, one record per line, top-to-bottom — trivial and unambiguous.
- **Verdict**: Fully genuinely born-digital and native-text-solvable. Content is 100% plain ASCII
  text, extraction is exact and reliable, and reading order is not in question since there is no
  real table/grid layout to get wrong — it is a flat line-oriented text dump.

---

## Summary

| File | Born-digital/native-solvable | Reading order | Extraction reliability caveat |
|---|---|---|---|
| text_dense__tableofcontent.pdf | Yes | Correct (2-col handled properly) | Broken font ToUnicode cmap mis-maps accented letters/apostrophes (é→È, '→í) despite correct visual rendering |
| text_misc__dash.pdf | Yes | Correct (single column, nested list) | None on extraction; but source content is genuinely redacted (dashes are the true "content") |
| text_misc__gridofimages.pdf | Partially | Correct (row-major grid order) | Only caption text is native-extractable; the 50 photos (the real content) are raster images with no text form |
| text_misc__notATable.pdf | Yes | Trivial/correct | None — clean flat delimited text, not an actual table |

SHA-256 of this file (computed after write):
