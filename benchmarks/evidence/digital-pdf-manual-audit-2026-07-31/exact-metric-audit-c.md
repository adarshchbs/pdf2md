# Candidate-blind native-PDF audit (set C)

Scope: inspected ONLY the five source PDFs listed below via PyMuPDF native text extraction
(`page.get_text("text")`) and 150-dpi page renders. No candidate outputs, benchmark
JSONL/references, evaluator reports, parser code, or run artifacts were opened.

Tool: PyMuPDF (pymupdf) 1.28.0 / MuPDF 1.29.0, via project `.venv` (`uv run python3`).

---

## 1. docs/text/text_simple__name_list.pdf

- 1 page, portrait (595x842 pt). Native text length: 4396 chars.
- Visible content: a single-column, centered list of ~60+ entries, each of the form
  "Yrd. Doç. Dr. <Name> <University> <Country>" (Turkish academic committee list),
  page number "VIII" at bottom right.
- Native extraction order: line-by-line top-to-bottom, exactly matching the visual
  render (checked full page render against extracted text — order and content match,
  including Turkish diacritics e.g. "Baştürk", "Necmettin Erbakan Üniversitesi").
- Assessment: genuinely born-digital, single simple text layout (no columns/tables/
  overlapping elements). Native text extraction is fully reliable for content and
  reading order. **Native-text-solvable: YES, high confidence.**

## 2. docs/text/text_simple__returns.pdf

- 1 page, portrait (612x792 pt, US Letter). Native text length: 1571 chars.
- Visible content: IRS Form 990 Schedule O, page 2 — a header strip (form/page id,
  "Employer identification number", org name "NEADS, INC.", EIN "23-7281887") followed
  by several ruled/underlined narrative paragraphs (each preceded by a
  "FORM 990, PART VI, SECTION B/..." label) and a final comma-separated state list.
- Native extraction order: the small header/table-cell fragments ("132212 11-11-21",
  "2", "Employer identification number", "Schedule O (Form 990) 2021" x2, "Page",
  "Name of the organization", "NEADS, INC.", "23-7281887") are emitted first, in a
  jumbled cell order relative to their visual position (this is typical PDF form
  header/table cell ordering, not a content-loss issue — all header tokens are present,
  just not in strict visual reading order). After the header block, all narrative
  paragraph text is extracted in exact top-to-bottom visual order, verbatim, matching
  the render (checked full page render: paragraph text, section labels, and trailing
  state-code list all match 1:1).
- Assessment: genuinely born-digital. Body/paragraph content (the substantive text
  that would be scored) extracts in correct order and is complete. Only the compact
  header/id strip has non-visual (cell-based) ordering, which is a minor, well-known
  PDF-extraction quirk and does not affect the paragraph content. **Native-text-solvable:
  YES for body content; header field order is jumbled but complete.**

## 3. docs/text/text_sparse__blank.pdf

- 1 page, portrait (612x792 pt). Native text length: 71 chars.
- Visible content: page is visually blank except a top header line
  "vhdl-style-guide Documentation, Release 3.0.0" (with underline rule), and a footer
  with page number "8" (left) and "Chapter 3.  Installation" (right, with rule above).
  No body content is rendered on the page at all (confirmed via full-page render —
  large blank white area between header and footer).
- Native extraction: 'vhdl-style-guide Documentation, Release 3.0.0\n8\nChapter 3.
  Installation\n' — captures exactly the header text, then footer page number, then
  footer chapter title. No missing content: there is no body to miss.
- Assessment: genuinely born-digital; this is an intentionally content-empty page
  (documentation chapter break/blank filler page) — the "sparse" label reflects that
  the page is genuinely near-empty, not that content was lost. Native extraction fully
  reliable. **Native-text-solvable: YES (trivially, because there is essentially no
  content to extract).**

## 4. docs/text/text_sparse__ikea.pdf

- 1 page, landscape (842x595 pt). Native text length: 76 chars. 3 raster images embedded.
- Visible content: three full-bleed photographs side-by-side of IKEA home/living-room
  and children's-room scenes (furniture, curtains, play tent, toys, two people), each
  photo overlaid with a small code label in its corner ("PH134503", "PH134740",
  "PH134501"), plus a page header "Everyday IKEA" (top-left) and "Extra images – Living
  with children" (top-right).
- Native extraction: 'PH134503\nPH134740\nPH134501\nExtra images – Living with
  children\nEveryday IKEA\n' — the three image codes extract in correct left-to-right
  order matching their visual position; the two caption/header lines are present but
  emitted after the codes, not matching visual top position (header line actually sits
  above the images in the render, extracted last instead of first — order quirk, not a
  content-loss issue since both strings are present verbatim).
- Assessment: genuinely born-digital in the sense that all text layers (codes,
  captions) are real embedded text and extract completely and correctly, just in a
  minor non-visual order. However, the overwhelming majority of visible page content is
  the three photographs themselves, which are not text and cannot be recovered via
  native text extraction — only the incidental code/caption labels are text-extractable.
  **Native-text-solvable: PARTIAL — the small amount of actual text is reliably
  extractable (with a minor ordering quirk on the 2 caption lines), but the page's
  primary visible content (3 photos) is inherently non-text and out of scope for
  text extraction entirely.**

## 5. docs/text/text_sparse__numbers.pdf

- 1 page, landscape (792x612 pt). Native text length: 115 chars.
- Visible content: a simple vertical list of consecutive integers 182 through 210,
  left-aligned, one per line (with slightly larger line spacing for the first 6 entries
  182–187, then tighter/uniform spacing for 188–210 — confirmed via render).
- Native extraction: '182\n183\n184\n...\n210\n' — exact sequential match to the
  rendered list, correct order, no gaps or duplicates.
- Assessment: genuinely born-digital, trivial single-column numeric list. Native
  extraction fully reliable for content and order. **Native-text-solvable: YES, high
  confidence.**

---

## Summary table

| File | Native-text-solvable | Notes |
|---|---|---|
| text_simple__name_list.pdf | YES | Clean single-column list, order matches render exactly |
| text_simple__returns.pdf | YES (body); header cell order jumbled | IRS 990 form; paragraph content complete & ordered; header id-strip cell order non-visual but complete |
| text_sparse__blank.pdf | YES (trivial) | Page is genuinely near-blank; only header/footer text exists and extracts correctly |
| text_sparse__ikea.pdf | PARTIAL | Page content is dominated by 3 photographs (non-text); only incidental code/caption text is native and extracts completely, with 2 caption lines in non-visual order |
| text_sparse__numbers.pdf | YES | Simple sequential numeric list, exact match to render |
