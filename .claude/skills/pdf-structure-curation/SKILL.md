---
name: pdf-structure-curation
description: Independently curate bronze PDF evidence into authoritative structure-aware silver Parquet, or adjudicate candidate/reference disagreements without contaminating the reference.
disable-model-invocation: false
element-schema-version: "1.0.0"
source-catalog-schema-version: "2.0.0"
renderer-semantics-version: "1.1.0"
evaluator-semantics-version: "1.1.0"
---

# PDF structure curation

You are building the ground truth (silver) that the extraction code is measured against. Truth comes from page evidence: bronze artifacts and the source PDF itself. Code output is never truth.

## Evidence rules

You may use: bronze page images, bronze LiteParse JSON/text/Markdown, the source PDF (text, fonts, drawings, geometry), `app/pdf2md/schema.py`, and the rendering rules in `app/pdf2md/tables.py`.

Initial silver creation is candidate-blind. Until your silver is written and its hash recorded, do not look at candidate output, the engine code (`engine.py`, `pymupdf_tables.py`), evaluation reports, source maps, or past adjudications. If you saw any of it, the annotation is contaminated: say so and let a fresh agent restart.

In later verification cycles, run the current code and compare its output with silver. Candidate output may point you at a possible mistake, but never copy anything from it. Prove every correction from bronze/PDF evidence, and record that evidence plus exactly what changed in silver. No independent proof, no correction.

## What one element is

- One Parquet row = one logical document element. Not one PDF block, line, or cell.
- Join a paragraph or table that continues across pages.
- Split things a PDF library wrongly glued together: stacked footnotes, bullets, captions, adjacent tables. Never merge adjacent tables through empty spacer rows.
- Keep element IDs stable across revisions; a materially different element gets a new ID.
- Annotate everything, including page chrome. Recurring headers/footers/page numbers stay in the file with `include_in_output=false`.

## Vocabulary (must match the code)

The schema stores types and roles as plain strings, so silver and candidate are only comparable if both use the same closed list. Never invent new values; adding one requires a matching code change and changes this file's version identity.

- `element_type`: `paragraph`, `heading`, `caption`, `footnote`, `code`, `table`, `figure`, `header`, `footer`, `note`.
- Paragraph roles: `body`, `subtitle`, `list_item`, `figure_text`, `figure_panel_heading`; plus `heading`, `caption`, `footnote`, `code` on those element types; `running_header`, `running_footer`, `page_number` on margins; `table_continuation_marker` on `note` elements. (`list_item_continuation` is a candidate-only artifact — never write it in silver.)
- A document title is a `heading` with level 1. A printed page number is a `header` or `footer` element (top or bottom half of the page) with role `page_number` — there is no `page_number` element type. Do not invent roles like `document_title` or `table_caption`.
- Every element except `table` and `figure` carries a `ParagraphStructure` with one of these roles; tables and figures leave it null.
- Non-table elements use `format="text"` and verbatim content — no `#`, no list markers, no Markdown. Only tables use `markdown`/`html`, matching their representation.
- A repeated "(continued)"-style line is a `note` with role `table_continuation_marker`, `include_in_output=false` — not table data, not a caption.

## Provenance

- Every fragment has the real 1-based page number and a bbox measured on that page. Never reuse a bbox across pages.
- Fill `source_item_ids` with IDs copied verbatim from explicit ID fields in bronze. Never make IDs up, and never use array positions, MCIDs, or text snippets as IDs.
- If the evidence has no explicit ID (including PDF-native-only evidence), leave the IDs empty and add a properties entry `key="source_note"` saying what you measured and where: page, evidence type, coordinate space/rotation, and the bbox or judgment it supports. It must be specific enough for someone else to reproduce — "measured from PDF" is not.
- Corrections may only remove an existing source ID that is outside the corrected bbox or belongs to another element; say why.
- Don't share source items between siblings unless the page really uses the same glyphs for both, and check that sibling bboxes don't overlap.
- On rotated pages, use the coordinate space the bronze manifest declares and confirm the bbox lands on the visible element.

## Recurring margins

- Chrome is established by recurrence: same position, style, and function across pages. Alternating (recto/verso) chrome is two patterns, not one.
- Keep the printed folio text exactly (Roman numerals, offsets); never substitute the PDF page index.
- A title on the left and "Page N" on the right are one running-header/footer element only when baseline evidence shows they sit on one line and that paired row recurs. Then keep visible text in visual order, union the bbox, and merge provenance. Same block, similar wording, or proximity alone is not enough.
- When the pieces are geometrically independent, keep separate header/footer and page-number elements.
- Don't pull page-specific text (section names, dates, continuation notes) into chrome just because it sits in the margin. Split it out when geometry permits; otherwise keep the block as content and flag the uncertainty.
- Annotate each page's occurrence separately; never collapse recurrences into one multi-page element or a synthetic merged string.

## Tables

Representation: Markdown only for a complete rectangular grid with exactly one header row, no spans, no nested or multi-paragraph cells, and no ambiguous continuation. The classifier also forces HTML when a grid wider than two columns has a body row with only one populated cell (a section-label band) — reason `sparse_section`. Everything else is minified HTML.

For HTML tables, record `classification_reasons` only from the list in `app/pdf2md/tables.py` (`no_header`, `multiple_header_rows`, `spanning_cells`, `incomplete_grid`, `nested_content`, `sparse_section`, `multi_paragraph_cell`, `ambiguous_continuation`, `geometry_ambiguous`). Grid reasons are recomputed automatically; the semantic ones exist only if you record them, and unknown strings are ignored — a missing one flips the representation check.

Structure:

- Derive topology from page evidence. During verification, candidate topology can point at a discrepancy but never answers it.
- Visual bands and rules are evidence, not rows. A wrapped label is one logical row; a rule can mark a section break or the boundary between two tables — use it, don't collapse across it.
- `header_row_count` counts logical header tiers, not text lines. A group header spanning leaf headers is two tiers even if either wraps.
- Record a span only when rules, alignment, background, or drawing geometry show its exact coverage — never from wording or nearby emptiness.
- Rebuild multiline cells from all their tokens in visual order, `\n` for intentional breaks, bbox covering every line.
- Cell roles: column headers `header`, stub/section labels `row_header`, values `body`. Assign cells row-major (top to bottom, left to right after spans), never in PDF object order.
- A full-width section label (like `ASSETS:`) stays inside the table as a spanning `row_header` row; it is not a header tier or a sibling heading.
- Titles, unit lines, and other centered page text above the grid stay separate elements unless they align into the table's own structure. Period labels become header cells only when they align one-to-one with value columns; a date stacked over a year in one column is one multiline cell, not two tiers. A centered group label spans exactly the columns its centering and rules cover.
- A detached currency symbol belongs to the amount that follows it in the same row and column; fold helper symbols into their amount cells rather than creating currency-only columns, and never add symbols the page doesn't print.
- Conserve every token you selected for the table exactly once — no invention, no omission, no duplicates in leftover elements. Header and section cells need provenance just like body cells.
- A real but sparse table stays a table (in HTML); don't flatten it to paragraphs, don't turn a lone total strip into an all-header table, and model visually separate sections as separate tables.
- Cross-page tables: one element, one fragment per page. Repeated header tiers and continuation captions are not data. A repeated trailing foot row is kept once (the final copy), and only collapsed when the identical uniform foot row appears on at least three adjoining fragments — this threshold is a shared convention with the extraction code, not an evidence rule.
- After any table edit, regenerate `content` with `render_table` and check byte equality.

## Text

- Classify from layout evidence (font, size, weight, indentation, position), not wording.
- List items: role `list_item`, the exact visible marker in `list_label` (`1.`, `(a)`, `•` — punctuation kept, whitespace not), the body in `content` without the marker. The renderer prints `content` verbatim, so don't duplicate or normalize the marker.
- `list_depth` is structural nesting starting at 0, inferred from indentation and sequence — never from marker style. `(i)` may be Roman or alphabetic; layout decides its depth.
- A hanging-indent continuation joins its item's `content` (across pages too, with fragments per page). A genuinely separate indented paragraph stays separate with null list fields.
- Not lists: numbered section headings (`4.1. Method Details`), years, equation numbers, citations, inline `(A) ... (B) ...` alternatives, and line-split cell contents inside a table region. Numbered headings stay headings with the number in `content`.
- Footnotes: strip the leading label from `content` into `FootnoteStructure.label`; one label = one element; put confidently associated hosts in `reference_element_ids` (authoritative — don't mirror into `linked_element_ids`). Keep inline markers visibly distinct in host text and record them in a `key="inline_footnote_markers"` properties entry (host location, marker, target ID). Never fuse a marker into a word (`Word` + `a` ≠ `Worda`), and verify numeric markers by font/baseline, not substring.
- Preserve code newlines and indentation. Keep semantic hyphens; drop a line-break hyphen only with glyph/margin evidence. Normalize ligatures with NFKC, keeping meaningful symbols.
- Multi-column reading order: column first, top to bottom, with full-width interruptions allowed. Inside a multi-panel figure, follow the figure's own panel layout, not the page columns.
- Figures: the container comes before its internal text children, each linked to it. Keep every extractable label (titles, axes, legends, ticks) exactly once in local reading order — a figure never swallows or duplicates its native text. Reclassify text as figure text only when a confident caption proves the region really is a figure; an uncaptioned raster may just be background art. Inside a figure, `figure_panel_heading` needs real evidence (like explicit `a)`/`(b)` markers); bold or big alone is `figure_text`. Never use `include_in_output=false` to dodge a metric.
- Captions associate by proximity and visual evidence, not just a "Table N" prefix; link caption and target reciprocally via `linked_element_ids`.

## Adjudication

Adjudication compares disputed alternatives against page evidence with option identity hidden: options carry no source IDs, metadata, or origin cues, in randomized order; the source map is only opened after deciding, to record provenance.

- Reference matches evidence → keep it.
- Reference is wrong (whether the candidate is right or both are wrong) → choose C and rebuild the element yourself from evidence. Candidate-only elements also need independent evidence before being added.
- Every correction records its proving evidence and exact churn; candidate agreement proves nothing. Rationales must be packet-specific; boilerplate is invalid. Confidence means evidence strength, and does not replace human audit.

## Comparability

- The four frontmatter versions must exactly match the schema, source catalog, renderer, and evaluator; benchmarks fail before extraction on any mismatch.
- Express every distinction in schema-visible fields; the evaluator cannot see prose.
- Source-catalog v2: cell assignment uses exclusive cell-content item IDs; shared rules support topology but never prove cell text.
- Stored table `content` is byte-identical to the canonical renderer, and both sides count cross-page rows by the same conventions above.
- Never hide content, shift roles, or suppress figure internals to improve a metric.
- Benchmarks fingerprint this whole file; changing it creates a new run identity, and archived runs are never rewritten.

## Validation checklist

1. Parquet roundtrips; the single document ID equals the bronze manifest `source_sha256`.
2. Element IDs unique; order contiguous from zero and visually correct.
3. All types and roles come from the vocabulary above.
4. Every link resolves.
5. Table grids don't overlap; canonical rendering equals stored content.
6. Source IDs fall inside their fragment bboxes and don't leak between siblings.
7. Cross-page fragments ordered; repeated headers/captions excluded from logical content.
8. Margins annotated but excluded from output.
9. Uncertainty reported, never papered over with invented geometry.
10. Initial silver: hash recorded before seeing candidate output. Later cycles: compare code output to silver, resolve discrepancies from evidence, and keep an audit record (including "no change").
