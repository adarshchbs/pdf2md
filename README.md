# pdf2md

A structure-aware PDF-to-Markdown engine built on PyMuPDF. The current rewrite focuses on extracting tables without losing their cell structure, classifying whether each table can be represented faithfully in Markdown, and rendering non-representable tables as compact embedded HTML.

The engine targets born-digital PDFs with reliable native text. Scanned and image-only PDFs are intentionally excluded from the initial corpus and pipeline.

## Current rewrite

The new typed engine lives in `app/pdf2md/` and provides:

- one ordered Parquet row per document element;
- cross-page source fragments and bounding boxes;
- nested table cells with row/column indices and spans;
- strict Markdown-table eligibility;
- minified, complete HTML using `<thead>` and `<tbody>` for all other tables;
- direct PyMuPDF table conversion and candidate Parquet output;
- deterministic page rendering and immutable LiteParse bronze bundles;
- silver/golden annotation provenance and revision metadata.

The legacy parser remains in `app/pymupdf_parser/` while the full-engine rewrite proceeds incrementally.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- born-digital PDF input with a usable native text layer

OCR is disabled in the bronze pipeline.

## Setup

```bash
uv sync --all-groups
make all
```

`make all` formats and lints the new engine with Ruff, runs strict BasedPyright checks, and executes the test suite.

## Document extraction

```bash
uv run pdf2md extract input.pdf output.md --parquet-path candidate.parquet
```

The command emits canonical Markdown and optionally writes the ordered candidate elements to Parquet. Recurring headers and footers remain annotated in Parquet but are omitted from canonical Markdown.

## Table-only extraction

```bash
uv run pdf2md extract-tables input.pdf candidate.parquet
```

Limit extraction to selected one-based pages by repeating `--page`:

```bash
uv run pdf2md extract-tables input.pdf candidate.parquet --page 7 --page 8
```

A table is emitted as a Markdown table only if it is a complete rectangular grid with exactly one header row, no row or column spans, no nested or multi-paragraph cells, and no ambiguous continuation. All other tables use compact HTML such as:

```html
<table><thead><tr><th rowspan="2">Region</th><th colspan="2">Revenue</th></tr><tr><th>2025</th><th>2026</th></tr></thead><tbody><tr><td>North</td><td>10</td><td>12</td></tr></tbody></table>
```

## Bronze data

A bronze bundle contains deterministic page images, raw LiteParse JSON, LiteParse Markdown, LiteParse plain text, parser versions, page-selection metadata, and SHA-256 hashes.

```bash
uv run pdf2md bronze input.pdf data/bronze/example \
  --requested-page 7 \
  --annotated-page 6 \
  --annotated-page 7 \
  --annotated-page 8 \
  --context-page 5 \
  --context-page 9
```

The annotated range is expected to be expanded so paragraphs and tables crossing its original boundary remain complete. Context pages are rendered and parsed but are not part of silver annotation.

Verify an existing immutable bundle with:

```bash
uv run pdf2md verify-bronze data/bronze/example
```

## API and frontend

The existing FastAPI and Vue application can still be run with:

```bash
docker compose up
```

The backend is built from `pyproject.toml` and `uv.lock` using Python 3.13.
