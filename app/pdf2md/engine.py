from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from html import escape as html_escape
from pathlib import Path
from typing import TypeGuard, cast

import pymupdf

from app.pdf2md.continuations import mark_selection_boundary_tables, merge_table_continuations
from app.pdf2md.figures import (
    FigureCandidate,
    classify_contained_figure_text,
    extract_page_figure_candidates,
    figure_elements,
    order_linked_figure_content,
    overlaps_table,
    reject_recurring_figures,
)
from app.pdf2md.pymupdf_geometry import unrotated_page_extent
from app.pdf2md.pymupdf_runtime import open_document
from app.pdf2md.pymupdf_tables import extract_page_table_elements
from app.pdf2md.region_reading_order import ReadingRegion, decide_reading_region_order
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FootnoteStructure,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
    inline_footnote_marker_ranges,
)
from app.pdf2md.semantic_text import (
    SemanticBlock,
    SemanticKind,
    TextBlock,
    TextLine,
    TextSpan,
    classify_semantic_blocks,
    infer_page_column_splits,
    is_caption,
    link_figure_captions,
    link_footnotes,
    normalize_bbox,
    normalize_page_size,
)
from app.pdf2md.source_catalog import (
    SourceCatalog,
    SourceItem,
    build_source_catalog,
    make_source_item,
    source_item_identity_sha256,
    validate_source_catalog,
)
from app.pdf2md.table_provenance import TableDiagnostic
from app.pdf2md.table_quality import normalize_table_text

_MARGIN_FRACTION = 0.12
_PYMUPDF_ITALIC_FLAG = 1 << 1
_PYMUPDF_BOLD_FLAG = 1 << 4
_RAW_SUPERSCRIPT_SEQUENCE_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,3}")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


@dataclass(frozen=True, slots=True)
class DecorationLine:
    x0: float
    y: float
    x1: float
    thickness: float
    page_width: float
    page_height: float
    path_index: int
    segment_count: int
    primitive_kind: str


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    elements: tuple[DocumentElement, ...]
    source_catalog: SourceCatalog
    table_diagnostics: tuple[TableDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        validate_source_catalog(list(self.elements), self.source_catalog)


@dataclass(slots=True)
class _TableSpanLedger:
    table_id: str
    page_number: int
    supported_span_ids: set[str]
    suppressed_span_ids: set[str]
    retained_overlap_span_ids: set[str]


@dataclass(frozen=True, slots=True)
class _TableWordSupport:
    table_id: str
    page_number: int
    words: tuple[SourceItem, ...]


def extract_document_with_catalog(
    pdf_path: Path,
    *,
    pages: list[int] | None = None,
    annotator: str = "pdf2md-pymupdf",
) -> ExtractedDocument:
    document_id = _sha256(pdf_path)
    with open_document(pdf_path) as document:
        document_page_count = document.page_count
        selected_pages = list(range(1, document_page_count + 1)) if pages is None else pages
        _validate_pages(selected_pages, document_page_count)

        text_blocks: list[TextBlock] = []
        source_items: list[SourceItem] = []
        all_tables: list[DocumentElement] = []
        table_diagnostics: list[TableDiagnostic] = []
        all_figure_candidates: list[FigureCandidate] = []
        page_column_splits: dict[int, tuple[float, ...]] = {}
        page_reading_atomic_bboxes: dict[int, tuple[tuple[float, float, float, float], ...]] = {}
        element_reading_rotations: dict[int, int] = {}
        for page_number in selected_pages:
            page = document[page_number - 1]
            rotation = int(page.rotation)
            source_cropbox = pymupdf.Rect(page.cropbox)
            tables = extract_page_table_elements(
                page,
                document_id,
                annotator=annotator,
                source_items=source_items,
                diagnostics=table_diagnostics,
            )
            # Rotating a PyMuPDF page to canonicalize table detection can reset a
            # non-default crop box. Restore the caller-visible source geometry.
            if page.cropbox != source_cropbox:
                page.set_cropbox(source_cropbox)
            table_bboxes = [element.fragments[0].bbox for element in tables]
            layout_blocks = _extract_page_text_blocks(
                page,
                [],
                document_id=document_id,
                source_items=source_items,
            )
            reading_rotation = _page_reading_rotation(layout_blocks)
            element_reading_rotations[page_number] = (rotation - reading_rotation) % 360
            figure_caption_bboxes = [
                block.bbox
                for block in layout_blocks
                if is_caption(block) and re.match(r"^fig(?:ure)?\.?\s+", block.text.strip(), re.IGNORECASE)
            ]
            figure_candidates = extract_page_figure_candidates(
                page,
                table_bboxes,
                figure_caption_bboxes=figure_caption_bboxes,
            )
            # A tightly captioned, bounded figure is stronger evidence than a table
            # detector interpreting its axes or diagram boxes as cells. Never emit both.
            tables = [
                table
                for table in tables
                if not any(
                    overlaps_table(candidate, table.fragments[0].bbox) for candidate in figure_candidates
                )
            ]
            table_bboxes = [element.fragments[0].bbox for element in tables]
            figure_bboxes = [
                BoundingBox(
                    x0=candidate.normalized_bbox[0],
                    y0=candidate.normalized_bbox[1],
                    x1=candidate.normalized_bbox[2],
                    y1=candidate.normalized_bbox[3],
                )
                for candidate in figure_candidates
            ]
            layout_atomic_bboxes = [*table_bboxes, *figure_bboxes]
            output_width, output_height = normalize_page_size(*unrotated_page_extent(page), rotation)
            reading_atomic_bboxes = [
                normalize_bbox(
                    _bbox_tuple(bbox),
                    output_width,
                    output_height,
                    -element_reading_rotations[page_number],
                )
                for bbox in layout_atomic_bboxes
            ]
            page_reading_atomic_bboxes[page_number] = tuple(reading_atomic_bboxes)
            column_evidence = [block for block in layout_blocks if block.reading_rotation == reading_rotation]
            page_column_splits[page_number] = infer_page_column_splits(
                column_evidence,
                atomic_bboxes=reading_atomic_bboxes,
            )
            table_support = _table_word_support(tables, source_items, page_number=page_number)
            suppression_ledgers = {
                table.element_id: _TableSpanLedger(
                    table_id=table.element_id,
                    page_number=page_number,
                    supported_span_ids=set(),
                    suppressed_span_ids=set(),
                    retained_overlap_span_ids=set(),
                )
                for table in tables
            }
            text_blocks.extend(
                _extract_page_text_blocks(
                    page,
                    table_bboxes,
                    document_id=document_id,
                    source_items=source_items,
                    table_support=table_support,
                    suppression_ledgers=suppression_ledgers,
                )
            )
            tables = [
                _with_table_span_suppression_ledger(table, suppression_ledgers[table.element_id])
                for table in tables
            ]
            all_tables.extend(tables)
            all_figure_candidates.extend(figure_candidates)

    recurring_threshold = max(2, (len(selected_pages) + 1) // 2)
    text_blocks = _consolidate_compound_margin_lines(text_blocks, min_pages=recurring_threshold)
    semantic_blocks = _consolidate_compound_margin_semantics(
        classify_semantic_blocks(
            text_blocks,
            recurring_min_pages=recurring_threshold,
            recurring_margin_fraction=_MARGIN_FRACTION,
            page_column_splits=page_column_splits,
            page_atomic_bboxes=page_reading_atomic_bboxes,
        )
    )
    text_elements = link_footnotes(
        _semantic_elements(semantic_blocks, document_id, annotator),
        semantic_blocks,
    )
    figures = figure_elements(
        reject_recurring_figures(
            all_figure_candidates,
            selected_page_count=len(selected_pages),
        ),
        document_id,
        annotator,
    )

    merged_tables = merge_table_continuations(all_tables, context_elements=text_elements)
    merged_tables = mark_selection_boundary_tables(
        merged_tables,
        selected_pages,
        document_page_count,
        context_elements=[*text_elements, *merged_tables],
    )
    tables_by_first_page: dict[int, list[DocumentElement]] = {}
    for table in merged_tables:
        tables_by_first_page.setdefault(table.fragments[0].page_number, []).append(table)
    figures_by_page: dict[int, list[DocumentElement]] = {}
    for figure in figures:
        figures_by_page.setdefault(figure.fragments[0].page_number, []).append(figure)
    text_by_page: dict[int, list[DocumentElement]] = {}
    for element in text_elements:
        text_by_page.setdefault(element.fragments[0].page_number, []).append(element)

    elements: list[DocumentElement] = []
    for page_number in selected_pages:
        elements.extend(
            _insert_tables_in_reading_order(
                text_by_page.get(page_number, []),
                [
                    *tables_by_first_page.get(page_number, []),
                    *figures_by_page.get(page_number, []),
                ],
                column_splits=page_column_splits[page_number],
                rotation=element_reading_rotations[page_number],
            )
        )

    elements = _classify_recurring_margins(
        elements,
        len(selected_pages),
        page_rotations=element_reading_rotations,
    )
    elements = link_figure_captions(elements, page_rotations=element_reading_rotations)
    # Preserve the settled atomic/panel reading order; this pass changes only
    # contained text semantics and provenance links.
    elements = classify_contained_figure_text(elements)
    elements = order_linked_figure_content(elements)
    ordered = tuple(_with_order(element, order) for order, element in enumerate(elements))
    catalog = build_source_catalog(document_id, source_items)
    return ExtractedDocument(
        elements=ordered,
        source_catalog=catalog,
        table_diagnostics=tuple(table_diagnostics),
    )


def extract_document_elements(
    pdf_path: Path,
    *,
    pages: list[int] | None = None,
    annotator: str = "pdf2md-pymupdf",
) -> list[DocumentElement]:
    return list(extract_document_with_catalog(pdf_path, pages=pages, annotator=annotator).elements)


_MARKDOWN_SENSITIVE = frozenset("\\`*_{}[]!|~")


def _escape_inline_text(text: str) -> str:
    escaped = ""
    for character in text:
        html_character = html_escape(character, quote=False)
        escaped += "\\" + html_character if character in _MARKDOWN_SENSITIVE else html_character
    if escaped.startswith(("#", ">", "+ ", "- ")):
        escaped = "\\" + escaped
    escaped = re.sub(r"^(\d+)\. ", r"\1\\. ", escaped)
    return escaped


def _apply_style_semantics(
    element: DocumentElement,
    content: str,
    *,
    content_start: int,
    canonical_footnote_labels: set[str] | None = None,
) -> str:
    """Render escaped source text from an explicit offset in unchanged content."""
    if element.content[content_start : content_start + len(content)] != content:
        raise ValueError("render content does not match its explicit source character range")
    semantic_by_character = [(False, False, False, False)] * len(content)
    for run in element.structure.style_runs:
        start = max(run.start, content_start) - content_start
        end = min(run.end, content_start + len(content)) - content_start
        if end <= start:
            continue
        flags = (run.bold, run.italic, run.underline, run.strikeout)
        semantic_by_character[start:end] = [flags] * (end - start)

    del canonical_footnote_labels
    syntax_by_character = [False] * len(content)
    content_end = content_start + len(content)
    for marker in inline_footnote_marker_ranges(element):
        if marker.end <= content_start or marker.start >= content_end:
            continue
        if marker.start < content_start or marker.end > content_end:
            raise ValueError("render range splits generated footnote marker syntax")
        start = marker.start - content_start
        end = marker.end - content_start
        syntax_by_character[start:end] = [True] * (end - start)

    rendered = ""
    index = 0
    while index < len(content):
        flags = semantic_by_character[index]
        generated_syntax = syntax_by_character[index]
        end = index + 1
        while (
            end < len(content)
            and semantic_by_character[end] == flags
            and syntax_by_character[end] == generated_syntax
        ):
            end += 1
        segment = content[index:end]
        if generated_syntax:
            rendered += segment
            index = end
            continue
        bold, italic, underline, strikeout = flags
        leading_length = len(segment) - len(segment.lstrip())
        trailing_length = len(segment) - len(segment.rstrip())
        core_end = len(segment) - trailing_length if trailing_length else len(segment)
        leading = segment[:leading_length]
        core = _escape_inline_text(segment[leading_length:core_end])
        trailing = segment[core_end:]
        if core:
            prefix = ("<u>" if underline else "") + ("~~" if strikeout else "")
            prefix += ("**" if bold else "") + ("*" if italic else "")
            suffix = ("*" if italic else "") + ("**" if bold else "")
            suffix += ("~~" if strikeout else "") + ("</u>" if underline else "")
            segment = leading + prefix + core + suffix + trailing
        rendered += segment
        index = end
    return rendered


def _render_block_semantics(
    element: DocumentElement,
    content: str,
    *,
    content_start: int,
    canonical_footnote_labels: set[str] | None = None,
) -> str:
    paragraph = element.structure.paragraph
    rendered = _apply_style_semantics(
        element,
        content,
        content_start=content_start,
        canonical_footnote_labels=canonical_footnote_labels,
    )
    if element.element_type == "heading" and paragraph is not None and paragraph.heading_level is not None:
        return f"{'#' * paragraph.heading_level} {rendered}"
    if paragraph is not None and paragraph.role == "list_item" and paragraph.list_label is not None:
        label = "-" if paragraph.list_label in {"•", "▪", "◦", "‣", "⁃", "*", "-"} else paragraph.list_label
        return f"{'  ' * (paragraph.list_depth or 0)}{label} {rendered}"
    return rendered


def _render_element(
    element: DocumentElement,
    *,
    markdown_footnote_label: str | None = None,
    reference_labels: Mapping[str, str] | None = None,
    canonical_footnote_labels: set[str] | None = None,
) -> str:
    content_start = len(element.content) - len(element.content.lstrip())
    content = element.content.strip()
    if element.element_type == "table":
        return content
    footnote = element.structure.footnote
    if (
        footnote is None
        or footnote.label is None
        or not footnote.association_confident
        or not footnote.reference_element_ids
    ):
        protected_labels = set() if canonical_footnote_labels is None else set(canonical_footnote_labels)
        if reference_labels is not None:
            protected_labels.update(reference_labels)
        content = _render_block_semantics(
            element,
            content,
            content_start=content_start,
            canonical_footnote_labels=protected_labels,
        )
        if reference_labels is not None:
            for source_label, rendered_label in reference_labels.items():
                content = content.replace(f"[^{source_label}]", f"[^{rendered_label}]")
        return content
    source_label = footnote.label
    rendered_label = markdown_footnote_label or source_label
    prefix = re.match(
        rf"^(?:\({re.escape(source_label)}\)|{re.escape(source_label)})[.)]?(?:\s+|(?=[A-Za-z]))",
        content,
    )
    body_with_whitespace = content[0 if prefix is None else prefix.end() :]
    body_leading = len(body_with_whitespace) - len(body_with_whitespace.lstrip())
    body = body_with_whitespace.strip()
    body_start = content_start + (0 if prefix is None else prefix.end()) + body_leading
    body = _apply_style_semantics(
        element,
        body,
        content_start=body_start,
        canonical_footnote_labels=canonical_footnote_labels,
    )
    continuation = "\n    ".join(body.splitlines())
    return f"[^{rendered_label}]: {continuation}"


def render_element(element: DocumentElement, *, canonical_footnote_labels: set[str] | None = None) -> str:
    """Render one canonical element without document-level footnote disambiguation."""
    return _render_element(element, canonical_footnote_labels=canonical_footnote_labels)


def _markdown_footnote_labels(elements: Sequence[DocumentElement]) -> dict[str, str]:
    """Assign document-global labels while retaining unique source labels verbatim."""
    grouped: dict[str, list[DocumentElement]] = defaultdict(list)
    for element in elements:
        footnote = element.structure.footnote
        if (
            footnote is not None
            and footnote.label is not None
            and footnote.association_confident
            and footnote.reference_element_ids
        ):
            grouped[footnote.label].append(element)

    result: dict[str, str] = {}
    for source_label, notes in grouped.items():
        if len(notes) == 1:
            result[notes[0].element_id] = source_label
            continue
        page_occurrences: Counter[int] = Counter()
        for note in notes:
            page_number = note.fragments[0].page_number
            page_occurrences[page_number] += 1
            occurrence = page_occurrences[page_number]
            suffix = f"-p{page_number}" + (f"-{occurrence}" if occurrence > 1 else "")
            result[note.element_id] = source_label + suffix
    generated_labels = [label.casefold() for label in result.values()]
    if len(generated_labels) != len(set(generated_labels)):
        raise ValueError("generated footnote labels must be unique")
    return result


@dataclass(frozen=True, slots=True)
class RenderedDocumentElement:
    element_id: str
    markdown: str


def render_document_elements(elements: Sequence[DocumentElement]) -> tuple[RenderedDocumentElement, ...]:
    """Render canonical Markdown once while retaining element boundaries."""
    labels_by_note_id = _markdown_footnote_labels(elements)
    notes_by_id = {
        element.element_id: element
        for element in elements
        if element.structure.footnote is not None and element.element_id in labels_by_note_id
    }
    rendered: list[RenderedDocumentElement] = []
    for element in elements:
        if not element.include_in_output or not element.content.strip():
            continue
        reference_labels: dict[str, str] = {}
        for linked_id in element.structure.linked_element_ids:
            note = notes_by_id.get(linked_id)
            if note is None:
                continue
            footnote = note.structure.footnote
            if footnote is None or footnote.label is None:
                raise AssertionError("renderable note is missing its footnote label")
            rendered_label = labels_by_note_id[linked_id]
            previous = reference_labels.setdefault(footnote.label, rendered_label)
            if previous != rendered_label:
                raise ValueError("one element cannot disambiguate repeated footnote labels")
        rendered.append(
            RenderedDocumentElement(
                element_id=element.element_id,
                markdown=_render_element(
                    element,
                    markdown_footnote_label=labels_by_note_id.get(element.element_id),
                    reference_labels=reference_labels,
                ),
            )
        )
    return tuple(rendered)


def render_document(elements: list[DocumentElement]) -> str:
    return "\n\n".join(item.markdown for item in render_document_elements(elements))


def _table_word_support(
    tables: Sequence[DocumentElement],
    source_items: Sequence[SourceItem],
    *,
    page_number: int,
) -> tuple[_TableWordSupport, ...]:
    items_by_id = {item.source_item_id: item for item in source_items}
    result: list[_TableWordSupport] = []
    for table_element in tables:
        table = table_element.structure.table
        if table is None:
            raise ValueError("table support requires table elements")
        ordered_ids = list(
            dict.fromkeys(
                source_id
                for cell in sorted(table.cells, key=lambda item: (item.row_index, item.column_index))
                for fragment in cell.fragments
                if fragment.page_number == page_number
                for source_id in fragment.source_item_ids
            )
        )
        words: list[SourceItem] = []
        for source_id in ordered_ids:
            item = items_by_id.get(source_id)
            if item is None:
                raise ValueError(f"dangling accepted table cell source_item_id: {source_id}")
            if (
                item.kind != "word"
                or item.coordinate_frame != "source_page"
                or item.document_id != table_element.document_id
                or item.page_number != page_number
            ):
                raise ValueError(f"invalid accepted table cell source_item_id: {source_id}")
            words.append(item)
        result.append(
            _TableWordSupport(
                table_id=table_element.element_id,
                page_number=page_number,
                words=tuple(words),
            )
        )
    return tuple(result)


def _source_item_bbox(item: SourceItem) -> tuple[float, float, float, float]:
    return item.x0, item.y0, item.x1, item.y1


def _bbox_intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    return max(width, 0.0) * max(height, 0.0)


def _span_supported_by_table(
    span_text: str,
    span_bbox: tuple[float, float, float, float],
    support: _TableWordSupport,
) -> tuple[bool, bool]:
    overlapping = [
        word for word in support.words if _bbox_intersection_area(span_bbox, _source_item_bbox(word)) > 0
    ]
    if not overlapping:
        return False, False
    compatible_words = []
    for word in overlapping:
        word_bbox = _source_item_bbox(word)
        word_area = (word_bbox[2] - word_bbox[0]) * (word_bbox[3] - word_bbox[1])
        if word_area > 0 and _bbox_intersection_area(span_bbox, word_bbox) / word_area >= 0.5:
            compatible_words.append(word)
    accounted_text = normalize_table_text(" ".join(word.text or "" for word in compatible_words))
    return True, bool(accounted_text and accounted_text == normalize_table_text(span_text))


def _record_table_span_disposition(
    span_id: str,
    span_text: str,
    span_bbox: tuple[float, float, float, float],
    table_support: Sequence[_TableWordSupport],
    ledgers: Mapping[str, _TableSpanLedger] | None,
) -> bool:
    if not table_support:
        return False
    if ledgers is None:
        raise ValueError("table support requires suppression ledgers")
    overlap: list[_TableWordSupport] = []
    supported: list[_TableWordSupport] = []
    for support in table_support:
        does_overlap, is_supported = _span_supported_by_table(span_text, span_bbox, support)
        if does_overlap:
            overlap.append(support)
        if is_supported:
            supported.append(support)
    if len(overlap) == 1 and supported == overlap:
        ledgers[overlap[0].table_id].suppressed_span_ids.add(span_id)
        return True
    supported_ids = {support.table_id for support in supported}
    for support in overlap:
        ledger = ledgers[support.table_id]
        if support.table_id in supported_ids:
            ledger.supported_span_ids.add(span_id)
        else:
            ledger.retained_overlap_span_ids.add(span_id)
    return False


def _with_table_span_suppression_ledger(
    table: DocumentElement,
    ledger: _TableSpanLedger,
) -> DocumentElement:
    supported = sorted(ledger.supported_span_ids)
    suppressed = sorted(ledger.suppressed_span_ids)
    retained = sorted(ledger.retained_overlap_span_ids)
    if suppressed:
        disposition = "suppressed_supported_spans"
    elif supported:
        disposition = "retained_ambiguous_support"
    elif retained:
        disposition = "retained_unsupported_overlap"
    else:
        disposition = "no_overlap"
    payload = {
        "counts": {
            "retained_overlap": len(retained),
            "supported": len(supported),
            "suppressed": len(suppressed),
        },
        "disposition": disposition,
        "page_number": ledger.page_number,
        "retained_overlap_span_ids": retained,
        "supported_span_ids": supported,
        "suppressed_span_ids": suppressed,
        "table_id": ledger.table_id,
    }
    prop = StructureProperty(
        key="table_span_suppression_v1",
        value=json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
    )
    return table.model_copy(
        update={
            "structure": table.structure.model_copy(
                update={"properties": [*table.structure.properties, prop]}
            )
        }
    )


def _extract_page_text_blocks(
    page: pymupdf.Page,
    table_bboxes: list[BoundingBox],
    *,
    document_id: str | None = None,
    source_items: list[SourceItem] | None = None,
    table_support: Sequence[_TableWordSupport] = (),
    suppression_ledgers: Mapping[str, _TableSpanLedger] | None = None,
) -> list[TextBlock]:
    page_index = page.number
    if page_index is None:
        raise ValueError("page must belong to an open PyMuPDF document")
    payload = cast(Mapping[str, object], page.get_text("dict", sort=False))
    page_width, page_height = unrotated_page_extent(page)
    return _text_blocks_from_pymupdf_dict(
        payload,
        page_number=page_index + 1,
        page_width=page_width,
        page_height=page_height,
        rotation=int(page.rotation),
        table_bboxes=table_bboxes,
        decoration_lines=_page_decoration_lines(page),
        document_id=document_id,
        source_items=source_items,
        table_support=table_support,
        suppression_ledgers=suppression_ledgers,
    )


def _normalized_embedded_footnote_markers(
    raw_text: str, normalized_text: str
) -> tuple[tuple[int, int, str], ...]:
    """Project explicit Unicode superscripts through NFKC without inferring plain digits."""
    markers: list[tuple[int, int, str]] = []
    for match in _RAW_SUPERSCRIPT_SEQUENCE_RE.finditer(raw_text):
        start = len(unicodedata.normalize("NFKC", raw_text[: match.start()]))
        end = len(unicodedata.normalize("NFKC", raw_text[: match.end()]))
        label = match.group().translate(_SUPERSCRIPT_DIGITS)
        if normalized_text[start:end] != label:
            raise ValueError("NFKC footnote marker projection does not match normalized text")
        markers.append((start, end, label))
    return tuple(markers)


def _text_blocks_from_pymupdf_dict(
    payload: Mapping[str, object],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    rotation: int,
    table_bboxes: list[BoundingBox],
    decoration_lines: Sequence[DecorationLine] = (),
    document_id: str | None = None,
    source_items: list[SourceItem] | None = None,
    table_support: Sequence[_TableWordSupport] = (),
    suppression_ledgers: Mapping[str, _TableSpanLedger] | None = None,
) -> list[TextBlock]:
    if type(page_number) is not int:
        raise TypeError("page_number must be an integer")
    if page_number < 1:
        raise ValueError("page_number must be at least one")
    if (document_id is None) != (source_items is None):
        raise ValueError("document_id and source_items must be provided together")
    if table_support and (document_id is None or source_items is None or suppression_ledgers is None):
        raise ValueError("table support requires source collection and suppression ledgers")
    normalized_width, normalized_height = normalize_page_size(page_width, page_height, rotation)
    native_tables = [
        _bbox(normalize_bbox(_bbox_tuple(bbox), normalized_width, normalized_height, -rotation))
        for bbox in table_bboxes
    ]
    raw_blocks = payload.get("blocks")
    if not _is_payload_sequence(raw_blocks):
        raise TypeError("PyMuPDF dict payload must contain a blocks sequence")

    result: list[TextBlock] = []
    duplicate_source_identities: dict[str, int] = {}
    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping):
            raise TypeError("each PyMuPDF block must be a mapping")
        block_type = raw_block.get("type")
        if type(block_type) is not int:
            raise TypeError("PyMuPDF block type must be an integer")
        if block_type != 0:
            continue
        raw_lines = raw_block.get("lines")
        if not _is_payload_sequence(raw_lines):
            raise TypeError("PyMuPDF text block lines must be a sequence")

        components: list[list[TextLine]] = []
        current_component: list[TextLine] = []

        def flush_component() -> None:
            if current_component:
                components.append(current_component.copy())
                current_component.clear()

        for raw_line in raw_lines:
            if not isinstance(raw_line, Mapping):
                raise TypeError("each PyMuPDF text line must be a mapping")
            raw_spans = raw_line.get("spans")
            if not _is_payload_sequence(raw_spans):
                raise TypeError("PyMuPDF text line spans must be a sequence")
            direction = _direction_sequence(raw_line.get("dir", (1.0, 0.0)))

            segments: list[list[TextSpan]] = []
            segment: list[TextSpan] = []
            leading_suppression = False
            had_suppression = False
            for raw_span in raw_spans:
                if not isinstance(raw_span, Mapping):
                    raise TypeError("each PyMuPDF text span must be a mapping")
                raw_text = raw_span.get("text")
                if not isinstance(raw_text, str):
                    raise TypeError("PyMuPDF span text must be a string")
                flags_value = raw_span.get("flags", 0)
                if type(flags_value) is not int:
                    raise TypeError("PyMuPDF span flags must be an integer")
                size_value = raw_span.get("size")
                if type(size_value) not in (int, float):
                    raise TypeError("PyMuPDF span size must be numeric and not boolean")
                numeric_size = cast(int | float, size_value)
                font_value = raw_span.get("font", "")
                if not isinstance(font_value, str):
                    raise TypeError("PyMuPDF span font must be a string")

                text = unicodedata.normalize("NFKC", raw_text)
                normalized_footnote_markers = _normalized_embedded_footnote_markers(raw_text, text)
                if not text:
                    continue
                raw_bbox = _coordinate_sequence(raw_span.get("bbox"), "span bbox")
                raw_origin = raw_span.get("origin")
                baseline_y: float | None = None
                if raw_origin is not None:
                    origin = _point_sequence(raw_origin, "span origin")
                    baseline_y = origin[1]
                underline, strikeout = _decoration_flags(
                    raw_bbox,
                    float(numeric_size),
                    baseline_y,
                    direction,
                    decoration_lines,
                    rotation,
                    native_tables,
                )
                source_item_id: str | None = None
                if document_id is not None and source_items is not None:
                    style = {
                        "flags": flags_value,
                        "font": font_value,
                        "size": float(numeric_size).hex(),
                    }
                    identity_sha256, _, _ = source_item_identity_sha256(
                        document_id=document_id,
                        page_number=page_number,
                        kind="span",
                        canonical_coordinates=raw_bbox,
                        text=raw_text,
                        style=style,
                    )
                    duplicate_source_identities[identity_sha256] = (
                        duplicate_source_identities.get(identity_sha256, 0) + 1
                    )
                    source_item = make_source_item(
                        document_id=document_id,
                        page_number=page_number,
                        kind="span",
                        coordinate_frame="source_page",
                        coordinates=raw_bbox,
                        canonical_coordinates=raw_bbox,
                        text=raw_text,
                        style=style,
                        duplicate_index=duplicate_source_identities[identity_sha256],
                    )
                    source_item_id = source_item.source_item_id
                    source_items.append(source_item)
                if raw_bbox[2] == raw_bbox[0] and raw_bbox[3] > raw_bbox[1] and segment:
                    # Some native fonts encode an overprinted character with zero
                    # advance (for example ``S`` followed by a zero-width ``t``).
                    # It has no independent box, but remains extractive text owned by
                    # the preceding span; dropping it corrupts the source string.
                    existing_text_length = len(segment[-1].text)
                    segment[-1] = replace(
                        segment[-1],
                        text=segment[-1].text + text,
                        normalized_footnote_markers=(
                            *segment[-1].normalized_footnote_markers,
                            *(
                                (start + existing_text_length, end + existing_text_length, label)
                                for start, end, label in normalized_footnote_markers
                            ),
                        ),
                        source_item_ids=tuple(
                            dict.fromkeys((
                                *segment[-1].source_item_ids,
                                *((source_item_id,) if source_item_id else ()),
                            ))
                        ),
                    )
                    continue
                suppressed = False
                if source_item_id is not None and normalize_table_text(text):
                    suppressed = _record_table_span_disposition(
                        source_item_id,
                        raw_text,
                        raw_bbox,
                        table_support,
                        suppression_ledgers,
                    )
                if suppressed:
                    had_suppression = True
                    if not segments and not segment:
                        leading_suppression = True
                    if segment:
                        segments.append(segment)
                        segment = []
                    continue

                segment.append(
                    TextSpan(
                        text=text,
                        bbox=raw_bbox,
                        font_size=float(numeric_size),
                        font_name=font_value,
                        is_bold=bool(flags_value & _PYMUPDF_BOLD_FLAG),
                        is_italic=bool(flags_value & _PYMUPDF_ITALIC_FLAG),
                        is_underline=underline,
                        is_strikeout=strikeout,
                        baseline_y=baseline_y,
                        source_item_ids=() if source_item_id is None else (source_item_id,),
                        normalized_footnote_markers=normalized_footnote_markers,
                    )
                )
            if segment:
                segments.append(segment)
            if not segments:
                if had_suppression:
                    flush_component()
                continue
            if leading_suppression:
                flush_component()
            for index, spans in enumerate(segments):
                if index:
                    flush_component()
                line_bbox = _union_bboxes([_bbox(span.bbox) for span in spans])
                line = TextLine(
                    spans=tuple(spans),
                    bbox=_bbox_tuple(line_bbox),
                    direction=direction,
                )
                if line.text:
                    current_component.append(line)
            if had_suppression:
                flush_component()
        flush_component()

        for lines in components:
            block_bbox = _union_bboxes([_bbox(line.bbox) for line in lines])
            result.append(
                TextBlock(
                    lines=tuple(lines),
                    bbox=_bbox_tuple(block_bbox),
                    page_number=page_number,
                    page_width=page_width,
                    page_height=page_height,
                    rotation=rotation,
                )
            )
    return result


def _consolidate_compound_margin_lines(blocks: Sequence[TextBlock], *, min_pages: int) -> list[TextBlock]:
    """Join recurring header text and its page label when native baselines agree."""
    if min_pages < 2:
        raise ValueError("compound margin consolidation requires at least two pages")

    top_blocks = [
        block
        for block in blocks
        if len(block.lines) == 1
        and block.reading_bbox[3] <= block.reading_page_size[1] * _MARGIN_FRACTION
        and abs(block.lines[0].direction[0]) >= 0.9
        and abs(block.lines[0].direction[1]) <= 0.1
    ]
    page_labels: dict[int, tuple[TextBlock, int]] = {}
    pages_by_offset: dict[int, set[int]] = defaultdict(set)
    for block in top_blocks:
        match = re.fullmatch(r"page\s+(\d+)", block.text.strip(), re.IGNORECASE)
        if match is None:
            continue
        printed_page = int(match.group(1))
        page_labels[id(block)] = block, printed_page
        pages_by_offset[printed_page - block.page_number].add(block.page_number)
    accepted_offsets = {
        offset for offset, page_numbers in pages_by_offset.items() if len(page_numbers) >= min_pages
    }

    recurring_pages: dict[str, set[int]] = defaultdict(set)
    for block in top_blocks:
        if id(block) in page_labels:
            continue
        key = " ".join(block.text.casefold().split())
        if key and "continued" not in key:
            recurring_pages[key].add(block.page_number)
    recurring_keys = {key for key, page_numbers in recurring_pages.items() if len(page_numbers) >= min_pages}

    companions_by_page: dict[int, list[TextBlock]] = defaultdict(list)
    for block in top_blocks:
        if " ".join(block.text.casefold().split()) in recurring_keys:
            companions_by_page[block.page_number].append(block)

    replacements: dict[int, TextBlock] = {}
    consumed: set[int] = set()
    block_indices = {id(block): index for index, block in enumerate(blocks)}
    for label, printed_page in page_labels.values():
        if printed_page - label.page_number not in accepted_offsets:
            continue
        compatible = [
            companion
            for companion in companions_by_page[label.page_number]
            if _same_native_margin_baseline(companion, label)
        ]
        if len(compatible) != 1:
            continue
        companion = compatible[0]
        merged = _merge_native_margin_blocks(companion, label)
        first_index = min(block_indices[id(companion)], block_indices[id(label)])
        replacements[first_index] = merged
        consumed.update((id(companion), id(label)))

    return [
        replacements[index] if index in replacements else block
        for index, block in enumerate(blocks)
        if id(block) not in consumed or index in replacements
    ]


def _same_native_margin_baseline(first: TextBlock, second: TextBlock) -> bool:
    if (
        first.page_number != second.page_number
        or first.page_width != second.page_width
        or first.page_height != second.page_height
        or first.rotation != second.rotation
    ):
        return False
    first_bbox = first.bbox
    second_bbox = second.bbox
    first_height = first_bbox[3] - first_bbox[1]
    second_height = second_bbox[3] - second_bbox[1]
    vertical_overlap = min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1])
    center_delta = abs((first_bbox[1] + first_bbox[3]) - (second_bbox[1] + second_bbox[3])) / 2
    horizontal_gap = max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)
    return (
        vertical_overlap / min(first_height, second_height) >= 0.6
        and center_delta <= max(first_height, second_height) * 0.25
        and horizontal_gap >= first.page_width * 0.08
    )


def _merge_native_margin_blocks(first: TextBlock, second: TextBlock) -> TextBlock:
    left, right = sorted((first, second), key=lambda item: item.bbox[0])
    bbox = (
        min(left.bbox[0], right.bbox[0]),
        min(left.bbox[1], right.bbox[1]),
        max(left.bbox[2], right.bbox[2]),
        max(left.bbox[3], right.bbox[3]),
    )
    left_span = left.lines[0].spans[0]
    right_span = right.lines[0].spans[0]
    merged_span = TextSpan(
        text=f"{left.text} {right.text}",
        bbox=bbox,
        font_size=(left.font_size + right.font_size) / 2,
        font_name=left_span.font_name if left_span.font_name == right_span.font_name else "",
        is_bold=left.is_bold and right.is_bold,
        is_italic=all(span.is_italic for item in (left, right) for span in item.lines[0].spans),
        is_underline=all(span.is_underline for item in (left, right) for span in item.lines[0].spans),
        is_strikeout=all(span.is_strikeout for item in (left, right) for span in item.lines[0].spans),
        baseline_y=(left_span.baseline_y if left_span.baseline_y == right_span.baseline_y else None),
        source_item_ids=tuple(
            dict.fromkeys(
                source_item_id
                for item in (left, right)
                for span in item.lines[0].spans
                for source_item_id in span.source_item_ids
            )
        ),
    )
    return TextBlock(
        lines=(TextLine(spans=(merged_span,), bbox=bbox, direction=left.lines[0].direction),),
        bbox=bbox,
        page_number=left.page_number,
        page_width=left.page_width,
        page_height=left.page_height,
        rotation=left.rotation,
    )


def _consolidate_compound_margin_semantics(
    semantics: tuple[SemanticBlock, ...],
) -> tuple[SemanticBlock, ...]:
    """Keep a same-baseline running header and page label as one semantic element."""
    headers_by_page: dict[int, list[tuple[int, SemanticBlock]]] = defaultdict(list)
    numbers_by_page: dict[int, list[tuple[int, SemanticBlock]]] = defaultdict(list)
    for index, semantic in enumerate(semantics):
        if len(semantic.source_blocks) != 1 or len(semantic.source_blocks[0].lines) != 1:
            continue
        page_number = semantic.source_blocks[0].page_number
        if semantic.kind == "recurring_margin" and semantic.margin_role == "running_header":
            headers_by_page[page_number].append((index, semantic))
        elif (
            semantic.kind == "page_number"
            and semantic.margin_role == "page_number"
            and re.fullmatch(r"page\s+\d+", semantic.text.strip(), re.IGNORECASE) is not None
        ):
            numbers_by_page[page_number].append((index, semantic))

    replacements: dict[int, SemanticBlock] = {}
    consumed: set[int] = set()
    for page_number in headers_by_page.keys() & numbers_by_page.keys():
        compatible = [
            (header_index, header, number_index, number)
            for header_index, header in headers_by_page[page_number]
            for number_index, number in numbers_by_page[page_number]
            if _same_native_margin_baseline(header.source_blocks[0], number.source_blocks[0])
        ]
        if len(compatible) != 1:
            continue
        header_index, header, number_index, number = compatible[0]
        merged_source = _merge_native_margin_blocks(header.source_blocks[0], number.source_blocks[0])
        replacement_index = min(header_index, number_index)
        replacements[replacement_index] = SemanticBlock(
            kind="recurring_margin",
            text=merged_source.text,
            source_blocks=(merged_source,),
            margin_role="running_header",
        )
        consumed.update((header_index, number_index))

    return tuple(
        replacements[index] if index in replacements else semantic
        for index, semantic in enumerate(semantics)
        if index not in consumed or index in replacements
    )


def _semantic_elements(
    semantic_blocks: tuple[SemanticBlock, ...],
    document_id: str,
    annotator: str,
) -> list[DocumentElement]:
    counts: Counter[tuple[int, SemanticKind]] = Counter()
    elements: list[DocumentElement] = []
    for semantic in semantic_blocks:
        page_numbers = semantic.page_numbers
        page_number = page_numbers[0]
        counts[(page_number, semantic.kind)] += 1
        element_type, role, include_in_output = _semantic_role(semantic)
        structure = ElementStructure(
            paragraph=ParagraphStructure(
                role=role,
                heading_level=semantic.heading_level,
                list_depth=semantic.list_depth,
                list_label=semantic.list_label,
                title_evidence=_title_evidence(semantic),
                heading_evidence=_heading_evidence(semantic),
                list_evidence=_list_evidence(semantic),
            ),
            footnote=(
                FootnoteStructure(label=None, reference_element_ids=[], association_confident=False)
                if semantic.kind == "footnote"
                else None
            ),
            style_runs=list(semantic.style_runs),
        )
        elements.append(
            DocumentElement(
                document_id=document_id,
                element_id=(f"page-{page_number}-{semantic.kind}-{counts[(page_number, semantic.kind)]}"),
                order=len(elements),
                element_type=element_type,
                content=semantic.text,
                format="text",
                include_in_output=include_in_output,
                fragments=[_fragment(source) for source in semantic.source_blocks],
                structure=structure,
                annotation=AnnotationMetadata(
                    stage="candidate",
                    revision=1,
                    annotator=annotator,
                    confidence=1.0,
                    adjudication_status="unreviewed",
                ),
            )
        )
    return elements


def _heading_evidence(semantic: SemanticBlock) -> list[str]:
    if semantic.kind != "heading":
        return []
    block = semantic.source_blocks[0]
    evidence = ["short_nonterminal_text", "inferred_heading_level"]
    if block.is_bold:
        evidence.append("bold_majority")
    return evidence


def _title_evidence(semantic: SemanticBlock) -> list[str]:
    if semantic.kind != "heading" or semantic.heading_level != 1:
        return []
    block = semantic.source_blocks[0]
    if block.page_number != 1 or block.reading_bbox[1] >= block.reading_page_size[1] * 0.35:
        return []
    return ["heading_level_1", "first_page_upper_region"]


def _list_evidence(semantic: SemanticBlock) -> list[str]:
    if semantic.paragraph_role != "list_item" or semantic.list_label is None:
        return []
    evidence = ["explicit_leading_label"]
    if semantic.list_depth is not None and semantic.list_depth > 0:
        evidence.append("nested_label_pattern")
    return evidence


def _semantic_role(semantic: SemanticBlock) -> tuple[str, str, bool]:
    if semantic.kind in {"recurring_margin", "page_number"}:
        if semantic.margin_role is None:
            raise ValueError("margin semantic is missing its role")
        if semantic.margin_role == "running_header":
            return "header", semantic.margin_role, False
        if semantic.margin_role == "running_footer":
            return "footer", semantic.margin_role, False
        block = semantic.source_blocks[0]
        bbox = block.normalized_bbox
        page_height = block.normalized_page_size[1]
        element_type = "header" if bbox[3] <= page_height / 2 else "footer"
        return element_type, semantic.margin_role, False
    if semantic.kind == "note":
        return "note", "table_continuation_marker", False
    role = semantic.paragraph_role or "body" if semantic.kind == "paragraph" else semantic.kind
    return semantic.kind, role, True


def _fragment(block: TextBlock) -> PageFragment:
    page_width, page_height = block.normalized_page_size
    bbox = block.normalized_bbox
    clipped = (
        max(0.0, bbox[0]),
        max(0.0, bbox[1]),
        min(page_width, bbox[2]),
        min(page_height, bbox[3]),
    )
    return PageFragment(
        page_number=block.page_number,
        page_width=page_width,
        page_height=page_height,
        bbox=_bbox(clipped),
        source_item_ids=list(
            dict.fromkeys(
                source_item_id
                for line in block.lines
                for span in line.spans
                for source_item_id in span.source_item_ids
            )
        ),
    )


def _page_reading_rotation(blocks: Sequence[TextBlock]) -> int:
    if not blocks:
        return 0
    weights: Counter[int] = Counter()
    for block in blocks:
        weights[block.reading_rotation] += max(1, len(block.text.strip()))
    return min(weights, key=lambda rotation: (-weights[rotation], rotation))


def _insert_tables_in_reading_order(
    text_elements: list[DocumentElement],
    atomic_elements: list[DocumentElement],
    *,
    column_splits: tuple[float, ...] | None = None,
    rotation: int = 0,
) -> list[DocumentElement]:
    """Place atomic tables or figures using a page's precomputed layout."""
    if not atomic_elements:
        return text_elements.copy()
    elements = [*text_elements, *atomic_elements]
    if len(elements) < 2:
        return elements
    first_fragment = elements[0].fragments[0]
    if first_fragment.page_width is None or first_fragment.page_height is None:
        raise ValueError("page elements must share one known canonical page size")
    page_width, _ = normalize_page_size(
        first_fragment.page_width,
        first_fragment.page_height,
        -rotation,
    )
    regular = [
        element
        for element in elements
        if _element_width(element, rotation) < page_width * 0.72
        and (element.format != "text" or _element_margin_region(element, rotation) is None)
    ]
    splits = _element_column_splits(regular, page_width, rotation) if column_splits is None else column_splits
    regions = tuple(
        ReadingRegion(
            region_id=str(index),
            bbox=_bbox_tuple(_element_reading_bbox(element, rotation)),
            kind=element.element_type,
            native_index=index,
            page_width=normalize_page_size(
                element.fragments[0].page_width,
                element.fragments[0].page_height,
                -rotation,
            )[0],
            page_height=normalize_page_size(
                element.fragments[0].page_width,
                element.fragments[0].page_height,
                -rotation,
            )[1],
        )
        for index, element in enumerate(elements)
        if element.fragments[0].page_width is not None and element.fragments[0].page_height is not None
    )
    if len(regions) != len(elements):
        raise ValueError("page elements must share one known canonical page size")
    decision = decide_reading_region_order(regions, column_splits=splits, rotation=rotation)
    ordered = [elements[int(region.region_id)] for region in decision.regions]
    if decision.status == "native_fallback":
        return ordered
    ordered = _order_element_margin_bands(ordered, rotation)
    return _order_encompassed_figure_panel_leads(ordered, rotation)


def _order_encompassed_figure_panel_leads(
    elements: list[DocumentElement], rotation: int
) -> list[DocumentElement]:
    """Keep same-row panel headings with their independently extracted lead text."""
    result = list(elements)
    figures = [element for element in result if element.element_type == "figure"]
    for figure in figures:
        figure_bbox = _element_reading_bbox(figure, rotation)
        fragment = figure.fragments[0]
        if fragment.page_width is None or fragment.page_height is None:
            raise ValueError("figure fragments require known page dimensions")
        page_width, page_height = normalize_page_size(
            fragment.page_width,
            fragment.page_height,
            -rotation,
        )
        headings = [
            element
            for element in result
            if element.element_type == "heading"
            and element is not figure
            and _bbox_contains(figure_bbox, _element_reading_bbox(element, rotation), threshold=0.95)
        ]
        heading_rows: list[list[DocumentElement]] = []
        for heading in sorted(headings, key=lambda item: _position_key(item, rotation)):
            row = next(
                (
                    candidates
                    for candidates in heading_rows
                    if _same_visual_row(
                        _element_reading_bbox(candidates[0], rotation),
                        _element_reading_bbox(heading, rotation),
                    )
                ),
                None,
            )
            if row is None:
                heading_rows.append([heading])
            else:
                row.append(heading)

        for row in heading_rows:
            ordered_headings = sorted(row, key=lambda item: _element_reading_bbox(item, rotation).x0)
            if len(ordered_headings) < 2 or any(
                _element_reading_bbox(left, rotation).x1 + page_width * 0.03
                > _element_reading_bbox(right, rotation).x0
                for left, right in zip(ordered_headings, ordered_headings[1:], strict=False)
            ):
                continue
            leads: list[DocumentElement] = []
            for heading in ordered_headings:
                heading_bbox = _element_reading_bbox(heading, rotation)
                candidates = [
                    element
                    for element in result
                    if element.element_type == "paragraph"
                    and _bbox_contains(
                        figure_bbox,
                        _element_reading_bbox(element, rotation),
                        threshold=0.95,
                    )
                    and _is_panel_lead(
                        heading_bbox,
                        _element_reading_bbox(element, rotation),
                        page_height,
                    )
                ]
                if not candidates:
                    break
                candidates.sort(key=lambda item: _position_key(item, rotation))
                nearest_y = _element_reading_bbox(candidates[0], rotation).y0
                nearest = [
                    candidate
                    for candidate in candidates
                    if abs(_element_reading_bbox(candidate, rotation).y0 - nearest_y) <= 1.0
                ]
                if len(nearest) != 1:
                    break
                leads.append(nearest[0])
            if len(leads) != len(ordered_headings) or len({id(lead) for lead in leads}) != len(leads):
                continue

            members = [figure, *ordered_headings, *leads]
            positions = sorted(result.index(member) for member in members)
            if positions != list(range(positions[0], positions[-1] + 1)):
                continue
            replacement = [
                figure,
                *(
                    member
                    for heading, lead in zip(ordered_headings, leads, strict=True)
                    for member in (heading, lead)
                ),
            ]
            result[positions[0] : positions[-1] + 1] = replacement
    return result


def _same_visual_row(first: BoundingBox, second: BoundingBox) -> bool:
    overlap = min(first.y1, second.y1) - max(first.y0, second.y0)
    return overlap / min(first.y1 - first.y0, second.y1 - second.y0) >= 0.8


def _bbox_contains(container: BoundingBox, child: BoundingBox, *, threshold: float) -> bool:
    x0 = max(container.x0, child.x0)
    y0 = max(container.y0, child.y0)
    x1 = min(container.x1, child.x1)
    y1 = min(container.y1, child.y1)
    if x1 <= x0 or y1 <= y0:
        return False
    child_area = (child.x1 - child.x0) * (child.y1 - child.y0)
    return (x1 - x0) * (y1 - y0) / child_area >= threshold


def _is_panel_lead(heading: BoundingBox, candidate: BoundingBox, page_height: float) -> bool:
    gap = candidate.y0 - heading.y1
    overlap = max(0.0, min(heading.x1, candidate.x1) - max(heading.x0, candidate.x0))
    narrower_width = min(heading.x1 - heading.x0, candidate.x1 - candidate.x0)
    return (
        candidate.x1 - candidate.x0 >= (heading.x1 - heading.x0) * 0.5
        and overlap / narrower_width >= 0.8
        and 0 <= gap <= page_height * 0.08
    )


def _order_element_margin_bands(elements: list[DocumentElement], rotation: int) -> list[DocumentElement]:
    """Restore vertical margin-band order after column-major atomic insertion."""
    by_region = {
        region: [
            element
            for element in elements
            if element.format == "text" and _element_margin_region(element, rotation) == region
        ]
        for region in ("top", "bottom")
    }

    def order_rows(items: list[DocumentElement]) -> list[DocumentElement]:
        rows: list[list[DocumentElement]] = []
        for element in sorted(items, key=lambda item: _position_key(item, rotation)):
            bbox = _element_reading_bbox(element, rotation)
            if not rows or bbox.y0 >= max(_element_reading_bbox(item, rotation).y1 for item in rows[-1]):
                rows.append([element])
            else:
                rows[-1].append(element)
        return [
            element
            for row in rows
            for element in sorted(
                row,
                key=lambda item: (
                    _element_reading_bbox(item, rotation).x0,
                    _element_reading_bbox(item, rotation).y0,
                ),
            )
        ]

    margin_ids = {id(element) for items in by_region.values() for element in items}
    middle = [element for element in elements if id(element) not in margin_ids]
    return [*order_rows(by_region["top"]), *middle, *order_rows(by_region["bottom"])]


def _element_margin_region(element: DocumentElement, rotation: int) -> str | None:
    fragment = element.fragments[0]
    if fragment.page_width is None or fragment.page_height is None:
        raise ValueError("page element fragments require known page dimensions")
    bbox = _element_reading_bbox(element, rotation)
    page_height = normalize_page_size(fragment.page_width, fragment.page_height, -rotation)[1]
    if bbox.y1 - bbox.y0 > page_height * 0.04:
        return None
    center_y = (bbox.y0 + bbox.y1) / 2
    if center_y <= page_height * _MARGIN_FRACTION:
        return "top"
    if center_y >= page_height * (1 - _MARGIN_FRACTION):
        return "bottom"
    return None


def _classify_recurring_margins(
    elements: list[DocumentElement],
    selected_page_count: int,
    *,
    page_rotations: Mapping[int, int] | None = None,
) -> list[DocumentElement]:
    candidates: list[tuple[int, str, str]] = []
    for index, element in enumerate(elements):
        if element.element_type != "paragraph" or len(element.fragments) != 1:
            continue
        fragment = element.fragments[0]
        if fragment.page_width is None or fragment.page_height is None:
            continue
        rotation = 0 if page_rotations is None else page_rotations.get(fragment.page_number, 0)
        bbox = _element_reading_bbox(element, rotation)
        page_height = normalize_page_size(fragment.page_width, fragment.page_height, -rotation)[1]
        if bbox.y1 <= page_height * _MARGIN_FRACTION:
            candidates.append((index, "header", _recurrence_key(element.content, fragment.page_number)))
        elif bbox.y0 >= page_height * (1 - _MARGIN_FRACTION):
            candidates.append((index, "footer", _recurrence_key(element.content, fragment.page_number)))

    threshold = max(2, (selected_page_count + 1) // 2)
    pages_by_key: dict[tuple[str, str], set[int]] = {}
    for index, kind, key in candidates:
        if key:
            pages_by_key.setdefault((kind, key), set()).add(elements[index].fragments[0].page_number)
    recurring = {key for key, page_numbers in pages_by_key.items() if len(page_numbers) >= threshold}
    result = list(elements)
    for index, kind, key in candidates:
        if (kind, key) not in recurring:
            continue
        element = result[index]
        role = "running_header" if kind == "header" else "running_footer"
        structure = element.structure.model_copy(update={"paragraph": ParagraphStructure(role=role)})
        payload = element.model_dump(mode="json")
        payload.update(
            element_type=kind,
            include_in_output=False,
            structure=structure.model_dump(mode="json"),
        )
        result[index] = DocumentElement.model_validate(payload)
    return result


def _element_reading_bbox(element: DocumentElement, rotation: int) -> BoundingBox:
    fragment = element.fragments[0]
    if fragment.page_width is None or fragment.page_height is None:
        raise ValueError("page element fragments require known page dimensions")
    return _bbox(
        normalize_bbox(
            _bbox_tuple(fragment.bbox),
            fragment.page_width,
            fragment.page_height,
            -rotation,
        )
    )


def _position_key(element: DocumentElement, rotation: int = 0) -> tuple[float, float]:
    bbox = _element_reading_bbox(element, rotation)
    return bbox.y0, bbox.x0


def _element_width(element: DocumentElement, rotation: int = 0) -> float:
    bbox = _element_reading_bbox(element, rotation)
    return bbox.x1 - bbox.x0


def _element_center_x(element: DocumentElement, rotation: int = 0) -> float:
    bbox = _element_reading_bbox(element, rotation)
    return (bbox.x0 + bbox.x1) / 2


def _element_column_splits(
    elements: list[DocumentElement], page_width: float, rotation: int = 0
) -> tuple[float, ...]:
    if len(elements) < 2:
        return ()
    centers = sorted(_element_center_x(element, rotation) for element in elements)
    tolerance = page_width * 0.03
    splits: list[float] = []
    for left_center, right_center in zip(centers, centers[1:], strict=False):
        if right_center - left_center < page_width * 0.12:
            continue
        split = (left_center + right_center) / 2
        has_left = any(
            _element_reading_bbox(element, rotation).x1 <= split + tolerance for element in elements
        )
        has_right = any(
            _element_reading_bbox(element, rotation).x0 >= split - tolerance for element in elements
        )
        if has_left and has_right:
            splits.append(split)
    return tuple(splits)


def join_text_lines(lines: list[str]) -> str:
    content = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if content.endswith("\N{SOFT HYPHEN}"):
            content = content[:-1] + stripped
        elif content.endswith("-"):
            content += stripped
        elif content:
            content += " " + stripped
        else:
            content = stripped
    return content


def _union_bboxes(bboxes: list[BoundingBox]) -> BoundingBox:
    if not bboxes:
        raise ValueError("cannot union an empty bbox collection")
    return BoundingBox(
        x0=min(bbox.x0 for bbox in bboxes),
        y0=min(bbox.y0 for bbox in bboxes),
        x1=max(bbox.x1 for bbox in bboxes),
        y1=max(bbox.y1 for bbox in bboxes),
    )


def _bbox(values: Sequence[float]) -> BoundingBox:
    if len(values) != 4:
        raise ValueError(f"bbox must have four coordinates: {values}")
    return BoundingBox(x0=values[0], y0=values[1], x1=values[2], y1=values[3])


def _page_decoration_lines(page: pymupdf.Page) -> tuple[DecorationLine, ...]:
    """Retain only isolated, interior, non-page-wide horizontal line paths."""
    page_width, page_height = unrotated_page_extent(page)
    result: list[DecorationLine] = []
    for path_index, drawing in enumerate(page.get_drawings()):
        if not isinstance(drawing, Mapping):
            raise TypeError("PyMuPDF drawing must be a mapping")
        width_value = drawing.get("width", 0.0)
        if width_value is None:
            stroke_width = 0.0
        elif type(width_value) in (int, float):
            stroke_width = float(width_value)
        else:
            raise TypeError("PyMuPDF drawing width must be numeric or null")
        items = drawing.get("items")
        if not _is_payload_sequence(items):
            raise TypeError("PyMuPDF drawing items must be a sequence")
        if len(items) != 1:
            continue
        item = items[0]
        if not _is_payload_sequence(item) or not item:
            raise TypeError("PyMuPDF drawing item must be a nonempty sequence")
        if item[0] != "l" or len(item) < 3:
            continue
        first = _native_point(item[1], "drawing line start")
        second = _native_point(item[2], "drawing line end")
        x0, x1 = sorted((first[0], second[0]))
        y = (first[1] + second[1]) / 2
        if (
            abs(first[1] - second[1]) > 0.25
            or x1 <= x0
            or x1 - x0 >= page_width * 0.5
            or y <= page_height * 0.02
            or y >= page_height * 0.98
            or not 0 <= stroke_width <= 2.0
        ):
            continue
        result.append(
            DecorationLine(
                x0=x0,
                y=y,
                x1=x1,
                thickness=stroke_width,
                page_width=page_width,
                page_height=page_height,
                path_index=path_index,
                segment_count=1,
                primitive_kind="line",
            )
        )
    return tuple(result)


def _native_point(value: object, field_name: str) -> tuple[float, float]:
    x = getattr(value, "x", None)
    y = getattr(value, "y", None)
    if type(x) not in (int, float) or type(y) not in (int, float):
        raise TypeError(f"PyMuPDF {field_name} must expose numeric x/y coordinates")
    numeric_x = cast(int | float, x)
    numeric_y = cast(int | float, y)
    point = float(numeric_x), float(numeric_y)
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"PyMuPDF {field_name} coordinates must be finite")
    return point


def _decoration_flags(
    bbox: tuple[float, float, float, float],
    font_size: float,
    baseline_y: float | None,
    direction: tuple[float, float],
    lines: Sequence[DecorationLine],
    page_rotation: int = 0,
    table_bboxes: Sequence[BoundingBox] = (),
) -> tuple[bool, bool]:
    """Derive decorations only when text and drawings share a proven coordinate frame."""
    if type(page_rotation) is not int or page_rotation % 90 != 0:
        raise ValueError("page rotation must be an integer multiple of 90 degrees")
    if page_rotation % 360 != 0:
        return False, False
    if baseline_y is None or abs(direction[0]) < 0.98 or abs(direction[1]) > 0.1:
        return False, False
    span_width = bbox[2] - bbox[0]
    if span_width <= 0:
        return False, False
    matches: list[tuple[bool, bool]] = []
    for line in lines:
        if line.segment_count != 1 or line.primitive_kind != "line":
            continue
        if any(
            table.x0 <= line.x1 and table.x1 >= line.x0 and table.y0 <= line.y <= table.y1
            for table in table_bboxes
        ):
            continue
        overlap = min(bbox[2], line.x1) - max(bbox[0], line.x0)
        if (
            overlap < span_width * 0.65
            or line.x0 < bbox[0] - font_size * 0.1
            or line.x1 > bbox[2] + font_size * 0.1
            or line.x1 - line.x0 > span_width * 1.12
            or line.thickness > min(1.25, font_size * 0.12)
            or line.x1 - line.x0 >= line.page_width * 0.5
            or line.y <= line.page_height * 0.02
            or line.y >= line.page_height * 0.98
        ):
            continue
        baseline_delta = line.y - baseline_y
        underline = 0 <= baseline_delta <= font_size * 0.2
        strikeout = -font_size * 0.45 <= baseline_delta <= -font_size * 0.2
        if underline or strikeout:
            matches.append((underline, strikeout))
    return matches[0] if len(matches) == 1 else (False, False)


def _point_sequence(value: object, field_name: str) -> tuple[float, float]:
    if not _is_payload_sequence(value) or len(value) != 2:
        raise ValueError(f"PyMuPDF {field_name} must have two coordinates: {value}")
    if not all(type(coordinate) in (int, float) for coordinate in value):
        raise TypeError(f"PyMuPDF {field_name} coordinates must be numeric and not boolean")
    numeric_values = cast(Sequence[int | float], value)
    point = float(numeric_values[0]), float(numeric_values[1])
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"PyMuPDF {field_name} coordinates must be finite")
    return point


def _coordinate_sequence(value: object, field_name: str) -> tuple[float, float, float, float]:
    if not _is_payload_sequence(value) or len(value) != 4:
        raise ValueError(f"PyMuPDF {field_name} must have four coordinates: {value}")
    if not all(type(coordinate) in (int, float) for coordinate in value):
        raise TypeError(f"PyMuPDF {field_name} coordinates must be numeric and not boolean")
    numeric_values = cast(Sequence[int | float], value)
    coordinates = cast(
        tuple[float, float, float, float], tuple(float(coordinate) for coordinate in numeric_values)
    )
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise ValueError(f"PyMuPDF {field_name} coordinates must be finite")
    return coordinates


def _direction_sequence(value: object) -> tuple[float, float]:
    if not _is_payload_sequence(value) or len(value) != 2:
        raise ValueError(f"PyMuPDF text direction must have two coordinates: {value}")
    if not all(type(coordinate) in (int, float) for coordinate in value):
        raise TypeError("PyMuPDF text direction coordinates must be numeric and not boolean")
    numeric_values = cast(Sequence[int | float], value)
    direction = cast(tuple[float, float], tuple(float(coordinate) for coordinate in numeric_values))
    if not all(math.isfinite(coordinate) for coordinate in direction):
        raise ValueError("PyMuPDF text direction coordinates must be finite")
    if math.hypot(*direction) == 0:
        raise ValueError("PyMuPDF text direction must be nonzero")
    return direction


def _is_payload_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _bbox_tuple(bbox: BoundingBox) -> tuple[float, float, float, float]:
    return bbox.x0, bbox.y0, bbox.x1, bbox.y1


def _recurrence_key(content: str, page_number: int) -> str:
    normalized = re.sub(rf"(?<!\d){page_number}(?!\d)", "#", content.casefold())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return "" if "continued" in normalized else normalized


def _validate_pages(pages: list[int], page_count: int) -> None:
    if not all(type(page) is int for page in pages):
        raise TypeError("pages must contain integers and not booleans")
    if pages != sorted(set(pages)):
        raise ValueError("pages must be sorted and unique")
    invalid = [page for page in pages if page < 1 or page > page_count]
    if invalid:
        raise ValueError(f"pages outside the document range: {invalid}")


def _with_order(element: DocumentElement, order: int) -> DocumentElement:
    payload = element.model_dump(mode="json")
    payload["order"] = order
    return DocumentElement.model_validate(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
