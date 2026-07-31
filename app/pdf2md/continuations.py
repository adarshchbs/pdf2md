from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from app.pdf2md.schema import (
    BoundingBox,
    DocumentElement,
    ElementStructure,
    PageFragment,
    StructureProperty,
    TableCell,
    TableStructure,
)
from app.pdf2md.tables import TableFeatures, classify_table, render_table

_TOP_PAGE_FRACTION = 0.25
_BOTTOM_PAGE_FRACTION = 0.75
_REPEATED_HEADER_BOTTOM_FRACTION = 0.65
_COLUMN_TOLERANCE = 0.03
_CAPTION_BOUNDARY_TOLERANCE = 0.01


@dataclass(frozen=True)
class _CaptionEvidence:
    identifier: str
    continued: bool
    element: DocumentElement


def mark_selection_boundary_tables(
    elements: list[DocumentElement],
    selected_pages: list[int],
    document_page_count: int,
    *,
    context_elements: list[DocumentElement] | None = None,
) -> list[DocumentElement]:
    """Mark tables that may continue beyond an extracted page selection."""
    selected = set(selected_pages)
    result: list[DocumentElement] = []
    for element in elements:
        if not element.fragments or element.structure.table is None:
            result.append(element)
            continue
        first = element.fragments[0]
        last = element.fragments[-1]
        missing_previous = first.page_number > 1 and first.page_number - 1 not in selected
        missing_next = last.page_number < document_page_count and last.page_number + 1 not in selected
        ambiguous = (
            missing_previous
            and _touches_selection_boundary(element, first, start=True)
            and not _has_substantive_boundary_context(
                element,
                first,
                context_elements,
                before=True,
            )
        ) or (
            missing_next
            and _touches_selection_boundary(element, last, start=False)
            and not _has_substantive_boundary_context(
                element,
                last,
                context_elements,
                before=False,
            )
        )
        result.append(_with_ambiguous_continuation(element) if ambiguous else element)
    return result


def merge_table_continuations(
    elements: list[DocumentElement],
    *,
    captions: list[DocumentElement] | None = None,
    context_elements: list[DocumentElement] | None = None,
) -> list[DocumentElement]:
    if captions is not None and context_elements is not None:
        raise ValueError("provide captions or context_elements, not both")
    context = context_elements if context_elements is not None else captions or []
    caption_evidence = _caption_evidence_by_page(context)
    merged: list[DocumentElement] = []
    for element in elements:
        if merged and _is_continuation(
            merged[-1],
            element,
            caption_evidence,
            context if context_elements is not None else None,
        ):
            merged[-1] = _merge_tables(merged[-1], element)
        else:
            if merged and _is_ambiguous_continuation(
                merged[-1],
                element,
                caption_evidence,
                context if context_elements is not None else None,
            ):
                merged[-1] = _with_ambiguous_continuation(merged[-1])
                element = _with_ambiguous_continuation(element)
            merged.append(element)
    return [_with_order(element, order) for order, element in enumerate(merged)]


def _is_continuation(
    previous: DocumentElement,
    current: DocumentElement,
    caption_evidence: dict[int, _CaptionEvidence],
    context_elements: list[DocumentElement] | None,
) -> bool:
    previous_table = previous.structure.table
    current_table = current.structure.table
    if previous_table is None or current_table is None:
        return False
    if previous.element_type != "table" or current.element_type != "table":
        return False
    if previous_table.column_count != current_table.column_count:
        return False
    if not previous.fragments or not current.fragments:
        return False

    previous_fragment = previous.fragments[-1]
    current_fragment = current.fragments[0]
    if current_fragment.page_number != previous_fragment.page_number + 1:
        return False
    previous_axis = _element_row_axis(previous, previous_fragment.page_number)
    current_axis = _element_row_axis(current, current_fragment.page_number)
    if previous_axis is None or current_axis is None or previous_axis != current_axis:
        return False
    previous_header = _header_signature(previous_table)
    current_header = _header_signature(current_table)
    headers_match = not (previous_header or current_header) or previous_header == current_header
    if _caption_identity_conflicts(
        previous,
        current,
        caption_evidence,
        context_elements,
        previous_axis,
    ):
        return False
    has_explicit_continuation = _has_matching_continued_caption(
        previous,
        current,
        caption_evidence,
        context_elements,
        previous_axis,
    )
    if not headers_match and not has_explicit_continuation:
        return False
    if not _adjoins_reading_boundaries(
        previous_fragment,
        current_fragment,
        previous_axis,
        repeated_header=bool(previous_header),
    ):
        return False
    return _columns_align(previous, current)


def _is_ambiguous_continuation(
    previous: DocumentElement,
    current: DocumentElement,
    caption_evidence: dict[int, _CaptionEvidence],
    context_elements: list[DocumentElement] | None,
) -> bool:
    previous_table = previous.structure.table
    current_table = current.structure.table
    if previous_table is None or current_table is None:
        return False
    if previous_table.column_count != current_table.column_count:
        return False
    if not previous.fragments or not current.fragments:
        return False
    previous_fragment = previous.fragments[-1]
    current_fragment = current.fragments[0]
    if current_fragment.page_number != previous_fragment.page_number + 1:
        return False
    previous_header = _header_signature(previous_table)
    current_header = _header_signature(current_table)
    if not previous_header or not current_header:
        return False
    row_axis = _element_row_axis(previous, previous_fragment.page_number)
    current_axis = _element_row_axis(current, current_fragment.page_number)
    if row_axis is None or row_axis != current_axis:
        return False
    if _caption_identity_conflicts(
        previous,
        current,
        caption_evidence,
        context_elements,
        row_axis,
    ):
        return False
    return bool(
        _adjoins_reading_boundaries(
            previous_fragment,
            current_fragment,
            row_axis,
            repeated_header=True,
        )
        and _columns_align(previous, current)
    )


def _with_ambiguous_continuation(element: DocumentElement) -> DocumentElement:
    existing = element.structure.table
    if existing is None:
        raise ValueError("ambiguous continuation requires a table element")
    features = TableFeatures(
        row_count=existing.row_count,
        column_count=existing.column_count,
        header_row_count=existing.header_row_count,
        cells=existing.cells,
        has_ambiguous_continuation=True,
    )
    representation, reasons = classify_table(features)
    reasons = list(dict.fromkeys([*existing.classification_reasons, *reasons]))
    table = existing.model_copy(
        update={
            "representation": representation,
            "classification_reasons": reasons,
        }
    )
    payload = element.model_dump(mode="json")
    payload.update(
        content=render_table(table),
        format=table.representation,
        structure=element.structure.model_copy(update={"table": table}).model_dump(mode="json"),
    )
    return DocumentElement.model_validate(payload)


def _caption_evidence_by_page(
    elements: list[DocumentElement],
) -> dict[int, _CaptionEvidence]:
    candidates: dict[int, list[_CaptionEvidence]] = {}
    for element in elements:
        paragraph = element.structure.paragraph
        if element.element_type != "caption" or paragraph is None or paragraph.role != "caption":
            continue
        parsed = _table_caption_evidence(element.content)
        if parsed is None:
            continue
        pages = {fragment.page_number for fragment in element.fragments}
        if len(pages) != 1:
            continue
        identifier, continued = parsed
        page_number = pages.pop()
        candidates.setdefault(page_number, []).append(
            _CaptionEvidence(identifier=identifier, continued=continued, element=element)
        )
    return {page_number: values[0] for page_number, values in candidates.items() if len(values) == 1}


def _table_caption_evidence(content: str) -> tuple[str, bool] | None:
    normalized = unicodedata.normalize("NFKC", content)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    match = re.match(
        r"^table\s+(\(?[A-Za-z0-9]+(?:\.[A-Za-z0-9]+|[-\N{EN DASH}\N{EM DASH}][A-Za-z0-9]+|\s+[-\N{EN DASH}\N{EM DASH}]\s*\d+[A-Za-z0-9]*)*\)?)(.*)$",
        normalized,
        re.IGNORECASE,
    )
    if match is None:
        return None
    raw_identifier, remainder = match.groups()
    bare_identifier = raw_identifier.strip("()")
    if not any(character.isdigit() for character in bare_identifier) and not (
        len(bare_identifier) == 1 or re.fullmatch(r"[IVXLCDM]+", bare_identifier, re.IGNORECASE)
    ):
        return None
    identifier = re.sub(r"\s*[-\N{EN DASH}\N{EM DASH}]\s*", "-", bare_identifier)
    identifier = re.sub(r"\s*\.\s*", ".", identifier).casefold()
    stripped_remainder = remainder.strip(" \t\r\n:;,.\N{EN DASH}\N{EM DASH}-")
    continued = not bool(re.search(r"\bnot\s+[([]?\s*continued\b", remainder, re.IGNORECASE)) and bool(
        re.search(r"(?:\(\s*continued\s*\)|\[\s*continued\s*\])", remainder, re.IGNORECASE)
        or re.fullmatch(r"continued", stripped_remainder, re.IGNORECASE)
        or re.match(r"^continued\s*[:\-\N{EN DASH}\N{EM DASH}]", stripped_remainder, re.IGNORECASE)
        or re.search(r"[:\-\N{EN DASH}\N{EM DASH}]\s*continued$", stripped_remainder, re.IGNORECASE)
    )
    return identifier, continued


def _is_continuation_marker(content: str) -> bool:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", content)).strip()
    caption = _table_caption_evidence(normalized)
    if caption is not None and caption[1]:
        return True
    return bool(
        re.fullmatch(
            r"(?:table\s+)?(?:continued|continues)(?:\s+(?:on|from|to)\s+(?:the\s+)?(?:next|previous|following|preceding)\s+page)?[.!]?",
            normalized,
            re.IGNORECASE,
        )
    )


def _page_section_identifier(elements: list[DocumentElement], page_number: int) -> str | None:
    identifiers = {
        match.group(1).casefold()
        for element in elements
        if any(fragment.page_number == page_number for fragment in element.fragments)
        and (match := re.search(r"\bpart\s+([IVXLCDM]+|\d+)\b", element.content, re.IGNORECASE)) is not None
    }
    return identifiers.pop() if len(identifiers) == 1 else None


def _caption_identity_conflicts(
    previous: DocumentElement,
    current: DocumentElement,
    evidence_by_page: dict[int, _CaptionEvidence],
    context_elements: list[DocumentElement] | None,
    row_axis: tuple[str, int],
) -> bool:
    previous_page = previous.fragments[-1].page_number
    current_page = current.fragments[0].page_number
    if context_elements is not None:
        previous_section = _page_section_identifier(context_elements, previous_page)
        current_section = _page_section_identifier(context_elements, current_page)
        if (
            previous_section is not None
            and current_section is not None
            and previous_section != current_section
        ):
            return True
    previous_evidence = evidence_by_page.get(previous_page)
    current_evidence = evidence_by_page.get(current_page)
    if previous_evidence is None or current_evidence is None:
        return False
    if previous_evidence.identifier != current_evidence.identifier:
        current_caption_bounds = _element_progress_bounds(
            current_evidence.element,
            current.fragments[0].page_number,
            row_axis,
        )
        current_table_bounds = _progress_bounds(current.fragments[0], row_axis)
        if current_caption_bounds is None:
            return True
        # A distinct caption preceding or overlapping the candidate identifies
        # that table and vetoes the merge. A caption wholly after the candidate
        # belongs to later page content, as in RP2040 Table 330 following the
        # top-of-page continuation of Table 329.
        if current_caption_bounds[0] < current_table_bounds[1]:
            return True
        return bool(
            context_elements is not None
            and _has_intervening_content(
                previous,
                current,
                context_elements,
                row_axis,
                ignored_ids={id(previous_evidence.element), id(current_evidence.element)},
            )
        )
    if context_elements is not None and not _captions_belong_to_adjoining_tables(
        previous,
        current,
        previous_evidence.element,
        current_evidence.element,
        context_elements,
        row_axis,
    ):
        return False
    return not current_evidence.continued


def _has_matching_continued_caption(
    previous: DocumentElement,
    current: DocumentElement,
    evidence_by_page: dict[int, _CaptionEvidence],
    context_elements: list[DocumentElement] | None,
    row_axis: tuple[str, int],
) -> bool:
    previous_page = previous.fragments[-1].page_number
    current_page = current.fragments[0].page_number
    previous_evidence = evidence_by_page.get(previous_page)
    current_evidence = evidence_by_page.get(current_page)
    if not (
        previous_evidence is not None
        and current_evidence is not None
        and current_evidence.continued
        and previous_evidence.identifier == current_evidence.identifier
    ):
        return False
    if context_elements is None:
        return True
    return _captions_belong_to_adjoining_tables(
        previous,
        current,
        previous_evidence.element,
        current_evidence.element,
        context_elements,
        row_axis,
    )


def _has_intervening_content(
    previous: DocumentElement,
    current: DocumentElement,
    context_elements: list[DocumentElement],
    row_axis: tuple[str, int],
    *,
    ignored_ids: set[int],
) -> bool:
    previous_fragment = previous.fragments[-1]
    current_fragment = current.fragments[0]
    previous_table_bounds = _progress_bounds(previous_fragment, row_axis)
    current_table_bounds = _progress_bounds(current_fragment, row_axis)
    for element in context_elements:
        if id(element) in ignored_ids or not element.include_in_output:
            continue
        if element.element_type in {"caption", "header", "footer", "note"}:
            continue
        previous_bounds = _element_progress_bounds(element, previous_fragment.page_number, row_axis)
        if previous_bounds is not None and previous_bounds[1] > previous_table_bounds[1]:
            return True
        current_bounds = _element_progress_bounds(element, current_fragment.page_number, row_axis)
        if current_bounds is not None and current_bounds[0] < current_table_bounds[0]:
            return True
    return False


def _captions_belong_to_adjoining_tables(
    previous: DocumentElement,
    current: DocumentElement,
    previous_caption: DocumentElement,
    current_caption: DocumentElement,
    context_elements: list[DocumentElement],
    row_axis: tuple[str, int],
) -> bool:
    previous_fragment = previous.fragments[-1]
    current_fragment = current.fragments[0]
    previous_caption_bounds = _element_progress_bounds(
        previous_caption, previous_fragment.page_number, row_axis
    )
    current_caption_bounds = _element_progress_bounds(current_caption, current_fragment.page_number, row_axis)
    if previous_caption_bounds is None or current_caption_bounds is None:
        return False
    previous_table_bounds = _progress_bounds(previous_fragment, row_axis)
    current_table_bounds = _progress_bounds(current_fragment, row_axis)
    previous_extent = _axis_extent(previous_fragment, row_axis[0])
    current_extent = _axis_extent(current_fragment, row_axis[0])
    if previous_extent is None or current_extent is None:
        return False
    if (
        previous_caption_bounds[0] > previous_table_bounds[0] + previous_extent * _CAPTION_BOUNDARY_TOLERANCE
        or current_caption_bounds[0] > current_table_bounds[0] + current_extent * _CAPTION_BOUNDARY_TOLERANCE
    ):
        return False

    ignored_ids = {id(previous_caption), id(current_caption)}
    for element in context_elements:
        if id(element) in ignored_ids or not element.include_in_output:
            continue
        if element.element_type in {"caption", "header", "footer", "note"}:
            continue
        previous_bounds = _element_progress_bounds(element, previous_fragment.page_number, row_axis)
        if previous_bounds is not None and previous_bounds[1] > previous_table_bounds[1]:
            return False
        current_bounds = _element_progress_bounds(element, current_fragment.page_number, row_axis)
        if current_bounds is None or current_bounds[0] >= current_table_bounds[0]:
            continue
        if current_bounds[1] <= current_caption_bounds[0]:
            return False
        if not _looks_like_table_residual(element, current_fragment, row_axis):
            return False
    return True


def _element_progress_bounds(
    element: DocumentElement,
    page_number: int,
    row_axis: tuple[str, int],
) -> tuple[float, float] | None:
    bounds = [
        _progress_bounds(fragment, row_axis)
        for fragment in element.fragments
        if fragment.page_number == page_number
    ]
    if not bounds:
        return None
    return min(value[0] for value in bounds), max(value[1] for value in bounds)


def _progress_bounds(fragment: PageFragment, row_axis: tuple[str, int]) -> tuple[float, float]:
    axis, direction = row_axis
    low, high = _fragment_axis_bounds(fragment, axis)
    if direction > 0:
        return low, high
    extent = _axis_extent(fragment, axis)
    if extent is None:
        raise ValueError("progress bounds require a known page extent")
    return extent - high, extent - low


def _axis_extent(fragment: PageFragment, axis: str) -> float | None:
    return fragment.page_width if axis == "x" else fragment.page_height


def _looks_like_table_residual(
    element: DocumentElement,
    table_fragment: PageFragment,
    row_axis: tuple[str, int],
) -> bool:
    column_axis = "y" if row_axis[0] == "x" else "x"
    table_low, table_high = _fragment_axis_bounds(table_fragment, column_axis)
    element_bounds = [
        _fragment_axis_bounds(fragment, column_axis)
        for fragment in element.fragments
        if fragment.page_number == table_fragment.page_number
    ]
    if not element_bounds:
        return False
    element_low = min(bounds[0] for bounds in element_bounds)
    element_high = max(bounds[1] for bounds in element_bounds)
    element_span = element_high - element_low
    table_span = table_high - table_low
    overlap = min(table_high, element_high) - max(table_low, element_low)
    return bool(
        overlap > 0 and element_span >= table_span * 0.8 and overlap / min(element_span, table_span) >= 0.9
    )


def _suppression_properties_for_consumer(
    properties: tuple[StructureProperty, ...] | list[StructureProperty],
    consumer_table_id: str,
) -> list[StructureProperty]:
    result: list[StructureProperty] = []
    for prop in properties:
        if prop.key != "table_span_suppression_v1":
            result.append(prop)
            continue
        payload = json.loads(prop.value)
        if not isinstance(payload, dict):
            raise ValueError("table span suppression ledger must be a JSON object")
        payload["table_id"] = consumer_table_id
        result.append(
            StructureProperty(
                key=prop.key,
                value=json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
            )
        )
    return result


def _merge_tables(previous: DocumentElement, current: DocumentElement) -> DocumentElement:
    previous_table = previous.structure.table
    current_table = current.structure.table
    if previous_table is None or current_table is None:
        raise ValueError("only table elements can be merged")

    previous_table = _collapse_repeated_trailing_foot_rows(previous, previous_table, current, current_table)
    repeated_header_rows = current_table.header_row_count
    joins_split_row = _first_body_row_continues_previous(previous_table, current_table)
    consumed_body_rows = 1 if joins_split_row else 0
    cells = list(previous_table.cells)
    if joins_split_row:
        cells = _merge_split_body_row(
            cells,
            current_table.cells,
            previous_table.row_count - 1,
            repeated_header_rows,
        )
    cells, extended_cell_ids = _extend_leading_rowspan_across_boundary(
        cells,
        current_table,
        previous_table.row_count,
        repeated_header_rows + consumed_body_rows,
    )
    row_offset = previous_table.row_count - consumed_body_rows
    continuation_cells = [
        TableCell.model_validate({
            **cell.model_dump(mode="json"),
            "row_index": row_offset + cell.row_index - repeated_header_rows,
        })
        for cell in current_table.cells
        if cell.row_index >= repeated_header_rows + consumed_body_rows and id(cell) not in extended_cell_ids
    ]
    cells.extend(continuation_cells)
    row_count = previous_table.row_count + current_table.row_count - repeated_header_rows - consumed_body_rows
    features = TableFeatures(
        row_count=row_count,
        column_count=previous_table.column_count,
        header_row_count=previous_table.header_row_count,
        cells=cells,
        has_ambiguous_continuation=any(
            "ambiguous_continuation" in table.classification_reasons
            for table in (previous_table, current_table)
        ),
    )
    representation, reasons = classify_table(features)
    table = TableStructure(
        row_count=row_count,
        column_count=previous_table.column_count,
        header_row_count=previous_table.header_row_count,
        representation=representation,
        classification_reasons=reasons,
        cells=cells,
    )
    payload = previous.model_dump(mode="json")
    payload.update(
        content=render_table(table),
        format=table.representation,
        fragments=[
            *previous.model_dump(mode="json")["fragments"],
            *current.model_dump(mode="json")["fragments"],
        ],
        structure=ElementStructure(
            paragraph=previous.structure.paragraph,
            table=table,
            figure=previous.structure.figure,
            footnote=previous.structure.footnote,
            linked_element_ids=[
                *previous.structure.linked_element_ids,
                current.element_id,
            ],
            properties=list(
                dict.fromkeys(
                    _suppression_properties_for_consumer(
                        [*previous.structure.properties, *current.structure.properties],
                        previous.element_id,
                    )
                )
            ),
        ).model_dump(mode="json"),
    )
    return DocumentElement.model_validate(payload)


def _collapse_repeated_trailing_foot_rows(
    previous_element: DocumentElement,
    previous: TableStructure,
    current_element: DocumentElement,
    current: TableStructure,
) -> TableStructure:
    """Retain one foot row after three adjoining continuation fragments prove repetition."""
    if len(previous_element.fragments) < 2 or not current_element.fragments:
        return previous
    previous_pages = [fragment.page_number for fragment in previous_element.fragments]
    current_page = current_element.fragments[0].page_number
    if len(previous_pages) != len(set(previous_pages)) or previous_pages[-2:] != [
        current_page - 2,
        current_page - 1,
    ]:
        return previous
    current_row = current.row_count - 1
    signature = _complete_row_signature(current, current_row)
    if (
        signature is None
        or current_row < current.header_row_count
        or not _has_uniform_table_foot_layout(signature, current.column_count)
    ):
        return previous
    if _row_source_page(current, current_row) != current_page:
        return previous

    repeated_rows: list[int] = []
    for page in previous_pages[-2:]:
        rows_on_page = {
            cell.row_index
            for cell in previous.cells
            if any(fragment.page_number == page for fragment in cell.fragments)
        }
        if not rows_on_page:
            return previous
        trailing_row = max(rows_on_page)
        if (
            trailing_row < previous.header_row_count
            or _row_source_page(previous, trailing_row) != page
            or _complete_row_signature(previous, trailing_row) != signature
        ):
            return previous
        repeated_rows.append(trailing_row)
    if len(set(repeated_rows)) != 2:
        return previous

    removed = set(repeated_rows)
    cells = [
        cell.model_copy(update={"row_index": cell.row_index - sum(row < cell.row_index for row in removed)})
        for cell in previous.cells
        if cell.row_index not in removed
    ]
    row_count = previous.row_count - len(removed)
    representation, reasons = classify_table(
        TableFeatures(
            row_count=row_count,
            column_count=previous.column_count,
            header_row_count=previous.header_row_count,
            cells=cells,
            has_ambiguous_continuation="ambiguous_continuation" in previous.classification_reasons,
        )
    )
    return previous.model_copy(
        update={
            "row_count": row_count,
            "representation": representation,
            "classification_reasons": reasons,
            "cells": cells,
        }
    )


def _complete_row_signature(
    table: TableStructure,
    row_index: int,
) -> tuple[tuple[int, int, str, str], ...] | None:
    cells = sorted(
        (cell for cell in table.cells if cell.row_index == row_index),
        key=lambda cell: cell.column_index,
    )
    if not cells or any(cell.rowspan != 1 for cell in cells):
        return None
    occupied = {
        column for cell in cells for column in range(cell.column_index, cell.column_index + cell.colspan)
    }
    if occupied != set(range(table.column_count)) or not any(cell.text.strip() for cell in cells):
        return None
    return tuple((cell.column_index, cell.colspan, cell.role, _normalize_text(cell.text)) for cell in cells)


def _has_uniform_table_foot_layout(
    signature: tuple[tuple[int, int, str, str], ...],
    column_count: int,
) -> bool:
    if any(role not in {"body", "row_header"} for _, _, role, _ in signature):
        return False
    populated = {text for _, _, _, text in signature if text}
    return bool(
        len(populated) == 1
        and (
            len(signature) > 1
            or any(column == 0 and colspan == column_count for column, colspan, _, _ in signature)
        )
    )


def _row_source_page(table: TableStructure, row_index: int) -> int | None:
    pages = {
        fragment.page_number
        for cell in table.cells
        if cell.row_index == row_index
        for fragment in cell.fragments
    }
    return pages.pop() if len(pages) == 1 else None


def _first_body_row_continues_previous(previous: TableStructure, current: TableStructure) -> bool:
    if previous.column_count < 3 or current.header_row_count >= current.row_count:
        return False
    split = (previous.column_count + 1) // 2
    previous_columns = _populated_columns(previous, previous.row_count - 1)
    current_columns = _populated_columns(current, current.header_row_count)
    return bool(
        current_columns
        and min(current_columns) >= split
        and previous_columns
        and min(previous_columns) < split
    )


def _populated_columns(table: TableStructure, row_index: int) -> set[int]:
    return {
        column
        for cell in table.cells
        if cell.text.strip() and cell.row_index <= row_index < cell.row_index + cell.rowspan
        for column in range(cell.column_index, cell.column_index + cell.colspan)
    }


def _extend_leading_rowspan_across_boundary(
    previous_cells: list[TableCell],
    current: TableStructure,
    previous_row_count: int,
    first_appended_row: int,
) -> tuple[list[TableCell], set[int]]:
    target_index = next(
        (
            index
            for index, cell in enumerate(previous_cells)
            if cell.column_index == 0
            and cell.role == "row_header"
            and cell.text.strip()
            and cell.row_index + cell.rowspan == previous_row_count
        ),
        None,
    )
    if target_index is None:
        return previous_cells, set()

    target = previous_cells[target_index]
    blank_continuations: list[TableCell] = []
    for row_index in range(first_appended_row, current.row_count):
        row_cells = [
            cell
            for cell in current.cells
            if cell.row_index == row_index
            and cell.column_index == target.column_index
            and cell.colspan == target.colspan
        ]
        if len(row_cells) != 1:
            break
        continuation = row_cells[0]
        if continuation.rowspan != 1 or continuation.text.strip():
            break
        blank_continuations.append(continuation)

    if not blank_continuations:
        return previous_cells, set()
    extended = list(previous_cells)
    extended[target_index] = target.model_copy(
        update={
            "rowspan": target.rowspan + len(blank_continuations),
            "fragments": [
                *target.fragments,
                *(fragment for cell in blank_continuations for fragment in cell.fragments),
            ],
        }
    )
    return extended, {id(cell) for cell in blank_continuations}


def _merge_split_body_row(
    previous_cells: list[TableCell],
    current_cells: list[TableCell],
    previous_row_index: int,
    current_row_index: int,
) -> list[TableCell]:
    merged = list(previous_cells)
    for continuation in current_cells:
        if continuation.row_index != current_row_index or not continuation.text.strip():
            continue
        target_index = next(
            (
                index
                for index, cell in enumerate(merged)
                if cell.row_index <= previous_row_index < cell.row_index + cell.rowspan
                and cell.column_index == continuation.column_index
                and cell.colspan == continuation.colspan
            ),
            None,
        )
        if target_index is None:
            merged.append(continuation.model_copy(update={"row_index": previous_row_index}))
            continue
        target = merged[target_index]
        merged[target_index] = target.model_copy(
            update={
                "text": " ".join(part for part in (target.text.strip(), continuation.text.strip()) if part),
                "fragments": [*target.fragments, *continuation.fragments],
            }
        )
    return merged


def _header_signature(table: TableStructure) -> tuple[tuple[int, int, int, int, str, str], ...]:
    if table.header_row_count == 0:
        return ()
    return tuple(
        (
            cell.row_index,
            cell.column_index,
            cell.rowspan,
            cell.colspan,
            cell.role,
            _normalize_text(cell.text),
        )
        for cell in sorted(
            (cell for cell in table.cells if cell.row_index < table.header_row_count),
            key=lambda cell: (cell.row_index, cell.column_index),
        )
    )


def _is_compact_titled_table(
    element: DocumentElement,
    fragment: PageFragment,
    row_axis: tuple[str, int],
    extent: float,
) -> bool:
    table = element.structure.table
    if table is None or table.row_count < 2:
        return False
    low, high = _progress_bounds(fragment, row_axis)
    if high - low > extent * _TOP_PAGE_FRACTION:
        return False
    first_row = [cell for cell in table.cells if cell.row_index == 0 and cell.text.strip()]
    if len(first_row) != 1:
        return False
    title = first_row[0].text.strip()
    return bool(
        1 < len(title.split()) <= 12
        and not _is_continuation_marker(title)
        and not title.endswith((".", ":", ";"))
    )


def _has_substantive_boundary_context(
    table: DocumentElement,
    fragment: PageFragment,
    context_elements: list[DocumentElement] | None,
    *,
    before: bool,
) -> bool:
    row_axis = _element_row_axis(table, fragment.page_number)
    if row_axis is None:
        return False
    table_bounds = _progress_bounds(fragment, row_axis)
    extent = _axis_extent(fragment, row_axis[0])
    if extent is None:
        return False
    if not before and _is_compact_titled_table(table, fragment, row_axis, extent):
        return True
    if context_elements is None:
        return False
    tolerance = extent * _CAPTION_BOUNDARY_TOLERANCE
    page_context: list[tuple[DocumentElement, tuple[float, float]]] = []
    for element in context_elements:
        if element.document_id != table.document_id or element.element_id == table.element_id:
            continue
        bounds = _element_progress_bounds(element, fragment.page_number, row_axis)
        if bounds is not None:
            page_context.append((element, bounds))

    # An explicit continuation label is direct evidence that a top-of-page
    # fragment began on an omitted page. Do not let an explanatory repeated
    # heading between that label and the detected grid cancel the evidence.
    if any(
        _is_continuation_marker(element.content)
        and (bounds[1] <= table_bounds[0] + tolerance if before else bounds[0] >= table_bounds[1] - tolerance)
        for element, bounds in page_context
    ):
        return False
    if any(
        re.search(r"\bpart\s+(?:[IVXLCDM]+|\d+)\b", element.content, re.IGNORECASE)
        and bounds[1] <= table_bounds[0] + tolerance
        and bounds[0] <= extent * _TOP_PAGE_FRACTION
        for element, bounds in page_context
        if element.element_type not in {"footer", "note"}
    ):
        return True

    for element, bounds in page_context:
        completion_context = bool(
            re.match(r"^(?:see|source|notes?)\b", element.content.strip(), re.IGNORECASE)
        )
        if not element.include_in_output and not completion_context:
            continue
        if element.element_type in {"header", "note"}:
            continue
        if element.element_type == "footer" and not completion_context:
            continue
        caption = _table_caption_evidence(element.content) if element.element_type == "caption" else None
        if caption is not None and caption[1]:
            continue
        trusted_caption = caption is not None
        closely_adjoining = (
            bounds[1] >= table_bounds[0] - tolerance if before else bounds[0] <= table_bounds[1] + tolerance
        )
        if not (completion_context or trusted_caption or closely_adjoining) and (
            bounds[1] <= extent * 0.12 or bounds[0] >= extent * 0.88
        ):
            continue
        if before and bounds[1] <= table_bounds[0] + tolerance:
            return True
        if not before and bounds[0] >= table_bounds[1] - tolerance:
            return True
    return False


def _touches_selection_boundary(element: DocumentElement, fragment: PageFragment, *, start: bool) -> bool:
    row_axis = _element_row_axis(element, fragment.page_number)
    if row_axis is None:
        return False
    axis, direction = row_axis
    extent = fragment.page_width if axis == "x" else fragment.page_height
    if extent is None:
        return False
    low, high = _fragment_axis_bounds(fragment, axis)
    progress_low = low if direction > 0 else extent - high
    progress_high = high if direction > 0 else extent - low
    return (
        progress_low <= extent * _TOP_PAGE_FRACTION
        if start
        else progress_high >= extent * _BOTTOM_PAGE_FRACTION
    )


def _adjoins_reading_boundaries(
    previous: PageFragment,
    current: PageFragment,
    row_axis: tuple[str, int],
    *,
    repeated_header: bool,
) -> bool:
    axis, direction = row_axis
    previous_extent = previous.page_width if axis == "x" else previous.page_height
    current_extent = current.page_width if axis == "x" else current.page_height
    if previous_extent is None or current_extent is None:
        return False
    previous_low, previous_high = _fragment_axis_bounds(previous, axis)
    current_low, current_high = _fragment_axis_bounds(current, axis)
    previous_progress_high = previous_high if direction > 0 else previous_extent - previous_low
    current_progress_low = current_low if direction > 0 else current_extent - current_high
    if current_progress_low > current_extent * _TOP_PAGE_FRACTION:
        return False
    if previous_progress_high >= previous_extent * _BOTTOM_PAGE_FRACTION:
        return True
    previous_progress_low = previous_low if direction > 0 else previous_extent - previous_high
    return bool(
        repeated_header
        and previous_progress_low <= previous_extent * _TOP_PAGE_FRACTION
        and previous_progress_high >= previous_extent * _REPEATED_HEADER_BOTTOM_FRACTION
    )


def _element_row_axis(element: DocumentElement, preferred_page: int) -> tuple[str, int] | None:
    page_numbers = list(
        dict.fromkeys([preferred_page, *(fragment.page_number for fragment in element.fragments)])
    )
    table = element.structure.table
    if table is None:
        raise ValueError("row axis requires a table element")
    evidence = [
        (
            axis,
            sum(
                1
                for cell in table.cells
                for fragment in cell.fragments
                if fragment.page_number == page_number
            ),
        )
        for page_number in page_numbers
        if (axis := _logical_row_axis(element, page_number)) is not None
    ]
    if not evidence:
        return None
    preferred = _logical_row_axis(element, preferred_page)
    axes = list(dict.fromkeys(axis for axis, _ in evidence))
    return max(
        axes,
        key=lambda axis: (
            sum(weight for candidate, weight in evidence if candidate == axis),
            sum(candidate == axis for candidate, _ in evidence),
            axis == preferred,
        ),
    )


def _logical_row_axis(element: DocumentElement, page_number: int) -> tuple[str, int] | None:
    table = element.structure.table
    if table is None:
        raise ValueError("row axis requires a table element")
    bounds_by_row: dict[float, list[BoundingBox]] = {}
    for cell in table.cells:
        logical_center = cell.row_index + (cell.rowspan - 1) / 2
        for fragment in cell.fragments:
            if fragment.page_number != page_number:
                continue
            bounds_by_row.setdefault(logical_center, []).append(fragment.bbox)
    if len(bounds_by_row) < 2:
        return None
    first_row = bounds_by_row[min(bounds_by_row)]
    last_row = bounds_by_row[max(bounds_by_row)]
    if table.row_count <= 2:
        first_point = (min(bbox.x0 for bbox in first_row), min(bbox.y0 for bbox in first_row))
        last_point = (min(bbox.x0 for bbox in last_row), min(bbox.y0 for bbox in last_row))
    else:
        first_point = (
            sum((bbox.x0 + bbox.x1) / 2 for bbox in first_row) / len(first_row),
            sum((bbox.y0 + bbox.y1) / 2 for bbox in first_row) / len(first_row),
        )
        last_point = (
            sum((bbox.x0 + bbox.x1) / 2 for bbox in last_row) / len(last_row),
            sum((bbox.y0 + bbox.y1) / 2 for bbox in last_row) / len(last_row),
        )
    dx = last_point[0] - first_point[0]
    dy = last_point[1] - first_point[1]
    if abs(dx) > abs(dy):
        return ("x", 1 if dx > 0 else -1)
    if abs(dy) > 0:
        return ("y", 1 if dy > 0 else -1)
    return None


def _columns_align(previous: DocumentElement, current: DocumentElement) -> bool:
    previous_page = previous.fragments[-1].page_number
    current_page = current.fragments[0].page_number
    previous_row_axis = _element_row_axis(previous, previous_page)
    current_row_axis = _element_row_axis(current, current_page)
    if previous_row_axis is None or current_row_axis is None or previous_row_axis != current_row_axis:
        return False
    column_axis = "y" if previous_row_axis[0] == "x" else "x"
    previous_extent = _normalized_table_axis_bounds(previous.fragments[-1], column_axis)
    current_extent = _normalized_table_axis_bounds(current.fragments[0], column_axis)
    if (
        previous_extent is None
        or current_extent is None
        or any(
            abs(left - right) > _COLUMN_TOLERANCE
            for left, right in zip(previous_extent, current_extent, strict=True)
        )
    ):
        return False
    previous_boundaries = _normalized_column_boundaries(previous, previous_page, column_axis)
    current_boundaries = _normalized_column_boundaries(current, current_page, column_axis)
    return len(previous_boundaries) == len(current_boundaries) and all(
        abs(left - right) <= _COLUMN_TOLERANCE
        for left, right in zip(previous_boundaries, current_boundaries, strict=True)
    )


def _normalized_table_axis_bounds(fragment: PageFragment, axis: str) -> tuple[float, float] | None:
    page_extent = fragment.page_width if axis == "x" else fragment.page_height
    if page_extent is None:
        return None
    low, high = _fragment_axis_bounds(fragment, axis)
    return low / page_extent, high / page_extent


def _normalized_column_boundaries(element: DocumentElement, page_number: int, axis: str) -> list[float]:
    table = element.structure.table
    if table is None:
        raise ValueError("column boundaries require a table element")
    page_fragments = [fragment for fragment in element.fragments if fragment.page_number == page_number]
    if len(page_fragments) != 1:
        raise ValueError(f"table must have exactly one fragment for page {page_number}")
    table_low, table_high = _fragment_axis_bounds(page_fragments[0], axis)
    span = table_high - table_low
    boundaries = {
        coordinate
        for cell in table.cells
        for fragment in cell.fragments
        if fragment.page_number == page_number
        for coordinate in _fragment_axis_bounds(fragment, axis)
    }
    return sorted((coordinate - table_low) / span for coordinate in boundaries)


def _fragment_axis_bounds(fragment: PageFragment, axis: str) -> tuple[float, float]:
    if axis == "x":
        return fragment.bbox.x0, fragment.bbox.x1
    return fragment.bbox.y0, fragment.bbox.y1


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _with_order(element: DocumentElement, order: int) -> DocumentElement:
    payload = element.model_dump(mode="json")
    payload["order"] = order
    return DocumentElement.model_validate(payload)
