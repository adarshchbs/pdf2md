from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from typing import Literal, cast

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment

from app.pdf2md.schema import BoundingBox, DocumentElement, SchemaModel, TableCell, TableStructure
from app.pdf2md.tables import render_table, strict_table_classification

EVALUATOR_SEMANTICS_VERSION = "1.1.0"

_ALIGNMENT_THRESHOLD = 0.35
_MIN_TEXT_SIMILARITY = 0.5
_MIN_FIGURE_GEOMETRY_IOU = 0.2
_MAX_PAGE_BACKGROUND_AREA_RATIO = 0.9
_MIN_COMPONENT_LENGTH = 4
_MIN_RELATION_COVERAGE = 0.8
_PROVENANCE_ONLY_PROPERTY_KEYS = frozenset({"source_note", "inline_footnote_markers"})

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]


class ElementAlignment(SchemaModel):
    candidate_index: int
    reference_index: int
    score: float


class EvaluationMetrics(SchemaModel):
    element_precision: float
    element_recall: float
    element_f1: float
    element_type_accuracy: float
    reading_order_accuracy: float | None
    normalized_character_error_rate: float
    mean_geometry_iou: float
    fragmentation_rate: float
    merge_rate: float
    content_fragmentation_rate: float = 0.0
    content_merge_rate: float = 0.0
    candidate_reference_identity_rate: float
    candidate_reference_content_identity_rate: float
    semantic_role_accuracy: float | None = None
    semantic_role_conditional_accuracy: float | None = None
    heading_detection_precision: float | None = None
    heading_detection_recall: float | None = None
    caption_detection_precision: float | None = None
    caption_detection_recall: float | None = None
    figure_detection_precision: float | None = None
    figure_detection_recall: float | None = None
    footnote_detection_precision: float | None = None
    footnote_detection_recall: float | None = None
    table_detection_precision: float | None = None
    table_detection_recall: float | None = None
    table_topology_accuracy: float | None = None
    table_grid_topology_accuracy: float | None = None
    table_cell_role_accuracy: float | None = None
    table_cell_role_precision: float | None = None
    table_cell_role_recall: float | None = None
    table_cell_role_f1: float | None = None
    table_cell_content_accuracy: float | None = None
    table_cell_content_precision: float | None = None
    table_cell_content_recall: float | None = None
    table_cell_content_f1: float | None = None
    table_cell_token_precision: float | None = None
    table_cell_token_recall: float | None = None
    table_cell_token_f1: float | None = None
    table_cell_content_assignment_accuracy: float | None = None
    table_representation_accuracy: float | None = None
    canonical_render_accuracy: float
    cross_page_continuity_accuracy: float | None = None


class EvaluationReport(SchemaModel):
    document_id: str
    candidate_count: int
    reference_count: int
    alignments: list[ElementAlignment]
    unmatched_candidate_indices: list[int]
    unmatched_reference_indices: list[int]
    metrics: EvaluationMetrics


class ReferenceChurnReport(SchemaModel):
    document_id: str
    previous_count: int
    current_count: int
    added_count: int
    removed_count: int
    changed_count: int
    unchanged_count: int
    churn_rate: float
    candidate_identity_rate: float | None = None
    candidate_content_identity_rate: float | None = None


class CandidateChurnReport(SchemaModel):
    document_id: str
    previous_stage: Literal["candidate"]
    current_stage: Literal["candidate"]
    previous_count: int
    current_count: int
    added_normalized_content_count: int
    removed_normalized_content_count: int
    content_fragmentation_hits: int
    content_fragmentation_rate: float
    content_merge_hits: int
    content_merge_rate: float
    alignments: tuple[ElementAlignment, ...]
    aligned_count: int
    exact_content_match_count: int
    exact_content_identity_rate: float
    exact_type_match_count: int
    exact_type_identity_rate: float
    exact_order_match_count: int
    exact_order_identity_rate: float
    exact_content_type_order_match_count: int
    exact_content_type_order_identity_rate: float


def evaluate_document(candidate: list[DocumentElement], reference: list[DocumentElement]) -> EvaluationReport:
    document_id = _validate_documents(candidate, reference)
    alignments = align_elements(candidate, reference)
    candidate_matches = {alignment.candidate_index for alignment in alignments}
    reference_matches = {alignment.reference_index for alignment in alignments}
    unmatched_candidate = [index for index in range(len(candidate)) if index not in candidate_matches]
    unmatched_reference = [index for index in range(len(reference)) if index not in reference_matches]

    same_type = [
        alignment
        for alignment in alignments
        if candidate[alignment.candidate_index].element_type
        == reference[alignment.reference_index].element_type
    ]
    true_positive_count = len(same_type)
    precision = _safe_divide(true_positive_count, len(candidate))
    recall = _safe_divide(true_positive_count, len(reference))
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    candidate_character_texts = _semantic_text_payloads(candidate, table_projection="character")
    reference_character_texts = _semantic_text_payloads(reference, table_projection="character")
    candidate_text = "\n".join(
        candidate_character_texts[index]
        for index in sorted(range(len(candidate)), key=lambda index: candidate[index].order)
    )
    reference_text = "\n".join(
        reference_character_texts[index]
        for index in sorted(range(len(reference)), key=lambda index: reference[index].order)
    )
    character_errors = Levenshtein.distance(candidate_text, reference_text)
    character_denominator = max(len(candidate_text), len(reference_text))

    table_metrics = _table_metrics(candidate, reference)
    semantic_metrics = _semantic_metrics(candidate, reference, alignments)
    content_fragmentation_hits, reference_relation_eligible = _content_relation_stats(reference, candidate)
    content_merge_hits, _ = _content_relation_stats(candidate, reference)
    cross_page_reference = [index for index, element in enumerate(reference) if len(element.fragments) > 1]
    matched_by_reference = {alignment.reference_index: alignment.candidate_index for alignment in alignments}
    cross_page_correct = sum(
        _fragment_pages(candidate[matched_by_reference[index]]) == _fragment_pages(reference[index])
        for index in cross_page_reference
        if index in matched_by_reference
    )

    metrics = EvaluationMetrics(
        element_precision=precision,
        element_recall=recall,
        element_f1=f1,
        element_type_accuracy=_safe_divide(len(same_type), len(alignments)),
        reading_order_accuracy=_reading_order_accuracy(candidate, reference, alignments),
        normalized_character_error_rate=_safe_divide(character_errors, character_denominator),
        mean_geometry_iou=_mean([
            _element_geometry_iou(
                candidate[alignment.candidate_index],
                reference[alignment.reference_index],
            )
            for alignment in alignments
        ]),
        fragmentation_rate=_fragmentation_rate(candidate, reference),
        merge_rate=_merge_rate(candidate, reference),
        content_fragmentation_rate=_safe_divide(content_fragmentation_hits, reference_relation_eligible),
        content_merge_rate=_safe_divide(
            min(content_merge_hits, reference_relation_eligible), reference_relation_eligible
        ),
        candidate_reference_identity_rate=_candidate_reference_identity_rate(candidate, reference),
        candidate_reference_content_identity_rate=_candidate_reference_content_identity_rate(
            candidate, reference
        ),
        canonical_render_accuracy=_canonical_render_accuracy(candidate, reference, alignments),
        cross_page_continuity_accuracy=(
            _safe_divide(cross_page_correct, len(cross_page_reference)) if cross_page_reference else None
        ),
        **semantic_metrics,
        **table_metrics,
    )
    return EvaluationReport(
        document_id=document_id,
        candidate_count=len(candidate),
        reference_count=len(reference),
        alignments=alignments,
        unmatched_candidate_indices=unmatched_candidate,
        unmatched_reference_indices=unmatched_reference,
        metrics=metrics,
    )


def evaluate_reference_churn(
    previous: list[DocumentElement],
    current: list[DocumentElement],
    *,
    candidate: list[DocumentElement] | None = None,
) -> ReferenceChurnReport:
    document_id = _validate_reference_revisions(previous, current)
    previous_by_id = {element.element_id: element for element in previous}
    current_by_id = {element.element_id: element for element in current}
    common_ids = previous_by_id.keys() & current_by_id.keys()
    changed = sum(
        _comparison_payload(previous_by_id[element_id]) != _comparison_payload(current_by_id[element_id])
        for element_id in common_ids
    )
    added = len(current_by_id.keys() - previous_by_id.keys())
    removed = len(previous_by_id.keys() - current_by_id.keys())
    churn = changed + added + removed
    denominator = max(len(previous), len(current))
    if candidate is not None:
        _validate_documents(candidate, current)
    return ReferenceChurnReport(
        document_id=document_id,
        previous_count=len(previous),
        current_count=len(current),
        added_count=added,
        removed_count=removed,
        changed_count=changed,
        unchanged_count=len(common_ids) - changed,
        churn_rate=_safe_divide(churn, denominator),
        candidate_identity_rate=(
            _candidate_reference_identity_rate(candidate, current) if candidate is not None else None
        ),
        candidate_content_identity_rate=(
            _candidate_reference_content_identity_rate(candidate, current) if candidate is not None else None
        ),
    )


def evaluate_candidate_churn(
    previous: list[DocumentElement], current: list[DocumentElement]
) -> CandidateChurnReport:
    document_id = _validate_candidate_revisions(previous, current)
    previous_content = Counter(_normalize_text(element.content) for element in previous)
    current_content = Counter(_normalize_text(element.content) for element in current)
    alignments = align_elements(previous, current)
    denominator = max(len(previous), len(current))
    fragmentation_hits, fragmentation_eligible = _content_relation_stats(previous, current)
    merge_hits, _ = _content_relation_stats(current, previous)

    exact_content_matches = sum((previous_content & current_content).values())
    exact_type_matches = sum(
        previous[alignment.candidate_index].element_type == current[alignment.reference_index].element_type
        for alignment in alignments
    )
    exact_order_matches = sum(
        previous[alignment.candidate_index].order == current[alignment.reference_index].order
        for alignment in alignments
    )
    previous_identity = Counter(
        (_normalize_text(element.content), element.element_type, element.order) for element in previous
    )
    current_identity = Counter(
        (_normalize_text(element.content), element.element_type, element.order) for element in current
    )
    exact_identity_matches = sum((previous_identity & current_identity).values())

    return CandidateChurnReport(
        document_id=document_id,
        previous_stage="candidate",
        current_stage="candidate",
        previous_count=len(previous),
        current_count=len(current),
        added_normalized_content_count=sum((current_content - previous_content).values()),
        removed_normalized_content_count=sum((previous_content - current_content).values()),
        content_fragmentation_hits=fragmentation_hits,
        content_fragmentation_rate=_safe_divide(fragmentation_hits, fragmentation_eligible),
        content_merge_hits=merge_hits,
        content_merge_rate=_safe_divide(min(merge_hits, fragmentation_eligible), fragmentation_eligible),
        alignments=tuple(alignments),
        aligned_count=len(alignments),
        exact_content_match_count=exact_content_matches,
        exact_content_identity_rate=_safe_divide(exact_content_matches, denominator),
        exact_type_match_count=exact_type_matches,
        exact_type_identity_rate=_safe_divide(exact_type_matches, denominator),
        exact_order_match_count=exact_order_matches,
        exact_order_identity_rate=_safe_divide(exact_order_matches, denominator),
        exact_content_type_order_match_count=exact_identity_matches,
        exact_content_type_order_identity_rate=_safe_divide(exact_identity_matches, denominator),
    )


def align_elements(
    candidate: list[DocumentElement], reference: list[DocumentElement]
) -> list[ElementAlignment]:
    if not candidate or not reference:
        return []
    candidate_by_id = {element.element_id: element for element in candidate}
    reference_by_id = {element.element_id: element for element in reference}
    scores = np.array([
        [
            _element_similarity(
                candidate_element,
                reference_element,
                candidate_by_id=candidate_by_id,
                reference_by_id=reference_by_id,
            )
            for reference_element in reference
        ]
        for candidate_element in candidate
    ])
    assignments = _threshold_assignment(scores, scores >= _ALIGNMENT_THRESHOLD)
    return sorted(
        [
            ElementAlignment(
                candidate_index=candidate_index,
                reference_index=reference_index,
                score=float(scores[candidate_index, reference_index]),
            )
            for candidate_index, reference_index in assignments
        ],
        key=lambda alignment: alignment.reference_index,
    )


def _threshold_assignment(scores: np.ndarray, eligible: np.ndarray) -> list[tuple[int, int]]:
    """Find a maximum-weight matching without letting rejected edges affect it."""
    if scores.shape != eligible.shape or scores.ndim != 2:
        raise ValueError("assignment scores and eligibility must be same-shaped matrices")
    weighted_scores = np.where(eligible, scores, 0.0)
    candidate_indices, reference_indices = linear_sum_assignment(weighted_scores, maximize=True)
    return [
        (int(candidate_index), int(reference_index))
        for candidate_index, reference_index in zip(candidate_indices, reference_indices, strict=True)
        if eligible[candidate_index, reference_index]
    ]


def _element_similarity(
    candidate: DocumentElement,
    reference: DocumentElement,
    *,
    candidate_by_id: dict[str, DocumentElement],
    reference_by_id: dict[str, DocumentElement],
) -> float:
    if candidate.element_type == "figure" or reference.element_type == "figure":
        return _figure_similarity(
            candidate,
            reference,
            candidate_by_id=candidate_by_id,
            reference_by_id=reference_by_id,
        )

    candidate_text = _character_error_text(candidate)
    reference_text = _character_error_text(reference)
    if not (_fragment_pages(candidate) & _fragment_pages(reference)):
        return 0.0
    geometry_score = _element_geometry_iou(candidate, reference)
    if geometry_score == 0 and (
        not _is_informative_text(candidate_text) or not _is_informative_text(reference_text)
    ):
        return 0.0
    text_score = ratio(candidate_text, reference_text) / 100
    if text_score < _MIN_TEXT_SIMILARITY:
        return 0.0
    if (candidate.element_type == "table") != (reference.element_type == "table"):
        return 0.0
    type_score = float(candidate.element_type == reference.element_type)
    return 0.45 * text_score + 0.35 * geometry_score + 0.2 * type_score


def _figure_similarity(
    candidate: DocumentElement,
    reference: DocumentElement,
    *,
    candidate_by_id: dict[str, DocumentElement],
    reference_by_id: dict[str, DocumentElement],
) -> float:
    if candidate.element_type != "figure" or reference.element_type != "figure":
        return 0.0
    candidate_figure = candidate.structure.figure
    reference_figure = reference.structure.figure
    if (
        candidate_figure is None
        or reference_figure is None
        or not candidate_figure.asset_path.strip()
        or not reference_figure.asset_path.strip()
    ):
        return 0.0
    if not candidate.fragments or not reference.fragments:
        return 0.0
    if not (_fragment_pages(candidate) & _fragment_pages(reference)):
        return 0.0
    if _is_page_background(candidate) or _is_page_background(reference):
        return 0.0

    geometry_score = _element_geometry_iou(candidate, reference)
    if geometry_score < _MIN_FIGURE_GEOMETRY_IOU:
        return 0.0

    candidate_caption = _linked_caption_text(candidate, candidate_by_id)
    reference_caption = _linked_caption_text(reference, reference_by_id)
    if candidate_caption and reference_caption:
        caption_score = ratio(candidate_caption, reference_caption) / 100
        if caption_score < _MIN_TEXT_SIMILARITY:
            return 0.0
        return 0.65 * geometry_score + 0.2 + 0.15 * caption_score
    return 0.8 * geometry_score + 0.2


def _linked_caption_text(figure: DocumentElement, elements_by_id: dict[str, DocumentElement]) -> str:
    structure = figure.structure.figure
    if structure is None:
        return ""
    linked_ids = [structure.caption_element_id, *figure.structure.linked_element_ids]
    captions = [
        _normalize_text(linked.content)
        for linked_id in linked_ids
        if linked_id is not None
        for linked in [elements_by_id.get(linked_id)]
        if linked is not None and linked.element_type == "caption"
    ]
    return " ".join(dict.fromkeys(caption for caption in captions if _is_informative_text(caption)))


def _is_page_background(element: DocumentElement) -> bool:
    for fragment in element.fragments:
        if fragment.page_width is None or fragment.page_height is None:
            continue
        bbox = fragment.bbox
        area_ratio = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0) / (fragment.page_width * fragment.page_height)
        if area_ratio >= _MAX_PAGE_BACKGROUND_AREA_RATIO:
            return True
    return False


def _align_table_structures(
    candidate: list[DocumentElement], reference: list[DocumentElement]
) -> list[tuple[TableStructure, TableStructure]]:
    if not candidate or not reference:
        return []
    geometry_scores = np.array([
        [_element_geometry_iou(candidate_element, reference_element) for reference_element in reference]
        for candidate_element in candidate
    ])
    text_scores = np.array([
        [_table_content_similarity(candidate_element, reference_element) for reference_element in reference]
        for candidate_element in candidate
    ])
    scores = 0.7 * geometry_scores + 0.3 * text_scores
    eligible = np.array([
        [
            bool(_fragment_pages(candidate_element) & _fragment_pages(reference_element))
            and (
                geometry_scores[candidate_index, reference_index] >= 0.05
                or (
                    text_scores[candidate_index, reference_index] >= _MIN_TEXT_SIMILARITY
                    and _is_informative_text(_table_text(candidate_element))
                    and _is_informative_text(_table_text(reference_element))
                )
            )
            for reference_index, reference_element in enumerate(reference)
        ]
        for candidate_index, candidate_element in enumerate(candidate)
    ])
    pairs: list[tuple[TableStructure, TableStructure]] = []
    for candidate_index, reference_index in _threshold_assignment(scores, eligible):
        candidate_table = candidate[candidate_index].structure.table
        reference_table = reference[reference_index].structure.table
        if candidate_table is None or reference_table is None:
            raise ValueError("table elements must carry table structure")
        pairs.append((candidate_table, reference_table))
    return pairs


def _table_text(element: DocumentElement) -> str:
    table = element.structure.table
    if table is None:
        return ""
    return _normalize_text(" ".join(cell.text for cell in table.cells if cell.text.strip()))


def _table_content_similarity(first: DocumentElement, second: DocumentElement) -> float:
    first_tokens = _table_token_counter(cast(TableStructure, first.structure.table))
    second_tokens = _table_token_counter(cast(TableStructure, second.structure.table))
    overlap = sum((first_tokens & second_tokens).values())
    return _safe_divide(2 * overlap, sum(first_tokens.values()) + sum(second_tokens.values()))


def _semantic_metrics(
    candidate: list[DocumentElement],
    reference: list[DocumentElement],
    alignments: list[ElementAlignment],
) -> dict[str, float | None]:
    semantic_pairs = [
        (candidate_role.role, reference_role.role)
        for alignment in alignments
        for candidate_role in [candidate[alignment.candidate_index].structure.paragraph]
        for reference_role in [reference[alignment.reference_index].structure.paragraph]
        if candidate_role is not None and reference_role is not None
    ]
    reference_role_count = sum(element.structure.paragraph is not None for element in reference)
    correct_role_count = sum(
        candidate_role == reference_role for candidate_role, reference_role in semantic_pairs
    )
    metrics: dict[str, float | None] = {
        "semantic_role_accuracy": (
            _safe_divide(correct_role_count, reference_role_count) if reference_role_count else None
        ),
        "semantic_role_conditional_accuracy": (
            _safe_divide(correct_role_count, len(semantic_pairs)) if semantic_pairs else None
        ),
    }
    for element_type in ("heading", "caption", "figure", "footnote"):
        candidate_count = sum(element.element_type == element_type for element in candidate)
        reference_count = sum(element.element_type == element_type for element in reference)
        true_positive_count = sum(
            candidate[alignment.candidate_index].element_type == element_type
            and reference[alignment.reference_index].element_type == element_type
            for alignment in alignments
        )
        metrics[f"{element_type}_detection_precision"] = (
            _safe_divide(true_positive_count, candidate_count) if candidate_count else None
        )
        metrics[f"{element_type}_detection_recall"] = (
            _safe_divide(true_positive_count, reference_count) if reference_count else None
        )
    return metrics


def _table_metrics(
    candidate: list[DocumentElement],
    reference: list[DocumentElement],
) -> dict[str, float | None]:
    candidate_elements = [
        element
        for element in candidate
        if element.element_type == "table" and element.structure.table is not None
    ]
    reference_elements = [
        element
        for element in reference
        if element.element_type == "table" and element.structure.table is not None
    ]
    candidate_table_count = len(candidate_elements)
    candidate_tables = [cast(TableStructure, element.structure.table) for element in candidate_elements]
    reference_tables = [cast(TableStructure, element.structure.table) for element in reference_elements]
    reference_table_count = len(reference_tables)
    typed_pairs = _align_table_structures(candidate_elements, reference_elements)
    topology_correct = sum(
        _table_topology(candidate_table) == _table_topology(reference_table)
        for candidate_table, reference_table in typed_pairs
    )
    grid_topology_correct = sum(
        _table_grid_topology(candidate_table) == _table_grid_topology(reference_table)
        for candidate_table, reference_table in typed_pairs
    )
    candidate_cells = sum(len(candidate_table.cells) for candidate_table in candidate_tables)
    reference_cells = sum(len(reference_table.cells) for reference_table in reference_tables)
    cell_role_correct = sum(
        _matching_cell_roles(candidate_table, reference_table)
        for candidate_table, reference_table in typed_pairs
    )
    cell_content_correct = sum(
        _matching_cell_content(candidate_table, reference_table)
        for candidate_table, reference_table in typed_pairs
    )
    role_precision, role_recall, role_f1 = _precision_recall_f1(
        cell_role_correct, candidate_cells, reference_cells
    )
    content_precision, content_recall, content_f1 = _precision_recall_f1(
        cell_content_correct, candidate_cells, reference_cells
    )
    candidate_token_count = sum(
        sum(_table_token_counter(candidate_table).values()) for candidate_table in candidate_tables
    )
    reference_token_count = sum(
        sum(_table_token_counter(reference_table).values()) for reference_table in reference_tables
    )
    matching_token_count = sum(
        _matching_table_tokens(candidate_table, reference_table)
        for candidate_table, reference_table in typed_pairs
    )
    token_precision, token_recall, token_f1 = _precision_recall_f1(
        matching_token_count, candidate_token_count, reference_token_count
    )
    assignment_correct, assignment_comparable = _cell_content_assignment_stats(
        candidate_tables, reference_tables, typed_pairs
    )
    representation_correct = sum(
        candidate_table.representation == strict_table_classification(candidate_table)[0]
        and strict_table_classification(candidate_table)[0] == strict_table_classification(reference_table)[0]
        for candidate_table, reference_table in typed_pairs
    )
    return {
        "table_detection_precision": (
            _safe_divide(len(typed_pairs), candidate_table_count) if candidate_table_count else None
        ),
        "table_detection_recall": (
            _safe_divide(len(typed_pairs), reference_table_count) if reference_table_count else None
        ),
        "table_topology_accuracy": (
            _safe_divide(topology_correct, reference_table_count) if reference_table_count else None
        ),
        "table_grid_topology_accuracy": (
            _safe_divide(grid_topology_correct, reference_table_count) if reference_table_count else None
        ),
        "table_cell_role_accuracy": (
            _safe_divide(cell_role_correct, reference_cells) if reference_table_count else None
        ),
        "table_cell_role_precision": role_precision,
        "table_cell_role_recall": role_recall,
        "table_cell_role_f1": role_f1,
        "table_cell_content_accuracy": (
            _safe_divide(cell_content_correct, reference_cells) if reference_table_count else None
        ),
        "table_cell_content_precision": content_precision,
        "table_cell_content_recall": content_recall,
        "table_cell_content_f1": content_f1,
        "table_cell_token_precision": token_precision,
        "table_cell_token_recall": token_recall,
        "table_cell_token_f1": token_f1,
        "table_cell_content_assignment_accuracy": (
            _safe_divide(assignment_correct, assignment_comparable) if assignment_comparable else None
        ),
        "table_representation_accuracy": (
            _safe_divide(representation_correct, reference_table_count) if reference_table_count else None
        ),
    }


def _table_topology(table: TableStructure) -> tuple[object, ...]:
    return (
        table.row_count,
        table.column_count,
        table.header_row_count,
        tuple(
            sorted(
                (
                    cell.row_index,
                    cell.column_index,
                    cell.rowspan,
                    cell.colspan,
                    cell.role,
                )
                for cell in table.cells
            )
        ),
    )


def _table_grid_topology(table: TableStructure) -> tuple[object, ...]:
    return (
        table.row_count,
        table.column_count,
        table.header_row_count,
        tuple(
            sorted((cell.row_index, cell.column_index, cell.rowspan, cell.colspan) for cell in table.cells)
        ),
    )


def _cells_by_position(table: TableStructure) -> dict[tuple[int, int, int, int], TableCell]:
    return {(cell.row_index, cell.column_index, cell.rowspan, cell.colspan): cell for cell in table.cells}


def _matching_cell_roles(candidate: TableStructure, reference: TableStructure) -> int:
    candidate_cells = _cells_by_position(candidate)
    return sum(
        key in candidate_cells and candidate_cells[key].role == cell.role
        for cell in reference.cells
        for key in [(cell.row_index, cell.column_index, cell.rowspan, cell.colspan)]
    )


def _matching_cell_content(candidate: TableStructure, reference: TableStructure) -> int:
    candidate_content = Counter(_normalize_text(cell.text) for cell in candidate.cells)
    reference_content = Counter(_normalize_text(cell.text) for cell in reference.cells)
    return sum((candidate_content & reference_content).values())


def _table_token_counter(table: TableStructure) -> Counter[str]:
    return Counter(
        token for cell in table.cells for token in re.findall(r"\w+|[^\w\s]", _normalize_text(cell.text))
    )


def _matching_table_tokens(candidate: TableStructure, reference: TableStructure) -> int:
    return sum((_table_token_counter(candidate) & _table_token_counter(reference)).values())


def _cell_content_assignment_stats(
    candidate_tables: list[TableStructure],
    reference_tables: list[TableStructure],
    pairs: list[tuple[TableStructure, TableStructure]],
) -> tuple[int, int]:
    """Score exact content at positions without dropping missing or extra cells."""
    correct = 0
    comparable = 0
    for candidate, reference in pairs:
        candidate_cells = _cells_by_position(candidate)
        reference_cells = _cells_by_position(reference)
        for position in candidate_cells.keys() | reference_cells.keys():
            candidate_cell = candidate_cells.get(position)
            reference_cell = reference_cells.get(position)
            candidate_text = _normalize_text(candidate_cell.text) if candidate_cell is not None else ""
            reference_text = _normalize_text(reference_cell.text) if reference_cell is not None else ""
            if not candidate_text and not reference_text:
                continue
            comparable += 1
            correct += candidate_text == reference_text

    paired_candidate_ids = {id(candidate) for candidate, _ in pairs}
    paired_reference_ids = {id(reference) for _, reference in pairs}
    comparable += sum(
        bool(_normalize_text(cell.text))
        for table in candidate_tables
        if id(table) not in paired_candidate_ids
        for cell in table.cells
    )
    comparable += sum(
        bool(_normalize_text(cell.text))
        for table in reference_tables
        if id(table) not in paired_reference_ids
        for cell in table.cells
    )
    return correct, comparable


def _precision_recall_f1(
    true_positive_count: int, candidate_count: int, reference_count: int
) -> tuple[float | None, float | None, float | None]:
    precision = _safe_divide(true_positive_count, candidate_count) if candidate_count else None
    recall = _safe_divide(true_positive_count, reference_count) if reference_count else None
    f1 = (
        _safe_divide(2 * true_positive_count, candidate_count + reference_count)
        if candidate_count or reference_count
        else None
    )
    return precision, recall, f1


def _reading_order_accuracy(
    candidate: list[DocumentElement],
    reference: list[DocumentElement],
    alignments: list[ElementAlignment],
) -> float | None:
    pairs = list(combinations(alignments, 2))
    if not pairs:
        return None
    correct = sum(
        (candidate[left.candidate_index].order < candidate[right.candidate_index].order)
        == (reference[left.reference_index].order < reference[right.reference_index].order)
        for left, right in pairs
    )
    return correct / len(pairs)


def _fragmentation_rate(candidate: list[DocumentElement], reference: list[DocumentElement]) -> float:
    fragmented = sum(_component_count(element, candidate) > 1 for element in reference)
    return _safe_divide(fragmented, len(reference))


def _merge_rate(candidate: list[DocumentElement], reference: list[DocumentElement]) -> float:
    merged = sum(_component_count(element, reference) > 1 for element in candidate)
    return _safe_divide(min(merged, len(reference)), len(reference))


def _component_count(target: DocumentElement, possible_components: list[DocumentElement]) -> int:
    target_text = _normalize_text(target.content)
    if len(target_text) < _MIN_COMPONENT_LENGTH:
        return 0
    return sum(
        component.element_type == target.element_type
        and _element_geometry_iou(component, target) > 0
        and _is_text_component(_normalize_text(component.content), target_text)
        for component in possible_components
    )


def _is_text_component(component: str, whole: str) -> bool:
    if len(component) < max(_MIN_COMPONENT_LENGTH, round(len(whole) * 0.15)) or component == whole:
        return False
    return component in whole or ratio(component, whole) / 100 >= _MIN_TEXT_SIMILARITY


def _content_relation_stats(
    targets: list[DocumentElement], possible_components: list[DocumentElement]
) -> tuple[int, int]:
    eligible_targets = [target for target in targets if len(_relation_text(target)) >= _MIN_COMPONENT_LENGTH]
    related = sum(
        _has_one_to_many_content_relation(target, possible_components) for target in eligible_targets
    )
    return related, len(eligible_targets)


def _has_one_to_many_content_relation(
    target: DocumentElement, possible_components: list[DocumentElement]
) -> bool:
    target_text = _relation_text(target)
    if not target_text:
        return False

    # A component must stay on the same page set and on the same side of the
    # table boundary. Geometry is deliberately not required: moved chart labels
    # are exactly the segmentation failure this metric is intended to expose.
    comparable_components = [
        component
        for component in possible_components
        if _fragment_pages(component) == _fragment_pages(target)
        and (component.element_type == "table") == (target.element_type == "table")
    ]
    component_texts = {
        component_text
        for component in comparable_components
        for component_text in [_relation_text(component)]
        if component_text and component_text != target_text and component_text in target_text
    }
    if len(component_texts) < 2:
        return False

    # An exact counterpart means the target itself is not fragmented/merged;
    # additional substrings are duplicate extras and must not manufacture a relation.
    if any(_relation_text(component) == target_text for component in comparable_components):
        return False

    # Count each distinct component text at most once and assign it to one fully
    # non-overlapping target span. Explore alternate occurrences because a greedy
    # longest-first choice can hide a valid set of disjoint components.
    required_coverage = _MIN_RELATION_COVERAGE * len(target_text)
    states = {0: 0}
    for component_text in sorted(component_texts, key=len, reverse=True):
        span_masks = {
            ((1 << len(component_text)) - 1) << start
            for start in _substring_starts(target_text, component_text)
        }
        next_states = dict(states)
        for covered, contributor_count in states.items():
            for span_mask in span_masks:
                if covered & span_mask:
                    continue
                combined = covered | span_mask
                combined_count = contributor_count + 1
                if combined_count > 1 and combined.bit_count() >= required_coverage:
                    return True
                next_states[combined] = max(next_states.get(combined, 0), combined_count)
        states = next_states
    return False


def _relation_text(element: DocumentElement) -> str:
    normalized = _normalize_text(element.content).casefold()
    if not _is_informative_text(normalized):
        return ""
    return re.sub(r"\s+", "", normalized)


def _substring_starts(whole: str, component: str) -> list[int]:
    starts: list[int] = []
    start = whole.find(component)
    while start >= 0:
        starts.append(start)
        start = whole.find(component, start + 1)
    return starts


def _candidate_reference_identity_rate(
    candidate: list[DocumentElement], reference: list[DocumentElement]
) -> float:
    candidate_payloads = Counter(
        _comparison_payload(element, include_provenance=False) for element in candidate
    )
    reference_payloads = Counter(
        _comparison_payload(element, include_provenance=False) for element in reference
    )
    return _safe_divide(sum((candidate_payloads & reference_payloads).values()), len(reference))


def _candidate_reference_content_identity_rate(
    candidate: list[DocumentElement], reference: list[DocumentElement]
) -> float:
    candidate_text = Counter(_semantic_text_payloads(candidate, table_projection="content"))
    reference_text = Counter(_semantic_text_payloads(reference, table_projection="content"))
    return _safe_divide(sum((candidate_text & reference_text).values()), len(reference))


def _canonical_render_accuracy(
    candidate: list[DocumentElement],
    reference: list[DocumentElement],
    alignments: list[ElementAlignment],
) -> float:
    candidate_render = _semantic_text_payloads(candidate, table_projection="render")
    reference_render = _semantic_text_payloads(reference, table_projection="render")
    correct = sum(
        candidate_render[alignment.candidate_index] == reference_render[alignment.reference_index]
        for alignment in alignments
    )
    return _safe_divide(correct, len(reference))


def _semantic_text_payloads(
    elements: list[DocumentElement], *, table_projection: Literal["content", "character", "render"]
) -> list[str]:
    """Compare footnote meaning independently of source or Markdown marker syntax.

    Footnote labels and reciprocal reference relationships remain part of the
    payload. Only the representational differences between a native marker
    (``text3``) and canonical Markdown (``text[^3]``), and between a labeled
    native note and its schema-separated body, are erased.
    """
    known_labels = {
        footnote.label
        for element in elements
        if (footnote := element.structure.footnote) is not None and footnote.label is not None
    }
    base_texts = [
        _semantic_element_text(
            element,
            known_labels=known_labels,
            table_projection=table_projection,
        )
        for element in elements
    ]
    indices_by_id: dict[str, list[int]] = {}
    for index, element in enumerate(elements):
        indices_by_id.setdefault(element.element_id, []).append(index)

    inbound_labels: list[list[str | None]] = [[] for _ in elements]
    outbound_targets: list[list[str]] = [[] for _ in elements]
    for note_index, element in enumerate(elements):
        footnote = element.structure.footnote
        if footnote is None:
            continue
        for reference_id in footnote.reference_element_ids:
            target_indices = indices_by_id.get(reference_id, [])
            if len(target_indices) != 1:
                outbound_targets[note_index].append(f"unresolved:{reference_id}")
                continue
            target_index = target_indices[0]
            inbound_labels[target_index].append(footnote.label)
            fingerprint = hashlib.sha256(base_texts[target_index].encode()).hexdigest()
            outbound_targets[note_index].append(fingerprint)

    payloads: list[str] = []
    for index, element in enumerate(elements):
        footnote = element.structure.footnote
        relation: dict[str, JsonValue] = {}
        if inbound_labels[index]:
            relation["footnote_references"] = cast(list[JsonValue], inbound_labels[index])
        if footnote is not None:
            relation["footnote"] = {
                "association_confident": footnote.association_confident,
                "label": footnote.label,
                "reference_targets": cast(list[JsonValue], outbound_targets[index]),
            }
        payload = base_texts[index]
        if relation:
            payload += "\n" + json.dumps(relation, sort_keys=True, ensure_ascii=False)
        payloads.append(payload)
    return payloads


def _semantic_element_text(
    element: DocumentElement,
    *,
    known_labels: set[str],
    table_projection: Literal["content", "character", "render"],
) -> str:
    table = element.structure.table
    if element.element_type == "table" and table is not None:
        if table_projection == "content":
            return _normalize_text(element.content)
        if table_projection == "character":
            return _character_error_text(element)
        return _normalize_text(_strict_render(element))

    content = element.content
    content = re.sub(
        r"\[\^(?P<label>[^\]\r\n]+)\]",
        lambda match: match.group("label") if match.group("label") in known_labels else match.group(0),
        content,
    )
    footnote = element.structure.footnote
    if footnote is None or footnote.label is None:
        return _normalize_text(content)

    label = footnote.label
    body = re.sub(
        rf"^(?:\({re.escape(label)}\)|{re.escape(label)})[.)]?(?:\s+|(?=[A-Za-z]))",
        "",
        content,
        count=1,
    ).strip()
    return _normalize_text(f"{label} {body}")


def _comparison_payload(element: DocumentElement, *, include_provenance: bool = True) -> str:
    payload = cast(
        JsonValue,
        element.model_dump(mode="json", exclude={"element_id", "order", "annotation"}),
    )
    if not include_provenance:
        payload = _remove_provenance(payload)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _remove_provenance(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, child in value.items():
            if key == "source_item_ids":
                normalized[key] = []
            elif key == "properties" and isinstance(child, list):
                normalized[key] = [
                    _remove_provenance(item)
                    for item in child
                    if not (isinstance(item, dict) and item.get("key") in _PROVENANCE_ONLY_PROPERTY_KEYS)
                ]
            else:
                normalized[key] = _remove_provenance(child)
        return normalized
    if isinstance(value, list):
        return [_remove_provenance(child) for child in value]
    if isinstance(value, float):
        return round(value, 3)
    return value


def _element_geometry_iou(first: DocumentElement, second: DocumentElement) -> float:
    first_by_page: dict[int, list[BoundingBox]] = {}
    second_by_page: dict[int, list[BoundingBox]] = {}
    for fragment in first.fragments:
        first_by_page.setdefault(fragment.page_number, []).append(fragment.bbox)
    for fragment in second.fragments:
        second_by_page.setdefault(fragment.page_number, []).append(fragment.bbox)

    intersection_area = 0.0
    union_area = 0.0
    for page in first_by_page.keys() | second_by_page.keys():
        first_boxes = first_by_page.get(page, [])
        second_boxes = second_by_page.get(page, [])
        first_area = _rectangle_union_area(first_boxes)
        second_area = _rectangle_union_area(second_boxes)
        intersections = [
            intersection
            for first_bbox in first_boxes
            for second_bbox in second_boxes
            if (intersection := _bbox_intersection(first_bbox, second_bbox)) is not None
        ]
        page_intersection = _rectangle_union_area(intersections)
        intersection_area += page_intersection
        union_area += first_area + second_area - page_intersection
    return _safe_divide(intersection_area, union_area)


def _bbox_intersection(first: BoundingBox, second: BoundingBox) -> BoundingBox | None:
    x0 = max(first.x0, second.x0)
    y0 = max(first.y0, second.y0)
    x1 = min(first.x1, second.x1)
    y1 = min(first.y1, second.y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _rectangle_union_area(boxes: list[BoundingBox]) -> float:
    x_edges = sorted({edge for box in boxes for edge in (box.x0, box.x1)})
    area = 0.0
    for x0, x1 in zip(x_edges, x_edges[1:], strict=False):
        intervals = sorted(
            (box.y0, box.y1) for box in boxes if box.x0 < x1 and box.x1 > x0 and box.y1 > box.y0
        )
        if not intervals:
            continue
        covered_y = 0.0
        current_y0, current_y1 = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 > current_y1:
                covered_y += current_y1 - current_y0
                current_y0, current_y1 = y0, y1
            else:
                current_y1 = max(current_y1, y1)
        area += (x1 - x0) * (covered_y + current_y1 - current_y0)
    return area


def _fragment_pages(element: DocumentElement) -> set[int]:
    return {fragment.page_number for fragment in element.fragments}


def _strict_render(element: DocumentElement) -> str:
    table = element.structure.table
    if element.element_type != "table" or table is None:
        return element.content
    representation, reasons = strict_table_classification(table)
    strict_table = table.model_copy(
        update={"representation": representation, "classification_reasons": reasons}
    )
    return render_table(strict_table)


def _character_error_text(element: DocumentElement) -> str:
    """Project an element to ordered source characters, excluding table syntax and topology."""
    table = element.structure.table
    if element.element_type != "table" or table is None:
        return _normalize_text(element.content)
    ordered_cells = sorted(table.cells, key=lambda cell: (cell.row_index, cell.column_index))
    return "".join(_encode_character_error_cell(cell.text) for cell in ordered_cells)


def _encode_character_error_cell(text: str) -> str:
    """Encode one normalized cell with an unambiguous, topology-neutral terminator."""
    normalized = _normalize_text(text)
    escaped = normalized.replace("\x01", "\x01\x01").replace("\x00", "\x01\x00")
    return f"{escaped}\x00"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _is_informative_text(value: str) -> bool:
    return any(character.isalnum() for character in value)


def _validate_candidate_revisions(previous: list[DocumentElement], current: list[DocumentElement]) -> str:
    elements = [*previous, *current]
    if not elements:
        raise ValueError("candidate churn requires at least one element")
    invalid_previous_stages = sorted({
        element.annotation.stage for element in previous if element.annotation.stage != "candidate"
    })
    if invalid_previous_stages:
        raise ValueError(
            f"previous candidate elements must have candidate stage, got {invalid_previous_stages}"
        )
    invalid_current_stages = sorted({
        element.annotation.stage for element in current if element.annotation.stage != "candidate"
    })
    if invalid_current_stages:
        raise ValueError(
            f"current candidate elements must have candidate stage, got {invalid_current_stages}"
        )
    document_ids = {element.document_id for element in elements}
    if len(document_ids) != 1:
        raise ValueError("candidate revisions must belong to one document")
    return next(iter(document_ids))


def _validate_reference_revisions(previous: list[DocumentElement], current: list[DocumentElement]) -> str:
    elements = [*previous, *current]
    document_ids = {element.document_id for element in elements}
    if not elements:
        raise ValueError("reference churn requires at least one element")
    if len(document_ids) != 1:
        raise ValueError("reference revisions must belong to one document")
    invalid_stages = sorted({
        element.annotation.stage
        for element in elements
        if element.annotation.stage not in {"silver", "golden"}
    })
    if invalid_stages:
        raise ValueError(f"reference revisions must have silver or golden stage, got {invalid_stages}")
    return next(iter(document_ids))


def _validate_documents(candidate: list[DocumentElement], reference: list[DocumentElement]) -> str:
    invalid_candidate_stages = sorted({
        element.annotation.stage for element in candidate if element.annotation.stage != "candidate"
    })
    if invalid_candidate_stages:
        raise ValueError(f"candidate elements must have candidate stage, got {invalid_candidate_stages}")
    invalid_reference_stages = sorted({
        element.annotation.stage
        for element in reference
        if element.annotation.stage not in {"silver", "golden"}
    })
    if invalid_reference_stages:
        raise ValueError(
            f"reference elements must have silver or golden stage, got {invalid_reference_stages}"
        )
    document_ids = {element.document_id for element in [*candidate, *reference]}
    if not document_ids:
        raise ValueError("evaluation requires at least one element")
    if len(document_ids) != 1:
        raise ValueError("candidate and reference must belong to one document")
    return next(iter(document_ids))


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
