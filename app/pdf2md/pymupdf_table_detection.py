from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import pymupdf

from app.pdf2md.source_catalog import content_addressed_source_item_id, source_item_identity_sha256
from app.pdf2md.table_provenance import (
    CoordinateFrame,
    FinderProvenance,
    GeometricBand,
    GeometricBandKind,
    NativeRule,
    NativeRuleId,
    NativeToken,
    NativeTokenId,
    TableProvenance,
)

BBox = tuple[float, float, float, float]

_MIN_RULE_WIDTH_RATIO = 0.20
_HORIZONTAL_TOLERANCE_RATIO = 0.002
_MAX_RULE_GAP_RATIO = 0.10
_MIN_BAND_HEIGHT_RATIO = 0.015
_BAND_MARGIN_RATIO = 0.006
_CAPTION_DISTANCE_RATIO = 0.10
_CAPTION_HORIZONTAL_SLOP_RATIO = 0.03
_DEDUPLICATION_TOLERANCE_RATIO = 0.002
_MIN_DUPLICATE_IOU = 0.90
_MIN_RECOVERY_CONSENSUS_IOU = 0.85
_MAX_RECOVERY_ROW_RATIO = 3.0
_MIN_POPULATED_RATIO = 0.25
_MIN_POPULATED_ROW_RATIO = 0.50
_MIN_RULES_PER_RECOVERY_BAND = 3
_MAX_DEFAULT_RECOVERY_OVERLAP = 0.10
_MIN_COMPLEX_VECTOR_ITEMS = 8
_COMPLEX_VECTOR_DENSITY = 48.0
_MIN_BORDERLESS_NUMERIC_RATIO = 0.2
_MIN_BORDERLESS_ROWS = 3
_MIN_BORDERLESS_COLUMNS = 2
_MIN_WEAK_DEFAULT_COVERAGE = 0.95
_MIN_RECONCILIATION_GAIN = 3
_MAX_SINGLE_ROW_SATELLITE_HEIGHT_RATIO = 0.02
_MAX_SINGLE_ROW_SATELLITE_GAP_RATIO = 0.03
_MIN_SATELLITE_RULE_COVERAGE = 0.80
_CAPTION_PATTERN = re.compile(
    r"^\s*(table|figure)(?:\s+(?:[A-Z]?\d+|[IVXLCDM]+)\b|\s*[:.\N{EM DASH}\N{EN DASH}-])",
    re.IGNORECASE,
)


class DetectedHeader(Protocol):
    @property
    def external(self) -> bool: ...

    @property
    def names(self) -> Sequence[str | None]: ...


class DetectedRow(Protocol):
    @property
    def cells(self) -> Sequence[Sequence[float] | None]: ...


class DetectedTable(Protocol):
    @property
    def bbox(self) -> Sequence[float]: ...

    def extract(self) -> list[list[str | None]]: ...


class SnapshotCapableTable(DetectedTable, Protocol):
    @property
    def cells(self) -> Sequence[Sequence[float]]: ...

    @property
    def header(self) -> DetectedHeader: ...

    @property
    def rows(self) -> Sequence[DetectedRow]: ...


class TableFinder(Protocol):
    @property
    def tables(self) -> Sequence[DetectedTable]: ...


@dataclass(frozen=True)
class _Caption:
    kind: str
    bbox: BBox


@dataclass(frozen=True)
class _Candidate:
    table: DetectedTable
    bbox: BBox
    priority: int
    ordinal: int


@dataclass(frozen=True, slots=True)
class _Rule:
    rule_id: NativeRuleId
    x0: float
    x1: float
    y: float


@dataclass(frozen=True, slots=True)
class _VerticalRule:
    x: float
    y0: float
    y1: float


@dataclass(frozen=True)
class _HeaderSnapshot:
    external: bool
    names: tuple[str | None, ...]


@dataclass(frozen=True)
class _RowSnapshot:
    cells: tuple[tuple[float, float, float, float] | None, ...]


@dataclass(frozen=True, slots=True)
class _TableSnapshot:
    bbox: tuple[float, float, float, float]
    cells: tuple[tuple[float, float, float, float], ...]
    header: _HeaderSnapshot
    rows: tuple[_RowSnapshot, ...] | None
    extracted_rows: tuple[tuple[str | None, ...], ...]
    provenance: TableProvenance

    @property
    def finder_provenance(self) -> FinderProvenance:
        return self.provenance.finder

    @property
    def geometric_bands(self) -> tuple[GeometricBand, ...]:
        return self.provenance.geometric_bands

    def extract(self) -> list[list[str | None]]:
        return [list(row) for row in self.extracted_rows]


@dataclass(frozen=True)
class _Coherence:
    row_count: int
    column_count: int
    populated_count: int
    populated_row_count: int


def detect_page_tables(page: pymupdf.Page, *, document_id: str | None = None) -> list[DetectedTable]:
    """Detect tables with authoritative defaults and strongly gated recovery."""
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    if not math.isfinite(page_width) or not math.isfinite(page_height):
        raise ValueError("page must have finite dimensions")
    if page_width <= 0 or page_height <= 0:
        raise ValueError("page must have positive dimensions")

    drawings = cast(list[Mapping[str, object]], page.get_drawings())
    captions = _captions(page)
    native_tokens = _native_word_tokens(page, document_id=document_id)
    native_rules = _native_transformed_rules(page, drawings, document_id=document_id)
    suppression_rules = _native_transformed_rules(page, drawings, stroked_only=True)
    vertical_rules = _native_transformed_vertical_rules(page, drawings)
    defaults = _find_candidates(
        page,
        finder_provenance=FinderProvenance.DEFAULT,
        priority=0,
        ordinal=0,
        native_tokens=native_tokens,
        native_rules=native_rules,
    )
    strict = _find_candidates(
        page,
        finder_provenance=FinderProvenance.LINES_STRICT,
        priority=1,
        ordinal=len(defaults),
        native_tokens=native_tokens,
        native_rules=native_rules,
        vertical_strategy="lines_strict",
        horizontal_strategy="lines_strict",
    )
    authoritative = _deduplicate(defaults, page_width, page_height)
    authoritative = [
        _detach_external_leading_sentence_banner(
            candidate,
            captions,
            suppression_rules,
            vertical_rules,
            page_width,
            page_height,
        )
        for candidate in authoritative
    ]
    authoritative = _suppress_rule_connected_external_strips(
        authoritative,
        captions,
        suppression_rules,
        vertical_rules,
        page_width,
        page_height,
    )
    recovery: list[_Candidate] = []
    ordinal = len(defaults) + len(strict)
    page_number = _page_number(page)
    rules = _horizontal_rules(drawings, page_width, page_height, page_number)

    for band in _horizontal_rule_bands(drawings, page_width, page_height, page_number):
        if not _is_strong_recovery_band(band.bbox, rules, captions, page_width, page_height):
            continue
        text = _find_candidates(
            page,
            finder_provenance=FinderProvenance.RULE_BAND_TEXT,
            priority=2,
            ordinal=ordinal,
            geometric_band=band,
            native_tokens=native_tokens,
            native_rules=native_rules,
            clip=band.bbox,
            vertical_strategy="text",
            horizontal_strategy="text",
        )
        ordinal += len(text)
        mixed = _find_candidates(
            page,
            finder_provenance=FinderProvenance.RULE_BAND_MIXED,
            priority=2,
            ordinal=ordinal,
            geometric_band=band,
            native_tokens=native_tokens,
            native_rules=native_rules,
            clip=band.bbox,
            vertical_strategy="text",
            horizontal_strategy="lines_strict",
        )
        ordinal += len(mixed)
        recovery.extend(_consensus_recovery(text, mixed, band.bbox))

    accepted_recovery = _filter_candidates(recovery, captions, drawings, page_width, page_height)
    additive_recovery = [
        candidate
        for candidate in accepted_recovery
        if all(
            _intersection_over_first(candidate.bbox, default.bbox) <= _MAX_DEFAULT_RECOVERY_OVERLAP
            for default in authoritative
        )
    ]
    detected = _deduplicate(
        [*authoritative, *additive_recovery],
        page_width,
        page_height,
    )
    weak_defaults = [
        candidate for candidate in authoritative if _candidate_coherence(candidate.table) is None
    ]
    if not detected or weak_defaults:
        borderless = _find_candidates(
            page,
            finder_provenance=FinderProvenance.BORDERLESS_TEXT,
            priority=3,
            ordinal=ordinal,
            native_tokens=native_tokens,
            native_rules=native_rules,
            vertical_strategy="text",
            horizontal_strategy="text",
        )
        borderless = [candidate for candidate in borderless if _is_likely_borderless_table(candidate.table)]
        if not detected:
            detected = _deduplicate(borderless, page_width, page_height)
        else:
            detected = _reconcile_weak_defaults(
                detected,
                weak_defaults,
                borderless,
                page_width,
                page_height,
            )
    return [candidate.table for candidate in detected]


def _find_candidates(
    page: pymupdf.Page,
    *,
    finder_provenance: FinderProvenance,
    priority: int,
    ordinal: int,
    native_tokens: tuple[NativeToken, ...],
    native_rules: tuple[NativeRule, ...],
    geometric_band: GeometricBand | None = None,
    **kwargs: object,
) -> list[_Candidate]:
    finder = cast(TableFinder | None, page.find_tables(**kwargs))
    if finder is None:
        raise RuntimeError(f"PyMuPDF did not return a table finder for options {kwargs}")
    candidates: list[_Candidate] = []
    for index, table in enumerate(finder.tables):
        snapshot = _snapshot_table(
            table,
            finder_provenance=finder_provenance,
            geometric_band=geometric_band,
            native_tokens=native_tokens,
            native_rules=native_rules,
        )
        candidates.append(
            _Candidate(
                table=snapshot,
                bbox=_bbox(snapshot.bbox, "table"),
                priority=priority,
                ordinal=ordinal + index,
            )
        )
    return candidates


def _detach_external_leading_sentence_banner(
    candidate: _Candidate,
    captions: Sequence[_Caption],
    rules: Sequence[NativeRule],
    vertical_rules: Sequence[_VerticalRule],
    page_width: float,
    page_height: float,
) -> _Candidate:
    """Crop a sentence-like banner only when the following ruled grid proves it external."""
    table = cast(_TableSnapshot, candidate.table)
    if table.rows is None or len(table.rows) < 3 or len(table.extracted_rows) != len(table.rows):
        return candidate
    first_values = table.extracted_rows[0]
    second_values = table.extracted_rows[1]
    populated_first = [index for index, value in enumerate(first_values) if value and value.strip()]
    populated_second = [index for index, value in enumerate(second_values) if value and value.strip()]
    if populated_first != [0] or len(populated_second) < 3:
        return candidate
    sentence = str(first_values[0]).rstrip()
    if re.search(r"[.!?](?:\s*\.)*$", sentence) is None:
        return candidate

    first_cells = tuple(cell for cell in table.rows[0].cells if cell is not None)
    second_cells = tuple(cell for cell in table.rows[1].cells if cell is not None)
    if not first_cells or len(second_cells) < 3:
        return candidate
    first_bbox = _enclosing_bbox(first_cells)
    second_bbox = _enclosing_bbox(second_cells)
    first_height = first_bbox[3] - first_bbox[1]
    second_height = second_bbox[3] - second_bbox[1]
    table_width = table.bbox[2] - table.bbox[0]
    if (
        first_height > second_height * 0.6
        or (first_cells[0][2] - first_cells[0][0]) / table_width < 0.8
        or any(caption.bbox[3] > first_bbox[1] and caption.bbox[1] < second_bbox[3] for caption in captions)
    ):
        return candidate

    horizontal_tolerance = page_width * _DEDUPLICATION_TOLERANCE_RATIO
    rule_tolerance = page_height * _HORIZONTAL_TOLERANCE_RATIO
    central_first_levels = tuple(
        level
        for level in _internal_vertical_grid_levels(first_bbox, vertical_rules, horizontal_tolerance)
        if table.bbox[0] + table_width * 0.05 < level < table.bbox[2] - table_width * 0.05
    )
    body_bbox = (table.bbox[0], second_bbox[1], table.bbox[2], table.bbox[3])
    body_levels = _internal_vertical_grid_levels(body_bbox, vertical_rules, horizontal_tolerance)
    boundary_supported = any(
        abs(rule.y - second_bbox[1]) <= rule_tolerance
        and max(0.0, min(rule.x1, table.bbox[2]) - max(rule.x0, table.bbox[0])) / table_width >= 0.9
        for rule in rules
    )
    if central_first_levels or len(body_levels) < 3 or not boundary_supported:
        return candidate

    remaining_rows = table.rows[1:]
    remaining_cells = tuple(
        dict.fromkeys(cell for row in remaining_rows for cell in row.cells if cell is not None)
    )
    cropped = _TableSnapshot(
        bbox=body_bbox,
        cells=remaining_cells,
        header=_HeaderSnapshot(external=False, names=second_values),
        rows=remaining_rows,
        extracted_rows=table.extracted_rows[1:],
        provenance=table.provenance,
    )
    return _Candidate(table=cropped, bbox=body_bbox, priority=candidate.priority, ordinal=candidate.ordinal)


def _suppress_rule_connected_external_strips(
    candidates: Sequence[_Candidate],
    captions: Sequence[_Caption],
    rules: Sequence[NativeRule],
    vertical_rules: Sequence[_VerticalRule],
    page_width: float,
    page_height: float,
) -> list[_Candidate]:
    """Drop only lower-cardinality strips proved external to an adjacent ruled grid."""
    retained: list[_Candidate] = []
    horizontal_tolerance = page_width * _DEDUPLICATION_TOLERANCE_RATIO
    max_height = page_height * _MAX_SINGLE_ROW_SATELLITE_HEIGHT_RATIO
    max_gap = page_height * _MAX_SINGLE_ROW_SATELLITE_GAP_RATIO
    rule_y_tolerance = page_height * _HORIZONTAL_TOLERANCE_RATIO
    for satellite in candidates:
        rows = satellite.table.extract()
        column_count = max((len(row) for row in rows), default=0)
        populated = sum(bool(value and str(value).strip()) for row in rows for value in row)
        satellite_height = satellite.bbox[3] - satellite.bbox[1]
        if len(rows) != 1 or column_count < 2 or populated != column_count or satellite_height > max_height:
            retained.append(satellite)
            continue

        is_satellite = any(
            _is_rule_connected_external_strip_of(
                satellite,
                core,
                captions,
                rules,
                vertical_rules,
                column_count,
                satellite_height,
                horizontal_tolerance,
                max_gap,
                rule_y_tolerance,
            )
            for core in candidates
            if core is not satellite
        )
        if not is_satellite:
            retained.append(satellite)
    return retained


def _is_rule_connected_external_strip_of(
    satellite: _Candidate,
    core: _Candidate,
    captions: Sequence[_Caption],
    rules: Sequence[NativeRule],
    vertical_rules: Sequence[_VerticalRule],
    column_count: int,
    satellite_height: float,
    horizontal_tolerance: float,
    max_gap: float,
    rule_y_tolerance: float,
) -> bool:
    core_rows = core.table.extract()
    core_column_count = max((len(row) for row in core_rows), default=0)
    if len(core_rows) < 3 or core_column_count <= column_count:
        return False
    if (
        abs(satellite.bbox[0] - core.bbox[0]) > horizontal_tolerance
        or abs(satellite.bbox[2] - core.bbox[2]) > horizontal_tolerance
        or core.bbox[3] - core.bbox[1] < satellite_height * 3
        or len(_internal_vertical_grid_levels(core.bbox, vertical_rules, horizontal_tolerance)) < 2
        or _internal_vertical_grid_levels(satellite.bbox, vertical_rules, horizontal_tolerance)
    ):
        return False

    if satellite.bbox[3] <= core.bbox[1]:
        gap = core.bbox[1] - satellite.bbox[3]
        union_y0, union_y1 = satellite.bbox[1], core.bbox[1]
        adjacent_core_edge = core.bbox[1]
    elif core.bbox[3] <= satellite.bbox[1]:
        gap = satellite.bbox[1] - core.bbox[3]
        union_y0, union_y1 = core.bbox[3], satellite.bbox[3]
        adjacent_core_edge = core.bbox[3]
    else:
        return False
    if gap > max_gap or any(
        caption.bbox[3] > union_y0 and caption.bbox[1] < union_y1 for caption in captions
    ):
        return False

    width = satellite.bbox[2] - satellite.bbox[0]
    supported_levels: list[float] = []
    for rule in sorted(rules, key=lambda item: (item.y, item.x0, item.x1)):
        if not union_y0 - rule_y_tolerance <= rule.y <= union_y1 + rule_y_tolerance:
            continue
        overlap = max(0.0, min(rule.x1, satellite.bbox[2]) - max(rule.x0, satellite.bbox[0]))
        if overlap / width < _MIN_SATELLITE_RULE_COVERAGE:
            continue
        if not supported_levels or rule.y - supported_levels[-1] > rule_y_tolerance:
            supported_levels.append(rule.y)
    required_edges = (satellite.bbox[1], satellite.bbox[3], adjacent_core_edge)
    return all(
        any(abs(level - edge) <= rule_y_tolerance for level in supported_levels) for edge in required_edges
    )


def _internal_vertical_grid_levels(
    bbox: BBox,
    rules: Sequence[_VerticalRule],
    tolerance: float,
) -> tuple[float, ...]:
    height = bbox[3] - bbox[1]
    by_level: list[tuple[list[float], list[tuple[float, float]]]] = []
    for rule in sorted(rules, key=lambda value: (value.x, value.y0, value.y1)):
        if not bbox[0] + tolerance < rule.x < bbox[2] - tolerance:
            continue
        interval = max(bbox[1], rule.y0), min(bbox[3], rule.y1)
        if interval[1] <= interval[0]:
            continue
        if by_level and rule.x - by_level[-1][0][-1] <= tolerance:
            by_level[-1][0].append(rule.x)
            by_level[-1][1].append(interval)
        else:
            by_level.append(([rule.x], [interval]))

    levels: list[float] = []
    for xs, intervals in by_level:
        merged: list[list[float]] = []
        for top, bottom in sorted(intervals):
            if merged and top <= merged[-1][1] + tolerance:
                merged[-1][1] = max(merged[-1][1], bottom)
            else:
                merged.append([top, bottom])
        coverage = sum(bottom - top for top, bottom in merged) / height
        if coverage >= _MIN_SATELLITE_RULE_COVERAGE:
            levels.append(sum(xs) / len(xs))
    return tuple(levels)


def _filter_candidates(
    candidates: Sequence[_Candidate],
    captions: Sequence[_Caption],
    drawings: Sequence[Mapping[str, object]],
    page_width: float,
    page_height: float,
) -> list[_Candidate]:
    return [
        candidate
        for candidate in candidates
        if _associated_caption(candidate.bbox, captions, page_width, page_height) != "figure"
        and not _is_vector_complex(candidate.bbox, drawings, page_width, page_height)
    ]


def _is_strong_recovery_band(
    band: BBox,
    rules: Sequence[_Rule],
    captions: Sequence[_Caption],
    page_width: float,
    page_height: float,
) -> bool:
    band_rules = sorted(
        rule.y for rule in rules if band[0] <= rule.x1 and rule.x0 <= band[2] and band[1] <= rule.y <= band[3]
    )
    tolerance = page_height * _HORIZONTAL_TOLERANCE_RATIO
    distinct_y: list[float] = []
    for y in band_rules:
        if not distinct_y or y - distinct_y[-1] > tolerance:
            distinct_y.append(y)
    return (
        len(distinct_y) >= _MIN_RULES_PER_RECOVERY_BAND
        and _associated_caption(band, captions, page_width, page_height) == "table"
    )


def _consensus_recovery(
    text: Sequence[_Candidate], mixed: Sequence[_Candidate], band: BBox
) -> list[_Candidate]:
    pairs = sorted(
        (
            (_iou(text_candidate.bbox, mixed_candidate.bbox), text_candidate, mixed_candidate)
            for text_candidate in text
            for mixed_candidate in mixed
            if _recovery_pair_is_coherent(text_candidate, mixed_candidate, band)
        ),
        key=lambda value: (-value[0], _candidate_tiebreak(value[1]), _candidate_tiebreak(value[2])),
    )
    used_text: set[int] = set()
    used_mixed: set[int] = set()
    recovered: list[_Candidate] = []
    for _, text_candidate, mixed_candidate in pairs:
        if text_candidate.ordinal in used_text or mixed_candidate.ordinal in used_mixed:
            continue
        used_text.add(text_candidate.ordinal)
        used_mixed.add(mixed_candidate.ordinal)
        recovered.append(_more_coherent(text_candidate, mixed_candidate))
    return recovered


def _recovery_pair_is_coherent(text: _Candidate, mixed: _Candidate, band: BBox) -> bool:
    text_coherence = _candidate_coherence(text.table)
    mixed_coherence = _candidate_coherence(mixed.table)
    if text_coherence is None or mixed_coherence is None:
        return False
    row_ratio = max(text_coherence.row_count, mixed_coherence.row_count) / min(
        text_coherence.row_count, mixed_coherence.row_count
    )
    column_ratio = max(text_coherence.column_count, mixed_coherence.column_count) / min(
        text_coherence.column_count, mixed_coherence.column_count
    )
    return (
        _iou(text.bbox, mixed.bbox) >= _MIN_RECOVERY_CONSENSUS_IOU
        and row_ratio <= _MAX_RECOVERY_ROW_RATIO
        and column_ratio <= _MAX_RECOVERY_ROW_RATIO
        and _intersection_over_first(text.bbox, band) >= _MIN_RECOVERY_CONSENSUS_IOU
        and _intersection_over_first(mixed.bbox, band) >= _MIN_RECOVERY_CONSENSUS_IOU
    )


def _reconcile_weak_defaults(
    detected: Sequence[_Candidate],
    weak_defaults: Sequence[_Candidate],
    borderless: Sequence[_Candidate],
    page_width: float,
    page_height: float,
) -> list[_Candidate]:
    replacements: dict[int, _Candidate] = {}
    used_borderless: set[int] = set()
    for weak in weak_defaults:
        eligible = [
            candidate
            for candidate in borderless
            if candidate.ordinal not in used_borderless and _is_strong_borderless_replacement(weak, candidate)
        ]
        if not eligible:
            continue
        replacement_score = max(
            (
                _candidate_coherence_score(candidate.table),
                _intersection_over_first(weak.bbox, candidate.bbox),
            )
            for candidate in eligible
        )
        replacement = min(
            (
                candidate
                for candidate in eligible
                if (
                    _candidate_coherence_score(candidate.table),
                    _intersection_over_first(weak.bbox, candidate.bbox),
                )
                == replacement_score
            ),
            key=_candidate_tiebreak,
        )
        replacements[weak.ordinal] = replacement
        used_borderless.add(replacement.ordinal)

    reconciled = [replacements.get(candidate.ordinal, candidate) for candidate in detected]
    return _deduplicate(reconciled, page_width, page_height)


def _is_strong_borderless_replacement(weak: _Candidate, borderless: _Candidate) -> bool:
    coherence = _candidate_coherence(borderless.table)
    if coherence is None:
        return False
    weak_rows = weak.table.extract()
    weak_row_count = len(weak_rows)
    weak_column_count = max((len(row) for row in weak_rows), default=0)
    weak_populated_count = sum(bool(value and value.strip()) for row in weak_rows for value in row)
    weak_capacity = weak_row_count * weak_column_count
    return (
        max(
            _intersection_over_first(weak.bbox, borderless.bbox),
            _intersection_over_first(borderless.bbox, weak.bbox),
        )
        >= _MIN_WEAK_DEFAULT_COVERAGE
        and coherence.row_count >= _MIN_BORDERLESS_ROWS
        and coherence.column_count >= _MIN_BORDERLESS_COLUMNS
        and coherence.populated_count
        >= max(
            _MIN_BORDERLESS_ROWS * _MIN_BORDERLESS_COLUMNS, weak_populated_count * _MIN_RECONCILIATION_GAIN
        )
        and coherence.row_count * coherence.column_count
        >= max(
            _MIN_BORDERLESS_ROWS * _MIN_BORDERLESS_COLUMNS,
            weak_capacity * _MIN_RECONCILIATION_GAIN,
        )
    )


def _candidate_coherence_score(table: DetectedTable) -> tuple[int, int, int]:
    coherence = _candidate_coherence(table)
    if coherence is None:
        raise ValueError("replacement candidates must be coherent")
    return (coherence.populated_count, coherence.populated_row_count, coherence.column_count)


def _is_likely_borderless_table(table: DetectedTable) -> bool:
    rows = table.extract()
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    if row_count < _MIN_BORDERLESS_ROWS or column_count < _MIN_BORDERLESS_COLUMNS:
        return False
    populated_rows = [row for row in rows if sum(bool(value and value.strip()) for value in row) >= 2]
    if len(populated_rows) < _MIN_BORDERLESS_ROWS:
        return False
    nonempty = [value.strip() for row in populated_rows for value in row if value and value.strip()]
    numeric_count = sum(_is_numeric_value(value) for value in nonempty)
    return numeric_count / len(nonempty) >= _MIN_BORDERLESS_NUMERIC_RATIO


def _is_numeric_value(value: str) -> bool:
    normalized = value.strip().replace(",", "").replace(" ", "")
    return bool(re.fullmatch(r"[$€£¥]?\(?[+-]?\d+(?:\.\d+)?%?\)?", normalized))


def _candidate_coherence(table: DetectedTable) -> _Coherence | None:
    rows = table.extract()
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)
    if row_count < 2 or column_count < 2:
        return None
    populated_by_row = [sum(bool(value and str(value).strip()) for value in row) for row in rows]
    populated_count = sum(populated_by_row)
    populated_row_count = sum(count >= 2 for count in populated_by_row)
    if (
        populated_count < 4
        or populated_count / (row_count * column_count) < _MIN_POPULATED_RATIO
        or populated_row_count / row_count < _MIN_POPULATED_ROW_RATIO
    ):
        return None
    return _Coherence(
        row_count=row_count,
        column_count=column_count,
        populated_count=populated_count,
        populated_row_count=populated_row_count,
    )


def _more_coherent(left: _Candidate, right: _Candidate) -> _Candidate:
    left_coherence = _candidate_coherence(left.table)
    right_coherence = _candidate_coherence(right.table)
    if left_coherence is None or right_coherence is None:
        raise ValueError("consensus candidates must be coherent")
    left_score = (left_coherence.populated_count, left_coherence.populated_row_count)
    right_score = (right_coherence.populated_count, right_coherence.populated_row_count)
    if left_score != right_score:
        return left if left_score > right_score else right
    return min((left, right), key=_candidate_tiebreak)


def _snapshot_table(
    table: DetectedTable,
    *,
    finder_provenance: FinderProvenance,
    geometric_band: GeometricBand | None,
    native_tokens: tuple[NativeToken, ...],
    native_rules: tuple[NativeRule, ...],
) -> _TableSnapshot:
    extracted_rows = tuple(tuple(row) for row in table.extract())
    cells: tuple[BBox, ...] = ()
    rows: tuple[_RowSnapshot, ...] | None = None
    header_snapshot = _HeaderSnapshot(external=True, names=())
    if hasattr(table, "cells"):
        snapshot_capable = cast(SnapshotCapableTable, table)
        cells = tuple(_bbox_like(cell, "table cell") for cell in snapshot_capable.cells)
    if hasattr(table, "header"):
        snapshot_capable = cast(SnapshotCapableTable, table)
        header = snapshot_capable.header
        header_snapshot = _HeaderSnapshot(
            external=bool(header.external),
            names=tuple(header.names),
        )
    if hasattr(table, "rows"):
        candidate_rows = cast(Sequence[object], getattr(table, "rows"))
        if all(hasattr(row, "cells") for row in candidate_rows):
            geometry_rows = cast(Sequence[DetectedRow], candidate_rows)
            rows = tuple(
                _RowSnapshot(
                    cells=tuple(
                        None if cell is None else _bbox_like(cell, "table row cell") for cell in row.cells
                    )
                )
                for row in geometry_rows
            )

    snapshot_bbox = _bbox(table.bbox, "table")
    # Retain a narrow, table-relative lookback so reconstruction can prove that
    # externally detected date/year tiers continue the grid. Keep this invariant
    # under uniformly scaled page geometry; the reconstruction gate, rather than
    # proximity alone, decides whether any of this text belongs to the table.
    header_lookback = float(snapshot_bbox[3] - snapshot_bbox[1]) * 0.5
    table_tokens = tuple(
        token
        for token in native_tokens
        if snapshot_bbox[0] <= (token.bbox[0] + token.bbox[2]) / 2 <= snapshot_bbox[2]
        and snapshot_bbox[1] - header_lookback
        <= token.baseline
        <= snapshot_bbox[3] + (token.bbox[3] - token.bbox[1]) * 0.25
    )
    table_rules = tuple(
        rule
        for rule in native_rules
        if snapshot_bbox[0] <= rule.x1
        and rule.x0 <= snapshot_bbox[2]
        and snapshot_bbox[1] - header_lookback <= rule.y <= snapshot_bbox[3]
    )
    evidence_rules = table_rules
    if geometric_band is not None:
        band_rule_ids = tuple(
            rule.rule_id
            for rule in native_rules
            if geometric_band.bbox[0] <= rule.x1
            and rule.x0 <= geometric_band.bbox[2]
            and geometric_band.bbox[1] <= rule.y <= geometric_band.bbox[3]
        )
        geometric_band = GeometricBand(
            band_id=geometric_band.band_id,
            kind=geometric_band.kind,
            frame=geometric_band.frame,
            bbox=geometric_band.bbox,
            native_rule_ids=band_rule_ids,
        )
        evidence_rules = tuple(rule for rule in native_rules if rule.rule_id in set(band_rule_ids))
    row_bands = tuple(
        GeometricBand(
            band_id=f"finder-row-{index:04d}",
            kind=GeometricBandKind.FINDER_ROW,
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=(band_bbox := _enclosing_bbox(tuple(cell for cell in row.cells if cell is not None))),
            native_token_ids=tuple(
                token.token_id
                for token in table_tokens
                if band_bbox[0] <= (token.bbox[0] + token.bbox[2]) / 2 <= band_bbox[2]
                and band_bbox[1] <= token.baseline <= band_bbox[3]
            ),
            native_rule_ids=tuple(
                rule.rule_id
                for rule in evidence_rules
                if band_bbox[0] <= rule.x1
                and rule.x0 <= band_bbox[2]
                and band_bbox[1] <= rule.y <= band_bbox[3]
            ),
        )
        for index, row in enumerate(rows or (), start=1)
        if any(cell is not None for cell in row.cells)
    )
    bands = (*row_bands, *((geometric_band,) if geometric_band is not None else ()))
    table_rule_ids = (
        geometric_band.native_rule_ids
        if geometric_band is not None
        else tuple(rule.rule_id for rule in evidence_rules)
    )
    return _TableSnapshot(
        bbox=snapshot_bbox,
        cells=cells,
        header=header_snapshot,
        rows=rows,
        extracted_rows=extracted_rows,
        provenance=TableProvenance(
            finder=finder_provenance,
            frame=CoordinateFrame.DETECTOR_PAGE,
            geometric_bands=bands,
            native_token_ids=tuple(token.token_id for token in table_tokens),
            native_rule_ids=table_rule_ids,
            native_tokens=table_tokens,
            native_rules=evidence_rules,
        ),
    )


def _page_number(page: pymupdf.Page) -> int:
    page_index = getattr(page, "number", 0)
    if page_index is None:
        raise ValueError("page must belong to an open PyMuPDF document")
    if type(page_index) is not int or page_index < 0:
        raise ValueError("page number must be a non-negative integer")
    return page_index + 1


def _native_word_tokens(page: pymupdf.Page, *, document_id: str | None = None) -> tuple[NativeToken, ...]:
    raw_words = cast(Sequence[Sequence[object]], page.get_text("words", sort=True))
    matrix = getattr(page, "rotation_matrix", pymupdf.Matrix(1, 0, 0, 1, 0, 0))
    page_number = _page_number(page)
    records: list[tuple[BBox, BBox, str, tuple[str, ...]]] = []
    for raw_word in raw_words:
        if len(raw_word) < 5:
            raise ValueError(f"native word must have at least five fields: {raw_word}")
        text = str(raw_word[4])
        if not text.strip():
            continue
        canonical_bbox = _bbox(raw_word[:4], "native word")
        rectangle = pymupdf.Rect(*canonical_bbox) * matrix
        bbox = _bbox_like(rectangle, "transformed native word")
        # PyMuPDF's block/line/word tuple, when present, is a deterministic tie
        # key for indistinguishable duplicate native observations. It is not part
        # of the content identity hash.
        tie_key = tuple(str(value) for value in raw_word[5:8])
        records.append((bbox, canonical_bbox, text, tie_key))
    records.sort(key=lambda item: (item[1][1], item[1][0], item[1][3], item[1][2], item[2], item[3]))
    duplicate_counts: dict[str, int] = {}
    tokens: list[NativeToken] = []
    for index, (bbox, canonical_bbox, text, _tie_key) in enumerate(records, start=1):
        if document_id is None:
            token_value = f"pymupdf:p{page_number:06d}:word:w{index:06d}"
        else:
            identity, _, _ = source_item_identity_sha256(
                document_id=document_id,
                page_number=page_number,
                kind="word",
                canonical_coordinates=canonical_bbox,
                text=text,
            )
            duplicate_counts[identity] = duplicate_counts.get(identity, 0) + 1
            token_value = content_addressed_source_item_id(
                document_id=document_id,
                page_number=page_number,
                kind="word",
                canonical_coordinates=canonical_bbox,
                text=text,
                duplicate_index=duplicate_counts[identity],
            )
        tokens.append(
            NativeToken(
                token_id=NativeTokenId(token_value),
                frame=CoordinateFrame.DETECTOR_PAGE,
                bbox=bbox,
                baseline=bbox[3],
                text=text,
                canonical_bbox=canonical_bbox,
            )
        )
    return tuple(sorted(tokens, key=lambda token: (token.bbox[1], token.bbox[0], token.token_id.value)))


def _native_transformed_vertical_rules(
    page: pymupdf.Page,
    drawings: Sequence[Mapping[str, object]],
) -> tuple[_VerticalRule, ...]:
    """Return stroked vertical segments; fill-only rectangle edges are not rule evidence."""
    matrix = getattr(page, "rotation_matrix", pymupdf.Matrix(1, 0, 0, 1, 0, 0))
    tolerance = min(float(page.rect.width), float(page.rect.height)) * _HORIZONTAL_TOLERANCE_RATIO
    vertical: set[tuple[float, float, float]] = set()
    for drawing in drawings:
        if drawing.get("color") is None:
            continue
        raw_items = drawing.get("items")
        if raw_items is None:
            raise ValueError("drawing is missing items")
        for raw_item in cast(Sequence[Sequence[object]], raw_items):
            if not raw_item:
                raise ValueError("drawing item must not be empty")
            segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
            if str(raw_item[0]) == "l":
                if len(raw_item) < 3:
                    raise ValueError(f"line drawing item is incomplete: {raw_item}")
                segments.append((_point(raw_item[1]), _point(raw_item[2])))
            elif str(raw_item[0]) == "re":
                if len(raw_item) < 2:
                    raise ValueError(f"rectangle drawing item is incomplete: {raw_item}")
                rectangle = _bbox_like(raw_item[1], "drawing rectangle")
                segments.extend([
                    ((rectangle[0], rectangle[1]), (rectangle[0], rectangle[3])),
                    ((rectangle[2], rectangle[1]), (rectangle[2], rectangle[3])),
                ])
            for start, end in segments:
                transformed_start = pymupdf.Point(*start) * matrix
                transformed_end = pymupdf.Point(*end) * matrix
                if abs(transformed_end.x - transformed_start.x) > tolerance:
                    continue
                y0 = min(float(transformed_start.y), float(transformed_end.y))
                y1 = max(float(transformed_start.y), float(transformed_end.y))
                if y1 - y0 > tolerance:
                    vertical.add(((float(transformed_start.x) + float(transformed_end.x)) / 2, y0, y1))
    return tuple(_VerticalRule(x=x, y0=y0, y1=y1) for x, y0, y1 in sorted(vertical))


def _native_transformed_rules(
    page: pymupdf.Page,
    drawings: Sequence[Mapping[str, object]],
    *,
    document_id: str | None = None,
    stroked_only: bool = False,
) -> tuple[NativeRule, ...]:
    matrix = getattr(page, "rotation_matrix", pymupdf.Matrix(1, 0, 0, 1, 0, 0))
    horizontal: set[tuple[float, float, float, tuple[float, float, float, float]]] = set()
    tolerance = min(float(page.rect.width), float(page.rect.height)) * _HORIZONTAL_TOLERANCE_RATIO
    for drawing in drawings:
        if stroked_only and drawing.get("color") is None:
            continue
        raw_items = drawing.get("items")
        if raw_items is None:
            raise ValueError("drawing is missing items")
        for raw_item in cast(Sequence[Sequence[object]], raw_items):
            if not raw_item:
                raise ValueError("drawing item must not be empty")
            segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
            if str(raw_item[0]) == "l":
                if len(raw_item) < 3:
                    raise ValueError(f"line drawing item is incomplete: {raw_item}")
                segments.append((_point(raw_item[1]), _point(raw_item[2])))
            elif str(raw_item[0]) == "re":
                if len(raw_item) < 2:
                    raise ValueError(f"rectangle drawing item is incomplete: {raw_item}")
                rectangle = _bbox_like(raw_item[1], "drawing rectangle")
                segments.extend([
                    ((rectangle[0], rectangle[1]), (rectangle[2], rectangle[1])),
                    ((rectangle[0], rectangle[3]), (rectangle[2], rectangle[3])),
                    ((rectangle[0], rectangle[1]), (rectangle[0], rectangle[3])),
                    ((rectangle[2], rectangle[1]), (rectangle[2], rectangle[3])),
                ])
            for start, end in segments:
                transformed_start = pymupdf.Point(*start) * matrix
                transformed_end = pymupdf.Point(*end) * matrix
                if abs(transformed_end.y - transformed_start.y) <= tolerance:
                    x0 = min(float(transformed_start.x), float(transformed_end.x))
                    x1 = max(float(transformed_start.x), float(transformed_end.x))
                    if x1 - x0 > tolerance:
                        canonical = (
                            min(start[0], end[0]),
                            min(start[1], end[1]),
                            max(start[0], end[0]),
                            max(start[1], end[1]),
                        )
                        horizontal.add((
                            x0,
                            x1,
                            (float(transformed_start.y) + float(transformed_end.y)) / 2,
                            canonical,
                        ))
    ordered = sorted(horizontal, key=lambda value: (value[3], value[2], value[0], value[1]))
    page_number = _page_number(page)
    duplicate_counts: dict[str, int] = {}
    rules: list[NativeRule] = []
    for index, (x0, x1, y, canonical) in enumerate(ordered, start=1):
        if document_id is None:
            rule_value = f"pymupdf:p{page_number:06d}:rule:r{index:06d}"
        else:
            identity, _, _ = source_item_identity_sha256(
                document_id=document_id,
                page_number=page_number,
                kind="rule",
                canonical_coordinates=canonical,
                text=None,
            )
            duplicate_counts[identity] = duplicate_counts.get(identity, 0) + 1
            rule_value = content_addressed_source_item_id(
                document_id=document_id,
                page_number=page_number,
                kind="rule",
                canonical_coordinates=canonical,
                text=None,
                duplicate_index=duplicate_counts[identity],
            )
        rules.append(
            NativeRule(
                rule_id=NativeRuleId(rule_value),
                frame=CoordinateFrame.DETECTOR_PAGE,
                x0=x0,
                x1=x1,
                y=y,
                canonical_geometry=canonical,
            )
        )
    return tuple(sorted(rules, key=lambda rule: (rule.y, rule.x0, rule.x1, rule.rule_id.value)))


def _captions(page: pymupdf.Page) -> list[_Caption]:
    blocks = cast(Sequence[Sequence[object]], page.get_text("blocks", sort=True))
    captions: list[_Caption] = []
    for block in blocks:
        if len(block) < 5:
            raise ValueError(f"text block must have at least five fields: {block}")
        match = _CAPTION_PATTERN.match(str(block[4]))
        if match is not None:
            captions.append(_Caption(kind=match.group(1).lower(), bbox=_bbox(block[:4], "caption")))
    return captions


def _horizontal_rule_bands(
    drawings: Sequence[Mapping[str, object]],
    page_width: float,
    page_height: float,
    page_number: int,
) -> list[GeometricBand]:
    rules = sorted(
        _horizontal_rules(drawings, page_width, page_height, page_number),
        key=lambda rule: (rule.y, rule.x0),
    )
    groups: list[list[_Rule]] = []
    max_gap = page_height * _MAX_RULE_GAP_RATIO
    for rule in rules:
        compatible_groups = [
            group
            for group in groups
            if rule.y - group[-1].y <= max_gap and _horizontal_overlap(rule, group[-1]) > 0
        ]
        if compatible_groups:
            compatible_groups[-1].append(rule)
        else:
            groups.append([rule])

    margin_x = page_width * _BAND_MARGIN_RATIO
    margin_y = page_height * _BAND_MARGIN_RATIO
    min_height = page_height * _MIN_BAND_HEIGHT_RATIO
    band_rules: dict[BBox, set[NativeRuleId]] = {}
    for group in groups:
        if len({rule.y for rule in group}) < 2 or group[-1].y - group[0].y < min_height:
            continue
        bbox = (
            max(0.0, min(rule.x0 for rule in group) - margin_x),
            max(0.0, group[0].y - margin_y),
            min(page_width, max(rule.x1 for rule in group) + margin_x),
            min(page_height, group[-1].y + margin_y),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        band_rules.setdefault(bbox, set()).update(rule.rule_id for rule in group)
    ordered = sorted(band_rules.items(), key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
    return [
        GeometricBand(
            band_id=f"rule-band-{index:04d}",
            kind=GeometricBandKind.RULE_BAND,
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=bbox,
            native_rule_ids=tuple(sorted(rule_ids)),
        )
        for index, (bbox, rule_ids) in enumerate(ordered, start=1)
    ]


def _horizontal_rules(
    drawings: Sequence[Mapping[str, object]],
    page_width: float,
    page_height: float,
    page_number: int,
) -> list[_Rule]:
    segments_found: set[tuple[float, float, float]] = set()
    min_width = page_width * _MIN_RULE_WIDTH_RATIO
    tolerance = page_height * _HORIZONTAL_TOLERANCE_RATIO
    for drawing in drawings:
        raw_items = drawing.get("items")
        if raw_items is None:
            raise ValueError("drawing is missing items")
        for raw_item in cast(Sequence[Sequence[object]], raw_items):
            if not raw_item:
                raise ValueError("drawing item must not be empty")
            kind = str(raw_item[0])
            segments: list[tuple[float, float, float, float]] = []
            if kind == "l":
                if len(raw_item) < 3:
                    raise ValueError(f"line drawing item is incomplete: {raw_item}")
                start = _point(raw_item[1])
                end = _point(raw_item[2])
                segments.append((start[0], start[1], end[0], end[1]))
            elif kind == "re":
                if len(raw_item) < 2:
                    raise ValueError(f"rectangle drawing item is incomplete: {raw_item}")
                rectangle = _bbox_like(raw_item[1], "drawing rectangle")
                segments.extend([
                    (rectangle[0], rectangle[1], rectangle[2], rectangle[1]),
                    (rectangle[0], rectangle[3], rectangle[2], rectangle[3]),
                ])
            for x0, y0, x1, y1 in segments:
                if abs(y1 - y0) <= tolerance and abs(x1 - x0) >= min_width:
                    segment = (min(x0, x1), max(x0, x1), (y0 + y1) / 2)
                    if not all(math.isfinite(value) for value in segment):
                        raise ValueError(f"horizontal rule must have finite coordinates: {segment}")
                    segments_found.add(segment)
    ordered = sorted(segments_found, key=lambda value: (value[2], value[0], value[1]))
    return [
        _Rule(
            rule_id=NativeRuleId(f"pymupdf:p{page_number:06d}:rule:r{index:06d}"),
            x0=x0,
            x1=x1,
            y=y,
        )
        for index, (x0, x1, y) in enumerate(ordered, start=1)
    ]


def _associated_caption(
    bbox: BBox, captions: Sequence[_Caption], page_width: float, page_height: float
) -> str | None:
    max_distance = page_height * _CAPTION_DISTANCE_RATIO
    horizontal_slop = page_width * _CAPTION_HORIZONTAL_SLOP_RATIO
    nearby: list[tuple[float, float, float, str]] = []
    for caption in captions:
        if min(bbox[2], caption.bbox[2]) + horizontal_slop < max(bbox[0], caption.bbox[0]):
            continue
        vertical_distance = max(caption.bbox[1] - bbox[3], bbox[1] - caption.bbox[3], 0.0)
        if vertical_distance <= max_distance:
            nearby.append((vertical_distance, caption.bbox[1], caption.bbox[0], caption.kind))
    if not nearby:
        return None
    return min(nearby)[3]


def _is_vector_complex(
    bbox: BBox,
    drawings: Sequence[Mapping[str, object]],
    page_width: float,
    page_height: float,
) -> bool:
    candidate_width = bbox[2] - bbox[0]
    candidate_height = bbox[3] - bbox[1]
    candidate_area_ratio = candidate_width * candidate_height / (page_width * page_height)
    complexity_threshold = max(
        _MIN_COMPLEX_VECTOR_ITEMS, math.ceil(candidate_area_ratio * _COMPLEX_VECTOR_DENSITY)
    )
    complex_items = 0
    axis_aligned_items = 0
    for drawing in drawings:
        drawing_bbox = _drawing_bbox(drawing)
        if drawing_bbox is None or not _intersects(bbox, drawing_bbox):
            continue
        raw_items = drawing.get("items")
        if raw_items is None:
            raise ValueError("drawing is missing items")
        for raw_item in cast(Sequence[Sequence[object]], raw_items):
            if not raw_item:
                raise ValueError("drawing item must not be empty")
            kind = str(raw_item[0])
            if kind == "re":
                axis_aligned_items += 1
            elif kind == "l" and len(raw_item) >= 3:
                start = _point(raw_item[1])
                end = _point(raw_item[2])
                if math.isclose(start[0], end[0]) or math.isclose(start[1], end[1]):
                    axis_aligned_items += 1
                else:
                    complex_items += 1
            else:
                complex_items += 1
    return complex_items >= complexity_threshold and complex_items > axis_aligned_items


def _candidate_tiebreak(
    candidate: _Candidate,
) -> tuple[int, BBox, tuple[BBox, ...], tuple[tuple[str, ...], ...]]:
    table = cast(_TableSnapshot, candidate.table)
    finder_order = {
        FinderProvenance.DEFAULT: 0,
        FinderProvenance.LINES_STRICT: 1,
        FinderProvenance.RULE_BAND_TEXT: 2,
        FinderProvenance.RULE_BAND_MIXED: 3,
        FinderProvenance.BORDERLESS_TEXT: 4,
    }
    extracted = tuple(
        tuple("\x00" if value is None else str(value) for value in row) for row in table.extracted_rows
    )
    return (finder_order[table.finder_provenance], table.bbox, table.cells, extracted)


def _deduplicate(candidates: Sequence[_Candidate], page_width: float, page_height: float) -> list[_Candidate]:
    tolerance = min(page_width, page_height) * _DEDUPLICATION_TOLERANCE_RATIO
    retained: list[_Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda value: (
            value.priority,
            -_area(value.bbox),
            value.bbox[1],
            value.bbox[0],
            _candidate_tiebreak(value),
        ),
    ):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(retained)
                if _same_candidate_bbox(candidate.bbox, existing.bbox, tolerance)
            ),
            None,
        )
        if duplicate_index is None:
            retained.append(candidate)
    return sorted(retained, key=lambda value: (value.bbox[1], value.bbox[0], value.bbox[3], value.bbox[2]))


def _same_candidate_bbox(left: BBox, right: BBox, tolerance: float) -> bool:
    return (
        all(abs(left[index] - right[index]) <= tolerance for index in range(4))
        or _iou(left, right) >= _MIN_DUPLICATE_IOU
        or _intersection_over_first(left, right) >= _MIN_DUPLICATE_IOU
        or _intersection_over_first(right, left) >= _MIN_DUPLICATE_IOU
    )


def _intersection_over_first(first: BBox, second: BBox) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return intersection_width * intersection_height / _area(first)


def _iou(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    union = _area(left) + _area(right) - intersection
    return intersection / union if union > 0 else 0.0


def _drawing_bbox(drawing: Mapping[str, object]) -> BBox | None:
    rectangle = drawing.get("rect")
    if rectangle is None:
        return None
    if isinstance(rectangle, Sequence):
        coordinates = cast(Sequence[object], rectangle)
        if len(coordinates) != 4:
            raise ValueError(f"drawing bbox must have four coordinates: {rectangle}")
        bbox = tuple(float(str(value)) for value in coordinates)
    else:
        bbox = (
            float(str(getattr(rectangle, "x0"))),
            float(str(getattr(rectangle, "y0"))),
            float(str(getattr(rectangle, "x1"))),
            float(str(getattr(rectangle, "y1"))),
        )
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"drawing bbox must contain finite coordinates: {bbox}")
    return cast(BBox, bbox)


def _enclosing_bbox(cells: tuple[BBox, ...]) -> BBox:
    if not cells:
        raise ValueError("cannot create a finder row band without cells")
    return _bbox(
        (
            min(cell[0] for cell in cells),
            min(cell[1] for cell in cells),
            max(cell[2] for cell in cells),
            max(cell[3] for cell in cells),
        ),
        "finder row band",
    )


def _bbox(values: Sequence[object], label: str) -> BBox:
    if len(values) != 4:
        raise ValueError(f"{label} bbox must have four coordinates: {values}")
    bbox = (
        float(str(values[0])),
        float(str(values[1])),
        float(str(values[2])),
        float(str(values[3])),
    )
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{label} bbox must contain finite coordinates: {bbox}")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{label} bbox must have positive dimensions: {bbox}")
    return bbox


def _bbox_like(value: object, label: str) -> BBox:
    if isinstance(value, Sequence):
        return _bbox(cast(Sequence[object], value), label)
    coordinates = value
    return _bbox(
        [
            getattr(coordinates, "x0"),
            getattr(coordinates, "y0"),
            getattr(coordinates, "x1"),
            getattr(coordinates, "y1"),
        ],
        label,
    )


def _point(value: object) -> tuple[float, float]:
    if isinstance(value, Sequence):
        coordinates = cast(Sequence[object], value)
        if len(coordinates) != 2:
            raise ValueError(f"point must have two coordinates: {value}")
        point = float(str(coordinates[0])), float(str(coordinates[1]))
    else:
        point = float(str(getattr(value, "x"))), float(str(getattr(value, "y")))
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError(f"point must have finite coordinates: {point}")
    return point


def _horizontal_overlap(left: _Rule, right: _Rule) -> float:
    return max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))


def _intersects(left: BBox, right: BBox) -> bool:
    return left[0] < right[2] and right[0] < left[2] and left[1] < right[3] and right[1] < left[3]


def _area(bbox: BBox) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
