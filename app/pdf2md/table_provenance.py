from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import median
from typing import Protocol, cast

BBox = tuple[float, float, float, float]
_FINDER_GRID_V1 = "finder_grid_v1"


class CoordinateFrame(StrEnum):
    """Coordinate system in which detector evidence was observed."""

    DETECTOR_PAGE = "detector_page"


class FinderProvenance(StrEnum):
    """The PyMuPDF finder pass that produced a table candidate."""

    DEFAULT = "default"
    LINES_STRICT = "lines_strict"
    RULE_BAND_TEXT = "rule_band_text"
    RULE_BAND_MIXED = "rule_band_mixed"
    BORDERLESS_TEXT = "borderless_text"


class GeometricBandKind(StrEnum):
    FINDER_ROW = "finder_row"
    RULE_BAND = "rule_band"


@dataclass(frozen=True, slots=True, order=True)
class NativeTokenId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("native token ID must not be empty")


@dataclass(frozen=True, slots=True, order=True)
class NativeRuleId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("native rule ID must not be empty")


@dataclass(frozen=True, slots=True)
class NativeToken:
    """One immutable native word in detector-page coordinates."""

    token_id: NativeTokenId
    frame: CoordinateFrame
    bbox: BBox
    baseline: float
    text: str
    canonical_bbox: BBox | None = None

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox, "native token")
        if self.canonical_bbox is not None:
            _validate_bbox(self.canonical_bbox, "canonical native token")
        if not math.isfinite(self.baseline):
            raise ValueError("native token baseline must be finite")
        if not self.text or not self.text.strip():
            raise ValueError("native token text must not be blank")


@dataclass(frozen=True, slots=True)
class NativeRule:
    """One transformed horizontal rule in detector-page coordinates."""

    rule_id: NativeRuleId
    frame: CoordinateFrame
    x0: float
    x1: float
    y: float
    canonical_geometry: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.x0, self.x1, self.y)):
            raise ValueError("native rule geometry must be finite")
        if self.x1 <= self.x0:
            raise ValueError("native rule must have positive width")
        if self.canonical_geometry is not None:
            if not all(math.isfinite(value) for value in self.canonical_geometry):
                raise ValueError("canonical native rule geometry must be finite")
            x0, y0, x1, y1 = self.canonical_geometry
            if x0 == x1 and y0 == y1:
                raise ValueError("canonical native rule must not be a point")


@dataclass(frozen=True, slots=True)
class GeometricBand:
    """Immutable detector geometry, deliberately separate from logical table rows."""

    band_id: str
    kind: GeometricBandKind
    frame: CoordinateFrame
    bbox: tuple[float, float, float, float]
    native_token_ids: tuple[NativeTokenId, ...] = ()
    native_rule_ids: tuple[NativeRuleId, ...] = ()

    def __post_init__(self) -> None:
        if not self.band_id:
            raise ValueError("geometric band ID must not be empty")
        _validate_bbox(self.bbox, "geometric band")
        _require_unique(self.native_token_ids, "native token IDs within a geometric band")
        _require_unique(self.native_rule_ids, "native rule IDs within a geometric band")


@dataclass(frozen=True, slots=True)
class TableProvenance:
    """Closed-lifetime evidence retained for one finder snapshot."""

    finder: FinderProvenance
    frame: CoordinateFrame
    geometric_bands: tuple[GeometricBand, ...] = ()
    native_token_ids: tuple[NativeTokenId, ...] = ()
    native_rule_ids: tuple[NativeRuleId, ...] = ()
    native_tokens: tuple[NativeToken, ...] = ()
    native_rules: tuple[NativeRule, ...] = ()

    def __post_init__(self) -> None:
        band_ids = tuple(band.band_id for band in self.geometric_bands)
        _require_unique(band_ids, "geometric band IDs")
        _require_unique(self.native_token_ids, "native token IDs")
        _require_unique(self.native_rule_ids, "native rule IDs")
        _require_unique(tuple(token.token_id for token in self.native_tokens), "native token evidence IDs")
        _require_unique(tuple(rule.rule_id for rule in self.native_rules), "native rule evidence IDs")
        if (
            self.native_tokens
            and tuple(token.token_id for token in self.native_tokens) != self.native_token_ids
        ):
            raise ValueError("native token evidence must match native token IDs")
        if self.native_rules and tuple(rule.rule_id for rule in self.native_rules) != self.native_rule_ids:
            raise ValueError("native rule evidence must match native rule IDs")
        if any(token.frame != self.frame for token in self.native_tokens):
            raise ValueError("native tokens must use the table provenance coordinate frame")
        if any(rule.frame != self.frame for rule in self.native_rules):
            raise ValueError("native rules must use the table provenance coordinate frame")
        if any(band.frame != self.frame for band in self.geometric_bands):
            raise ValueError("geometric bands must use the table provenance coordinate frame")

        known_tokens = set(self.native_token_ids)
        known_rules = set(self.native_rule_ids)
        referenced_tokens = {item for band in self.geometric_bands for item in band.native_token_ids}
        referenced_rules = {item for band in self.geometric_bands for item in band.native_rule_ids}
        dangling_tokens = referenced_tokens - known_tokens
        dangling_rules = referenced_rules - known_rules
        if dangling_tokens:
            raise ValueError(f"geometric bands reference unknown native token IDs: {sorted(dangling_tokens)}")
        if dangling_rules:
            raise ValueError(f"geometric bands reference unknown native rule IDs: {sorted(dangling_rules)}")


class _FinderHeader(Protocol):
    @property
    def external(self) -> bool: ...

    @property
    def names(self) -> Sequence[str | None]: ...


class _FinderRow(Protocol):
    @property
    def cells(self) -> Sequence[Sequence[float] | None]: ...


class FinderSnapshot(Protocol):
    """Minimum finder snapshot protocol accepted by the reconstruction seam."""

    @property
    def bbox(self) -> Sequence[float]: ...

    @property
    def cells(self) -> Sequence[Sequence[float]]: ...

    @property
    def header(self) -> _FinderHeader: ...

    def extract(self) -> list[list[str | None]]: ...


@dataclass(frozen=True, slots=True)
class FinderHeaderInput:
    external: bool
    names: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class LogicalCellInput:
    """One finder-grid text/geometry association consumed by conversion."""

    row_index: int
    column_index: int
    bbox: BBox
    text: str
    rowspan: int = 1

    def __post_init__(self) -> None:
        if self.row_index < 0 or self.column_index < 0:
            raise ValueError("logical cell indices must be non-negative")
        if self.rowspan < 1:
            raise ValueError("logical cell rowspan must be positive")
        _validate_bbox(self.bbox, "logical cell")


@dataclass(frozen=True, slots=True)
class ProvenanceSummary:
    finder: FinderProvenance
    frame: CoordinateFrame
    geometric_band_ids: tuple[str, ...]
    native_token_count: int
    native_rule_count: int


@dataclass(frozen=True, slots=True)
class TableReconstruction:
    """Immutable finder_grid_v1 inputs retained behind the conversion boundary."""

    adapter: str
    bbox: BBox
    cells: tuple[BBox, ...]
    logical_cells: tuple[LogicalCellInput, ...]
    extracted_rows: tuple[tuple[str | None, ...], ...]
    header: FinderHeaderInput
    provenance: ProvenanceSummary | None
    logical_grid_indices: bool = False
    recovered_header_row_count: int | None = None

    def __post_init__(self) -> None:
        if self.adapter != _FINDER_GRID_V1:
            raise ValueError(f"unsupported table reconstruction adapter: {self.adapter}")
        _validate_bbox(self.bbox, "table")
        if not self.cells:
            raise ValueError("finder grid must contain at least one cell")


def reconstruct_table(
    snapshot: FinderSnapshot,
    evidence: TableProvenance | None,
) -> TableReconstruction:
    """Adapt a finder snapshot to immutable finder_grid_v1 reconstruction inputs."""
    table_bbox = _bbox(snapshot.bbox, "table")
    cells = tuple(_coordinates(cell, "finder grid cell") for cell in snapshot.cells)
    if not cells:
        raise ValueError("finder grid must contain at least one cell")
    extracted_rows = tuple(tuple(row) for row in snapshot.extract())
    header = FinderHeaderInput(
        external=bool(snapshot.header.external),
        names=tuple(snapshot.header.names),
    )
    geometry_rows = _geometry_rows(snapshot)
    logical_cells = (
        _logical_cells_from_rows(cells, geometry_rows, extracted_rows)
        if geometry_rows is not None
        else _logical_cells_from_flat_grid(cells, extracted_rows)
    )
    provenance = None
    logical_grid_indices = False
    recovered_header_row_count: int | None = None
    if evidence is not None:
        provenance = ProvenanceSummary(
            finder=evidence.finder,
            frame=evidence.frame,
            geometric_band_ids=tuple(band.band_id for band in evidence.geometric_bands),
            native_token_count=len(evidence.native_token_ids),
            native_rule_count=len(evidence.native_rule_ids),
        )
    if evidence is not None:
        external_header = _reconstruct_external_financial_header(
            table_bbox,
            cells,
            geometry_rows,
            extracted_rows,
            evidence,
        )
        if external_header is not None:
            table_bbox, cells, logical_cells, extracted_rows, header = external_header
            logical_grid_indices = True
        else:
            external_grid_header = _reconstruct_rule_connected_external_header(
                table_bbox,
                cells,
                logical_cells,
                geometry_rows,
                extracted_rows,
                evidence,
            )
            if external_grid_header is not None:
                (
                    table_bbox,
                    cells,
                    logical_cells,
                    extracted_rows,
                    header,
                    recovered_header_row_count,
                ) = external_grid_header
                logical_grid_indices = True
            else:
                ruled_inset = _reconstruct_ruled_inset_grid(
                    table_bbox,
                    geometry_rows,
                    extracted_rows,
                    evidence,
                )
                if ruled_inset is not None:
                    cells, logical_cells, extracted_rows = ruled_inset
                    header = FinderHeaderInput(external=False, names=extracted_rows[0])
                    logical_grid_indices = True
                else:
                    baseline_grid = _reconstruct_rule_backed_baseline_grid(
                        table_bbox,
                        geometry_rows,
                        extracted_rows,
                        evidence,
                    )
                    if baseline_grid is not None:
                        cells, logical_cells, extracted_rows = baseline_grid
                        header = FinderHeaderInput(external=False, names=extracted_rows[0])
                        recovered_header_row_count = 1
                        logical_grid_indices = True
                    else:
                        ruled_form = _reconstruct_rule_partitioned_form_grid(
                            table_bbox,
                            geometry_rows,
                            extracted_rows,
                            evidence,
                        )
                        if ruled_form is not None:
                            table_bbox, cells, logical_cells, extracted_rows = ruled_form
                            header = FinderHeaderInput(external=False, names=extracted_rows[0])
                            logical_grid_indices = True
                        elif evidence.finder == FinderProvenance.BORDERLESS_TEXT:
                            reconstructed = _reconstruct_borderless_text(
                                table_bbox,
                                cells,
                                geometry_rows,
                                extracted_rows,
                                evidence,
                            )
                            if reconstructed is not None:
                                table_bbox, cells, logical_cells, extracted_rows = reconstructed
                                logical_grid_indices = True
    return TableReconstruction(
        adapter=_FINDER_GRID_V1,
        bbox=table_bbox,
        cells=cells,
        logical_cells=logical_cells,
        extracted_rows=extracted_rows,
        header=header,
        provenance=provenance,
        logical_grid_indices=logical_grid_indices,
        recovered_header_row_count=recovered_header_row_count,
    )


_CURRENCY_SYMBOLS = frozenset({"$", "€", "£", "¥"})
_NUMERIC_TOKEN = re.compile(r"\(?[+-]?[\d,.]+%?\)?$")
_YEAR_TOKEN = re.compile(r"(?:19|20)\d{2}$")
_MONTH_TOKEN = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _ExternalHeader:
    top: float
    bottom: float
    date_labels: tuple[str, ...]
    date_tokens: tuple[tuple[NativeToken, ...], ...]
    spanning_text: str | None
    spanning_tokens: tuple[NativeToken, ...]
    body_prefix: tuple[str, ...] | None
    body_prefix_top: float | None


def _reconstruct_external_financial_header(
    table_bbox: BBox,
    finder_cells: tuple[BBox, ...],
    geometry_rows: tuple[tuple[BBox | None, ...], ...] | None,
    finder_rows: tuple[tuple[str | None, ...], ...],
    evidence: TableProvenance,
) -> (
    tuple[
        BBox,
        tuple[BBox, ...],
        tuple[LogicalCellInput, ...],
        tuple[tuple[str | None, ...], ...],
        FinderHeaderInput,
    ]
    | None
):
    """Absorb a rule-connected date/year tier and collapse financial helper columns."""
    if geometry_rows is None or not evidence.native_tokens or not evidence.native_rules:
        return None
    boundaries = _finder_column_boundaries(
        table_bbox, geometry_rows, finder_rows, collapse_empty_helpers=True
    )
    if boundaries is None or len(boundaries) < 4:
        return None
    external = _external_header_evidence(
        table_bbox, boundaries, evidence.native_tokens, evidence.native_rules
    )
    if external is None:
        return None

    body_rows = _collapsed_body_rows(boundaries, geometry_rows, finder_rows)
    if not body_rows:
        return None
    extracted: list[tuple[str | None, ...]] = []
    cells: list[BBox] = []
    logical_cells: list[LogicalCellInput] = []

    date_top = min(token.bbox[1] for group in external.date_tokens for token in group)
    if external.spanning_text is not None:
        split = date_top
        values: tuple[str | None, ...] = ("", external.spanning_text, *(None for _ in boundaries[2:-1]))
        extracted.append(values)
        top_cells = [
            (0, 0, (boundaries[0], external.top, boundaries[1], split), ""),
            (0, 1, (boundaries[1], external.top, boundaries[-1], split), external.spanning_text),
        ]
        for row_index, column_index, bbox, text in top_cells:
            cells.append(bbox)
            logical_cells.append(LogicalCellInput(row_index, column_index, bbox, text))

    date_row_index = len(extracted)
    date_values = ("", *external.date_labels)
    extracted.append(date_values)
    date_bottom = external.body_prefix_top or external.bottom
    for column_index in range(len(boundaries) - 1):
        bbox = (boundaries[column_index], date_top, boundaries[column_index + 1], date_bottom)
        text = date_values[column_index]
        cells.append(bbox)
        logical_cells.append(LogicalCellInput(date_row_index, column_index, bbox, text))

    if external.body_prefix is not None:
        prefix_row_index = len(extracted)
        extracted.append(external.body_prefix)
        for column_index, text in enumerate(external.body_prefix):
            bbox = (boundaries[column_index], date_bottom, boundaries[column_index + 1], external.bottom)
            cells.append(bbox)
            logical_cells.append(LogicalCellInput(prefix_row_index, column_index, bbox, text))

    for body_offset, row in enumerate(body_rows, start=len(extracted)):
        values, row_cells = row
        extracted.append(values)
        for column_index, column_end, bbox, text in row_cells:
            cells.append(bbox)
            logical_cells.append(LogicalCellInput(body_offset, column_index, bbox, text))
            if column_end > column_index + 1:
                # A single geometry cell represents the span; omitted positions are
                # encoded as None in the extracted grid.
                continue

    expanded_bbox = (table_bbox[0], external.top, table_bbox[2], table_bbox[3])
    return (
        expanded_bbox,
        tuple(cells),
        tuple(logical_cells),
        tuple(extracted),
        FinderHeaderInput(external=False, names=extracted[0]),
    )


def _finder_column_boundaries(
    table_bbox: BBox,
    geometry_rows: tuple[tuple[BBox | None, ...], ...],
    finder_rows: tuple[tuple[str | None, ...], ...],
    *,
    collapse_empty_helpers: bool,
) -> tuple[float, ...] | None:
    column_count = max((len(row) for row in finder_rows), default=0)
    if column_count < 3 or len(geometry_rows) != len(finder_rows):
        return None
    widths: list[list[float]] = [[] for _ in range(column_count)]
    right_edges: list[list[float]] = [[] for _ in range(column_count)]
    present = [0] * column_count
    nonempty = [0] * column_count
    for geometry_row, text_row in zip(geometry_rows, finder_rows, strict=True):
        if len(geometry_row) != len(text_row):
            return None
        for index, (bbox, value) in enumerate(zip(geometry_row, text_row, strict=True)):
            if bbox is None:
                continue
            present[index] += 1
            widths[index].append(bbox[2] - bbox[0])
            right_edges[index].append(bbox[2])
            nonempty[index] += bool(value and str(value).strip())

    table_width = table_bbox[2] - table_bbox[0]
    separators = [
        index
        for index in range(1, column_count - 1)
        if present[index] >= max(2, len(geometry_rows) // 2)
        and nonempty[index] == 0
        and widths[index]
        and median(widths[index]) <= table_width * 0.03
    ]
    stub_right = median(right_edges[0]) if right_edges[0] else 0.0
    if collapse_empty_helpers and separators:
        boundaries = (
            table_bbox[0],
            stub_right,
            *(median(right_edges[index]) for index in separators),
            table_bbox[2],
        )
    else:
        common = [
            row for row in geometry_rows if len(row) == column_count and all(cell is not None for cell in row)
        ]
        if not common:
            return None
        boundaries = (
            table_bbox[0],
            *(median(cast(BBox, row[index])[2] for row in common) for index in range(column_count - 1)),
            table_bbox[2],
        )
    minimum_width = table_width * 0.02
    if any(right - left < minimum_width for left, right in zip(boundaries, boundaries[1:], strict=False)):
        return None
    return boundaries


def _external_header_evidence(
    table_bbox: BBox,
    boundaries: tuple[float, ...],
    tokens: tuple[NativeToken, ...],
    rules: tuple[NativeRule, ...],
) -> _ExternalHeader | None:
    numeric_column_count = len(boundaries) - 2
    nearby = tuple(
        token
        for token in tokens
        if token.baseline < table_bbox[1]
        and boundaries[0] <= (token.bbox[0] + token.bbox[2]) / 2 <= boundaries[-1]
    )
    if not nearby:
        return None
    token_height = median(token.bbox[3] - token.bbox[1] for token in nearby)
    if token_height <= 0:
        return None

    date_groups: list[tuple[NativeToken, ...]] = []
    labels: list[str] = []
    for left, right in zip(boundaries[1:-1], boundaries[2:], strict=True):
        column_tokens = tuple(
            token for token in nearby if left <= (token.bbox[0] + token.bbox[2]) / 2 <= right
        )
        years = [token for token in column_tokens if _YEAR_TOKEN.fullmatch(token.text.strip()) is not None]
        if not years:
            return None
        year = max(years, key=lambda token: (token.baseline, token.bbox[0]))
        group = tuple(
            sorted(
                (
                    token
                    for token in column_tokens
                    if year.baseline - token_height * 2.0
                    <= token.baseline
                    <= year.baseline + token_height * 0.25
                ),
                key=lambda token: (token.baseline, token.bbox[0], token.token_id.value),
            )
        )
        label = " ".join(token.text for token in group)
        if not group or not _is_date_or_year_label(label):
            return None
        date_groups.append(group)
        labels.append(label)
    if len(date_groups) != numeric_column_count or numeric_column_count < 2:
        return None

    date_top = min(token.bbox[1] for group in date_groups for token in group)
    date_baseline = max(token.baseline for group in date_groups for token in group)
    if table_bbox[1] - date_baseline > token_height * 2.25:
        return None
    bottom_rule = _supporting_rule_after(
        rules,
        boundaries[1],
        boundaries[-1],
        date_baseline,
        date_baseline + token_height * 1.5,
        token_height,
    )
    if bottom_rule is None:
        return None

    date_ids = {token.token_id for group in date_groups for token in group}
    upper_candidates = tuple(
        token
        for token in nearby
        if token.token_id not in date_ids
        and token.bbox[3] <= date_top
        and date_top - token.bbox[3] <= token_height
        and boundaries[1] <= (token.bbox[0] + token.bbox[2]) / 2 <= boundaries[-1]
    )
    spanning_tokens: tuple[NativeToken, ...] = ()
    spanning_text: str | None = None
    if upper_candidates:
        nearest_baseline = max(token.baseline for token in upper_candidates)
        spanning_tokens = tuple(
            sorted(
                (
                    token
                    for token in upper_candidates
                    if abs(token.baseline - nearest_baseline) <= token_height * 0.3
                ),
                key=lambda token: (token.bbox[0], token.token_id.value),
            )
        )
        span_bbox = _tokens_bbox(spanning_tokens)
        numeric_center = (boundaries[1] + boundaries[-1]) / 2
        if (
            not spanning_tokens
            or abs((span_bbox[0] + span_bbox[2]) / 2 - numeric_center)
            > (boundaries[-1] - boundaries[1]) * 0.2
            or _supporting_rule_after(
                rules,
                boundaries[1],
                boundaries[-1],
                max(token.baseline for token in spanning_tokens),
                date_top + token_height * 0.5,
                token_height,
            )
            is None
        ):
            spanning_tokens = ()
        else:
            spanning_text = " ".join(token.text for token in spanning_tokens)

    prefix_tokens = tuple(
        token
        for token in nearby
        if token.token_id not in date_ids
        and token.token_id not in {item.token_id for item in spanning_tokens}
        and token.baseline > bottom_rule
        and token.baseline <= table_bbox[1] + token_height * 0.5
    )
    body_prefix = _external_body_prefix(boundaries, prefix_tokens)
    if (
        body_prefix is not None
        and _supporting_rule_after(
            rules,
            boundaries[0],
            boundaries[-1],
            bottom_rule,
            table_bbox[1] + token_height,
            token_height,
        )
        is None
    ):
        body_prefix = None
    top = min(
        token.bbox[1] for token in (*spanning_tokens, *(token for group in date_groups for token in group))
    )
    return _ExternalHeader(
        top=top,
        # Logical cells remain contiguous even when typography leaves breathing
        # room between the ruled header and the first body baseline.
        bottom=table_bbox[1],
        date_labels=tuple(labels),
        date_tokens=tuple(date_groups),
        spanning_text=spanning_text,
        spanning_tokens=spanning_tokens,
        body_prefix=body_prefix,
        body_prefix_top=bottom_rule if body_prefix is not None else None,
    )


def _external_body_prefix(
    boundaries: tuple[float, ...], tokens: tuple[NativeToken, ...]
) -> tuple[str, ...] | None:
    columns: list[list[NativeToken]] = [[] for _ in boundaries[:-1]]
    for token in tokens:
        center = (token.bbox[0] + token.bbox[2]) / 2
        column_index = _interval_index(boundaries, center)
        if column_index is not None:
            columns[column_index].append(token)
    numeric_columns = sum(
        any(_NUMERIC_TOKEN.fullmatch(token.text.strip()) for token in column) for column in columns[1:]
    )
    if not columns[0] or (numeric_columns < 2 and any(columns[1:])):
        return None
    return tuple(
        " ".join(token.text for token in sorted(column, key=lambda item: item.bbox[0])) for column in columns
    )


def _is_date_or_year_label(value: str) -> bool:
    words = value.replace(",", " ").split()
    return bool(
        any(_YEAR_TOKEN.fullmatch(word) for word in words)
        and (len(words) == 1 or any(_MONTH_TOKEN.fullmatch(word) for word in words))
    )


def _supporting_rule_after(
    rules: tuple[NativeRule, ...],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    token_height: float,
) -> float | None:
    tolerance = max(0.75, token_height * 0.15)
    levels: list[list[tuple[float, float]]] = []
    ys: list[float] = []
    for rule in sorted(rules, key=lambda item: (item.y, item.x0, item.x1)):
        if not y0 - tolerance <= rule.y <= y1:
            continue
        if not ys or abs(rule.y - ys[-1]) > tolerance:
            ys.append(rule.y)
            levels.append([])
        levels[-1].append((max(x0, rule.x0), min(x1, rule.x1)))
    required = (x1 - x0) * 0.75
    for y, intervals in zip(ys, levels, strict=True):
        merged: list[list[float]] = []
        for left, right in sorted(intervals):
            if right <= left:
                continue
            if merged and left <= merged[-1][1] + tolerance:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        if sum(right - left for left, right in merged) >= required:
            return y
    return None


def _collapsed_body_rows(
    boundaries: tuple[float, ...],
    geometry_rows: tuple[tuple[BBox | None, ...], ...],
    finder_rows: tuple[tuple[str | None, ...], ...],
) -> list[tuple[tuple[str | None, ...], list[tuple[int, int, BBox, str]]]]:
    result: list[tuple[tuple[str | None, ...], list[tuple[int, int, BBox, str]]]] = []
    for geometry_row, text_row in zip(geometry_rows, finder_rows, strict=True):
        if not any(value and str(value).strip() for value in text_row):
            continue
        present_cells = tuple(cell for cell in geometry_row if cell is not None)
        if not present_cells:
            continue
        row_top = min(cell[1] for cell in present_cells)
        row_bottom = max(cell[3] for cell in present_cells)
        assigned: dict[tuple[int, int], list[str]] = {}
        for bbox, value in zip(geometry_row, text_row, strict=True):
            if bbox is None:
                continue
            start = _nearest_boundary_index(boundaries, bbox[0])
            end = _nearest_boundary_index(boundaries, bbox[2])
            if end <= start:
                center = (bbox[0] + bbox[2]) / 2
                start = _interval_index(boundaries, center) or 0
                end = start + 1
            text = str(value or "").strip()
            if text:
                assigned.setdefault((start, end), []).append(text)
        if not assigned:
            continue
        if (0, len(boundaries) - 1) in assigned:
            text = " ".join(assigned[(0, len(boundaries) - 1)])
            values = (text, *(None for _ in boundaries[1:-1]))
            row_cells = [(0, len(boundaries) - 1, (boundaries[0], row_top, boundaries[-1], row_bottom), text)]
            result.append((values, row_cells))
            continue

        values_list: list[str | None] = []
        row_cells = []
        for column_index in range(len(boundaries) - 1):
            pieces = [
                text
                for (start, end), texts in sorted(assigned.items())
                if start <= column_index < end
                for text in texts
                if start == column_index
            ]
            text = " ".join(pieces)
            values_list.append(text)
            bbox = (boundaries[column_index], row_top, boundaries[column_index + 1], row_bottom)
            row_cells.append((column_index, column_index + 1, bbox, text))
        result.append((tuple(values_list), row_cells))
    return result


def _nearest_boundary_index(boundaries: tuple[float, ...], value: float) -> int:
    return min(range(len(boundaries)), key=lambda index: abs(boundaries[index] - value))


def _tokens_bbox(tokens: tuple[NativeToken, ...]) -> BBox:
    if not tokens:
        raise ValueError("cannot enclose an empty native token collection")
    return (
        min(token.bbox[0] for token in tokens),
        min(token.bbox[1] for token in tokens),
        max(token.bbox[2] for token in tokens),
        max(token.bbox[3] for token in tokens),
    )


def _reconstruct_rule_connected_external_header(
    table_bbox: BBox,
    finder_cells: tuple[BBox, ...],
    body_logical_cells: tuple[LogicalCellInput, ...],
    geometry_rows: tuple[tuple[BBox | None, ...], ...] | None,
    finder_rows: tuple[tuple[str | None, ...], ...],
    evidence: TableProvenance,
) -> (
    tuple[
        BBox,
        tuple[BBox, ...],
        tuple[LogicalCellInput, ...],
        tuple[tuple[str | None, ...], ...],
        FinderHeaderInput,
        int,
    ]
    | None
):
    """Recover a tiered header only when native rules connect it to a numeric grid."""
    if geometry_rows is None or not evidence.native_tokens or not evidence.native_rules:
        return None
    boundaries = _finder_column_boundaries(
        table_bbox, geometry_rows, finder_rows, collapse_empty_helpers=False
    )
    if boundaries is None or len(boundaries) < 5:
        return None
    column_count = len(boundaries) - 1
    if len(finder_rows) < 3 or sum(_numeric_row(value) for value in finder_rows) < 3:
        return None

    nearby = tuple(
        token
        for token in evidence.native_tokens
        if token.baseline < table_bbox[1]
        and boundaries[0] <= (token.bbox[0] + token.bbox[2]) / 2 <= boundaries[-1]
    )
    if not nearby:
        return None
    token_height = median(token.bbox[3] - token.bbox[1] for token in nearby)
    if token_height <= 0:
        return None

    levels = _rule_coverage_levels(
        evidence.native_rules,
        boundaries[0],
        boundaries[-1],
        table_bbox[1] - token_height * 10,
        table_bbox[1] + token_height * 0.5,
        token_height,
    )
    if not levels or abs(levels[-1][0] - table_bbox[1]) > token_height * 0.5:
        return None
    connected: list[tuple[float, float]] = [levels[-1]]
    for level in reversed(levels[:-1]):
        if connected[-1][0] - level[0] > token_height * 3:
            break
        connected.append(level)
    connected.reverse()
    if not 4 <= len(connected) <= 6 or connected[-1][1] < 0.8:
        return None
    if any(coverage < 0.3 for _, coverage in connected):
        return None

    first_bottom = connected[0][0]
    selected = tuple(
        token
        for token in nearby
        if first_bottom - token_height * 2 <= token.baseline <= connected[-1][0] + token_height * 0.25
    )
    if len(selected) < column_count * 2:
        return None
    header_top = min(token.bbox[1] for token in selected)
    y_boundaries = (header_top, *(level for level, _ in connected))
    if any(right <= left for left, right in zip(y_boundaries, y_boundaries[1:], strict=False)):
        return None

    values: list[tuple[str | None, ...]] = []
    header_cells: list[LogicalCellInput] = []
    used_ids: set[NativeTokenId] = set()
    dense_rows = 0
    for row_index, (top, bottom) in enumerate(zip(y_boundaries, y_boundaries[1:], strict=False)):
        columns: list[list[NativeToken]] = [[] for _ in range(column_count)]
        for token in selected:
            if not top <= token.baseline <= bottom + token_height * 0.15:
                continue
            column_index = _interval_index(boundaries, (token.bbox[0] + token.bbox[2]) / 2)
            if column_index is None or token.token_id in used_ids:
                return None
            columns[column_index].append(token)
            used_ids.add(token.token_id)
        populated = sum(bool(column) for column in columns)
        dense_rows += populated >= max(3, math.ceil(column_count * 0.4))
        row_values: list[str | None] = []
        for column_index, column in enumerate(columns):
            text = " ".join(
                token.text
                for token in sorted(
                    column,
                    key=lambda item: (item.baseline, item.bbox[0], item.token_id.value),
                )
            )
            row_values.append(text)
            bbox = (boundaries[column_index], top, boundaries[column_index + 1], bottom)
            header_cells.append(LogicalCellInput(row_index, column_index, bbox, text))
        values.append(tuple(row_values))
    if used_ids != {token.token_id for token in selected} or dense_rows < 3:
        return None

    header_row_count = len(values)
    shifted_body = tuple(
        LogicalCellInput(
            row_index=cell.row_index + header_row_count,
            column_index=cell.column_index,
            bbox=cell.bbox,
            text=cell.text,
            rowspan=cell.rowspan,
        )
        for cell in body_logical_cells
    )
    logical_cells = (*header_cells, *shifted_body)
    return (
        (table_bbox[0], header_top, table_bbox[2], table_bbox[3]),
        (*tuple(cell.bbox for cell in header_cells), *finder_cells),
        logical_cells,
        (*values, *finder_rows),
        FinderHeaderInput(external=False, names=values[0]),
        header_row_count,
    )


def _numeric_row(row: tuple[str | None, ...]) -> bool:
    return (
        sum(
            _NUMERIC_TOKEN.fullmatch(str(value).strip().replace("−", "-").replace("–", "-")) is not None
            for value in row
            if value and str(value).strip()
        )
        >= 3
    )


def _rule_coverage_levels(
    rules: tuple[NativeRule, ...],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    token_height: float,
) -> list[tuple[float, float]]:
    tolerance = max(0.75, token_height * 0.15)
    grouped: list[tuple[list[float], list[tuple[float, float]]]] = []
    for rule in sorted(rules, key=lambda item: (item.y, item.x0, item.x1)):
        if not y0 <= rule.y <= y1:
            continue
        interval = (max(x0, rule.x0), min(x1, rule.x1))
        if interval[1] <= interval[0]:
            continue
        if grouped and abs(rule.y - median(grouped[-1][0])) <= tolerance:
            grouped[-1][0].append(rule.y)
            grouped[-1][1].append(interval)
        else:
            grouped.append(([rule.y], [interval]))

    levels: list[tuple[float, float]] = []
    for ys, intervals in grouped:
        merged: list[list[float]] = []
        for left, right in sorted(intervals):
            if merged and left <= merged[-1][1] + tolerance:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        levels.append((median(ys), sum(right - left for left, right in merged) / (x1 - x0)))
    return levels


def _reconstruct_rule_partitioned_form_grid(
    table_bbox: BBox,
    geometry_rows: tuple[tuple[BBox | None, ...], ...] | None,
    finder_rows: tuple[tuple[str | None, ...], ...],
    evidence: TableProvenance,
) -> tuple[BBox, tuple[BBox, ...], tuple[LogicalCellInput, ...], tuple[tuple[str | None, ...], ...]] | None:
    """Split only oversized finder cells whose internal rows have independent rule evidence."""
    if (
        evidence.finder != FinderProvenance.DEFAULT
        or geometry_rows is None
        or len(geometry_rows) != len(finder_rows)
        or len(geometry_rows) < 8
        or not evidence.native_tokens
        or not evidence.native_rules
    ):
        return None

    retained_start = 0
    while retained_start < len(finder_rows) - 1:
        row = finder_rows[retained_start]
        geometry = geometry_rows[retained_start]
        if any(value and str(value).strip() for value in row):
            break
        row_cells = tuple(cell for cell in geometry if cell is not None)
        if any(
            any(
                cell[0] <= (token.bbox[0] + token.bbox[2]) / 2 <= cell[2]
                and cell[1] <= token.baseline <= cell[3]
                for cell in row_cells
            )
            for token in evidence.native_tokens
        ):
            break
        retained_start += 1
    if retained_start:
        geometry_rows = geometry_rows[retained_start:]
        finder_rows = finder_rows[retained_start:]

    row_bands = tuple(_enclosing_optional_row(row) for row in geometry_rows)
    if any(
        right[0] < left[0] or abs(left[1] - right[0]) > max(0.75, (table_bbox[3] - table_bbox[1]) * 0.002)
        for left, right in zip(row_bands, row_bands[1:], strict=False)
    ):
        return None
    y_boundaries = (row_bands[0][0], *(band[1] for band in row_bands))
    table_width = table_bbox[2] - table_bbox[0]
    tolerance = max(0.75, (table_bbox[3] - table_bbox[1]) * 0.002)

    unique_cells: dict[BBox, str] = {}
    for geometry_row, text_row in zip(geometry_rows, finder_rows, strict=True):
        for bbox, value in zip(geometry_row, text_row, strict=True):
            if bbox is not None:
                unique_cells.setdefault(bbox, str(value or "").strip())

    split_cells: set[BBox] = set()
    split_text: dict[tuple[BBox, int], str] = {}
    split_bboxes: dict[tuple[BBox, int], BBox] = {}
    for bbox in unique_cells:
        start = _near_boundary(y_boundaries, bbox[1], tolerance)
        end = _near_boundary(y_boundaries, bbox[3], tolerance)
        if start is None or end is None or end - start < 3 or bbox[2] - bbox[0] < table_width * 0.20:
            continue
        outside_width = table_width - (bbox[2] - bbox[0])
        if outside_width <= table_width * 0.10:
            continue
        internal_boundaries = y_boundaries[start + 1 : end]
        if not internal_boundaries or not all(
            _rule_coverage_outside_cell(
                level,
                _form_partition_bbox(
                    bbox,
                    tuple(unique_cells),
                    level - tolerance,
                    level + tolerance,
                ),
                table_bbox,
                evidence.native_rules,
                tolerance,
            )
            >= 0.20
            for level in internal_boundaries
        ):
            continue
        occupied_rows = 0
        for row_index in range(start, end):
            row_bbox = _form_partition_bbox(
                bbox,
                tuple(unique_cells),
                y_boundaries[row_index],
                y_boundaries[row_index + 1],
            )
            tokens = _tokens_in_partition(
                evidence.native_tokens,
                row_bbox[0],
                row_bbox[2],
                row_bbox[1],
                row_bbox[3],
            )
            text = _form_partition_text(tokens)
            split_bboxes[(bbox, row_index)] = row_bbox
            split_text[(bbox, row_index)] = text
            occupied_rows += bool(text)
        if occupied_rows < 3:
            continue
        split_cells.add(bbox)

    if not split_cells:
        return None

    rebuilt: list[tuple[int, BBox, str]] = []
    for bbox, text in unique_cells.items():
        start = _near_boundary(y_boundaries, bbox[1], tolerance)
        end = _near_boundary(y_boundaries, bbox[3], tolerance)
        if start is None or end is None or end <= start:
            return None
        if bbox in split_cells:
            rebuilt.extend(
                (
                    row_index,
                    split_bboxes[(bbox, row_index)],
                    split_text[(bbox, row_index)],
                )
                for row_index in range(start, end)
            )
        else:
            rebuilt.append((start, bbox, text))

    x_boundaries = tuple(sorted({value for _, bbox, _ in rebuilt for value in (bbox[0], bbox[2])}))
    if len(x_boundaries) < 3:
        return None
    logical_cells: list[LogicalCellInput] = []
    values: list[list[str | None]] = [[None for _ in x_boundaries[:-1]] for _ in row_bands]
    occupied: set[tuple[int, int]] = set()
    for row_index, bbox, text in sorted(rebuilt, key=lambda item: (item[0], item[1][0], item[1][2])):
        column_start = _near_boundary(x_boundaries, bbox[0], tolerance)
        column_end = _near_boundary(x_boundaries, bbox[2], tolerance)
        row_end = _near_boundary(y_boundaries, bbox[3], tolerance)
        if column_start is None or column_end is None or row_end is None:
            return None
        positions = {
            (row, column) for row in range(row_index, row_end) for column in range(column_start, column_end)
        }
        if occupied.intersection(positions):
            return None
        occupied.update(positions)
        logical_cells.append(
            LogicalCellInput(
                row_index=row_index,
                column_index=column_start,
                bbox=bbox,
                text=text,
                rowspan=row_end - row_index,
            )
        )
        values[row_index][column_start] = text
    return (
        table_bbox,
        tuple(cell.bbox for cell in logical_cells),
        tuple(logical_cells),
        tuple(tuple(row) for row in values),
    )


def _enclosing_optional_row(row: tuple[BBox | None, ...]) -> tuple[float, float]:
    cells = tuple(cell for cell in row if cell is not None)
    if not cells:
        raise ValueError("finder form row must contain geometry")
    top = min(cell[1] for cell in cells)
    bottoms = [cell[3] for cell in cells if cell[3] > top]
    return top, min(bottoms)


def _rule_coverage_outside_cell(
    y: float,
    cell: BBox,
    table_bbox: BBox,
    rules: tuple[NativeRule, ...],
    tolerance: float,
) -> float:
    regions = ((table_bbox[0], cell[0]), (cell[2], table_bbox[2]))
    available = sum(right - left for left, right in regions)
    intervals: list[tuple[float, float]] = []
    for rule in rules:
        if abs(rule.y - y) > tolerance:
            continue
        intervals.extend(
            (max(left, rule.x0), min(right, rule.x1))
            for left, right in regions
            if min(right, rule.x1) > max(left, rule.x0)
        )
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return sum(right - left for left, right in merged) / available if available > 0 else 0.0


def _form_partition_bbox(
    container: BBox,
    cells: tuple[BBox, ...],
    y0: float,
    y1: float,
) -> BBox:
    nested_starts = [
        cell[0]
        for cell in cells
        if cell != container
        and container[0] < cell[0] < container[2]
        and cell[2] <= container[2]
        and cell[1] < y1
        and y0 < cell[3]
    ]
    right = min(nested_starts, default=container[2])
    return container[0], y0, right, y1


def _tokens_in_partition(
    tokens: tuple[NativeToken, ...],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[NativeToken, ...]:
    return tuple(
        token
        for token in tokens
        if x0 <= (token.bbox[0] + token.bbox[2]) / 2 <= x1 and y0 <= token.baseline <= y1
    )


def _form_partition_text(tokens: tuple[NativeToken, ...]) -> str:
    if not tokens:
        return ""
    token_height = median(token.bbox[3] - token.bbox[1] for token in tokens)
    rows = _baseline_rows(tokens, token_height)
    ordered = [
        token for row in rows for token in sorted(row, key=lambda item: (item.bbox[0], item.token_id.value))
    ]
    return " ".join(token.text for token in ordered if token.text.strip() != ".")


def _reconstruct_rule_backed_baseline_grid(
    table_bbox: BBox,
    geometry_rows: tuple[tuple[BBox | None, ...], ...] | None,
    finder_rows: tuple[tuple[str | None, ...], ...],
    evidence: TableProvenance,
) -> tuple[tuple[BBox, ...], tuple[LogicalCellInput, ...], tuple[tuple[str | None, ...], ...]] | None:
    """Split finder rows that compressed a ruled numeric table across baselines."""
    if geometry_rows is None or not evidence.native_tokens or not evidence.native_rules:
        return None
    if not 2 <= len(finder_rows) <= 5 or len(geometry_rows) != len(finder_rows):
        return None

    tokens = tuple(
        token
        for token in evidence.native_tokens
        if table_bbox[0] <= (token.bbox[0] + token.bbox[2]) / 2 <= table_bbox[2]
        and table_bbox[1] <= token.baseline <= table_bbox[3]
    )
    if len(tokens) < 12 or len({token.bbox for token in tokens}) != len(tokens):
        return None
    token_height = median(token.bbox[3] - token.bbox[1] for token in tokens)
    if token_height <= 0:
        return None

    rule_levels = _rule_coverage_levels(
        evidence.native_rules,
        table_bbox[0],
        table_bbox[2],
        table_bbox[1] - token_height * 0.25,
        table_bbox[3] + token_height * 0.25,
        token_height,
    )
    if len([level for level in rule_levels if level[1] >= 0.8]) < 3:
        return None

    baseline_rows = _baseline_rows(tokens, token_height)
    if len(baseline_rows) < max(6, len(finder_rows) * 2):
        return None
    numeric_centers = sorted(
        (token.bbox[0] + token.bbox[2]) / 2
        for token in tokens
        if _NUMERIC_TOKEN.fullmatch(token.text.strip().replace("−", "-").replace("–", "-")) is not None
    )
    center_groups: list[list[float]] = []
    center_tolerance = token_height * 1.5
    for center in numeric_centers:
        if center_groups and center - median(center_groups[-1]) <= center_tolerance:
            center_groups[-1].append(center)
        else:
            center_groups.append([center])
    minimum_support = max(3, math.ceil((len(baseline_rows) - 1) * 0.5))
    anchors = tuple(median(group) for group in center_groups if len(group) >= minimum_support)
    if not 2 <= len(anchors) <= 8:
        return None

    first_anchor = anchors[0]
    qualifying_rows: list[int] = []
    stub_right_edges: list[float] = []
    for row_index, row in enumerate(baseline_rows):
        value_columns: set[int] = set()
        numeric_columns: set[int] = set()
        for token in row:
            center = (token.bbox[0] + token.bbox[2]) / 2
            nearest = min(range(len(anchors)), key=lambda index: abs(anchors[index] - center))
            if abs(anchors[nearest] - center) > center_tolerance:
                continue
            value_columns.add(nearest)
            if _NUMERIC_TOKEN.fullmatch(token.text.strip().replace("−", "-").replace("–", "-")) is not None:
                numeric_columns.add(nearest)
        stub = [
            token for token in row if (token.bbox[0] + token.bbox[2]) / 2 < first_anchor - center_tolerance
        ]
        if len(value_columns) == len(anchors) and len(numeric_columns) >= len(anchors) - 1 and stub:
            qualifying_rows.append(row_index)
            stub_right_edges.extend(token.bbox[2] for token in stub)
    if len(qualifying_rows) < minimum_support or not stub_right_edges:
        return None
    first_data_row = qualifying_rows[0]
    if qualifying_rows != list(range(first_data_row, len(baseline_rows))):
        return None
    if first_data_row < 1:
        return None
    first_row = 0 if first_data_row <= 2 else first_data_row - 1
    selected_rows = baseline_rows[first_row:]

    stub_right = max(stub_right_edges)
    if stub_right >= first_anchor:
        return None
    boundaries = (
        table_bbox[0],
        (stub_right + first_anchor) / 2,
        *((left + right) / 2 for left, right in zip(anchors, anchors[1:], strict=False)),
        table_bbox[2],
    )
    if any(right <= left for left, right in zip(boundaries, boundaries[1:], strict=False)):
        return None

    header_columns = {
        _interval_index(boundaries, (token.bbox[0] + token.bbox[2]) / 2) for token in selected_rows[0]
    }
    if header_columns != set(range(len(boundaries) - 1)):
        return None

    baselines = [median(token.baseline for token in row) for row in selected_rows]
    y_boundaries = [table_bbox[1]]
    y_boundaries.extend((left + right) / 2 for left, right in zip(baselines, baselines[1:], strict=False))
    y_boundaries.append(table_bbox[3])
    if any(right <= left for left, right in zip(y_boundaries, y_boundaries[1:], strict=False)):
        return None

    cells: list[BBox] = []
    logical_cells: list[LogicalCellInput] = []
    extracted: list[tuple[str | None, ...]] = []
    for row_index, row in enumerate(selected_rows):
        columns: list[list[NativeToken]] = [[] for _ in boundaries[:-1]]
        for token in row:
            column_index = _interval_index(boundaries, (token.bbox[0] + token.bbox[2]) / 2)
            if column_index is None:
                return None
            columns[column_index].append(token)
        values: list[str | None] = []
        for column_index, column in enumerate(columns):
            bbox = (
                boundaries[column_index],
                y_boundaries[row_index],
                boundaries[column_index + 1],
                y_boundaries[row_index + 1],
            )
            text = " ".join(
                token.text
                for token in sorted(
                    column, key=lambda item: (item.baseline, item.bbox[0], item.token_id.value)
                )
            )
            cells.append(bbox)
            logical_cells.append(LogicalCellInput(row_index, column_index, bbox, text))
            values.append(text)
        extracted.append(tuple(values))
    return tuple(cells), tuple(logical_cells), tuple(extracted)


def _reconstruct_ruled_inset_grid(
    table_bbox: BBox,
    geometry_rows: tuple[tuple[BBox | None, ...], ...] | None,
    finder_rows: tuple[tuple[str | None, ...], ...],
    evidence: TableProvenance,
) -> tuple[tuple[BBox, ...], tuple[LogicalCellInput, ...], tuple[tuple[str | None, ...], ...]] | None:
    """Collapse repeated narrow inset tracks using rule-backed logical rows.

    Some PDFs draw both a ruled cell rectangle and an inset text rectangle. A
    finder can expose the left inset, text box, and right inset as three physical
    columns and each text line as a physical row. Collapse only when every column
    group has that same narrow-wide-narrow topology and native rules independently
    prove the logical row boundaries.
    """
    if geometry_rows is None or not evidence.native_rules or len(geometry_rows) != len(finder_rows):
        return None
    column_count = max((len(row) for row in finder_rows), default=0)
    if (
        column_count < 6
        or column_count % 3
        or any(
            len(geometry_row) != column_count or len(text_row) != column_count
            for geometry_row, text_row in zip(geometry_rows, finder_rows, strict=True)
        )
    ):
        return None

    densest = max(geometry_rows, key=lambda row: sum(cell is not None for cell in row))
    if any(cell is None for cell in densest):
        return None
    physical_boundaries = tuple([cast(BBox, densest[0])[0], *(cast(BBox, cell)[2] for cell in densest)])
    if any(right <= left for left, right in zip(physical_boundaries, physical_boundaries[1:], strict=False)):
        return None
    table_width = table_bbox[2] - table_bbox[0]
    track_widths = tuple(
        right - left for left, right in zip(physical_boundaries, physical_boundaries[1:], strict=False)
    )
    helper_limit = table_width * 0.015
    for index in range(0, column_count, 3):
        left_helper, content, right_helper = track_widths[index : index + 3]
        if (
            left_helper > helper_limit
            or right_helper > helper_limit
            or content < table_width * 0.05
            or content < max(left_helper, right_helper) * 8
        ):
            return None
    x_boundaries = tuple(physical_boundaries[index] for index in range(0, column_count + 1, 3))
    y_boundaries = _rule_backed_row_boundaries(table_bbox, x_boundaries, evidence.native_rules)
    if y_boundaries is None:
        return None

    row_count = len(y_boundaries) - 1
    logical_column_count = len(x_boundaries) - 1
    x_tolerance = max(0.75, table_width * 0.002)
    table_height = table_bbox[3] - table_bbox[1]
    y_tolerance = max(0.75, table_height * 0.004)
    records = [
        (bbox, str(value).strip())
        for geometry_row, text_row in zip(geometry_rows, finder_rows, strict=True)
        for bbox, value in zip(geometry_row, text_row, strict=True)
        if bbox is not None and value is not None and str(value).strip()
    ]

    direct: list[tuple[int, int, int, int, BBox, str]] = []
    direct_bboxes: set[BBox] = set()
    occupied: set[tuple[int, int]] = set()
    for bbox, text in sorted(records, key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2])):
        column_start = _near_boundary(x_boundaries, bbox[0], x_tolerance)
        column_end = _near_boundary(x_boundaries, bbox[2], x_tolerance)
        row_start = _near_boundary(y_boundaries, bbox[1], y_tolerance)
        row_end = _near_boundary(y_boundaries, bbox[3], y_tolerance)
        if (
            column_start is None
            or column_end is None
            or row_start is None
            or row_end is None
            or column_end <= column_start
            or row_end <= row_start
        ):
            continue
        positions = {
            (row, column) for row in range(row_start, row_end) for column in range(column_start, column_end)
        }
        if occupied.intersection(positions):
            continue
        direct.append((row_start, column_start, row_end, column_end, bbox, text))
        direct_bboxes.add(bbox)
        occupied.update(positions)

    logical_cells: list[LogicalCellInput] = []
    values: list[list[str | None]] = [["" for _ in range(logical_column_count)] for _ in range(row_count)]
    for row_start, column_start, row_end, column_end, bbox, text in direct:
        logical_cells.append(
            LogicalCellInput(
                row_index=row_start,
                column_index=column_start,
                bbox=(
                    x_boundaries[column_start],
                    y_boundaries[row_start],
                    x_boundaries[column_end],
                    y_boundaries[row_end],
                ),
                text=text,
                rowspan=row_end - row_start,
            )
        )
        values[row_start][column_start] = text
        for row in range(row_start, row_end):
            for column in range(column_start, column_end):
                if (row, column) != (row_start, column_start):
                    values[row][column] = None

    for row in range(row_count):
        if any((row, column) in occupied for column in range(logical_column_count)):
            continue
        spanning = [
            (bbox, text)
            for bbox, text in records
            if bbox not in direct_bboxes
            and bbox[2] - bbox[0] >= table_width * 0.75
            and y_boundaries[row] <= (bbox[1] + bbox[3]) / 2 <= y_boundaries[row + 1]
        ]
        if len(spanning) == 1:
            bbox, text = spanning[0]
            logical_cells.append(
                LogicalCellInput(
                    row_index=row,
                    column_index=0,
                    bbox=(x_boundaries[0], y_boundaries[row], x_boundaries[-1], y_boundaries[row + 1]),
                    text=text,
                )
            )
            values[row] = [text, *(None for _ in range(logical_column_count - 1))]
            occupied.update((row, column) for column in range(logical_column_count))

    for row in range(row_count):
        for column in range(logical_column_count):
            if (row, column) in occupied:
                continue
            pieces = [
                (bbox, text)
                for bbox, text in records
                if bbox not in direct_bboxes
                and y_boundaries[row] <= (bbox[1] + bbox[3]) / 2 <= y_boundaries[row + 1]
                and x_boundaries[column] <= (bbox[0] + bbox[2]) / 2 <= x_boundaries[column + 1]
            ]
            ordered_text: list[str] = []
            for _, text in sorted(pieces, key=lambda item: (item[0][1], item[0][0], item[0][3])):
                if not ordered_text or text != ordered_text[-1]:
                    ordered_text.append(text)
            text = " ".join(ordered_text)
            bbox = (
                x_boundaries[column],
                y_boundaries[row],
                x_boundaries[column + 1],
                y_boundaries[row + 1],
            )
            logical_cells.append(LogicalCellInput(row, column, bbox, text))
            values[row][column] = text
            occupied.add((row, column))

    logical_cells.sort(key=lambda cell: (cell.row_index, cell.column_index))
    if not logical_cells or not any(value and value.strip() for value in values[0]):
        return None
    return (
        tuple(cell.bbox for cell in logical_cells),
        tuple(logical_cells),
        tuple(tuple(row) for row in values),
    )


def _rule_backed_row_boundaries(
    table_bbox: BBox,
    column_boundaries: tuple[float, ...],
    rules: tuple[NativeRule, ...],
) -> tuple[float, ...] | None:
    table_width = table_bbox[2] - table_bbox[0]
    table_height = table_bbox[3] - table_bbox[1]
    tolerance = max(0.75, table_height * 0.003)
    x_tolerance = max(0.75, table_width * 0.002)
    levels: list[tuple[list[float], list[tuple[float, float]]]] = []
    for rule in sorted(rules, key=lambda item: (item.y, item.x0, item.x1)):
        if not table_bbox[1] - tolerance <= rule.y <= table_bbox[3] + tolerance:
            continue
        left = max(table_bbox[0], rule.x0)
        right = min(table_bbox[2], rule.x1)
        if (
            right <= left
            or _near_boundary(column_boundaries, left, x_tolerance) is None
            or _near_boundary(column_boundaries, right, x_tolerance) is None
        ):
            continue
        if levels and abs(rule.y - median(levels[-1][0])) <= tolerance:
            levels[-1][0].append(rule.y)
            levels[-1][1].append((left, right))
        else:
            levels.append(([rule.y], [(left, right)]))

    boundaries: list[float] = []
    for ys, intervals in levels:
        merged: list[list[float]] = []
        for left, right in sorted(intervals):
            if merged and left <= merged[-1][1] + tolerance:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        coverage = sum(right - left for left, right in merged) / table_width
        if coverage >= 0.80:
            boundaries.append(median(ys))
    if (
        len(boundaries) < 3
        or abs(boundaries[0] - table_bbox[1]) > tolerance
        or abs(boundaries[-1] - table_bbox[3]) > tolerance
    ):
        return None
    return tuple(boundaries)


def _near_boundary(boundaries: tuple[float, ...], value: float, tolerance: float) -> int | None:
    index = min(range(len(boundaries)), key=lambda candidate: abs(boundaries[candidate] - value))
    return index if abs(boundaries[index] - value) <= tolerance else None


def _reconstruct_borderless_text(
    table_bbox: BBox,
    finder_cells: tuple[BBox, ...],
    geometry_rows: tuple[tuple[BBox | None, ...], ...] | None,
    finder_rows: tuple[tuple[str | None, ...], ...],
    evidence: TableProvenance,
) -> tuple[BBox, tuple[BBox, ...], tuple[LogicalCellInput, ...], tuple[tuple[str | None, ...], ...]] | None:
    """Rebuild a strongly gated text table, otherwise retain finder_grid_v1 unchanged."""
    if not evidence.native_tokens or geometry_rows is None:
        return None
    column_count = max((len(row) for row in finder_rows), default=0)
    if column_count < 2 or column_count > 20 or any(len(row) != column_count for row in geometry_rows):
        return None
    x_boundaries = _logical_column_boundaries(table_bbox, geometry_rows, column_count, evidence.native_tokens)
    if x_boundaries is None:
        return None
    column_count = len(x_boundaries) - 1

    tokens = tuple(
        token
        for token in evidence.native_tokens
        if x_boundaries[0] <= (token.bbox[0] + token.bbox[2]) / 2 <= x_boundaries[-1]
        and table_bbox[1] <= token.baseline <= table_bbox[3] + (token.bbox[3] - token.bbox[1]) * 0.25
    )
    if len(tokens) < 4 or len({token.bbox for token in tokens}) != len(tokens):
        return None
    token_height = median(token.bbox[3] - token.bbox[1] for token in tokens)
    if (
        token_height <= 0
        or sum(token.bbox[3] - token.bbox[1] > token_height * 3 for token in tokens) > len(tokens) * 0.2
    ):
        return None

    top_boundary = _rule_supported_top_boundary(table_bbox, tokens, evidence.native_rules)
    selected_tokens = tuple(token for token in tokens if token.baseline > top_boundary)
    if len(selected_tokens) < 4:
        return None
    baseline_rows = _baseline_rows(selected_tokens, token_height)
    baseline_rows, reconstructed_bottom = _trim_isolated_trailing_text_row(
        baseline_rows,
        table_bbox,
        token_height,
    )
    baseline_rows = _merge_wrapped_sparse_rows(baseline_rows, x_boundaries, token_height)
    if len(baseline_rows) < 3 or len(baseline_rows) > len(geometry_rows):
        return None

    row_assignments: list[list[list[NativeToken]]] = []
    used_ids: set[NativeTokenId] = set()
    for row in baseline_rows:
        columns: list[list[NativeToken]] = [[] for _ in range(column_count)]
        associated_tokens, original_ids = _associate_currency_tokens(row, token_height)
        if used_ids.intersection(original_ids):
            return None
        for token in associated_tokens:
            center = (token.bbox[0] + token.bbox[2]) / 2
            column_index = _interval_index(x_boundaries, center)
            if column_index is None:
                return None
            columns[column_index].append(token)
        used_ids.update(original_ids)
        row_assignments.append(columns)
    retained_token_ids = {token.token_id for row in baseline_rows for token in row}
    if used_ids != retained_token_ids:
        return None
    populated_rows = sum(sum(bool(column) for column in columns) >= 2 for columns in row_assignments)
    if populated_rows / len(row_assignments) < 0.5:
        return None

    baselines = [median(token.baseline for token in row) for row in baseline_rows]
    y_boundaries = [top_boundary]
    y_boundaries.extend((left + right) / 2 for left, right in zip(baselines, baselines[1:], strict=False))
    y_boundaries.append(
        reconstructed_bottom
        if reconstructed_bottom is not None
        else max(table_bbox[3], baselines[-1] + token_height * 0.25)
    )
    if any(right <= left for left, right in zip(y_boundaries, y_boundaries[1:], strict=False)):
        return None

    reconstructed_cells: list[BBox] = []
    logical_cells: list[LogicalCellInput] = []
    extracted: list[tuple[str | None, ...]] = []
    for row_index, (row, columns) in enumerate(zip(baseline_rows, row_assignments, strict=True)):
        values: list[str | None] = [""] * column_count
        spans = _rule_supported_spans(
            row,
            columns,
            x_boundaries,
            y_boundaries[row_index],
            y_boundaries[row_index + 1],
            evidence.native_rules,
        )
        if row_index > 0:
            spans = {start: end for start, end in spans.items() if not (start == 0 and end == column_count)}
        consumed: set[int] = set()
        for column_index in range(column_count):
            if column_index in consumed:
                values[column_index] = None
                continue
            span_end = spans.get(column_index, column_index + 1)
            span_tokens = [
                token for span_column in range(column_index, span_end) for token in columns[span_column]
            ]
            bbox = (
                x_boundaries[column_index],
                y_boundaries[row_index],
                x_boundaries[span_end],
                y_boundaries[row_index + 1],
            )
            text = " ".join(
                token.text
                for token in sorted(
                    span_tokens,
                    key=lambda item: (item.baseline, item.bbox[0], item.token_id.value),
                )
            )
            reconstructed_cells.append(bbox)
            logical_cells.append(
                LogicalCellInput(
                    row_index=row_index,
                    column_index=column_index,
                    bbox=bbox,
                    text=text,
                )
            )
            values[column_index] = text
            consumed.update(range(column_index + 1, span_end))
        extracted.append(tuple(values))
    reconstructed_bbox = (table_bbox[0], top_boundary, table_bbox[2], y_boundaries[-1])
    return reconstructed_bbox, tuple(reconstructed_cells), tuple(logical_cells), tuple(extracted)


def _logical_column_boundaries(
    table_bbox: BBox,
    geometry_rows: tuple[tuple[BBox | None, ...], ...],
    column_count: int,
    native_tokens: tuple[NativeToken, ...],
) -> tuple[float, ...] | None:
    boundary_candidates: list[list[float]] = [[] for _ in range(column_count + 1)]
    boundary_candidates[0].append(table_bbox[0])
    boundary_candidates[-1].append(table_bbox[2])
    for row in geometry_rows:
        for index, cell in enumerate(row):
            if cell is None:
                continue
            boundary_candidates[index].append(cell[0])
            boundary_candidates[index + 1].append(cell[2])
    if any(not candidates for candidates in boundary_candidates):
        return None
    raw_boundaries = tuple(median(candidates) for candidates in boundary_candidates)
    numeric_centers = [
        [
            (token.bbox[0] + token.bbox[2]) / 2
            for token in native_tokens
            if _NUMERIC_TOKEN.fullmatch(token.text.strip()) is not None
            and raw_boundaries[index] <= (token.bbox[0] + token.bbox[2]) / 2 < raw_boundaries[index + 1]
        ]
        for index in range(1, column_count)
    ]
    support_threshold = max(3, math.ceil(max((len(values) for values in numeric_centers), default=0) * 0.25))
    supported_indices = tuple(
        index for index, values in enumerate(numeric_centers, start=1) if len(values) >= support_threshold
    )
    if len(supported_indices) >= 2:
        numeric_anchors = tuple(median(numeric_centers[index - 1]) for index in supported_indices)
        interior = tuple(
            (left + right) / 2 for left, right in zip(numeric_anchors, numeric_anchors[1:], strict=False)
        )
        boundaries = (
            raw_boundaries[0],
            raw_boundaries[supported_indices[0]],
            *interior,
            raw_boundaries[-1],
        )
    else:
        boundaries = raw_boundaries
    minimum_width = (table_bbox[2] - table_bbox[0]) * 0.005
    if any(right - left <= minimum_width for left, right in zip(boundaries, boundaries[1:], strict=False)):
        return None
    return boundaries


def _rule_supported_top_boundary(
    table_bbox: BBox,
    tokens: tuple[NativeToken, ...],
    rules: tuple[NativeRule, ...],
) -> float:
    width = table_bbox[2] - table_bbox[0]
    height = table_bbox[3] - table_bbox[1]
    candidates: list[float] = []
    token_height = median(token.bbox[3] - token.bbox[1] for token in tokens)
    for rule in rules:
        overlap = max(0.0, min(rule.x1, table_bbox[2]) - max(rule.x0, table_bbox[0]))
        if (
            overlap < width * 0.8
            or not table_bbox[1] <= rule.y <= table_bbox[1] + height * 0.20
            or rule.y - table_bbox[1] < token_height * 2
        ):
            continue
        above = [token for token in tokens if token.baseline <= rule.y]
        below = [token for token in tokens if token.baseline > rule.y]
        if above and below and len(above) <= max(12, math.ceil(len(tokens) * 0.08)):
            candidates.append(rule.y)
    return min(candidates, default=table_bbox[1])


def _baseline_rows(tokens: tuple[NativeToken, ...], token_height: float) -> list[list[NativeToken]]:
    tolerance = max(0.5, token_height * 0.2)
    rows: list[list[NativeToken]] = []
    for token in sorted(tokens, key=lambda item: (item.baseline, item.bbox[0], item.token_id.value)):
        if not rows or abs(token.baseline - median(item.baseline for item in rows[-1])) > tolerance:
            rows.append([token])
        else:
            rows[-1].append(token)
    return rows


def _trim_isolated_trailing_text_row(
    rows: list[list[NativeToken]],
    table_bbox: BBox,
    token_height: float,
) -> tuple[list[list[NativeToken]], float | None]:
    """Exclude a geometrically isolated prose line appended below a numeric grid."""
    if len(rows) < 4:
        return rows, None
    baselines = [median(token.baseline for token in row) for row in rows]
    ordinary_gaps = [right - left for left, right in zip(baselines[:-2], baselines[1:-1], strict=False)]
    if not ordinary_gaps or baselines[-1] - baselines[-2] < median(ordinary_gaps) * 2:
        return rows, None
    trailing = rows[-1]
    if any(_NUMERIC_TOKEN.fullmatch(token.text.strip()) for token in trailing):
        return rows, None
    trailing_bbox = _tokens_bbox(tuple(trailing))
    table_width = table_bbox[2] - table_bbox[0]
    table_center = (table_bbox[0] + table_bbox[2]) / 2
    trailing_center = (trailing_bbox[0] + trailing_bbox[2]) / 2
    if (
        trailing_bbox[2] - trailing_bbox[0] < table_width * 0.40
        or abs(trailing_center - table_center) > table_width * 0.15
    ):
        return rows, None
    bottom = max(token.bbox[3] for token in rows[-2])
    if bottom < baselines[-2] or bottom - baselines[-2] > token_height:
        return rows, None
    return rows[:-1], bottom


def _merge_wrapped_sparse_rows(
    rows: list[list[NativeToken]],
    boundaries: tuple[float, ...],
    token_height: float,
) -> list[list[NativeToken]]:
    """Join a sparse upper line into repeated columns on the immediately following line."""
    merged: list[list[NativeToken]] = []
    index = 0
    while index < len(rows):
        if index + 1 >= len(rows):
            merged.append(rows[index])
            break
        upper = rows[index]
        lower = rows[index + 1]
        upper_columns = {
            column
            for token in upper
            if (column := _interval_index(boundaries, (token.bbox[0] + token.bbox[2]) / 2)) is not None
        }
        lower_columns = {
            column
            for token in lower
            if (column := _interval_index(boundaries, (token.bbox[0] + token.bbox[2]) / 2)) is not None
        }
        baseline_gap = median(token.baseline for token in lower) - median(token.baseline for token in upper)
        upper_left = min(token.bbox[0] for token in upper)
        lower_left = min(token.bbox[0] for token in lower)
        wrapped_single_column = (
            len(upper) >= 6
            and len(lower) <= 3
            and _interval_index(boundaries, (upper[0].bbox[0] + upper[0].bbox[2]) / 2) == 0
            and lower_columns == {0}
            and lower_left - upper_left >= token_height * 0.75
        )
        repeated_sparse_columns = 2 <= len(upper_columns) <= max(
            2, (len(boundaries) - 1) // 2
        ) and upper_columns.issubset(lower_columns)
        if (
            (wrapped_single_column or repeated_sparse_columns)
            and 0 < baseline_gap <= token_height * 1.75
            and not any(_NUMERIC_TOKEN.fullmatch(token.text.strip()) for token in upper)
        ):
            merged.append([*upper, *lower])
            index += 2
        else:
            merged.append(upper)
            index += 1
    return merged


def _associate_currency_tokens(
    row: list[NativeToken], token_height: float
) -> tuple[list[NativeToken], set[NativeTokenId]]:
    ordered = sorted(row, key=lambda token: (token.bbox[0], token.token_id.value))
    consumed: set[NativeTokenId] = set()
    associated: list[NativeToken] = []
    for token in ordered:
        if token.token_id in consumed:
            continue
        if token.text.strip() in _CURRENCY_SYMBOLS:
            amounts = [
                candidate
                for candidate in ordered
                if candidate.bbox[0] >= token.bbox[2]
                and candidate.token_id not in consumed
                and _NUMERIC_TOKEN.fullmatch(candidate.text.strip()) is not None
                and candidate.bbox[0] - token.bbox[2] <= token_height * 3
            ]
            if amounts:
                amount = min(
                    amounts, key=lambda candidate: (candidate.bbox[0] - token.bbox[2], candidate.bbox[0])
                )
                consumed.add(amount.token_id)
                associated.append(
                    NativeToken(
                        token_id=token.token_id,
                        frame=token.frame,
                        bbox=(
                            amount.bbox[0],
                            min(token.bbox[1], amount.bbox[1]),
                            amount.bbox[2],
                            max(token.bbox[3], amount.bbox[3]),
                        ),
                        baseline=(token.baseline + amount.baseline) / 2,
                        text=f"{token.text.strip()} {amount.text.strip()}",
                    )
                )
                consumed.add(token.token_id)
                continue
        associated.append(token)
        consumed.add(token.token_id)

    phrases: list[NativeToken] = []
    for token in sorted(associated, key=lambda item: item.bbox[0]):
        previous = phrases[-1] if phrases else None
        if (
            previous is not None
            and abs(token.baseline - previous.baseline) <= token_height * 0.2
            and token.bbox[0] - previous.bbox[2] <= token_height * 0.6
            and _NUMERIC_TOKEN.fullmatch(previous.text.strip()) is None
            and _NUMERIC_TOKEN.fullmatch(token.text.strip()) is None
            and not any(symbol in previous.text or symbol in token.text for symbol in _CURRENCY_SYMBOLS)
        ):
            phrases[-1] = NativeToken(
                token_id=previous.token_id,
                frame=previous.frame,
                bbox=(
                    previous.bbox[0],
                    min(previous.bbox[1], token.bbox[1]),
                    token.bbox[2],
                    max(previous.bbox[3], token.bbox[3]),
                ),
                baseline=(previous.baseline + token.baseline) / 2,
                text=f"{previous.text} {token.text}",
            )
        else:
            phrases.append(token)
    return phrases, consumed


def _interval_index(boundaries: tuple[float, ...], center: float) -> int | None:
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        if left <= center < right or (index == len(boundaries) - 2 and center == right):
            return index
    return None


def _rule_supported_spans(
    row: list[NativeToken],
    columns: list[list[NativeToken]],
    x_boundaries: tuple[float, ...],
    row_top: float,
    row_bottom: float,
    rules: tuple[NativeRule, ...],
) -> dict[int, int]:
    if not row:
        return {}
    baseline = median(token.baseline for token in row)
    column_widths = [right - left for left, right in zip(x_boundaries, x_boundaries[1:], strict=False)]
    tolerance = max(
        (row_bottom - row_top) * 0.35,
        median(column_widths) * 0.4,
    )
    spans: dict[int, int] = {}
    for rule in rules:
        if not baseline <= rule.y <= row_bottom + tolerance:
            continue
        start = min(range(len(x_boundaries)), key=lambda index: abs(x_boundaries[index] - rule.x0))
        end = min(range(len(x_boundaries)), key=lambda index: abs(x_boundaries[index] - rule.x1))
        if end - start <= 1:
            continue
        if abs(x_boundaries[start] - rule.x0) > tolerance or abs(x_boundaries[end] - rule.x1) > tolerance:
            continue
        occupied = [index for index in range(start, end) if columns[index]]
        if len(occupied) != 1:
            continue
        if start in spans:
            continue
        spans[start] = end
    return spans


def _geometry_rows(snapshot: FinderSnapshot) -> tuple[tuple[BBox | None, ...], ...] | None:
    candidate = getattr(snapshot, "rows", None)
    if candidate is None:
        return None
    rows = cast(Sequence[object], candidate)
    if not all(hasattr(row, "cells") for row in rows):
        raise ValueError("finder grid rows must expose cells")
    geometry_rows = cast(Sequence[_FinderRow], rows)
    return tuple(
        tuple(None if cell is None else _bbox(cell, "finder grid row cell") for cell in row.cells)
        for row in geometry_rows
    )


def _logical_cells_from_rows(
    cells: tuple[BBox, ...],
    geometry_rows: tuple[tuple[BBox | None, ...], ...],
    extracted_rows: tuple[tuple[str | None, ...], ...],
) -> tuple[LogicalCellInput, ...]:
    if len(geometry_rows) != len(extracted_rows):
        raise ValueError(
            "finder grid geometry row count does not match extracted row count: "
            f"{len(geometry_rows)} != {len(extracted_rows)}"
        )
    known_cells = set(cells)
    referenced_cells: set[BBox] = set()
    logical_cells: list[LogicalCellInput] = []
    for row_index, (geometry_row, extracted_row) in enumerate(
        zip(geometry_rows, extracted_rows, strict=True)
    ):
        if len(geometry_row) != len(extracted_row):
            raise ValueError(
                f"finder grid row {row_index} geometry width does not match extracted width: "
                f"{len(geometry_row)} != {len(extracted_row)}"
            )
        for column_index, (bbox, value) in enumerate(zip(geometry_row, extracted_row, strict=True)):
            if bbox is None:
                continue
            if bbox not in known_cells:
                raise ValueError(f"finder grid row {row_index} references unknown cell bbox: {bbox}")
            referenced_cells.add(bbox)
            logical_cells.append(
                LogicalCellInput(
                    row_index=row_index,
                    column_index=column_index,
                    bbox=bbox,
                    text=value or "",
                )
            )
    dangling_cells = known_cells - referenced_cells
    if dangling_cells:
        raise ValueError(
            f"finder grid contains cells not referenced by geometry rows: {sorted(dangling_cells)}"
        )
    return tuple(logical_cells)


def _logical_cells_from_flat_grid(
    cells: tuple[BBox, ...],
    extracted_rows: tuple[tuple[str | None, ...], ...],
) -> tuple[LogicalCellInput, ...]:
    x_starts = sorted({cell[0] for cell in cells})
    y_starts = sorted({cell[1] for cell in cells})
    return tuple(
        LogicalCellInput(
            row_index=row_index,
            column_index=column_index,
            bbox=bbox,
            text=extracted_rows[row_index][column_index] or "",
        )
        for bbox in cells
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]
        for row_index in [y_starts.index(bbox[1])]
        for column_index in [x_starts.index(bbox[0])]
        if row_index < len(extracted_rows) and column_index < len(extracted_rows[row_index])
    )


def _bbox(values: Sequence[float], label: str) -> BBox:
    bbox = _coordinates(values, label)
    _validate_bbox(bbox, label)
    return bbox


def _coordinates(values: Sequence[float], label: str) -> BBox:
    if len(values) != 4:
        raise ValueError(f"{label} bbox must contain four coordinates: {values}")
    bbox = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{label} bbox must contain four finite coordinates: {bbox}")
    return cast(BBox, bbox)


def _validate_bbox(bbox: tuple[float, float, float, float], label: str) -> None:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"{label} bbox must contain four finite coordinates: {bbox}")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"{label} bbox must have positive dimensions: {bbox}")


def _require_unique(values: tuple[object, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
