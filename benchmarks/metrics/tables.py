"""Table product metrics over lightweight cell mappings."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from benchmarks.metrics import Metric

_NUMBER_PATTERN = re.compile(r"(?<![\w])(?:\([+\-]?\d[\d\s,.'’]*\)|[+\-]?\d[\d\s,.'’]*)(?:[%‰])?(?![\w])")
_METRIC_NAMES = (
    "numeric_recall",
    "exact_numeric_cell_precision",
    "exact_numeric_cell_recall",
    "exact_numeric_cell_f1",
    "span_structure_precision",
    "span_structure_recall",
    "span_structure_f1",
    "row_exact",
    "column_exact",
)


@dataclass(frozen=True, slots=True)
class _Cell:
    row: int
    column: int
    text: str
    rowspan: int
    colspan: int


def normalize_unicode(value: str) -> str:
    """Normalize compatibility forms, Unicode digits, spacing, and case."""

    normalized = unicodedata.normalize("NFKC", value)
    characters: list[str] = []
    for character in normalized:
        if unicodedata.category(character) == "Nd":
            characters.append(str(unicodedata.decimal(character)))
        elif character in {"−", "‐", "‑", "‒", "–", "—"}:
            characters.append("-")
        elif character == "٫":
            characters.append(".")
        elif character == "٬":
            characters.append(",")
        else:
            characters.append(character)
    return " ".join("".join(characters).casefold().split())


def normalize_number(value: str) -> str:
    """Return a canonical decimal representation for one numeric token."""

    normalized = normalize_unicode(value).strip()
    negative_parentheses = normalized.startswith("(") and normalized.endswith(")")
    if negative_parentheses:
        normalized = normalized[1:-1].strip()
    suffix = ""
    if normalized.endswith(("%", "‰")):
        suffix, normalized = normalized[-1], normalized[:-1].strip()
    normalized = normalized.replace("’", "").replace("'", "").replace(" ", "")

    if "," in normalized and "." in normalized:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif "," in normalized:
        parts = normalized.split(",")
        normalized = "".join(parts) if all(len(part) == 3 for part in parts[1:]) else ".".join(parts)
    elif normalized.count(".") > 1:
        parts = normalized.split(".")
        if all(len(part) == 3 for part in parts[1:]):
            normalized = "".join(parts)

    if negative_parentheses:
        normalized = f"-{normalized.lstrip('+')}"
    try:
        number = Decimal(normalized)
    except InvalidOperation as error:
        raise ValueError(f"not a numeric token: {value!r}") from error
    if not number.is_finite():
        raise ValueError("numeric tokens must be finite")
    if number == 0:
        number = abs(number)
    canonical = format(number.normalize(), "f")
    return f"{canonical}{suffix}"


def numeric_tokens(value: str) -> tuple[str, ...]:
    """Extract normalized numeric tokens from arbitrary cell text."""

    normalized = normalize_unicode(value)
    return tuple(normalize_number(match.group()) for match in _NUMBER_PATTERN.finditer(normalized))


def table_metrics(
    candidate_cells: Iterable[Mapping[str, object]],
    reference_cells: Iterable[Mapping[str, object]],
    *,
    supported_metrics: Iterable[str] = _METRIC_NAMES,
) -> Mapping[str, Metric]:
    """Compute numeric, spanning-cell, and exact-axis table metrics.

    Cell mappings require zero-based ``row``, ``column``, and string ``text``.
    ``rowspan`` and ``colspan`` default to one. Numeric recall is an inventory
    metric; exact numeric cells additionally require the same grid position.
    """

    candidate = _parse_cells(candidate_cells)
    reference = _parse_cells(reference_cells)
    supported = frozenset(supported_metrics)
    unknown = supported.difference(_METRIC_NAMES)
    if unknown:
        raise ValueError(f"unknown supported table metrics: {sorted(unknown)}")

    candidate_numbers = Counter(token for cell in candidate for token in numeric_tokens(cell.text))
    reference_numbers = Counter(token for cell in reference for token in numeric_tokens(cell.text))
    number_matches = sum((candidate_numbers & reference_numbers).values())

    candidate_numeric_cells = Counter(
        (cell.row, cell.column, tokens) for cell in candidate if (tokens := numeric_tokens(cell.text))
    )
    reference_numeric_cells = Counter(
        (cell.row, cell.column, tokens) for cell in reference if (tokens := numeric_tokens(cell.text))
    )
    numeric_cell_matches = sum((candidate_numeric_cells & reference_numeric_cells).values())

    candidate_spans = Counter(
        (cell.row, cell.column, cell.rowspan, cell.colspan)
        for cell in candidate
        if cell.rowspan > 1 or cell.colspan > 1
    )
    reference_spans = Counter(
        (cell.row, cell.column, cell.rowspan, cell.colspan)
        for cell in reference
        if cell.rowspan > 1 or cell.colspan > 1
    )
    span_matches = sum((candidate_spans & reference_spans).values())

    metrics = {
        "numeric_recall": Metric.ratio(number_matches, sum(reference_numbers.values())),
        "exact_numeric_cell_precision": Metric.ratio(
            numeric_cell_matches, sum(candidate_numeric_cells.values())
        ),
        "exact_numeric_cell_recall": Metric.ratio(
            numeric_cell_matches, sum(reference_numeric_cells.values())
        ),
        "exact_numeric_cell_f1": _f1(
            numeric_cell_matches,
            sum(candidate_numeric_cells.values()),
            sum(reference_numeric_cells.values()),
        ),
        "span_structure_precision": Metric.ratio(span_matches, sum(candidate_spans.values())),
        "span_structure_recall": Metric.ratio(span_matches, sum(reference_spans.values())),
        "span_structure_f1": _f1(span_matches, sum(candidate_spans.values()), sum(reference_spans.values())),
        "row_exact": _axis_exact(candidate, reference, axis="row"),
        "column_exact": _axis_exact(candidate, reference, axis="column"),
    }
    for name in _METRIC_NAMES:
        if name not in supported:
            metrics[name] = Metric.unsupported(n=metrics[name].n)
    return MappingProxyType(metrics)


def _parse_cells(cells: Iterable[Mapping[str, object]]) -> tuple[_Cell, ...]:
    parsed: list[_Cell] = []
    anchors: set[tuple[int, int]] = set()
    for raw in cells:
        row = _integer_field(raw, "row", default=None)
        column = _integer_field(raw, "column", default=None)
        rowspan = _integer_field(raw, "rowspan", default=1)
        colspan = _integer_field(raw, "colspan", default=1)
        text = raw.get("text")
        if not isinstance(text, str):
            raise TypeError("cell text must be a string")
        if row < 0 or column < 0:
            raise ValueError("cell row and column must be non-negative")
        if rowspan < 1 or colspan < 1:
            raise ValueError("cell spans must be positive")
        anchor = (row, column)
        if anchor in anchors:
            raise ValueError(f"duplicate cell anchor: {anchor}")
        anchors.add(anchor)
        parsed.append(_Cell(row, column, text, rowspan, colspan))
    return tuple(sorted(parsed, key=lambda cell: (cell.row, cell.column)))


def _integer_field(raw: Mapping[str, object], name: str, *, default: int | None) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"cell {name} must be an integer")
    return value


def _f1(matches: int, candidate_count: int, reference_count: int) -> Metric:
    denominator = candidate_count + reference_count
    return Metric.ratio(2 * matches, denominator)


def _axis_exact(candidate: tuple[_Cell, ...], reference: tuple[_Cell, ...], *, axis: str) -> Metric:
    index = 0 if axis == "row" else 1

    def inventory(cells: tuple[_Cell, ...]) -> dict[int, tuple[tuple[int, str, int, int], ...]]:
        grouped: dict[int, list[tuple[int, str, int, int]]] = {}
        for cell in cells:
            anchor = (cell.row, cell.column)
            other = anchor[1 - index]
            grouped.setdefault(anchor[index], []).append((
                other,
                normalize_unicode(cell.text),
                cell.rowspan,
                cell.colspan,
            ))
        return {key: tuple(sorted(value)) for key, value in grouped.items()}

    candidate_inventory = inventory(candidate)
    reference_inventory = inventory(reference)
    population = set(candidate_inventory) | set(reference_inventory)
    exact = sum(candidate_inventory.get(key) == reference_inventory.get(key) for key in population)
    return Metric.ratio(exact, len(population))
