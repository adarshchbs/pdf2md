from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.pdf2md.schema import TableStructure
from app.pdf2md.table_provenance import TableDiagnostic, TableDiagnosticOutcome, TableDiagnosticStage

_MIN_DIAGRAM_SLOTS = 9
_MAX_DIAGRAM_POPULATED_CELLS = 2
_MAX_DIAGRAM_CHARACTERS = 20
_MIN_SPARSE_GRID_SLOTS = 100
_MAX_SPARSE_GRID_DENSITY = 0.05
_MAX_SPARSE_GRID_CHARACTERS_PER_SLOT = 2.0


@dataclass(frozen=True)
class TableQualityFeatures:
    grid_slots: int
    populated_cells: int
    populated_density: float
    normalized_character_count: int


@dataclass(frozen=True)
class TableQualityDecision:
    accepted: bool
    rejection_reason: str | None
    features: TableQualityFeatures
    diagnostic: TableDiagnostic


def validate_table_candidate(
    table: TableStructure,
    *,
    nearby_text: str = "",
) -> TableQualityDecision:
    features = table_quality_features(table)
    reason: str | None = None
    if features.populated_cells == 0:
        reason = "empty"
    elif (
        features.grid_slots >= _MIN_DIAGRAM_SLOTS
        and features.populated_cells <= _MAX_DIAGRAM_POPULATED_CELLS
        and features.normalized_character_count <= _MAX_DIAGRAM_CHARACTERS
    ):
        reason = "diagram_sparse"
    elif _looks_like_table_of_contents(table, nearby_text):
        reason = "table_of_contents"
    elif (
        features.grid_slots >= _MIN_SPARSE_GRID_SLOTS
        and features.populated_density < _MAX_SPARSE_GRID_DENSITY
        and features.normalized_character_count / features.grid_slots < _MAX_SPARSE_GRID_CHARACTERS_PER_SLOT
    ):
        reason = "pathological_sparse"
    accepted = reason is None
    return TableQualityDecision(
        accepted=accepted,
        rejection_reason=reason,
        features=features,
        diagnostic=TableDiagnostic(
            stage=TableDiagnosticStage.TABLE_QUALITY,
            outcome=(TableDiagnosticOutcome.ACCEPTED if accepted else TableDiagnosticOutcome.REJECTED),
            reason=reason or "accepted",
            metrics=(
                ("grid_slots", features.grid_slots),
                ("normalized_character_count", features.normalized_character_count),
                ("populated_cells", features.populated_cells),
                ("populated_density", features.populated_density),
            ),
        ),
    )


def normalize_table_text(value: str) -> str:
    """Return the shared NFKC and whitespace-normalized table/provenance text form."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def table_quality_features(table: TableStructure) -> TableQualityFeatures:
    normalized_text = [normalize_table_text(cell.text) for cell in table.cells]
    populated = [text for text in normalized_text if text]
    grid_slots = table.row_count * table.column_count
    return TableQualityFeatures(
        grid_slots=grid_slots,
        populated_cells=len(populated),
        populated_density=len(populated) / grid_slots,
        normalized_character_count=sum(len(re.sub(r"\s+", "", text)) for text in populated),
    )


def _looks_like_table_of_contents(table: TableStructure, nearby_text: str) -> bool:
    context = normalize_table_text(nearby_text).casefold()
    if not re.search(r"\b(?:table\s+of\s+contents|contents)\b", context):
        return False

    rows: dict[int, list[tuple[int, str]]] = {}
    for cell in table.cells:
        text = normalize_table_text(cell.text)
        if text:
            rows.setdefault(cell.row_index, []).append((cell.column_index, text))

    qualified_pages: list[int] = []
    dotted_count = 0
    two_field_count = 0
    for row_index in sorted(rows):
        ordered = [text for _, text in sorted(rows[row_index])]
        row_text = " ".join(ordered)
        if re.fullmatch(r"(?:table\s+of\s+contents|contents)", row_text, flags=re.IGNORECASE):
            continue
        match = re.fullmatch(r"(.+?)(\d{1,4}|[IVXLCDM]{1,8})", row_text, flags=re.IGNORECASE)
        if match is None or not re.search(r"[A-Za-z]", match.group(1)):
            continue
        qualified_pages.append(_page_number(match.group(2)))
        dotted_count += int(bool(re.search(r"\.{3,}", match.group(1))))
        two_field_count += int(len(ordered) >= 2)

    if len(qualified_pages) < 5:
        return False
    monotonic_count = sum(
        left <= right for left, right in zip(qualified_pages, qualified_pages[1:], strict=False)
    )
    monotonic_ratio = monotonic_count / max(len(qualified_pages) - 1, 1)
    return monotonic_ratio >= 0.8 and (
        dotted_count / len(qualified_pages) >= 0.6 or two_field_count / len(qualified_pages) >= 0.8
    )


def _page_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(value.upper()):
        current = values[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total
