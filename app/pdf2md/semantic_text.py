from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from statistics import median
from typing import Literal, TypeAlias

from app.pdf2md.schema import DocumentElement, ElementStructure

BBox: TypeAlias = tuple[float, float, float, float]
SemanticKind: TypeAlias = Literal[
    "paragraph", "heading", "caption", "footnote", "note", "recurring_margin", "page_number"
]
MarginRole: TypeAlias = Literal["running_header", "running_footer", "page_number"]

_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?\.?|table|chart|image)\s+(?:[A-Z]?\d+[A-Z]?|[IVXLC]+)\s*[.:—-]",
    re.IGNORECASE,
)
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_FOOTNOTE_LABEL_PATTERN = r"(?:[*†‡]|[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,3}|\d{1,3})"
_FOOTNOTE_RE = re.compile(
    rf"^(?:[*†‡][.)]?|\({_FOOTNOTE_LABEL_PATTERN}\)[.)]?|{_FOOTNOTE_LABEL_PATTERN}[.)]?)"
    r"(?=\s+\S|[A-Z])"
)
_FOOTNOTE_PREFIX_RE = re.compile(
    rf"^(?P<prefix>(?:\((?P<parenthesized>{_FOOTNOTE_LABEL_PATTERN})\)|"
    rf"(?P<plain>{_FOOTNOTE_LABEL_PATTERN}))[.)]?)(?:\s+|(?=[A-Za-z]))"
)
_UNICODE_REFERENCE_RE = re.compile(r"(?<=[A-Za-z]{3})(?P<label>[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,3})(?=\W|$)")
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S", re.IGNORECASE)
_LIST_LABEL_RE = re.compile(
    r"^(?P<label>(?:\((?:[A-Za-z]|[ivxlcdm]{2,}|\d{1,3})\)|\d+(?:\.\d+)*[.)]|[A-Za-z][.)]|[•▪◦‣⁃*-]))\s+",
    re.IGNORECASE,
)
_CITATION_ENTRY_RE = re.compile(
    r"^[A-Z][A-Za-z'’-]+,\s+(?:[A-Z](?:[A-Za-z'’-]+)?(?:[.,]|\s)){1,4}.*\(?(?:18|19|20)\d{2}\)?",
)
_EQUATION_RE = re.compile(r"(?:^|\s)(?:[A-Za-z][A-Za-z0-9_]*\s*)?[=≈≠≤≥<>±∑∫](?:\s|$)")
_TERMINAL_RE = re.compile(r"[.!?][\"'’”)]*$")
_TABLE_CONTINUATION_MARKER_RE = re.compile(r"continued\s*(?:→|>)", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)?(?P<number>\d{1,4}|[ivxlcdm]{1,8})(?:\s*/\s*\d{1,4})?$",
    re.IGNORECASE,
)
_ROMAN_NUMBER_RE = re.compile(r"^[ivxlcdm]{1,8}$", re.IGNORECASE)
_PARENTHETICAL_QUALIFIER_RE = re.compile(r"^\([^()\n]{1,80}\)$")


def _is_number(value: object) -> bool:
    return type(value) in (int, float)


def _validate_bbox(bbox: BBox) -> None:
    if len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    if not all(_is_number(value) for value in bbox):
        raise TypeError("bbox coordinates must be numeric and not boolean")
    if not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox coordinates must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("bbox must have positive width and height")


@dataclass(frozen=True, slots=True)
class TextSpan:
    text: str
    bbox: BBox
    font_size: float
    font_name: str = ""
    is_bold: bool = False
    is_italic: bool = False
    source_item_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox)
        if not _is_number(self.font_size):
            raise TypeError("font_size must be numeric and not boolean")
        if not math.isfinite(self.font_size) or self.font_size <= 0:
            raise ValueError("font_size must be positive and finite")
        if any(not source_item_id for source_item_id in self.source_item_ids):
            raise ValueError("text span source item IDs must not be empty")
        if len(self.source_item_ids) != len(set(self.source_item_ids)):
            raise ValueError("text span source item IDs must be unique")


@dataclass(frozen=True, slots=True)
class TextLine:
    spans: tuple[TextSpan, ...]
    bbox: BBox
    direction: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox)
        if not self.spans:
            raise ValueError("a text line requires at least one span")
        if len(self.direction) != 2 or not all(_is_number(value) for value in self.direction):
            raise TypeError("text direction must contain two numeric coordinates")
        if not all(math.isfinite(value) for value in self.direction):
            raise ValueError("text direction coordinates must be finite")
        if math.hypot(*self.direction) == 0:
            raise ValueError("text direction must be nonzero")

    @property
    def text(self) -> str:
        return _join_spans(self.spans, self.direction)

    @property
    def font_size(self) -> float:
        return _weighted_font_size(self.spans)


@dataclass(frozen=True, slots=True)
class TextBlock:
    lines: tuple[TextLine, ...]
    bbox: BBox
    page_number: int
    page_width: float
    page_height: float
    rotation: int = 0

    def __post_init__(self) -> None:
        _validate_bbox(self.bbox)
        if not self.lines:
            raise ValueError("a text block requires at least one line")
        if type(self.page_number) is not int:
            raise TypeError("page_number must be an integer and not boolean")
        if self.page_number < 1:
            raise ValueError("page_number must be at least one")
        if not _is_number(self.page_width) or not _is_number(self.page_height):
            raise TypeError("page dimensions must be numeric and not boolean")
        if not math.isfinite(self.page_width) or not math.isfinite(self.page_height):
            raise ValueError("page dimensions must be finite")
        if self.page_width <= 0 or self.page_height <= 0:
            raise ValueError("page dimensions must be positive")
        if type(self.rotation) is not int:
            raise TypeError("rotation must be an integer and not boolean")
        if self.rotation % 90 != 0:
            raise ValueError("rotation must be a multiple of 90 degrees")

    @property
    def text(self) -> str:
        return join_text_parts(tuple(line.text for line in self.lines))

    @property
    def font_size(self) -> float:
        return median(line.font_size for line in self.lines)

    @property
    def is_bold(self) -> bool:
        spans = tuple(span for line in self.lines for span in line.spans if span.text.strip())
        return (
            bool(spans)
            and sum(len(span.text) for span in spans if span.is_bold)
            >= sum(len(span.text) for span in spans) / 2
        )

    @property
    def reading_rotation(self) -> int:
        """Use page rotation only when it makes the encoded text baseline upright."""
        native_score = _upright_direction_score(self.lines, 0)
        rotated_score = _upright_direction_score(self.lines, self.rotation)
        return self.rotation if rotated_score > native_score + 0.5 else 0

    @property
    def reading_bbox(self) -> BBox:
        """Return geometry in the orientation that makes the encoded text upright."""
        return normalize_bbox(self.bbox, self.page_width, self.page_height, self.reading_rotation)

    @property
    def reading_page_size(self) -> tuple[float, float]:
        return normalize_page_size(self.page_width, self.page_height, self.reading_rotation)

    @property
    def normalized_bbox(self) -> BBox:
        """Return geometry in the page's displayed orientation."""
        return normalize_bbox(self.bbox, self.page_width, self.page_height, self.rotation)

    @property
    def normalized_page_size(self) -> tuple[float, float]:
        return normalize_page_size(self.page_width, self.page_height, self.rotation)


@dataclass(frozen=True, slots=True)
class SemanticBlock:
    kind: SemanticKind
    text: str
    source_blocks: tuple[TextBlock, ...]
    heading_level: int | None = None
    paragraph_role: str | None = None
    list_depth: int | None = None
    list_label: str | None = None
    margin_role: MarginRole | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("semantic text cannot be blank")
        if not self.source_blocks:
            raise ValueError("a semantic block requires at least one source block")
        if self.kind == "heading":
            if self.heading_level is None or not 1 <= self.heading_level <= 6:
                raise ValueError("headings require a level from one to six")
        elif self.heading_level is not None:
            raise ValueError("only headings may have a heading level")
        if self.paragraph_role is not None and self.kind != "paragraph":
            raise ValueError("only paragraphs may override the paragraph role")
        if (self.list_depth is None) != (self.list_label is None):
            raise ValueError("list depth and label must be provided together")
        if self.list_depth is not None and self.paragraph_role != "list_item":
            raise ValueError("list metadata requires the list_item role")
        if self.kind in {"recurring_margin", "page_number"}:
            if self.margin_role is None:
                raise ValueError("margin semantics require a margin role")
            if self.kind == "page_number" and self.margin_role != "page_number":
                raise ValueError("page numbers require the page_number margin role")
        elif self.margin_role is not None:
            raise ValueError("only margin semantics may have a margin role")

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(block.page_number for block in self.source_blocks))


def normalize_page_size(width: float, height: float, rotation: int) -> tuple[float, float]:
    """Return page dimensions in upright coordinates."""
    if not _is_number(width) or not _is_number(height):
        raise TypeError("page dimensions must be numeric and not boolean")
    if not math.isfinite(width) or not math.isfinite(height):
        raise ValueError("page dimensions must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("page dimensions must be positive")
    if type(rotation) is not int:
        raise TypeError("rotation must be an integer and not boolean")
    normalized_rotation = rotation % 360
    if normalized_rotation not in (0, 90, 180, 270):
        raise ValueError("rotation must be a multiple of 90 degrees")
    return (height, width) if normalized_rotation in (90, 270) else (width, height)


def normalize_point(x: float, y: float, width: float, height: float, rotation: int) -> tuple[float, float]:
    """Rotate a PDF point clockwise into upright, top-left-origin coordinates."""
    normalize_page_size(width, height, rotation)
    match rotation % 360:
        case 0:
            return x, y
        case 90:
            return height - y, x
        case 180:
            return width - x, height - y
        case 270:
            return y, width - x
        case _:
            raise AssertionError("rotation was validated")


def normalize_bbox(bbox: BBox, page_width: float, page_height: float, rotation: int) -> BBox:
    """Return the axis-aligned upright box enclosing all four rotated corners."""
    _validate_bbox(bbox)
    points = (
        normalize_point(bbox[0], bbox[1], page_width, page_height, rotation),
        normalize_point(bbox[2], bbox[1], page_width, page_height, rotation),
        normalize_point(bbox[0], bbox[3], page_width, page_height, rotation),
        normalize_point(bbox[2], bbox[3], page_width, page_height, rotation),
    )
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    return min(xs), min(ys), max(xs), max(ys)


def _upright_direction_score(lines: tuple[TextLine, ...], rotation: int) -> float:
    weighted_x = 0.0
    weighted_y = 0.0
    for line in lines:
        dx, dy = line.direction
        match rotation % 360:
            case 0:
                rotated_x, rotated_y = dx, dy
            case 90:
                rotated_x, rotated_y = -dy, dx
            case 180:
                rotated_x, rotated_y = -dx, -dy
            case 270:
                rotated_x, rotated_y = dy, -dx
            case _:
                raise ValueError("rotation must be a multiple of 90 degrees")
        weight = max(1, len(line.text.strip()))
        weighted_x += rotated_x * weight
        weighted_y += rotated_y * weight
    magnitude = math.hypot(weighted_x, weighted_y)
    return weighted_x / magnitude if magnitude else -1.0


def _span_reading_interval(span: TextSpan, direction: tuple[float, float]) -> tuple[float, float]:
    magnitude = math.hypot(*direction)
    axis_x, axis_y = direction[0] / magnitude, direction[1] / magnitude
    projections = tuple(
        x * axis_x + y * axis_y
        for x, y in (
            (span.bbox[0], span.bbox[1]),
            (span.bbox[0], span.bbox[3]),
            (span.bbox[2], span.bbox[1]),
            (span.bbox[2], span.bbox[3]),
        )
    )
    return min(projections), max(projections)


def _join_spans(spans: tuple[TextSpan, ...], direction: tuple[float, float]) -> str:
    """Join style runs using gaps projected onto the line's reading direction."""
    result = ""
    previous: TextSpan | None = None
    for span in spans:
        text = span.text
        if not text:
            previous = span
            continue
        reading_gap = 0.0
        if previous is not None:
            _, previous_end = _span_reading_interval(previous, direction)
            current_start, _ = _span_reading_interval(span, direction)
            reading_gap = current_start - previous_end
        if (
            result
            and previous is not None
            and not result[-1].isspace()
            and not text[0].isspace()
            and reading_gap >= min(previous.font_size, span.font_size) * 0.15
        ):
            result += " "
        result += text
        previous = span
    return result.strip()


def join_text_parts(parts: tuple[str, ...]) -> str:
    """Join wrapped text, removing only an explicit Unicode soft hyphen."""
    result = ""
    for raw_part in parts:
        part = raw_part.strip()
        if not part:
            continue
        if not result:
            result = part
        elif result.endswith("­"):
            result = result[:-1] + part.lstrip()
        elif result.endswith("-"):
            result += part.lstrip()
        else:
            result += " " + part
    return re.sub(r"[ \t]+", " ", result).strip()


def infer_page_column_splits(
    blocks: Sequence[TextBlock], *, atomic_bboxes: Sequence[BBox] = ()
) -> tuple[float, ...]:
    """Infer stable columns from native blocks before atomic-content suppression."""
    if len(blocks) < 2:
        return ()
    page_numbers = {block.page_number for block in blocks}
    page_sizes = {block.reading_page_size for block in blocks}
    if len(page_numbers) != 1 or len(page_sizes) != 1:
        raise ValueError("column inference requires blocks from one canonical page")
    page_width = next(iter(page_sizes))[0]
    regular = [block for block in blocks if _bbox_width(block.reading_bbox) < page_width * 0.72]
    if not atomic_bboxes:
        return _supported_splits(_column_splits(regular, page_width), regular, page_width)

    unsuppressed_evidence = [
        block
        for block in regular
        if not any(_bbox_overlap_fraction(block.reading_bbox, bbox) >= 0.5 for bbox in atomic_bboxes)
    ]
    supported = _supported_splits(
        _column_splits(unsuppressed_evidence, page_width), unsuppressed_evidence, page_width
    )
    return supported or _aligned_track_splits(unsuppressed_evidence, page_width)


def column_aware_reading_order(
    blocks: tuple[TextBlock, ...] | list[TextBlock],
    *,
    page_column_splits: Mapping[int, tuple[float, ...]] | None = None,
) -> tuple[TextBlock, ...]:
    """Order each page by vertical bands and then by fixed or detected columns."""
    pages: dict[int, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        pages[block.page_number].append(block)
    ordered: list[TextBlock] = []
    for page_number in sorted(pages):
        splits = None if page_column_splits is None else page_column_splits.get(page_number, ())
        ordered.extend(_order_page(pages[page_number], splits=splits))
    return tuple(ordered)


def reading_order(blocks: tuple[TextBlock, ...] | list[TextBlock]) -> tuple[TextBlock, ...]:
    """Alias for column-aware reading order."""
    return column_aware_reading_order(blocks)


def _order_page(blocks: list[TextBlock], *, splits: tuple[float, ...] | None = None) -> list[TextBlock]:
    if len(blocks) < 2:
        return blocks.copy()
    page_width = blocks[0].reading_page_size[0]
    regular = [block for block in blocks if _bbox_width(block.reading_bbox) < page_width * 0.72]
    effective_splits = _column_splits(regular, page_width) if splits is None else splits
    if not effective_splits:
        return _order_inline_margin_fragments(sorted(blocks, key=_position_key))

    spanning = sorted(
        (
            block
            for block in blocks
            if block not in regular or _crosses_splits(block.reading_bbox, effective_splits)
        ),
        key=_position_key,
    )
    result: list[TextBlock] = []
    remaining = [block for block in blocks if block not in spanning]
    boundary = -math.inf
    for separator in spanning:
        separator_bbox = separator.reading_bbox
        band = [block for block in remaining if boundary <= block.reading_bbox[1] < separator_bbox[1]]
        result.extend(_order_columns(band, effective_splits))
        result.append(separator)
        remaining = [block for block in remaining if block not in band]
        boundary = separator_bbox[3]
    result.extend(_order_columns(remaining, effective_splits))
    return _order_inline_margin_fragments(result)


def _inline_margin_region(block: TextBlock) -> Literal["top", "bottom"] | None:
    bbox = block.reading_bbox
    if bbox[3] - bbox[1] > block.reading_page_size[1] * 0.04:
        return None
    return _margin_region(block, 0.12)


def _order_inline_margin_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    """Order margin bands top-to-bottom, with same-row fragments left-to-right."""
    top = _order_margin_rows([block for block in blocks if _inline_margin_region(block) == "top"])
    middle = [block for block in blocks if _inline_margin_region(block) is None]
    bottom = _order_margin_rows([block for block in blocks if _inline_margin_region(block) == "bottom"])
    return [*top, *middle, *bottom]


def _order_margin_rows(blocks: list[TextBlock]) -> list[TextBlock]:
    rows: list[list[TextBlock]] = []
    for block in sorted(blocks, key=_position_key):
        if not rows or block.reading_bbox[1] >= max(item.reading_bbox[3] for item in rows[-1]):
            rows.append([block])
        else:
            rows[-1].append(block)
    return [
        block
        for row in rows
        for block in sorted(row, key=lambda item: (item.reading_bbox[0], item.reading_bbox[1]))
    ]


def _column_splits(blocks: list[TextBlock], page_width: float) -> tuple[float, ...]:
    if len(blocks) < 2:
        return ()
    centers = sorted(_center_x(block) for block in blocks)
    tolerance = page_width * 0.03
    splits: list[float] = []
    for left_center, right_center in zip(centers, centers[1:], strict=False):
        if right_center - left_center < page_width * 0.12:
            continue
        split = (left_center + right_center) / 2
        has_left = any(block.reading_bbox[2] <= split + tolerance for block in blocks)
        has_right = any(block.reading_bbox[0] >= split - tolerance for block in blocks)
        if has_left and has_right:
            splits.append(split)
    return tuple(splits)


def _supported_splits(
    splits: tuple[float, ...], blocks: list[TextBlock], page_width: float
) -> tuple[float, ...]:
    tolerance = page_width * 0.03
    page_height = blocks[0].reading_page_size[1] if blocks else 0.0
    evidence = [
        bbox
        for block in blocks
        for bbox in [block.reading_bbox]
        if _bbox_width(bbox) >= page_width * 0.05
        and bbox[3] - bbox[1] <= _bbox_width(bbox) * 2
        and page_height * 0.05 < (bbox[1] + bbox[3]) / 2 < page_height * 0.95
    ]
    return tuple(
        split
        for split in splits
        if sum(bbox[2] <= split + tolerance for bbox in evidence) >= 2
        and sum(bbox[0] >= split - tolerance for bbox in evidence) >= 2
    )


def _aligned_track_splits(blocks: list[TextBlock], page_width: float) -> tuple[float, ...]:
    tolerance = page_width * 0.04
    boxes = sorted(
        (
            block.reading_bbox
            for block in blocks
            if page_width * 0.18 <= _bbox_width(block.reading_bbox) <= page_width * 0.6
        ),
        key=lambda bbox: (bbox[0], bbox[2]),
    )
    clusters: list[list[BBox]] = []
    for bbox in boxes:
        cluster = next(
            (
                candidate
                for candidate in clusters
                if abs(median(item[0] for item in candidate) - bbox[0]) <= tolerance
                and abs(median(item[2] for item in candidate) - bbox[2]) <= tolerance
            ),
            None,
        )
        if cluster is None:
            clusters.append([bbox])
        else:
            cluster.append(bbox)
    tracks = [
        (median(item[0] for item in cluster), median(item[2] for item in cluster), len(cluster))
        for cluster in clusters
        if len(cluster) >= 2
    ]
    candidates: list[tuple[int, float, tuple[tuple[float, float, int], ...]]] = []
    for count in range(2, min(3, len(tracks)) + 1):
        for selected in combinations(tracks, count):
            ordered = tuple(sorted(selected))
            if all(
                left[1] <= right[0] + tolerance for left, right in zip(ordered, ordered[1:], strict=False)
            ):
                candidates.append((
                    sum(track[2] for track in ordered),
                    ordered[-1][1] - ordered[0][0],
                    ordered,
                ))
    if not candidates:
        return ()
    selected = max(candidates, key=lambda item: (item[0], item[1]))[2]
    return tuple((left[1] + right[0]) / 2 for left, right in zip(selected, selected[1:], strict=False))


def _bbox_overlap_fraction(first: BBox, second: BBox) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / ((_bbox_width(first)) * (first[3] - first[1]))


def _crosses_splits(bbox: BBox, splits: tuple[float, ...]) -> bool:
    return any(bbox[0] < split < bbox[2] for split in splits)


def _order_columns(blocks: list[TextBlock], splits: tuple[float, ...]) -> list[TextBlock]:
    columns: list[list[TextBlock]] = [[] for _ in range(len(splits) + 1)]
    for block in blocks:
        column = sum(_center_x(block) >= split for split in splits)
        columns[column].append(block)
    return [block for column in columns for block in sorted(column, key=_position_key)]


def _position_key(block: TextBlock) -> tuple[float, float]:
    bbox = block.reading_bbox
    return bbox[1], bbox[0]


def _center_x(block: TextBlock) -> float:
    bbox = block.reading_bbox
    return (bbox[0] + bbox[2]) / 2


def _bbox_width(bbox: BBox) -> float:
    return bbox[2] - bbox[0]


def should_join_paragraphs(previous: TextBlock, current: TextBlock) -> bool:
    """Require strong geometric, typographic, and linguistic same-page evidence."""
    if previous.page_number != current.page_number:
        return False
    if (
        _TERMINAL_RE.search(previous.text.rstrip())
        or _LIST_LABEL_RE.match(current.text.strip())
        or not _starts_like_continuation(current.text)
    ):
        return False
    if abs(previous.font_size - current.font_size) > max(0.75, previous.font_size * 0.08):
        return False

    previous_box = previous.reading_bbox
    current_box = current.reading_bbox
    page_width, page_height = previous.reading_page_size
    same_column = _horizontal_overlap(previous_box, current_box) >= 0.5
    if same_column:
        line_height = median(line.bbox[3] - line.bbox[1] for line in (*previous.lines, *current.lines))
        gap = current_box[1] - previous_box[3]
        left_offset = current_box[0] - previous_box[0]
        aligned_or_hanging = -max(4.0, page_width * 0.015) <= left_offset <= max(4.0, page_width * 0.04)
        return aligned_or_hanging and -line_height * 0.25 <= gap <= line_height * 1.5

    moves_right = current_box[0] > previous_box[0]
    at_physical_edges = previous_box[3] >= page_height * 0.82 and current_box[1] <= page_height * 0.18
    # Footnotes can end the usable body columns well above the physical page
    # bottom. Only relax that edge when the continuation starts at the very top.
    at_footnoted_body_edges = previous_box[3] >= page_height * 0.68 and current_box[1] <= page_height * 0.1
    return moves_right and (at_physical_edges or at_footnoted_body_edges)


def _atomic_is_in_column(bbox: BBox, splits: tuple[float, ...], column: int) -> bool:
    if any(bbox[0] < split < bbox[2] for split in splits):
        return True
    center = (bbox[0] + bbox[2]) / 2
    return sum(center >= split for split in splits) == column


def _should_join_cross_page_paragraphs(
    previous: SemanticBlock,
    current: SemanticBlock,
    page_column_splits: Mapping[int, tuple[float, ...]] | None,
    page_atomic_bboxes: Mapping[int, Sequence[BBox]] | None,
) -> bool:
    """Accept only a lower-case continuation across adjoining physical page edges."""
    if (
        previous.kind != "paragraph"
        or current.kind != "paragraph"
        or previous.paragraph_role is not None
        or current.paragraph_role is not None
    ):
        return False
    previous_block = previous.source_blocks[-1]
    current_block = current.source_blocks[0]
    if current_block.page_number != previous_block.page_number + 1:
        return False
    if (
        _TERMINAL_RE.search(previous.text.rstrip())
        or _LIST_LABEL_RE.match(current.text.strip())
        or _FOOTNOTE_RE.match(current.text.strip())
        or not _starts_like_cross_page_continuation(current.text)
        or abs(previous_block.font_size - current_block.font_size)
        > max(0.75, previous_block.font_size * 0.08)
        or previous_block.is_bold != current_block.is_bold
    ):
        return False

    previous_box = previous_block.reading_bbox
    current_box = current_block.reading_bbox
    previous_width, previous_height = previous_block.reading_page_size
    current_width, current_height = current_block.reading_page_size
    if previous_box[3] < previous_height * 0.8 or current_box[1] > current_height * 0.2:
        return False

    previous_splits = (
        () if page_column_splits is None else page_column_splits.get(previous_block.page_number, ())
    )
    current_splits = (
        () if page_column_splits is None else page_column_splits.get(current_block.page_number, ())
    )
    previous_column = sum(_center_x(previous_block) >= split for split in previous_splits)
    current_column = sum(_center_x(current_block) >= split for split in current_splits)
    if page_atomic_bboxes is not None:
        previous_atomic = page_atomic_bboxes.get(previous_block.page_number, ())
        current_atomic = page_atomic_bboxes.get(current_block.page_number, ())
        if any(
            _atomic_is_in_column(bbox, previous_splits, previous_column) and bbox[1] >= previous_box[3]
            for bbox in previous_atomic
        ) or any(
            _atomic_is_in_column(bbox, current_splits, current_column) and bbox[3] <= current_box[1]
            for bbox in current_atomic
        ):
            return False
    if previous_splits:
        # Multi-column reading proceeds from the final column on one page to the
        # first column on the next. Requiring both physical edges prevents a
        # lower-case caption or side note from becoming continuation evidence.
        return bool(
            previous_column == len(previous_splits)
            and current_column == 0
            and current_box[0] <= current_width * 0.2
            and previous_box[0] >= previous_width * 0.35
        )

    normalized_previous = tuple(value / previous_width for value in (previous_box[0], previous_box[2]))
    normalized_current = tuple(value / current_width for value in (current_box[0], current_box[2]))
    overlap = max(
        0.0,
        min(normalized_previous[1], normalized_current[1])
        - max(normalized_previous[0], normalized_current[0]),
    )
    narrower_width = min(
        normalized_previous[1] - normalized_previous[0],
        normalized_current[1] - normalized_current[0],
    )
    # A missed column split must not turn two aligned narrow columns into a
    # putative single-column page boundary. Without explicit column evidence,
    # require both endpoints to occupy most of the normalized text track.
    return bool(
        current_column == 0
        and narrower_width >= 0.6
        and overlap / narrower_width >= 0.8
        and abs(normalized_previous[0] - normalized_current[0]) <= 0.04
    )


def _merge_cross_page_footnote_continuations(
    semantics: Sequence[SemanticBlock],
) -> tuple[SemanticBlock, ...]:
    """Join only uniquely supported small-type footer continuations across pages."""
    continuation_to_note: dict[int, int] = {}
    for continuation_index, continuation in enumerate(semantics):
        if continuation.kind != "paragraph" or continuation.paragraph_role is not None:
            continue
        continuation_block = continuation.source_blocks[0]
        continuation_bbox = continuation_block.reading_bbox
        continuation_height = continuation_block.reading_page_size[1]
        if continuation_bbox[1] < continuation_height * 0.85 or not _starts_like_cross_page_continuation(
            continuation.text
        ):
            continue
        candidates: list[int] = []
        for note_index, note in enumerate(semantics):
            if note.kind != "footnote" or not note.text.rstrip().endswith("-"):
                continue
            note_block = note.source_blocks[-1]
            note_bbox = note_block.reading_bbox
            if (
                continuation_block.page_number != note_block.page_number + 1
                or note_bbox[3] < note_block.reading_page_size[1] * 0.85
                or abs(note_block.font_size - continuation_block.font_size)
                > max(0.75, note_block.font_size * 0.08)
            ):
                continue
            candidates.append(note_index)
        if len(candidates) == 1:
            continuation_to_note[continuation_index] = candidates[0]

    result = list(semantics)
    for continuation_index, note_index in continuation_to_note.items():
        note = result[note_index]
        continuation = result[continuation_index]
        result[note_index] = replace(
            note,
            text=join_text_parts((note.text, continuation.text)),
            source_blocks=(*note.source_blocks, *continuation.source_blocks),
        )
    return tuple(semantic for index, semantic in enumerate(result) if index not in continuation_to_note)


def _merge_cross_page_paragraphs(
    semantics: Sequence[SemanticBlock],
    page_column_splits: Mapping[int, tuple[float, ...]] | None,
    page_atomic_bboxes: Mapping[int, Sequence[BBox]] | None,
) -> tuple[SemanticBlock, ...]:
    """Merge proven continuations while leaving page chrome and footnotes in place."""
    result: list[SemanticBlock] = []
    transparent_kinds: frozenset[SemanticKind] = frozenset({"recurring_margin", "page_number", "footnote"})
    for semantic in semantics:
        previous_index = next(
            (
                index
                for index in range(len(result) - 1, -1, -1)
                if result[index].kind not in transparent_kinds
            ),
            None,
        )
        if previous_index is not None and _should_join_cross_page_paragraphs(
            result[previous_index], semantic, page_column_splits, page_atomic_bboxes
        ):
            previous = result[previous_index]
            result[previous_index] = replace(
                previous,
                text=join_text_parts((previous.text, semantic.text)),
                source_blocks=(*previous.source_blocks, *semantic.source_blocks),
            )
            continue
        result.append(semantic)
    return tuple(result)


def join_paragraphs(
    blocks: tuple[TextBlock, ...] | list[TextBlock],
    *,
    page_column_splits: Mapping[int, tuple[float, ...]] | None = None,
) -> tuple[SemanticBlock, ...]:
    """Join only adjacent paragraph continuations in column-aware reading order."""
    ordered = column_aware_reading_order(blocks, page_column_splits=page_column_splits)
    groups: list[list[TextBlock]] = []
    for block in ordered:
        if groups and should_join_paragraphs(groups[-1][-1], block):
            groups[-1].append(block)
        else:
            groups.append([block])
    result: list[SemanticBlock] = []
    for group in groups:
        text = join_text_parts(tuple(block.text for block in group))
        label, depth, content = _list_item_parts(text, group[0])
        result.append(
            SemanticBlock(
                kind="paragraph",
                text=content,
                source_blocks=tuple(group),
                paragraph_role="list_item" if label is not None else None,
                list_depth=depth,
                list_label=label,
            )
        )
    return tuple(result)


def _list_item_parts(text: str, source: TextBlock) -> tuple[str | None, int | None, str]:
    bbox = source.reading_bbox
    page_height = source.reading_page_size[1]
    match = _LIST_LABEL_RE.match(text)
    if match is None:
        return None, None, text
    label = match.group("label")
    in_margin = bbox[3] <= page_height * 0.1 or bbox[1] >= page_height * 0.9
    if in_margin and label[0].isdigit() and label[:-1].count(".") >= 1:
        return None, None, text
    content = text[match.end() :].strip()
    if not content or _looks_like_non_list_label(label, content):
        return None, None, text
    if label.startswith("("):
        marker = label[1:-1]
        depth = 2 if marker.casefold() in {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii"} else 1
    elif label[0].isdigit() and "." in label[:-1]:
        depth = label[:-1].count(".")
    else:
        depth = 0
    return label, depth, content


def _looks_like_non_list_label(label: str, content: str) -> bool:
    if label[0].isdigit():
        numeric_label = label.rstrip(".)")
        if numeric_label.isdigit() and 1800 <= int(numeric_label) <= 2999:
            return True
        if _EQUATION_RE.search(content) or _CITATION_ENTRY_RE.match(content):
            return True
        if not _TERMINAL_RE.search(content) and not content.endswith((":", ";")) and _is_title_case(content):
            return True
    if label in {"•", "▪", "◦", "‣", "⁃", "*", "-"}:
        tokens = content.split()
        if len(tokens) >= 2 and all(len(token.strip(".,:;()")) == 1 for token in tokens):
            return True
    return False


def _is_title_case(text: str) -> bool:
    words = tuple(re.findall(r"[A-Za-z][A-Za-z'’-]*", text))
    if not words or len(words) > 16:
        return False
    significant = tuple(
        word
        for word in words
        if word.casefold() not in {"a", "an", "and", "for", "in", "of", "on", "the", "to"}
    )
    return bool(significant) and all(word.isupper() or word[0].isupper() for word in significant)


def _is_list_item_continuation(previous: SemanticBlock, current: SemanticBlock) -> bool:
    if (
        previous.paragraph_role != "list_item"
        or previous.list_label is None
        or not previous.list_label.startswith("(")
        or current.kind != "paragraph"
        or current.paragraph_role is not None
        or previous.page_numbers != current.page_numbers
    ):
        return False
    previous_bbox = previous.source_blocks[-1].reading_bbox
    current_bbox = current.source_blocks[0].reading_bbox
    page_width, page_height = current.source_blocks[0].reading_page_size
    return (
        current_bbox[0] - previous_bbox[0] >= page_width * 0.01
        and current_bbox[1] - previous_bbox[3] <= page_height * 0.04
    )


def is_caption(block: TextBlock) -> bool:
    return bool(_CAPTION_RE.match(block.text.strip())) and len(block.text) <= 5_000


def is_table_continuation_marker(block: TextBlock) -> bool:
    return (
        block.reading_bbox[1] >= block.reading_page_size[1] * 0.8
        and _TABLE_CONTINUATION_MARKER_RE.fullmatch(block.text.strip()) is not None
    )


def _line_has_leading_footnote_marker(line: TextLine) -> bool:
    nonblank = tuple(span for span in line.spans if span.text.strip())
    if len(nonblank) < 2:
        return False
    marker, following = nonblank[0], nonblank[1]
    label = marker.text.strip().translate(_SUPERSCRIPT_DIGITS)
    marker_center = (marker.bbox[1] + marker.bbox[3]) / 2
    following_center = (following.bbox[1] + following.bbox[3]) / 2
    return (
        re.fullmatch(r"(?:\d{1,3}|[*†‡])", label) is not None
        and marker.font_size <= following.font_size * 0.95
        and marker_center <= following_center - following.font_size * 0.05
        and marker.bbox[0] <= following.bbox[0]
    )


def _has_leading_footnote_marker_span(block: TextBlock) -> bool:
    return _line_has_leading_footnote_marker(block.lines[0])


def _looks_like_numeric_table_row(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 6:
        return False
    numeric = sum(
        re.fullmatch(r"[+−-]?\d+(?:[.,]\d+)?%?", token.strip("()[]")) is not None for token in tokens
    )
    return numeric / len(tokens) >= 0.7


def is_footnote(block: TextBlock, body_font_size: float) -> bool:
    if body_font_size <= 0:
        raise ValueError("body_font_size must be positive")
    bbox = block.reading_bbox
    page_height = block.reading_page_size[1]
    explicit_marker_span = _has_leading_footnote_marker_span(block)
    return (
        not _looks_like_numeric_table_row(block.text)
        and bbox[1] >= page_height * 0.72
        and (
            (block.font_size <= body_font_size * 0.85 and bool(_FOOTNOTE_RE.match(block.text.strip())))
            or (explicit_marker_span and block.font_size <= body_font_size * 1.02)
        )
    )


def _split_compound_footnote_blocks(blocks: Sequence[TextBlock]) -> tuple[TextBlock, ...]:
    result: list[TextBlock] = []
    for block in blocks:
        marker_starts = [
            index for index, line in enumerate(block.lines) if _line_has_leading_footnote_marker(line)
        ]
        if (
            len(marker_starts) < 2
            or marker_starts[0] != 0
            or block.reading_bbox[1] < block.reading_page_size[1] * 0.72
        ):
            result.append(block)
            continue
        boundaries = [*marker_starts, len(block.lines)]
        for start, end in zip(boundaries, boundaries[1:], strict=False):
            lines = block.lines[start:end]
            bbox = (
                min(line.bbox[0] for line in lines),
                min(line.bbox[1] for line in lines),
                max(line.bbox[2] for line in lines),
                max(line.bbox[3] for line in lines),
            )
            result.append(
                TextBlock(
                    lines=lines,
                    bbox=bbox,
                    page_number=block.page_number,
                    page_width=block.page_width,
                    page_height=block.page_height,
                    rotation=block.rotation,
                )
            )
    return tuple(result)


def is_heading(block: TextBlock, body_font_size: float) -> bool:
    if body_font_size <= 0:
        raise ValueError("body_font_size must be positive")
    text = block.text.strip()
    if (
        not text
        or len(text) > 180
        or len(block.lines) > 3
        or _TERMINAL_RE.search(text)
        or _PARENTHETICAL_QUALIFIER_RE.fullmatch(text)
    ):
        return False
    has_typographic_signal = block.font_size >= body_font_size * 1.15 or (
        block.is_bold and block.font_size >= body_font_size
    )
    return has_typographic_signal and (len(text.split()) <= 16 or bool(_NUMBERED_HEADING_RE.match(text)))


def heading_level(block: TextBlock, body_font_size: float) -> int:
    if not is_heading(block, body_font_size):
        raise ValueError("block is not a heading")
    ratio = block.font_size / body_font_size
    if ratio >= 1.7:
        return 1
    if ratio >= 1.4:
        return 2
    if ratio >= 1.15:
        return 3
    return 4


def _is_heading_subtitle(block: TextBlock, previous: SemanticBlock | None) -> bool:
    """Recognize a short parenthetical qualifier below a heading without absorbing it."""
    if previous is None or previous.kind != "heading":
        return False
    text = block.text.strip()
    if _PARENTHETICAL_QUALIFIER_RE.fullmatch(text) is None or len(text.split()) > 8:
        return False
    heading = previous.source_blocks[-1]
    if block.page_number != heading.page_number or block.font_size > heading.font_size * 1.05:
        return False
    heading_bbox = heading.reading_bbox
    subtitle_bbox = block.reading_bbox
    page_width, page_height = block.reading_page_size
    center_delta = abs(_center_x(block) - _center_x(heading))
    gap = subtitle_bbox[1] - heading_bbox[3]
    line_height = max(heading_bbox[3] - heading_bbox[1], subtitle_bbox[3] - subtitle_bbox[1])
    return center_delta <= page_width * 0.03 and -line_height * 0.25 <= gap <= max(
        line_height * 0.75, page_height * 0.015
    )


def _margin_region(block: TextBlock, margin_fraction: float) -> Literal["top", "bottom"] | None:
    bbox = block.reading_bbox
    page_height = block.reading_page_size[1]
    center_y = (bbox[1] + bbox[3]) / 2
    if center_y <= page_height * margin_fraction:
        return "top"
    if center_y >= page_height * (1 - margin_fraction):
        return "bottom"
    return None


def _block_from_lines(lines: tuple[TextLine, ...], source: TextBlock) -> TextBlock:
    bboxes = tuple(line.bbox for line in lines)
    return TextBlock(
        lines=lines,
        bbox=(
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        ),
        page_number=source.page_number,
        page_width=source.page_width,
        page_height=source.page_height,
        rotation=source.rotation,
    )


def _line_reading_bbox(line: TextLine, source: TextBlock) -> BBox:
    return normalize_bbox(
        line.bbox,
        source.page_width,
        source.page_height,
        source.reading_rotation,
    )


def _visual_line_rows(block: TextBlock) -> tuple[tuple[TextLine, ...], ...]:
    """Group native lines into displayed rows using rotation-normalized geometry."""
    rows: list[list[TextLine]] = []
    for line in sorted(
        block.lines,
        key=lambda item: (
            _line_reading_bbox(item, block)[1],
            _line_reading_bbox(item, block)[0],
            _line_reading_bbox(item, block)[3],
        ),
    ):
        line_bbox = _line_reading_bbox(line, block)
        if not rows:
            rows.append([line])
            continue
        row_bboxes = tuple(_line_reading_bbox(item, block) for item in rows[-1])
        row_top = min(bbox[1] for bbox in row_bboxes)
        row_bottom = max(bbox[3] for bbox in row_bboxes)
        overlap = min(row_bottom, line_bbox[3]) - max(row_top, line_bbox[1])
        shorter = min(row_bottom - row_top, line_bbox[3] - line_bbox[1])
        if overlap > 0 and overlap / shorter >= 0.5:
            rows[-1].append(line)
        else:
            rows.append([line])
    return tuple(tuple(sorted(row, key=lambda item: _line_reading_bbox(item, block)[0])) for row in rows)


def _row_is_short_heading(row: Sequence[TextLine]) -> bool:
    spans = tuple(span for line in row for span in line.spans if span.text.strip())
    text = join_text_parts(tuple(line.text for line in row)).strip()
    has_bold_signal = any(span.is_bold for span in spans)
    numbered_heading = _NUMBERED_HEADING_RE.match(text) is not None and _is_title_case(text)
    has_heading_shape = _is_title_case(text) or numbered_heading
    starts_list = _LIST_LABEL_RE.match(text) is not None and not numbered_heading
    return bool(
        spans
        and has_bold_signal
        and has_heading_shape
        and len(text) <= 180
        and (3 <= len(text.split()) <= 16 or numbered_heading)
        and _TERMINAL_RE.search(text) is None
        and _CAPTION_RE.match(text) is None
        and _PARENTHETICAL_QUALIFIER_RE.fullmatch(text) is None
        and not starts_list
    )


def _row_has_bold_majority(row: Sequence[TextLine]) -> bool:
    spans = tuple(span for line in row for span in line.spans if span.text.strip())
    return (
        bool(spans)
        and sum(len(span.text.strip()) for span in spans if span.is_bold)
        >= sum(len(span.text.strip()) for span in spans) / 2
    )


def _split_heading_style_barriers(
    blocks: Sequence[TextBlock], *, protected_ids: set[int]
) -> tuple[TextBlock, ...]:
    """Split a short bold visual row from following prose in a mixed native block."""
    result: list[TextBlock] = []
    for block in blocks:
        if id(block) in protected_ids:
            result.append(block)
            continue
        rows = _visual_line_rows(block)
        boundaries = [
            index
            for index in range(1, len(rows))
            if _row_is_short_heading(rows[index - 1])
            and not _row_has_bold_majority(rows[index])
            and (next_text := join_text_parts(tuple(line.text for line in rows[index])).strip())
            and _TERMINAL_RE.search(next_text) is not None
            and _CAPTION_RE.match(next_text) is None
            and _LIST_LABEL_RE.match(next_text) is None
            and _PARENTHETICAL_QUALIFIER_RE.fullmatch(next_text) is None
            and not _starts_like_continuation(next_text)
        ]
        if len(boundaries) != 1:
            result.append(block)
            continue
        boundary = boundaries[0]
        if any(_row_has_bold_majority(row) for row in rows[boundary + 1 :]):
            result.append(block)
            continue
        before = tuple(line for row in rows[:boundary] for line in row)
        after = tuple(line for row in rows[boundary:] for line in row)
        if not before or not after:
            raise AssertionError("validated style barrier must have text on both sides")
        result.extend((_block_from_lines(before, block), _block_from_lines(after, block)))
    return tuple(result)


def _split_block_at_spans(block: TextBlock, selected_spans: frozenset[int]) -> tuple[TextBlock, TextBlock]:
    selected_lines: list[TextLine] = []
    residual_lines: list[TextLine] = []
    for line in block.lines:
        selected = tuple(span for span in line.spans if id(span) in selected_spans and span.text.strip())
        residual = tuple(span for span in line.spans if id(span) not in selected_spans and span.text.strip())
        for spans, destination in ((selected, selected_lines), (residual, residual_lines)):
            if spans:
                destination.append(
                    TextLine(
                        spans=spans,
                        bbox=(
                            min(span.bbox[0] for span in spans),
                            min(span.bbox[1] for span in spans),
                            max(span.bbox[2] for span in spans),
                            max(span.bbox[3] for span in spans),
                        ),
                    )
                )
    if not selected_lines or not residual_lines:
        raise ValueError("a compound margin block requires selected and residual text")
    return (
        _block_from_lines(tuple(selected_lines), block),
        _block_from_lines(tuple(residual_lines), block),
    )


def _roman_value(token: str) -> int | None:
    normalized = token.casefold()
    if _ROMAN_NUMBER_RE.fullmatch(normalized) is None:
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for character in reversed(normalized):
        value = values[character]
        total += -value if value < previous else value
        previous = max(previous, value)
    # Reject arbitrary strings made from Roman glyphs (for example, "civil").
    encoded = ""
    remainder = total
    for value, glyph in (
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ):
        count, remainder = divmod(remainder, value)
        encoded += glyph * count
    return total if encoded == normalized else None


def _page_number_value(text: str) -> int | None:
    match = _PAGE_NUMBER_RE.fullmatch(text.strip())
    if match is None:
        return None
    token = match.group("number")
    return int(token) if token.isdigit() else _roman_value(token)


def _isolated_page_number_spans(block: TextBlock) -> tuple[frozenset[int], int] | None:
    nonblank_lines = tuple(line for line in block.lines if line.text.strip())
    for line in nonblank_lines if len(nonblank_lines) >= 2 else ():
        value = _page_number_value(line.text)
        if value is None:
            continue
        overlapping = tuple(
            other
            for other in nonblank_lines
            if other is not line and min(line.bbox[3], other.bbox[3]) > max(line.bbox[1], other.bbox[1])
        )
        if overlapping:
            horizontal_gap = min(
                max(other.bbox[0] - line.bbox[2], line.bbox[0] - other.bbox[2], 0.0) for other in overlapping
            )
            if horizontal_gap < block.reading_page_size[0] * 0.08:
                continue
        return frozenset(id(span) for span in line.spans if span.text.strip()), value
    for line in nonblank_lines:
        nonblank_spans = tuple(span for span in line.spans if span.text.strip())
        for span in nonblank_spans:
            value = _page_number_value(span.text)
            others = tuple(other for other in nonblank_spans if other is not span)
            if value is None or not others:
                continue
            horizontal_gap = min(
                max(other.bbox[0] - span.bbox[2], span.bbox[0] - other.bbox[2], 0.0) for other in others
            )
            if horizontal_gap >= block.reading_page_size[0] * 0.08:
                return frozenset((id(span),)), value
    return None


def _split_recurring_page_numbers(
    blocks: Sequence[TextBlock], *, min_pages: int, margin_fraction: float
) -> tuple[tuple[TextBlock, ...], set[int], set[int]]:
    pages: dict[int, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        pages[block.page_number].append(block)
    candidates: list[tuple[TextBlock, frozenset[int] | None, int]] = []
    for block in blocks:
        region = _margin_region(block, margin_fraction)
        later_count = sum(other.reading_bbox[1] > block.reading_bbox[1] for other in pages[block.page_number])
        in_variable_bottom_margin = (
            block.reading_bbox[1] >= block.reading_page_size[1] * 0.4 and later_count <= 2
        )
        if region is None and not in_variable_bottom_margin:
            continue
        value = _page_number_value(block.text)
        if value is not None:
            candidates.append((block, None, value))
            continue
        isolated = _isolated_page_number_spans(block)
        if isolated is not None:
            candidates.append((block, isolated[0], isolated[1]))

    pages_by_offset: dict[int, set[int]] = defaultdict(set)
    for block, _, value in candidates:
        pages_by_offset[value - block.page_number].add(block.page_number)
    accepted_offsets = {
        offset for offset, page_numbers in pages_by_offset.items() if len(page_numbers) >= min_pages
    }

    selected = {
        id(block): span for block, span, value in candidates if value - block.page_number in accepted_offsets
    }
    result: list[TextBlock] = []
    page_number_ids: set[int] = set()
    companion_ids: set[int] = set()
    for block in blocks:
        if id(block) not in selected:
            result.append(block)
            continue
        span = selected[id(block)]
        if span is None:
            result.append(block)
            page_number_ids.add(id(block))
            continue
        number, residual = _split_block_at_spans(block, span)
        result.extend((number, residual))
        page_number_ids.add(id(number))
        companion_ids.add(id(residual))
    return tuple(result), page_number_ids, companion_ids


def _split_validated_footer_companions(
    blocks: Sequence[TextBlock],
    *,
    page_number_ids: set[int],
    min_pages: int,
    margin_fraction: float,
) -> tuple[tuple[TextBlock, ...], set[int]]:
    """Split out one footer label paired with each validated native-baseline page number."""
    numbers_by_page: dict[int, list[TextBlock]] = defaultdict(list)
    blocks_by_page: dict[int, list[TextBlock]] = defaultdict(list)
    for block in blocks:
        blocks_by_page[block.page_number].append(block)
        if id(block) in page_number_ids and _margin_region(block, margin_fraction) == "bottom":
            numbers_by_page[block.page_number].append(block)

    # (page number, source block, selected source spans, candidate component)
    matches: dict[int, tuple[TextBlock, TextBlock, frozenset[int] | None, TextBlock]] = {}
    for page_number, numbers in numbers_by_page.items():
        compatible: list[tuple[TextBlock, TextBlock, frozenset[int] | None, TextBlock]] = []
        for number in numbers:
            for source in blocks_by_page[page_number]:
                if id(source) in page_number_ids:
                    continue
                variants: list[tuple[frozenset[int] | None, TextBlock]] = [(None, source)]
                if not _is_conservative_footer_companion(source, number, margin_fraction):
                    variants = []
                    seen: set[frozenset[int]] = set()
                    for line in source.lines:
                        line_spans = frozenset(id(span) for span in line.spans if span.text.strip())
                        if line_spans and line_spans not in seen:
                            seen.add(line_spans)
                            variants.append((line_spans, _block_from_lines((line,), source)))
                        for span in line.spans:
                            if not span.text.strip():
                                continue
                            span_ids = frozenset((id(span),))
                            if span_ids in seen:
                                continue
                            seen.add(span_ids)
                            variants.append((
                                span_ids,
                                _block_from_lines(
                                    (TextLine(spans=(span,), bbox=span.bbox, direction=line.direction),),
                                    source,
                                ),
                            ))
                compatible.extend(
                    (number, source, selected, candidate)
                    for selected, candidate in variants
                    if _is_conservative_footer_companion(candidate, number, margin_fraction)
                )
        if len(compatible) == 1:
            matches[page_number] = compatible[0]

    matched_pages = set(matches)
    if not _has_recurrence_support(matched_pages, set(numbers_by_page), min_pages):
        return tuple(blocks), set()
    sides_by_parity: dict[int, set[bool]] = defaultdict(set)
    for page_number, (number, _, _, companion) in matches.items():
        sides_by_parity[page_number % 2].add(_center_x(companion) < _center_x(number))
    if any(len(sides) != 1 for sides in sides_by_parity.values()):
        return tuple(blocks), set()
    if len(sides_by_parity) == 2 and next(iter(sides_by_parity[0])) == next(iter(sides_by_parity[1])):
        return tuple(blocks), set()

    selected_by_source = {id(source): selected for _, source, selected, _ in matches.values()}
    result: list[TextBlock] = []
    companion_ids: set[int] = set()
    for block in blocks:
        if id(block) not in selected_by_source:
            result.append(block)
            continue
        selected = selected_by_source[id(block)]
        if selected is None:
            result.append(block)
            companion_ids.add(id(block))
            continue
        companion, residual = _split_block_at_spans(block, selected)
        result.extend((companion, residual))
        companion_ids.add(id(companion))
    return tuple(result), companion_ids


def _is_conservative_footer_companion(
    candidate: TextBlock, number: TextBlock, margin_fraction: float
) -> bool:
    text = candidate.text.strip()
    if (
        _margin_region(candidate, margin_fraction) != "bottom"
        or len(candidate.lines) != 1
        or len(text) > 120
        or "continued" in text.casefold()
        or _PAGE_NUMBER_RE.fullmatch(text) is not None
        or _looks_like_table_note(text)
        or _TERMINAL_RE.search(text) is not None
        or abs(candidate.font_size - number.font_size) > max(1.0, number.font_size * 0.2)
        or candidate.page_width != number.page_width
        or candidate.page_height != number.page_height
        or candidate.rotation != number.rotation
    ):
        return False
    candidate_bbox = candidate.bbox
    number_bbox = number.bbox
    candidate_height = candidate_bbox[3] - candidate_bbox[1]
    number_height = number_bbox[3] - number_bbox[1]
    vertical_overlap = min(candidate_bbox[3], number_bbox[3]) - max(candidate_bbox[1], number_bbox[1])
    center_delta = abs((candidate_bbox[1] + candidate_bbox[3]) - (number_bbox[1] + number_bbox[3])) / 2
    horizontal_gap = max(candidate_bbox[0] - number_bbox[2], number_bbox[0] - candidate_bbox[2], 0.0)
    return (
        vertical_overlap / min(candidate_height, number_height) >= 0.6
        and center_delta <= max(candidate_height, number_height) * 0.25
        and horizontal_gap >= candidate.page_width * 0.08
    )


def _looks_like_table_note(text: str) -> bool:
    stripped = text.strip()
    # Alternating publication footers can name the current table in an
    # all-caps, slash-separated label (for example, UNDP statistical annexes).
    # Same-baseline folio geometry and parity recurrence establish those in
    # _split_validated_footer_companions; do not reject them by prefix alone.
    is_designed_table_footer = "/" in stripped and stripped == stripped.upper()
    if is_designed_table_footer:
        return False
    return (
        re.match(
            r"^(?:fig(?:ure)?\.?|table|note(?:s)?\s*:|source(?:s)?\s*:)",
            stripped,
            re.IGNORECASE,
        )
        is not None
    )


def _is_repeated_top_table_label(text: str, region: Literal["top", "bottom"] | None) -> bool:
    """Allow only exact all-caps table identifiers to prove recurring top chrome."""
    stripped = " ".join(text.split())
    return bool(
        region == "top"
        and stripped == stripped.upper()
        and re.fullmatch(r"TABLE\s+[A-Z]?\d+[A-Z]?(?:[.-]\d+)*", stripped) is not None
    )


def _has_recurrence_support(page_numbers: set[int], available_pages: set[int], min_pages: int) -> bool:
    if len(page_numbers) >= min_pages:
        return True
    if len(page_numbers) < 2:
        return False
    parity_pages = {page for page in available_pages if page % 2 == next(iter(page_numbers)) % 2}
    return page_numbers == parity_pages


def _split_recurring_margin_spans(
    blocks: Sequence[TextBlock], *, min_pages: int, margin_fraction: float
) -> tuple[tuple[TextBlock, ...], set[int]]:
    """Separate recurring chrome from unrelated text sharing one native PDF line."""
    available_pages = {block.page_number for block in blocks}
    candidates: dict[tuple[str, Literal["top", "bottom"]], list[tuple[TextBlock, frozenset[int]]]] = (
        defaultdict(list)
    )
    for block in blocks:
        region = _margin_region(block, margin_fraction)
        spans = tuple(span for line in block.lines for span in line.spans if span.text.strip())
        if region is None or len(spans) < 2:
            continue
        for line in block.lines:
            line_spans = frozenset(id(span) for span in line.spans if span.text.strip())
            normalized_line = " ".join(line.text.casefold().split())
            if (
                line_spans
                and len(block.lines) >= 2
                and len(normalized_line) >= 3
                and "continued" not in normalized_line
            ):
                candidates[(normalized_line, region)].append((block, line_spans))
        for span in spans:
            others = tuple(other for other in spans if other is not span)
            horizontal_gap = min(
                max(other.bbox[0] - span.bbox[2], span.bbox[0] - other.bbox[2], 0.0) for other in others
            )
            normalized = " ".join(span.text.casefold().split())
            if (
                horizontal_gap >= block.reading_page_size[0] * 0.08
                and len(normalized) >= 3
                and "continued" not in normalized
            ):
                candidates[(normalized, region)].append((block, frozenset((id(span),))))

    selected_by_block: dict[int, set[int]] = defaultdict(set)
    for occurrences in candidates.values():
        pages = {block.page_number for block, _ in occurrences}
        if _has_recurrence_support(pages, available_pages, min_pages):
            for block, span_ids in occurrences:
                selected_by_block[id(block)].update(span_ids)

    result: list[TextBlock] = []
    recurring_ids: set[int] = set()
    for block in blocks:
        selected = frozenset(selected_by_block.get(id(block), set()))
        span_ids = frozenset(id(span) for line in block.lines for span in line.spans if span.text.strip())
        if not selected or selected == span_ids:
            result.append(block)
            continue
        recurring, residual = _split_block_at_spans(block, selected)
        result.extend((recurring, residual))
        recurring_ids.add(id(recurring))
    return tuple(result), recurring_ids


def _recurring_margin_roles(
    blocks: Sequence[TextBlock],
    *,
    min_pages: int,
    margin_fraction: float,
    split_margin_ids: set[int] | None = None,
    page_number_companion_ids: set[int] | None = None,
    validated_companion_ids: set[int] | None = None,
) -> dict[int, MarginRole]:
    keyed: dict[tuple[str, Literal["top", "bottom"]], list[TextBlock]] = defaultdict(list)
    pages = defaultdict(list)
    for block in blocks:
        pages[block.page_number].append(block)
        region = _margin_region(block, margin_fraction)
        normalized_text = _normalized_recurrence_text(block.text, block.page_number)
        if (
            region is not None
            and normalized_text
            and "continued" not in normalized_text
            and (not _looks_like_table_note(block.text) or _is_repeated_top_table_label(block.text, region))
        ):
            keyed[(normalized_text, region)].append(block)
            tokens = normalized_text.split()
            if len(tokens) >= 2:
                keyed[("\x00".join(sorted(tokens)), region)].append(block)

    accepted: dict[int, MarginRole] = {}
    available_pages = set(pages)
    for (_, region), candidates in keyed.items():
        candidate_pages = {block.page_number for block in candidates}
        if _has_recurrence_support(candidate_pages, available_pages, min_pages):
            role: MarginRole = "running_header" if region == "top" else "running_footer"
            accepted.update((id(block), role) for block in candidates)

    for block in blocks:
        if split_margin_ids is not None and id(block) in split_margin_ids:
            region = _margin_region(block, margin_fraction)
            if region is None:
                raise ValueError("split recurring margin text must remain in a margin region")
            accepted[id(block)] = "running_header" if region == "top" else "running_footer"

    if page_number_companion_ids is not None:
        for block in blocks:
            if id(block) not in page_number_companion_ids:
                continue
            region = _margin_region(block, margin_fraction)
            if region is not None:
                accepted[id(block)] = "running_header" if region == "top" else "running_footer"

    if validated_companion_ids is not None:
        accepted.update((block_id, "running_footer") for block_id in validated_companion_ids)

    # Financial statements often place the same note immediately below a variable-height
    # table rather than at the physical page edge. Accept it only when it recurs on enough
    # distinct pages, stays in the lower page half, and has at most two later text blocks.
    repeated = defaultdict(list)
    for block in blocks:
        if block.reading_bbox[1] >= block.reading_page_size[1] * 0.4:
            key = re.sub(r"(?<!\w)\d+(?!\w)", "#", " ".join(block.text.casefold().split()))
            if (
                key
                and len(block.text) <= 180
                and "continued" not in key
                and not _looks_like_table_note(block.text)
            ):
                repeated[key].append(block)
    for candidates in repeated.values():
        if len({block.page_number for block in candidates}) < min_pages:
            continue
        sentence_note = all(_TERMINAL_RE.search(block.text.rstrip()) for block in candidates)
        varying_numbered_template = len({
            " ".join(block.text.casefold().split()) for block in candidates
        }) > 1 and all(re.search(r"\d", block.text) for block in candidates)
        numbered_identifier = all(
            re.search(r"\d", block.text) and len(block.text.split()) >= 2 for block in candidates
        )
        if not sentence_note and not varying_numbered_template and not numbered_identifier:
            continue
        if all(
            sum(other.reading_bbox[1] > block.reading_bbox[1] for other in pages[block.page_number]) <= 2
            for block in candidates
        ):
            accepted.update((id(block), "running_footer") for block in candidates)
    return accepted


def recurring_margin_candidates(
    blocks: tuple[TextBlock, ...] | list[TextBlock], *, min_pages: int = 2, margin_fraction: float = 0.12
) -> tuple[TextBlock, ...]:
    """Find normalized strings repeated in a consistent top or bottom margin."""
    if min_pages < 2:
        raise ValueError("min_pages must be at least two")
    if not 0 < margin_fraction < 0.5:
        raise ValueError("margin_fraction must be between zero and one half")
    roles = _recurring_margin_roles(
        blocks,
        min_pages=min_pages,
        margin_fraction=margin_fraction,
    )
    return tuple(block for block in column_aware_reading_order(blocks) if id(block) in roles)


def _split_disconnected_directional_blocks(blocks: Sequence[TextBlock]) -> list[TextBlock]:
    """Split native blocks that incorrectly group separate rotated labels."""
    result: list[TextBlock] = []
    for block in blocks:
        groups: list[list[TextLine]] = []
        for line in block.lines:
            vertical = abs(line.direction[1]) >= 0.9 and abs(line.direction[0]) <= 0.1
            if groups and vertical:
                previous = groups[-1][-1]
                previous_vertical = abs(previous.direction[1]) >= 0.9 and abs(previous.direction[0]) <= 0.1
                overlap = max(0.0, min(previous.bbox[3], line.bbox[3]) - max(previous.bbox[1], line.bbox[1]))
                shorter = min(previous.bbox[3] - previous.bbox[1], line.bbox[3] - line.bbox[1])
                if previous_vertical and overlap / shorter < 0.5:
                    groups.append([line])
                    continue
            if groups:
                groups[-1].append(line)
            else:
                groups.append([line])
        for lines in groups:
            if len(lines) == len(block.lines):
                result.append(block)
                continue
            bboxes = tuple(line.bbox for line in lines)
            result.append(
                TextBlock(
                    lines=tuple(lines),
                    bbox=(
                        min(bbox[0] for bbox in bboxes),
                        min(bbox[1] for bbox in bboxes),
                        max(bbox[2] for bbox in bboxes),
                        max(bbox[3] for bbox in bboxes),
                    ),
                    page_number=block.page_number,
                    page_width=block.page_width,
                    page_height=block.page_height,
                    rotation=block.rotation,
                )
            )
    return result


def _consolidate_adjacent_directional_blocks(blocks: Sequence[TextBlock]) -> list[TextBlock]:
    """Join adjacent native line blocks that form one rotated multiline label."""
    candidates = [
        block
        for block in blocks
        if len(block.lines) == 1
        and abs(block.lines[0].direction[1]) >= 0.9
        and abs(block.lines[0].direction[0]) <= 0.1
    ]
    neighbors: dict[int, set[int]] = {id(block): set() for block in candidates}
    by_id = {id(block): block for block in candidates}
    for left, right in combinations(candidates, 2):
        if left.page_number != right.page_number:
            continue
        left_line, right_line = left.lines[0], right.lines[0]
        if left_line.direction[1] * right_line.direction[1] < 0.9:
            continue
        if any(_PARENTHETICAL_QUALIFIER_RE.fullmatch(candidate.text.strip()) for candidate in (left, right)):
            continue
        overlap = max(0.0, min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]))
        shorter = min(left.bbox[3] - left.bbox[1], right.bbox[3] - right.bbox[1])
        left_width = left.bbox[2] - left.bbox[0]
        right_width = right.bbox[2] - right.bbox[0]
        center_gap = abs((left.bbox[0] + left.bbox[2] - right.bbox[0] - right.bbox[2]) / 2)
        if (
            overlap / shorter >= 0.8
            and center_gap <= max(left_width, right_width) * 1.25
            and abs(left.font_size - right.font_size) <= max(0.75, left.font_size * 0.08)
            and left.is_bold == right.is_bold
        ):
            neighbors[id(left)].add(id(right))
            neighbors[id(right)].add(id(left))

    components: list[list[TextBlock]] = []
    remaining = set(neighbors)
    while remaining:
        seed = remaining.pop()
        member_ids = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            additions = neighbors[current] - member_ids
            member_ids.update(additions)
            frontier.extend(additions)
            remaining.difference_update(additions)
        if len(member_ids) > 1:
            components.append([by_id[block_id] for block_id in member_ids])

    replacement_by_id: dict[int, TextBlock] = {}
    consumed: set[int] = set()
    indices = {id(block): index for index, block in enumerate(blocks)}
    for component in components:
        reverse = component[0].lines[0].direction[1] > 0
        ordered = sorted(component, key=lambda block: block.bbox[0], reverse=reverse)
        lines = tuple(line for block in ordered for line in block.lines)
        bboxes = tuple(block.bbox for block in ordered)
        merged = TextBlock(
            lines=lines,
            bbox=(
                min(bbox[0] for bbox in bboxes),
                min(bbox[1] for bbox in bboxes),
                max(bbox[2] for bbox in bboxes),
                max(bbox[3] for bbox in bboxes),
            ),
            page_number=ordered[0].page_number,
            page_width=ordered[0].page_width,
            page_height=ordered[0].page_height,
            rotation=ordered[0].rotation,
        )
        first_id = id(min(component, key=lambda block: indices[id(block)]))
        replacement_by_id[first_id] = merged
        consumed.update(id(block) for block in component)

    return [
        replacement_by_id[id(block)] if id(block) in replacement_by_id else block
        for block in blocks
        if id(block) not in consumed or id(block) in replacement_by_id
    ]


def classify_semantic_blocks(
    blocks: tuple[TextBlock, ...] | list[TextBlock],
    *,
    body_font_size: float | None = None,
    recurring_min_pages: int = 2,
    recurring_margin_fraction: float = 0.12,
    page_column_splits: Mapping[int, tuple[float, ...]] | None = None,
    page_atomic_bboxes: Mapping[int, Sequence[BBox]] | None = None,
) -> tuple[SemanticBlock, ...]:
    """Classify strong semantic signals, then conservatively join remaining prose."""
    if recurring_min_pages < 2:
        raise ValueError("recurring_min_pages must be at least two")
    if not 0 < recurring_margin_fraction < 0.5:
        raise ValueError("recurring_margin_fraction must be between zero and one half")
    directional_blocks = _split_compound_footnote_blocks(
        _consolidate_adjacent_directional_blocks(_split_disconnected_directional_blocks(blocks))
    )
    numbered_blocks, page_number_ids, page_number_companion_ids = _split_recurring_page_numbers(
        directional_blocks,
        min_pages=recurring_min_pages,
        margin_fraction=recurring_margin_fraction,
    )
    numbered_blocks, validated_companion_ids = _split_validated_footer_companions(
        numbered_blocks,
        page_number_ids=page_number_ids,
        min_pages=recurring_min_pages,
        margin_fraction=recurring_margin_fraction,
    )
    split_blocks, split_margin_ids = _split_recurring_margin_spans(
        numbered_blocks,
        min_pages=recurring_min_pages,
        margin_fraction=recurring_margin_fraction,
    )
    split_blocks = _split_heading_style_barriers(
        split_blocks,
        protected_ids={
            *page_number_ids,
            *page_number_companion_ids,
            *validated_companion_ids,
            *split_margin_ids,
        },
    )
    ordered = column_aware_reading_order(split_blocks, page_column_splits=page_column_splits)
    if not ordered:
        return ()
    effective_body_size = body_font_size if body_font_size is not None else infer_body_font_size(ordered)
    margin_roles = _recurring_margin_roles(
        ordered,
        min_pages=recurring_min_pages,
        margin_fraction=recurring_margin_fraction,
        split_margin_ids=split_margin_ids,
        page_number_companion_ids=page_number_companion_ids,
        validated_companion_ids=validated_companion_ids,
    )
    exact_margin_pages: dict[tuple[str, MarginRole], set[int]] = defaultdict(set)
    for block in ordered:
        role = margin_roles.get(id(block))
        if role is not None:
            exact_text = " ".join(block.text.casefold().split())
            exact_margin_pages[(exact_text, role)].add(block.page_number)
    available_pages = {candidate.page_number for candidate in ordered}
    exact_margin_ids: set[int] = set()
    for block in ordered:
        role = margin_roles.get(id(block))
        if role is None:
            continue
        key = (" ".join(block.text.casefold().split()), role)
        if _has_recurrence_support(exact_margin_pages[key], available_pages, recurring_min_pages):
            exact_margin_ids.add(id(block))
    result: list[SemanticBlock] = []
    pending: list[TextBlock] = []

    def flush_pending() -> None:
        if not pending:
            return
        for semantic in join_paragraphs(pending, page_column_splits=page_column_splits):
            if result and _is_list_item_continuation(result[-1], semantic):
                semantic = replace(semantic, paragraph_role="list_item_continuation")
            result.append(semantic)
        pending.clear()

    for block in ordered:
        kind: SemanticKind | None = None
        level: int | None = None
        paragraph_role: str | None = None
        margin_role: MarginRole | None = None
        if id(block) in page_number_ids:
            kind = "page_number"
            margin_role = "page_number"
        elif is_caption(block):
            kind = "caption"
        elif is_table_continuation_marker(block):
            kind = "note"
        elif is_footnote(block, effective_body_size):
            kind = "footnote"
        elif id(block) in margin_roles and (
            id(block) in exact_margin_ids
            or id(block) in page_number_companion_ids
            or id(block) in validated_companion_ids
            or not is_heading(block, effective_body_size)
        ):
            kind = "recurring_margin"
            margin_role = margin_roles[id(block)]
        elif _is_heading_subtitle(block, result[-1] if result and not pending else None):
            kind = "paragraph"
            paragraph_role = "subtitle"
        elif is_heading(block, effective_body_size):
            kind = "heading"
            level = heading_level(block, effective_body_size)
        if kind is None:
            pending.append(block)
        else:
            flush_pending()
            result.append(
                SemanticBlock(
                    kind=kind,
                    text=block.text,
                    source_blocks=(block,),
                    heading_level=level,
                    paragraph_role=paragraph_role,
                    margin_role=margin_role,
                )
            )
    flush_pending()
    with_footnote_continuations = _merge_cross_page_footnote_continuations(result)
    return _merge_cross_page_paragraphs(with_footnote_continuations, page_column_splits, page_atomic_bboxes)


@dataclass(frozen=True, slots=True)
class _FootnoteMarker:
    label: str
    text: str
    previous_text: str
    next_text: str


def _normalize_footnote_label(label: str) -> str:
    return label.translate(_SUPERSCRIPT_DIGITS)


def _footnote_label(text: str) -> str | None:
    match = _FOOTNOTE_PREFIX_RE.match(text.strip())
    if match is None:
        return None
    raw_label = match.group("parenthesized") or match.group("plain")
    return _normalize_footnote_label(raw_label)


def _span_center(span: TextSpan) -> tuple[float, float]:
    return (span.bbox[0] + span.bbox[2]) / 2, (span.bbox[1] + span.bbox[3]) / 2


def _bbox_distance(first: BBox, second: BBox) -> float:
    x_gap = max(first[0] - second[2], second[0] - first[2], 0.0)
    y_gap = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(x_gap, y_gap)


def _is_superscript_neighbor(marker: TextSpan, neighbor: TextSpan, line: TextLine) -> bool:
    """Compare script position in the line's coordinate frame, not native page axes."""
    direction_length = math.hypot(*line.direction)
    baseline_x = line.direction[0] / direction_length
    baseline_y = line.direction[1] / direction_length
    # A clockwise page coordinate system makes the reading-up normal (dy, -dx).
    up_x, up_y = baseline_y, -baseline_x
    marker_center = _span_center(marker)
    neighbor_center = _span_center(neighbor)
    upward_offset = (marker_center[0] - neighbor_center[0]) * up_x + (
        marker_center[1] - neighbor_center[1]
    ) * up_y
    return bool(
        marker.font_size <= neighbor.font_size * 0.82
        and upward_offset >= neighbor.font_size * 0.05
        and _bbox_distance(marker.bbox, neighbor.bbox) <= neighbor.font_size * 0.6
    )


def _has_math_notation_context(semantic: SemanticBlock) -> bool:
    return re.search(r"[=≈≠≤≥<>±∑∫∂λγΩΩω∥]", semantic.text) is not None


def _looks_like_formula_script(marker: _FootnoteMarker) -> bool:
    previous = marker.previous_text.strip()
    following = marker.next_text.strip()
    if not previous or len(previous) > 3 or re.fullmatch(r"[A-Za-z0-9_ΩΩωλγ∥∂]+", previous) is None:
        return False
    return not following or re.match(r"[A-Za-z0-9_+−=()[\]{}]", following) is not None


def _semantic_reference_markers(semantic: SemanticBlock) -> tuple[_FootnoteMarker, ...]:
    markers: list[_FootnoteMarker] = []
    for block in semantic.source_blocks:
        for line in block.lines:
            spans = line.spans
            for index, span in enumerate(spans):
                stripped = span.text.strip()
                if re.fullmatch(_FOOTNOTE_LABEL_PATTERN, stripped) is not None:
                    previous = next(
                        (candidate for candidate in reversed(spans[:index]) if candidate.text.strip()),
                        None,
                    )
                    following = next(
                        (candidate for candidate in spans[index + 1 :] if candidate.text.strip()),
                        None,
                    )
                    regular_neighbors = [
                        candidate
                        for candidate in (previous, following)
                        if candidate is not None and candidate.font_size >= span.font_size / 0.82
                    ]
                    if regular_neighbors:
                        nearest = min(
                            regular_neighbors,
                            key=lambda candidate: _bbox_distance(span.bbox, candidate.bbox),
                        )
                        if _is_superscript_neighbor(span, nearest, line):
                            markers.append(
                                _FootnoteMarker(
                                    label=_normalize_footnote_label(stripped),
                                    text=stripped,
                                    previous_text="" if previous is None else previous.text.rstrip(),
                                    next_text="" if following is None else following.text.lstrip(),
                                )
                            )
                for match in _UNICODE_REFERENCE_RE.finditer(span.text):
                    markers.append(
                        _FootnoteMarker(
                            label=_normalize_footnote_label(match.group("label")),
                            text=match.group("label"),
                            previous_text=span.text[: match.start()],
                            next_text=span.text[match.end() :],
                        )
                    )
    return tuple(dict.fromkeys(markers))


def _replace_reference_marker(content: str, marker: _FootnoteMarker) -> str:
    replacement = f"[^{marker.label}]"
    if marker.text not in content:
        return content
    before = marker.previous_text[-24:]
    after = marker.next_text[:24]
    pattern = re.compile(
        (re.escape(before) + r"\s*" if before else "")
        + f"(?P<marker>{re.escape(marker.text)})"
        + (r"\s*" + re.escape(after) if after else "")
    )
    match = pattern.search(content)
    if match is not None:
        start, end = match.span("marker")
        return content[:start] + replacement + content[end:]
    if content.count(marker.text) == 1:
        return content.replace(marker.text, replacement)
    return content


def link_footnotes(
    elements: list[DocumentElement],
    semantic_blocks: Sequence[SemanticBlock],
) -> list[DocumentElement]:
    """Reciprocally link only typographically explicit markers to nearby labeled notes."""
    if len(elements) != len(semantic_blocks):
        raise ValueError("semantic elements and blocks must have matching lengths")
    element_ids = [element.element_id for element in elements]
    if len(element_ids) != len(set(element_ids)):
        raise ValueError("footnote linking requires unique element IDs")

    notes_by_label: dict[str, list[int]] = defaultdict(list)
    labels_by_index: dict[int, str] = {}
    for index, (element, semantic) in enumerate(zip(elements, semantic_blocks, strict=True)):
        if element.element_type != "footnote" or semantic.kind != "footnote":
            continue
        label = _footnote_label(element.content)
        if label is not None:
            labels_by_index[index] = label
            notes_by_label[label].append(index)

    links_by_reference: dict[int, list[tuple[int, _FootnoteMarker]]] = defaultdict(list)
    for reference_index, (element, semantic) in enumerate(zip(elements, semantic_blocks, strict=True)):
        if element.element_type in {"footnote", "header", "footer", "note"}:
            continue
        reference_page = semantic.page_numbers[-1]
        for marker in _semantic_reference_markers(semantic):
            if _looks_like_formula_script(marker):
                continue
            candidates = notes_by_label.get(marker.label, [])
            same_page = [
                index for index in candidates if semantic_blocks[index].page_numbers[0] == reference_page
            ]
            if len(same_page) == 1:
                note_index = same_page[0]
            else:
                # Mathematical superscripts are common and can coincide with a note
                # label on an adjoining page. Cross-page association needs prose-like
                # context; same-page repeated equation references remain supported.
                if _has_math_notation_context(semantic):
                    continue
                nearby = [
                    index
                    for index in candidates
                    if abs(semantic_blocks[index].page_numbers[0] - reference_page) <= 1
                ]
                if len(nearby) != 1:
                    continue
                note_index = nearby[0]
            if _replace_reference_marker(element.content, marker) == element.content:
                continue
            pair = (note_index, marker)
            if pair not in links_by_reference[reference_index]:
                links_by_reference[reference_index].append(pair)

    references_by_note: dict[int, list[int]] = defaultdict(list)
    for reference_index, links in links_by_reference.items():
        for note_index, _marker in links:
            if reference_index not in references_by_note[note_index]:
                references_by_note[note_index].append(reference_index)

    result: list[DocumentElement] = []
    for index, element in enumerate(elements):
        structure = element.structure
        content = element.content
        linked_ids = list(structure.linked_element_ids)
        for note_index, marker in links_by_reference.get(index, []):
            note_id = elements[note_index].element_id
            if note_id not in linked_ids:
                linked_ids.append(note_id)
            content = _replace_reference_marker(content, marker)

        footnote = structure.footnote
        if footnote is not None and index in labels_by_index:
            reference_ids = [elements[item].element_id for item in references_by_note.get(index, [])]
            footnote = footnote.model_copy(
                update={
                    "label": labels_by_index[index],
                    "reference_element_ids": reference_ids,
                    "association_confident": bool(reference_ids),
                }
            )
            for reference_id in reference_ids:
                if reference_id not in linked_ids:
                    linked_ids.append(reference_id)

        updated_structure = ElementStructure.model_validate(
            structure.model_dump(mode="json")
            | {
                "footnote": None if footnote is None else footnote.model_dump(mode="json"),
                "linked_element_ids": linked_ids,
            }
        )
        result.append(element.model_copy(update={"content": content, "structure": updated_structure}))
    return result


def link_figure_captions(
    elements: list[DocumentElement],
    *,
    page_rotations: Mapping[int, int] | None = None,
) -> list[DocumentElement]:
    """Reciprocally link unambiguous Figure captions in page-reading coordinates."""
    typed_elements = list(elements)
    figures = [element for element in typed_elements if element.element_type == "figure"]
    captions = [
        element
        for element in typed_elements
        if element.element_type == "caption"
        and re.match(r"^fig(?:ure)?\.?\s+", element.content.strip(), re.IGNORECASE)
    ]
    possible_by_caption: dict[str, list[DocumentElement]] = defaultdict(list)
    possible_by_figure: dict[str, list[DocumentElement]] = defaultdict(list)
    for caption in captions:
        caption_fragment = caption.fragments[0]
        if caption_fragment.page_width is None or caption_fragment.page_height is None:
            raise ValueError("caption fragments require known page dimensions")
        reading_rotation = (
            0 if page_rotations is None else page_rotations.get(caption_fragment.page_number, 0)
        )
        caption_bbox = normalize_bbox(
            (
                caption_fragment.bbox.x0,
                caption_fragment.bbox.y0,
                caption_fragment.bbox.x1,
                caption_fragment.bbox.y1,
            ),
            caption_fragment.page_width,
            caption_fragment.page_height,
            -reading_rotation,
        )
        reading_page_height = normalize_page_size(
            caption_fragment.page_width,
            caption_fragment.page_height,
            -reading_rotation,
        )[1]
        for figure in figures:
            figure_fragment = figure.fragments[0]
            if figure_fragment.page_number != caption_fragment.page_number:
                continue
            figure_bbox = normalize_bbox(
                (
                    figure_fragment.bbox.x0,
                    figure_fragment.bbox.y0,
                    figure_fragment.bbox.x1,
                    figure_fragment.bbox.y1,
                ),
                caption_fragment.page_width,
                caption_fragment.page_height,
                -reading_rotation,
            )
            horizontal_overlap = max(
                0.0,
                min(figure_bbox[2], caption_bbox[2]) - max(figure_bbox[0], caption_bbox[0]),
            )
            narrower_width = min(
                figure_bbox[2] - figure_bbox[0],
                caption_bbox[2] - caption_bbox[0],
            )
            gap = caption_bbox[1] - figure_bbox[3]
            caption_height = caption_bbox[3] - caption_bbox[1]
            vertical_overlap = max(
                0.0,
                min(figure_bbox[3], caption_bbox[3]) - max(figure_bbox[1], caption_bbox[1]),
            )
            caption_area = (caption_bbox[2] - caption_bbox[0]) * caption_height
            overlap_area = horizontal_overlap * vertical_overlap
            caption_inside_lower_figure = (
                overlap_area / caption_area >= 0.8
                and caption_bbox[1] >= figure_bbox[1] + (figure_bbox[3] - figure_bbox[1]) * 0.35
            )
            caption_below_figure = (
                horizontal_overlap / narrower_width >= 0.5 and 0 <= gap <= reading_page_height * 0.04
            )
            if caption_below_figure or caption_inside_lower_figure:
                possible_by_caption[caption.element_id].append(figure)
                possible_by_figure[figure.element_id].append(caption)

    links = {
        (figures_for_caption[0].element_id, caption.element_id)
        for caption in captions
        for figures_for_caption in [possible_by_caption[caption.element_id]]
        if len(figures_for_caption) == 1 and len(possible_by_figure[figures_for_caption[0].element_id]) == 1
    }
    if not links:
        return typed_elements

    linked_by_id = {left: right for left, right in links} | {right: left for left, right in links}
    result: list[DocumentElement] = []
    for element in typed_elements:
        linked_id = linked_by_id.get(element.element_id)
        if linked_id is None:
            result.append(element)
            continue
        structure = element.structure
        figure = structure.figure
        if figure is not None:
            figure = figure.model_copy(update={"caption_element_id": linked_id})
        updated_structure = ElementStructure.model_validate(
            structure.model_dump(mode="json")
            | {
                "figure": None if figure is None else figure.model_dump(mode="json"),
                "linked_element_ids": [*structure.linked_element_ids, linked_id],
            }
        )
        result.append(element.model_copy(update={"structure": updated_structure}))
    return result


def infer_body_font_size(blocks: tuple[TextBlock, ...] | list[TextBlock]) -> float:
    if not blocks:
        raise ValueError("cannot infer body font size without blocks")
    weighted = Counter[float]()
    for block in blocks:
        for line in block.lines:
            for span in line.spans:
                weighted[round(span.font_size, 2)] += max(1, len(span.text.strip()))
    if not weighted:
        raise ValueError("cannot infer body font size from blank spans")
    return weighted.most_common(1)[0][0]


def _weighted_font_size(spans: tuple[TextSpan, ...]) -> float:
    values = [span.font_size for span in spans for _ in range(max(1, len(span.text.strip())))]
    return median(values)


def _starts_like_continuation(text: str) -> bool:
    stripped = text.lstrip()
    return bool(stripped) and (stripped[0].islower() or stripped[0].isdigit() or stripped[0] in ",;:)]")


def _starts_like_cross_page_continuation(text: str) -> bool:
    """Exclude numeric starts that can denote lists, folios, or index entries."""
    stripped = text.lstrip()
    return bool(stripped) and (stripped[0].islower() or stripped[0] in ",;:)]")


def _horizontal_overlap(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    return overlap / min(_bbox_width(left), _bbox_width(right))


def _normalized_recurrence_text(text: str, page_number: int) -> str:
    stripped = " ".join(text.split())
    normalized = stripped.casefold()
    page_token = re.compile(rf"(?<!\d){page_number}(?!\d)")
    normalized = page_token.sub("#", normalized)
    is_language_marker = re.fullmatch(r"[A-Z]{2}", stripped) is not None
    return normalized if len(normalized) >= 3 or is_language_marker else ""
