from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal, TypeAlias

BBox: TypeAlias = tuple[float, float, float, float]

_PANEL_KINDS = frozenset({"caption", "figure", "table"})
_TOP_KINDS = frozenset({"header"})
_BOTTOM_KINDS = frozenset({"footer", "page_number"})


@dataclass(frozen=True, slots=True)
class ReadingRegion:
    """One indivisible page region whose internal/native order is already settled."""

    region_id: str
    bbox: BBox
    kind: str
    native_index: int
    page_width: float
    page_height: float

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("reading region ID cannot be empty")
        if not self.kind:
            raise ValueError("reading region kind cannot be empty")
        if type(self.native_index) is not int or self.native_index < 0:
            raise ValueError("reading region native_index must be a non-negative integer")
        if len(self.bbox) != 4 or not all(type(value) in (int, float) for value in self.bbox):
            raise ValueError("reading region bbox must contain four numeric non-boolean coordinates")
        if not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("reading region bbox must contain four finite coordinates")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("reading region bbox must have positive width and height")
        if type(self.page_width) not in (int, float) or type(self.page_height) not in (int, float):
            raise ValueError("reading region page dimensions must be numeric and non-boolean")
        if not math.isfinite(self.page_width) or not math.isfinite(self.page_height):
            raise ValueError("reading region page dimensions must be finite")
        if self.page_width <= 0 or self.page_height <= 0:
            raise ValueError("reading region page dimensions must be positive")


ReadingOrderStatus: TypeAlias = Literal["ordered", "native_fallback"]


@dataclass(frozen=True, slots=True)
class ReadingOrderDecision:
    regions: tuple[ReadingRegion, ...]
    status: ReadingOrderStatus


def order_reading_regions(
    regions: tuple[ReadingRegion, ...] | list[ReadingRegion],
    *,
    column_splits: tuple[float, ...],
    rotation: int = 0,
) -> tuple[ReadingRegion, ...]:
    return decide_reading_region_order(
        regions,
        column_splits=column_splits,
        rotation=rotation,
    ).regions


def decide_reading_region_order(
    regions: tuple[ReadingRegion, ...] | list[ReadingRegion],
    *,
    column_splits: tuple[float, ...],
    rotation: int = 0,
) -> ReadingOrderDecision:
    """Order unambiguous page regions with a conservative precedence graph.

    Region payloads are never split or rewritten. A single body track retains native
    region order. Proven columns use column-major flow, while an all-panel band uses
    visual row-major flow. Full-width regions interrupt and restart those flows.
    Any geometric overlap that would require guessing fails closed to the input order.
    """
    items = tuple(regions)
    if len(items) < 2:
        return ReadingOrderDecision(items, "ordered")
    if type(rotation) is not int or rotation % 90 != 0:
        raise ValueError("reading rotation must be an integer multiple of 90 degrees")
    if len({item.region_id for item in items}) != len(items):
        raise ValueError("reading region IDs must be unique")
    # Adapters canonicalize page geometry before constructing regions. Tolerate a
    # minority of legacy/mixed-direction extents here rather than making ordering fail.
    page_size_counts = Counter((item.page_width, item.page_height) for item in items)
    page_width, _ = max(
        page_size_counts,
        key=lambda size: (
            page_size_counts[size],
            size[0] * size[1],
            size[1],
            -size[0],
        ),
    )
    if tuple(sorted(set(column_splits))) != column_splits or any(
        type(split) not in (int, float) or not math.isfinite(split) or not 0 < split < page_width
        for split in column_splits
    ):
        raise ValueError("column splits must be finite, unique, sorted, and inside the page")

    children_by_container: dict[str, list[ReadingRegion]] = {}
    container_by_child: dict[str, str] = {}
    unexplained_overlaps: list[tuple[ReadingRegion, ReadingRegion]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if not _boxes_overlap(left.bbox, right.bbox):
                continue
            relation = _expected_containment(left, right)
            if relation is None:
                unexplained_overlaps.append((left, right))
                continue
            container, child = relation
            previous_container = container_by_child.setdefault(child.region_id, container.region_id)
            if previous_container != container.region_id:
                return ReadingOrderDecision(items, "native_fallback")
            children_by_container.setdefault(container.region_id, []).append(child)
    if any(
        container_by_child.get(left.region_id) != container_by_child.get(right.region_id)
        or left.region_id not in container_by_child
        for left, right in unexplained_overlaps
    ):
        return ReadingOrderDecision(items, "native_fallback")

    contained_ids = set(container_by_child)
    layout_items = [item for item in items if item.region_id not in contained_ids]
    top = [item for item in layout_items if item.kind in _TOP_KINDS]
    bottom = [item for item in layout_items if item.kind in _BOTTOM_KINDS]
    body = [item for item in layout_items if item.kind not in _TOP_KINDS | _BOTTOM_KINDS]

    ordered_top = _order_furniture_rows(top)
    ordered_bottom = _order_furniture_rows(bottom)
    ordered_body, body_fallback = _order_body(body, column_splits, page_width)
    if body_fallback:
        return ReadingOrderDecision(items, "native_fallback")
    layout_order = [*ordered_top, *ordered_body, *ordered_bottom]
    desired = [
        member
        for item in layout_order
        for member in (
            item,
            *sorted(children_by_container.get(item.region_id, ()), key=_geometry_key),
        )
    ]

    # Materialize the region sequence as a DAG. Keeping graph construction here makes
    # precedence explicit and leaves room for independent attachment edges without
    # coupling ordering to document rendering.
    adjacency: dict[str, set[str]] = {item.region_id: set() for item in items}
    indegree = {item.region_id: 0 for item in items}
    for left, right in zip(desired, desired[1:], strict=False):
        if right.region_id not in adjacency[left.region_id]:
            adjacency[left.region_id].add(right.region_id)
            indegree[right.region_id] += 1
    by_id = {item.region_id: item for item in items}
    rank = {item.region_id: index for index, item in enumerate(desired)}
    ready = [(rank[region_id], region_id) for region_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    result: list[ReadingRegion] = []
    while ready:
        _, region_id = heapq.heappop(ready)
        result.append(by_id[region_id])
        for successor in adjacency[region_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, (rank[successor], successor))
    if len(result) != len(items):
        return ReadingOrderDecision(items, "native_fallback")
    return ReadingOrderDecision(tuple(result), "ordered")


def _order_body(
    regions: list[ReadingRegion], column_splits: tuple[float, ...], page_width: float
) -> tuple[list[ReadingRegion], bool]:
    if len(regions) < 2:
        return regions.copy(), False
    if not column_splits:
        if regions and all(item.kind in _PANEL_KINDS for item in regions):
            return _order_panel_rows(regions), False
        if any(item.kind in _PANEL_KINDS | {"heading"} for item in regions):
            return sorted(regions, key=_geometry_key), False
        return sorted(regions, key=lambda item: (item.native_index, item.region_id)), False

    separators = sorted(
        [
            item
            for item in regions
            if _width(item.bbox) >= page_width * 0.72 or _crosses_split(item.bbox, column_splits)
        ],
        key=_geometry_key,
    )
    regular = [item for item in regions if item not in separators]
    if any(
        not (left.bbox[3] <= right.bbox[1]) for left, right in zip(separators, separators[1:], strict=False)
    ):
        return regions.copy(), True

    bands: list[list[ReadingRegion]] = [[] for _ in range(len(separators) + 1)]
    for item in regular:
        if any(
            separator.bbox[1] < item.bbox[3] and item.bbox[1] < separator.bbox[3] for separator in separators
        ):
            return regions.copy(), True
        band_index = sum(separator.bbox[3] <= item.bbox[1] for separator in separators)
        bands[band_index].append(item)

    result: list[ReadingRegion] = []
    for index, band in enumerate(bands):
        result.extend(_order_band(band, column_splits))
        if index < len(separators):
            result.append(separators[index])
    return result, False


def _order_band(regions: list[ReadingRegion], column_splits: tuple[float, ...]) -> list[ReadingRegion]:
    if len(regions) < 2:
        return regions.copy()
    if regions and all(item.kind in _PANEL_KINDS for item in regions):
        return _order_panel_rows(regions)

    columns: list[list[ReadingRegion]] = [[] for _ in range(len(column_splits) + 1)]
    for item in regions:
        column = sum(_center_x(item.bbox) >= split for split in column_splits)
        columns[column].append(item)
    return [item for column in columns for item in sorted(column, key=_geometry_key)]


def _order_furniture_rows(regions: list[ReadingRegion]) -> list[ReadingRegion]:
    rows: list[list[ReadingRegion]] = []
    for item in sorted(regions, key=_geometry_key):
        if not rows or item.bbox[1] >= max(candidate.bbox[3] for candidate in rows[-1]):
            rows.append([item])
        else:
            rows[-1].append(item)
    return [
        item
        for row in rows
        for item in sorted(
            row, key=lambda candidate: (candidate.bbox[0], candidate.bbox[1], candidate.region_id)
        )
    ]


def _order_panel_rows(regions: list[ReadingRegion]) -> list[ReadingRegion]:
    rows: list[list[ReadingRegion]] = []
    for item in sorted(regions, key=_geometry_key):
        if not rows:
            rows.append([item])
            continue
        row_top = min(candidate.bbox[1] for candidate in rows[-1])
        row_bottom = max(candidate.bbox[3] for candidate in rows[-1])
        overlap = min(row_bottom, item.bbox[3]) - max(row_top, item.bbox[1])
        shorter = min(row_bottom - row_top, item.bbox[3] - item.bbox[1])
        if overlap > 0 and overlap / shorter >= 0.5:
            rows[-1].append(item)
        else:
            rows.append([item])
    return [
        item
        for row in rows
        for item in sorted(
            row, key=lambda candidate: (candidate.bbox[0], candidate.bbox[1], candidate.region_id)
        )
    ]


def _expected_containment(
    first: ReadingRegion, second: ReadingRegion
) -> tuple[ReadingRegion, ReadingRegion] | None:
    for container, child in ((first, second), (second, first)):
        if (
            container.kind in {"figure", "table"}
            and child.kind not in {"figure", "table", "header", "footer", "page_number"}
            and _contains(container.bbox, child.bbox)
        ):
            return container, child
    return None


def _contains(container: BBox, child: BBox) -> bool:
    intersection_width = max(0.0, min(container[2], child[2]) - max(container[0], child[0]))
    intersection_height = max(0.0, min(container[3], child[3]) - max(container[1], child[1]))
    child_area = _width(child) * (child[3] - child[1])
    container_area = _width(container) * (container[3] - container[1])
    return bool(
        intersection_width * intersection_height / child_area >= 0.98 and container_area >= child_area * 1.05
    )


def _boxes_overlap(first: BBox, second: BBox) -> bool:
    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(
        first[1], second[1]
    )


def _crosses_split(bbox: BBox, splits: tuple[float, ...]) -> bool:
    return any(bbox[0] < split < bbox[2] for split in splits)


def _width(bbox: BBox) -> float:
    return bbox[2] - bbox[0]


def _center_x(bbox: BBox) -> float:
    return (bbox[0] + bbox[2]) / 2


def _geometry_key(item: ReadingRegion) -> tuple[float, float, float, float, str]:
    return item.bbox[1], item.bbox[0], item.bbox[3], item.bbox[2], item.region_id
