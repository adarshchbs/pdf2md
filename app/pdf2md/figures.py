from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeGuard, cast

import pymupdf

from app.pdf2md.pymupdf_geometry import unrotated_page_extent
from app.pdf2md.pymupdf_runtime import pymupdf_session
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FigureStructure,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
)
from app.pdf2md.semantic_text import BBox, normalize_bbox, normalize_page_size

_MIN_WIDTH_FRACTION = 0.08
_MIN_HEIGHT_FRACTION = 0.05
_MIN_AREA_FRACTION = 0.006
_MAX_AREA_FRACTION = 0.82
_TABLE_OVERLAP_THRESHOLD = 0.2
_VECTOR_JOIN_GAP = 8.0


@dataclass(frozen=True, slots=True)
class FigureCandidate:
    page_number: int
    native_bbox: BBox
    page_width: float
    page_height: float
    rotation: int
    asset_path: str
    sha256: str
    width: int
    height: int
    evidence_kind: str

    @property
    def normalized_bbox(self) -> BBox:
        return normalize_bbox(
            self.native_bbox,
            self.page_width,
            self.page_height,
            self.rotation,
        )

    @property
    def normalized_page_size(self) -> tuple[float, float]:
        return normalize_page_size(self.page_width, self.page_height, self.rotation)


def extract_page_figure_candidates(
    page: pymupdf.Page,
    table_bboxes: Sequence[BoundingBox],
    *,
    figure_caption_bboxes: Sequence[BBox] = (),
) -> list[FigureCandidate]:
    """Extract conservative raster and strongly bounded vector figure references."""
    with pymupdf_session():
        return _extract_page_figure_candidates(
            page,
            table_bboxes,
            figure_caption_bboxes=figure_caption_bboxes,
        )


def _extract_page_figure_candidates(
    page: pymupdf.Page,
    table_bboxes: Sequence[BoundingBox],
    *,
    figure_caption_bboxes: Sequence[BBox],
) -> list[FigureCandidate]:
    page_index = page.number
    if page_index is None:
        raise ValueError("page must belong to an open PyMuPDF document")
    page_number = page_index + 1
    # PyMuPDF text, image, and drawing coordinates are relative to the crop box,
    # even when the media box has a different size or origin.
    page_width, page_height = unrotated_page_extent(page)
    rotation = int(page.rotation)
    output_width, output_height = normalize_page_size(page_width, page_height, rotation)
    native_tables = [
        normalize_bbox(_bbox_tuple(bbox), output_width, output_height, -rotation) for bbox in table_bboxes
    ]
    payload = cast(Mapping[str, object], page.get_text("dict", sort=False))
    raster = raster_candidates_from_pymupdf_dict(
        payload,
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        rotation=rotation,
        table_bboxes=native_tables,
        figure_caption_bboxes=figure_caption_bboxes,
        ignore_degenerate_images=True,
    )
    drawings = page.get_drawings()
    vector = vector_candidates_from_drawings(
        cast(Sequence[Mapping[str, object]], drawings),
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        rotation=rotation,
        table_bboxes=native_tables,
        occupied_bboxes=[candidate.native_bbox for candidate in raster],
        figure_caption_bboxes=figure_caption_bboxes,
    )
    return _reconcile_raster_vector_candidates(raster, vector)


def raster_candidates_from_pymupdf_dict(
    payload: Mapping[str, object],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    rotation: int,
    table_bboxes: Sequence[BBox] = (),
    figure_caption_bboxes: Sequence[BBox] = (),
    ignore_degenerate_images: bool = False,
) -> list[FigureCandidate]:
    _validate_page_inputs(page_number, page_width, page_height, rotation)
    raw_blocks = payload.get("blocks")
    if not _is_payload_sequence(raw_blocks):
        raise TypeError("PyMuPDF dict payload must contain a blocks sequence")

    candidates: list[FigureCandidate] = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, Mapping):
            raise TypeError("each PyMuPDF block must be a mapping")
        block_type = raw_block.get("type")
        if type(block_type) is not int:
            raise TypeError("PyMuPDF block type must be an integer")
        if block_type != 1:
            continue
        width = raw_block.get("width")
        height = raw_block.get("height")
        if type(width) is not int or type(height) is not int:
            raise TypeError("PyMuPDF image width and height must be integers and not booleans")
        if width < 1 or height < 1:
            if ignore_degenerate_images:
                continue
            raise ValueError("PyMuPDF image width and height must be positive")
        image = raw_block.get("image")
        if not isinstance(image, bytes):
            raise TypeError("PyMuPDF image payload must be bytes")
        if not image:
            raise ValueError("PyMuPDF image payload must not be empty")
        bbox = _image_bbox(
            raw_block.get("bbox"),
            page_width,
            page_height,
            ignore_degenerate=ignore_degenerate_images,
        )
        if bbox is None:
            continue
        digest = hashlib.sha256(image).hexdigest()
        caption_confirms_figure = any(
            _is_tightly_adjacent_caption(bbox, caption, page_height) for caption in figure_caption_bboxes
        )
        if (
            not _eligible_region(
                bbox,
                page_width,
                page_height,
                table_bboxes,
                figure_caption_bboxes,
            )
            or (_is_edge_bleed(bbox, page_width, page_height) and not caption_confirms_figure)
            or (
                _is_uncaptioned_raster_ornament(bbox, page_width, page_height) and not caption_confirms_figure
            )
        ):
            continue
        candidates.append(
            FigureCandidate(
                page_number=page_number,
                native_bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                rotation=rotation,
                asset_path=f"pdf://page/{page_number}/image/{digest}",
                sha256=digest,
                width=width,
                height=height,
                evidence_kind="raster_image_block",
            )
        )
    return _reject_raster_fragments(candidates)


def vector_candidates_from_drawings(
    drawings: Sequence[object],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    rotation: int,
    table_bboxes: Sequence[BBox] = (),
    occupied_bboxes: Sequence[BBox] = (),
    figure_caption_bboxes: Sequence[BBox] = (),
) -> list[FigureCandidate]:
    """Find only dense, two-dimensional drawing clusters with an explicit boundary."""
    _validate_page_inputs(page_number, page_width, page_height, rotation)
    records: list[tuple[BBox, int, bool]] = []
    for drawing in drawings:
        if not isinstance(drawing, Mapping):
            raise TypeError("each PyMuPDF drawing must be a mapping")
        rect = _rect_bbox(drawing.get("rect"), "drawing rect")
        items = drawing.get("items")
        if not _is_payload_sequence(items):
            raise TypeError("PyMuPDF drawing items must be a sequence")
        item_count = len(items)
        has_boundary = any(_is_payload_sequence(item) and len(item) > 0 and item[0] == "re" for item in items)
        records.append((rect, item_count, has_boundary))

    clusters: list[list[tuple[BBox, int, bool]]] = []
    for record in records:
        touching = [cluster for cluster in clusters if any(_near(record[0], item[0]) for item in cluster)]
        if not touching:
            clusters.append([record])
            continue
        merged = [record]
        for cluster in touching:
            merged.extend(cluster)
            clusters.remove(cluster)
        clusters.append(merged)

    result: list[FigureCandidate] = []
    for cluster in clusters:
        bbox = _union(item[0] for item in cluster)
        item_count = sum(item[1] for item in cluster)
        boundary_count = sum(item[2] for item in cluster)
        bounded = any(item[2] and _bbox_iou(item[0], bbox) >= 0.8 for item in cluster) or (
            len(cluster) >= 10 and boundary_count >= 3
        )
        two_dimensional = _cluster_has_two_dimensional_spread(cluster, bbox)
        caption_confirms_figure = any(
            _caption_supports_figure(bbox, caption, page_height) for caption in figure_caption_bboxes
        )
        if (
            item_count < 6
            or not bounded
            or not two_dimensional
            or not caption_confirms_figure
            or _is_page_spanning_edge_vector(bbox, page_width, page_height)
            or not _eligible_region(
                bbox,
                page_width,
                page_height,
                table_bboxes,
                figure_caption_bboxes,
            )
            or any(_overlap_fraction(bbox, occupied) >= 0.5 for occupied in occupied_bboxes)
        ):
            continue
        signature = _vector_signature(cluster, bbox)
        pixel_width = max(1, round(bbox[2] - bbox[0]))
        pixel_height = max(1, round(bbox[3] - bbox[1]))
        result.append(
            FigureCandidate(
                page_number=page_number,
                native_bbox=bbox,
                page_width=page_width,
                page_height=page_height,
                rotation=rotation,
                asset_path=f"pdf://page/{page_number}/vector/{signature}",
                sha256=signature,
                width=pixel_width,
                height=pixel_height,
                evidence_kind="bounded_vector_region",
            )
        )
    return result


def _reject_raster_fragments(candidates: Sequence[FigureCandidate]) -> list[FigureCandidate]:
    unique: list[FigureCandidate] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for candidate in candidates:
        key = (candidate.sha256, tuple(round(value, 3) for value in candidate.native_bbox))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    duplicate_placements = {
        index
        for index, candidate in enumerate(unique)
        if any(
            peer_index < index
            and candidate.sha256 == peer.sha256
            and _overlap_fraction(candidate.native_bbox, peer.native_bbox) >= 0.8
            and _overlap_fraction(peer.native_bbox, candidate.native_bbox) >= 0.8
            for peer_index, peer in enumerate(unique)
        )
    }
    ambiguous_layers = {
        index
        for index, candidate in enumerate(unique)
        for peer_index, peer in enumerate(unique)
        if index != peer_index
        and candidate.sha256 != peer.sha256
        and _overlap_fraction(candidate.native_bbox, peer.native_bbox) >= 0.8
        and _overlap_fraction(peer.native_bbox, candidate.native_bbox) >= 0.8
    }
    gallery_members: set[int] = set()
    for candidate in unique:
        width = candidate.native_bbox[2] - candidate.native_bbox[0]
        height = candidate.native_bbox[3] - candidate.native_bbox[1]
        peers = [
            peer_index
            for peer_index, peer in enumerate(unique)
            if abs((peer.native_bbox[2] - peer.native_bbox[0]) - width) <= width * 0.08
            and abs((peer.native_bbox[3] - peer.native_bbox[1]) - height) <= height * 0.08
            and (
                abs(peer.native_bbox[1] - candidate.native_bbox[1]) <= height * 0.12
                or abs(peer.native_bbox[0] - candidate.native_bbox[0]) <= width * 0.12
            )
        ]
        if len(peers) >= 3:
            gallery_members.update(peers)
    rejected = duplicate_placements | ambiguous_layers | gallery_members
    return [candidate for index, candidate in enumerate(unique) if index not in rejected]


def _reconcile_raster_vector_candidates(
    raster: Sequence[FigureCandidate], vector: Sequence[FigureCandidate]
) -> list[FigureCandidate]:
    raster_inside_vector: set[int] = set()
    for bounded_vector in vector:
        enclosed = [
            index
            for index, image in enumerate(raster)
            if _overlap_fraction(image.native_bbox, bounded_vector.native_bbox) >= 0.8
        ]
        if len(enclosed) >= 2:
            raster_inside_vector.update(enclosed)
    return [image for index, image in enumerate(raster) if index not in raster_inside_vector] + list(vector)


def reject_recurring_figures(
    candidates: Sequence[FigureCandidate], *, selected_page_count: int
) -> list[FigureCandidate]:
    """Reject repeated assets as logos or recurring ornaments, conservatively."""
    if selected_page_count < 0:
        raise ValueError("selected_page_count cannot be negative")
    pages_by_signature: dict[tuple[str, str], set[int]] = {}
    for candidate in candidates:
        pages_by_signature.setdefault((candidate.evidence_kind, candidate.sha256), set()).add(
            candidate.page_number
        )
    recurring = {signature for signature, pages in pages_by_signature.items() if len(pages) >= 2}
    return [
        candidate for candidate in candidates if (candidate.evidence_kind, candidate.sha256) not in recurring
    ]


def figure_elements(
    candidates: Sequence[FigureCandidate], document_id: str, annotator: str
) -> list[DocumentElement]:
    counts: Counter[int] = Counter()
    elements: list[DocumentElement] = []
    for candidate in candidates:
        page_width, page_height = candidate.normalized_page_size
        bbox = _clip_bbox(candidate.normalized_bbox, page_width, page_height)
        if bbox is None:
            continue
        counts[candidate.page_number] += 1
        elements.append(
            DocumentElement(
                document_id=document_id,
                element_id=f"page-{candidate.page_number}-figure-{counts[candidate.page_number]}",
                order=len(elements),
                element_type="figure",
                content="",
                format="text",
                fragments=[
                    PageFragment(
                        page_number=candidate.page_number,
                        page_width=page_width,
                        page_height=page_height,
                        bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                    )
                ],
                structure=ElementStructure(
                    figure=FigureStructure(
                        asset_path=candidate.asset_path,
                        sha256=candidate.sha256,
                        width=candidate.width,
                        height=candidate.height,
                    ),
                    properties=[
                        StructureProperty(
                            key="source_note",
                            value=(
                                f"page {candidate.page_number} PyMuPDF native "
                                f"{candidate.evidence_kind}; native bbox {candidate.native_bbox}; "
                                f"rotation {candidate.rotation} normalized to displayed coordinates"
                            ),
                        )
                    ],
                ),
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


_PANEL_HEADING_RE = re.compile(r"^\s*(?:\([a-z]\)|[a-z][.)])(?:\s|$)", re.IGNORECASE)


def classify_contained_figure_text(elements: Sequence[DocumentElement]) -> list[DocumentElement]:
    """Conservatively type and link native text wholly contained by one figure.

    An extractive figure is additive: native text remains output-visible. A
    confident caption link is required before containment can change text roles;
    an uncaptioned raster may instead be page artwork behind ordinary prose.
    Only an explicit alphabetic panel marker preserves heading semantics inside
    the figure; other heading-like chart labels become generic figure text.
    """
    figures_by_page: dict[int, list[DocumentElement]] = {}
    for element in elements:
        figure = element.structure.figure
        if (
            element.element_type == "figure"
            and len(element.fragments) == 1
            and figure is not None
            and figure.caption_element_id is not None
        ):
            figures_by_page.setdefault(element.fragments[0].page_number, []).append(element)

    result: list[DocumentElement] = []
    for element in elements:
        if element.element_type not in {"heading", "paragraph"} or len(element.fragments) != 1:
            result.append(element)
            continue
        fragment = element.fragments[0]
        child_bbox = _bbox_tuple(fragment.bbox)
        containers = [
            figure
            for figure in figures_by_page.get(fragment.page_number, [])
            if _overlap_fraction(child_bbox, _bbox_tuple(figure.fragments[0].bbox)) >= 0.95
        ]
        if len(containers) != 1:
            result.append(element)
            continue

        figure = containers[0]
        structure = element.structure
        linked_ids = list(structure.linked_element_ids)
        if figure.element_id not in linked_ids:
            linked_ids.append(figure.element_id)

        element_type = element.element_type
        paragraph = structure.paragraph
        if element_type == "heading" and _PANEL_HEADING_RE.match(element.content):
            heading_level = None if paragraph is None else paragraph.heading_level
            paragraph = ParagraphStructure(role="figure_panel_heading", heading_level=heading_level)
        elif element_type == "heading":
            element_type = "paragraph"
            paragraph = ParagraphStructure(role="figure_text")
        elif paragraph is not None and paragraph.role == "body":
            paragraph = ParagraphStructure(role="figure_text")

        updated_structure = structure.model_copy(
            update={"paragraph": paragraph, "linked_element_ids": linked_ids}
        )
        result.append(
            element.model_copy(update={"element_type": element_type, "structure": updated_structure})
        )
    return result


def order_linked_figure_content(elements: Sequence[DocumentElement]) -> list[DocumentElement]:
    """Move an internal caption only within a closed, contiguous figure component."""
    ordered = list(elements)
    elements_by_id: dict[str, list[DocumentElement]] = {}
    for element in ordered:
        elements_by_id.setdefault(element.element_id, []).append(element)
    figures = [element for element in ordered if element.element_type == "figure"]

    for figure in figures:
        figure_structure = figure.structure.figure
        if (
            figure_structure is None
            or figure_structure.caption_element_id is None
            or len(figure.fragments) != 1
            or len(elements_by_id.get(figure.element_id, [])) != 1
            or not _has_valid_bbox(figure)
        ):
            continue

        caption_id = figure_structure.caption_element_id
        caption_matches = elements_by_id.get(caption_id, [])
        if len(caption_matches) != 1:
            continue
        caption = caption_matches[0]
        if (
            caption.element_type != "caption"
            or len(caption.fragments) != 1
            or not _has_valid_bbox(caption)
            or len(figure.structure.linked_element_ids) != len(set(figure.structure.linked_element_ids))
            or figure.structure.linked_element_ids.count(caption_id) != 1
            or caption.structure.linked_element_ids != [figure.element_id]
        ):
            continue

        caption_claimants = {
            candidate.element_id
            for candidate in figures
            if candidate.structure.figure is not None
            and (
                candidate.structure.figure.caption_element_id == caption_id
                or caption_id in candidate.structure.linked_element_ids
            )
        }
        if caption_claimants != {figure.element_id}:
            continue

        linked_content = [
            element
            for element in ordered
            if element.element_id != caption_id and figure.element_id in element.structure.linked_element_ids
        ]
        if not linked_content or any(
            not _is_extractive_figure_child(element, figure.element_id)
            or len(elements_by_id.get(element.element_id, [])) != 1
            for element in linked_content
        ):
            continue
        if any(
            {
                candidate.element_id
                for candidate in figures
                if child.element_id in candidate.structure.linked_element_ids
            }
            - {figure.element_id}
            for child in linked_content
        ):
            continue

        forward_link_ids = set(figure.structure.linked_element_ids) - {caption_id}
        if forward_link_ids - {element.element_id for element in linked_content}:
            continue

        figure_page = figure.fragments[0].page_number
        figure_bbox = _bbox_tuple(figure.fragments[0].bbox)
        owned = {figure.element_id, caption_id, *(element.element_id for element in linked_content)}
        if any(
            len(element.fragments) != 1
            or not _has_valid_bbox(element)
            or element.fragments[0].page_number != figure_page
            or not _bbox_is_contained(_bbox_tuple(element.fragments[0].bbox), figure_bbox)
            for element in [caption, *linked_content]
        ):
            continue

        caption_index = next(index for index, element in enumerate(ordered) if element is caption)
        last_content_index = max(
            index
            for index, element in enumerate(ordered)
            if any(element is child for child in linked_content)
        )
        if caption_index >= last_content_index:
            continue
        if any(
            ordered[index].element_id not in owned for index in range(caption_index, last_content_index + 1)
        ):
            continue

        ordered.pop(caption_index)
        last_content_index -= 1
        ordered.insert(last_content_index + 1, caption)
    return ordered


def _is_extractive_figure_child(element: DocumentElement, figure_id: str) -> bool:
    paragraph = element.structure.paragraph
    if element.structure.linked_element_ids != [figure_id] or paragraph is None:
        return False
    return (element.element_type, paragraph.role) in {
        ("paragraph", "figure_text"),
        ("heading", "figure_panel_heading"),
    }


def _has_valid_bbox(element: DocumentElement) -> bool:
    if len(element.fragments) != 1:
        return False
    bbox = _bbox_tuple(element.fragments[0].bbox)
    return all(math.isfinite(value) for value in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _bbox_is_contained(inner: BBox, outer: BBox) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def overlaps_table(candidate: FigureCandidate, table_bbox: BoundingBox) -> bool:
    """Return whether a table occupies a material fraction of a figure candidate."""
    return _overlap_fraction(candidate.normalized_bbox, _bbox_tuple(table_bbox)) >= _TABLE_OVERLAP_THRESHOLD


def _is_edge_bleed(bbox: BBox, page_width: float, page_height: float) -> bool:
    area_fraction = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (page_width * page_height)
    tolerance = min(page_width, page_height) * 0.005
    touches_edge = (
        bbox[0] <= tolerance
        or bbox[1] <= tolerance
        or bbox[2] >= page_width - tolerance
        or bbox[3] >= page_height - tolerance
    )
    return area_fraction >= 0.1 and touches_edge


def _is_uncaptioned_raster_ornament(bbox: BBox, page_width: float, page_height: float) -> bool:
    """Identify only strongly elongated, low-area brand and decorative strips."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area_fraction = width * height / (page_width * page_height)
    elongation = max(width / height, height / width)
    return (elongation >= 3 and area_fraction <= 0.06) or (elongation >= 6 and area_fraction <= 0.1)


def _is_page_spanning_edge_vector(bbox: BBox, page_width: float, page_height: float) -> bool:
    """Reject over-joined vector clusters that absorb document-page boundaries."""
    area_fraction = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (page_width * page_height)
    tolerance = min(page_width, page_height) * 0.005
    touches_edge = (
        bbox[0] <= tolerance
        or bbox[1] <= tolerance
        or bbox[2] >= page_width - tolerance
        or bbox[3] >= page_height - tolerance
    )
    return area_fraction >= 0.5 and touches_edge


def _eligible_region(
    bbox: BBox,
    page_width: float,
    page_height: float,
    table_bboxes: Sequence[BBox],
    figure_caption_bboxes: Sequence[BBox],
) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area_fraction = width * height / (page_width * page_height)
    margin_ornament = area_fraction < 0.06 and (
        bbox[3] <= page_height * 0.18 or bbox[1] >= page_height * 0.82
    )
    overlaps_table_bbox = any(
        _overlap_fraction(bbox, table) >= _TABLE_OVERLAP_THRESHOLD for table in table_bboxes
    )
    caption_confirms_figure = any(
        _is_tightly_adjacent_caption(bbox, caption, page_height) for caption in figure_caption_bboxes
    )
    return (
        (not margin_ornament or caption_confirms_figure)
        and width >= page_width * _MIN_WIDTH_FRACTION
        and height >= page_height * _MIN_HEIGHT_FRACTION
        and _MIN_AREA_FRACTION <= area_fraction <= _MAX_AREA_FRACTION
        and (not overlaps_table_bbox or caption_confirms_figure)
    )


def _is_tightly_adjacent_caption(figure: BBox, caption: BBox, page_height: float) -> bool:
    horizontal_overlap = max(0.0, min(figure[2], caption[2]) - max(figure[0], caption[0]))
    narrower_width = min(figure[2] - figure[0], caption[2] - caption[0])
    if narrower_width <= 0:
        return False
    gap = caption[1] - figure[3]
    return horizontal_overlap / narrower_width >= 0.5 and 0 <= gap <= page_height * 0.04


def _caption_supports_figure(figure: BBox, caption: BBox, page_height: float) -> bool:
    if _is_tightly_adjacent_caption(figure, caption, page_height):
        return True
    horizontal_overlap = max(0.0, min(figure[2], caption[2]) - max(figure[0], caption[0]))
    vertical_overlap = max(0.0, min(figure[3], caption[3]) - max(figure[1], caption[1]))
    caption_area = (caption[2] - caption[0]) * (caption[3] - caption[1])
    if caption_area <= 0:
        return False
    figure_width = figure[2] - figure[0]
    figure_height = figure[3] - figure[1]
    if figure_width <= 0 or figure_height <= 0:
        return False
    contained_fraction = horizontal_overlap * vertical_overlap / caption_area
    caption_width = caption[2] - caption[0]
    relative_caption_top = (caption[1] - figure[1]) / figure_height
    return contained_fraction >= 0.8 and (
        relative_caption_top >= 0.65
        or (relative_caption_top >= 0.35 and caption_width <= figure_width * 0.55)
    )


def _cluster_has_two_dimensional_spread(cluster: Sequence[tuple[BBox, int, bool]], bbox: BBox) -> bool:
    if len(cluster) < 3:
        return False
    centers_x = [(item[0][0] + item[0][2]) / 2 for item in cluster]
    centers_y = [(item[0][1] + item[0][3]) / 2 for item in cluster]
    heights = [item[0][3] - item[0][1] for item in cluster]
    vertical_evidence = max(centers_y) - min(centers_y) >= (bbox[3] - bbox[1]) * 0.2 or (
        len({round(height, 1) for height in heights}) >= 3
        and max(heights) - min(heights) >= (bbox[3] - bbox[1]) * 0.1
    )
    return max(centers_x) - min(centers_x) >= (bbox[2] - bbox[0]) * 0.2 and vertical_evidence


def _vector_signature(cluster: Sequence[tuple[BBox, int, bool]], bbox: BBox) -> str:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    payload = ";".join(
        f"{round((item[0][0] - bbox[0]) / width, 3)},"
        f"{round((item[0][1] - bbox[1]) / height, 3)},"
        f"{round((item[0][2] - bbox[0]) / width, 3)},"
        f"{round((item[0][3] - bbox[1]) / height, 3)},"
        f"{item[1]},{int(item[2])}"
        for item in sorted(cluster, key=lambda value: value[0])
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_page_inputs(page_number: int, page_width: float, page_height: float, rotation: int) -> None:
    if type(page_number) is not int:
        raise TypeError("page_number must be an integer and not boolean")
    if page_number < 1:
        raise ValueError("page_number must be at least one")
    normalize_page_size(page_width, page_height, rotation)


def _image_bbox(
    value: object,
    page_width: float,
    page_height: float,
    *,
    ignore_degenerate: bool = False,
) -> BBox | None:
    """Validate image geometry, ignoring transformed images that cannot occupy page area."""
    if not _is_payload_sequence(value) or len(value) != 4:
        raise ValueError(f"PyMuPDF image bbox must have four coordinates: {value}")
    if not all(type(coordinate) in (int, float) for coordinate in value):
        raise TypeError("PyMuPDF image bbox coordinates must be numeric and not boolean")
    coordinates = cast(BBox, tuple(float(cast(int | float, coordinate)) for coordinate in value))
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise ValueError("PyMuPDF image bbox coordinates must be finite")
    if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
        x_extent = sorted((coordinates[0], coordinates[2]))
        y_extent = sorted((coordinates[1], coordinates[3]))
        wholly_off_page = (
            x_extent[1] <= 0 or x_extent[0] >= page_width or y_extent[1] <= 0 or y_extent[0] >= page_height
        )
        if wholly_off_page or ignore_degenerate:
            return None
        raise ValueError("PyMuPDF image bbox must have positive width and height")
    return _clip_bbox(coordinates, page_width, page_height)


def _clip_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox | None:
    clipped = (
        max(0.0, bbox[0]),
        max(0.0, bbox[1]),
        min(page_width, bbox[2]),
        min(page_height, bbox[3]),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _rect_bbox(value: object, field_name: str) -> BBox:
    raw_value: object = (value.x0, value.y0, value.x1, value.y1) if isinstance(value, pymupdf.Rect) else value
    if not _is_payload_sequence(raw_value) or len(raw_value) != 4:
        raise ValueError(f"PyMuPDF {field_name} must have four coordinates: {raw_value}")
    if not all(type(coordinate) in (int, float) for coordinate in raw_value):
        raise TypeError(f"PyMuPDF {field_name} coordinates must be numeric and not boolean")
    coordinates = cast(BBox, tuple(float(cast(int | float, coordinate)) for coordinate in raw_value))
    if not all(math.isfinite(coordinate) for coordinate in coordinates):
        raise ValueError(f"PyMuPDF {field_name} coordinates must be finite")
    if coordinates[2] < coordinates[0] or coordinates[3] < coordinates[1]:
        raise ValueError(f"PyMuPDF {field_name} has inverted coordinates")
    return coordinates


def _is_payload_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _near(first: BBox, second: BBox) -> bool:
    return not (
        first[2] + _VECTOR_JOIN_GAP < second[0]
        or second[2] + _VECTOR_JOIN_GAP < first[0]
        or first[3] + _VECTOR_JOIN_GAP < second[1]
        or second[3] + _VECTOR_JOIN_GAP < first[1]
    )


def _union(bboxes: Iterable[BBox]) -> BBox:
    values = list(bboxes)
    if not values:
        raise ValueError("cannot union empty drawing geometry")
    return (
        min(bbox[0] for bbox in values),
        min(bbox[1] for bbox in values),
        max(bbox[2] for bbox in values),
        max(bbox[3] for bbox in values),
    )


def _overlap_fraction(first: BBox, second: BBox) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / ((first[2] - first[0]) * (first[3] - first[1]))


def _bbox_iou(first: BBox, second: BBox) -> float:
    intersection = _overlap_fraction(first, second) * (first[2] - first[0]) * (first[3] - first[1])
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _bbox_tuple(bbox: BoundingBox) -> BBox:
    return bbox.x0, bbox.y0, bbox.x1, bbox.y1
