from itertools import permutations

import pytest

from app.pdf2md.engine import (
    _insert_tables_in_reading_order,  # pyright: ignore[reportPrivateUsage]
    _page_reading_rotation,  # pyright: ignore[reportPrivateUsage]
)
from app.pdf2md.region_reading_order import (
    ReadingRegion,
    decide_reading_region_order,
    order_reading_regions,
)
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FigureStructure,
    PageFragment,
    ParagraphStructure,
)
from app.pdf2md.semantic_text import (
    TextBlock,
    TextLine,
    TextSpan,
    column_aware_reading_order,
    normalize_bbox,
)


def region(
    region_id: str,
    bbox: tuple[float, float, float, float],
    *,
    kind: str = "body",
    native_index: int = 0,
    page_width: float = 600,
    page_height: float = 800,
) -> ReadingRegion:
    return ReadingRegion(
        region_id=region_id,
        bbox=bbox,
        kind=kind,
        native_index=native_index,
        page_width=page_width,
        page_height=page_height,
    )


def ids(items: tuple[ReadingRegion, ...]) -> list[str]:
    return [item.region_id for item in items]


def test_single_column_preserves_native_region_order() -> None:
    items = (
        region("native-first", (40, 200, 560, 225), native_index=0),
        region("native-second", (40, 100, 560, 125), native_index=1),
    )

    assert order_reading_regions(items, column_splits=()) == items


def test_engine_page_rotation_tie_break_is_permutation_invariant() -> None:
    horizontal_bbox = (40.0, 100.0, 140.0, 125.0)
    horizontal_span = TextSpan("equal", horizontal_bbox, 10)
    horizontal = TextBlock(
        lines=(TextLine((horizontal_span,), horizontal_bbox, (1.0, 0.0)),),
        bbox=horizontal_bbox,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
    )
    rotated_bbox = normalize_bbox(horizontal_bbox, 600, 800, -90)
    rotated_span = TextSpan("equal", rotated_bbox, 10)
    rotated = TextBlock(
        lines=(TextLine((rotated_span,), rotated_bbox, (0.0, -1.0)),),
        bbox=rotated_bbox,
        page_number=1,
        page_width=800,
        page_height=600,
        rotation=90,
    )

    assert _page_reading_rotation((horizontal, rotated)) == 0
    assert _page_reading_rotation((rotated, horizontal)) == 0


def test_equal_frequency_equal_area_page_extent_tie_is_permutation_invariant() -> None:
    items = (
        region("left-top", (40, 100, 270, 125), page_width=600, page_height=800),
        region("left-bottom", (40, 500, 270, 525), page_width=800, page_height=600),
        region("right-top", (330, 100, 560, 125), page_width=600, page_height=800),
        region("right-bottom", (330, 500, 560, 525), page_width=800, page_height=600),
    )
    expected = ["left-top", "left-bottom", "right-top", "right-bottom"]

    for permuted in permutations(items):
        decision = decide_reading_region_order(permuted, column_splits=(300,))
        assert decision.status == "ordered"
        assert ids(decision.regions) == expected


def test_two_body_columns_are_column_major_and_geometry_authoritative() -> None:
    items = (
        region("left-top", (40, 100, 270, 125)),
        region("left-bottom", (40, 500, 270, 525)),
        region("right-top", (330, 100, 560, 125)),
        region("right-bottom", (330, 500, 560, 525)),
    )
    expected = ["left-top", "left-bottom", "right-top", "right-bottom"]

    for permuted in permutations(items):
        assert ids(order_reading_regions(permuted, column_splits=(300,))) == expected


def test_panel_caption_grid_is_row_major_not_body_column_major() -> None:
    items = (
        region("top-left", (40, 100, 270, 125), kind="caption"),
        region("top-right", (330, 100, 560, 125), kind="caption"),
        region("bottom-left", (40, 500, 270, 525), kind="caption"),
        region("bottom-right", (330, 500, 560, 525), kind="caption"),
    )

    assert ids(order_reading_regions(tuple(reversed(items)), column_splits=(300,))) == [
        "top-left",
        "top-right",
        "bottom-left",
        "bottom-right",
    ]


def test_full_width_interruption_restarts_column_flow() -> None:
    items = (
        region("right-below", (330, 500, 560, 525)),
        region("left-above", (40, 100, 270, 125)),
        region("interruption", (40, 300, 560, 340), kind="table"),
        region("right-above", (330, 100, 560, 125)),
        region("left-below", (40, 500, 270, 525)),
    )

    assert ids(order_reading_regions(items, column_splits=(300,))) == [
        "left-above",
        "right-above",
        "interruption",
        "left-below",
        "right-below",
    ]


def test_headers_and_footers_form_deterministic_page_bands() -> None:
    items = (
        region("footer-right", (420, 770, 560, 785), kind="footer"),
        region("body", (40, 200, 560, 225)),
        region("header-right", (500, 15, 560, 30), kind="header"),
        region("footer-left", (40, 770, 80, 785), kind="page_number"),
        region("header-left", (40, 15, 200, 30), kind="header"),
    )

    assert ids(order_reading_regions(items, column_splits=())) == [
        "header-left",
        "header-right",
        "body",
        "footer-left",
        "footer-right",
    ]


def test_contained_figure_children_follow_container_without_triggering_peer_fallback() -> None:
    items = (
        region("body-after", (40, 600, 560, 625), native_index=0),
        region("caption", (100, 430, 500, 455), kind="caption", native_index=1),
        region("internal-text", (100, 180, 500, 220), native_index=2),
        region("figure", (60, 100, 540, 500), kind="figure", native_index=3),
    )

    ordered = order_reading_regions(items, column_splits=(300,))

    assert ids(ordered) == ["figure", "internal-text", "caption", "body-after"]
    assert {id(item) for item in ordered} == {id(item) for item in items}


def _document_element(
    element_id: str,
    bbox: tuple[float, float, float, float],
    *,
    element_type: str = "paragraph",
    linked_ids: tuple[str, ...] = (),
    source_id: str,
) -> DocumentElement:
    fragment = PageFragment(
        page_number=1,
        page_width=600,
        page_height=800,
        bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
        source_item_ids=[source_id],
    )
    structure = (
        ElementStructure(
            figure=FigureStructure(
                asset_path="pdf://synthetic/figure",
                sha256="0" * 64,
                width=480,
                height=400,
                caption_element_id="caption",
            ),
            linked_element_ids=list(linked_ids),
        )
        if element_type == "figure"
        else ElementStructure(
            paragraph=ParagraphStructure(role="caption" if element_type == "caption" else "body"),
            linked_element_ids=list(linked_ids),
        )
    )
    return DocumentElement(
        document_id="synthetic-document",
        element_id=element_id,
        order=0,
        element_type=element_type,
        content="" if element_type == "figure" else element_id,
        format="text",
        fragments=[fragment],
        structure=structure,
        annotation=AnnotationMetadata(
            stage="candidate",
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        ),
    )


def test_engine_containment_order_preserves_figure_links_and_provenance() -> None:
    figure = _document_element(
        "figure",
        (60, 100, 540, 500),
        element_type="figure",
        linked_ids=("internal-text", "caption"),
        source_id="figure-source",
    )
    internal = _document_element(
        "internal-text",
        (100, 180, 500, 220),
        linked_ids=("figure",),
        source_id="internal-source",
    )
    caption = _document_element(
        "caption",
        (100, 430, 500, 455),
        element_type="caption",
        linked_ids=("figure",),
        source_id="caption-source",
    )
    after = _document_element("after", (40, 600, 560, 625), source_id="after-source")
    original = [after, caption, internal, figure]

    ordered = _insert_tables_in_reading_order(
        [after, caption, internal],
        [figure],
        column_splits=(300,),
    )

    assert [item.element_id for item in ordered] == ["figure", "internal-text", "caption", "after"]
    assert {id(item) for item in ordered} == {id(item) for item in original}
    assert figure.structure.linked_element_ids == ["internal-text", "caption"]
    assert caption.structure.linked_element_ids == ["figure"]
    assert [item.fragments[0].source_item_ids for item in ordered] == [
        ["figure-source"],
        ["internal-source"],
        ["caption-source"],
        ["after-source"],
    ]


def test_engine_preserves_native_order_after_ambiguous_peer_fallback() -> None:
    right_first = _document_element(
        "right-first",
        (300, 755, 560, 780),
        source_id="right-source",
    )
    left_second = _document_element(
        "left-second",
        (40, 750, 400, 775),
        source_id="left-source",
    )
    figure = _document_element(
        "separate-figure",
        (60, 300, 540, 500),
        element_type="figure",
        source_id="figure-source",
    )

    for peers in ([right_first, left_second], [left_second, right_first]):
        ordered = _insert_tables_in_reading_order(
            peers,
            [figure],
            column_splits=(300,),
        )
        assert [item.element_id for item in ordered] == [
            peers[0].element_id,
            peers[1].element_id,
            "separate-figure",
        ]
    decision = decide_reading_region_order(
        (
            region("right-first", (300, 755, 560, 780), native_index=0),
            region("left-second", (40, 750, 400, 775), native_index=1),
            region("separate-figure", (60, 300, 540, 500), kind="figure", native_index=2),
        ),
        column_splits=(300,),
    )
    assert decision.status == "native_fallback"


def test_overlapping_children_of_one_figure_do_not_trigger_peer_fallback() -> None:
    items = (
        region("body-after", (40, 600, 560, 625), native_index=0),
        region("panel-heading", (100, 180, 500, 220), kind="heading", native_index=1),
        region("panel-text", (100, 205, 500, 245), native_index=2),
        region("figure", (60, 100, 540, 500), kind="figure", native_index=3),
    )

    decision = decide_reading_region_order(items, column_splits=(300,))

    assert decision.status == "ordered"
    assert ids(decision.regions) == ["figure", "panel-heading", "panel-text", "body-after"]


def test_contained_table_text_uses_the_same_expected_containment_relation() -> None:
    items = (
        region("table-caption", (90, 110, 510, 135), kind="caption", native_index=0),
        region("cell-text", (100, 180, 250, 205), native_index=1),
        region("table", (60, 80, 540, 300), kind="table", native_index=2),
    )

    assert ids(order_reading_regions(items, column_splits=(300,))) == [
        "table",
        "table-caption",
        "cell-text",
    ]


def test_overlapping_regions_fail_closed_to_native_order() -> None:
    items = (
        region("native-first", (40, 100, 300, 200), native_index=0),
        region("native-second", (200, 150, 560, 250), native_index=1),
        region("native-third", (40, 300, 270, 325), native_index=2),
    )

    assert order_reading_regions(items, column_splits=(300,)) == items


def test_translation_and_scaling_are_metamorphically_invariant() -> None:
    source = (
        region("left", (40, 100, 270, 125)),
        region("right", (330, 100, 560, 125)),
        region("wide", (40, 300, 560, 340), kind="figure"),
    )

    def transform(item: ReadingRegion, scale: float, dx: float, dy: float) -> ReadingRegion:
        x0, y0, x1, y1 = item.bbox
        return ReadingRegion(
            region_id=item.region_id,
            bbox=(x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy),
            kind=item.kind,
            native_index=item.native_index,
            page_width=item.page_width * scale + 2 * dx,
            page_height=item.page_height * scale + 2 * dy,
        )

    baseline = ids(order_reading_regions(source, column_splits=(300,)))
    transformed = tuple(transform(item, 1.75, 31, 47) for item in reversed(source))
    assert ids(order_reading_regions(transformed, column_splits=(300 * 1.75 + 31,))) == baseline


def test_quarter_turn_canonical_geometry_has_identical_order() -> None:
    visual = (
        region("left-top", (40, 100, 270, 125)),
        region("left-bottom", (40, 500, 270, 525)),
        region("right-top", (330, 100, 560, 125)),
        region("right-bottom", (330, 500, 560, 525)),
    )
    expected = ids(order_reading_regions(visual, column_splits=(300,)))

    for rotation in (0, 90, 180, 270):
        assert ids(order_reading_regions(visual, column_splits=(300,), rotation=rotation)) == expected


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_semantic_adapter_orders_rotated_two_column_pages_in_display_coordinates(rotation: int) -> None:
    visual_boxes = {
        "left-top": (40.0, 100.0, 270.0, 125.0),
        "left-bottom": (40.0, 500.0, 270.0, 525.0),
        "right-top": (330.0, 100.0, 560.0, 125.0),
        "right-bottom": (330.0, 500.0, 560.0, 525.0),
    }
    directions = {0: (1.0, 0.0), 90: (0.0, -1.0), 180: (-1.0, 0.0), 270: (0.0, 1.0)}
    native_width, native_height = (800.0, 600.0) if rotation in (90, 270) else (600.0, 800.0)
    blocks = []
    for text, visual_bbox in reversed(tuple(visual_boxes.items())):
        native_bbox = normalize_bbox(visual_bbox, 600, 800, -rotation)
        span = TextSpan(text=text, bbox=native_bbox, font_size=10)
        blocks.append(
            TextBlock(
                lines=(TextLine(spans=(span,), bbox=native_bbox, direction=directions[rotation]),),
                bbox=native_bbox,
                page_number=1,
                page_width=native_width,
                page_height=native_height,
                rotation=rotation,
            )
        )

    ordered = column_aware_reading_order(blocks, page_column_splits={1: (300,)})

    assert [item.text for item in ordered] == [
        "left-top",
        "left-bottom",
        "right-top",
        "right-bottom",
    ]


@pytest.mark.parametrize("scale", [0.75, 1.0, 1.8])
def test_semantic_adapter_uses_one_page_frame_for_mixed_text_directions(scale: float) -> None:
    rotation = 90
    native_width, native_height = 800 * scale, 600 * scale
    directions = {"body": (0.0, -1.0), "label": (1.0, 0.0)}
    visual = (
        ("left-top", (40, 100, 270, 125), "body"),
        ("left-bottom", (40, 500, 270, 525), "body"),
        ("right-top", (330, 100, 560, 125), "body"),
        ("right-bottom", (330, 500, 560, 525), "body"),
        ("side-label", (5, 300, 20, 420), "label"),
    )
    blocks = []
    for text, bbox, kind in reversed(visual):
        scaled_bbox = tuple(value * scale for value in bbox)
        native_bbox = normalize_bbox(scaled_bbox, 600 * scale, 800 * scale, -rotation)  # type: ignore[arg-type]
        span = TextSpan(text=text, bbox=native_bbox, font_size=10 * scale)
        blocks.append(
            TextBlock(
                lines=(TextLine((span,), native_bbox, directions[kind]),),
                bbox=native_bbox,
                page_number=1,
                page_width=native_width,
                page_height=native_height,
                rotation=rotation,
            )
        )

    ordered = column_aware_reading_order(
        blocks,
        page_column_splits={1: (300 * scale,)},
    )

    assert [item.text for item in ordered] == [
        "left-top",
        "side-label",
        "left-bottom",
        "right-top",
        "right-bottom",
    ]


@pytest.mark.parametrize(
    ("bbox", "page_width", "page_height"),
    [
        ((True, 1, 2, 3), 600, 800),
        ((0, 1, 2, 3), True, 800),
        ((0, 1, 2, 3), 600, False),
    ],
)
def test_region_geometry_rejects_boolean_numbers(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
) -> None:
    with pytest.raises(ValueError, match="numeric|dimensions"):
        region("invalid", bbox, page_width=page_width, page_height=page_height)


def test_invalid_or_duplicate_regions_fail_fast() -> None:
    first = region("duplicate", (40, 100, 270, 125))
    second = region("duplicate", (330, 100, 560, 125))
    with pytest.raises(ValueError, match="unique"):
        order_reading_regions((first, second), column_splits=(300,))
