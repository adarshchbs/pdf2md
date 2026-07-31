from dataclasses import FrozenInstanceError

import pytest

from app.pdf2md.semantic_text import (
    SemanticBlock,
    TextBlock,
    TextLine,
    TextSpan,
    classify_semantic_blocks,
    column_aware_reading_order,
    infer_page_column_splits,
    is_caption,
    is_footnote,
    is_heading,
    join_paragraphs,
    normalize_bbox,
    recurring_margin_candidates,
    should_join_paragraphs,
)


def block(
    text: str,
    bbox: tuple[float, float, float, float],
    *,
    page: int = 1,
    size: float = 10,
    bold: bool = False,
    page_width: float = 600,
    page_height: float = 800,
    rotation: int = 0,
    direction: tuple[float, float] = (1.0, 0.0),
) -> TextBlock:
    span = TextSpan(text=text, bbox=bbox, font_size=size, font_name="Synthetic", is_bold=bold)
    line = TextLine(spans=(span,), bbox=bbox, direction=direction)
    return TextBlock(
        lines=(line,),
        bbox=bbox,
        page_number=page,
        page_width=page_width,
        page_height=page_height,
        rotation=rotation,
    )


def compound_block(
    parts: tuple[tuple[str, tuple[float, float, float, float]], ...],
    *,
    page: int,
    page_width: float = 600,
    page_height: float = 800,
    size: float = 8,
    rotation: int = 0,
) -> TextBlock:
    spans = tuple(TextSpan(text=text, bbox=bbox, font_size=size) for text, bbox in parts)
    bbox = (
        min(span.bbox[0] for span in spans),
        min(span.bbox[1] for span in spans),
        max(span.bbox[2] for span in spans),
        max(span.bbox[3] for span in spans),
    )
    return TextBlock(
        lines=(TextLine(spans=spans, bbox=bbox),),
        bbox=bbox,
        page_number=page,
        page_width=page_width,
        page_height=page_height,
        rotation=rotation,
    )


def multiline_block(
    lines: tuple[str, ...], bbox: tuple[float, float, float, float], *, size: float = 10
) -> TextBlock:
    line_height = (bbox[3] - bbox[1]) / len(lines)
    text_lines = tuple(
        TextLine(
            spans=(
                TextSpan(
                    text=text,
                    bbox=(
                        bbox[0],
                        bbox[1] + index * line_height,
                        bbox[2],
                        bbox[1] + (index + 1) * line_height,
                    ),
                    font_size=size,
                ),
            ),
            bbox=(bbox[0], bbox[1] + index * line_height, bbox[2], bbox[1] + (index + 1) * line_height),
        )
        for index, text in enumerate(lines)
    )
    return TextBlock(
        lines=text_lines,
        bbox=bbox,
        page_number=1,
        page_width=600,
        page_height=800,
    )


def test_records_are_deeply_immutable_and_validate_geometry() -> None:
    text_block = block("Immutable", (10, 10, 100, 30))

    with pytest.raises(FrozenInstanceError):
        text_block.page_number = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="positive width"):
        TextSpan(text="bad", bbox=(10, 10, 10, 20), font_size=10)
    with pytest.raises(ValueError, match="multiple of 90"):
        TextBlock(
            lines=text_block.lines,
            bbox=text_block.bbox,
            page_number=1,
            page_width=600,
            page_height=800,
            rotation=45,
        )
    with pytest.raises(ValueError, match="dimensions must be finite"):
        TextBlock(
            lines=text_block.lines,
            bbox=text_block.bbox,
            page_number=1,
            page_width=float("inf"),
            page_height=800,
        )
    with pytest.raises(TypeError, match="numeric and not boolean"):
        TextSpan(text="bad", bbox=(10, 10, 20, 20), font_size=True)
    with pytest.raises(TypeError, match="integer and not boolean"):
        TextBlock(
            lines=text_block.lines,
            bbox=text_block.bbox,
            page_number=True,
            page_width=600,
            page_height=800,
        )
    with pytest.raises(TypeError, match="dimensions must be numeric and not boolean"):
        TextBlock(
            lines=text_block.lines,
            bbox=text_block.bbox,
            page_number=1,
            page_width=True,
            page_height=800,
        )


def test_rotation_normalizes_all_bbox_corners() -> None:
    bbox = (10.0, 20.0, 30.0, 60.0)

    assert normalize_bbox(bbox, 200, 100, 0) == bbox
    assert normalize_bbox(bbox, 200, 100, 90) == (40.0, 10.0, 80.0, 30.0)
    assert normalize_bbox(bbox, 200, 100, 180) == (170.0, 40.0, 190.0, 80.0)
    assert normalize_bbox(bbox, 200, 100, 270) == (20.0, 170.0, 60.0, 190.0)


@pytest.mark.parametrize(
    ("rotation", "direction"),
    [(90, (0.0, -1.0)), (180, (-1.0, 0.0)), (270, (0.0, 1.0))],
)
def test_reading_geometry_applies_only_page_rotation_that_makes_text_upright(
    rotation: int, direction: tuple[float, float]
) -> None:
    visual_page_width, visual_page_height = (800, 600) if rotation in (90, 270) else (600, 800)
    visual_bbox = (40.0, 50.0, 300.0, 75.0)
    native_bbox = normalize_bbox(
        visual_bbox,
        visual_page_width,
        visual_page_height,
        -rotation,
    )
    compensated = block(
        "Visually upright header",
        native_bbox,
        rotation=rotation,
        direction=direction,
    )
    uncompensated = block("Native orientation control", native_bbox, rotation=rotation)

    assert compensated.reading_rotation == rotation
    assert compensated.reading_bbox == pytest.approx(visual_bbox)
    assert uncompensated.reading_rotation == 0
    assert uncompensated.reading_bbox == pytest.approx(native_bbox)


def test_reading_order_finishes_left_column_before_right_column() -> None:
    heading = block("A Generalized Technical Report", (40, 20, 560, 50), size=18, bold=True)
    left_top = block("left one", (40, 100, 270, 130))
    left_bottom = block("left two.", (40, 500, 270, 530))
    right_top = block("right one", (330, 100, 560, 130))
    right_bottom = block("right two.", (330, 500, 560, 530))

    result = column_aware_reading_order([right_bottom, left_bottom, right_top, heading, left_top])

    assert [item.text for item in result] == [
        "A Generalized Technical Report",
        "left one",
        "left two.",
        "right one",
        "right two.",
    ]


def test_reading_order_finishes_each_of_three_columns_before_the_next() -> None:
    blocks = [
        block("left top", (40, 100, 270, 130), page_width=900),
        block("left bottom", (40, 500, 270, 530), page_width=900),
        block("middle top", (335, 100, 565, 130), page_width=900),
        block("middle bottom", (335, 500, 565, 530), page_width=900),
        block("right top", (630, 100, 860, 130), page_width=900),
        block("right bottom", (630, 500, 860, 530), page_width=900),
    ]

    result = column_aware_reading_order(list(reversed(blocks)))

    assert [item.text for item in result] == [
        "left top",
        "left bottom",
        "middle top",
        "middle bottom",
        "right top",
        "right bottom",
    ]


def test_bottom_margin_table_continuation_marker_is_an_excluded_note_semantic() -> None:
    marker = block("Continued →", (470, 734, 560, 742), size=8)

    result = classify_semantic_blocks([marker], body_font_size=10)

    assert [(item.kind, item.text) for item in result] == [("note", "Continued →")]


def test_bottom_margin_band_keeps_vertical_order_across_columns_and_inline_ltr() -> None:
    blocks = [
        block("left body", (40, 100, 270, 130)),
        block("right body", (330, 100, 560, 130)),
        block("Continued →", (470, 730, 560, 744), size=8),
        block("42", (40, 770, 60, 784), size=8),
        block("Report footer", (420, 769, 520, 785), size=8),
    ]

    result = column_aware_reading_order(blocks, page_column_splits={1: (300,)})

    assert [item.text for item in result] == [
        "left body",
        "right body",
        "Continued →",
        "42",
        "Report footer",
    ]


def test_ipcc_style_side_ornament_does_not_create_a_third_column() -> None:
    blocks = [
        block("vertical section tab", (2, 300, 18, 470)),
        block("left one", (40, 50, 297, 220)),
        block("left two", (40, 240, 297, 460)),
        block("left three", (40, 480, 297, 550)),
        block("right one", (306, 50, 563, 145)),
        block("right two", (306, 155, 563, 345)),
        block("70 Short footnote", (40, 640, 112, 650)),
    ]

    assert infer_page_column_splits(blocks) == pytest.approx((301.5,))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_column_support_uses_native_page_height_for_all_rotations(rotation: int) -> None:
    blocks = [
        block("left one", (40, 690, 270, 710), rotation=rotation),
        block("left two", (40, 730, 270, 750), rotation=rotation),
        block("right one", (330, 690, 560, 710), rotation=rotation),
        block("right two", (330, 730, 560, 750), rotation=rotation),
    ]

    assert infer_page_column_splits(blocks) == pytest.approx((300,))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_full_width_separator_restarts_column_order(rotation: int) -> None:
    left_above = block("left above.", (40, 100, 270, 130), rotation=rotation)
    right_above = block("right above.", (330, 100, 560, 130), rotation=rotation)
    separator = block("2 Methods", (40, 300, 560, 335), size=15, bold=True, rotation=rotation)
    left_below = block("left below.", (40, 400, 270, 430), rotation=rotation)
    right_below = block("right below.", (330, 400, 560, 430), rotation=rotation)

    result = column_aware_reading_order([right_below, right_above, separator, left_below, left_above])

    assert [item.text for item in result] == [
        "left above.",
        "right above.",
        "2 Methods",
        "left below.",
        "right below.",
    ]


def test_joining_requires_a_lowercase_aligned_continuation() -> None:
    first = block("The browser uses a conservative", (40, 100, 560, 120))
    continuation = block("layout strategy for this content.", (40, 123, 560, 143))
    new_paragraph = block("A New Paragraph Starts Here", (40, 146, 560, 166))

    assert should_join_paragraphs(first, continuation)
    result = join_paragraphs([first, continuation, new_paragraph])
    assert [item.text for item in result] == [
        "The browser uses a conservative layout strategy for this content.",
        "A New Paragraph Starts Here",
    ]


def test_hanging_indent_list_item_joins_and_preserves_list_metadata() -> None:
    first = block("(b) Systems intended to evaluate learning outcomes, including", (94, 100, 560, 120))
    continuation = block("the learning process in educational institutions;", (108, 123, 500, 143))

    result = join_paragraphs([first, continuation])

    assert len(result) == 1
    assert result[0].text == (
        "Systems intended to evaluate learning outcomes, including "
        "the learning process in educational institutions;"
    )
    assert result[0].paragraph_role == "list_item"
    assert result[0].list_label == "(b)"
    assert result[0].list_depth == 1


def test_numbered_running_footer_is_not_classified_as_a_list_item() -> None:
    footer = block("2.19. GPIO 290", (40, 760, 560, 780))

    result = join_paragraphs([footer])

    assert result[0].text == "2.19. GPIO 290"
    assert result[0].paragraph_role is None
    assert result[0].list_label is None


@pytest.mark.parametrize(
    "text",
    [
        "(2024) Growth accelerated after the intervention.",
        "(However) this parenthesized transition begins ordinary prose.",
        "2024. Growth accelerated after the intervention.",
        "3.2. Ordinary Numbered Heading",
        "1. x = y + z",
        "1. Smith, J. (2020). A cited work.",
        "- P U D",
    ],
)
def test_list_classifier_rejects_common_non_list_labels(text: str) -> None:
    result = join_paragraphs([block(text, (40, 100, 560, 120))])

    assert result[0].text == text
    assert result[0].paragraph_role is None
    assert result[0].list_label is None
    assert result[0].list_depth is None


@pytest.mark.parametrize(
    ("text", "label", "depth", "content"),
    [
        ("• Actual bullet item", "•", 0, "Actual bullet item"),
        (
            "1. Actual numeric item with explanatory content.",
            "1.",
            0,
            "Actual numeric item with explanatory content.",
        ),
        (
            "1.2. Nested numeric item with explanatory content.",
            "1.2.",
            1,
            "Nested numeric item with explanatory content.",
        ),
        ("(a) Parenthesized alphabetic item.", "(a)", 1, "Parenthesized alphabetic item."),
        ("(iv) Nested parenthesized Roman item.", "(iv)", 2, "Nested parenthesized Roman item."),
    ],
)
def test_list_classifier_strips_supported_labels_and_assigns_depth(
    text: str, label: str, depth: int, content: str
) -> None:
    result = join_paragraphs([block(text, (40, 100, 560, 120))])

    assert result[0].text == content
    assert result[0].paragraph_role == "list_item"
    assert result[0].list_label == label
    assert result[0].list_depth == depth


def test_adjacent_numeric_items_are_not_joined_as_continuation_text() -> None:
    first = block("1. First numeric item without punctuation", (40, 100, 560, 120))
    second = block("2. Second numeric item follows", (40, 123, 560, 143))

    result = join_paragraphs([first, second])

    assert [(item.text, item.list_label) for item in result] == [
        ("First numeric item without punctuation", "1."),
        ("Second numeric item follows", "2."),
    ]


def test_indented_prose_after_parenthesized_item_is_a_list_continuation() -> None:
    item = block("(a) remote identification systems.", (94, 100, 300, 120))
    qualification = block(
        "This does not include systems used only for verification.",
        (108, 140, 560, 160),
    )

    result = classify_semantic_blocks([item, qualification], body_font_size=10)

    assert [(semantic.paragraph_role, semantic.list_label) for semantic in result] == [
        ("list_item", "(a)"),
        ("list_item_continuation", None),
    ]


def test_cross_column_join_requires_page_edge_and_continuation_evidence() -> None:
    bottom_left = block("The residual network continues", (40, 680, 270, 790))
    top_right = block("with an identity mapping.", (330, 10, 560, 100))
    lower_right = block("with an identity mapping.", (330, 250, 560, 280))

    assert should_join_paragraphs(bottom_left, top_right)
    assert not should_join_paragraphs(bottom_left, lower_right)
    assert join_paragraphs([top_right, bottom_left])[0].text == (
        "The residual network continues with an identity mapping."
    )


def test_cross_column_join_uses_body_edge_above_footnotes() -> None:
    bottom_left = block("Average annual emissions growth between", (40, 478, 297, 550))
    top_right = block("2010 and 2019 slowed compared to the prior decade.", (306, 46, 563, 142))
    separate_paragraph = block("2010 starts a separate paragraph.", (306, 90, 563, 142))

    assert should_join_paragraphs(bottom_left, top_right)
    assert not should_join_paragraphs(bottom_left, separate_paragraph)
    assert join_paragraphs([top_right, bottom_left])[0].text == (
        "Average annual emissions growth between 2010 and 2019 slowed compared to the prior decade."
    )


def test_never_joins_paragraphs_across_pages() -> None:
    previous = block("Climate projections continue", (40, 760, 560, 790), page=1)
    current = block("under the assessed scenario.", (40, 10, 560, 40), page=2)

    assert not should_join_paragraphs(previous, current)
    assert len(join_paragraphs([previous, current])) == 2


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_classification_joins_proven_single_column_cross_page_continuation_with_chrome(
    scale: float,
) -> None:
    def scaled(text: str, bbox: tuple[float, float, float, float], page: int, size: float = 10) -> TextBlock:
        return block(
            text,
            tuple(value * scale for value in bbox),  # type: ignore[arg-type]
            page=page,
            size=size * scale,
            page_width=600 * scale,
            page_height=800 * scale,
        )

    blocks = [
        scaled("Synthetic Manual", (40, 15, 180, 30), 1, 8),
        scaled("The file operator indicates that the contents of", (40, 700, 560, 750), 1),
        scaled("Page 1", (270, 770, 330, 785), 1, 8),
        scaled("Synthetic Manual", (40, 15, 180, 30), 2, 8),
        scaled("the file will be read in.", (40, 45, 560, 75), 2),
        scaled("Page 2", (270, 770, 330, 785), 2, 8),
    ]

    result = classify_semantic_blocks(blocks, body_font_size=10 * scale, recurring_min_pages=2)

    paragraphs = [item for item in result if item.kind == "paragraph"]
    assert len(paragraphs) == 1
    assert paragraphs[0].text == (
        "The file operator indicates that the contents of the file will be read in."
    )
    assert paragraphs[0].page_numbers == (1, 2)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_cross_page_join_is_quarter_turn_invariant(rotation: int) -> None:
    directions = {0: (1.0, 0.0), 90: (0.0, -1.0), 180: (-1.0, 0.0), 270: (0.0, 1.0)}

    def visual_block(text: str, visual_bbox: tuple[float, float, float, float], page: int) -> TextBlock:
        native_width, native_height = (800, 600) if rotation in (90, 270) else (600, 800)
        native_bbox = normalize_bbox(visual_bbox, 600, 800, -rotation)
        return block(
            text,
            native_bbox,
            page=page,
            page_width=native_width,
            page_height=native_height,
            rotation=rotation,
            direction=directions[rotation],
        )

    previous = visual_block("The rotated paragraph continues", (40, 700, 560, 750), 1)
    current = visual_block("across the displayed page boundary.", (40, 45, 560, 75), 2)

    result = classify_semantic_blocks([previous, current], body_font_size=10)

    assert [(item.text, item.page_numbers) for item in result] == [
        ("The rotated paragraph continues across the displayed page boundary.", (1, 2))
    ]


def test_cross_page_join_cycles_from_last_column_to_first_and_preserves_hard_hyphen() -> None:
    previous = block("The same analysis applies to the sub-", (330, 700, 560, 750), page=1)
    current = block("leading pair in the next event.", (40, 45, 270, 75), page=2)

    result = classify_semantic_blocks(
        [previous, current],
        body_font_size=10,
        page_column_splits={1: (300,), 2: (300,)},
    )

    assert [(item.text, item.page_numbers) for item in result] == [
        ("The same analysis applies to the sub-leading pair in the next event.", (1, 2))
    ]


@pytest.mark.parametrize(("x_scale", "y_scale"), [(0.5, 2.0), (1.7, 0.6)])
def test_cross_page_join_is_affine_and_input_order_invariant(x_scale: float, y_scale: float) -> None:
    def transformed(
        text: str,
        bbox: tuple[float, float, float, float],
        page: int,
    ) -> TextBlock:
        return block(
            text,
            (
                bbox[0] * x_scale,
                bbox[1] * y_scale,
                bbox[2] * x_scale,
                bbox[3] * y_scale,
            ),
            page=page,
            size=10 * y_scale,
            page_width=600 * x_scale,
            page_height=800 * y_scale,
        )

    previous = transformed("The transformed paragraph continues", (40, 700, 560, 750), 1)
    current = transformed("across the physical page boundary.", (40, 45, 560, 75), 2)

    forward = classify_semantic_blocks([previous, current], body_font_size=10 * y_scale)
    reverse = classify_semantic_blocks([current, previous], body_font_size=10 * y_scale)

    expected = [("The transformed paragraph continues across the physical page boundary.", (1, 2))]
    assert [(item.text, item.page_numbers) for item in forward] == expected
    assert [(item.text, item.page_numbers) for item in reverse] == expected


@pytest.mark.parametrize(
    ("previous", "current", "splits"),
    [
        (
            block("A narrow left-column paragraph continues", (40, 700, 270, 750), page=1),
            block("into unrelated narrow-column prose.", (40, 45, 270, 75), page=2),
            None,
        ),
        (
            block("Institutional affiliations continue", (40, 700, 560, 750), page=1),
            block("35 Nevis Laboratory, Columbia University", (40, 45, 560, 75), page=2),
            None,
        ),
        (
            block("An index entry reaches the page edge", (40, 700, 560, 750), page=1),
            block("6.1.3 Alternate Method for Computing a Digest", (40, 45, 560, 75), page=2),
            None,
        ),
    ],
)
def test_cross_page_join_rejects_missed_columns_and_bare_numbered_entries(
    previous: TextBlock,
    current: TextBlock,
    splits: dict[int, tuple[float, ...]] | None,
) -> None:
    result = classify_semantic_blocks(
        [previous, current],
        body_font_size=10,
        page_column_splits=splits,
    )

    assert [item.page_numbers for item in result] == [(1,), (2,)]


@pytest.mark.parametrize(
    "atomic_bboxes",
    [
        {1: ((40, 755, 560, 790),)},
        {2: ((40, 5, 560, 40),)},
    ],
)
def test_cross_page_join_rejects_intervening_table_or_figure_geometry(
    atomic_bboxes: dict[int, tuple[tuple[float, float, float, float], ...]],
) -> None:
    previous = block("A paragraph before atomic page content", (40, 700, 560, 750), page=1)
    current = block("unrelated prose follows the atomic content.", (40, 45, 560, 75), page=2)

    result = classify_semantic_blocks(
        [previous, current],
        body_font_size=10,
        page_atomic_bboxes=atomic_bboxes,
    )

    assert [item.page_numbers for item in result] == [(1,), (2,)]


def test_atomic_geometry_in_an_already_read_opposite_column_does_not_block_join() -> None:
    previous = block("The final-column paragraph continues", (330, 700, 560, 750), page=1)
    current = block("onto the first column of the next page.", (40, 45, 270, 75), page=2)

    result = classify_semantic_blocks(
        [previous, current],
        body_font_size=10,
        page_column_splits={1: (300,), 2: (300,)},
        page_atomic_bboxes={1: ((40, 755, 270, 790),)},
    )

    assert [item.page_numbers for item in result] == [(1, 2)]


def test_cross_page_join_removes_only_explicit_soft_hyphen() -> None:
    previous = block("The system supports inter­", (40, 700, 560, 750), page=1)
    current = block("operability across implementations.", (40, 45, 560, 75), page=2)

    result = classify_semantic_blocks([previous, current], body_font_size=10)

    assert [item.text for item in result] == ["The system supports interoperability across implementations."]


def test_cross_page_join_skips_an_explicit_footnote_without_absorbing_it() -> None:
    previous = block("The residual architecture uses a", (40, 650, 560, 680), page=1)
    footnote = block("4 Supporting implementation detail.", (40, 700, 400, 720), page=1, size=8)
    current = block("parameter-free identity shortcut.", (40, 45, 560, 75), page=2)

    result = classify_semantic_blocks([previous, footnote, current], body_font_size=10)

    assert [(item.kind, item.text, item.page_numbers) for item in result] == [
        ("paragraph", "The residual architecture uses a parameter-free identity shortcut.", (1, 2)),
        ("footnote", "4 Supporting implementation detail.", (1,)),
    ]


def test_cross_page_footer_footnote_continuation_requires_unique_hyphenated_evidence() -> None:
    note = block("4 Bidirectional Trans-", (320, 750, 560, 765), page=1, size=8)
    continuation = block(
        "former terminology continues here.",
        (40, 735, 280, 765),
        page=2,
        size=8,
    )

    result = classify_semantic_blocks([continuation, note], body_font_size=10)

    assert [(item.kind, item.text, item.page_numbers) for item in result] == [
        ("footnote", "4 Bidirectional Trans-former terminology continues here.", (1, 2))
    ]


def test_cross_page_footer_prose_is_not_absorbed_without_unique_hyphenated_note() -> None:
    notes = [
        block("4 First possible Trans-", (40, 750, 280, 765), page=1, size=8),
        block("5 Second possible trans-", (320, 750, 560, 765), page=1, size=8),
    ]
    continuation = block("former-looking footer prose.", (40, 735, 280, 765), page=2, size=8)

    ambiguous = classify_semantic_blocks([continuation, *notes], body_font_size=10)
    no_hyphen = classify_semantic_blocks(
        [
            block("4 Complete note.", (320, 750, 560, 765), page=1, size=8),
            continuation,
        ],
        body_font_size=10,
    )

    assert all(item.page_numbers != (1, 2) for item in ambiguous)
    assert all(item.page_numbers != (1, 2) for item in no_hyphen)


@pytest.mark.parametrize(
    ("previous", "intervening", "current"),
    [
        (
            block("A completed paragraph.", (40, 700, 560, 750), page=1),
            None,
            block("another paragraph starts here.", (40, 45, 560, 75), page=2),
        ),
        (
            block("A paragraph before a figure", (40, 700, 560, 750), page=1),
            block("Figure 2. Separate evidence", (40, 10, 560, 35), page=2, size=9),
            block("continued prose is not adjacent.", (40, 45, 560, 75), page=2),
        ),
        (
            block("A paragraph before a new section", (40, 700, 560, 750), page=1),
            block("2 New Section", (40, 10, 560, 35), page=2, size=14, bold=True),
            block("continued-looking prose remains separate.", (40, 45, 560, 75), page=2),
        ),
        (
            block("1. A list item that reaches the page edge", (40, 700, 560, 750), page=1),
            None,
            block("continued-looking prose remains separate.", (40, 45, 560, 75), page=2),
        ),
        (
            block("A left-column fragment", (40, 700, 270, 750), page=1),
            None,
            block("continued-looking prose remains separate.", (40, 45, 270, 75), page=2),
        ),
    ],
)
def test_cross_page_continuation_controls_fail_closed(
    previous: TextBlock,
    intervening: TextBlock | None,
    current: TextBlock,
) -> None:
    blocks = [previous, *(() if intervening is None else (intervening,)), current]
    result = classify_semantic_blocks(
        blocks,
        body_font_size=10,
        page_column_splits={1: (300,), 2: (300,)} if previous.bbox[2] <= 270 else None,
    )

    assert not any(item.page_numbers == (1, 2) for item in result)


def test_only_soft_hyphen_is_dehyphenated() -> None:
    soft = multiline_block(("inter­", "operability"), (40, 100, 560, 140))
    semantic = multiline_block(("user-", "agent"), (40, 200, 560, 240))

    assert soft.text == "interoperability"
    assert semantic.text == "user-agent"


def test_caption_classifier_requires_an_explicit_label_and_separator() -> None:
    caption = block("Figure 3. Synthetic architecture overview", (80, 500, 520, 530), size=9)
    abbreviated = block("Fig. 3. Synthetic architecture overview", (80, 540, 520, 570), size=9)
    prose = block("Figure skating provides the motivating example.", (40, 500, 560, 530))

    assert is_caption(caption)
    assert is_caption(abbreviated)
    assert not is_caption(prose)


def test_style_boundaries_add_spacing_only_for_a_geometric_word_gap() -> None:
    spaced = TextLine(
        spans=(
            TextSpan(text="Semantic", bbox=(10, 10, 55, 20), font_size=10),
            TextSpan(text="integration", bbox=(58, 10, 110, 20), font_size=10, is_bold=True),
        ),
        bbox=(10, 10, 110, 20),
    )
    touching = TextLine(
        spans=(
            TextSpan(text="inter", bbox=(10, 30, 35, 40), font_size=10),
            TextSpan(text="operability", bbox=(35, 30, 90, 40), font_size=10, is_italic=True),
        ),
        bbox=(10, 30, 90, 40),
    )

    assert spaced.text == "Semantic integration"
    assert touching.text == "interoperability"


def test_footnote_classifier_requires_marker_small_type_and_bottom_margin() -> None:
    footnote = block("1. Synthetic supporting detail", (40, 700, 560, 720), size=8)
    body_number = block("1. Main result", (40, 300, 560, 325), size=10)

    assert is_footnote(footnote, body_font_size=10)
    assert not is_footnote(body_number, body_font_size=10)


def test_numeric_table_row_in_bottom_region_is_not_a_footnote() -> None:
    table_row = block("3 768 12 5.84 77.9 79.8 88.4 6 768 3 5.24", (40, 700, 560, 720), size=8)

    assert not is_footnote(table_row, body_font_size=10)
    result = classify_semantic_blocks([table_row], body_font_size=10)
    assert [(item.kind, item.text) for item in result] == [("paragraph", table_row.text)]


def test_superscript_footnote_marker_joined_to_uppercase_prose_is_accepted() -> None:
    footnote = block("3We tested additional training iterations.", (310, 685, 545, 713), size=8)
    numbered_body = block("34-layer residual network", (40, 300, 280, 325), size=8)

    assert is_footnote(footnote, body_font_size=10)
    assert not is_footnote(numbered_body, body_font_size=10)


def test_bare_numeric_footnote_markers_remain_separate_semantic_blocks() -> None:
    first = block("72 Supporting detail without marker punctuation", (40, 680, 560, 702), size=8)
    second = block("73 Another independently numbered detail", (40, 712, 560, 746), size=8)

    result = classify_semantic_blocks([first, second], body_font_size=10)

    assert [(item.kind, item.text) for item in result] == [
        ("footnote", first.text),
        ("footnote", second.text),
    ]


def test_compound_native_footnote_block_splits_at_each_typographic_label() -> None:
    lines = tuple(
        TextLine(
            spans=(
                TextSpan(text=str(label), bbox=(40, y, 45, y + 6), font_size=6),
                TextSpan(text=text, bbox=(46, y + 1.5, 300, y + 10.5), font_size=8),
            ),
            bbox=(40, y, 300, y + 10.5),
        )
        for label, text, y in (
            (1, "First source note.", 730),
            (2, "Second source note.", 742),
            (3, "Third source note.", 754),
        )
    )
    grouped = TextBlock(
        lines=lines,
        bbox=(40, 730, 300, 764.5),
        page_number=1,
        page_width=600,
        page_height=800,
    )

    result = classify_semantic_blocks([grouped], body_font_size=10)

    assert [(item.kind, item.text) for item in result] == [
        ("footnote", "1 First source note."),
        ("footnote", "2 Second source note."),
        ("footnote", "3 Third source note."),
    ]


def test_heading_classifier_uses_typography_and_avoids_sentence_prose() -> None:
    heading = block("3.2 Deep Residual Learning", (40, 100, 560, 130), size=14, bold=True)
    sentence = block(
        "This larger sentence ends like ordinary prose.",
        (40, 160, 560, 190),
        size=14,
        bold=True,
    )

    assert is_heading(heading, body_font_size=10)
    assert not is_heading(sentence, body_font_size=10)


@pytest.mark.parametrize(("scale", "instruction_italic"), [(0.5, False), (1.0, True), (2.0, True)])
def test_mixed_native_block_splits_short_bold_visual_heading_from_instruction(
    scale: float, instruction_italic: bool
) -> None:
    def styled_line(
        text: str,
        bbox: tuple[float, float, float, float],
        *,
        bold: bool = False,
        italic: bool = False,
    ) -> TextLine:
        scaled = tuple(value * scale for value in bbox)
        span = TextSpan(
            text=text,
            bbox=scaled,  # type: ignore[arg-type]
            font_size=10 * scale,
            is_bold=bold,
            is_italic=italic,
        )
        return TextLine(spans=(span,), bbox=scaled)  # type: ignore[arg-type]

    lines = (
        styled_line("Part IX", (40, 40, 80, 52), bold=True),
        styled_line("Statement of Functional Expenses", (90, 40, 260, 52), bold=True),
        styled_line(
            "All organizations must complete the applicable columns.",
            (40, 54, 520, 66),
            italic=instruction_italic,
        ),
    )
    grouped = TextBlock(
        lines=lines,
        bbox=(40 * scale, 40 * scale, 520 * scale, 66 * scale),
        page_number=1,
        page_width=600 * scale,
        page_height=800 * scale,
    )

    result = classify_semantic_blocks([grouped], body_font_size=10 * scale)

    assert [(item.kind, item.text) for item in result] == [
        ("heading", "Part IX Statement of Functional Expenses"),
        ("paragraph", "All organizations must complete the applicable columns."),
    ]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(
    ("heading_parts", "following", "expected_kinds"),
    [
        (
            (("Part X", True), ("Statement of Program Service Accomplishments", False)),
            "Organizations must report each program service activity.",
            ("heading", "paragraph"),
        ),
        (
            (("3.2 Deep Residual Learning", True),),
            "We evaluate the resulting models below.",
            ("heading", "paragraph"),
        ),
        (
            (("(In thousands)", True),),
            "Amounts are shown for the current period.",
            ("paragraph",),
        ),
        (
            (("1. Actual numeric item with explanatory content.", True),),
            "The same item wraps onto another native line.",
            ("paragraph",),
        ),
        (
            (("Methods and Evaluation Protocol", True),),
            "for all reported experimental settings",
            ("paragraph",),
        ),
        (
            (("Figure 2. Synthetic architecture", True),),
            "The arrows show the forward computation.",
            ("caption",),
        ),
        (
            (("Encoder", True),),
            "Feature map produced by this branch.",
            ("paragraph",),
        ),
    ],
)
def test_heading_style_split_semantics_are_quarter_turn_invariant(
    rotation: int,
    heading_parts: tuple[tuple[str, bool], ...],
    following: str,
    expected_kinds: tuple[str, ...],
) -> None:
    directions = {0: (1.0, 0.0), 90: (0.0, -1.0), 180: (-1.0, 0.0), 270: (0.0, 1.0)}
    native_width, native_height = (800, 600) if rotation in (90, 270) else (600, 800)

    def native_bbox(visual_bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return normalize_bbox(visual_bbox, 600, 800, -rotation)

    heading_spans: list[TextSpan] = []
    x = 40.0
    for text, bold in heading_parts:
        width = max(40.0, len(text) * 5.0)
        bbox = native_bbox((x, 100.0, x + width, 114.0))
        heading_spans.append(TextSpan(text=text, bbox=bbox, font_size=12, is_bold=bold))
        x += width + 8.0
    heading_bbox = (
        min(span.bbox[0] for span in heading_spans),
        min(span.bbox[1] for span in heading_spans),
        max(span.bbox[2] for span in heading_spans),
        max(span.bbox[3] for span in heading_spans),
    )
    following_bbox = native_bbox((40.0, 118.0, 550.0, 132.0))
    following_span = TextSpan(text=following, bbox=following_bbox, font_size=10)
    lines = (
        TextLine(spans=tuple(heading_spans), bbox=heading_bbox, direction=directions[rotation]),
        TextLine(spans=(following_span,), bbox=following_bbox, direction=directions[rotation]),
    )
    grouped_bbox = (
        min(line.bbox[0] for line in lines),
        min(line.bbox[1] for line in lines),
        max(line.bbox[2] for line in lines),
        max(line.bbox[3] for line in lines),
    )
    grouped = TextBlock(
        lines=lines,
        bbox=grouped_bbox,
        page_number=1,
        page_width=native_width,
        page_height=native_height,
        rotation=rotation,
    )

    result = classify_semantic_blocks([grouped], body_font_size=10)

    assert tuple(item.kind for item in result) == expected_kinds
    heading_text = " ".join(text for text, _ in heading_parts)
    if heading_text.startswith("1. "):
        expected_texts = (f"{heading_text.removeprefix('1. ')} {following}",)
        assert result[0].list_label == "1."
    else:
        expected_texts = (
            (heading_text, following) if len(expected_kinds) == 2 else (f"{heading_text} {following}",)
        )
    assert tuple(item.text for item in result) == expected_texts


@pytest.mark.parametrize(
    ("regular_text", "regular_y", "second_bold", "extra_transition"),
    [
        ("All organizations complete the columns.", 40, False, False),
        ("All organizations complete the columns.", 54, True, False),
        ("All organizations complete the columns.", 54, False, True),
        ("(In millions, except per-share amounts)", 54, False, False),
        ("continue across this wrapped visual line.", 54, False, False),
    ],
)
def test_mixed_style_barrier_fails_closed_for_ambiguous_controls(
    regular_text: str, regular_y: int, second_bold: bool, extra_transition: bool
) -> None:
    def line(text: str, bbox: tuple[float, float, float, float], *, bold: bool) -> TextLine:
        span = TextSpan(text=text, bbox=bbox, font_size=10, is_bold=bold)
        return TextLine(spans=(span,), bbox=bbox)

    lines = [
        line("Part IX Statement", (40, 40, 240, 52), bold=True),
        line(regular_text, (40, regular_y, 520, regular_y + 12), bold=second_bold),
    ]
    if extra_transition:
        lines.extend([
            line("Second heading", (40, 68, 180, 80), bold=True),
            line("Additional prose.", (40, 82, 260, 94), bold=False),
        ])
    grouped = TextBlock(
        lines=tuple(lines),
        bbox=(40, 40, 520, max(item.bbox[3] for item in lines)),
        page_number=1,
        page_width=600,
        page_height=800,
    )

    result = classify_semantic_blocks([grouped], body_font_size=10)

    assert len(result) == 1


def test_parenthetical_qualifier_is_not_a_heading_without_context() -> None:
    qualifier = block("(In thousands)", (250, 100, 350, 125), size=12, bold=True)
    embedded_parenthetical = block("Appendix (Revised)", (40, 150, 300, 180), size=14, bold=True)

    result = classify_semantic_blocks([qualifier], body_font_size=10)

    assert not is_heading(qualifier, body_font_size=10)
    assert is_heading(embedded_parenthetical, body_font_size=10)
    assert [(item.kind, item.paragraph_role, item.text) for item in result] == [
        ("paragraph", None, "(In thousands)"),
    ]


@pytest.mark.parametrize("scale", [1.0, 1.5])
def test_rotated_parenthetical_heading_qualifier_stays_a_separate_subtitle(scale: float) -> None:
    def rotated_block(text: str, bbox: tuple[float, float, float, float]) -> TextBlock:
        scaled_bbox = (
            bbox[0] * scale,
            bbox[1] * scale,
            bbox[2] * scale,
            bbox[3] * scale,
        )
        span = TextSpan(text=text, bbox=scaled_bbox, font_size=12 * scale, is_bold=True)
        line = TextLine(spans=(span,), bbox=scaled_bbox, direction=(0.0, -1.0))
        return TextBlock(
            lines=(line,),
            bbox=scaled_bbox,
            page_number=1,
            page_width=612 * scale,
            page_height=792 * scale,
            rotation=90,
        )

    title = rotated_block("Consolidated Statement of Functional Expenses", (88, 279, 102, 513))
    qualifier = rotated_block("(In thousands)", (104, 361, 118, 432))

    result = classify_semantic_blocks([qualifier, title], body_font_size=10 * scale)

    assert [(item.kind, item.paragraph_role, item.text) for item in result] == [
        ("heading", None, title.text),
        ("paragraph", "subtitle", qualifier.text),
    ]


@pytest.mark.parametrize(("scale", "reverse_lines"), [(1.0, False), (1.5, True)])
def test_disconnected_rotated_labels_split_without_splitting_multiline_label(
    scale: float, reverse_lines: bool
) -> None:
    def vertical_line(text: str, bbox: tuple[float, float, float, float]) -> TextLine:
        scaled = (bbox[0] * scale, bbox[1] * scale, bbox[2] * scale, bbox[3] * scale)
        span = TextSpan(text=text, bbox=scaled, font_size=12 * scale, is_bold=True)
        return TextLine(spans=(span,), bbox=scaled, direction=(0.0, -1.0))

    lines = [
        vertical_line("of GHGs in the atmosphere", (78, 358, 91, 498)),
        vertical_line("Increased emissions of", (67, 565, 80, 683)),
    ]
    if reverse_lines:
        lines.reverse()
    grouped = TextBlock(
        lines=tuple(lines),
        bbox=(67 * scale, 358 * scale, 91 * scale, 683 * scale),
        page_number=1,
        page_width=612 * scale,
        page_height=792 * scale,
    )
    one_label = TextBlock(
        lines=(
            vertical_line("Increased concentrations", (67, 361, 80, 492)),
            vertical_line("of GHGs in the atmosphere", (78, 358, 91, 498)),
        ),
        bbox=(67 * scale, 358 * scale, 91 * scale, 498 * scale),
        page_number=1,
        page_width=612 * scale,
        page_height=792 * scale,
    )

    separate_label_blocks = [
        TextBlock(
            lines=(line,),
            bbox=line.bbox,
            page_number=1,
            page_width=612 * scale,
            page_height=792 * scale,
        )
        for line in one_label.lines
    ]
    if reverse_lines:
        separate_label_blocks.reverse()

    split = classify_semantic_blocks([grouped], body_font_size=10 * scale)
    retained = classify_semantic_blocks([one_label], body_font_size=10 * scale)
    consolidated = classify_semantic_blocks(separate_label_blocks, body_font_size=10 * scale)

    assert {item.text for item in split} == {
        "of GHGs in the atmosphere",
        "Increased emissions of",
    }
    assert len(split) == 2
    expected = [("heading", "Increased concentrations of GHGs in the atmosphere")]
    assert [(item.kind, item.text) for item in retained] == expected
    assert [(item.kind, item.text) for item in consolidated] == expected


@pytest.mark.parametrize("direction", [(0.0, -1.0), (0.0, 1.0)])
def test_rotated_multiline_label_reconstruction_is_translation_and_order_invariant(
    direction: tuple[float, float],
) -> None:
    scale = 1.25
    dx, dy = 43.0, 61.0
    first_x, second_x = (67.0, 78.0) if direction[1] < 0 else (78.0, 67.0)

    def directional_block(text: str, x: float) -> TextBlock:
        bbox = (
            x * scale + dx,
            358.0 * scale + dy,
            (x + 13.0) * scale + dx,
            498.0 * scale + dy,
        )
        span = TextSpan(text=text, bbox=bbox, font_size=12 * scale, is_bold=True)
        line = TextLine(spans=(span,), bbox=bbox, direction=direction)
        return TextBlock(
            lines=(line,),
            bbox=bbox,
            page_number=1,
            page_width=612 * scale + 2 * dx,
            page_height=792 * scale + 2 * dy,
        )

    blocks = [
        directional_block("Increased concentrations", first_x),
        directional_block("of GHGs in the atmosphere", second_x),
    ]

    forward = classify_semantic_blocks(blocks, body_font_size=10 * scale)
    reversed_input = classify_semantic_blocks(list(reversed(blocks)), body_font_size=10 * scale)

    expected = [("heading", "Increased concentrations of GHGs in the atmosphere")]
    assert [(item.kind, item.text) for item in forward] == expected
    assert [(item.kind, item.text) for item in reversed_input] == expected


def test_recurring_margin_candidates_generalize_only_matching_page_numbers() -> None:
    headers = [
        block(f"Synthetic Browser Manual — {page}", (40, 15, 560, 35), page=page, size=8)
        for page in range(1, 4)
    ]
    page_numbers = [block(f"Page {page}", (270, 770, 330, 790), page=page, size=8) for page in range(1, 4)]
    distinct_releases = [
        block(f"Release {year}", (40, 45, 200, 65), page=page, size=8)
        for page, year in enumerate((2025, 2026, 2027), start=1)
    ]
    body = block("Synthetic body text.", (40, 200, 560, 230), page=1)

    result = recurring_margin_candidates([body, *headers, *page_numbers, *distinct_releases])

    assert {item.text for item in result} == {item.text for item in [*headers, *page_numbers]}


def test_recurring_bold_margin_text_takes_precedence_over_heading_typography() -> None:
    blocks = [
        block("SYNTHETIC TABLE LABEL", (40, 15, 250, 35), page=page, size=14, bold=True)
        for page in range(1, 4)
    ]

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    assert [(item.kind, item.margin_role) for item in result] == [("recurring_margin", "running_header")] * 3


def test_repeated_all_caps_table_identifier_is_top_chrome_but_caption_like_controls_are_not() -> None:
    repeated = [block("TABLE 1", (40, 15, 120, 35), page=page, size=14, bold=True) for page in range(10, 13)]
    controls = [
        block("Table 1. Results", (40, 15, 180, 35), page=page, size=14, bold=True) for page in range(20, 23)
    ]

    repeated_result = classify_semantic_blocks(repeated, body_font_size=10, recurring_min_pages=2)
    control_result = classify_semantic_blocks(controls, body_font_size=10, recurring_min_pages=2)

    assert [(item.kind, item.margin_role) for item in repeated_result] == [
        ("recurring_margin", "running_header")
    ] * 3
    assert all(item.kind != "recurring_margin" for item in control_result)


def test_split_margin_line_restores_left_to_right_order_and_language_markers() -> None:
    blocks = [
        compound_block(
            (
                ("Synthetic Journal", (40, 20, 180, 34)),
                ("EN", (540, 14, 560, 32)),
            ),
            page=page,
        )
        for page in range(1, 3)
    ]

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    assert [item.text for item in result] == ["Synthetic Journal", "EN"] * 2
    assert all(item.kind == "recurring_margin" and item.margin_role == "running_header" for item in result)


def test_multi_page_margin_roles_split_offset_page_numbers_and_alternating_headers() -> None:
    blocks: list[TextBlock] = []
    for page, printed_page in zip(range(10, 14), range(201, 205), strict=True):
        if page % 2:
            parts = (
                (str(printed_page), (40, 20, 65, 32)),
                (f"Section token {page}", (360, 20, 560, 32)),
            )
        else:
            parts = (
                (f"Chapter token {page}", (40, 20, 240, 32)),
                (str(printed_page), (535, 20, 560, 32)),
            )
        blocks.extend((
            compound_block(parts, page=page),
            block(f"{page - 9}. Real Numbered Heading", (40, 120, 560, 145), page=page, size=14, bold=True),
            block("Repeated body control text.", (40, 300, 560, 320), page=page),
        ))

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    assert sum(item.kind == "page_number" for item in result) == 4
    assert {(item.kind, item.margin_role) for item in result if item.margin_role is not None} == {
        ("page_number", "page_number"),
        ("recurring_margin", "running_header"),
    }
    assert sum(item.kind == "recurring_margin" for item in result) == 4
    assert sum(item.kind == "heading" and "Real Numbered Heading" in item.text for item in result) == 4
    assert (
        sum(item.kind == "paragraph" and item.text == "Repeated body control text." for item in result) == 4
    )


def test_roman_front_matter_and_printed_page_offsets_are_page_numbers() -> None:
    roman = [
        block(label, (280, 770, 320, 790), page=page, size=8)
        for page, label in enumerate(("i", "ii", "iii", "iv", "v"), start=1)
    ]
    offset = [
        block(str(printed), (280, 770, 320, 790), page=page, size=8)
        for page, printed in zip(range(10, 15), range(201, 206), strict=True)
    ]

    roman_result = classify_semantic_blocks(roman, body_font_size=10, recurring_min_pages=3)
    offset_result = classify_semantic_blocks(offset, body_font_size=10, recurring_min_pages=3)

    assert [(item.kind, item.margin_role) for item in roman_result] == [("page_number", "page_number")] * 5
    assert [(item.kind, item.margin_role) for item in offset_result] == [("page_number", "page_number")] * 5


def test_five_page_alternating_headers_have_parity_aware_recurrence() -> None:
    blocks = [
        block(
            "Odd-page report title" if page % 2 else "Even-page chapter title",
            (40, 15, 300, 30),
            page=page,
            size=8,
        )
        for page in range(1, 6)
    ]

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=3)

    assert [(item.kind, item.margin_role) for item in result] == [("recurring_margin", "running_header")] * 5


def test_mirrored_unique_footer_companions_follow_validated_page_numbers() -> None:
    blocks: list[TextBlock] = []
    for page in range(1, 5):
        if page % 2:
            number_bbox = (540, 770, 560, 784)
            companion_bbox = (40, 770, 220, 784)
        else:
            number_bbox = (40, 770, 60, 784)
            companion_bbox = (380, 770, 560, 784)
        blocks.extend((
            block(str(page + 100), number_bbox, page=page, size=8),
            block(f"Unique footer section {page}", companion_bbox, page=page, size=8),
            block(f"{page}. Real Body Heading", (40, 120, 400, 145), page=page, size=14, bold=True),
        ))

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    assert sum(item.kind == "page_number" for item in result) == 4
    assert [item.text for item in result if item.margin_role == "running_footer"] == [
        f"Unique footer section {page}" for page in range(1, 5)
    ]
    assert sum(item.kind == "heading" and "Real Body Heading" in item.text for item in result) == 4


def test_all_caps_table_title_is_an_alternating_folio_companion() -> None:
    blocks: list[TextBlock] = []
    for page, printed_page in zip(range(288, 291), range(274, 277), strict=True):
        if page % 2:
            parts = (
                ("TABLE 1 / HUMAN DEVELOPMENT INDEX AND ITS COMPONENTS", (190, 748, 430, 758)),
                (str(printed_page), (555, 748, 569, 758)),
            )
        else:
            parts = (
                (str(printed_page), (43, 748, 57, 758)),
                ("HUMAN DEVELOPMENT REPORT 2023/2024", (221, 748, 384, 758)),
            )
        blocks.append(compound_block(parts, page=page, page_width=612, page_height=792, size=7))

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    for page, printed_page in zip(range(288, 291), range(274, 277), strict=True):
        page_result = [item for item in result if item.page_numbers == (page,)]
        assert [item.text for item in page_result] == [
            *(
                [str(printed_page), "HUMAN DEVELOPMENT REPORT 2023/2024"]
                if page % 2 == 0
                else ["TABLE 1 / HUMAN DEVELOPMENT INDEX AND ITS COMPONENTS", str(printed_page)]
            )
        ]
        assert {item.margin_role for item in page_result} == {"page_number", "running_footer"}


def test_footer_companion_splits_from_higher_continued_marker() -> None:
    blocks: list[TextBlock] = []
    for page in range(1, 5):
        if page % 2:
            number_bbox = (540, 770, 560, 784)
            footer_bbox = (40, 770, 220, 784)
        else:
            number_bbox = (40, 770, 60, 784)
            footer_bbox = (380, 770, 560, 784)
        blocks.extend((
            block(str(page + 100), number_bbox, page=page, size=8),
            compound_block(
                (
                    ("Continued →", (470, 740, 560, 754)),
                    (f"Unique footer section {page}", footer_bbox),
                ),
                page=page,
            ),
        ))

    result = classify_semantic_blocks(
        blocks,
        body_font_size=10,
        recurring_min_pages=2,
        page_column_splits={page: (300,) for page in range(1, 5)},
    )

    for page in range(1, 5):
        page_result = [item for item in result if item.page_numbers == (page,)]
        assert [item.text for item in page_result] == [
            "Continued →",
            *(
                [f"Unique footer section {page}", str(page + 100)]
                if page % 2
                else [str(page + 100), f"Unique footer section {page}"]
            ),
        ]
        assert next(item for item in page_result if item.text.startswith("Unique footer")).margin_role == (
            "running_footer"
        )


def test_three_page_slice_accepts_odd_only_footer_companion() -> None:
    blocks = [
        item
        for page in range(1, 4)
        for item in (
            block(str(page + 40), (540, 770, 560, 784), page=page, size=8),
            *((block("Odd folio label", (40, 770, 180, 784), page=page, size=8),) if page % 2 else ()),
        )
    ]

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    assert sum(item.kind == "page_number" for item in result) == 3
    assert [item.text for item in result if item.margin_role == "running_footer"] == [
        "Odd folio label",
        "Odd folio label",
    ]


@pytest.mark.parametrize("prefix", ["Table", "Note:", "Source:"])
def test_page_number_companion_rejects_unique_table_notes(prefix: str) -> None:
    notes = [f"{prefix} page-specific values {page}" for page in range(1, 4)]
    blocks = [
        item
        for page, note in enumerate(notes, start=1)
        for item in (
            block(str(page + 20), (540, 770, 560, 784), page=page, size=8),
            block(note, (40, 770, 220, 784), page=page, size=8),
        )
    ]

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    assert not any(item.text in notes and item.margin_role == "running_footer" for item in result)
    assert sum(item.text in notes for item in result) == 3


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_compound_header_splits_recurring_fragment_without_losing_page_heading(rotation: int) -> None:
    blocks = [
        compound_block(
            (
                ("Synthetic recurring report", (40, 20, 230, 32)),
                (f"Unique section {page + 40}", (360, 20, 560, 32)),
            ),
            page=page,
            rotation=rotation,
        )
        for page in range(1, 4)
    ]

    result = classify_semantic_blocks(blocks, body_font_size=10, recurring_min_pages=2)

    assert sum(item.kind == "recurring_margin" for item in result) == 3
    retained = [item for item in result if item.kind == "paragraph"]
    assert [item.text for item in retained] == ["Unique section 41", "Unique section 42", "Unique section 43"]
    assert all(item.margin_role is None and item.paragraph_role is None for item in retained)


def test_margin_false_positive_controls_remain_canonical_content() -> None:
    controls: list[TextBlock] = []
    for page in range(1, 4):
        controls.extend((
            block(f"Unique Section Heading {page + 20}", (40, 20, 560, 42), page=page, size=14, bold=True),
            block("Repeated body control text.", (40, 300, 560, 320), page=page),
            block(f"2026-08-{page:02d}", (40, 60, 150, 75), page=page, size=8),
            block("Table 7 (continued)", (40, 700, 200, 715), page=page, size=8),
        ))
    controls.append(block("One-page publisher chrome", (40, 15, 250, 30), page=4, size=8))

    result = classify_semantic_blocks(controls, body_font_size=10, recurring_min_pages=2)

    assert all(item.kind not in {"recurring_margin", "page_number"} for item in result)
    assert sum(item.kind == "heading" for item in result) == 3
    assert sum(item.text == "Repeated body control text." for item in result) == 3
    assert sum(item.text == "Table 7 (continued)" for item in result) == 3
    assert any(item.text == "One-page publisher chrome" for item in result)
    assert {item.text for item in result} == {item.text for item in controls}


def test_standalone_page_number_requires_recurrence_across_distinct_pages() -> None:
    page_number = block("Page 9", (270, 770, 330, 790), page=1, size=8)
    body_heading = block("9. Real Numbered Heading", (40, 100, 560, 130), size=14, bold=True)

    result = classify_semantic_blocks([page_number, body_heading], body_font_size=10)

    assert [(item.kind, item.margin_role) for item in result] == [
        ("heading", None),
        ("paragraph", None),
    ]


def test_repeated_variable_height_statement_note_is_a_running_footer() -> None:
    notes = [
        block("See accompanying notes to the statements.", (40, y, 400, y + 12), page=page, size=8)
        for page, y in ((1, 350), (2, 500), (3, 430))
    ]
    physical_footers = [
        block(f"Page {page}", (500, 770, 560, 785), page=page, size=8) for page in range(1, 4)
    ]

    result = classify_semantic_blocks([*notes, *physical_footers], body_font_size=10, recurring_min_pages=2)

    assert [item.margin_role for item in result if item.text.startswith("See accompanying")] == [
        "running_footer"
    ] * 3
    assert [item.margin_role for item in result if item.text.startswith("Page")] == ["page_number"] * 3


def test_semantic_pipeline_propagates_frozen_columns_through_paragraph_joining() -> None:
    heading = block("Synthetic Report", (40, 20, 560, 50), size=18, bold=True)
    ornament = block("Section tab.", (2, 300, 18, 470))
    left_top = block("Left top.", (40, 100, 270, 130))
    left_bottom = block("Left bottom.", (40, 500, 270, 530))
    right_top = block("Right top.", (330, 100, 560, 130))
    right_bottom = block("Right bottom.", (330, 500, 560, 530))

    result = classify_semantic_blocks(
        [right_bottom, ornament, left_bottom, right_top, heading, left_top],
        body_font_size=10,
        page_column_splits={1: (300,)},
    )

    assert [item.text for item in result] == [
        "Synthetic Report",
        "Left top.",
        "Section tab.",
        "Left bottom.",
        "Right top.",
        "Right bottom.",
    ]


def test_semantic_pipeline_keeps_ambiguous_blocks_as_paragraphs() -> None:
    heading = block("Technical Summary", (40, 100, 560, 135), size=17, bold=True)
    prose = block("Assessment text remains conservative.", (40, 160, 560, 180), size=10)
    caption = block("Table 2. Synthetic assessment levels", (80, 300, 520, 320), size=9)
    footnote = block("*. Synthetic qualification", (40, 720, 560, 740), size=8)

    result = classify_semantic_blocks([footnote, caption, prose, heading], body_font_size=10)

    assert [(item.kind, item.text) for item in result] == [
        ("heading", "Technical Summary"),
        ("paragraph", "Assessment text remains conservative."),
        ("caption", "Table 2. Synthetic assessment levels"),
        ("footnote", "*. Synthetic qualification"),
    ]
    assert result[0].heading_level == 1
    assert isinstance(result[0], SemanticBlock)
