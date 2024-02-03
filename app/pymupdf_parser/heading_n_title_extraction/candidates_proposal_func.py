from typing import Any, List, Dict
from app.pymupdf_parser.alignment.horizontal_margin import check_center_align
from app.pymupdf_parser.parameter.parsed_content import Block
from app.pymupdf_parser.utils.number_of_fonts_in_line_n_block import get_block_font
from app.pymupdf_parser.utils.pdf_statitics import FontSizeStat, FontTypeStat
import numpy as np

no_of_positive_function = 3


def is_bold(block_no: int, parsed_doc: List[Block]):
    block: Block = parsed_doc[block_no]
    span_emphasis = [span.emphasis for span in block.spans]
    if 0 in span_emphasis or 2 in span_emphasis:
        return -1
    elif 3 in span_emphasis:
        return 1
    else:
        return 2


def is_font_bigger_than_normal(
    block_no: int, parsed_doc: List[Block], font_size_stat: FontSizeStat
):
    block: Block = parsed_doc[block_no]
    return font_size_stat.is_bigger_than_usual_font(block.block_size)


def is_font_different_than_normal(
    block_no: int, parsed_doc: List[Block], font_type_stat: FontTypeStat
):
    block: Block = parsed_doc[block_no]
    return font_type_stat.is_font_different_than_usual_font(block.block_font)


def is_block_center_aligned(
    block_no: int,
    parsed_doc: List[Block],
    horizontal_margin: Dict[int, Any],
):
    block: Block = parsed_doc[block_no]

    x_cord = np.array(block.block_bbox)[[0, 2]]
    if block.page_no in horizontal_margin and check_center_align(
        x_cord, horizontal_margin[block.page_no], 4
    ):
        return 1
    else:
        return 0


def is_no_of_char_in_block_normal_for_heading(block_no: int, parsed_doc: List[Block]):
    block: Block = parsed_doc[block_no]
    if len(block.block_text) >= 150:
        return -1 * (no_of_positive_function + 1)
    elif 120 < len(block.block_text) < 150 or len(block.block_text) <= 2:
        return -2
    else:
        return 0


def does_next_block_contain_larger_no_character(block_no: int, parsed_doc: List[Block]):
    if block_no >= len(parsed_doc) - 2:
        return 0
    if block_no == 0:
        return 0.5
    scores = []
    for i in [-1, 1, 2]:
        next_block = parsed_doc[block_no + i]
        if len(next_block.block_text) >= 75:
            scores.append(1)
        elif 45 < len(next_block.block_text) < 75:
            scores.append(0.5)
        else:
            scores.append(-1)
    if scores == [-1, -1, -1]:
        return -0.5
    elif scores[0] == 1 or scores[1] == 1:
        return 0.5
    else:
        return 0


def score_based_on_no_of_line_in_next_block(block_no: int, parsed_doc: List[Block]):
    if block_no >= len(parsed_doc) - 2:
        return 0
    if block_no == 0:
        return 0.5
    scores = []
    for i in [-1, 1, 2]:
        next_block = parsed_doc[block_no + i]
        if next_block.number_of_line >= 2:
            scores.append(1)
        else:
            scores.append(-1)
    if scores == [-1, -1, -1]:
        return -0.5
    elif scores[0] == 1 or scores[1] == 1:
        return 0.5
    else:
        return 0


def score_for_numbered_block(block_no: int, parsed_doc: List[Block]):
    """
    If the block is numbered, then there should be at least one unnumbered block between them.
    """
    block = parsed_doc[block_no]
    if not block.header.text:
        return 0
    if block_no == len(parsed_doc) - 1:
        return 0

    # block.header.tag has len greater than 3 if it correspond to numbered list
    # For ex: (2, 0, 0, 'upper_roman', 'number')
    if len(block.header.tag) <= 3:
        return -1 * (no_of_positive_function + 1)
    next_block = parsed_doc[block_no + 1]
    if len(next_block.header.tag) <= 3:
        return 2
    elif next_block.header.tag[3:] != block.header.tag[3:]:
        return 2
    elif next_block.header.tag != block.header.tag:
        return 1
    else:
        return -1


def no_of_word_in_sweet_spot(block_no: int, parsed_doc: List[Block]):
    block = parsed_doc[block_no]
    number_of_words_in_block = len(block.block_text.split(" "))
    return 0.5 if number_of_words_in_block in {2, 3, 4} else 0
