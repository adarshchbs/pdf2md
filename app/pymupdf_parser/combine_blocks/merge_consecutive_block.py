"""Copyright (C) 2022 Adarsh Gupta"""
import operator
import numpy as np
from copy import deepcopy

from app.pymupdf_parser.parameter.parsed_content import Block
from app.pymupdf_parser.utils.font_characteristics import font_characteristics


def combine_block_bbox(b1: np.ndarray, b2: np.ndarray):
    return np.array(
        [
            min([b1[0], b2[0]]),
            min([b1[1], b2[1]]),
            max([b1[2], b2[2]]),
            max([b1[3], b2[3]]),
        ]
    )


def merge_consecutive_blocks(previous_block: Block, current_block: Block):
    """
    > If the font characteristics of the last span of the previous block are the same as the font
    characteristics of the current span, then merge the current span with the last span of the previous
    block. Otherwise, append the current span to the previous block
    """

    for current_span in current_block.spans:
        previous_span = previous_block.spans[-1]

        current_block_span_fonts = font_characteristics(current_span)
        previous_block_span_fonts = font_characteristics(previous_span)

        if current_block_span_fonts == previous_block_span_fonts:
            previous_span.text = "".join([previous_span.text, current_span.text])
        else:
            previous_block.spans.append(deepcopy(current_span))

    previous_block.last_line_y_end = current_block.last_line_y_end
    previous_block.block_bbox = combine_block_bbox(
        previous_block.block_bbox, current_block.block_bbox
    )
    previous_block.number_of_line = (
        previous_block.number_of_line + current_block.number_of_line
    )

    for key, count in current_block.number_of_characters_per_font.items():
        previous_block.number_of_characters_per_font[key] += count

    for key, count in current_block.number_of_characters_per_emphasis.items():
        previous_block.number_of_characters_per_emphasis[key] += count

    majority_font_description = max(
        previous_block.number_of_characters_per_font.items(), key=operator.itemgetter(1)
    )[0]
    (
        previous_block.block_font,
        previous_block.block_size,
    ) = majority_font_description

    previous_block.block_emphasis = max(
        previous_block.number_of_characters_per_emphasis.items(),
        key=operator.itemgetter(1),
    )[0]

    return previous_block
