"""Copyright (C) 2022 Adarsh Gupta"""
import logging
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from app.pymupdf_parser.combine_blocks.merge_consecutive_block import (
    merge_consecutive_blocks,
)
from app.pymupdf_parser.parameter.parsed_content import Block
from app.pymupdf_parser.utils.font_characteristics import (
    block_font_characteristics,
    font_characteristics,
)
from app.pymupdf_parser.utils.iou import iou_xaxis


def combine_blocks_wo_linespacing(
    content: List[Block],
    pdf_statitics: Dict[Tuple[int, float], List[float]],
) -> Tuple[List[Block], Dict[Tuple[int, float], List[float]]]:

    modified_content: List[Block] = []

    for block in content:
        if modified_content:
            previous_block = modified_content[-1]

            current_span = block.spans[0]
            previous_span = previous_block.spans[-1]

            current_block_fonts = block_font_characteristics(block)
            previous_block_fonts = block_font_characteristics(previous_block)

            current_block_span_fonts = font_characteristics(current_span)
            previous_block_span_fonts = font_characteristics(previous_span)
            font_size_emphasis_match = (
                current_block_fonts == previous_block_fonts
            ) or (current_block_span_fonts == previous_block_span_fonts)

            previous_block_ending = previous_span.text.strip()[-1]
            current_block_start = current_span.text.strip()[0]

            ending_starting_match = (
                previous_block_ending not in {".", "?", "!"}
                and current_block_start.islower()
            )

            iou_x = iou_xaxis(
                np.array(block.block_bbox), np.array(previous_block.block_bbox)
            )
            current_block_not_contains_header = not block.header.text.strip()

            if (
                font_size_emphasis_match
                and ending_starting_match
                and (iou_x > 0.85)
                and current_block_not_contains_header
                and (previous_block.page_no == block.page_no)
            ):
                if (
                    block.number_of_line == 1
                    and (block.block_bbox[3] - previous_block.block_bbox[3]) > 0
                ):
                    pdf_statitics[
                        (block.block_font, np.round(block.block_size, 1))
                    ].append(
                        np.round(block.block_bbox[3] - previous_block.block_bbox[3], 1)
                    )
                logging.info(
                    f""" Merged with above block due to last and first character:
                            {''.join([span.text for span in block.spans])}"""
                )
                previous_block = merge_consecutive_blocks(previous_block, block)
            else:
                modified_content.append(deepcopy(block))
        else:
            modified_content.append(deepcopy(block))
    return modified_content, pdf_statitics
