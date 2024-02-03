"""Copyright (C) 2022 Adarsh Gupta"""
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent, _Blocks, _Lines
from app.pymupdf_parser.utils.combine_bboxes import combine_line_bboxes
from app.pymupdf_parser.utils.number_of_fonts_in_line_n_block import (
    get_block_font,
    get_first_span_of_line_font,
    get_last_span_of_line_font,
)


def get_block_bbox_from_lines_bboxes(
    mu_line_list: List[_Lines],
) -> Tuple[float, float, float, float]:
    line_bboxes = [line.bbox for line in mu_line_list]
    return combine_line_bboxes(line_bboxes)


def create_block_from_lines(mu_line_list: List[_Lines]) -> _Blocks:
    return _Blocks(
        get_block_bbox_from_lines_bboxes(mu_line_list),
        mu_line_list,
        len(mu_line_list),
        0,
    )


def split_block_recessively(
    block: _Blocks,
    block_list: List[_Blocks],
    font_description: Dict[str, Tuple[int, int]],
):
    if len(block.lines) == 1 or get_block_font(block, font_description):
        block_list.append(block)
        return block_list

    mu_line_list: List[_Lines] = []
    mu_last_span_of_line_font_list: List[Tuple[float, float]] = []

    for i, line in enumerate(block.lines):
        if mu_line_list:
            first_span_of_line_font = get_first_span_of_line_font(
                line, font_description
            )
            if first_span_of_line_font == mu_last_span_of_line_font_list[-1]:
                mu_line_list.append(line)
                mu_last_span_of_line_font_list.append(
                    get_last_span_of_line_font(line, font_description)
                )
            else:
                upper_block = create_block_from_lines(mu_line_list)
                block_list.append(upper_block)
                if i == len(block.lines):
                    return block_list
                lower_block = create_block_from_lines(block.lines[i:])

                return split_block_recessively(
                    lower_block, block_list, font_description
                )

        else:
            mu_line_list.append(line)
            mu_last_span_of_line_font_list.append(
                get_last_span_of_line_font(line, font_description)
            )
    upper_block = create_block_from_lines(mu_line_list)
    block_list.append(upper_block)
    return block_list


def split_blocks(
    unparsed_contents: List[PyMuPdfContent],
    font_description: Dict[str, Tuple[int, int]],
):

    modified_contents: List[PyMuPdfContent] = []
    for content in unparsed_contents:
        temp_content = PyMuPdfContent([], content.width, content.height)
        for block in content.blocks:
            mu_block_list = []
            mu_block_list = split_block_recessively(
                block, mu_block_list, font_description
            )
            for b in mu_block_list:
                temp_content.blocks.append(b)

        modified_contents.append(temp_content)

    return modified_contents
