"""Copyright (C) 2022 Adarsh Gupta"""
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple, DefaultDict

import numpy as np
from app.pymupdf_parser.parameter.parsed_content import Block

from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent, _Blocks
from app.pymupdf_parser.utils.approximate_numbers import ApproximateNumbersWithThreshold
from app.pymupdf_parser.utils.number_of_fonts_in_line_n_block import get_line_font


def get_previous_line(block: _Blocks, line_index: int):
    return None if line_index == 0 else block.lines[line_index - 1]


def lines_spacing_stats(
    unparsed_contents: List[PyMuPdfContent],
    font_description: Dict[str, Tuple[int, int]],
) -> Dict[Tuple[int, float], List[float]]:
    """
    It takes a list of PyMuPdfContent objects and a dictionary of font descriptions,
    and returns a dictionary of (font-as-int,font-size) tuples to lists of line spacing values

    :return: A dictionary with the (font-as-int, font-size) as the key and
    a list of the spacing between lines as the value.
    """
    spacing_stats: DefaultDict[Tuple[int, float], List[float]] = defaultdict(list)
    for content in unparsed_contents:
        for block in content.blocks:
            for line_no, line in enumerate(block.lines):
                previous_line = get_previous_line(block, line_no)
                if previous_line is None:
                    continue
                previous_line_font = get_line_font(previous_line, font_description)
                line_font = get_line_font(line, font_description)
                if previous_line_font is None or line_font is None:
                    continue

                if (
                    previous_line_font == line_font
                    and (line.bbox[3] - previous_line.bbox[3]) > 0.05
                ):
                    # TODO change the threshold as percent of font size
                    spacing_stats[line_font].append(
                        np.round(line.bbox[3] - previous_line.bbox[3], 1)
                    )
    return spacing_stats


class FontSizeStat:
    def __init__(self, parsed_doc: List[Block]) -> None:
        font_type_dict = defaultdict(int)
        for block in parsed_doc:
            font_type_dict[block.block_size] += len(block.block_text)
        font_size = np.array(list(font_type_dict.keys()))
        threshold = min(font_size) / 20
        self.ap = ApproximateNumbersWithThreshold(font_size, threshold=threshold)

        font_size_dict_with_approx = defaultdict(int)
        for size, value in font_type_dict.items():
            font_size_dict_with_approx[self.ap.approx(size)] += value

        total_number_char_in_doc = sum(font_size_dict_with_approx.values())
        self.font_size_dict_percentage_in_doc = {
            size: no_of_char / total_number_char_in_doc
            for size, no_of_char in font_size_dict_with_approx.items()
        }
        self.usual_font_size = {
            size
            for size, percentage in self.font_size_dict_percentage_in_doc.items()
            if percentage > 0.10
        }
        self.smallest_usual_font = min(self.usual_font_size)

    def is_bigger_than_usual_font(self, size):
        size = self.ap.approx(size)
        if size in self.usual_font_size:
            return -1
        elif size < self.smallest_usual_font:
            return -2
        else:
            return 1


class FontTypeStat:
    def __init__(self, parsed_doc: List[Block]) -> None:
        font_type_dict = defaultdict(int)
        for block in parsed_doc:
            font_type_dict[block.block_font] += len(block.block_text)

        total_number_char_in_doc = sum(font_type_dict.values())
        self.font_type_dict_percentage_in_doc = {
            font_type: no_of_char / total_number_char_in_doc
            for font_type, no_of_char in font_type_dict.items()
        }
        self.usual_font_type = {
            font_type
            for font_type, percentage in self.font_type_dict_percentage_in_doc.items()
            if percentage > 0.10
        }

    def is_font_different_than_usual_font(self, font_type: int):
        return 0 if font_type in self.usual_font_type else 1
