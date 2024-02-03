"""Copyright (C) 2022 Adarsh Gupta"""
from typing import Dict, Tuple, Optional

import numpy as np
from app.pymupdf_parser.parameter.pymupdf_content import _Blocks, _Lines


def get_line_font(
    line: _Lines, font_description: Dict[str, Tuple[int, int]]
) -> Optional[Tuple[int, float]]:
    """
    It takes a line and a dictionary of font names to (font-in-int,font-emphasis), and returns the
    (font-in-int,font-size) of the line if all spans in the line have the same font-in-int
    """

    font_property = {
        (font_description[span.font][0], np.round(span.size)) for span in line.spans
    }

    return font_property.pop() if len(font_property) == 1 else None


def get_last_span_of_line_font(
    line: _Lines, font_description: Dict[str, Tuple[int, int]]
) -> Tuple[int, float]:
    """
    It takes a line and a font description and returns the (font-in-int,font-size) of the last span of
    the line
    """
    span = line.spans[-1]
    return (font_description[span.font][0], np.round(span.size))


def get_first_span_of_line_font(
    line: _Lines, font_description: Dict[str, Tuple[int, int]]
) -> Tuple[int, float]:
    """
    It takes a line and a font description and returns the (font-in-int,font-size) of the first span of
    the line
    """

    span = line.spans[0]
    return (font_description[span.font][0], np.round(span.size))


def get_block_font(
    block: _Blocks, font_description: Dict[str, Tuple[int, int]]
) -> Optional[Tuple[int, float]]:
    """
    It takes a block and a dictionary of font descriptions and returns the (font-in-int,font-size) of the block if
    all the spans in the block have the same font and size.
    """

    font_property = {
        (font_description[span.font][0], np.round(span.size))
        for line in block.lines
        for span in line.spans
    }

    return font_property.pop() if len(font_property) == 1 else None
