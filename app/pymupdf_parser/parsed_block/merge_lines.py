"""Copyright (C) 2022 Adarsh Gupta"""
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent
from app.pymupdf_parser.utils.iou import iou_yaxis


def merge_lines_in_block(
    unparsed_contents: List[PyMuPdfContent],
    font_description: Dict[str, Tuple[int, int]],
):
    """
    Merge consective lines in blocks if they both have same font and
    font size. And their bbox y_cordinates overlap more than 40%.
    """
    modified_contents: List[PyMuPdfContent] = []
    for content in unparsed_contents:
        temp_content = PyMuPdfContent([], content.width, content.height)
        for block in content.blocks:
            temp_block = deepcopy(block)
            temp_block.lines = []
            temp_line = None
            temp_line_font = np.nan
            for line in block.lines:
                line_font_set = set()
                for span in line.spans:
                    if span.text not in {"", " ", "  ", "   "}:
                        font = font_description[span.font][0]
                        line_font_set.add(f"{font}\_{span.size:.1f}")

                if len(line_font_set) == 1:
                    line_font = line_font_set.pop()

                elif not line_font_set:
                    continue
                else:
                    line_font = np.nan

                if (temp_line_font == line_font) and (
                    iou_yaxis(np.array(line.bbox), np.array(temp_line.bbox))
                    > 0.4  ## hyperparameter (compare line Y coordinates)
                ):
                    for span in line.spans:
                        temp_line.spans.append(span)

                else:
                    temp_line = deepcopy(line)
                    temp_block.lines.append(temp_line)
                    temp_line_font = line_font

            temp_content.blocks.append(temp_block)
        modified_contents.append(temp_content)

    return modified_contents
