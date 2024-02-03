"""Copyright (C) 2022 Adarsh Gupta"""
import logging
from copy import deepcopy
from typing import Dict, List

import numpy as np

from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent
from app.pymupdf_parser.utils.iou import iou_yaxis


def combine_block_bbox(b1, b2):
    return (
        min([b1[0], b2[0]]),
        min([b1[1], b2[1]]),
        max([b1[2], b2[2]]),
        max([b1[3], b2[3]]),
    )


def merge_block_to_right(unparsed_contents: List[PyMuPdfContent]):
    """
    Merge consective lines in blocks if they both have same font and
    font size. And their bbox y_cordinates overlap more than 65%.
    """
    for content in unparsed_contents:
        i = 0
        while i < len(content.blocks[:-1]):
            current_block = content.blocks[i]
            next_block = content.blocks[i + 1]

            no_of_lines = len(current_block.lines)

            if (
                (no_of_lines == 1)
                and (
                    iou_yaxis(np.array(current_block.bbox), np.array(next_block.bbox))
                    > 0.65
                )
                and ((next_block.bbox[0] - current_block.bbox[2]) < 60)
            ):  ## hyperparameter (compare line Y coordinates)
                logging.info(
                    f"Merged to right {''.join([span.text for span in current_block.lines[0].spans]) }"
                )
                for span in next_block.lines[0].spans:
                    current_block.lines[0].spans.append(span)

                current_block.lines[0].bbox = (
                    current_block.lines[0].bbox[0],
                ) + next_block.lines[0].bbox[1:]
                next_block.lines[0] = deepcopy(current_block.lines[0])
                next_block.bbox = combine_block_bbox(
                    next_block.lines[0].bbox, next_block.bbox
                )
                content.blocks.pop(i)
            else:
                i += 1
    return unparsed_contents
