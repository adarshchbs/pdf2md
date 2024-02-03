"""Copyright (C) 2022 Adarsh Gupta"""
import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import numpy as np

from app.pymupdf_parser.parameter.parsed_content import Block, Span
from app.pymupdf_parser.utils.font_characteristics import font_characteristics


class OrderedDictWithDefaultList(OrderedDict):
    def __missing__(self, key):
        value = []
        self[key] = value
        return value


def merge_sub_script_with_text(block: Block) -> Block:
    lines_of_spans = OrderedDictWithDefaultList()
    for span in block.spans:
        lines_of_spans[span.end_span_line_number].append(span)

    list_of_lines: List[List[Span]] = []

    for _, line in lines_of_spans.items():
        if len(line) == 1:
            list_of_lines.append(line)
            continue

        mask = np.ones(len(line), dtype=bool)
        previous_span_block_height = 0
        previous_span_font_size = 9999
        skip = False
        for i, span in enumerate(line):
            j = i
            if skip:
                skip = False
                continue

            if (span.size > 0.95 * previous_span_font_size) or len(span.text) > 4:
                continue

            if previous_span_block_height != 0:
                has_smaller_font_size = 0
                if span.size < 0.8 * previous_span_font_size:
                    has_smaller_font_size += 2
                elif span.size < 0.90 * previous_span_font_size:
                    has_smaller_font_size += 1

                is_below_previous_span = 0
                if span.end_span_origin[1] > (
                    previous_span_block_y + 0.05 * previous_span_block_height
                ):
                    is_below_previous_span += 2
                elif span.end_span_origin[1] > (
                    previous_span_block_y + 0.02 * previous_span_block_height
                ):
                    is_below_previous_span += 1
                subscript_score = is_below_previous_span + has_smaller_font_size

                if subscript_score >= 1 and is_below_previous_span >= 1:
                    line[i - 1].text += "_{" + span.text + "} "
                    mask[i] = False
                    logging.info(f"-----{span.text}")
                    if (i + 2) < len(line) and font_characteristics(
                        line[i - 1]
                    ) == font_characteristics(line[i + 1]):
                        line[i - 1].text += line[i + 1].text
                        line[i - 1].end_span_bbox = deepcopy(line[i + 1].end_span_bbox)
                        line[i - 1].end_span_origin = deepcopy(
                            line[i + 1].end_span_origin
                        )
                        mask[i + 1] = False
                        skip = True
                        j = i + 1

            previous_span_block_height = (
                line[j].end_span_bbox[3] - line[j].end_span_bbox[1]
            )
            previous_span_font_size = line[j].size
            previous_span_block_y = line[j].end_span_origin[1]

        list_of_lines.append(list(np.array(line)[mask]))

    modified_blocks = deepcopy(block)
    modified_blocks.spans = []
    for line in list_of_lines:
        modified_blocks.spans.extend(iter(line))
    return modified_blocks
