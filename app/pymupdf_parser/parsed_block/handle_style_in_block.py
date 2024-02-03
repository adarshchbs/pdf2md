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


def first_character_different(block: Block):
    """
    If a single word is splitted into two spans due different font size
    of the inside the word, then merge then to create a single word.
    This usually happens in a header. So, we will also note this
    occurance for later analysis.
    """
    lines_of_spans = OrderedDictWithDefaultList()
    for span in block.spans:
        lines_of_spans[span.end_span_line_number].append(span)

    list_of_lines: List[List[Span]] = []

    for _, line in lines_of_spans.items():
        if len(line) == 1:
            list_of_lines.append(line)
            continue

        mask = np.ones(len(line), dtype=bool)

        for i in range(len(line)):
            number_of_span = len(line) - 1
            if i >= number_of_span:
                continue
            current_span: Span = line[i]
            next_span: Span = line[i + 1]
            if (
                (current_span.text[-1].isalpha())
                and (i != number_of_span)
                and next_span.text[0].isalpha()
                and current_span.font == next_span.font
            ):
                mask[i] = False
                logging.info(
                    f"first character different {current_span.text}-{next_span.text}"
                )
                next_span.text = current_span.text + next_span.text
                next_span.style = 3  ## 3 stand for first character of the word is of different font or size

        line = list(np.array(line)[mask])
        mask = np.ones(len(line), dtype=bool)
        if len(line) > 1:
            previous_span = line[0]
            for i, span in enumerate(line[1:]):
                if font_characteristics(previous_span) == font_characteristics(span):
                    previous_span.text += span.text
                    mask[i + 1] = False
                else:
                    previous_span = span
        list_of_lines.append(list(np.array(line)[mask]))

    modified_blocks = deepcopy(block)
    modified_blocks.spans = []
    for line in list_of_lines:
        modified_blocks.spans.extend(iter(line))
    return modified_blocks
