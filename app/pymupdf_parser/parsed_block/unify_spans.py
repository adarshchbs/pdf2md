"""Copyright (C) 2022 Adarsh Gupta"""
import operator
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np

from app.pymupdf_parser.parameter.parsed_content import Block, Span
from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent, _Blocks


def unify_spans(
    block: _Blocks,
    font_description,
) -> Optional[Block]:
    parsed_block = Block(
        np.array([]),
        [],
        "",
        -1,
        np.nan,
        -1,
        0,
        np.nan,
        np.nan,
        {},
        {},
        np.nan,
    )
    for line_no, line in enumerate(block.lines):
        for span in line.spans:
            if not span.text.strip():
                continue
            span_font_description = (
                font_description[span.font][0],
                np.round(span.size, 1),
                font_description[span.font][1],
            )
            if parsed_block.spans and (
                current_span_font_description == span_font_description
            ):
                is_last_char_space = current_parsed_span.text[-1].isspace()
                is_first_char_space = span.text[0].isspace()
                if is_first_char_space or is_last_char_space:
                    current_parsed_span.text = "".join(
                        [current_parsed_span.text, span.text]
                    )
                else:
                    ## If spans donot contain space in between then
                    ## add an extra space
                    current_parsed_span.text = " ".join(
                        [current_parsed_span.text, span.text]
                    )
                current_parsed_span.number_of_char += len(span.text)
                current_parsed_span.number_of_span += 1
                current_parsed_span.end_span_origin = deepcopy(span.origin)
                current_parsed_span.end_span_bbox = deepcopy(span.bbox)
                current_parsed_span.end_span_line_number = line_no
            elif parsed_block.spans and (span.flags & 1):
                continue
                current_parsed_span.text = "".join(
                    [current_parsed_span.text, "^{" + span.text + "}"]
                )
                current_parsed_span.number_of_char += len(span.text)
                current_parsed_span.number_of_span += 1
                current_parsed_span.end_span_origin = deepcopy(span.origin)
                current_parsed_span.end_span_bbox = deepcopy(span.bbox)
                current_parsed_span.end_span_line_number = line_no
            else:
                current_parsed_span = Span(
                    span.text,
                    font_description[span.font][0],
                    np.round(span.size, 1),
                    font_description[span.font][1],
                    len(span.text),
                    deepcopy(span.origin),
                    deepcopy(span.bbox),
                    deepcopy(span.origin),
                    deepcopy(span.bbox),
                    1,
                    line_no,
                    span.flags,
                )
                current_span_font_description = span_font_description
                parsed_block.spans.append(current_parsed_span)

    # parsed_block.text = "".join([span.text for span in parsed_block.spans])
    # parsed_block.text = parsed_block.text.strip()
    parsed_block.header = block.header
    parsed_block.number_of_line = len(block.lines)
    parsed_block.block_bbox = block.bbox
    parsed_block.first_line_y_end = block.lines[0].bbox[3]
    parsed_block.last_line_y_end = block.lines[-1].bbox[3]

    """ Assign the most occurring font, size and emphasis as
    block font, size and emphasis """
    number_of_characters_per_font = defaultdict(int)
    number_of_characters_per_emphasis = defaultdict(int)
    if len(parsed_block.spans) == 0:
        return None
    for span in parsed_block.spans:
        span_font_description = (span.font, np.round(span.size, 1))
        number_of_characters_per_font[span_font_description] += span.number_of_char
        number_of_characters_per_emphasis[span.emphasis] += span.number_of_char

    # if block.header.text:
    #     number_of_characters_per_font[
    #         (block.header.tag[0], block.header.tag[2])
    #     ] += len(block.header.text)
    #     number_of_characters_per_emphasis[block.header.tag[1]] += len(
    #         block.header.text
    #     )

    majority_font_description = max(
        number_of_characters_per_font.items(), key=operator.itemgetter(1)
    )[0]
    (
        parsed_block.block_font,
        parsed_block.block_size,
    ) = majority_font_description

    parsed_block.block_emphasis = max(
        number_of_characters_per_emphasis.items(), key=operator.itemgetter(1)
    )[0]

    parsed_block.number_of_characters_per_font = number_of_characters_per_font
    parsed_block.number_of_characters_per_emphasis = number_of_characters_per_emphasis

    return parsed_block


def parse_document(
    unparsed_contents: List[PyMuPdfContent], font_description
) -> List[Block]:
    parsed_doc = []
    for page_no, content in enumerate(unparsed_contents):
        for block in content.blocks:
            parsed_block = unify_spans(block, font_description)
            if parsed_block is None:
                continue
            parsed_block.page_no = page_no
            parsed_block.page_width = content.width
            parsed_block.page_height = content.height
            parsed_doc.append(parsed_block)

    return parsed_doc
