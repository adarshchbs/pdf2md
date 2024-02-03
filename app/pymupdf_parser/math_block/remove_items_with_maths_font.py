from copy import deepcopy
import logging
from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent
from typing import List, Dict, DefaultDict, Tuple
from collections import defaultdict
from app.pymupdf_parser.data_type_for_content.convert_pymupdf import (
    convert_pymupdf_from_list_to_dict_type,
    convert_pymupdf_from_dict_to_list_type,
)
from app.pymupdf_parser.parameter.pymupdf_content_dict import PyMuPdfContentDict


def remove_items_with_maths_font(pymu_doc_in_dc: List[PyMuPdfContent]):
    pymu_doc_in_dc_dict = convert_pymupdf_from_list_to_dict_type(pymu_doc_in_dc)
    for content in pymu_doc_in_dc_dict:
        for block_no in list(content.blocks.keys()):
            block = content.blocks[block_no]
            for line_no in list(block.lines.keys()):
                line = block.lines[line_no]
                for span_no in list(line.spans.keys()):
                    span = line.spans[span_no]
                    if "math" in span.font.lower():
                        logging.info(f"text with math font = {span.text}")
                        line.spans.pop(span_no)
                if len(line.spans) == 0:
                    block.lines.pop(line_no)
            if len(block.lines) == 0:
                content.blocks.pop(block_no)
    pymu_doc_in_dc = convert_pymupdf_from_dict_to_list_type(pymu_doc_in_dc_dict)
    return pymu_doc_in_dc
