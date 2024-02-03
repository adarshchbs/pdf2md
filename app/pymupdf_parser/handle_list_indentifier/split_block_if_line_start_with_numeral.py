from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent, _Blocks
from typing import List, Tuple, Optional
import re
from app.pymupdf_parser.handle_list_indentifier.numbered_list_indentifier import (
    pattern_list,
)
from app.pymupdf_parser.parsed_block.split_block import create_block_from_lines


def check_numeral_and_split(block: _Blocks) -> Optional[Tuple[_Blocks, _Blocks]]:
    for line_no in range(1, len(block.lines)):
        line = block.lines[line_no]
        searchable_text = "".join([span.text for span in line.spans[:3]])
        for pattern in pattern_list:
            if not re.search(pattern, searchable_text.strip()):
                continue
            current_line_bbox = line.bbox
            last_line_bbox = block.lines[line_no - 1].bbox
            is_lower_than_previous_line = (current_line_bbox[3] - last_line_bbox[3]) > (
                last_line_bbox[3] - last_line_bbox[1]
            ) / 3
            if is_lower_than_previous_line:
                return create_block_from_lines(
                    block.lines[:line_no]
                ), create_block_from_lines(block.lines[line_no:])


def split_block_if_line_start_with_numeral(
    pymu_doc: List[PyMuPdfContent],
) -> List[PyMuPdfContent]:
    for pymu_content in pymu_doc:
        for block_no, block in enumerate(pymu_content.blocks):
            if splited_block := check_numeral_and_split(block):
                pymu_content.blocks[block_no] = splited_block[0]
                pymu_content.blocks.insert(block_no + 1, splited_block[1])
    return pymu_doc
