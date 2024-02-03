"""Copyright (C) 2022 Adarsh Gupta"""
from typing import Dict, List

from app.pymupdf_parser.parameter.parsed_content import Block


def modify_parsed_content(content: List[Block], function) -> List[Block]:
    """
    It takes a list of blocks and a function that can modify a block.
    It apply the function to each block and return the output.
    """
    modified_content = []
    for block in content:
        modified_block = function(block)
        modified_content.append(modified_block)

    return modified_content
