"""Copyright (C) 2022 Adarsh Gupta"""
from app.pymupdf_parser.parameter.parsed_content import Block, Span


def font_characteristics(span: Span):
    """
    It returns font characteristics as a tuple.
    The tuple = (font-as-int, size,emphasis)
    emphasis := {bold, italics, bold-italics}
    """
    return (span.font, span.size, span.emphasis)


def block_font_characteristics(block: Block):
    """
    It returns font characteristics as a tuple.
    The tuple = (font-as-int, size,emphasis)
    emphasis := {bold, italics, bold-italics}
    """
    return (block.block_font, block.block_size, block.block_emphasis)
