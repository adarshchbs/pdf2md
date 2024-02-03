from app.pymupdf_parser.parameter.pymupdf_content import (
    PyMuPdfContent,
    _Blocks,
    _Lines,
    _Spans,
)
from app.pymupdf_parser.parameter.pymupdf_content_dict import (
    PyMuPdfContentDict,
    _BlocksDict,
    _LinesDict,
)
from typing import List
from collections import OrderedDict


def convert_pymupdf_from_list_to_dict_type(
    pymu_doc: List[PyMuPdfContent],
) -> List[PyMuPdfContentDict]:
    pymu_doc_dict: List[PyMuPdfContentDict] = []
    for pymu_content in pymu_doc:
        pymu_content_dict = PyMuPdfContentDict(
            OrderedDict(), pymu_content.width, pymu_content.height
        )
        for block_no, block in enumerate(pymu_content.blocks):
            block_dict = _BlocksDict(
                block.bbox, OrderedDict(), block.number, block.type, block.header
            )
            for line_no, line in enumerate(block.lines):
                line_dict = _LinesDict(line.bbox, line.dir, OrderedDict(), line.wmode)
                for span_no, span in enumerate(line.spans):
                    line_dict.spans[span_no] = span
                block_dict.lines[line_no] = line_dict
            pymu_content_dict.blocks[block_no] = block_dict
        pymu_doc_dict.append(pymu_content_dict)

    return pymu_doc_dict


def convert_pymupdf_from_dict_to_list_type(
    pymu_doc_dict: List[PyMuPdfContentDict],
) -> List[PyMuPdfContent]:
    pymu_doc: List[PyMuPdfContent] = []
    for pymu_content_dict in pymu_doc_dict:
        pymu_content = PyMuPdfContent(
            [], pymu_content_dict.width, pymu_content_dict.height
        )
        for block_dict in pymu_content_dict.blocks.values():
            block = _Blocks(
                block_dict.bbox,
                [],
                block_dict.number,
                block_dict.type,
                block_dict.header,
            )
            for line_dict in block_dict.lines.values():
                line = _Lines(line_dict.bbox, line_dict.dir, [], line_dict.wmode)
                for span in line_dict.spans.values():
                    line.spans.append(span)
                block.lines.append(line)
            pymu_content.blocks.append(block)
        pymu_doc.append(pymu_content)

    return pymu_doc
