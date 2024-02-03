from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent
from typing import List, Dict, DefaultDict, Tuple
from collections import defaultdict
from app.pymupdf_parser.data_type_for_content.convert_pymupdf import (
    convert_pymupdf_from_list_to_dict_type,
    convert_pymupdf_from_dict_to_list_type,
)
from app.pymupdf_parser.parameter.pymupdf_content_dict import PyMuPdfContentDict


def remove_header_n_footer_from_pydoc(
    header: Dict[int, List[Tuple[int, int]]],
    footer: Dict[int, List[Tuple[int, int]]],
    pymu_doc_in_dc: List[PyMuPdfContent],
):
    """
    header: Dict[page_no,List[block_no,line_no]]
    footer: Dict[page_no,List[block_no,line_no]]

    return
        (
            Dict[page_no, List[header_text]],
            Dict[page_no, List[header_text]]
        )
    """
    footer_cleaned = defaultdict(list)
    for page_no, block_line_no_list in footer.items():
        for block_line_no in block_line_no_list:
            if block_line_no not in header[page_no]:
                footer_cleaned[page_no].append(block_line_no)

    header_text: Dict[int, List[str]] = defaultdict(list)
    footer_text: Dict[int, List[str]] = defaultdict(list)

    pymu_doc_in_dc_dict = convert_pymupdf_from_list_to_dict_type(pymu_doc_in_dc)
    remove_line_n_block(header, header_text, pymu_doc_in_dc_dict, "header")
    remove_line_n_block(footer_cleaned, footer_text, pymu_doc_in_dc_dict, "footer")
    pymu_doc_in_dc = convert_pymupdf_from_dict_to_list_type(pymu_doc_in_dc_dict)
    return pymu_doc_in_dc, header_text, footer_text


def remove_line_n_block(
    block_n_line_no: Dict[int, List[Tuple[int, int]]],
    extracted_text: Dict[int, List[str]],
    pymu_doc_in_dc_dict: List[PyMuPdfContentDict],
    header_or_footer: str,
):
    for page_no, header_index_list in block_n_line_no.items():
        for block_no, line_no in header_index_list:
            block = pymu_doc_in_dc_dict[page_no].blocks[block_no]
            line = block.lines.pop(line_no)
            if len(block.lines) == 0:
                pymu_doc_in_dc_dict[page_no].blocks.pop(block_no)

            line_text = "".join([span.text for span in line.spans.values()])
            extracted_text[page_no].append(line_text)

            print(header_or_footer, line_text)
