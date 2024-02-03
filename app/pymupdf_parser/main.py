"""Copyright (C) 2022 Adarsh Gupta"""
import logging
from pathlib import Path
import time
from typing import Optional

import click
import fitz
import numpy as np
from dacite.core import from_dict

from app.pymupdf_parser.parameter.parsed_content import Block
from app.pymupdf_parser.utils.iou import iou_yaxis
from app.pymupdf_parser.math_block.remove_items_with_maths_font import (
    remove_items_with_maths_font,
)
from .combine_blocks.combine_blocks_with_linespacing import (
    combine_blocks_with_linespacing,
    create_clusters,
)
from .combine_blocks.combine_blocks_wo_linespacing import combine_blocks_wo_linespacing
from .combine_blocks.combine_mu_block_to_right import merge_block_to_right
from .figure_detection.remove_figure import remove_block_lie_inside_figure
from .fonts.fonts_description import font_description, fonts_in_document
from .handle_list_indentifier.numbered_list_indentifier import NumberedListIdentifier
from .handle_list_indentifier.unnumbered_list_indentifier import SeperateListIdentifier
from .header_n_footer.header_n_footer import HeaderNFooter
from .parameter.pymupdf_content import PyMuPdfContent
from .parsed_block.handle_style_in_block import first_character_different
from .parsed_block.handle_subscript import merge_sub_script_with_text
from .parsed_block.merge_lines import merge_lines_in_block
from .parsed_block.page_margin import get_page_margin
from .parsed_block.remove_blank_blocks import (
    remove_blank_blocks,
    remove_blank_lines,
    remove_blank_spans,
    remove_tables,
)
from .parsed_block.split_block import split_blocks
from .parsed_block.unify_spans import parse_document
from .table_parser.table_detection import TableDetection
from .utils.modify_parsed_block import modify_parsed_content
from .utils.pdf_statitics import lines_spacing_stats
from .header_n_footer.remove_hnf_from_pydoc import remove_header_n_footer_from_pydoc
from .handle_list_indentifier.split_block_if_line_start_with_numeral import (
    split_block_if_line_start_with_numeral,
)
from .footnotes.identify_footnotes import filter_footnotes


# @click.command()
# @click.option("-p", "--file_path", default=None)
def main(file_path):  # sourcery skip: low-code-quality
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Create and configure logger
    logging.basicConfig(
        filename=f"{__name__}.log", format="%(asctime)s %(message)s", filemode="w"
    )
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.INFO)

    if file_path is None:
        logger.info("file path not passed, taking default file path")
        file_path = (
            "/Users/adarshgupta/Projects/pdf_parser/pdf/Attention-Is-All-You-Need.pdf"
        )

    print("file_path: ", file_path)
    pymu_doc = fitz.open(file_path)  # type: ignore

    # document_layout_dict = get_layout(file_path)
    # document_layout_dict = layoutparser(file_path)
    doc_in_dict = []  # list of dictionaries. Each dictionary follows pattern of pymupdf, but still is a dict
    for page in pymu_doc:
        content = page.get_textpage().extractDICT()
        content["width"] = np.ceil(page.rect.width)  # actual dimensions of the page
        content["height"] = np.ceil(page.rect.height)
        doc_in_dict.append(content)

    # ak_doc = ak.from_iter(doc_in_dict)  # in awkward array
    # horizontal_margin = horizontal_margin_per_page(ak_doc)
    table_detection = TableDetection(file_path)
    table_bboxes, box_bboxes = table_detection.propose()
    # box_bbox is bounding box for any text that is in a box, will not b excluded
    # logger.debug(table_bboxes)

    doc_in_dict = remove_block_lie_inside_figure(doc_in_dict, file_path)
    # figue info is obtained from pdfminer, figures are removed in this step

    # Add the doc_in_dict pages to pymupdf dataclass, exclude tables bounding box in each page
    t_start = time.time()
    pymu_doc_in_dc = []
    for page_content, table_bbox in zip(doc_in_dict, table_bboxes):
        content = from_dict(data_class=PyMuPdfContent, data=page_content)
        remove_tables(content, np.array(table_bbox))
        content = remove_blank_blocks(content)
        content = remove_blank_lines(content)
        content = remove_blank_spans(content)
        pymu_doc_in_dc.append(
            content,
        )
    logger.info(f"Time taken to load dict in dataclass {time.time()-t_start} seconds")

    # for page, tables in zip(pymu_doc, table_bboxes):
    #     for bbox in tables:
    #         page.add_highlight_annot(bbox)
    # pymu_doc.save("/Users/adarshgupta/Projects/pdf_parser/pdf/unbounded_table.pdf")

    # pymu_doc_in_dc = remove_figure_block(pymu_doc_in_dc, file_path)
    t_start = time.time()
    logger.info(
        f"Time taken to load dataclass in awkward array =  {time.time()-t_start} seconds"
    )

    margin = get_page_margin(pymu_doc_in_dc, file_path)
    logger.debug(f"Margin from get_page_margin: {margin}")

    t_start = time.time()
    fonts = fonts_in_document(pymu_doc_in_dc)
    fonts_description_dic = font_description(fonts)
    logger.info(f"Time taken to get font descriptions{time.time()-t_start} seconds")

    pymu_doc_in_dc = merge_lines_in_block(pymu_doc_in_dc, fonts_description_dic)
    pymu_doc_in_dc = split_blocks(pymu_doc_in_dc, fonts_description_dic)
    pymu_doc_in_dc = merge_block_to_right(pymu_doc_in_dc)
    pymu_doc_in_dc = split_block_if_line_start_with_numeral(pymu_doc_in_dc)

    spacing_stats = lines_spacing_stats(pymu_doc_in_dc, fonts_description_dic)

    separate_list_indentifiers = SeperateListIdentifier(
        pymu_doc_in_dc, fonts_description_dic, margin
    )

    separate_list_indentifiers.apply()

    header_n_footer = HeaderNFooter(pymu_doc_in_dc)
    header, footer = header_n_footer.apply()
    pymu_doc_in_dc, header_text, footer_text = remove_header_n_footer_from_pydoc(
        header, footer, pymu_doc_in_dc
    )
    pymu_doc_in_dc = remove_items_with_maths_font(pymu_doc_in_dc)
    parsed_document = parse_document(
        pymu_doc_in_dc, fonts_description_dic
    )  # concept of page is removed, parsed_doc is a list of blocks

    t_start = time.time()
    # note
    parsed_document = modify_parsed_content(parsed_document, merge_sub_script_with_text)
    parsed_document = modify_parsed_content(parsed_document, first_character_different)

    print(f"Time taken for subscript and first character {time.time()-t_start} seconds")

    numbered_list_indentifier = NumberedListIdentifier(parsed_document)
    numbered_list_indentifier.apply()

    parsed_document, spacing_stats = combine_blocks_wo_linespacing(
        parsed_document, spacing_stats
    )
    dbscan_models = create_clusters(spacing_stats)
    parsed_document = combine_blocks_with_linespacing(parsed_document, dbscan_models)
    parsed_document, footnotes = filter_footnotes(parsed_document)

    logger.info(
        """Parsed Document
                    
                """
    )
    # Combine text from all the spans inside a block and print
    # Also check if the next block has y overlap greater than 30%
    # then combine that block text inside the previous text
    paragraphs: list[str] = []
    previous_block: Optional[Block] = None
    previous_text: str = ""
    for block in parsed_document:
        # pymu_doc[block.page_no].add_highlight_annot(block.block_bbox)
        block.block_text = " ".join([span.text for span in block.spans])
        if block.block_emphasis in {1, 3}:
            block.block_text = f"**{block.block_text.strip(' ')}**"

        if block.header.text:
            if block.header.text in block.block_text:
                block.block_text = block.block_text.replace(
                    block.header.text,
                    f"{block.header.text.strip(':. ')}. "
                    if any(c.isalnum() for c in block.header.text)
                    else f"{block.header.text} ",
                    1,
                )
            else:
                block.block_text = block.header.text + " " + block.block_text

            print(block.block_text)
        block_text = block.block_text

        if previous_block is None:
            paragraphs.append(block_text)
        elif iou_yaxis(previous_block.block_bbox, block.block_bbox) > 0.1:
            block_text = " ".join([previous_text, block_text])
        else:
            paragraphs.append(block_text)
        previous_block = block
        previous_text = block_text

    # for i, paragraph in enumerate(paragraphs[:-1]):
    #     if len(paragraph.split()) > 15:
    #         paragraphs[i + 1] = "\n" + paragraphs[i + 1]

    # with open(Path(file_path).with_suffix(".md"), "w") as f:
    #     f.write("\n\n".join(paragraphs))
    return "\n\n".join(paragraphs)


# if __name__ == "__main__":
#     main()
