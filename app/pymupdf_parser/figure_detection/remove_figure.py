import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTFigure, LTImage

from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent
from app.pymupdf_parser.utils.iou import intersection_over_first_bbox_area


def get_figure_bbox(filename: str):
    # Getting the bounding box of the figures and images in the pdf.
    bbox_dict: Dict[int, List] = defaultdict(list)
    for i, page_layout in enumerate(extract_pages(filename)):
        for element in page_layout:
            if type(element) in {LTFigure, LTImage}:
                bbox = element.bbox
                height = page_layout.bbox[3]
                bbox_dict[i].append(
                    np.array([bbox[0], height - bbox[3], bbox[2], height - bbox[1]])
                )

    return bbox_dict


def remove_block_lie_inside_figure(document: List, filename: str):
    bbox_dict = get_figure_bbox(filename)
    ret_document = []
    print(f"{bbox_dict=}")
    for page_no, mu_content in enumerate(document):
        if page_no in bbox_dict.keys():

            new_content = {
                "blocks": [],
                "width": mu_content["width"],
                "height": mu_content["height"],
            }
            for block in mu_content["blocks"]:
                is_inside_figure = False
                for figure_bbox in bbox_dict[page_no]:
                    if (
                        intersection_over_first_bbox_area(
                            np.array(block["bbox"]), figure_bbox
                        )
                        > 0.8
                    ):
                        is_inside_figure = True
                        block_text = "".join(
                            [
                                span["text"]
                                for line in block["lines"]
                                for span in line["spans"]
                            ]
                        )
                        logging.info(f"Inside figure {block_text}")

                if not is_inside_figure:
                    new_content["blocks"].append(block)
            ret_document.append(new_content)
        else:
            ret_document.append(mu_content)
    return ret_document


# def remove_figure_block(document: List[PyMuPdfContent], filename: str):
#     """
#     It removes blocks that are inside a figure
#     """
#     bbox_dict = get_figure_bbox(filename)
#     for page_no in bbox_dict.keys():
#         print(f"{page_no=}")
#         mu_content = document[page_no]
#         i = 0
#         while i < len(mu_content.blocks):
#             block = mu_content.blocks[i]
#             for bbox in bbox_dict[page_no]:
#                 if intersection_over_first_bbox_area(np.array(block.bbox), bbox) > 0.8:
#                     block_text = "".join(
#                         [span.text for line in block.lines for span in line.spans]
#                     )
#                     logging.info(f"Inside figure {block_text}")
#                     mu_content.blocks.pop(i)
#                     break
#             else:
#                 i += 1

#     return document
