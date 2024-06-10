from collections import OrderedDict
import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
from typing import List

import fitz
import numpy as np
import time

from app.pymupdf_parser.table_parser import (
    add_table_boundary,
    cluster_lines,
    cv_operations,
)
from app.pymupdf_parser.table_parser.cv_operations import LineType
from app.pymupdf_parser.table_parser import bounded_table
from app.utils.pix2np import pix2np
from app.pymupdf_parser.utils.filter_item_inside_bbox import (
    filter_lines_which_are_inside_table,
)

def tableFinder(
    img
):
    table_bboxes = []
    boxes_bboxes = []
    image_width = img.shape[1]
    image_height = img.shape[0]
    # page_width_height_dict = (image_width, image_width)
    # TODO: change the acceptable_size to (min-span-width-in-the-page, 1.2* page-margin-width)
    horizontal_lines_acceptable_size = (image_width / 30, 0.9 * image_width)
    # TODO: change the acceptable_size to (min-span-height-in-the-page,  page-margin-height)
    vertical_lines_acceptable_size = (image_height / 75, 0.9 * image_height)
    ver_kernel, hor_kernel, kernel = cv_operations.get_morphological_kernel(
        img
    )
    
    img, thresh = bounded_table.adaptive_threshold(img)
    (
        vertical_lines,
        horizontal_lines,
    ) = bounded_table.get_vertical_n_horizontal_lines(
        vertical_lines_acceptable_size,
        horizontal_lines_acceptable_size,
        ver_kernel,  # type: ignore
        hor_kernel,  # type: ignore
        thresh,
    )
    if len(horizontal_lines) and len(vertical_lines):

        horizontal_lines = cluster_lines.cluster_lines(
            np.array(
                [line.three_point_representation for line in horizontal_lines]
            ),
            LineType.horizontal,
        )

        vertical_lines = cluster_lines.cluster_lines(
            np.array(
                [line.three_point_representation for line in vertical_lines]
            ),
            LineType.vertical,
        )

        "Draw the detected lines on a black canvas"
        img_after_cluster = bounded_table.draw_vertical_n_horizontal_lines(
            img, vertical_lines, horizontal_lines
        )
        new_vertical_lines = (
            add_table_boundary.find_left_most_n_right_most_vertical_lines(
                img_after_cluster
            )
        )
        if new_vertical_lines:
            vertical_lines.extend(new_vertical_lines)
            img_after_cluster = bounded_table.draw_vertical_n_horizontal_lines(
                img, vertical_lines, horizontal_lines
            )

        tables, boxes = bounded_table.detect_cell_of_table(
            vertical_lines,
            horizontal_lines,
            img_after_cluster,  # type: ignore
            vertical_lines_acceptable_size,
            horizontal_lines_acceptable_size,
        )
        table_bboxes.extend([table.bbox for table in tables])
        boxes_bboxes.extend([box.bbox for box in boxes])
        horizontal_lines = filter_lines_which_are_inside_table(
            horizontal_lines,
            [table.bbox for table in tables] + [box.bbox for box in boxes],
        )

    if len(horizontal_lines):
        horizontal_lines_per_page = OrderedDict()
        for line_no, line in enumerate(horizontal_lines):
            horizontal_lines_per_page[line_no] = line
    return table_bboxes, boxes_bboxes

# if __name__ == "__main__":
#     doc = fitz.open("/Users/adarshgupta/Projects/pdf_parser/pdf/1706.03762.pdf")
#     image_list = [pix2np(page.get_pixmap()) for page in doc]
#     document_table: List[List[np.ndarray]] = []
#     document_boxes: List[List[np.ndarray]] = []
#     for img in image_list:
#         # btd = TableFinder(img)
#         start_time = time.time()
#         table_bboxes, boxes_bboxes = tableFinder(img)
#         print("total time:",time.time()-start_time)
#         document_table.append(table_bboxes)
#         document_boxes.append(boxes_bboxes)
#     print("End")
