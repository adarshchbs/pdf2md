from collections import OrderedDict
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

from app.pymupdf_parser.table_parser.remove_recurring_horizontal_lines import (
    RecurringHorizontalLines,
)

from typing import List, Dict, Tuple

import fitz
import numpy as np

from app.pymupdf_parser.table_parser import (
    add_table_boundary,
    cluster_lines,
    cv_operations,
)
from app.pymupdf_parser.table_parser.create_table_from_cell_bbox import (
    Table,
)
from app.pymupdf_parser.table_parser.cv_operations import LineParameters, LineType
from app.pymupdf_parser.table_parser.unbounded_table_detection import (
    group_and_sort_left_right_up_down,
)
from app.pymupdf_parser.table_parser import bounded_table
from app.utils.pix2np import pix2np
import awkward as ak
from app.pymupdf_parser.table_parser.unbounded_table_detection import (
    UnboundedTableDetection,
)
from app.pymupdf_parser.utils.filter_item_inside_bbox import (
    filter_lines_which_are_inside_table,
)
from app.pymupdf_parser.table_parser.table_finder import tableFinder

class TableDetection:
    def __init__(self, file_path: str):
        self.doc = fitz.open(file_path)  # type: ignore
        self.image_list = [pix2np(page.get_pixmap()) for page in self.doc]

        # self.unbounded_table_detection = UnboundedTableDetection(
        #     ak_doc,
        #     database_path,
        # )

    def propose(
        self,
    ):  # sourcery skip: inline-immediately-returned-variable, low-code-quality, use-named-expression
        # previous_image_width = np.nan
        document_table: List[List[np.ndarray]] = []
        # horizontal_lines_per_page: Dict[int, OrderedDict[int, LineParameters]] = {}
        # page_width_height_dict: Dict[int, Tuple[float, float]] = {}
        document_boxes: List[List[np.ndarray]] = []
        for page_no, (page, img) in enumerate(zip(self.doc, self.image_list)):
            img_table, img_boxes = tableFinder(img)
            document_table.append(img_table)
            document_boxes.append(img_boxes)
            # table_bboxes = []
            # boxes_bboxes = []
            # image_width = img.shape[1]
            # image_height = img.shape[0]
            # page_width_height_dict[page_no] = (image_width, image_width)
            # # TODO: change the acceptable_size to (min-span-width-in-the-page, 1.2* page-margin-width)
            # horizontal_lines_acceptable_size = (image_width / 30, 0.9 * image_width)
            # # TODO: change the acceptable_size to (min-span-height-in-the-page,  page-margin-height)
            # vertical_lines_acceptable_size = (image_height / 75, 0.9 * image_height)

            # if previous_image_width != image_width:
            #     ver_kernel, hor_kernel, kernel = cv_operations.get_morphological_kernel(
            #         img
            #     )
            #     previous_image_width = image_width
            # img, thresh = bounded_table.adaptive_threshold(img)
            # (
            #     vertical_lines,
            #     horizontal_lines,
            # ) = bounded_table.get_vertical_n_horizontal_lines(
            #     vertical_lines_acceptable_size,
            #     horizontal_lines_acceptable_size,
            #     ver_kernel,  # type: ignore
            #     hor_kernel,  # type: ignore
            #     thresh,
            # )
            # if len(horizontal_lines) and len(vertical_lines):

            #     horizontal_lines = cluster_lines.cluster_lines(
            #         np.array(
            #             [line.three_point_representation for line in horizontal_lines]
            #         ),
            #         LineType.horizontal,
            #     )

            #     vertical_lines = cluster_lines.cluster_lines(
            #         np.array(
            #             [line.three_point_representation for line in vertical_lines]
            #         ),
            #         LineType.vertical,
            #     )

            #     "Draw the detected lines on a black canvas"
            #     img_after_cluster = bounded_table.draw_vertical_n_horizontal_lines(
            #         img, vertical_lines, horizontal_lines
            #     )
            #     new_vertical_lines = (
            #         add_table_boundary.find_left_most_n_right_most_vertical_lines(
            #             img_after_cluster
            #         )
            #     )
            #     if new_vertical_lines:
            #         vertical_lines.extend(new_vertical_lines)
            #         img_after_cluster = bounded_table.draw_vertical_n_horizontal_lines(
            #             img, vertical_lines, horizontal_lines
            #         )

            #     tables, boxes = bounded_table.detect_cell_of_table(
            #         vertical_lines,
            #         horizontal_lines,
            #         img_after_cluster,  # type: ignore
            #         vertical_lines_acceptable_size,
            #         horizontal_lines_acceptable_size,
            #     )
            #     table_bboxes.extend([table.bbox for table in tables])
            #     boxes_bboxes.extend([box.bbox for box in boxes])
            #     # for table in tables:
            #     #     page.add_highlight_annot(table.bbox)
            #     horizontal_lines = filter_lines_which_are_inside_table(
            #         horizontal_lines,
            #         [table.bbox for table in tables] + [box.bbox for box in boxes],
            #     )

            # if len(horizontal_lines):
            #     horizontal_lines_per_page[page_no] = OrderedDict()
            #     for line_no, line in enumerate(horizontal_lines):
            #         horizontal_lines_per_page[page_no][line_no] = line

            #     # for bbox in unbounded_tables_bbox:
            #     #     page.add_highlight_annot(bbox)

            # document_table.append(table_bboxes)
            # document_boxes.append(boxes_bboxes)
        # rhl = RecurringHorizontalLines(
        #     horizontal_lines_per_page, page_width_height_dict
        # )
        # horizontal_lines_per_page = rhl.apply()
        # for page_no, horizontal_lines in horizontal_lines_per_page.items():
        #     if len(horizontal_lines) == 0:
        #         continue
        #     unbounded_tables_bbox = self.unbounded_table_detection.detect(
        #         page_no, list(horizontal_lines.values())
        #     )
        #     table_bboxes = document_table[page_no]
        #     table_bboxes.extend(unbounded_tables_bbox)
            # for bbox in unbounded_tables_bbox:
            #     self.doc[page_no].add_highlight_annot(bbox)
        # for page_no, boxes_bboxes in enumerate(document_boxes):
        #     for bbox in boxes_bboxes:
        #         self.doc[page_no].add_highlight_annot(bbox)
        # self.doc.save("/Users/adarshgupta/Projects/pdf_parser/pdf/unbounded_table.pdf")
        return document_table, document_boxes


# if __name__ == "__main__":
#     btd = TableDetection("/Users/adarshgupta/Projects/pdf_parser/pdf/1706.03762.pdf")
#     btd.propose()
#     print("End")
