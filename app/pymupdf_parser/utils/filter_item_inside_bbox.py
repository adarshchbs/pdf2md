from typing import List

import numpy as np
from app.pymupdf_parser.utils.iou import intersection_over_first_bbox_area
from app.pymupdf_parser.table_parser.cv_operations import LineParameters
import numpy as np
from functools import partial


def filter_lines_which_are_inside_table(
    lines: List[LineParameters], table_bbox_list: List[np.ndarray]
):
    """
    > For each line, check if it is inside any of the table bounding boxes. If it is, then don't add it
    to the list of filtered lines

    :param lines: List[LineParameters]
    :param table_bbox_list: list of bounding boxes of tables
    :return: a list of lines that are not inside a table.
    """
    t = 5
    filtered_lines = []
    for line in lines:
        for bbox in table_bbox_list:
            bbox = np.array(
                [bbox[0] - t, bbox[1] - 2 * t, bbox[2] + t, bbox[3] + 2 * t]
            )
            if intersection_over_first_bbox_area(line.bbox, bbox) > 0.6:
                break
        else:  # no break
            filtered_lines.append(line)

    return filtered_lines
