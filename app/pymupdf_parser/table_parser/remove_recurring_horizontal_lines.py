from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple, Union

from app.pymupdf_parser.table_parser.cv_operations import LineParameters
from app.pymupdf_parser.utils.iou import iou_xaxis
import numpy as np


class RecurringHorizontalLines:
    def __init__(
        self,
        lines_per_page: Dict[int, OrderedDict[int, LineParameters]],
        page_width_height_dict: Dict[int, Tuple[float, float]],
    ) -> None:
        self.minimum_apperance = max([3, (len(page_width_height_dict) - 2) / 3])
        self.minimum_apperance = min([10, self.minimum_apperance])
        self.original_lines_per_page = lines_per_page
        self.lines_per_page = deepcopy(lines_per_page)
        self.page_width_height_dict = page_width_height_dict

    def propose(self):
        for page_no, line_list in self.lines_per_page.items():
            page_height = self.page_width_height_dict[page_no][1]
            filtered_line = OrderedDict()
            for line_no, line in line_list.items():
                if (
                    line.constant_cordinates > 3 * page_height / 4
                    or line.constant_cordinates < page_height / 4
                ):
                    filtered_line[line_no] = line

            self.lines_per_page[page_no] = filtered_line

    def critise(self):
        all_lines = [
            line
            for lines_list in self.lines_per_page.values()
            for line in lines_list.values()
        ]
        for page_no, line_list in self.lines_per_page.items():
            vertical_threshold = self.page_width_height_dict[page_no][1] / 200
            for line_no, line in line_list.items():
                score = sum(
                    iou_xaxis(line.bbox, compare_line.bbox) > 0.95
                    and np.abs(
                        line.constant_cordinates - compare_line.constant_cordinates
                    )
                    < vertical_threshold
                    for compare_line in all_lines
                )

                if score >= self.minimum_apperance:
                    self.original_lines_per_page[page_no].pop(line_no)

    def apply(self):
        self.propose()
        self.critise()
        return self.original_lines_per_page
