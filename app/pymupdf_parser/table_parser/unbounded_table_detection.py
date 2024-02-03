from collections import deque
from typing import Deque, List

import awkward as ak
import numpy as np
from matplotlib.pyplot import table

from app.pymupdf_parser.sql_utils.spans_with_bbox import (
    number_of_spans_per_line_inside_bbox,
    put_text_with_their_center_in_database,
)
from app.pymupdf_parser.table_parser.create_table_from_cell_bbox import merge_bboxes
from app.utils.save_to_database import Connection
from app.pymupdf_parser.table_parser.cv_operations import LineParameters


class UnboundedTableDetection:
    def __init__(self, unparsed_contents_ak: ak.Array, db_path: str):
        self.con = Connection(db_path)
        put_text_with_their_center_in_database(unparsed_contents_ak, self.con)

    def detect(self, page_no: int, horizontal_lines: List[LineParameters]):
        horizontal_three_point_rep = np.array(
            [line.three_point_representation for line in horizontal_lines]
        )
        table_bboxes = []
        grouped_sorted_lines = group_and_sort_left_right_up_down(
            horizontal_three_point_rep
        )
        for group in grouped_sorted_lines:
            if len(group) < 3:
                continue
            horizontal_cells_list = self.threshold_on_no_of_spans_per_line(
                page_no, group
            )

            table_bbox = self.collect_cells_if_they_are_continous(horizontal_cells_list)

            table_bboxes.extend(table_bbox)

        return table_bboxes

    def collect_cells_if_they_are_continous(self, horizontal_cells_list):
        table_bbox = []
        for horizontal_cells in horizontal_cells_list:
            if len(horizontal_cells) < 2:
                continue
            bbox = merge_bboxes(horizontal_cells[0], horizontal_cells[1])
            if len(horizontal_cells) > 2:
                for cell_bbox in horizontal_cells[2:]:
                    bbox = merge_bboxes(bbox, cell_bbox)

            table_bbox.append(bbox)
        return table_bbox

    def threshold_on_no_of_spans_per_line(self, page_no, group):
        horizontal_cells_list = [[]]
        for h1, h2 in zip(group[:-1], group[1:]):
            bbox = np.array([h1[1], h1[0], h2[2], h2[0]])
            no_of_spans = number_of_spans_per_line_inside_bbox(self.con, page_no, bbox)
            if len(no_of_spans) > 0 and (
                # Table should contain more than 2 span per line
                np.mean((no_of_spans > 2) * 1)
                >= 0.5
            ):  # 60% of lines should contain more than one span
                horizontal_cells_list[-1].append(bbox)
            else:
                horizontal_cells_list.append([])
        return horizontal_cells_list


def intersection_over_union_of_intervals(
    interval_1: np.ndarray, interval_2: np.ndarray
):
    concated_array = np.concatenate([interval_1, interval_2])
    concated_array = np.sort(concated_array)
    return (concated_array[2] - concated_array[1]) / (
        concated_array[3] - concated_array[0]
    )


def union_of_intervals(interval_1: np.ndarray, interval_2: np.ndarray):
    concated_array = np.concatenate([interval_1, interval_2])
    concated_array = np.sort(concated_array)
    return np.array([concated_array[0], concated_array[3]])


def group_and_sort_left_right_up_down(horizontal_lines: np.ndarray):
    arg_sort = np.lexsort(
        (
            horizontal_lines[:, 1],
            horizontal_lines[:, 0],
        )
    )
    horizontal_lines = horizontal_lines[arg_sort]
    grouped_sorted_lines: List[Deque[np.ndarray]] = []
    grouped_sorted_lines_np: List[np.ndarray] = []
    length = len(horizontal_lines)
    for i in range(length):
        for group in grouped_sorted_lines:
            if (
                intersection_over_union_of_intervals(
                    horizontal_lines[i, 1:], group[0][1:]
                )
                > 0.95
            ):  # threshold
                group[0][1:] = union_of_intervals(group[0][1:], horizontal_lines[i, 1:])
                group.append(horizontal_lines[i])
                break
        else:  # no break
            grouped_sorted_lines.append(deque([horizontal_lines[i]]))
            grouped_sorted_lines[-1].append(horizontal_lines[i])
    for group in grouped_sorted_lines:
        group.popleft()
        grouped_sorted_lines_np.append(np.array(group))
    return grouped_sorted_lines_np
