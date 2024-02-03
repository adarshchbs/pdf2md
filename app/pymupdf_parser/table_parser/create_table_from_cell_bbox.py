from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from app.pymupdf_parser.utils.iou import intersection_over_first_bbox_area


@dataclass
class Table:
    bbox: np.ndarray
    cells: List
    cell_count: int = 1


def create_table(
    bboxes,
    vertical_lines_acceptable_size: Tuple[float, float],
    horizontal_lines_acceptable_size: Tuple[float, float],
):
    bboxes = sorted(bboxes, key=lambda x: x[0])
    bboxes = sorted(bboxes, key=lambda x: x[1])

    tables: List[Table] = []
    for cell_bbox in bboxes:
        for table in tables:
            if (
                is_cell_insecting_with_table(cell_bbox, table.bbox)
                or intersection_over_first_bbox_area(table.bbox, cell_bbox) > 0
            ):
                table.bbox = merge_bboxes(table.bbox, cell_bbox)
                table.cell_count += 1
                table.cells.append(cell_bbox)
                break
        else:  # no break
            tables.append(Table(bbox=cell_bbox, cells=[cell_bbox]))

    tables = filter_table_cells(
        tables, vertical_lines_acceptable_size, horizontal_lines_acceptable_size
    )
    # print(len(tables))
    tables, boxes = remove_table_with_less_than_two_cells(tables)

    return tables, boxes


def is_cell_insecting_with_table(cell_bbox: np.ndarray, table_bbox: np.ndarray):
    threshold = 7
    if (table_bbox[1] > cell_bbox[3] + threshold) or (
        cell_bbox[1] > table_bbox[3] + threshold
    ):
        return 0
    elif (table_bbox[0] > cell_bbox[2] + threshold) or (
        cell_bbox[0] > table_bbox[2] + threshold
    ):
        return 0
    else:
        return 1


def merge_bboxes(bbox_1: np.ndarray, bbox_2: np.ndarray):

    left_upper_point = np.min(np.array([bbox_1[:2], bbox_2[:2]]), axis=0)
    right_lower_point = np.max(np.array([bbox_1[2:], bbox_2[2:]]), axis=0)
    # print(f"{bbox_1=} {bbox_2=} {left_upper_point=} {right_lower_point=} ")

    return np.concatenate([left_upper_point, right_lower_point])


def filter_table_cells(
    tables: List[Table],
    vertical_lines_acceptable_size: Tuple[float, float],
    horizontal_lines_acceptable_size: Tuple[float, float],
):
    for table in tables:
        cells = []
        for cell_bbox in table.cells:
            horizontal_condition = (
                horizontal_lines_acceptable_size[0]
                <= cell_bbox[2] - cell_bbox[0]
                <= horizontal_lines_acceptable_size[1]
            )
            vertical_condition = (
                vertical_lines_acceptable_size[0]
                <= cell_bbox[3] - cell_bbox[1]
                <= vertical_lines_acceptable_size[1]
            )
            if horizontal_condition and vertical_condition:
                cells.append(cell_bbox)

        table.cells = cells
        table.cell_count = len(cells)

    return tables


def remove_table_with_less_than_two_cells(tables: List[Table]):
    return [table for table in tables if table.cell_count >= 2], [
        box for box in tables if box.cell_count == 1
    ]
