from collections import Counter
from typing import List

import cv2
import numpy as np
from numpy_groupies.aggregate_numpy import aggregate
from scipy.ndimage import label

from app.pymupdf_parser.table_parser.cv_operations import (
    LineParameters, LineType, line_parameter_from_three_point_representation)


def find_left_most_n_right_most_vertical_lines(img_vh: np.ndarray):
    structure = np.ones((3, 3), dtype=np.int32)
    fill_value = -999
    labeled, ncomponents = label(img_vh, structure=structure)  # type: ignore

    new_vertical_lines: List[LineParameters] = []

    for i in range(1, ncomponents + 1):

        i_th_component = np.nonzero(labeled == i)
        y_cordinates, x_cordinates = i_th_component
        min_x_cordinates = aggregate(
            y_cordinates, x_cordinates, "min", fill_value=fill_value
        )
        min_x_cordinates = min_x_cordinates[min_x_cordinates > fill_value]
        max_x_cordinates = aggregate(
            y_cordinates, x_cordinates, "max", fill_value=fill_value
        )
        max_x_cordinates = max_x_cordinates[max_x_cordinates > fill_value]
        min_x_counter = Counter(min_x_cordinates)
        max_x_counter = Counter(max_x_cordinates)

        if left_line := left_most_n_right_most_vertical_line(
            x_cordinates, y_cordinates, min_x_counter, False
        ):
            new_vertical_lines.append(left_line)
        if right_line := left_most_n_right_most_vertical_line(
            x_cordinates, y_cordinates, max_x_counter, True
        ):
            new_vertical_lines.append(right_line)

    return new_vertical_lines


def left_most_n_right_most_vertical_line(
    x_cordinates, y_cordinates, x_counter, reverse: bool
):
    x_cordinate_sorted = sorted(x_counter.keys(), reverse=reverse)
    extreme_x = x_cordinate_sorted[0]
    extreme_x_count = x_counter[extreme_x]
    extreme_x_array = [extreme_x]
    for x in x_cordinate_sorted[1:]:
        if np.abs(extreme_x - x) > 8:  # threshold
            break
        if x_counter[x] > x_counter[extreme_x]:
            extreme_x = x
        extreme_x_count += x_counter[x]
        extreme_x_array.append(x)
    if extreme_x_count > 1:
        corresponding_y_cordinates = []
        for x in extreme_x_array:
            corresponding_y_cord = y_cordinates[np.nonzero(x_cordinates == x)[0]]
            corresponding_y_cordinates.extend(corresponding_y_cord)

        min_y = min(corresponding_y_cordinates)
        max_y = max(corresponding_y_cordinates)

        return line_parameter_from_three_point_representation(
            np.array([extreme_x, min_y, max_y]), LineType.vertical
        )
    else:
        return None
