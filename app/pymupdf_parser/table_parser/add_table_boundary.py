from collections import Counter
from typing import List

import cv2
import numpy as np
from numpy_groupies.aggregate_numpy import aggregate
from scipy.ndimage import label

from app.pymupdf_parser.table_parser.cv_operations import (
    LineParameters,
    LineType,
    line_parameter_from_three_point_representation,
)


def find_left_most_n_right_most_vertical_lines(img_vh: np.ndarray):
    """
    Finds the leftmost and rightmost vertical lines in an image.

    Args:
        img_vh (np.ndarray): The input image.

    Returns:
        List[LineParameters]: A list of LineParameters representing the leftmost and rightmost vertical lines.
    """
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
    """
    This function finds the leftmost or rightmost vertical line in a set of connected components in an image.

    Parameters:
    x_cordinates (numpy.ndarray): The x coordinates of the pixels in the connected components.
    y_cordinates (numpy.ndarray): The y coordinates of the pixels in the connected components.
    x_counter (collections.Counter): A counter object that counts the occurrences of each x coordinate.
    reverse (bool): A flag that determines whether to find the leftmost line (False) or the rightmost line (True).

    Returns:
    LineParameter or None: A LineParameter object representing the extreme line if one is found, otherwise None.

    The function works by sorting the x coordinates and finding the one that occurs most frequently.
    It considers x coordinates as part of the same line if their absolute difference is less than or equal to 8.
    If the count of the most frequent x coordinate is greater than 1,
    it finds the corresponding y coordinates and returns a LineParameter object representing the line.
    If the count is not greater than 1, it returns None.
    """
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
