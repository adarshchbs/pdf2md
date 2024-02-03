"""Copyright (C) 2022 Adarsh Gupta"""

import itertools
import logging
from typing import List, Tuple

import numpy as np
from nptyping import NDArray


def check_x_alignment(bbox_list: NDArray, font_list: List[Tuple], margin: NDArray):
    """
    It checks if any two or more boxes are aligned in the x-axis upto a threshold

    :param bbox_list: a list of bounding boxes
    :param font_list: a list of (font name and size) corresponding to the bboxes
    :param margin: horizontal margin of the page
    :return: A boolean array of length n, where n is the number of bounding boxes.
    """
    margin_distance = set()
    for m1 in margin[:, 0]:
        for m2 in margin[:, 0]:
            margin_distance.add(np.abs(m1 - m2))

    n = len(bbox_list)
    is_font_same = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        if font_list[i] == font_list[j] and i != j:
            is_font_same[i, j] = 1

    is_x_align = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        if i != j:
            distance = np.min(
                [np.abs(bbox_list[i][0] - bbox_list[j][0] - m) for m in margin_distance]
                + [
                    np.abs(bbox_list[i][0] - bbox_list[j][0] + m)
                    for m in margin_distance
                ]
            )
            is_x_align[i, j] = np.exp(
                -distance / (2 * bbox_list[i][3] - bbox_list[i][1])
            )

    alignment = is_font_same * is_x_align
    alignment = np.sum(alignment, axis=1)

    alignment = alignment > 0.5

    return alignment
