"""Copyright (C) 2022 Adarsh Gupta"""
from typing import List, Tuple
import numpy as np


def combine_span_bboxes(bboxes: List[np.ndarray]):
    """
    It takes a list of bounding boxes, and returns
    a single bounding box that contains all of them
    """
    return np.array([bboxes[0][0], bboxes[0][1], bboxes[-1][2], bboxes[0][3]])


def combine_line_bboxes(bboxes: List[np.ndarray]):
    """
    It takes a list of bounding boxes, and returns
    a single bounding box that contains all of them
    """
    x_last = max(x[2] for x in bboxes)
    x_start = min(x[0] for x in bboxes)
    return np.array([x_start, bboxes[0][1], x_last, bboxes[-1][3]])
