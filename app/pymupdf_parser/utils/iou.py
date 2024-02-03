"""Copyright (C) 2022 Adarsh Gupta"""
import numpy as np
from numba import njit, prange
from typing import Dict, Tuple


@njit()
def range_intersection(range_1, range_2) -> float:
    """
    It takes two ranges (e.g. [0, 1] and [0.5, 1.5]) and returns the fraction of the first range that is
    covered by the second range

    For ex: [0, 1] and [0.5, 1.5] return 1/3
    """
    if (range_2[0] >= range_1[1]) or (range_1[0] >= range_2[1]):
        return 0

    cordinates = np.array([range_1[0], range_1[1], range_2[0], range_2[1]])
    cordinates = np.sort(cordinates)
    return (cordinates[2] - cordinates[1]) / (cordinates[3] - cordinates[0])


@njit()
def iou_xaxis(bbox_1, bbox_2) -> float:
    """
    > bbox = [xmin, ymin, xmax, ymax]
    > Given two bounding boxes, return the intersection over union of the two boxes along the x-axis
    """
    if (bbox_2[0] >= bbox_1[2]) or (bbox_1[0] >= bbox_2[2]):
        return 0

    x_cordinates = np.array([bbox_1[0], bbox_1[2], bbox_2[0], bbox_2[2]])
    x_cordinates = np.sort(x_cordinates)
    return (x_cordinates[2] - x_cordinates[1]) / (x_cordinates[3] - x_cordinates[0])


@njit()
def iou_yaxis(bbox_1, bbox_2) -> float:
    """
    > bbox = [xmin, ymin, xmax, ymax]
    > Given two bounding boxes, return the intersection over union of the two boxes along the y-axis
    """
    if (bbox_2[1] >= bbox_1[3]) or (bbox_1[1] >= bbox_2[3]):
        return 0

    y_cordinates = np.array([bbox_1[1], bbox_1[3], bbox_2[1], bbox_2[3]])
    y_cordinates = np.sort(y_cordinates)
    return (y_cordinates[2] - y_cordinates[1]) / (y_cordinates[3] - y_cordinates[0])


@njit()
def intersection_over_first_bbox_area(bbox_1, bbox_2) -> float:
    """
    > If the two bboxes don't overlap, return 0. Otherwise, return the area of the intersection divided
    by the area of the first bbox

    :param bbox_1: The first bounding box
    :param bbox_2: the bbox that we want to check if it's inside the first bbox
    :return: The area of the intersection of the two bounding boxes
    """
    if (bbox_2[1] >= bbox_1[3]) or (bbox_1[1] >= bbox_2[3]):
        return 0
    elif (bbox_2[0] >= bbox_1[2]) or (bbox_1[0] >= bbox_2[2]):
        return 0

    x_cordinates = np.array([bbox_1[0], bbox_1[2], bbox_2[0], bbox_2[2]])
    x_cordinates = np.sort(x_cordinates)
    x_intersection = (x_cordinates[2] - x_cordinates[1]) / (bbox_1[2] - bbox_1[0])

    y_cordinates = np.array([bbox_1[1], bbox_1[3], bbox_2[1], bbox_2[3]])
    y_cordinates = np.sort(y_cordinates)
    y_intersection = (y_cordinates[2] - y_cordinates[1]) / (bbox_1[3] - bbox_1[1])

    return x_intersection * y_intersection


@njit(parallel=True)
def iou_over_list_of_bbox(list_bbox_1, list_bbox_2):
    """
    For each pair of bounding boxes, we calculate the intersection over union (IoU) of the two bounding
    boxes.

    If the IoU is greater than 0.5, we set the value of the corresponding cell in the matrix to the IoU.
    Otherwise, we set the value to 0.

    The matrix is symmetric, so we only need to calculate the upper triangle.

    We use the `numba` library to speed up the function.

    :param list_bbox_1: list of bounding boxes, each bounding box is a list of 4 elements: [xmin, ymin,
    xmax, ymax]
    :param list_bbox_2: list of bounding boxes
    :return: the intersection over union of two bounding boxes.
    """
    ret = np.zeros((len(list_bbox_1), len(list_bbox_1)))
    for i in prange(len(list_bbox_1)):
        for j in prange(i):
            value = iou_xaxis(list_bbox_1[i], list_bbox_2[j]) * iou_yaxis(
                list_bbox_1[i], list_bbox_2[j]
            )
            if value > 0.5:
                ret[i, j] = value
                ret[j, i] = value

    return ret


def convert_array_to_dict(ret) -> Dict[Tuple[int, int], float]:
    """
    It takes a 2D array and returns a dictionary where the keys are the non-zero indices of the array
    and the values are the corresponding values in the array

    :param ret: the 2-D matrix
    :return: A dictionary of the non-zero elements of the matrix.
    """
    ret_dict = {}
    non_zeros_index = np.nonzero(ret)
    for i in range(len(non_zeros_index[0])):
        x, y = non_zeros_index[0][i], non_zeros_index[1][i]
        ret_dict[(x, y)] = ret[x, y]
        ret_dict[(y, x)] = ret[x, y]
    return ret_dict
