import numpy as np
from numba import njit
from app.pymupdf_parser.utils.iou import range_intersection
from typing import Union
import pandas as pd


@njit
def moving_avg_cluster_1d(
    array: np.ndarray, threshold: Union[float, int] = 5
) -> np.ndarray:
    """
    It takes an array and a threshold, and returns an array of the same length, where each element is
    the index of the cluster that the corresponding element of the input array belongs to

    :param array: 1-D array to be clustered
    :param threshold: The threshold for the moving average, defaults to 5
    :return: The index of the cluster that each element belongs to.
    """
    length = len(array)
    sort_undo_arg = np.argsort(np.argsort(array))
    array = np.sort(array)

    index_array = np.zeros(length, dtype=np.int32)
    index = 0

    index_array[0] = 0
    moving_avg = array[0]
    moving_count = 1
    for i in range(1, length):
        if np.abs(array[i] - moving_avg) > threshold:
            index += 1
            moving_avg = array[i]
            moving_count = 1
        else:
            moving_count += 1
            moving_avg += (array[i] - moving_avg) / moving_count

        index_array[i] = index
        # print(f"{array[i]=} , {moving_avg=} , {moving_count=} ")

    return index_array[sort_undo_arg]


@njit
def dbscan_1d(array: np.ndarray, threshold=5) -> np.ndarray:
    """
    It takes an 1-D array, and assigns a cluster index to each element based on the distance
    between it and the previous element

    :param array: 1-D array to be clustered
    :param threshold: the maximum distance between two points to be considered in the same cluster,
    defaults to 5 (optional)
    :return: The index of the cluster that each point belongs to.
    """
    length = len(array)
    sort_undo_arg = np.argsort(np.argsort(array))
    array = np.sort(array)

    index_array = np.zeros(length, dtype=np.int32)
    index = 0

    index_array[0] = 0
    for i in range(1, length):
        if np.abs(array[i] - array[i - 1]) > threshold:
            index += 1

        index_array[i] = index

    return index_array[sort_undo_arg]


def cluster_range(array: np.ndarray) -> np.ndarray:
    """
    It takes an 2-D array of size (N,2) and assigns each range to a cluster based on whether
    it overlaps with the previous range

    :param array: the array of ranges to cluster
    :type array: np.ndarray[N,2]
    :return: an array of integers, where each integer represents the cluster that the corresponding
    element of the input array belongs to.
    """
    length = len(array)
    sort_arg = np.argsort(array[:, 0])
    sort_undo_arg = np.argsort(sort_arg)

    array = array[sort_arg]

    index_array = np.zeros(length, dtype=np.int32)
    index = 0
    index_array[0] = 0

    for i in range(1, length):
        if range_intersection(array[i], array[i - 1]) < 0.1:  # threshold parameter
            index += 1

        index_array[i] = index

    return index_array[sort_undo_arg]
