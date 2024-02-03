import sys
from os.path import dirname, realpath

sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))
import numpy as np
from nptyping import Int, NDArray, Shape
from numba import njit
from sklearn.cluster import DBSCAN

from app.pymupdf_parser.table_parser.cv_operations import (
    LineType, line_parameter_from_three_point_representation)


@njit()
def is_range_not_intersecting(
    first_range: NDArray[Shape["2"], Int],
    second_range: NDArray[Shape["2"], Int],
):
    if first_range[1] < second_range[0] or first_range[0] > second_range[1]:
        return 1
    else:
        return 0


@njit()
def numba_max(array):
    max_value = array[0]
    for i in range(len(array)):
        if array[i] > max_value:
            max_value = array[i]
    return max_value


@njit
def line_distance(
    first_line: NDArray[Shape["3"], Int], second_line: NDArray[Shape["3"], Int]
) -> float:
    constant_cordinate_distance = np.abs(first_line[0] - second_line[0])
    variable_cordinate_concat = np.sort(
        np.array([first_line[1], first_line[2], second_line[1], second_line[2]])
    )[1:3]
    variable_cordinate_distance = (
        variable_cordinate_concat[1] - variable_cordinate_concat[0]
    )

    return numba_max(
        [
            2 * constant_cordinate_distance,
            variable_cordinate_distance
            * is_range_not_intersecting(first_line[1:], second_line[1:]),
        ]
    )


@njit
def combine_lines(lines: NDArray[Shape["*,3"], Int]):
    constant_cordinate = np.int32(np.mean(lines[:, 0]))
    return np.array(
        [constant_cordinate, np.min(lines[:, 1]), np.max(lines[:, 2])], dtype=np.int32
    )


def cluster_lines(lines: NDArray[Shape["*,3"], Int], line_type: LineType):
    model = DBSCAN(eps=8, min_samples=1, metric=line_distance)  # type: ignore
    cluster_labels = model.fit_predict(lines)
    unique_lines = []
    for label in set(cluster_labels):
        mask = cluster_labels == label
        clustered_lines = lines[mask]
        unique_lines.append(combine_lines(clustered_lines))
    unique_lines = [
        line_parameter_from_three_point_representation(line, line_type)
        for line in unique_lines
    ]
    return unique_lines


if __name__ == "__main__":
    print(line_distance(np.array([25, 3, 5]), np.array([25, 1, 10])))
    print(line_distance(np.array([29, 30, 100]), np.array([20, 50, 150])))
    print(line_distance(np.array([35, 1, 10]), np.array([38, 3, 5])))
    print(line_distance(np.array([12, 3, 5]), np.array([13, 3, 5])))
    print(line_distance(np.array([10, 3, 5]), np.array([20, 7, 10])))
