from itertools import product

import numpy as np

from app.pymupdf_parser.utils.iou import intersection_over_first_bbox_area


def line_inside_horizontal_span(
    vertical_lines: np.ndarray, horizontal_lines: np.ndarray
):

    if top_horizontal_span := vertical_lines[1] == horizontal_lines[0]:
        return 1, horizontal_lines[0]
    elif (
        inside_horizontal_span := vertical_lines[1]
        < horizontal_lines[0]
        < vertical_lines[2]
    ):
        return 2, horizontal_lines[0]
    elif bottom_horizontal_span := vertical_lines[2] == horizontal_lines[0]:
        return 3, horizontal_lines[0]
    else:
        return 0, np.nan


def line_inside_vertical_span(vertical_lines: np.ndarray, horizontal_lines: np.ndarray):

    if left_vertical_span := horizontal_lines[1] == vertical_lines[0]:
        return 1, vertical_lines[0]
    elif (
        inside_vertical_span := horizontal_lines[1]
        < vertical_lines[0]
        < horizontal_lines[2]
    ):
        return 2, vertical_lines[0]

    elif right_vertical_span := horizontal_lines[2] == vertical_lines[0]:
        return 3, vertical_lines[0]
    else:
        return 0, np.nan


def start_end_points(
    vertical_lines_array: np.ndarray, horizontal_lines_array: np.ndarray
):
    cell_start_cord = []
    cell_end_cord = []

    for h, v in product(horizontal_lines_array, vertical_lines_array):
        k1, hi = line_inside_horizontal_span(v, h)
        k2, vi = line_inside_vertical_span(v, h)
        if (k1, k2) in {(1, 1), (1, 2), (2, 1), (2, 2)}:
            cell_start_cord.append(((vi, hi)))
        if (k1, k2) in {(3, 3), (3, 2), (2, 3), (2, 2)}:
            cell_end_cord.append(((vi, hi)))

    return np.array(cell_start_cord), np.array(cell_end_cord)


def check_if_lines_exists_between_points(
    image: np.ndarray,
    start_point: np.ndarray,
    end_point: np.ndarray,
    threshold: int = 4,
):
    distance = np.abs(end_point[1] - start_point[1])
    left_line_exists: bool = np.sum(
        image[
            start_point[1] : end_point[1],
            start_point[0] - threshold : start_point[0] + threshold,
        ]
    ) > max([0.9 * distance, distance - 15])

    right_line_exists: bool = np.sum(
        image[
            start_point[1] : end_point[1],
            end_point[0] - threshold : end_point[0] + threshold,
        ]
    ) > max([0.9 * distance, distance - 15])

    distance = np.abs(end_point[0] - start_point[0])
    up_line_exists: bool = np.sum(
        image[
            start_point[1] - threshold : start_point[1] + threshold,
            start_point[0] : end_point[0],
        ]
    ) > max([0.9 * distance, distance - 15])

    bottom_line_exists: bool = np.sum(
        image[
            end_point[1] - threshold : end_point[1] + threshold,
            start_point[0] : end_point[0],
        ]
    ) > max([0.9 * distance, distance - 15])

    return (
        left_line_exists and right_line_exists and up_line_exists and bottom_line_exists
    )


def remove_overlapping_cells(cell_bboxes: np.ndarray):
    mask = np.ones(cell_bboxes.shape[0], dtype=bool)
    discard_cell = set()
    for i in range(cell_bboxes.shape[0]):
        bbox_1 = cell_bboxes[i]
        for j in range(cell_bboxes.shape[0]):
            if i == j or j in discard_cell:
                continue
            bbox_2 = cell_bboxes[j]
            intersection = intersection_over_first_bbox_area(bbox_2, bbox_1)

            if intersection > 0.9:
                print(f"{bbox_1=}   {bbox_2=} {intersection=}")
                mask[i] = False
                discard_cell.add(i)
                break
    return cell_bboxes[mask]


def get_cell_bboxes(
    img_vh: np.ndarray, cell_start_cord: np.ndarray, cell_end_cord: np.ndarray
):
    cell_bboxes = np.zeros((cell_start_cord.shape[0], 4))
    for i in range(cell_start_cord.shape[0]):
        start = cell_start_cord[i]
        best_end_distance = np.infty
        best_end_cord = np.zeros(2)
        for j in range(cell_end_cord.shape[0]):
            end = cell_end_cord[j]
            if (
                end[0] > start[0]
                and end[1] > start[1]
                and (distance := np.linalg.norm(end - start)) < best_end_distance
                and check_if_lines_exists_between_points(img_vh, start, end)
            ):
                best_end_cord = end
                best_end_distance = distance

        if best_end_distance < np.infty:
            cell_bboxes[i] = np.concatenate((start, best_end_cord))

    cell_bboxes: np.ndarray = cell_bboxes[np.sum(cell_bboxes, axis=1) > 0]
    cell_bboxes = remove_overlapping_cells(cell_bboxes)
    return cell_bboxes
