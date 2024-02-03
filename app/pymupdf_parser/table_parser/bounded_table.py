from typing import List

import cv2
import numpy as np
from app.pymupdf_parser.table_parser import cv_operations, find_cell_of_table
from app.pymupdf_parser.table_parser.create_table_from_cell_bbox import create_table
from app.pymupdf_parser.table_parser.cv_operations import LineParameters, LineType
from nptyping import NDArray
from skimage import morphology


def detect_cell_of_table(
    vertical_lines: List[LineParameters],
    horizontal_lines: List[LineParameters],
    img_after_cluster: np.ndarray,
    vertical_lines_acceptable_size,
    horizontal_lines_acceptable_size,
):
    horizontal_lines_3_point_rep = np.array(
        [line.three_point_representation for line in horizontal_lines]
    )
    vertical_lines_3_point_rep = np.array(
        [line.three_point_representation for line in vertical_lines]
    )
    intersection_points = np.array(
        list(
            find_cell_of_table.intersection_points_between_lines(
                vertical_lines_3_point_rep, horizontal_lines_3_point_rep
            )
        )
    )
    intersection_groups = find_cell_of_table.group_point_with_same_x_or_y(
        intersection_points
    )

    graph = find_cell_of_table.ConnectedPointGraph(
        intersection_groups, img_after_cluster
    )
    cell_bboxes = list(graph.get_unique_cell_bbox())
    cell_bboxes = np.array(cell_bboxes, dtype=np.int32)

    # cell_image = np.ones(img_after_cluster.shape, dtype=np.uint8) * 255

    # for bbox in table_cell_bboxes:
    #     cell_image: NDArray = cv2.rectangle(
    #         cell_image, bbox[:2], bbox[2:], (0, 255, 0), 1
    #     )

    # return cell_image
    return create_table(
        cell_bboxes,
        vertical_lines_acceptable_size,
        horizontal_lines_acceptable_size,
    )


def adaptive_threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.BORDER_REPLICATE, cv2.THRESH_BINARY_INV, 3, 2
    )

    thresh: NDArray = (
        morphology.remove_small_objects(
            thresh.astype(bool), min_size=10, connectivity=10
        )
        * 255
    ).astype(np.uint8)

    return img, thresh


def get_vertical_n_horizontal_lines(
    vertical_lines_acceptable_size,
    horizontal_lines_acceptable_size,
    ver_kernel,
    hor_kernel,
    thresh,
):
    "Use vertical kernel to detect all  the vertical lines in the image"
    vertical_lines = cv_operations.detect_vertical_n_horizontal_lines(
        thresh, ver_kernel, vertical_lines_acceptable_size, 4, LineType.vertical
    )
    "Use horizontal kernel to detect all  the horizontal lines in the image"
    horizontal_lines = cv_operations.detect_vertical_n_horizontal_lines(
        thresh,
        hor_kernel,
        horizontal_lines_acceptable_size,
        4,
        LineType.horizontal,
    )

    return vertical_lines, horizontal_lines


def draw_vertical_n_horizontal_lines(
    img: NDArray,
    vertical_lines: List[LineParameters],
    horizontal_lines: List[LineParameters],
):
    img_vh = np.zeros(img.shape, dtype=np.uint8)
    img_vh = cv_operations.draw_lines(
        img_vh,
        horizontal_lines + vertical_lines,
    )

    img_vh: np.ndarray = (
        morphology.remove_small_objects(
            img_vh.astype(bool), min_size=100, connectivity=100
        )
        * 255
    ).astype(np.uint8)

    return img_vh
