from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
from nptyping import Int, NDArray, Shape


class LineType(Enum):
    horizontal = 1
    vertical = 2


@dataclass(frozen=True)
class LineParameters:
    line_type: LineType
    constant_cordinates: np.int32
    variable_cordinates: NDArray[Shape["2"], Int]
    three_point_representation: NDArray[
        Shape["[constant_cordinate,variable_cordinate_0,variable_cordinate_1]"], Int
    ]
    bbox: NDArray[Shape["[x_0,y_0,x_1,y_2]"], Int]


def line_parameter_from_bbox(line: NDArray[Shape["4"], Int], line_type: LineType):
    if line_type is LineType.vertical:
        constant_cordinates = line[0]
        variable_cordinates: NDArray[Shape["2"], Int] = np.array([line[1], line[3]])

    elif line_type is LineType.horizontal:
        constant_cordinates = line[1]
        variable_cordinates: NDArray[Shape["2"], Int] = np.array([line[0], line[2]])
    else:
        raise ValueError("Not correct line shape")

    three_point_representation = np.hstack([constant_cordinates, variable_cordinates])

    if line_type is LineType.vertical:
        bbox = np.array(
            [
                constant_cordinates,
                variable_cordinates[0],
                constant_cordinates + 1,
                variable_cordinates[1],
            ]
        )
    else:
        bbox = np.array(
            [
                variable_cordinates[0],
                constant_cordinates,
                variable_cordinates[1],
                constant_cordinates + 1,
            ]
        )

    return LineParameters(
        deepcopy(line_type),
        constant_cordinates,
        variable_cordinates,
        three_point_representation,
        bbox,
    )


def line_parameter_from_three_point_representation(
    line: NDArray[Shape["3"], Int], line_type: LineType
):
    if line_type is LineType.vertical:
        bbox = np.array([line[0], line[1], line[0] + 1, line[2]])
    elif line_type is LineType.horizontal:
        bbox = np.array([line[1], line[0], line[2], line[0] + 1])
    else:
        raise ValueError(f"{line_type} is not correct line shape")

    return LineParameters(deepcopy(line_type), line[0], line[1:], line, bbox)


def get_morphological_kernel(img: NDArray):
    "Length(width) of kernel as 150th of total width"
    kernel_len = np.array(img).shape[1] // 150
    "Defining a vertical kernel to detect all vertical lines of image"
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    " Defining a horizontal kernel to detect all horizontal lines of image"
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    "A kernel of 2x2"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return ver_kernel, hor_kernel, kernel


def detect_vertical_n_horizontal_lines(
    img: NDArray,
    kernel,
    acceptable_size: Tuple[float, float],
    iterations: int,
    kernel_type: LineType,
):
    img = deepcopy(img)
    """Possible value of "kernel_type" variable are 'vertical' and 'horizontal'"""
    image_1 = cv2.erode(img, kernel, iterations=iterations)
    detected_lines = cv2.dilate(image_1, kernel, iterations=iterations)
    detected_lines, _ = cv2.findContours(
        detected_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    detected_lines = [cv2.boundingRect(c) for c in detected_lines]
    return filter_lines_outside_of_acceptable_range(
        acceptable_size, kernel_type, detected_lines
    )


def filter_lines_outside_of_acceptable_range(
    acceptable_size, kernel_type, detected_lines
):
    detected_lines_ret: List[LineParameters] = []
    for c in detected_lines:
        if (
            kernel_type is LineType.vertical
            and acceptable_size[0] < c[3] < acceptable_size[1]
        ):
            detected_lines_ret.append(
                line_parameter_from_bbox(
                    np.array([c[0], c[1], c[0], c[1] + c[3]]), kernel_type
                )
            )
        if (
            kernel_type is LineType.horizontal
            and acceptable_size[0] < c[2] < acceptable_size[1]
        ):
            detected_lines_ret.append(
                line_parameter_from_bbox(
                    np.array([c[0], c[1], c[0] + c[2], c[1]]), kernel_type
                )
            )
    return detected_lines_ret


def draw_lines(
    input_image: NDArray,
    bboxes: List[LineParameters],
):
    """Possible value of "line_type" variable are 'vertical' and 'horizontal'"""
    for line in bboxes:
        bbox = line.bbox
        input_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
    return input_image


if __name__ == "__main__":
    line = line_parameter_from_bbox(np.array([10, 20, 10, 30]), LineType.vertical)
    print(line)
    print(
        line_parameter_from_three_point_representation(
            line.three_point_representation, LineType.vertical
        )
    )

    line = line_parameter_from_bbox(np.array([10, 20, 30, 20]), LineType.horizontal)
    print(line)
    print(
        line_parameter_from_three_point_representation(
            line.three_point_representation, LineType.horizontal
        )
    )
