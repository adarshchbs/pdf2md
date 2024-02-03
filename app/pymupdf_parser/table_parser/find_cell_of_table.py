from collections import OrderedDict, defaultdict
from typing import List, Set, Tuple

import numpy as np
from sortedcontainers import SortedSet


def group_point_with_same_x_or_y(points: np.ndarray):
    same_x_points = defaultdict(lambda: SortedSet(key=lambda x: x[1]))
    same_y_points = defaultdict(lambda: SortedSet(key=lambda x: x[0]))

    for p in points:
        p: Tuple[int, int] = tuple(p)
        same_x_points[p[0]].add(p)  # type: ignore
        same_y_points[p[1]].add(p)  # type: ignore

    ret_list = list(same_x_points.values())
    ret_list.extend(iter(same_y_points.values()))
    return ret_list


def intersection_points_between_two_lines(
    vertical_line: np.ndarray, horizontal_line: np.ndarray, threshold: int
):
    intersection_point = np.ones(2) * np.nan
    if vertical_line[1] - threshold < horizontal_line[0] < vertical_line[2] + threshold:
        intersection_point[1] = horizontal_line[0]
    if (
        horizontal_line[1] - threshold
        < vertical_line[0]
        < horizontal_line[2] + threshold
    ):
        intersection_point[0] = vertical_line[0]

    return intersection_point


def does_line_exists_between_point(img_vh: np.ndarray, point_1, point_2, threshold=5):
    x1, x2 = point_1[0], point_2[0] + 1
    y1, y2 = point_1[1], point_2[1] + 1
    point_1, point_2 = np.array(point_1), np.array(point_2)
    distance_between_points = np.sum(np.abs(point_2 - point_1))  # type: ignore
    return np.sum(img_vh[y1:y2, x1:x2]) / 255 > (distance_between_points - threshold)


def intersection_points_between_lines(
    vertical_lines: np.ndarray, horizontal_lines: np.ndarray
):
    for vl in vertical_lines:
        for hl in horizontal_lines:
            intersection_point = intersection_points_between_two_lines(vl, hl, 5)
            if not np.any(np.isnan(intersection_point)):
                yield np.array(intersection_point, dtype=np.int32)


class ConnectedPointGraph:
    def __init__(self, grouped_points_list: List[SortedSet], img_vh: np.ndarray):
        self.direction = OrderedDict(
            [
                (1, [2, 1]),
                (2, [-1, 2]),
                (-1, [-2, -1]),
            ]
        )

        self.graph = defaultdict(dict)
        for sorted_list in grouped_points_list:
            for point_1, point_2 in zip(sorted_list[:-1], sorted_list[1:]):
                if does_line_exists_between_point(img_vh, point_1, point_2, 5):
                    if point_1[1] == point_2[1]:  # same y
                        self.graph[point_1][1] = point_2
                        self.graph[point_2][-1] = point_1
                    elif point_1[0] == point_2[0]:  # same x
                        self.graph[point_1][2] = point_2
                        self.graph[point_2][-2] = point_1

    @property
    def nodes(self):
        return list(self.graph)

    def find_cell(self, point):  # sourcery skip: low-code-quality
        if 1 not in self.graph[tuple(point)]:
            return None
        next_point = self.graph[tuple(point)][1]
        cell_points = [next_point]
        direction = iter(self.direction.items())
        _, index = next(direction)
        while True:
            first_option, second_option = index[0], index[1]

            if first_option in self.graph[next_point]:
                next_point = self.graph[next_point][first_option]
                cell_points.append(next_point)
                if first_option == -2:
                    if next_point[1] <= point[1]:
                        return cell_points if next_point == point else None
                    else:
                        continue
                _, index = next(direction)

            elif second_option in self.graph[next_point]:
                next_point = self.graph[next_point][second_option]
            else:
                return None

    def get_bbox(self, vertexes: np.ndarray):
        assert vertexes.shape[1] == 2 and vertexes.shape[0] >= 4
        left_upper_point = np.min(vertexes, axis=0)
        right_lower_point = np.max(vertexes, axis=0)

        return np.concatenate([left_upper_point, right_lower_point])

    def get_unique_cell_bbox(
        self,
    ):
        unique_bbox: Set[Tuple[int, int, int, int]] = set()
        for node in self.nodes:
            if cell_vertexes := self.find_cell(node):
                bbox = self.get_bbox(np.array(cell_vertexes, dtype=int))
                bbox = tuple(bbox)
                unique_bbox.add(bbox)

        return unique_bbox
