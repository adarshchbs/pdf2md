from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import awkward as ak
import numpy as np
import polars as pl

from app.pymupdf_parser.utils.approximate_numbers import ApproximateNumbersWithThreshold
from app.pymupdf_parser.utils.cluster import moving_avg_cluster_1d


def get_approx_extrema(array: np.ndarray, return_max: bool):
    cluster_index = moving_avg_cluster_1d(array, 4)  # threshold
    index, counts = np.unique(cluster_index, return_counts=True)
    if return_max:
        index = index[::-1]
        counts = counts[::-1]
    for i, c in zip(index, counts, strict=True):
        if c > 3:
            c_i = i
            break
    else:  # no break
        return np.nan
    return np.mean(array[cluster_index == c_i])


def horizontal_start_end_of_page(ak_doc: ak.Array) -> dict[int, Tuple[float, float]]:
    margin: Dict[int, Tuple[float, float]] = OrderedDict()
    for page_no, content in enumerate(ak_doc):
        line_bboxes = np.array([r.to_list() for r in ak.flatten(content["blocks", :, "lines", "bbox"])])
        if line_bboxes.any():
            min_margin = get_approx_extrema(line_bboxes[:, 0], return_max=False)
            max_margin = get_approx_extrema(line_bboxes[:, 2], return_max=True)
            margin[page_no] = (min_margin, max_margin)
    return margin


def put_cordinate_information_df(ak_doc):
    x_cordinates = ak_doc[:, "blocks", :, "bbox"]
    x_cordinates_flatten = np.array(ak.flatten(x_cordinates)[["0", "2"]].to_list()).flatten()
    apx = ApproximateNumbersWithThreshold(x_cordinates_flatten, threshold=3)
    x_with_no_of_lines = {"x_0": [], "x_1": [], "page_no": [], "no_of_lines": []}
    start_end_of_page = horizontal_start_end_of_page(ak_doc)
    for page_no, content in enumerate(ak_doc):
        if page_no not in start_end_of_page:
            continue
        x_cordinates = content["blocks", :, "bbox"][["0", "2"]].to_list()
        whole_margin = start_end_of_page[page_no]
        x_0 = [apx.approx(x[0]) for x in x_cordinates] + [apx.approx(whole_margin[0], False)]
        x_1 = [apx.approx(x[1]) for x in x_cordinates] + [apx.approx(whole_margin[1], False)]

        no_of_lines = [len(i) for i in content["blocks", :, "lines"]]
        x_with_no_of_lines["x_0"].extend(x_0)
        x_with_no_of_lines["x_1"].extend(x_1)
        x_with_no_of_lines["page_no"].extend([page_no] * (len(x_cordinates) + 1))
        x_with_no_of_lines["no_of_lines"].extend(no_of_lines + [5])

    return pl.DataFrame(x_with_no_of_lines)


def horizontal_margin_per_page(ak_doc: ak.Array) -> Dict[int, Any]:
    x_with_no_of_lines = put_cordinate_information_df(ak_doc)

    filtered_df = (
        x_with_no_of_lines
        .with_columns(pl.col("no_of_lines").sum().over("x_0", "x_1").alias("sum_of_lines"))
        .filter(pl.col("sum_of_lines") > 5)
        .group_by("x_0", "x_1", "page_no")
        .agg(
            pl.col("no_of_lines").sum().alias("lines_in_this_page"),
            pl.col("sum_of_lines").max().alias("lines_in_doc"),
        )
        .filter(pl.col("lines_in_this_page") > 1)
    )

    margin_appear_in_minimum_k_page = max(1, len(ak_doc) / 4)
    filtered_df = (
        filtered_df
        .with_columns(pl.col("page_no").count().over("x_0", "x_1").alias("appear_in_n_page"))
        .filter(pl.col("appear_in_n_page") >= margin_appear_in_minimum_k_page)
        .sort(
            ["page_no", "lines_in_doc", "x_0", "x_1"],
            descending=[False, True, False, False],
        )
    )

    return {
        key[0]: group.select("x_0", "x_1").to_numpy()
        for key, group in filtered_df.group_by("page_no", maintain_order=True)
    }


def check_center_align(block_x: Union[np.ndarray, List[float]], margin_array: np.ndarray, threshold=4):
    block_x = np.array(block_x)
    is_center_align = False
    for m in margin_array:
        diff = block_x - m
        abs_diff = np.abs(diff)
        if (
            diff[0] > threshold
            and -1 * diff[1] > threshold
            # and np.abs(np.mean(block_x) - np.mean(m)) < threshold
            and np.abs(abs_diff[0] - abs_diff[1]) < threshold
        ):
            is_center_align = True

    return is_center_align
