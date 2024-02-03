from typing import Any, Dict, Hashable, Tuple, List, Union
import numpy as np
import awkward as ak
from app.pymupdf_parser.utils.cluster import moving_avg_cluster_1d
from collections import OrderedDict
from app.pymupdf_parser.utils.approximate_numbers import ApproximateNumbersWithThreshold
import pandas as pd
from pandasql import sqldf


def get_approx_extrema(array: np.ndarray, return_max: bool):
    cluster_index = moving_avg_cluster_1d(array, 4)  # threshold
    index, counts = np.unique(cluster_index, return_counts=True)
    if return_max:
        index = index[::-1]
        counts = counts[::-1]
    for i, c in zip(index, counts):
        if c > 3:
            c_i = i
            break
    else:  # no break
        return np.nan
    return np.mean(array[cluster_index == c_i])


def horizontal_start_end_of_page(ak_doc: ak.Array) -> dict[int, Tuple[float, float]]:
    margin: Dict[int, Tuple[float, float]] = OrderedDict()
    for page_no, content in enumerate(ak_doc):
        line_bboxes = np.array(
            [r.to_list() for r in ak.flatten(content["blocks", :, "lines", "bbox"])]
        )
        if line_bboxes.any():
            min_margin = get_approx_extrema(line_bboxes[:, 0], return_max=False)
            max_margin = get_approx_extrema(line_bboxes[:, 2], return_max=True)
            margin[page_no] = (min_margin, max_margin)
    return margin


def put_cordinate_information_df(ak_doc):
    x_cordinates = ak_doc[:, "blocks", :, "bbox"]
    x_cordinates_flatten = np.array(
        ak.flatten(x_cordinates)[["0", "2"]].to_list()
    ).flatten()
    apx = ApproximateNumbersWithThreshold(x_cordinates_flatten, threshold=3)
    x_with_no_of_lines = {"x_0": [], "x_1": [], "page_no": [], "no_of_lines": []}
    start_end_of_page = horizontal_start_end_of_page(ak_doc)
    for page_no, content in enumerate(ak_doc):
        if page_no not in start_end_of_page:
            continue
        x_cordinates = content["blocks", :, "bbox"][["0", "2"]].to_list()
        whole_margin = start_end_of_page[page_no]
        x_0 = [apx.approx(x[0]) for x in x_cordinates] + [
            apx.approx(whole_margin[0], False)
        ]
        x_1 = [apx.approx(x[1]) for x in x_cordinates] + [
            apx.approx(whole_margin[1], False)
        ]

        no_of_lines = [len(i) for i in content["blocks", :, "lines"]]
        x_with_no_of_lines["x_0"].extend(x_0)
        x_with_no_of_lines["x_1"].extend(x_1)
        x_with_no_of_lines["page_no"].extend([page_no] * (len(x_cordinates) + 1))
        x_with_no_of_lines["no_of_lines"].extend(no_of_lines + [5])

    x_with_no_of_lines = pd.DataFrame(x_with_no_of_lines)
    return x_with_no_of_lines


def horizontal_margin_per_page(ak_doc: ak.Array) -> Dict[int, Any]:
    x_with_no_of_lines = put_cordinate_information_df(ak_doc)

    filtered_df: pd.DataFrame = sqldf(
        """
        --sql
        select x.x_0,
            x.x_1,
            x.page_no,
            sum(x.no_of_lines) lines_in_this_page,
            max(x.sum_of_lines) lines_in_doc
        from 
        (select x_0,x_1, page_no,no_of_lines,
        sum(no_of_lines) over (partition by x_0,x_1) sum_of_lines
        from x_with_no_of_lines
        order by page_no,sum_of_lines desc, x_0, x_1) x
        where x.sum_of_lines > 5
        group by x.x_0, x.x_1, x.page_no
        order by x.page_no, lines_in_doc desc, x.x_0, x.x_1
        --endsql
        """
    )  # type: ignore

    """to be included in margin -> no of lines
    in each page should be greater than 1"""
    filtered_df = filtered_df[filtered_df.lines_in_this_page > 1]
    margin_appear_in_minimum_k_page = max([1, len(ak_doc) / 4])
    filtered_df: pd.DataFrame = sqldf(
        f"""
        --sql
        -- same margin should appear in atleast len(ak_doc)/4 pages
        select * from
        (
            select *,
            count(page_no) over (partition by x_0,x_1) appear_in_n_page
            from filtered_df
        ) x
        where x.appear_in_n_page >= {margin_appear_in_minimum_k_page}
        order by page_no, lines_in_doc desc, x_0, x_1
        --end-sql
        """
    )  # type: ignore

    grouped = filtered_df.groupby("page_no")
    return {
        key: grouped.get_group(key)[["x_0", "x_1"]].values for key in grouped.groups
    }  # type: ignore


def check_center_align(
    block_x: Union[np.ndarray, List[float]], margin_array: np.ndarray, threshold=4
):
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
