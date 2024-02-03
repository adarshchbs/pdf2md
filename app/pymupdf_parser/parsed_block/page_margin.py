"""Copyright (C) 2022 Adarsh Gupta"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import fitz
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN

# from pyrsistent import T
from app.pymupdf_parser.parameter.pymupdf_content import (PyMuPdfContent,
                                                          _Blocks)


def min_approx_x_with_minimum_k_occurrences(x: List[float], k: int, threshold: float):
    count = Counter(x)
    sorted_keys = sorted(count.keys())
    for i, key in enumerate(sorted_keys):
        relevant_keys = [key]
        for j in range(i - 1, -1, -1):
            if np.abs(sorted_keys[j] - key) < threshold:
                relevant_keys.append(sorted_keys[j])
            else:
                break
        for j in range(i + 1, len(sorted_keys)):
            if np.abs(sorted_keys[j] - key) < threshold:
                relevant_keys.append(sorted_keys[j])
            else:
                break

        n = sum(count[j] for j in relevant_keys)

        if n >= k:
            return key

    return np.nan


def get_start_end_x(unparsed_contents: List[PyMuPdfContent], file_path: str):
    doc = fitz.open(file_path)
    start_end_x_list: List[List[float]] = []
    for page, content in zip(doc, unparsed_contents):
        page_block_margin = []
        width = page.rect.width
        previous_block = None
        for block in content.blocks:
            if previous_block is not None:
                bbox = previous_block.lines[-1].spans[-1].bbox
                span_height = bbox[3] - bbox[1]
                if (
                    (previous_block.bbox[3] - 0.2 * span_height)
                    > block.bbox[1]
                    > (previous_block.bbox[3] - 6 * span_height)
                ):
                    continue
            if (block.bbox[2] - block.bbox[0]) > width / 5:
                # print(len(block.lines), [block.bbox[0], block.bbox[2]])
                page_block_margin.extend(
                    [block.bbox[0], block.bbox[2]] for _ in range(len(block.lines))
                )

            previous_block = block
        start_end_x_list.extend(page_block_margin)
    return start_end_x_list


def get_page_margin(unparsed_contents: List[PyMuPdfContent], file_path: str):

    start_end_x_list = get_start_end_x(unparsed_contents, file_path)
    if len(unparsed_contents) < 8:
        min_samples = len(start_end_x_list) // 6
    elif 7 < len(unparsed_contents) < 20:
        min_samples = len(start_end_x_list) // 8
    else:
        min_samples = len(start_end_x_list) // 10
    cluster = DBSCAN(eps=4, min_samples=min_samples)
    cluster.fit(np.array(start_end_x_list))
    labels = defaultdict(list)
    for i, x_core in enumerate(cluster.components_):
        labels[cluster.labels_[cluster.core_sample_indices_[i]]].append(x_core)
    margin: List[float] = [
        stats.mode(np.array(value))[0][0] for key, value in labels.items()
    ]

    return np.array(sorted(margin, key=lambda x: x[0]))


if __name__ == "__main__":
    print(min_approx_x_with_minimum_k_occurrences(np.random.random(100), 4, 0.1))
