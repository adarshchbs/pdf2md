"""Copyright (C) 2022 Adarsh Gupta"""
import logging
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN

from app.pymupdf_parser.combine_blocks.merge_consecutive_block import (
    merge_consecutive_blocks,
)
from app.pymupdf_parser.parameter.parsed_content import Block
from app.pymupdf_parser.utils.font_characteristics import (
    block_font_characteristics,
    font_characteristics,
)


def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):  # type: ignore
    # sourcery skip: inline-immediately-returned-variable, move-assign-in-block, use-next
    y_new = -1
    X_new = np.array([X_new])
    # Find a core sample closer than EPS
    for i, x_core in enumerate(dbscan_model.components_):
        if metric(X_new, x_core) < dbscan_model.eps:
            # Assign label of x_core to x_new
            y_new = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
            break

    return y_new


def create_clusters(spacing_stats: Dict[Tuple[int, float], List[float]]):
    """
    It takes a dictionary of lists of floats, and returns a dictionary of DBSCAN models

    :param spacing_stats: A dictionary of tuples (font_as_int, font_size) to a list of line spacings
    :type spacing_stats: Dict[Tuple[int, float], List[float]]
    :return: A dictionary of DBSCAN models, where the key is a tuple of (font_as_int, font_size) and
    the value is the DBSCAN model.
    """
    dbscan_models = {}
    for key, array in spacing_stats.items():
        eps = max(array) / 100
        cluster = DBSCAN(eps=eps, min_samples=3)
        cluster.fit(np.array(array).reshape(-1, 1))
        dbscan_models[key] = cluster

    return dbscan_models


def combine_blocks_with_linespacing(
    content: List[Block],
    dbscan_models: Dict[Tuple[int, float], DBSCAN],
):
    """
    > IF (the current block is not a header block)
        AND
    ( font characteristics of the current block and the previous block are the same
        OR
    the font characteristics of the current block's first span and the previous
    block's last span are the same)
        AND
    ( the current block's first line y end is greater than the previous block's last line y end)
        AND
    (the current block's page number is the same as the previous block's page number)
        AND
    (y_distance_between_blocks follows distance between line distribution in
        similiar paragraphs)
        THEN
    merge the current block with the previous block
    """
    modified_content: List[Block] = []

    for block in content:
        if (modified_content and block.header.text) or not modified_content:
            modified_content.append(deepcopy(block))
        else:
            previous_block = modified_content[-1]
            current_span = block.spans[0]
            previous_span = previous_block.spans[-1]

            current_block_fonts = block_font_characteristics(block)
            previous_block_fonts = block_font_characteristics(previous_block)

            current_block_span_fonts = font_characteristics(current_span)
            previous_block_span_fonts = font_characteristics(previous_span)
            font_size_emphasis_match = (
                current_block_fonts == previous_block_fonts
            ) or (current_block_span_fonts == previous_block_span_fonts)

            previous_block_last_line_y_end = previous_block.last_line_y_end
            current_block_first_line_y_end = block.first_line_y_end

            y_distance_between_blocks = (
                current_block_first_line_y_end - previous_block_last_line_y_end
            )

            if (
                font_size_emphasis_match
                and ((block.block_font, np.round(block.block_size, 1)) in dbscan_models)
                and (y_distance_between_blocks > 0)
                and (previous_block.page_no == block.page_no)
            ):
                model = dbscan_models[(block.block_font, np.round(block.block_size, 1))]
                prediction = dbscan_predict(model, y_distance_between_blocks)
                if prediction != -1:
                    logging.info(
                        f"""Line Spacing Matched so Merged with above block:
                                {''.join([span.text for span in block.spans])}"""
                    )
                    previous_block = merge_consecutive_blocks(previous_block, block)
                else:
                    modified_content.append(deepcopy(block))

            else:
                modified_content.append(deepcopy(block))
    return modified_content
