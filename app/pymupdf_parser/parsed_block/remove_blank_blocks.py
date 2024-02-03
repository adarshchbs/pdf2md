"""Copyright (C) 2022 Adarsh Gupta"""
import logging
from copy import deepcopy

import numpy as np
from nptyping import NDArray

from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent, _Blocks
from app.pymupdf_parser.utils.iou import intersection_over_first_bbox_area, iou_yaxis


def remove_blank_blocks(content: PyMuPdfContent):
    parsed_content = PyMuPdfContent([], content.width, content.height)

    for block in content.blocks:
        contains_text = False
        for line in block.lines:
            for span in line.spans:
                if span.text not in {"", " ", "  ", "   "}:
                    contains_text = True
                    break
            else:
                # Continue if the inner loop wasn't broken.
                continue
            # Inner loop was broken, break the outer.
            break

        if contains_text == True:
            parsed_content.blocks.append(block)
        else:
            if len(parsed_content.blocks) == 0:
                continue
            for line in block.lines:
                for span in line.spans:
                    if span.text in {" ", "  ", "   "}:
                        current_span = span
                        break
                else:
                    # Continue if the inner loop wasn't broken.
                    continue
                previous_block_last_line = parsed_content.blocks[-1].lines[-1]
                last_span = previous_block_last_line.spans[-1]
                previous_span_bbox = np.array(previous_block_last_line.spans[-1].bbox)
                current_bbox = np.array(current_span.bbox)
                iou_y = iou_yaxis(previous_span_bbox, current_bbox)
                is_current_span_right_after_previous_block = (
                    current_bbox[0]
                    >= (previous_span_bbox[0] + previous_span_bbox[2]) / 2
                )

                if is_current_span_right_after_previous_block and iou_y > 0.8:
                    candidate_span = deepcopy(current_span)
                    candidate_span.font = last_span.font
                    candidate_span.size = last_span.size
                    previous_block_last_line.spans.append(candidate_span)

    return parsed_content


def remove_blank_lines(content: PyMuPdfContent):
    parsed_content = PyMuPdfContent([], content.width, content.height)
    for block in content.blocks:
        parsed_block = deepcopy(block)
        parsed_block.lines = []
        for line in block.lines:

            contains_text = any(
                span.text not in {"", " ", "  ", "   "} for span in line.spans
            )

            if contains_text:
                parsed_block.lines.append(line)
            else:
                if not parsed_block.lines:
                    continue
                for span in line.spans:
                    if not span.text.strip():
                        current_span = span
                        break
                else:
                    # Continue if the inner loop wasn't broken.
                    continue
                previous_line = parsed_block.lines[-1]
                last_span = previous_line.spans[-1]
                previous_span_bbox = np.array(previous_line.spans[-1].bbox)
                current_bbox = np.array(current_span.bbox)
                iou_y = iou_yaxis(previous_span_bbox, current_bbox)
                is_current_span_right_after_previous_block = (
                    current_bbox[0]
                    >= (previous_span_bbox[0] + previous_span_bbox[2]) / 2
                )

                if is_current_span_right_after_previous_block and iou_y > 0.8:
                    candidate_span = deepcopy(current_span)
                    candidate_span.font = last_span.font
                    candidate_span.size = last_span.size
                    previous_line.spans.append(candidate_span)
        if parsed_block.lines:
            parsed_content.blocks.append(parsed_block)
    return parsed_content


def remove_blank_spans(content: PyMuPdfContent):
    parsed_content = PyMuPdfContent([], content.width, content.height)
    for block in content.blocks:
        parsed_block = deepcopy(block)
        parsed_block.lines = []
        for line in block.lines:
            parsed_line = deepcopy(line)
            parsed_line.spans = []
            for span in line.spans:
                if span.text.strip():
                    parsed_line.spans.append(span)
                elif len(parsed_line.spans) > 0:
                    parsed_line.spans[-1].text = parsed_line.spans[-1].text + span.text
            if parsed_line.spans:
                parsed_block.lines.append(parsed_line)
        if parsed_block.lines:
            parsed_content.blocks.append(parsed_block)
    return parsed_content


def remove_tables(
    mu_content: PyMuPdfContent,
    pred_boxes: NDArray,
):
    for block in mu_content.blocks:
        for line in block.lines:
            i = 0
            while i < len(line.spans):
                span = line.spans[i]
                for bbox in pred_boxes:
                    if (
                        intersection_over_first_bbox_area(np.array(span.bbox), bbox)
                        > 0.7
                    ):
                        line.spans.pop(i)
                        logging.info(f" Inside Table : {span.text}")
                        break
                else:  # no break
                    i += 1

    # i = 0
    # while i < len(mu_content.blocks):
    #     block = mu_content.blocks[i]
    #     for bbox in pred_boxes:
    #         if intersection_over_first_bbox_area(np.array(block.bbox), bbox) > 0.8:
    #             block_text = "".join(
    #                 [span.text for line in block.lines for span in line.spans]
    #             )
    #             logging.info(f"Inside table {block_text}")
    #             mu_content.blocks.pop(i)
    #             break
    #     else:
    #         i += 1
