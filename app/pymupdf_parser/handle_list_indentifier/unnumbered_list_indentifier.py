"""Copyright (C) 2022 Adarsh Gupta"""
import logging
import re
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from nptyping import NDArray

# from pyrsistent import T
from app.pymupdf_parser.parameter.parsed_content import Header
from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent, _Blocks
from app.pymupdf_parser.utils.check_alignment import check_x_alignment
from app.pymupdf_parser.utils.combine_bboxes import (
    combine_line_bboxes,
    combine_span_bboxes,
)

# TODO put this into config
pattern = (
    r"\A(●|•|◘|○|◙|⦿|‣|⁃|⁌|⁍|➡|◆|◇|◌|◈|➢|➣|➤|➥|➦|➧|➱|➮|▷|▶|▸|▹|▻|⊛|=>|->|>|\*|o|\-)\s"
)
pattern = re.compile(pattern)


def bbox_after_header_separation(block: _Blocks):
    if len(block.lines) == 1:
        block_bbox_list = [
            np.array(span.bbox) for span in block.lines[0].spans if span.text.strip()
        ]
        return combine_span_bboxes(block_bbox_list)
    else:
        first_line_bbox_list = [
            np.array(span.bbox) for span in block.lines[0].spans if span.text.strip()
        ]
        # print(first_line_bbox_list)
        first_line_bbox = combine_span_bboxes(first_line_bbox_list)
        lines_bbox_list = [first_line_bbox] + [
            np.array(line.bbox) for line in block.lines[1:]
        ]
        return combine_line_bboxes(lines_bbox_list)


class SeperateListIdentifier:
    def __init__(
        self,
        document: List[PyMuPdfContent],
        font_description: Dict[str, Tuple[int, int]],
        margin: np.ndarray,
    ):
        self.document = document
        self.font_description = font_description
        self.margin = margin

    def propose(self):
        self.proposal: Dict[str, List[_Blocks]] = defaultdict(list)
        self.proposal_bbox: Dict[str, List[NDArray]] = defaultdict(list)
        self.proposal_font: Dict[str, List[Tuple]] = defaultdict(list)

        for content in self.document:
            for block in content.blocks:
                first_line = block.lines[0]
                first_span = first_line.spans[:2]

                # print(first_span.text)
                if first_span is None:
                    continue

                if result := re.search(
                    pattern, " ".join(span.text for span in first_span)
                ):
                    match = result.group().strip()
                    self.proposal[match].append(block)
                    self.proposal_bbox[match].append(np.array(first_span[0].bbox))
                    self.proposal_font[match].append(
                        (first_span[0].font, np.round(first_span[0].size, 1))
                    )

    def critise(self):
        for key, block_list in self.proposal.items():
            bbox_list = np.array(self.proposal_bbox[key])
            font_list = self.proposal_font[key]

            alignment = check_x_alignment(bbox_list, font_list, self.margin)

            self.proposal[key] = [b for i, b in enumerate(block_list) if alignment[i]]
            logging.info(
                f"Misaligned candidate list identifers = {[b for i, b in enumerate(block_list) if not alignment[i]]}"
            )
            self.proposal_bbox[key] = np.array(self.proposal_bbox[key])[alignment]
            self.proposal_font[key] = [
                f for i, f in enumerate(self.proposal_font[key]) if alignment[i]
            ]

    def apply(self):
        self.propose()
        self.critise()

        for key, block_list in self.proposal.items():
            if len(block_list) < 2:
                continue

            bbox_list = self.proposal_bbox[key]
            font_list = self.proposal_font[key]
            for i, block in enumerate(block_list):
                block.header = Header(
                    text=key,
                    bbox=bbox_list[i],
                    tag=self.font_description[font_list[i][0]] + (font_list[i][1],),
                )
                if (
                    block.lines[0].spans[0].text.strip() == key
                    and len(block.lines[0].spans) > 1
                ):
                    block.lines[0].spans.pop(0)
                    block.bbox = tuple(bbox_after_header_separation(block))

                else:
                    block.lines[0].spans[0].text = re.sub(
                        pattern,
                        "",
                        block.lines[0].spans[0].text,
                        0,
                    ).strip()
