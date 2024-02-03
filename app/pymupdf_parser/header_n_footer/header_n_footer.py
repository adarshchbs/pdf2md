"""Copyright (C) 2022 Adarsh Gupta"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from app.pymupdf_parser.header_n_footer.text_similarity import text_similarity
from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent, _Blocks, _Lines
from app.pymupdf_parser.utils.iou import iou_xaxis, iou_yaxis
import re


def header_footer_preprocess(text: str):
    text = re.sub(r"(\^|\_)\{(.+?)\}", r"\2", text)
    text = " ".join(text.split())
    return text


class HeaderNFooter:
    def __init__(self, mu_document: List[PyMuPdfContent]):

        self.mu_document = mu_document
        self.mu_document_lines: Dict[int, Dict[Tuple[int, int], _Lines]] = {}
        for page_no, mu_content in enumerate(mu_document):
            mu_content_lines = {}
            for block_no, block in enumerate(mu_content.blocks):
                for line_no, line in enumerate(block.lines):
                    mu_content_lines[(block_no, line_no)] = line
            self.mu_document_lines[page_no] = mu_content_lines

    def propose(self):
        self.header_proposal: Dict[int, List[Tuple[int, int]]] = {}
        self.footer_proposal: Dict[int, List[Tuple[int, int]]] = {}
        for page_no, mu_content_lines in self.mu_document_lines.items():
            ## sort the line keys by y_cordinates of the origin of first span of the blocks.
            sorted_content = sorted(
                mu_content_lines.keys(),
                key=lambda x: mu_content_lines[x].spans[0].origin[1],
            )
            self.header_proposal[page_no] = sorted_content[:4]
            self.footer_proposal[page_no] = sorted_content[-4:][::-1]

    def critise(self, header_or_footer_list: Dict[int, List[Tuple[int, int]]]):
        score: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(dict)

        for page_no, line_keys in header_or_footer_list.items():
            min_page_no = max([0, page_no - 8])
            max_page_no = min([len(header_or_footer_list), page_no + 8])
            neighboring_pages_lines_keys = {
                i: header_or_footer_list[i]
                for i in range(min_page_no, max_page_no)
                if i != page_no
            }

            for line_no, first_key in enumerate(line_keys):
                similarity = 0
                line_1 = self.mu_document_lines[page_no][first_key]
                text_1 = "".join([span.text for span in line_1.spans]).strip()
                text_1 = header_footer_preprocess(text_1)

                for (
                    neighbour_page_no,
                    neighbour_page_key,
                ) in neighboring_pages_lines_keys.items():

                    for second_key in neighbour_page_key:
                        line_2 = self.mu_document_lines[neighbour_page_no][second_key]
                        text_2 = "".join(span.text for span in line_2.spans).strip()
                        text_2 = header_footer_preprocess(text_2)
                        sim = text_similarity(text_1, text_2)
                        if sim > 0.5:
                            bbox_overlap = iou_xaxis(
                                line_1.bbox, line_2.bbox
                            ) * iou_yaxis(line_1.bbox, line_2.bbox)
                            bbox_overlap = 1 if bbox_overlap > 0.5 else bbox_overlap
                            similarity += sim * bbox_overlap

                score[page_no][first_key] = similarity * (
                    1 if line_no in {0, 1} else 1 / 2
                )

        return_dict = defaultdict(list)

        for page_no, line_keys in score.items():
            for key, sim in line_keys.items():
                if sim >= 1:
                    return_dict[page_no].append(key)

        return return_dict

    def apply(
        self,
    ) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
        self.propose()
        footer = self.critise(self.footer_proposal)
        header = self.critise(self.header_proposal)

        return header, footer
