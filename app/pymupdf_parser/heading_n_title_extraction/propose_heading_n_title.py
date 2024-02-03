from pprint import pprint
from typing import Any, Dict, List
from app.pymupdf_parser.heading_n_title_extraction.candidates_proposal_func import (
    does_next_block_contain_larger_no_character,
    is_no_of_char_in_block_normal_for_heading,
    is_block_center_aligned,
    is_font_bigger_than_normal,
    is_bold,
    score_based_on_no_of_line_in_next_block,
    score_for_numbered_block,
    no_of_word_in_sweet_spot,
    is_font_different_than_normal,
)
from app.pymupdf_parser.parameter.parsed_content import Block
from app.pymupdf_parser.alignment.horizontal_margin import (
    check_center_align,
)
from app.pymupdf_parser.utils.pdf_statitics import FontSizeStat, FontTypeStat
import numpy as np
from termcolor import colored

scoring_functions = [
    is_bold,
    is_no_of_char_in_block_normal_for_heading,
    does_next_block_contain_larger_no_character,
    score_based_on_no_of_line_in_next_block,
    score_for_numbered_block,
    no_of_word_in_sweet_spot,
]


def propose_heading_n_title(
    parsed_doc: List[Block], horizontal_margin: Dict[int, Any], pymu_doc
):  # sourcery skip: merge-dict-assign, merge-list-append
    font_size_stat = FontSizeStat(parsed_doc)
    font_type_stat = FontTypeStat(parsed_doc)
    for block_no, block in enumerate(parsed_doc):
        scores = {}

        scores[is_font_bigger_than_normal.__name__] = is_font_bigger_than_normal(
            block_no, parsed_doc, font_size_stat
        )
        scores[is_block_center_aligned.__name__] = is_block_center_aligned(
            block_no, parsed_doc, horizontal_margin
        )
        scores[is_font_different_than_normal.__name__] = is_font_different_than_normal(
            block_no, parsed_doc, font_type_stat
        )

        for func in scoring_functions:
            scores[func.__name__] = func(block_no, parsed_doc)

        scores_list = np.array(list(scores.values()))

        total_score = np.sum(scores_list)
        number_of_positive_signal = np.sum(1 * (scores_list > 0))

        if total_score >= 0 and number_of_positive_signal >= 2:
            print(colored(f"{total_score=}", "blue"))
            print(f"{block.block_text=}")
            pprint(scores)
            print("-------------------------")

        if scores[is_bold.__name__] > 0 and (
            total_score < 0 or number_of_positive_signal < 2
        ):
            print(colored(f"{total_score=}", "red"))
            print(colored(f"{block.block_text=}", "red"))
            pprint(scores)
            print("-------------------------")

        if total_score >= 0 and number_of_positive_signal >= 2:
            pymu_doc[block.page_no].add_highlight_annot(block.block_bbox)

    pymu_doc.save("/Users/adarshgupta/Projects/pdf_parser/pdf/heading_n_title.pdf")
