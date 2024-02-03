"""Copyright (C) 2022 Adarsh Gupta"""

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from app.pymupdf_parser.handle_list_indentifier.classify_numeral import classify_numeral
from app.pymupdf_parser.handle_list_indentifier.numeral_distance import is_numeral_legit
from app.pymupdf_parser.parameter.parsed_content import Block, Header
from app.pymupdf_parser.utils.font_characteristics import font_characteristics
from sklearn.cluster import DBSCAN
import logging


@dataclass
class NumberedIdWithClasses:
    numbered_id: str
    block: Block
    numeral_sign: List[str]
    class_separators: str
    numeral_value: np.ndarray


character = (
    r"(i|ii|iii|iv|v|vi|vii|viii|ix|I|II|III|IV|V|VI|VII|VIII|IX|[0-9]{1,3}|[a-z,A-Z])"
)
middle_character = f"({character}" + r"\s?\.\s?)?"
start_token = r"\A\(?\[?"
end_token = r"\s?(\.|\:|\)|\]|\s)"
negative_lookahead_with_space = r"(?!\s{0,3}\%|\s{0,3}\(|\s{1,3}[0-9]|\s{1,3}[a-z])"
"""
negative lookahead: Find expression A where expression B does not follow: A(?!B)
"""

vanilla_pattern = (
    start_token
    + middle_character * 3
    + character
    + end_token
    + negative_lookahead_with_space
)
vanilla_pattern = re.compile(vanilla_pattern)

bracket_start_token = r"\A(\(|\[)"
bracket_end_token = r"(\)|\]|\s)"
bracket_negative_lookahead_with_space = r"(?!\s{0,3}\%|\s{0,3}\()"
bracket_pattern = (
    bracket_start_token
    + middle_character * 3
    + character
    + bracket_end_token
    + bracket_negative_lookahead_with_space
)

word_re = r"\A[a-z,A-Z]{2,}\s{1,}"
word_end_token = r"(\.|\:|\s\:|\)|\]|\s|$)"

word_pattern = (
    word_re
    + middle_character * 3
    + character
    + word_end_token
    # + bracket_negative_lookahead_with_space
)

pattern_list = [vanilla_pattern, bracket_pattern, word_pattern]


class NumberedListIdentifier:
    def __init__(self, parsed_content: List[Block]):
        self.parsed_content = parsed_content
        self.pattern_list = pattern_list

    def propose(self):
        self.proposal: List[Tuple[str, Block]] = []
        self.proposal_font: List[Tuple[int, float, int]] = []
        for block in self.parsed_content:
            searchable_text = "".join([span.text for span in block.spans[:3]])
            for pattern in self.pattern_list:
                if search_item := re.search(pattern, searchable_text.strip()):
                    numbered_id = search_item.group().strip()
                    # logging.info(f"{numbered_id=} ---- {searchable_text=}")
                    self.proposal.append((numbered_id, block))
                    first_span = block.spans[0]
                    self.proposal_font.append(font_characteristics(first_span))
                    break

    def critise(self):
        self.proposal_dict_with_classes: Dict[
            Tuple[int, float, int, Tuple], List[NumberedIdWithClasses]
        ] = defaultdict(list)

        """
        key: (font, font_size, font_emphasis,(numeral_classes))
        value: List[NumberedIdWithClasses]
        """

        self.proposal_dict: Dict[
            Tuple[int, float, int], List[Tuple[str, Block]]
        ] = defaultdict(list)

        if not self.proposal_font:
            return
        font_size_list = np.array(
            [font_char[1] for font_char in self.proposal_font]
        ).reshape(-1, 1)
        dbscan_models = DBSCAN(eps=0.15, min_samples=1)
        font_size_class_list = dbscan_models.fit_predict(font_size_list)

        self.proposal_font = [
            (font_char[0], size, font_char[2])
            for font_char, size in zip(self.proposal_font, font_size_class_list)
        ]

        for (font_char, (numbered_id, block)) in zip(self.proposal_font, self.proposal):
            self.proposal_dict[font_char].append((numbered_id, block))

        for (font_char, l) in self.proposal_dict.items():
            for (numbered_id, block) in l:
                numeral_sign = [i for i in re.split(r"[^0-9a-zA-Z]", numbered_id) if i]
                class_separators = "".join(
                    [i for i in re.split(r"[0-9a-zA-Z]", numbered_id) if i]
                )
                classes = [tuple(classify_numeral(sign)[0]) for sign in numeral_sign]
                numeral_value_list = [
                    tuple(classify_numeral(sign)[1]) for sign in numeral_sign
                ]
                for class_tuple, value_tuple in zip(
                    itertools.product(*classes), itertools.product(*numeral_value_list)
                ):
                    numbered_id_with_classes = NumberedIdWithClasses(
                        numbered_id=numbered_id,
                        block=block,
                        numeral_sign=numeral_sign,
                        class_separators=class_separators,
                        numeral_value=np.array(value_tuple),
                    )
                    self.proposal_dict_with_classes[font_char + class_tuple].append(
                        numbered_id_with_classes
                    )

        is_numeral_legit(self.proposal_dict_with_classes)
        # for key, value in self.proposal_dict_with_classes.items():
        #     for a in value:
        #         print(a.numbered_id)

    def apply(self):
        self.propose()
        self.critise()

        for key, dataclasses_list in self.proposal_dict_with_classes.items():
            for dc in dataclasses_list:
                dc.block.header.text = dc.numbered_id
                dc.block.header.tag = key


if __name__ == "__main__":
    examples = [
        "(i) Abc",
        "(a) xye",
        "(1) uwerasd",
        "(a) the description",
        "1. Something",
        "2 . Something",
        "3 . 3 . Something",
        "4.5.9. Different",
        "4.6 around",
    ]
    for text in examples:
        for pattern in pattern_list:
            search_item = re.search(pattern, text)
            print(f" examples 1:  {text=} , {search_item=}")
            if search_item:
                break

    word_examples = [
        "Chapter 1 This is a good",
        "Figure 1.6: dt:birdtest: bird test images",
        "Algorithm 2 DecisionTreeTest(tree, test point)",
        "the other three features from Figure 1.3",
        "A1 other features",
        "Subsection 1.2.3 ",
        "A 1 teaching.com",
        "Chart 3",
        "Table 1. The Static Effects of Brexit on U.K. Living Standards",
    ]
    for text in word_examples:
        search_item = re.search(word_pattern, text)
        print(f" word example:  {text=} , {search_item=} matched using word pattern")
