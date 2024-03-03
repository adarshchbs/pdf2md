"""Copyright (C) 2022 Adarsh Gupta"""
import re
from typing import Dict, List, Tuple

import community as community
from rapidfuzz.distance.Levenshtein import normalized_similarity
import networkx as nx

from app.pymupdf_parser.parameter.pymupdf_content import PyMuPdfContent


def fonts_in_document(unparsed_contents: List[PyMuPdfContent]) -> List[str]:
    """
    It takes a list of PyMuPdfContent objects, and returns a list of all the fonts used in the document
    """
    font_set = set()

    for content in unparsed_contents:
        for key in content.blocks:
            for line in key.lines:
                for text in line.spans:
                    font_set.add(text.font)

    return list(font_set)


def font_description(fonts: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    It takes a list of fonts and returns a dictionary that maps each font to a tuple of two numbers. The
    first number is the font's family (i.e. which group it belongs to) and the second number is the
    font's "emphasis" (i.e. if its bold or italic or bold-italic)
    :return: A dictionary with the font name as the key and a tuple of the font family and the font
    emphasis as the value.
    """
    print(f"{fonts=}")
    best_partition = partition_fonts_in_family(fonts)

    font_description: Dict[str, Tuple[int, int]] = {}

    bold_keyword = {"bold"}
    italic_keyword = {"italic", "ital", "oblique"}

    for font in best_partition:
        f = font.lower()
        f = "".join(filter(str.isalnum, f))
        bold_emphasis = sum(keyword in f for keyword in bold_keyword)
        italic_emphasis = sum(2 for keyword in italic_keyword if keyword in f)
        italic_emphasis = min([italic_emphasis, 2])
        bold_emphasis = min([bold_emphasis, 1])
        font_emphasis = italic_emphasis + bold_emphasis

        font_description[font] = (best_partition[font], font_emphasis)

    return font_description


def partition_fonts_in_family(fonts) -> Dict[str, int]:
    """
    It takes a list of fonts and returns a dictionary where the keys are the name of the font and the
    values font family as int
    """
    G = nx.Graph()
    to_remove = ["medium", "regular", "bold", "italic", "ital", "roman"]
    p = re.compile("|".join(map(re.escape, to_remove)))

    for i, f1 in enumerate(fonts):
        for f2 in fonts[i + 1 :]:
            font_1 = f1.lower()
            font_2 = f2.lower()
            font_1 = "".join(filter(str.isalnum, font_1))
            font_2 = "".join(filter(str.isalnum, font_2))
            font_1 = p.sub("", font_1)
            font_2 = p.sub("", font_2)
            differences = normalized_similarity(font_1, font_2)
            if differences < 0.6:
                differences = 0
            G.add_edge(f1, f2, weight=differences)

    best_partition = community.community_louvain.best_partition(G, resolution=1)
    return best_partition


if __name__ == "__main__":
    fonts = ["Serif", "Sans-Serif", "Monospace"]
    print(font_description(fonts))
