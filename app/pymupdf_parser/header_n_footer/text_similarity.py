"""Copyright (C) 2022 Adarsh Gupta"""

import re
from typing import Dict

import numpy as np
from rapidfuzz.distance.Levenshtein import distance as levenshtein

from app.pymupdf_parser.handle_list_indentifier.classify_numeral import numeral_dict

TEXT_SIMILARITY = "hard"  ## options: hard and soft

normalized_distance = {
    i: 1 + np.random.random() / 100
    if np.abs(i) <= 8
    else 0.50 + np.random.random() / 100
    for i in range(-20, 20)
}

normalized_distance.pop(0)


def get_normalized_distance(distance):  # sourcery skip: assign-if-exp
    if distance in normalized_distance:
        return normalized_distance[distance]
    else:
        return 0


def levenshtein_distance(text_1, text_2):
    max_length = max([len(text_1), len(text_2)])
    min_length = min([len(text_1), len(text_2)])
    diff = levenshtein(text_1, text_2)
    if diff > 3:
        return 0
    sim = min_length - diff
    return sim / max_length


def classify_text(text) -> Dict:
    text_class = {
        key: numeral_set[text]
        for key, numeral_set in numeral_dict.items()
        if text in numeral_set
    }

    if not text_class:
        text_class["text"] = text

    return text_class


def text_similarity(text_1, text_2):
    text_1_class = classify_text(text_1)
    text_2_class = classify_text(text_2)

    similarity = {0}

    for class_1, value_1 in text_1_class.items():
        for class_2, value_2 in text_2_class.items():
            if class_1 == class_2:
                if class_1 == "text":
                    text_1 = re.sub(r"[0-9]", "&", text_1)
                    text_2 = re.sub(r"[0-9]", "&", text_2)
                    if TEXT_SIMILARITY == "soft":
                        sim = levenshtein_distance(text_1, text_2)
                    else:
                        sim = 1 * (text_1.lower() == text_2.lower())

                    if sim > 0.5:
                        similarity.add(sim + np.random.random() / 100)

                else:
                    distance = value_1 - value_2
                    similarity.add(
                        get_normalized_distance(distance)
                        * (2 if value_1 in {1, 2, 3, 4, 5, 6} else 1)
                    )

    return sum(similarity)
