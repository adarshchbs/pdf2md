"""Copyright (C) 2022 Adarsh Gupta"""
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from nptyping import NDArray
from numba import njit

from app.pymupdf_parser.parameter.parsed_content import Block


@dataclass
class NumberedIdWithClasses:
    numbered_id: str
    block: Block
    numeral_sign: List[str]
    class_separators: str
    numeral_value: NDArray


@njit
def normalised_distance_value(n: int) -> float:
    if n == -1:
        return 0.5
    elif n == -2:
        return 0.251
    elif n == -3:
        return 0.253
    elif n == 1:
        return 0.501
    elif n == 2:
        return 0.252
    elif n == 3:
        return 0.254
    elif n == 4:
        return 0.255
    else:
        return 0.0


def numeral_distance_between_array(x, y):
    d = x - y

    if d[-1]:
        """last element should only change"""
        return normalised_distance_value(d[-1])
    else:
        return 0.0


def is_numeral_legit(
    numeral_with_classes_dict: Dict[
        Tuple[int, float, int, Tuple], List[NumberedIdWithClasses]
    ]
):  # sourcery skip: set-comprehension
    for key in deepcopy(list(numeral_with_classes_dict.keys())):
        numbered_id_with_classes_list = numeral_with_classes_dict[key]
        if len(numbered_id_with_classes_list) < 2:
            numeral_with_classes_dict.pop(key)

        else:
            i = 0
            while i < len(numbered_id_with_classes_list):
                n1 = numbered_id_with_classes_list[i]
                similarity_with_other_id = set()
                for n2 in numbered_id_with_classes_list:
                    similarity_with_other_id.add(
                        numeral_distance_between_array(
                            n1.numeral_value, n2.numeral_value
                        )
                        * (n1.class_separators.strip() == n2.class_separators.strip())
                        * (1 + (n1.numeral_value[-1] in {1, 2, 3}))
                    )
                # print(n1.numbered_id, n1.numeral_value[-1])
                similarity_with_other_id = sum(similarity_with_other_id)
                if similarity_with_other_id < 1:
                    logging.info(
                        f"Numeral {n1.numbered_id} has similarity = {similarity_with_other_id}"
                    )
                    numbered_id_with_classes_list.pop(i)
                elif key[3] == ("lower_case") and (not n1.class_separators):
                    logging.info(
                        f"single lower case character - '{n1.numeral_sign[0]}', so removed"
                    )
                    numbered_id_with_classes_list.pop(i)
                else:
                    logging.info(
                        f"Numeral {n1.numbered_id} has similarity = {similarity_with_other_id}"
                    )
                    i += 1
