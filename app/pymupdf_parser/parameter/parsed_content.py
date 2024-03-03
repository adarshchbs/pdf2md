"""Copyright (C) 2022 Adarsh Gupta"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
from nptyping import Int, NDArray, Shape


@dataclass
class Span:
    text: str
    font: int
    size: float
    emphasis: int
    number_of_char: int
    start_span_origin: np.ndarray
    start_span_bbox: np.ndarray
    end_span_origin: np.ndarray
    end_span_bbox: np.ndarray
    number_of_span: int
    end_span_line_number: int
    flags: int
    style: int = 0
    span_center_align: bool = False


@dataclass
class Header:
    text: str
    bbox: NDArray[Shape["4"], Int]
    tag: Tuple


@dataclass
class Block:
    block_bbox: NDArray[Shape["4"], Int]
    spans: List[Span]
    block_text: str
    block_font: int
    block_size: float
    block_emphasis: int
    number_of_line: int
    first_line_y_end: float
    last_line_y_end: float
    number_of_characters_per_font: Dict[Tuple, int]
    number_of_characters_per_emphasis: Dict[int, int]
    page_no: Union[int, float]
    page_width: float = np.nan
    page_height: float = np.nan
    header: Header = Header("", np.array([]), ())
    block_center_align: bool = False
