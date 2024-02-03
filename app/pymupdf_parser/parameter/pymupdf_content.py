"""Copyright (C) 2022 Adarsh Gupta"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from app.pymupdf_parser.parameter.parsed_content import Header


@dataclass
class _Spans:
    bbox: Tuple[float, float, float, float]
    origin: Tuple[float, float]
    ascender: float
    color: int
    descender: float
    flags: int # superscript/subscript basically a proxy for emphasis - not reliable
    font: str # font of the span text
    size: float # font size
    text: str # text belonging to the span
    style: int = 0


@dataclass
class _Lines:
    bbox: Tuple[float, float, float, float]
    dir: Tuple
    spans: List[_Spans]
    wmode: int # which direction the text is written in 


@dataclass
class _Blocks: # a paragraph level object
    bbox: Tuple[float, float, float, float]
    lines: List[_Lines]
    number: int
    type: int
    header: Header = Header("", np.array([]), ())


@dataclass
class PyMuPdfContent:
    blocks: List[_Blocks]
    width: float
    height: float
