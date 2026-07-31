"""Copyright (C) 2022 Adarsh Gupta"""

from dataclasses import dataclass, field
from typing import OrderedDict, Tuple

import numpy as np

from app.pymupdf_parser.parameter.parsed_content import Header
from app.pymupdf_parser.parameter.pymupdf_content import _Spans


@dataclass
class _LinesDict:
    bbox: Tuple[float, float, float, float]
    dir: Tuple
    spans: OrderedDict[int, _Spans]
    wmode: int


@dataclass
class _BlocksDict:
    bbox: Tuple[float, float, float, float]
    lines: OrderedDict[int, _LinesDict]
    number: int
    type: int
    header: Header = field(default_factory=lambda: Header("", np.array([]), ()))


@dataclass
class PyMuPdfContentDict:
    blocks: OrderedDict[int, _BlocksDict]
    width: float
    height: float
