from __future__ import annotations

import math

import pymupdf

from app.pdf2md.pymupdf_runtime import pymupdf_session


def unrotated_page_extent(page: pymupdf.Page) -> tuple[float, float]:
    """Return crop-relative extent under the guarded modern-pipeline runtime."""
    with pymupdf_session():
        coordinates = tuple(float(value) for value in page.cropbox)
        if len(coordinates) != 4:
            raise ValueError(f"PyMuPDF crop box must have four coordinates: {coordinates}")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"PyMuPDF crop box must contain finite coordinates: {coordinates}")
        width = coordinates[2] - coordinates[0]
        height = coordinates[3] - coordinates[1]
        if width <= 0 or height <= 0:
            raise ValueError(f"PyMuPDF crop box must have positive dimensions: {coordinates}")
        return width, height
