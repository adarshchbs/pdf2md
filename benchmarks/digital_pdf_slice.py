from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pymupdf


@dataclass(frozen=True)
class DigitalSliceConfig:
    min_native_characters_per_page: int = 20
    scanned_image_coverage: float = 0.9
    scanned_page_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.min_native_characters_per_page < 1:
            raise ValueError("minimum native characters must be positive")
        if not 0 < self.scanned_image_coverage <= 1:
            raise ValueError("scanned image coverage must be in (0, 1]")
        if not 0 < self.scanned_page_ratio <= 1:
            raise ValueError("scanned page ratio must be in (0, 1]")


@dataclass(frozen=True)
class PageEvidence:
    native_characters: int
    image_coverage: float
    has_glyphless_font: bool

    def __post_init__(self) -> None:
        if self.native_characters < 0:
            raise ValueError("native character count cannot be negative")
        if not 0 <= self.image_coverage <= 1:
            raise ValueError("image coverage must be in [0, 1]")


@dataclass(frozen=True)
class DigitalPdfClassification:
    eligible: bool
    reason: str
    page_count: int
    scanned_page_count: int
    pages: tuple[PageEvidence, ...]


def classify_pdf(
    pdf_path: Path,
    config: DigitalSliceConfig | None = None,
) -> DigitalPdfClassification:
    """Classify a PDF using native document evidence, never benchmark references."""
    config = config or DigitalSliceConfig()
    if pdf_path.suffix.lower() != ".pdf":
        return DigitalPdfClassification(
            eligible=False,
            reason="non_pdf_input",
            page_count=0,
            scanned_page_count=0,
            pages=(),
        )
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            return DigitalPdfClassification(
                eligible=False,
                reason="encrypted_pdf",
                page_count=document.page_count,
                scanned_page_count=0,
                pages=(),
            )
        evidence = tuple(_page_evidence(page) for page in document)
    return classify_document_evidence(evidence, config)


def classify_document_evidence(
    pages: list[PageEvidence] | tuple[PageEvidence, ...],
    config: DigitalSliceConfig,
) -> DigitalPdfClassification:
    if not pages:
        raise ValueError("PDF must contain at least one page")
    immutable_pages = tuple(pages)
    scanned_page_count = sum(
        page.native_characters < config.min_native_characters_per_page
        and page.image_coverage >= config.scanned_image_coverage
        for page in immutable_pages
    )
    if any(page.has_glyphless_font for page in immutable_pages):
        return DigitalPdfClassification(
            eligible=False,
            reason="ocr_text_layer",
            page_count=len(immutable_pages),
            scanned_page_count=scanned_page_count,
            pages=immutable_pages,
        )
    if scanned_page_count / len(immutable_pages) >= config.scanned_page_ratio:
        return DigitalPdfClassification(
            eligible=False,
            reason="scanned_raster_majority",
            page_count=len(immutable_pages),
            scanned_page_count=scanned_page_count,
            pages=immutable_pages,
        )
    return DigitalPdfClassification(
        eligible=True,
        reason="born_digital",
        page_count=len(immutable_pages),
        scanned_page_count=scanned_page_count,
        pages=immutable_pages,
    )


def parsebench_test_id(category: str, relative_path: Path) -> str:
    inference_group = {
        "text_content": "text",
        "text_formatting": "text",
    }.get(category, category)
    return f"{inference_group}/{relative_path.stem}"


def _page_evidence(page: pymupdf.Page) -> PageEvidence:
    page_area = float(page.rect.width * page.rect.height)
    covered_area = 0.0
    if page_area > 0:
        for image in page.get_image_info():
            bbox = pymupdf.Rect(image["bbox"]) & page.rect
            if not bbox.is_empty:
                covered_area += float(bbox.width * bbox.height)
    font_fields = (
        value.casefold()
        for font in page.get_fonts()
        for value in font
        if isinstance(value, str)
    )
    return PageEvidence(
        native_characters=len(cast(str, page.get_text("text")).strip()),
        image_coverage=min(1.0, covered_area / page_area) if page_area > 0 else 0.0,
        has_glyphless_font=any("glyphlessfont" in value for value in font_fields),
    )
