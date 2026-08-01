from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pymupdf
import pytest

from app.pdf2md.engine import extract_document_elements
from app.pdf2md.figures import (
    order_linked_figure_content,
    raster_candidates_from_pymupdf_dict,
    vector_candidates_from_drawings,
)
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FigureStructure,
    PageFragment,
    ParagraphStructure,
)
from app.pdf2md.semantic_text import normalize_bbox


def _pixmap(width: int, height: int, color: int) -> pymupdf.Pixmap:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pixmap.clear_with(color)
    return pixmap


def _insert_image(page: pymupdf.Page, rect: tuple[float, float, float, float], color: int) -> None:
    page.insert_image(pymupdf.Rect(rect), pixmap=_pixmap(120, 80, color))


def _save(document: pymupdf.Document, path: Path) -> None:
    document.save(path)
    document.close()


def _figure_order_element(
    element_id: str,
    element_type: str,
    bbox: tuple[float, float, float, float],
    *,
    page_number: int = 1,
    linked_ids: list[str] | None = None,
    role: str | None = None,
    caption_id: str | None = None,
) -> DocumentElement:
    structure = ElementStructure(
        linked_element_ids=[] if linked_ids is None else linked_ids,
        paragraph=(
            None
            if role is None
            else ParagraphStructure(role=role, heading_level=3 if element_type == "heading" else None)
        ),
        figure=(
            FigureStructure(
                asset_path="pdf://page/1/vector/test",
                sha256="0" * 64,
                width=100,
                height=100,
                caption_element_id=caption_id,
            )
            if element_type == "figure"
            else None
        ),
    )
    return DocumentElement(
        document_id="figure-order-test",
        element_id=element_id,
        order=0,
        element_type=element_type,
        content="" if element_type == "figure" else element_id,
        format="text",
        fragments=[
            PageFragment(
                page_number=page_number,
                page_width=200,
                page_height=200,
                bbox=BoundingBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
            )
        ],
        structure=structure,
        annotation=AnnotationMetadata(
            stage="candidate",
            revision=1,
            annotator="test",
            confidence=1.0,
            adjudication_status="unreviewed",
        ),
    )


def _rotate_square_bbox(
    bbox: tuple[float, float, float, float], rotation: int
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    if rotation == 0:
        return bbox
    if rotation == 90:
        return 200 - y1, x0, 200 - y0, x1
    if rotation == 180:
        return 200 - x1, 200 - y1, 200 - x0, 200 - y0
    if rotation == 270:
        return y0, 200 - x1, y1, 200 - x0
    raise ValueError(f"unsupported rotation: {rotation}")


def _closed_figure_component(rotation: int = 0) -> list[DocumentElement]:
    figure = _figure_order_element(
        "figure",
        "figure",
        _rotate_square_bbox((20, 20, 180, 180), rotation),
        linked_ids=["caption"],
        caption_id="caption",
    )
    caption = _figure_order_element(
        "caption",
        "caption",
        _rotate_square_bbox((30, 50, 170, 70), rotation),
        linked_ids=["figure"],
        role="caption",
    )
    panel = _figure_order_element(
        "panel",
        "heading",
        _rotate_square_bbox((30, 80, 90, 100), rotation),
        linked_ids=["figure"],
        role="figure_panel_heading",
    )
    label = _figure_order_element(
        "label",
        "paragraph",
        _rotate_square_bbox((30, 110, 150, 130), rotation),
        linked_ids=["figure"],
        role="figure_text",
    )
    return [figure, caption, panel, label]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_linked_internal_caption_reorders_only_closed_rotated_component(rotation: int) -> None:
    elements = _closed_figure_component(rotation)

    first = order_linked_figure_content(elements)
    repeated = order_linked_figure_content(elements)

    assert [element.element_id for element in first] == ["figure", "panel", "label", "caption"]
    assert [element.model_dump(mode="json") for element in first] == [
        element.model_dump(mode="json") for element in repeated
    ]


def test_linked_caption_ordering_handles_multiple_disjoint_figures_deterministically() -> None:
    first = _closed_figure_component()
    second = [
        element.model_copy(
            update={
                "element_id": f"{element.element_id}-2",
                "structure": element.structure.model_copy(
                    update={
                        "linked_element_ids": [f"{link}-2" for link in element.structure.linked_element_ids],
                        "figure": (
                            element.structure.figure.model_copy(update={"caption_element_id": "caption-2"})
                            if element.structure.figure is not None
                            else None
                        ),
                    }
                ),
            }
        )
        for element in _closed_figure_component()
    ]

    ordered = order_linked_figure_content([*first, *second])

    assert [element.element_id for element in ordered] == [
        "figure",
        "panel",
        "label",
        "caption",
        "figure-2",
        "panel-2",
        "label-2",
        "caption-2",
    ]


def test_linked_caption_ordering_fails_closed_for_reviewed_link_negatives() -> None:
    base = _closed_figure_component()
    figure, caption, panel, label = base
    other_figure = _figure_order_element(
        "other-figure",
        "figure",
        (20, 20, 180, 180),
        linked_ids=["caption"],
        caption_id="caption",
    )
    missing_caption_figure = figure.model_copy(
        update={
            "structure": figure.structure.model_copy(
                update={
                    "figure": figure.structure.figure.model_copy(
                        update={"caption_element_id": "missing-caption"}
                    )
                    if figure.structure.figure is not None
                    else None
                }
            )
        }
    )
    stale_child_claimant = _figure_order_element(
        "stale-child-claimant",
        "figure",
        (20, 20, 180, 180),
        linked_ids=["other-caption", "panel"],
        caption_id="other-caption",
    )
    other_caption = _figure_order_element(
        "other-caption",
        "caption",
        (30, 140, 170, 160),
        linked_ids=["stale-child-claimant"],
        role="caption",
    )
    cases = [
        [missing_caption_figure, caption, panel, label],
        [
            figure.model_copy(
                update={
                    "structure": figure.structure.model_copy(
                        update={"linked_element_ids": ["caption", "label", "label"]}
                    )
                }
            ),
            caption,
            panel,
            label,
        ],
        [
            figure.model_copy(
                update={"structure": figure.structure.model_copy(update={"linked_element_ids": []})}
            ),
            caption,
            panel,
            label,
        ],
        [
            figure,
            caption.model_copy(
                update={
                    "structure": caption.structure.model_copy(
                        update={"linked_element_ids": ["missing-figure"]}
                    )
                }
            ),
            panel,
            label,
        ],
        [figure, caption, panel, label, other_figure],
        [figure, caption, panel, label, stale_child_claimant, other_caption],
        [
            figure,
            caption,
            panel.model_copy(
                update={
                    "structure": panel.structure.model_copy(
                        update={"linked_element_ids": ["figure", "other-figure"]}
                    )
                }
            ),
            label,
        ],
    ]

    for elements in cases:
        assert order_linked_figure_content(elements) == elements


def test_linked_caption_ordering_fails_closed_for_page_geometry_and_semantic_negatives() -> None:
    figure, caption, panel, label = _closed_figure_component()
    cross_page = label.model_copy(
        update={
            "fragments": [label.fragments[0].model_copy(update={"page_number": 2})],
        }
    )
    outside = label.model_copy(
        update={
            "fragments": [
                label.fragments[0].model_copy(update={"bbox": BoundingBox(x0=181, y0=110, x1=195, y1=130)})
            ]
        }
    )
    mostly_contained = label.model_copy(
        update={
            "fragments": [
                label.fragments[0].model_copy(update={"bbox": BoundingBox(x0=30, y0=110, x1=181, y1=130)})
            ]
        }
    )
    body = label.model_copy(
        update={
            "structure": label.structure.model_copy(update={"paragraph": ParagraphStructure(role="body")})
        }
    )
    wrong_heading_representation = panel.model_copy(update={"element_type": "paragraph"})
    degenerate = label.model_copy(
        update={
            "fragments": [
                label.fragments[0].model_copy(
                    update={"bbox": BoundingBox.model_construct(x0=30, y0=110, x1=30, y1=130)}
                )
            ]
        }
    )

    for child in (
        cross_page,
        outside,
        mostly_contained,
        body,
        wrong_heading_representation,
        degenerate,
    ):
        elements = [figure, caption, panel, child]
        assert order_linked_figure_content(elements) == elements

    cross_page_caption = caption.model_copy(
        update={
            "fragments": [caption.fragments[0].model_copy(update={"page_number": 2})],
        }
    )
    elements = [figure, cross_page_caption, panel, label]
    assert order_linked_figure_content(elements) == elements


def test_linked_caption_never_crosses_unrelated_text_and_ordinary_caption_is_stable() -> None:
    figure, caption, panel, label = _closed_figure_component()
    unrelated = _figure_order_element(
        "ordinary-body",
        "paragraph",
        (30, 100, 150, 110),
        role="body",
    )
    interrupted = [figure, caption, panel, unrelated, label]
    already_ordered = [figure, panel, label, caption, unrelated]

    assert order_linked_figure_content(interrupted) == interrupted
    assert order_linked_figure_content(already_ordered) == already_ordered


def test_raster_figure_and_caption_are_separate_reciprocally_linked_elements(tmp_path: Path) -> None:
    path = tmp_path / "figure-caption.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((40, 70), "Native body text before the figure.")
    _insert_image(page, (80, 120, 520, 420), 0x336699)
    page.insert_text((80, 445), "Figure 1. Native raster evidence", fontsize=9)
    page.insert_text((40, 520), "Native body text after the figure.")
    _save(document, path)

    elements = extract_document_elements(path)

    figures = [element for element in elements if element.element_type == "figure"]
    captions = [element for element in elements if element.element_type == "caption"]
    assert len(figures) == len(captions) == 1
    figure = figures[0]
    caption = captions[0]
    assert figure.content == ""
    assert figure.structure.figure is not None
    assert figure.structure.figure.asset_path.startswith("pdf://page/1/image/")
    assert not figure.structure.figure.asset_path.startswith("data:")
    assert figure.structure.figure.caption_element_id == caption.element_id
    assert figure.structure.linked_element_ids == [caption.element_id]
    assert caption.structure.linked_element_ids == [figure.element_id]
    assert caption.content == "Figure 1. Native raster evidence"
    body_after = next(
        element for element in elements if element.content == "Native body text after the figure."
    )
    assert elements.index(figure) < elements.index(caption) < elements.index(body_after)


def test_strongly_bounded_vector_region_emits_an_extractive_figure(tmp_path: Path) -> None:
    path = tmp_path / "bounded-vector.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.draw_rect(pymupdf.Rect(80, 100, 520, 400), color=(0, 0, 0))
    for x, y, height in ((120, 180, 120), (190, 220, 80), (260, 150, 150), (330, 200, 100), (400, 130, 170)):
        page.draw_rect(pymupdf.Rect(x, y, x + 35, y + height), color=(0, 0, 0), fill=(0.2, 0.4, 0.7))
    page.insert_text((100, 385), "0 10 20 30 40 50", fontsize=7)
    page.insert_text((90, 115), "validation error (%)", fontsize=7)
    page.insert_text((80, 425), "Figure 8. Bounded native vector evidence", fontsize=9)
    _save(document, path)

    elements = extract_document_elements(path)

    figure = next(element for element in elements if element.element_type == "figure")
    caption = next(element for element in elements if element.element_type == "caption")
    assert figure.structure.figure is not None
    assert figure.structure.figure.asset_path.startswith("pdf://page/1/vector/")
    assert figure.structure.figure.caption_element_id == caption.element_id
    assert caption.structure.linked_element_ids == [figure.element_id]
    assert all(element.element_type != "table" for element in elements)
    residual_text = " ".join(element.content for element in elements if element.element_type != "figure")
    assert "0 10 20 30 40 50" in residual_text
    assert "validation error (%)" in residual_text


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize(
    ("scale", "translation", "reverse_text"),
    [(1.0, (0.0, 0.0), False), (1.5, (35.0, 45.0), True)],
)
def test_contained_figure_text_is_conserved_and_only_explicit_panels_remain_headings(
    tmp_path: Path,
    rotation: int,
    scale: float,
    translation: tuple[float, float],
    reverse_text: bool,
) -> None:
    path = tmp_path / f"contained-figure-text-{rotation}-{scale}.pdf"
    dx, dy = translation

    def transform(x: float, y: float) -> tuple[float, float]:
        return x * scale + dx, y * scale + dy

    document = pymupdf.open()
    page = document.new_page(width=600 * scale + 2 * dx, height=800 * scale + 2 * dy)
    page.draw_rect(pymupdf.Rect(*transform(80, 100), *transform(520, 430)), color=(0, 0, 0))
    for x, height in ((120, 120), (190, 80), (260, 150), (330, 100), (400, 170)):
        page.draw_rect(
            pymupdf.Rect(*transform(x, 400 - height), *transform(x + 35, 400)),
            color=(0, 0, 0),
            fill=(0.2, 0.4, 0.7),
        )
    text = [
        ((40, 70), "Document section", 14, 0),
        ((100, 130), "d) Humans are responsible", 12, 0),
        ((115, 385), "Increased concentrations", 12, 90),
    ]
    if reverse_text:
        text.reverse()
    for point, content, fontsize, rotate in text:
        page.insert_text(
            (point[0] * scale + dx, point[1] * scale + dy),
            content,
            fontsize=fontsize * scale,
            fontname="hebo",
            rotate=rotate,
        )
    page.insert_text(transform(80, 455), "Figure 8. Contained text evidence", fontsize=9 * scale)
    page.set_rotation(rotation)
    _save(document, path)

    elements = extract_document_elements(path)
    repeated = extract_document_elements(path)

    assert [element.model_dump(mode="json") for element in elements] == [
        element.model_dump(mode="json") for element in repeated
    ]
    figure = next(element for element in elements if element.element_type == "figure")
    panel = next(element for element in elements if element.content == "d) Humans are responsible")
    label = next(element for element in elements if element.content == "Increased concentrations")
    outside = next(element for element in elements if element.content == "Document section")
    panel_paragraph = panel.structure.paragraph
    label_paragraph = label.structure.paragraph
    outside_paragraph = outside.structure.paragraph
    assert panel_paragraph is not None
    assert label_paragraph is not None
    assert outside_paragraph is not None
    assert (panel.element_type, panel_paragraph.role, panel.include_in_output) == (
        "heading",
        "figure_panel_heading",
        True,
    )
    assert (label.element_type, label_paragraph.role, label.include_in_output) == (
        "paragraph",
        "figure_text",
        True,
    )
    assert figure.element_id in panel.structure.linked_element_ids
    assert figure.element_id in label.structure.linked_element_ids
    assert outside.element_type == "heading"
    assert outside_paragraph.role == "heading"
    assert figure.element_id not in outside.structure.linked_element_ids
    assert elements.index(figure) < elements.index(panel) < elements.index(label)


def test_uncaptioned_page_artwork_does_not_change_contained_prose_roles(tmp_path: Path) -> None:
    path = tmp_path / "uncaptioned-page-artwork.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    _insert_image(page, (60, 120, 540, 700), 0xDDEEFF)
    page.insert_text((90, 180), "Ordinary editorial heading", fontsize=16, fontname="hebo")
    page.insert_text(
        (90, 230),
        "Ordinary narrative prose remains document content over page artwork.",
        fontsize=10,
    )
    _save(document, path)

    elements = extract_document_elements(path)

    figure = next(element for element in elements if element.element_type == "figure")
    heading = next(element for element in elements if element.content == "Ordinary editorial heading")
    paragraph = next(
        element
        for element in elements
        if element.content == "Ordinary narrative prose remains document content over page artwork."
    )
    assert figure.structure.figure is not None
    assert figure.structure.figure.caption_element_id is None
    assert heading.element_type == "heading"
    assert heading.structure.paragraph is not None
    assert heading.structure.paragraph.role == "heading"
    assert paragraph.structure.paragraph is not None
    assert paragraph.structure.paragraph.role == "body"
    assert figure.element_id not in heading.structure.linked_element_ids
    assert figure.element_id not in paragraph.structure.linked_element_ids


def test_ipcc_page_59_preserves_panel_heading_and_demotes_rotated_group_label() -> None:
    path = Path("data/corpus/candidates/institutional/ipcc-ar6-syr.pdf")

    elements = extract_document_elements(path, pages=[59])

    figure = next(element for element in elements if element.element_type == "figure")
    panel = next(element for element in elements if element.content == "d) Humans are responsible")
    concentration = next(
        element
        for element in elements
        if element.content == "Increased concentrations of GHGs in the atmosphere"
    )
    emissions = next(
        element for element in elements if element.content == "Increased emissions of greenhouse gases (GHGs)"
    )
    panel_paragraph = panel.structure.paragraph
    assert panel_paragraph is not None
    assert (panel.element_type, panel_paragraph.role, panel.include_in_output) == (
        "heading",
        "figure_panel_heading",
        True,
    )
    for element in (concentration, emissions):
        paragraph = element.structure.paragraph
        assert paragraph is not None
        assert (element.element_type, paragraph.role, element.include_in_output) == (
            "paragraph",
            "figure_text",
            True,
        )
    assert all(
        figure.element_id in element.structure.linked_element_ids
        for element in (panel, concentration, emissions)
    )
    rendered_text = " ".join(element.content for element in elements if element.include_in_output)
    for native_text in (
        "d) Humans are responsible",
        "Increased concentrations",
        "of GHGs in the atmosphere",
        "Increased emissions of",
    ):
        assert rendered_text.count(native_text) == 1
    assert elements.index(figure) < elements.index(panel) < elements.index(concentration)
    caption = next(element for element in elements if element.element_type == "caption")
    linked_figure_text = [
        element
        for element in elements
        if figure.element_id in element.structure.linked_element_ids
        and element.element_type in {"heading", "paragraph"}
    ]
    assert linked_figure_text
    first_later_child = next(
        element for element in linked_figure_text if elements.index(element) > elements.index(caption)
    )
    intervening = elements[elements.index(caption) + 1 : elements.index(first_later_child)]
    assert not intervening
    assert elements.index(caption) < elements.index(first_later_child)
    assert any(
        element.structure.paragraph is not None and element.structure.paragraph.role == "body"
        for element in elements[elements.index(caption) + 1 :]
    )


def test_uncaptioned_bounded_vector_chart_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "uncaptioned-vector.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.draw_rect(pymupdf.Rect(80, 100, 520, 400), color=(0, 0, 0))
    for x, height in ((120, 120), (190, 80), (260, 150), (330, 100), (400, 170)):
        page.draw_rect(
            pymupdf.Rect(x, 380 - height, x + 35, 380),
            color=(0, 0, 0),
            fill=(0.2, 0.4, 0.7),
        )
    page.insert_text((40, 500), "Native body text remains canonical.")
    _save(document, path)

    elements = extract_document_elements(path)

    assert all(element.element_type != "figure" for element in elements)
    assert "Native body text remains canonical." in {element.content for element in elements}


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_vector_figure_accepts_caption_inside_its_lower_region(tmp_path: Path, rotation: int) -> None:
    path = tmp_path / f"inside-caption-vector-{rotation}.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.draw_rect(pymupdf.Rect(60, 80, 540, 620), color=(0, 0, 0))
    for x, height in ((100, 120), (180, 190), (260, 150), (340, 230), (420, 170)):
        page.draw_rect(
            pymupdf.Rect(x, 400 - height, x + 45, 400),
            color=(0, 0, 0),
            fill=(0.2, 0.4, 0.7),
        )
    page.insert_text((80, 450), "Upper chart annotation remains extractive.", fontsize=10)
    page.insert_text((80, 500), "Figure 2.1: Native caption inside the lower panel", fontsize=9)
    page.insert_text((80, 550), "Lower chart annotation remains extractive.", fontsize=10)
    page.set_rotation(rotation)
    _save(document, path)

    elements = extract_document_elements(path)

    figure = next(element for element in elements if element.element_type == "figure")
    caption = next(element for element in elements if element.element_type == "caption")
    upper = next(
        element for element in elements if element.content == "Upper chart annotation remains extractive."
    )
    lower = next(
        element for element in elements if element.content == "Lower chart annotation remains extractive."
    )
    assert figure.structure.figure is not None
    assert figure.structure.figure.caption_element_id == caption.element_id
    assert caption.structure.linked_element_ids == [figure.element_id]
    assert all(figure.element_id in element.structure.linked_element_ids for element in (upper, lower))
    assert elements.index(figure) < elements.index(upper) < elements.index(lower) < elements.index(caption)


def test_mid_panel_full_width_caption_does_not_confirm_outer_container_as_figure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outer-editorial-panel.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.draw_rect(pymupdf.Rect(50, 80, 550, 700), color=(0, 0, 0))
    page.insert_text((70, 120), "Box 4.3 Ordinary editorial context", fontsize=14, fontname="hebo")
    page.insert_text(
        (70, 170),
        "This prose explains the topic before the independently captioned chart.",
        fontsize=10,
    )
    page.insert_text(
        (70, 340),
        "Figure 4.5 Broad chart caption spanning almost the entire editorial panel width",
        fontsize=9,
    )
    page.draw_rect(pymupdf.Rect(70, 370, 530, 650), color=(0, 0, 0))
    for x, height in ((110, 80), (190, 140), (270, 110), (350, 170), (430, 130)):
        page.draw_rect(
            pymupdf.Rect(x, 620 - height, x + 35, 620),
            color=(0, 0, 0),
            fill=(0.2, 0.4, 0.7),
        )
    _save(document, path)

    elements = extract_document_elements(path)

    figures = [element for element in elements if element.element_type == "figure"]
    assert len(figures) <= 1
    if figures:
        assert figures[0].fragments[0].bbox.y0 >= 370
    heading = next(element for element in elements if element.content == "Box 4.3 Ordinary editorial context")
    prose = next(
        element
        for element in elements
        if element.content == "This prose explains the topic before the independently captioned chart."
    )
    assert heading.structure.paragraph is not None
    assert heading.structure.paragraph.role == "heading"
    assert prose.structure.paragraph is not None
    assert prose.structure.paragraph.role == "body"


def test_ambiguous_and_missing_captions_do_not_create_links(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous-caption.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    _insert_image(page, (80, 100, 300, 280), 0xAA3300)
    _insert_image(page, (250, 100, 520, 280), 0x0033AA)
    page.insert_text((180, 305), "Figure 2. Ambiguous shared caption", fontsize=9)
    _insert_image(page, (80, 420, 520, 650), 0x339933)
    page.insert_text((40, 730), "Ordinary native body control text.")
    _save(document, path)

    elements = extract_document_elements(path)

    figures = [element for element in elements if element.element_type == "figure"]
    captions = [element for element in elements if element.element_type == "caption"]
    assert len(figures) == 3
    assert len(captions) == 1
    assert all(figure.structure.figure is not None for figure in figures)
    assert all(
        figure.structure.figure.caption_element_id is None for figure in figures if figure.structure.figure
    )
    assert all(not figure.structure.linked_element_ids for figure in figures)
    assert captions[0].structure.linked_element_ids == []


def test_raster_region_overlapping_a_table_is_excluded() -> None:
    payload = {
        "blocks": [
            {
                "type": 1,
                "bbox": (80, 100, 520, 400),
                "width": 120,
                "height": 80,
                "image": b"native-image-payload",
            }
        ]
    }

    candidates = raster_candidates_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        table_bboxes=[(60, 80, 540, 420)],
    )

    assert candidates == []


def test_repeated_raster_logo_is_rejected_across_selected_pages(tmp_path: Path) -> None:
    path = tmp_path / "repeated-logo.pdf"
    document = pymupdf.open()
    logo = _pixmap(120, 80, 0x224466)
    for page_number in (1, 2):
        page = document.new_page(width=600, height=800)
        page.insert_image(pymupdf.Rect(40, 30, 160, 90), pixmap=logo)
        page.insert_text((40, 180), f"Unique native page body {page_number}.")
    _save(document, path)

    elements = extract_document_elements(path)

    assert all(element.element_type != "figure" for element in elements)
    assert {element.content for element in elements if element.include_in_output} == {
        "Unique native page body 1.",
        "Unique native page body 2.",
    }


def test_rotated_figure_geometry_is_normalized_once_and_source_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "rotated-figure.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    _insert_image(page, (10, 20, 100, 80), 0x884422)
    page.insert_text((110, 60), "Native control text.", fontsize=8)
    page.set_rotation(90)
    _save(document, path)
    original = path.read_bytes()

    elements = extract_document_elements(path, pages=[1])

    figure = next(element for element in elements if element.element_type == "figure")
    expected = normalize_bbox((10, 20, 100, 80), 200, 100, 90)
    assert figure.fragments[0].bbox == BoundingBox(
        x0=expected[0], y0=expected[1], x1=expected[2], y1=expected[3]
    )
    assert figure.fragments[0].page_width == 100
    assert figure.fragments[0].page_height == 200
    assert path.read_bytes() == original


def test_figure_uses_existing_column_plan_for_atomic_insertion(tmp_path: Path) -> None:
    path = tmp_path / "two-column-figure.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((40, 100), "Left column first has broad geometric evidence.", fontsize=10)
    _insert_image(page, (40, 200, 270, 400), 0x663399)
    page.insert_text((40, 500), "Left column second has broad geometric evidence.", fontsize=10)
    page.insert_text((330, 100), "Right column first has broad geometric evidence.", fontsize=10)
    page.insert_text((330, 500), "Right column second has broad geometric evidence.", fontsize=10)
    _save(document, path)

    elements = extract_document_elements(path)

    assert [(element.element_type, element.content) for element in elements] == [
        ("paragraph", "Left column first has broad geometric evidence."),
        ("figure", ""),
        ("paragraph", "Left column second has broad geometric evidence."),
        ("paragraph", "Right column first has broad geometric evidence."),
        ("paragraph", "Right column second has broad geometric evidence."),
    ]


def test_uncaptioned_elongated_raster_ornament_is_rejected_but_caption_bypasses_gate() -> None:
    payload = {
        "blocks": [
            {
                "type": 1,
                "bbox": (50, 300, 550, 350),
                "width": 1000,
                "height": 100,
                "image": b"elongated-native-brand-strip",
            }
        ]
    }

    uncaptioned = raster_candidates_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
    )
    captioned = raster_candidates_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        figure_caption_bboxes=[(50, 355, 550, 370)],
    )

    assert uncaptioned == []
    assert len(captioned) == 1


def test_overlapping_placements_of_same_raster_asset_emit_once() -> None:
    payload = {
        "blocks": [
            {
                "type": 1,
                "bbox": (100, 200, 300, 320),
                "width": 400,
                "height": 240,
                "image": b"same-native-image",
            },
            {
                "type": 1,
                "bbox": (110, 200, 300, 320),
                "width": 400,
                "height": 240,
                "image": b"same-native-image",
            },
        ]
    }

    candidates = raster_candidates_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
    )

    assert len(candidates) == 1
    assert candidates[0].native_bbox == (100.0, 200.0, 300.0, 320.0)


def test_page_spanning_edge_vector_cluster_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "page-spanning-vector.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.draw_rect(pymupdf.Rect(10, 0, 590, 500), color=(0, 0, 0))
    for x, height in ((80, 180), (170, 120), (260, 220), (350, 160), (440, 200)):
        page.draw_rect(
            pymupdf.Rect(x, 360 - height, x + 45, 360),
            color=(0, 0, 0),
            fill=(0.2, 0.4, 0.7),
        )
    page.insert_text((80, 410), "Figure 3. False page-spanning vector cluster", fontsize=9)
    _save(document, path)

    elements = extract_document_elements(path)

    assert all(element.element_type != "figure" for element in elements)
    assert "Figure 3. False page-spanning vector cluster" in {element.content for element in elements}


def test_no_image_control_and_page_subset_do_not_emit_figures(tmp_path: Path) -> None:
    path = tmp_path / "no-image.pdf"
    document = pymupdf.open()
    for page_number in (1, 2):
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 100), f"Native no-image control page {page_number}.")
    _save(document, path)

    elements = extract_document_elements(path, pages=[2])

    assert all(element.element_type != "figure" for element in elements)
    assert {fragment.page_number for element in elements for fragment in element.fragments} == {2}


def test_captioned_raster_wins_over_false_table_without_losing_native_text(tmp_path: Path) -> None:
    path = tmp_path / "captioned-chart-grid.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    _insert_image(page, (80, 100, 520, 400), 0x336699)
    for x in (80, 300, 520):
        page.draw_line((x, 100), (x, 400))
    for y in (100, 250, 400):
        page.draw_line((80, y), (520, y))
    for x, y, text in ((100, 140, "A"), (320, 140, "B"), (100, 290, "C"), (320, 290, "D")):
        page.insert_text((x, y), text)
    page.insert_text((80, 425), "Figure 9. Explicit bounded chart evidence", fontsize=9)
    _save(document, path)

    elements = extract_document_elements(path)

    assert [element.element_type for element in elements].count("figure") == 1
    assert all(element.element_type != "table" for element in elements)
    assert {element.content for element in elements if element.content in {"A B", "C D"}} == {
        "A B",
        "C D",
    }
    figure = next(element for element in elements if element.element_type == "figure")
    caption = next(element for element in elements if element.element_type == "caption")
    assert figure.structure.figure is not None
    assert figure.structure.figure.caption_element_id == caption.element_id
    assert caption.structure.linked_element_ids == [figure.element_id]


def test_cropbox_origin_and_rotation_use_crop_relative_geometry(tmp_path: Path) -> None:
    path = tmp_path / "cropped-rotated-figure.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    _insert_image(page, (150, 200, 450, 500), 0x557799)
    page.insert_text((140, 170), "Crop-relative native control text.")
    page.set_cropbox(pymupdf.Rect(100, 100, 550, 700))
    page.set_rotation(90)
    _save(document, path)
    original = path.read_bytes()

    first = extract_document_elements(path, pages=[1])
    second = extract_document_elements(path, pages=[1])

    figure = next(element for element in first if element.element_type == "figure")
    expected = normalize_bbox((50, 150, 350, 350), 450, 600, 90)
    assert figure.fragments[0].bbox == BoundingBox(
        x0=expected[0], y0=expected[1], x1=expected[2], y1=expected[3]
    )
    assert figure.fragments[0].page_width == 600
    assert figure.fragments[0].page_height == 450
    assert "Crop-relative native control text." in {element.content for element in first}
    assert [element.model_dump(mode="json") for element in first] == [
        element.model_dump(mode="json") for element in second
    ]
    assert path.read_bytes() == original


def test_degenerate_vector_cluster_with_caption_is_rejected_without_division_error() -> None:
    candidates = vector_candidates_from_drawings(
        [
            {
                "rect": (80, 100, 500, 100),
                "items": [("re", (80, 100, 500, 100)) for _ in range(6)],
            }
        ],
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
        figure_caption_bboxes=[(80, 110, 500, 125)],
    )

    assert candidates == []


def test_malformed_drawing_payloads_fail_fast() -> None:
    with pytest.raises(TypeError, match="drawing must be a mapping"):
        vector_candidates_from_drawings(
            ["not-a-mapping"],
            page_number=1,
            page_width=600,
            page_height=800,
            rotation=0,
        )


def test_wholly_off_page_transformed_image_is_not_a_candidate() -> None:
    candidates = raster_candidates_from_pymupdf_dict(
        {
            "blocks": [
                {
                    "type": 1,
                    "bbox": (0, 320, -64, 500),
                    "width": 120,
                    "height": 80,
                    "image": b"valid-off-page-image",
                }
            ]
        },
        page_number=1,
        page_width=600,
        page_height=800,
        rotation=0,
    )

    assert candidates == []


def test_malformed_image_payloads_fail_fast() -> None:
    valid = {
        "type": 1,
        "bbox": (80, 100, 520, 400),
        "width": 120,
        "height": 80,
        "image": b"native-image-payload",
    }

    def parse(block: Mapping[str, object]) -> None:
        raster_candidates_from_pymupdf_dict(
            {"blocks": [dict(block)]},
            page_number=1,
            page_width=600,
            page_height=800,
            rotation=0,
        )

    with pytest.raises(TypeError, match="payload must be bytes"):
        parse(valid | {"image": "not-bytes"})
    with pytest.raises(ValueError, match="must not be empty"):
        parse(valid | {"image": b""})
    with pytest.raises(TypeError, match="width and height must be integers"):
        parse(valid | {"width": True})
    with pytest.raises(ValueError, match="positive width and height"):
        parse(valid | {"bbox": (80, 100, 80, 400)})
