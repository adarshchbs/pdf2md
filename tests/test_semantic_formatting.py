import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf
import pytest
from pydantic import ValidationError

from app.pdf2md.engine import (
    DecorationLine,
    _decoration_flags,  # pyright: ignore[reportPrivateUsage]
    _semantic_elements,  # pyright: ignore[reportPrivateUsage]
    _text_blocks_from_pymupdf_dict,  # pyright: ignore[reportPrivateUsage]
    extract_document_elements,
    render_document,
)
from app.pdf2md.evaluation import evaluate_document
from app.pdf2md.schema import (
    LEGACY_DOCUMENT_SCHEMA,
    PREVIOUS_DOCUMENT_SCHEMA,
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FootnoteStructure,
    InlineFootnoteMarkerRange,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
    TextStyleRun,
    inline_footnote_marker_ranges,
    read_document_elements,
    write_document_elements,
)
from app.pdf2md.semantic_text import (
    SemanticBlock,
    TextBlock,
    TextLine,
    TextSpan,
    classify_semantic_blocks,
    join_paragraphs,
    link_footnotes,
)


def _annotation() -> AnnotationMetadata:
    return AnnotationMetadata(
        stage="candidate",
        revision=1,
        annotator="test",
        confidence=1,
        adjudication_status="unreviewed",
    )


def _fragment() -> PageFragment:
    return PageFragment(
        page_number=1,
        page_width=200,
        page_height=200,
        bbox=BoundingBox(x0=10, y0=10, x1=190, y1=30),
    )


def _element(
    content: str,
    *,
    element_type: str = "paragraph",
    paragraph: ParagraphStructure | None = None,
    runs: list[TextStyleRun] | None = None,
) -> DocumentElement:
    return DocumentElement(
        document_id="document",
        element_id="element",
        order=0,
        element_type=element_type,
        content=content,
        format="text",
        fragments=[_fragment()],
        structure=ElementStructure(
            paragraph=paragraph or ParagraphStructure(role="body"),
            style_runs=runs or [],
        ),
        annotation=_annotation(),
    )


def _footnote_element(footnote: FootnoteStructure) -> DocumentElement:
    return DocumentElement(
        document_id="document",
        element_id="footnote",
        order=0,
        element_type="footnote",
        content="1. Note.",
        format="text",
        fragments=[_fragment()],
        structure=ElementStructure(
            paragraph=ParagraphStructure(role="footnote"),
            footnote=footnote,
        ),
        annotation=_annotation(),
    )


def _block(lines: tuple[TextLine, ...]) -> TextBlock:
    return TextBlock(
        lines=lines,
        bbox=(10, 10, 190, 50),
        page_number=1,
        page_width=200,
        page_height=200,
    )


def test_mixed_style_spans_preserve_plain_content_offsets_and_render_semantics() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "dir": [1, 0],
                        "spans": [
                            {
                                "text": "Plain ",
                                "bbox": [10, 10, 42, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 0,
                            },
                            {
                                "text": "bold",
                                "bbox": [42, 10, 66, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 16,
                            },
                            {
                                "text": " and ",
                                "bbox": [66, 10, 91, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 0,
                            },
                            {
                                "text": "italic",
                                "bbox": [91, 10, 125, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 2,
                            },
                        ],
                    }
                ],
            }
        ]
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=200,
        page_height=200,
        rotation=0,
        table_bboxes=[],
    )
    semantic = classify_semantic_blocks(blocks, body_font_size=10)[0]
    element = _semantic_elements((semantic,), "document", "test")[0]

    assert element.content == "Plain bold and italic"
    assert [(run.start, run.end, run.bold, run.italic) for run in element.structure.style_runs] == [
        (0, 6, False, False),
        (6, 10, True, False),
        (10, 15, False, False),
        (15, 21, False, True),
    ]
    assert render_document([element]) == "Plain **bold** and *italic*"


def test_list_marker_projection_does_not_search_repeated_body_text() -> None:
    marker = TextSpan("(a) ", (10, 10, 30, 20), 10, "Body")
    repeated = TextSpan("(a) body", (31, 10, 75, 20), 10, "Body", is_bold=True)
    line = TextLine((marker, repeated), (10, 10, 75, 20))
    semantic = join_paragraphs([_block((line,))])[0]

    assert semantic.text == "(a) body"
    assert semantic.list_label == "(a)"
    assert [(run.start, run.end, run.bold) for run in semantic.style_runs] == [(0, 8, True)]


def test_nfkc_expansion_keeps_explicit_style_mapping() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "oﬃce",
                                "bbox": [10, 10, 50, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 16,
                            }
                        ]
                    }
                ],
            }
        ]
    }
    block = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=200,
        page_height=200,
        rotation=0,
        table_bboxes=[],
    )[0]
    semantic = classify_semantic_blocks([block], body_font_size=10)[0]

    assert semantic.text == "office"
    assert [(run.start, run.end, run.bold) for run in semantic.style_runs] == [(0, 6, True)]


def test_join_and_soft_hyphen_dehyphenation_rebase_style_offsets_deterministically() -> None:
    first = TextLine(
        spans=(TextSpan("inter\N{SOFT HYPHEN}", (10, 10, 40, 20), 10, "Body", is_bold=True),),
        bbox=(10, 10, 40, 20),
    )
    second = TextLine(
        spans=(TextSpan("operability", (10, 25, 65, 35), 10, "Body", is_italic=True),),
        bbox=(10, 25, 65, 35),
    )
    semantic = SemanticBlock(
        kind="paragraph", text="interoperability", source_blocks=(_block((first, second)),)
    )

    assert [(run.start, run.end, run.bold, run.italic) for run in semantic.style_runs] == [
        (0, 5, True, False),
        (5, 16, False, True),
    ]
    assert (
        semantic.style_runs
        == SemanticBlock(
            kind="paragraph",
            text="interoperability",
            source_blocks=(_block((first, second)),),
        ).style_runs
    )


def _line(x0: float, y: float, x1: float, *, segments: int = 1) -> DecorationLine:
    return DecorationLine(x0, y, x1, 0.5, 200, 200, 0, segments, "line")


def test_footnote_marker_replacement_rebases_later_styles_without_substring_matching() -> None:
    line = TextLine(
        (
            TextSpan("Bold", (10, 10, 35, 20), 10, "Body", is_bold=True),
            TextSpan("1", (35, 7, 39, 13), 6, "Body"),
            TextSpan(" tail", (39, 10, 65, 20), 10, "Body", is_italic=True),
        ),
        (10, 7, 65, 20),
    )
    reference = SemanticBlock(kind="paragraph", text="Bold1 tail", source_blocks=(_block((line,)),))
    note_line = TextLine((TextSpan("1. Note.", (10, 30, 55, 40), 8, "Body"),), (10, 30, 55, 40))
    note = SemanticBlock(kind="footnote", text="1. Note.", source_blocks=(_block((note_line,)),))
    semantics = (reference, note)

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)[0]

    assert linked.content == "Bold[^1] tail"
    assert [(run.start, run.end, run.bold, run.italic) for run in linked.structure.style_runs] == [
        (0, 4, True, False),
        (8, 13, False, True),
    ]
    assert inline_footnote_marker_ranges(linked) == (InlineFootnoteMarkerRange(start=4, end=8, label="1"),)


def test_repeated_mixed_split_markers_are_resolved_against_original_content() -> None:
    line = TextLine(
        (
            TextSpan("Alpha", (10, 10, 40, 20), 10, "Body", is_bold=True),
            TextSpan("1", (40, 7, 44, 13), 6, "Body"),
            TextSpan(" beta ", (44, 10, 78, 20), 10, "Body"),
            TextSpan("2", (78, 7, 82, 13), 6, "Body"),
            TextSpan(" gamma ", (82, 10, 122, 20), 10, "Body", is_italic=True),
            TextSpan("1", (122, 7, 126, 13), 6, "Body"),
            TextSpan(" end", (126, 10, 148, 20), 10, "Body", is_bold=True),
        ),
        (10, 7, 148, 20),
    )
    reference = SemanticBlock(
        kind="paragraph", text="Alpha1 beta 2 gamma 1 end", source_blocks=(_block((line,)),)
    )
    note_1_line = TextLine((TextSpan("1. First.", (10, 150, 55, 160), 8, "Body"),), (10, 150, 55, 160))
    note_2_line = TextLine((TextSpan("2. Second.", (10, 165, 60, 175), 8, "Body"),), (10, 165, 60, 175))
    semantics = (
        reference,
        SemanticBlock(kind="footnote", text="1. First.", source_blocks=(_block((note_1_line,)),)),
        SemanticBlock(kind="footnote", text="2. Second.", source_blocks=(_block((note_2_line,)),)),
    )

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)
    relinked = link_footnotes(linked, semantics)

    assert linked[0].content == "Alpha[^1] beta [^2] gamma [^1] end"
    assert relinked == linked
    assert linked[0].content.count("[^1]") == 2
    assert inline_footnote_marker_ranges(linked[0]) == (
        InlineFootnoteMarkerRange(start=5, end=9, label="1"),
        InlineFootnoteMarkerRange(start=15, end=19, label="2"),
        InlineFootnoteMarkerRange(start=26, end=30, label="1"),
    )
    styled = [
        (linked[0].content[run.start : run.end], run.bold, run.italic)
        for run in linked[0].structure.style_runs
        if run.bold or run.italic
    ]
    assert styled == [
        ("Alpha", True, False),
        (" gamma ", False, True),
        (" end", True, False),
    ]


def test_literal_and_generated_same_label_use_exact_marker_occurrences() -> None:
    line = TextLine(
        (
            TextSpan("Literal [^1] Word", (10, 10, 100, 20), 10, "Body", is_bold=True),
            TextSpan("1", (100, 7, 104, 13), 6, "Body"),
        ),
        (10, 7, 104, 20),
    )
    note_line = TextLine(
        (TextSpan("1. Note.", (10, 150, 55, 160), 8, "Body"),),
        (10, 150, 55, 160),
    )
    semantics = (
        SemanticBlock(
            kind="paragraph",
            text="Literal [^1] Word1",
            source_blocks=(_block((line,)),),
        ),
        SemanticBlock(
            kind="footnote",
            text="1. Note.",
            source_blocks=(_block((note_line,)),),
        ),
    )

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)
    validated = [DocumentElement.model_validate_json(element.model_dump_json()) for element in linked]

    assert validated[0].content == "Literal [^1] Word[^1]"
    assert inline_footnote_marker_ranges(validated[0]) == (
        InlineFootnoteMarkerRange(start=17, end=21, label="1"),
    )
    assert render_document([validated[0]]) == "**Literal \\[^1\\] Word**[^1]"
    assert link_footnotes(validated, semantics) == validated


def test_legacy_label_only_marker_metadata_never_protects_literal_syntax() -> None:
    element = _element("Literal [^1]").model_copy(
        update={
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="body"),
                properties=[StructureProperty(key="inline_footnote_markers", value='["1"]')],
            )
        }
    )

    assert inline_footnote_marker_ranges(element) == ()
    assert render_document([element]) == "Literal \\[^1\\]"


@pytest.mark.parametrize(
    "properties",
    [
        [
            StructureProperty(
                key="inline_footnote_markers",
                value='[{"start":8,"end":12,"label":"1"},{"start":8,"end":12,"label":"1"}]',
            )
        ],
        [
            StructureProperty(
                key="inline_footnote_markers",
                value='[{"start":30,"end":34,"label":"1"}]',
            )
        ],
        [
            StructureProperty(key="inline_footnote_markers", value='["1"]'),
            StructureProperty(key="inline_footnote_markers", value='["1"]'),
        ],
        [
            StructureProperty(
                key="inline_footnote_markers",
                value='["1",{"start":8,"end":12,"label":"1"}]',
            )
        ],
    ],
)
def test_invalid_exact_marker_metadata_fails_closed(
    properties: list[StructureProperty],
) -> None:
    element = _element("Literal [^1]").model_copy(
        update={
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="body"),
                properties=properties,
            )
        }
    )

    with pytest.raises(ValueError):
        inline_footnote_marker_ranges(element)


def test_identical_repeated_marker_contexts_are_not_deduplicated() -> None:
    first = TextLine(
        (
            TextSpan("Same", (10, 10, 35, 20), 10, "Body"),
            TextSpan("1", (35, 7, 39, 13), 6, "Body"),
        ),
        (10, 7, 39, 20),
    )
    second = TextLine(
        (
            TextSpan("Same", (10, 25, 35, 35), 10, "Body"),
            TextSpan("1", (35, 22, 39, 28), 6, "Body"),
        ),
        (10, 22, 39, 35),
    )
    reference = SemanticBlock(kind="paragraph", text="Same1 Same1", source_blocks=(_block((first, second)),))
    note_line = TextLine((TextSpan("1. Note.", (10, 150, 55, 160), 8, "Body"),), (10, 150, 55, 160))
    semantics = (
        reference,
        SemanticBlock(kind="footnote", text="1. Note.", source_blocks=(_block((note_line,)),)),
    )

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)

    assert linked[0].content == "Same[^1] Same[^1]"


def test_nfkc_normalized_repeated_footnote_markers_are_all_rewritten() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Word",
                                "bbox": [10, 10, 40, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 0,
                            },
                            {"text": "１", "bbox": [40, 7, 44, 13], "size": 6, "font": "Body", "flags": 0},
                            {
                                "text": " and again",
                                "bbox": [44, 10, 100, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 0,
                            },
                            {"text": "１", "bbox": [100, 7, 104, 13], "size": 6, "font": "Body", "flags": 0},
                        ]
                    }
                ],
            },
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "１. Note.",
                                "bbox": [10, 150, 55, 160],
                                "size": 8,
                                "font": "Body",
                                "flags": 0,
                            }
                        ]
                    }
                ],
            },
        ],
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=200,
        page_height=200,
        rotation=0,
        table_bboxes=[],
    )
    semantics = (
        SemanticBlock(kind="paragraph", text=blocks[0].text, source_blocks=(blocks[0],)),
        SemanticBlock(kind="footnote", text=blocks[1].text, source_blocks=(blocks[1],)),
    )

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)

    assert blocks[0].text == "Word1 and again1"
    assert linked[0].content == "Word[^1] and again[^1]"
    assert inline_footnote_marker_ranges(linked[0]) == (
        InlineFootnoteMarkerRange(start=4, end=8, label="1"),
        InlineFootnoteMarkerRange(start=18, end=22, label="1"),
    )


def test_embedded_unicode_superscripts_survive_nfkc_with_split_styles_and_repetition() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Word¹",
                                "bbox": [10, 10, 44, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 16,
                            },
                            {
                                "text": " and again",
                                "bbox": [44, 10, 100, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 2,
                            },
                            {
                                "text": "¹",
                                "bbox": [100, 7, 104, 13],
                                "size": 6,
                                "font": "Body",
                                "flags": 0,
                            },
                        ]
                    }
                ],
            },
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "¹. Note.",
                                "bbox": [10, 150, 55, 160],
                                "size": 8,
                                "font": "Body",
                                "flags": 0,
                            }
                        ]
                    }
                ],
            },
        ],
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=200,
        page_height=200,
        rotation=0,
        table_bboxes=[],
    )
    semantics = (
        SemanticBlock(kind="paragraph", text=blocks[0].text, source_blocks=(blocks[0],)),
        SemanticBlock(kind="footnote", text=blocks[1].text, source_blocks=(blocks[1],)),
    )

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)

    assert blocks[0].text == "Word1 and again1"
    assert linked[0].content == "Word[^1] and again[^1]"
    assert inline_footnote_marker_ranges(linked[0]) == (
        InlineFootnoteMarkerRange(start=4, end=8, label="1"),
        InlineFootnoteMarkerRange(start=18, end=22, label="1"),
    )
    styled = [
        (linked[0].content[run.start : run.end], run.bold, run.italic)
        for run in linked[0].structure.style_runs
        if run.bold or run.italic
    ]
    assert styled == [("Word", True, False), (" and again", False, True)]


@pytest.mark.parametrize(
    ("spans", "normalized", "linked_content"),
    [
        (
            [
                {"text": "oﬃce¹", "bbox": [10, 10, 48, 20], "size": 10, "font": "Body", "flags": 16},
            ],
            "office1",
            "office[^1]",
        ),
        (
            [
                {"text": "Word", "bbox": [10, 10, 40, 20], "size": 10, "font": "Body", "flags": 16},
                {"text": "¹", "bbox": [40, 10, 44, 20], "size": 10, "font": "Body", "flags": 2},
            ],
            "Word1",
            "Word[^1]",
        ),
        (
            [
                {"text": "Word", "bbox": [10, 10, 40, 20], "size": 10, "font": "Body", "flags": 16},
                {"text": "¹", "bbox": [40, 10, 40, 20], "size": 10, "font": "Body", "flags": 2},
            ],
            "Word1",
            "Word[^1]",
        ),
        (
            [
                {"text": "Word¹", "bbox": [10, 10, 44, 20], "size": 10, "font": "Body", "flags": 16},
                {"text": " again", "bbox": [44, 10, 76, 20], "size": 10, "font": "Body", "flags": 0},
                {"text": "¹", "bbox": [76, 10, 80, 20], "size": 10, "font": "Body", "flags": 2},
            ],
            "Word1 again1",
            "Word[^1] again[^1]",
        ),
    ],
)
def test_raw_superscript_provenance_is_projected_across_normalized_logical_spans(
    spans: list[dict[str, object]], normalized: str, linked_content: str
) -> None:
    payload = {
        "blocks": [
            {"type": 0, "lines": [{"spans": spans}]},
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "¹. Note.",
                                "bbox": [10, 150, 55, 160],
                                "size": 8,
                                "font": "Body",
                                "flags": 0,
                            }
                        ]
                    }
                ],
            },
        ],
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=200,
        page_height=200,
        rotation=0,
        table_bboxes=[],
    )
    semantics = (
        SemanticBlock(kind="paragraph", text=blocks[0].text, source_blocks=(blocks[0],)),
        SemanticBlock(kind="footnote", text=blocks[1].text, source_blocks=(blocks[1],)),
    )

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)

    assert blocks[0].text == normalized
    assert linked[0].content == linked_content
    assert all(
        linked[0].content[run.start : run.end] not in {"[^1]", "1"} for run in linked[0].structure.style_runs
    )


def test_nfkc_does_not_reinterpret_embedded_ordinary_digits_as_footnotes() -> None:
    payload = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "Word1 and again1",
                                "bbox": [10, 10, 104, 20],
                                "size": 10,
                                "font": "Body",
                                "flags": 0,
                            }
                        ]
                    }
                ],
            },
            {
                "type": 0,
                "lines": [
                    {
                        "spans": [
                            {
                                "text": "1. Note.",
                                "bbox": [10, 150, 55, 160],
                                "size": 8,
                                "font": "Body",
                                "flags": 0,
                            }
                        ]
                    }
                ],
            },
        ],
    }
    blocks = _text_blocks_from_pymupdf_dict(
        payload,
        page_number=1,
        page_width=200,
        page_height=200,
        rotation=0,
        table_bboxes=[],
    )
    semantics = (
        SemanticBlock(kind="paragraph", text=blocks[0].text, source_blocks=(blocks[0],)),
        SemanticBlock(kind="footnote", text=blocks[1].text, source_blocks=(blocks[1],)),
    )

    linked = link_footnotes(_semantic_elements(semantics, "document", "test"), semantics)

    assert linked[0].content == "Word1 and again1"
    assert linked[0].structure.linked_element_ids == []


def test_decoration_geometry_accepts_baseline_local_lines_and_rejects_rules() -> None:
    bbox = (10.0, 10.0, 60.0, 20.0)
    underline = _decoration_flags(bbox, 10, 18, (1, 0), [_line(11, 19, 59)])
    strikeout = _decoration_flags(bbox, 10, 18, (1, 0), [_line(11, 14.5, 59)])
    long_rule = _decoration_flags(bbox, 10, 18, (1, 0), [_line(0, 19, 100)])
    wrong_height = _decoration_flags(bbox, 10, 18, (1, 0), [_line(11, 22, 59)])
    missing_baseline = _decoration_flags(bbox, 10, None, (1, 0), [_line(11, 19, 59)])
    rotated = _decoration_flags(bbox, 10, 18, (0, 1), [_line(11, 19, 59)])

    assert underline == (True, False)
    assert strikeout == (False, True)
    assert long_rule == wrong_height == missing_baseline == rotated == (False, False)
    assert _decoration_flags((110, 110, 160, 120), 10, 118, (1, 0), [_line(111, 119, 159)]) == underline
    assert _decoration_flags(bbox, 10, 18, (1, 0), [_line(11, 19, 59), _line(12, 19.2, 58)]) == (False, False)
    assert _decoration_flags(bbox, 10, 18, (1, 0), [_line(11, 19, 59, segments=2)]) == (
        False,
        False,
    )
    table = BoundingBox(x0=0, y0=0, x1=100, y1=100)
    assert _decoration_flags(bbox, 10, 18, (1, 0), [_line(11, 19, 59)], table_bboxes=[table]) == (
        False,
        False,
    )


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_synthetic_pdf_decoration_classification_fails_closed_on_rotated_pages(
    tmp_path: Path, rotation: int
) -> None:
    pdf_path = tmp_path / f"decorations-{rotation}.pdf"
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 40), "underlined", fontsize=12)
    page.draw_line((20, 41), (77, 41), width=0.5)
    page.insert_text((20, 70), "not underlined", fontsize=12)
    page.draw_line((0, 71), (200, 71), width=0.5)
    page.set_rotation(rotation)
    document.save(pdf_path)
    document.close()

    elements = extract_document_elements(pdf_path)
    paragraph = next(element for element in elements if element.content.startswith("underlined"))
    first_end = len("underlined")
    second_start = paragraph.content.index("not")
    underlined = any(
        run.underline and run.start == 0 and run.end == first_end for run in paragraph.structure.style_runs
    )

    assert underlined is (rotation == 0)
    assert not any(run.underline and run.end > second_start for run in paragraph.structure.style_runs)


def test_heading_hierarchy_list_evidence_and_rendering_are_schema_visible() -> None:
    title_span = TextSpan("Document Title", (10, 10, 100, 25), 18, "Display", is_bold=True)
    title_block = _block((TextLine((title_span,), title_span.bbox),))
    title = SemanticBlock(
        kind="heading", text="Document Title", source_blocks=(title_block,), heading_level=1
    )
    item_span = TextSpan("1. First item", (10, 35, 100, 45), 10, "Body")
    item_block = _block((TextLine((item_span,), item_span.bbox),))
    item = SemanticBlock(
        kind="paragraph",
        text="First item",
        source_blocks=(item_block,),
        paragraph_role="list_item",
        list_depth=0,
        list_label="1.",
    )
    title_element, item_element = _semantic_elements((title, item), "document", "test")
    item_element = item_element.model_copy(update={"order": 1, "element_id": "item"})

    assert title_element.structure.paragraph is not None
    assert title_element.structure.paragraph.title_evidence == ["heading_level_1", "first_page_upper_region"]
    assert "inferred_heading_level" in title_element.structure.paragraph.heading_evidence
    assert item_element.structure.paragraph is not None
    assert item_element.structure.paragraph.list_evidence == ["explicit_leading_label"]
    assert render_document([title_element, item_element]) == "# **Document Title**\n\n1. First item"


def test_supported_style_rendering_uses_html_underline_and_markdown_strikeout() -> None:
    element = _element(
        "under and strike",
        runs=[
            TextStyleRun(start=0, end=5, underline=True),
            TextStyleRun(start=10, end=16, strikeout=True),
        ],
    )
    assert render_document([element]) == "<u>under</u> and ~~strike~~"


def test_renderer_rejects_colliding_generated_footnote_labels() -> None:
    first = _footnote_element(
        FootnoteStructure(label="1", reference_element_ids=["reference-1"], association_confident=True)
    )
    second = first.model_copy(
        update={
            "element_id": "footnote-2",
            "order": 1,
            "fragments": [_fragment().model_copy(update={"page_number": 2})],
        }
    )
    colliding = first.model_copy(
        update={
            "element_id": "footnote-3",
            "order": 2,
            "structure": first.structure.model_copy(
                update={
                    "footnote": FootnoteStructure(
                        label="1-p1",
                        reference_element_ids=["reference-3"],
                        association_confident=True,
                    )
                }
            ),
        }
    )

    with pytest.raises(ValueError, match="generated footnote labels must be unique"):
        render_document([first, second, colliding])


def test_renderer_escapes_source_markup_but_not_generated_semantic_tokens() -> None:
    heading = _element(
        "<Title> *literal* [link]",
        element_type="heading",
        paragraph=ParagraphStructure(role="heading", heading_level=2),
        runs=[TextStyleRun(start=8, end=17, bold=True)],
    )

    assert render_document([heading]) == ("## &lt;Title&gt; **\\*literal\\*** \\[link\\]")
    literal_reference = _element("source [^literal]")
    assert render_document([literal_reference]) == "source \\[^literal\\]"
    assert heading.content == "<Title> *literal* [link]"


def test_evaluator_compares_rendered_formatting_without_changing_character_error_text() -> None:
    reference = _element("<same> *text*", runs=[TextStyleRun(start=7, end=13, bold=True)])
    candidate = reference.model_copy(
        update={
            "structure": reference.structure.model_copy(update={"style_runs": []}),
            "annotation": reference.annotation.model_copy(update={"stage": "candidate"}),
        }
    )
    reference = reference.model_copy(
        update={"annotation": reference.annotation.model_copy(update={"stage": "silver"})}
    )

    report = evaluate_document([candidate], [reference])
    assert report.metrics.normalized_character_error_rate == 0
    assert report.metrics.canonical_render_accuracy == 0


def test_style_ranges_fail_closed_on_overlap_and_content_overflow() -> None:
    with pytest.raises(ValidationError, match="sorted and non-overlapping"):
        _element(
            "abcdef",
            runs=[TextStyleRun(start=0, end=4, bold=True), TextStyleRun(start=3, end=6, italic=True)],
        )
    with pytest.raises(ValidationError, match="content length"):
        _element("abc", runs=[TextStyleRun(start=0, end=4, bold=True)])


def test_schema_rejects_cross_field_vocabulary_and_format_mismatches() -> None:
    base = _element("plain")
    invalid_payloads = [
        base.model_dump(mode="json") | {"element_type": "unknown"},
        base.model_dump(mode="json") | {"format": "markdown"},
        base.model_dump(mode="json") | {"structure": ElementStructure().model_dump(mode="json")},
        base.model_dump(mode="json")
        | {
            "structure": ElementStructure(paragraph=ParagraphStructure(role="label")).model_dump(mode="json"),
        },
        base.model_dump(mode="json")
        | {
            "element_type": "heading",
            "structure": ElementStructure(
                paragraph=ParagraphStructure(role="body", heading_level=1)
            ).model_dump(mode="json"),
        },
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            DocumentElement.model_validate(payload)

    with pytest.raises(ValidationError, match="list_item requires"):
        ParagraphStructure(role="list_item")
    with pytest.raises(ValidationError, match="title evidence"):
        ParagraphStructure(role="body", title_evidence=["guess"])


@pytest.mark.parametrize(
    "payload",
    [
        {"label": None, "reference_element_ids": ["reference"], "association_confident": True},
        {"label": "1", "reference_element_ids": [], "association_confident": True},
        {"label": "1", "reference_element_ids": ["reference"], "association_confident": False},
        {"label": "1", "reference_element_ids": [""], "association_confident": True},
        {"label": "1", "reference_element_ids": ["reference", "reference"], "association_confident": True},
    ],
)
def test_footnote_structure_rejects_inconsistent_associations(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FootnoteStructure.model_validate(payload)


@pytest.mark.parametrize(
    "label",
    ["", " ", "bad label", "bad]label", "bad\nlabel", "^1", "\\", "1\\", "a\\b", "\\escaped"],
)
def test_generated_footnote_labels_reject_markdown_breaking_and_control_syntax(label: str) -> None:
    with pytest.raises(ValidationError, match="label"):
        FootnoteStructure(label=label, reference_element_ids=[], association_confident=False)
    with pytest.raises(ValidationError, match="label"):
        InlineFootnoteMarkerRange(start=0, end=1, label=label)


def test_inline_marker_json_rejects_escaped_backslashes_and_preserves_safe_unicode() -> None:
    payload = _element("Word [^1\\]").model_dump(mode="json")
    payload["structure"]["properties"] = [
        {
            "key": "inline_footnote_markers",
            "value": json.dumps([{"start": 5, "end": 10, "label": "1\\"}]),
        }
    ]
    with pytest.raises(ValidationError, match="label"):
        DocumentElement.model_validate(payload)

    unicode_payload = _element("Word [^脚注]").model_dump(mode="json")
    unicode_payload["structure"]["properties"] = [
        {
            "key": "inline_footnote_markers",
            "value": json.dumps(
                [{"start": 5, "end": 10, "label": "脚注"}],
                ensure_ascii=False,
            ),
        }
    ]
    unicode_element = DocumentElement.model_validate(unicode_payload)
    assert inline_footnote_marker_ranges(unicode_element) == (
        InlineFootnoteMarkerRange(start=5, end=10, label="脚注"),
    )


def test_legacy_footnote_association_is_migrated_to_safe_unassociated_state(tmp_path: Path) -> None:
    current = _footnote_element(
        FootnoteStructure(label="1", reference_element_ids=[], association_confident=False)
    )
    legacy_record = current.model_dump(mode="json")
    legacy_record["schema_version"] = "1.1.0"
    legacy_record["structure"]["footnote"] = {
        "label": "unsafe label]",
        "reference_element_ids": ["reference", "reference", ""],
        "association_confident": True,
    }
    legacy_path = tmp_path / "legacy-unsafe-footnote.parquet"
    pq.write_table(pa.Table.from_pylist([legacy_record], schema=PREVIOUS_DOCUMENT_SCHEMA), legacy_path)

    migrated = read_document_elements(legacy_path)[0]

    assert migrated.structure.footnote == FootnoteStructure(
        label=None,
        reference_element_ids=[],
        association_confident=False,
    )


def test_style_roundtrip_is_lossless_and_legacy_parquet_gets_optional_defaults(tmp_path: Path) -> None:
    current = _element(
        "styled", runs=[TextStyleRun(start=0, end=6, bold=True, font_family="Body", font_size=10)]
    )
    current_path = tmp_path / "current.parquet"
    write_document_elements([current], current_path)
    assert read_document_elements(current_path) == [current]

    legacy_record = current.model_dump(mode="json")
    legacy_record["schema_version"] = "1.0.0"
    legacy_record["structure"].pop("style_runs")
    for key in ("title_evidence", "heading_evidence", "list_evidence"):
        legacy_record["structure"]["paragraph"].pop(key)
    legacy_path = tmp_path / "legacy.parquet"
    pq.write_table(pa.Table.from_pylist([legacy_record], schema=LEGACY_DOCUMENT_SCHEMA), legacy_path)

    migrated = read_document_elements(legacy_path)[0]
    assert migrated.schema_version == "1.2.0"
    assert migrated.content == current.content
    assert migrated.structure.style_runs == []
    assert migrated.structure.paragraph is not None
    assert migrated.structure.paragraph.heading_evidence == []

    previous_record = current.model_dump(mode="json")
    previous_record["schema_version"] = "1.1.0"
    previous_path = tmp_path / "previous.parquet"
    pq.write_table(pa.Table.from_pylist([previous_record], schema=PREVIOUS_DOCUMENT_SCHEMA), previous_path)
    assert read_document_elements(previous_path)[0] == current

    deprecated_record = current.model_dump(mode="json")
    deprecated_record["schema_version"] = "1.1.0"
    deprecated_record["structure"]["paragraph"]["role"] = "label"
    deprecated_path = tmp_path / "deprecated-role.parquet"
    pq.write_table(
        pa.Table.from_pylist([deprecated_record], schema=PREVIOUS_DOCUMENT_SCHEMA), deprecated_path
    )
    deprecated = read_document_elements(deprecated_path)[0]
    assert deprecated.structure.paragraph is not None
    assert deprecated.structure.paragraph.role == "body"
