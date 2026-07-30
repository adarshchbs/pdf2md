import pytest

from app.pdf2md.evaluation import (
    CandidateChurnReport,
    align_elements,
    evaluate_candidate_churn,
    evaluate_document,
    evaluate_reference_churn,
)
from app.pdf2md.schema import (
    AnnotationMetadata,
    BoundingBox,
    DocumentElement,
    ElementStructure,
    FigureStructure,
    FootnoteStructure,
    PageFragment,
    ParagraphStructure,
    StructureProperty,
    TableCell,
    TableRepresentation,
    TableStructure,
)
from app.pdf2md.tables import render_table


def paragraph(
    element_id: str,
    content: str,
    order: int,
    y0: float,
) -> DocumentElement:
    return DocumentElement(
        document_id="doc",
        element_id=element_id,
        order=order,
        element_type="paragraph",
        content=content,
        format="text",
        fragments=[
            PageFragment(
                page_number=1,
                page_width=100,
                page_height=100,
                bbox=BoundingBox(x0=0, y0=y0, x1=100, y1=y0 + 10),
            )
        ],
        structure=ElementStructure(paragraph=ParagraphStructure(role="body")),
        annotation=AnnotationMetadata(
            stage="silver",
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        ),
    )


def table(element_id: str, representation: TableRepresentation) -> DocumentElement:
    cells = [
        TableCell(
            row_index=row,
            column_index=column,
            role="header" if row == 0 else "body",
            text=(("Name", "Value"), ("North", "10"))[row][column],
            fragments=[
                PageFragment(
                    page_number=1,
                    page_width=100,
                    page_height=100,
                    bbox=BoundingBox(
                        x0=column * 50,
                        y0=row * 10,
                        x1=(column + 1) * 50,
                        y1=(row + 1) * 10,
                    ),
                )
            ],
        )
        for row in range(2)
        for column in range(2)
    ]
    structure = TableStructure(
        row_count=2,
        column_count=2,
        header_row_count=1,
        representation=representation,
        classification_reasons=[] if representation == "markdown" else ["nested_content"],
        cells=cells,
    )
    return DocumentElement(
        document_id="doc",
        element_id=element_id,
        order=0,
        element_type="table",
        content=render_table(structure),
        format=representation,
        fragments=[
            PageFragment(
                page_number=1,
                page_width=100,
                page_height=100,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20),
            )
        ],
        structure=ElementStructure(table=structure),
        annotation=AnnotationMetadata(
            stage="silver",
            revision=1,
            annotator="test",
            confidence=1,
            adjudication_status="unreviewed",
        ),
    )


def figure(
    element_id: str,
    page_number: int,
    content: str = "",
    *,
    bbox: tuple[float, float, float, float] = (0, 0, 100, 10),
    caption_element_id: str | None = None,
    asset_path: str | None = None,
) -> DocumentElement:
    element = paragraph(element_id, content, 0, 0)
    return element.model_copy(
        update={
            "element_type": "figure",
            "fragments": [
                element.fragments[0].model_copy(
                    update={
                        "page_number": page_number,
                        "bbox": BoundingBox(
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3],
                        ),
                    }
                )
            ],
            "structure": ElementStructure(
                figure=FigureStructure(
                    asset_path=asset_path or f"pdf://page/{page_number}/figure/{element_id}",
                    sha256="0" * 64,
                    width=100,
                    height=100,
                    caption_element_id=caption_element_id,
                ),
                linked_element_ids=[caption_element_id] if caption_element_id else [],
            ),
        }
    )


def caption(element_id: str, content: str, order: int, y0: float = 80) -> DocumentElement:
    element = paragraph(element_id, content, order, y0)
    return element.model_copy(
        update={
            "element_type": "caption",
            "structure": ElementStructure(paragraph=ParagraphStructure(role="caption")),
        }
    )


def footnote(
    element_id: str,
    content: str,
    order: int,
    label: str,
    reference_element_ids: list[str],
) -> DocumentElement:
    element = paragraph(element_id, content, order, 70)
    return element.model_copy(
        update={
            "element_type": "footnote",
            "structure": ElementStructure(
                footnote=FootnoteStructure(
                    label=label,
                    reference_element_ids=reference_element_ids,
                    association_confident=bool(reference_element_ids),
                )
            ),
        }
    )


def as_candidate(elements: list[DocumentElement]) -> list[DocumentElement]:
    return [
        element.model_copy(
            update={
                "annotation": element.annotation.model_copy(
                    update={"stage": "candidate", "annotator": "candidate"}
                )
            }
        )
        for element in elements
    ]


def test_perfect_candidate_scores_one() -> None:
    reference = [
        paragraph("one", "First paragraph", 0, 0),
        paragraph("two", "Second paragraph", 1, 20),
    ]

    report = evaluate_document(as_candidate(reference), reference)

    assert report.metrics.element_f1 == 1
    assert report.metrics.reading_order_accuracy == 1
    assert report.metrics.normalized_character_error_rate == 0
    assert report.metrics.canonical_render_accuracy == 1


def test_footnote_metrics_compare_semantics_independently_of_canonical_markdown_syntax() -> None:
    reference = [
        paragraph("reference-first", "First claim1.", 0, 0),
        paragraph("reference-second", "Second claim1.", 1, 20),
        footnote(
            "reference-note",
            "Supporting body.",
            2,
            "1",
            ["reference-first", "reference-second"],
        ),
    ]
    candidate = as_candidate([
        paragraph("candidate-first", "First claim[^1].", 0, 0),
        paragraph("candidate-second", "Second claim[^1].", 1, 20),
        footnote(
            "candidate-note",
            "1 Supporting body.",
            2,
            "1",
            ["candidate-first", "candidate-second"],
        ),
    ])

    report = evaluate_document(candidate, reference)

    assert report.metrics.candidate_reference_content_identity_rate == 1
    assert report.metrics.canonical_render_accuracy == 1
    assert report.metrics.normalized_character_error_rate == 0
    assert report.metrics.footnote_detection_precision == 1
    assert report.metrics.footnote_detection_recall == 1


def test_footnote_semantic_projection_penalizes_adversarial_link_and_content_errors() -> None:
    reference = [
        paragraph("reference-first", "First claim1.", 0, 0),
        paragraph("reference-second", "Second claim1.", 1, 20),
        footnote(
            "reference-note",
            "Supporting body.",
            2,
            "1",
            ["reference-first", "reference-second"],
        ),
    ]
    first = paragraph("candidate-first", "First claim[^1].", 0, 0)
    second = paragraph("candidate-second", "Second claim[^1].", 1, 20)

    adversarial_candidates = [
        [
            first.model_copy(update={"content": "First claim[^2]."}),
            second.model_copy(update={"content": "Second claim[^2]."}),
            footnote(
                "candidate-note",
                "2 Supporting body.",
                2,
                "2",
                ["candidate-first", "candidate-second"],
            ),
        ],
        [
            first.model_copy(update={"content": "First claim."}),
            second,
            footnote(
                "candidate-note",
                "1 Supporting body.",
                2,
                "1",
                ["candidate-second"],
            ),
        ],
        [
            first,
            second,
            footnote(
                "candidate-note",
                "1 Supporting body.",
                2,
                "1",
                ["candidate-first", "candidate-second", "candidate-first"],
            ),
        ],
        [
            first,
            second,
            footnote(
                "candidate-note",
                "1 Wrong body.",
                2,
                "1",
                ["candidate-first", "candidate-second"],
            ),
        ],
        [
            first,
            second,
            footnote(
                "candidate-note",
                "1 Supporting body.",
                2,
                "1",
                ["candidate-second", "candidate-first"],
            ),
        ],
    ]

    for candidate in adversarial_candidates:
        report = evaluate_document(as_candidate(candidate), reference)
        assert report.metrics.candidate_reference_content_identity_rate < 1
        assert report.metrics.canonical_render_accuracy < 1
        assert report.metrics.normalized_character_error_rate > 0


def test_footnote_semantic_projection_penalizes_an_extra_distinct_reference() -> None:
    reference = [
        paragraph("reference-first", "First claim1.", 0, 0),
        paragraph("reference-unlinked", "Unlinked text.", 1, 20),
        footnote("reference-note", "Supporting body.", 2, "1", ["reference-first"]),
    ]
    candidate = as_candidate([
        paragraph("candidate-first", "First claim[^1].", 0, 0),
        paragraph("candidate-unlinked", "Unlinked text.", 1, 20),
        footnote(
            "candidate-note",
            "1 Supporting body.",
            2,
            "1",
            ["candidate-first", "candidate-unlinked"],
        ),
    ])

    report = evaluate_document(candidate, reference)

    assert report.metrics.candidate_reference_content_identity_rate < 1
    assert report.metrics.canonical_render_accuracy < 1
    assert report.metrics.normalized_character_error_rate > 0


def test_reading_order_uses_content_alignment_not_list_position() -> None:
    first = paragraph("one", "First paragraph", 0, 0)
    second = paragraph("two", "Second paragraph", 1, 20)
    candidate = [
        second.model_copy(update={"order": 0}),
        first.model_copy(update={"order": 1}),
    ]

    report = evaluate_document(as_candidate(candidate), [first, second])

    assert report.metrics.element_recall == 1
    assert report.metrics.reading_order_accuracy == 0


def test_missing_element_reduces_recall_and_counts_text_omission() -> None:
    first = paragraph("one", "First paragraph", 0, 0)
    second = paragraph("two", "Second paragraph", 1, 20)

    report = evaluate_document(as_candidate([first]), [first, second])

    assert report.metrics.element_precision == 1
    assert report.metrics.element_recall == 0.5
    assert report.metrics.normalized_character_error_rate > 0
    assert report.unmatched_reference_indices == [1]


def test_wrong_text_at_same_geometry_does_not_align() -> None:
    reference = [paragraph("reference", "Substantive reference content", 0, 0)]
    candidate = as_candidate([paragraph("candidate", ".", 0, 0)])

    report = evaluate_document(candidate, reference)

    assert report.metrics.element_f1 == 0
    assert report.alignments == []
    assert report.metrics.reading_order_accuracy is None


def test_fragmentation_is_reported() -> None:
    reference = [paragraph("reference", "First sentence. Second sentence.", 0, 0)]
    candidate = as_candidate([
        paragraph("first", "First sentence.", 0, 0),
        paragraph("second", "Second sentence.", 1, 5),
    ])

    report = evaluate_document(candidate, reference)

    assert report.metrics.fragmentation_rate == 1


def test_content_fragmentation_detects_moved_reordered_numeric_labels() -> None:
    reference = [
        paragraph("labels", "10 20 30 40", 0, 0),
        paragraph("noninformative", "...", 1, 90),
    ]
    candidate = as_candidate([
        paragraph("forty", "40", 0, 80),
        paragraph("ten", "10", 1, 20),
        paragraph("thirty", "30", 2, 60),
        paragraph("twenty", "20", 3, 40),
    ])

    report = evaluate_document(candidate, reference)

    assert report.metrics.fragmentation_rate == 0
    assert report.metrics.content_fragmentation_rate == 1
    assert report.metrics.content_merge_rate == 0


def test_content_merge_detects_reordered_reference_fragments() -> None:
    reference = [
        paragraph("beta", "Beta", 0, 20),
        paragraph("alpha", "Alpha", 1, 0),
        paragraph("gamma", "Gamma", 2, 40),
    ]
    candidate = as_candidate([paragraph("merged", "Alpha Beta Gamma", 0, 80)])

    report = evaluate_document(candidate, reference)

    assert report.metrics.content_fragmentation_rate == 0
    assert report.metrics.content_merge_rate == pytest.approx(1 / 3)


def test_content_relations_do_not_reward_duplicates_or_cross_page_matches() -> None:
    reference = [paragraph("labels", "10 20 30", 0, 0)]
    duplicate = paragraph("duplicate", "10", 1, 20)
    cross_page = paragraph("cross-page", "20 30", 2, 40)
    cross_page = cross_page.model_copy(
        update={"fragments": [cross_page.fragments[0].model_copy(update={"page_number": 2})]}
    )
    candidate = as_candidate([
        paragraph("ten", "10", 0, 10),
        duplicate,
        cross_page,
        paragraph("empty", "...", 3, 60),
    ])

    report = evaluate_document(candidate, reference)

    assert report.metrics.content_fragmentation_rate == 0


def test_content_relations_exclude_table_mismatches_and_one_to_one_elements() -> None:
    target = paragraph("target", "Name Value North 10", 0, 0)
    first_table = table("first-table", "html").model_copy(update={"content": "Name Value"})
    second_table = table("second-table", "html").model_copy(update={"content": "North 10"})
    exact = paragraph("exact", "Stable paragraph", 1, 20)

    mismatched_report = evaluate_document(as_candidate([first_table, second_table]), [target])
    exact_report = evaluate_document(as_candidate([exact]), [exact])

    assert mismatched_report.metrics.content_fragmentation_rate == 0
    assert exact_report.metrics.content_fragmentation_rate == 0
    assert exact_report.metrics.content_merge_rate == 0


def test_content_relations_do_not_count_split_duplicates_beside_exact_element() -> None:
    target = paragraph("target", "Alpha Beta", 0, 0)
    candidate = as_candidate([
        paragraph("exact", "Alpha Beta", 0, 0),
        paragraph("alpha", "Alpha", 1, 20),
        paragraph("beta", "Beta", 2, 40),
    ])

    report = evaluate_document(candidate, [target])

    assert report.metrics.content_fragmentation_rate == 0


def test_content_relations_do_not_combine_overlapping_near_duplicates() -> None:
    target = paragraph("target", "abcdefghij", 0, 0)
    candidate = as_candidate([
        paragraph("left", "abcdefgh", 0, 20),
        paragraph("right", "bcdefghij", 1, 40),
    ])

    report = evaluate_document(candidate, [target])

    assert report.metrics.content_fragmentation_rate == 0


def test_content_relations_find_disjoint_components_despite_overlapping_distractor() -> None:
    target = paragraph("target", "abcdefghij", 0, 0)
    candidate = as_candidate([
        paragraph("left", "abcde", 0, 20),
        paragraph("overlap", "cdefgh", 1, 40),
        paragraph("right", "fghij", 2, 60),
    ])

    report = evaluate_document(candidate, [target])

    assert report.metrics.content_fragmentation_rate == 1


def test_merge_rates_use_reference_denominator_not_candidate_granularity() -> None:
    reference = [
        paragraph("alpha", "Alpha", 0, 0),
        paragraph("beta", "Beta", 1, 5),
        paragraph("stable", "Stable", 2, 40),
    ]
    merged = paragraph("merged", "Alpha Beta", 0, 0).model_copy(
        update={
            "fragments": [
                paragraph("geometry", "", 0, 0)
                .fragments[0]
                .model_copy(update={"bbox": BoundingBox(x0=0, y0=0, x1=100, y1=15)})
            ]
        }
    )
    compact = as_candidate([merged, paragraph("stable-candidate", "Stable", 1, 40)])
    padded = as_candidate([
        *compact,
        paragraph("noise-one", "Unmatched noise one", 2, 60),
        paragraph("noise-two", "Unmatched noise two", 3, 80),
    ])

    compact_report = evaluate_document(compact, reference)
    padded_report = evaluate_document(padded, reference)

    assert compact_report.metrics.content_merge_rate == pytest.approx(1 / 3)
    assert padded_report.metrics.content_merge_rate == compact_report.metrics.content_merge_rate
    assert compact_report.metrics.merge_rate == pytest.approx(1 / 3)
    assert padded_report.metrics.merge_rate == compact_report.metrics.merge_rate


def test_table_detection_alignment_is_independent_of_rendering_format() -> None:
    reference = [table("reference", "html")]
    candidate = as_candidate([table("candidate", "markdown")])

    report = evaluate_document(candidate, reference)

    assert report.metrics.table_detection_precision == 1
    assert report.metrics.table_detection_recall == 1
    assert report.metrics.table_grid_topology_accuracy == 1
    assert report.metrics.table_representation_accuracy == 0


def test_representation_accuracy_uses_strict_reference_eligibility() -> None:
    fragment = PageFragment(
        page_number=1,
        page_width=100,
        page_height=100,
        bbox=BoundingBox(x0=0, y0=0, x1=100, y1=30),
    )
    values = [
        ["Item", "Value", "Unit"],
        ["Operating activities", "", ""],
        ["Cash", "10", "USD"],
    ]
    cells = [
        TableCell(
            row_index=row,
            column_index=column,
            role="header" if row == 0 else "row_header" if column == 0 else "body",
            text=value,
            fragments=[fragment],
        )
        for row, values_by_column in enumerate(values)
        for column, value in enumerate(values_by_column)
    ]
    stale_reference_structure = TableStructure(
        row_count=3,
        column_count=3,
        header_row_count=1,
        representation="markdown",
        cells=cells,
    )
    strict_candidate_structure = stale_reference_structure.model_copy(
        update={"representation": "html", "classification_reasons": ["nested_content"]}
    )
    reference = DocumentElement(
        document_id="doc",
        element_id="reference",
        order=0,
        element_type="table",
        content=(
            "| Item | Value | Unit |\n| --- | --- | --- |\n"
            "| Operating activities |  |  |\n| Cash | 10 | USD |"
        ),
        format="markdown",
        fragments=[fragment],
        structure=ElementStructure(table=stale_reference_structure),
        annotation=paragraph("metadata", "x", 0, 0).annotation,
    )
    candidate = reference.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(strict_candidate_structure),
            "format": "html",
            "structure": ElementStructure(table=strict_candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate]), [reference])

    assert report.metrics.table_representation_accuracy == 1
    assert report.metrics.canonical_render_accuracy == 1
    assert report.metrics.normalized_character_error_rate == 0


def test_representation_accuracy_penalizes_candidate_eligibility_defects() -> None:
    reference = table("reference", "markdown")
    reference_structure = reference.structure.table
    assert reference_structure is not None
    extra_cells = [
        reference_structure.cells[row * 2].model_copy(
            update={
                "column_index": 2,
                "role": "header" if row == 0 else "body",
                "text": "Unit" if row == 0 else "USD",
            }
        )
        for row in range(2)
    ]
    reference_structure = reference_structure.model_copy(
        update={"column_count": 3, "cells": [*reference_structure.cells, *extra_cells]}
    )
    reference = reference.model_copy(
        update={
            "content": render_table(reference_structure),
            "structure": ElementStructure(table=reference_structure),
        }
    )
    defective_cells = [
        cell.model_copy(update={"text": ""}) if cell.row_index == 1 and cell.column_index in {1, 2} else cell
        for cell in reference_structure.cells
    ]
    stale_candidate_structure = reference_structure.model_copy(update={"cells": defective_cells})
    candidate = reference.model_copy(
        update={
            "element_id": "candidate",
            "structure": ElementStructure(table=stale_candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate]), [reference])

    assert report.metrics.table_representation_accuracy == 0
    assert report.metrics.canonical_render_accuracy == 0
    assert report.metrics.normalized_character_error_rate > 0


def test_character_error_rate_is_bounded() -> None:
    reference = [paragraph("reference", "short reference", 0, 0)]
    candidate = as_candidate([
        paragraph(f"candidate-{index}", "unrelated candidate insertion", index, index * 10)
        for index in range(10)
    ])

    report = evaluate_document(candidate, reference)

    assert 0 <= report.metrics.normalized_character_error_rate <= 1


def test_character_error_rate_ignores_table_renderer_only_markup() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    candidate_structure = reference_structure.model_copy(
        update={"classification_reasons": [*reference_structure.classification_reasons, "nested_content"]}
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": reference_table.content.replace("<th", '<th scope="col"'),
            "structure": ElementStructure(table=candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.normalized_character_error_rate == 0
    assert report.metrics.candidate_reference_content_identity_rate == 0
    assert report.metrics.canonical_render_accuracy == 1
    assert report.metrics.table_cell_content_f1 == 1
    assert report.metrics.table_topology_accuracy == 1


def test_character_error_rate_still_penalizes_changed_table_cell_text() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    candidate_structure = reference_structure.model_copy(
        update={
            "cells": [
                cell.model_copy(update={"text": "11"}) if cell.text == "10" else cell
                for cell in reference_structure.cells
            ]
        }
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(candidate_structure),
            "structure": ElementStructure(table=candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.normalized_character_error_rate > 0
    assert report.metrics.table_cell_content_f1 is not None
    assert report.metrics.table_cell_content_f1 < 1
    assert report.metrics.table_cell_content_assignment_accuracy is not None
    assert report.metrics.table_cell_content_assignment_accuracy < 1


def test_character_error_rate_ignores_table_topology_but_topology_metrics_do_not() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    flattened_structure = reference_structure.model_copy(
        update={
            "row_count": 1,
            "column_count": 4,
            "header_row_count": 1,
            "classification_reasons": ["spanning_cells"],
            "cells": [
                cell.model_copy(update={"row_index": 0, "column_index": index, "role": "header"})
                for index, cell in enumerate(reference_structure.cells)
            ],
        }
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(flattened_structure),
            "structure": ElementStructure(table=flattened_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.normalized_character_error_rate == 0
    assert report.metrics.table_topology_accuracy == 0
    assert report.metrics.table_grid_topology_accuracy == 0


def test_character_error_rate_ignores_spans_roles_and_table_metadata() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    changed_cells = [
        reference_structure.cells[0].model_copy(update={"colspan": 2}),
        reference_structure.cells[1].model_copy(update={"column_index": 2}),
        reference_structure.cells[2].model_copy(update={"role": "row_header"}),
        reference_structure.cells[3],
    ]
    changed_structure = reference_structure.model_copy(
        update={
            "column_count": 3,
            "classification_reasons": ["spanning_cells", "accessibility_headers"],
            "cells": changed_cells,
        }
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": "<table aria-describedby='renderer-only'>ignored</table>",
            "structure": ElementStructure(table=changed_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.normalized_character_error_rate == 0
    assert report.metrics.table_topology_accuracy == 0
    assert report.metrics.table_grid_topology_accuracy == 0
    assert report.metrics.table_cell_content_f1 == 1


def test_character_error_rate_tracks_table_source_order_not_cell_list_order() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    permuted_structure = reference_structure.model_copy(update={"cells": reference_structure.cells[::-1]})
    permuted_table = reference_table.model_copy(
        update={
            "element_id": "permuted",
            "content": "renderer output is intentionally irrelevant",
            "structure": ElementStructure(table=permuted_structure),
        }
    )
    swapped_structure = reference_structure.model_copy(
        update={
            "cells": [
                cell.model_copy(update={"text": reference_structure.cells[3 - index].text})
                for index, cell in enumerate(reference_structure.cells)
            ]
        }
    )
    swapped_table = reference_table.model_copy(
        update={
            "element_id": "swapped",
            "content": render_table(swapped_structure),
            "structure": ElementStructure(table=swapped_structure),
        }
    )

    permuted = evaluate_document(as_candidate([permuted_table]), [reference_table])
    swapped = evaluate_document(as_candidate([swapped_table]), [reference_table])

    assert permuted.metrics.normalized_character_error_rate == 0
    assert swapped.metrics.normalized_character_error_rate > 0
    assert swapped.metrics.table_grid_topology_accuracy == 1
    assert swapped.metrics.table_cell_content_f1 == 1
    assert swapped.metrics.table_cell_content_assignment_accuracy == 0


def test_character_error_rate_preserves_empty_and_duplicate_cell_multiplicity() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    cells = [
        cell.model_copy(update={"text": text})
        for cell, text in zip(reference_structure.cells, ("Same", "", "Same", "Café\nline"), strict=True)
    ]
    reference_structure = reference_structure.model_copy(update={"cells": cells})
    reference_table = reference_table.model_copy(
        update={
            "content": render_table(reference_structure),
            "structure": ElementStructure(table=reference_structure),
        }
    )
    normalized_cells = [
        cell.model_copy(update={"text": "Café  line" if index == 3 else cell.text})
        for index, cell in enumerate(cells)
    ]
    normalized_structure = reference_structure.model_copy(
        update={"classification_reasons": ["nested_content"], "cells": normalized_cells}
    )
    normalized_table = reference_table.model_copy(
        update={
            "element_id": "normalized",
            "content": "<table aria-label='ignored'><tr><td>ignored</td></tr></table>",
            "structure": ElementStructure(table=normalized_structure),
        }
    )
    missing_empty_structure = reference_structure.model_copy(update={"cells": [cells[0], *cells[2:]]})
    missing_empty_table = reference_table.model_copy(
        update={
            "element_id": "missing-empty",
            "content": render_table(missing_empty_structure),
            "structure": ElementStructure(table=missing_empty_structure),
        }
    )
    missing_duplicate_structure = reference_structure.model_copy(update={"cells": cells[1:]})
    missing_duplicate_table = reference_table.model_copy(
        update={
            "element_id": "missing-duplicate",
            "content": render_table(missing_duplicate_structure),
            "structure": ElementStructure(table=missing_duplicate_structure),
        }
    )

    normalized = evaluate_document(as_candidate([normalized_table]), [reference_table])
    missing_empty = evaluate_document(as_candidate([missing_empty_table]), [reference_table])
    missing_duplicate = evaluate_document(as_candidate([missing_duplicate_table]), [reference_table])

    assert normalized.metrics.normalized_character_error_rate == 0
    assert missing_empty.metrics.normalized_character_error_rate > 0
    assert missing_duplicate.metrics.normalized_character_error_rate > 0


def test_character_error_rate_penalizes_repeated_empty_cell_changes() -> None:
    base_table = table("base", "html")
    base_structure = base_table.structure.table
    assert base_structure is not None

    def empty_table(element_id: str, cell_count: int) -> DocumentElement:
        cells = [
            base_structure.cells[index].model_copy(
                update={"row_index": 0, "column_index": index, "role": "header", "text": ""}
            )
            for index in range(cell_count)
        ]
        structure = base_structure.model_copy(
            update={
                "row_count": 1,
                "column_count": max(cell_count, 1),
                "header_row_count": 1,
                "cells": cells,
            }
        )
        return base_table.model_copy(
            update={
                "element_id": element_id,
                "content": "renderer output is intentionally irrelevant",
                "structure": ElementStructure(table=structure),
            }
        )

    one_empty = empty_table("one", 1)
    two_empty = empty_table("two", 2)

    missing_repeated_empty = evaluate_document(as_candidate([one_empty]), [two_empty])
    extra_repeated_empty = evaluate_document(as_candidate([two_empty]), [one_empty])

    assert missing_repeated_empty.metrics.normalized_character_error_rate == pytest.approx(1 / 2)
    assert extra_repeated_empty.metrics.normalized_character_error_rate == pytest.approx(1 / 2)


def test_character_error_rate_counts_unmatched_table_source_text() -> None:
    body = paragraph("body", "Body", 0, 40)
    extra_table = table("extra", "html").model_copy(update={"order": 1})

    extra_report = evaluate_document(as_candidate([body, extra_table]), [body])
    missing_report = evaluate_document(as_candidate([body]), [body, extra_table])

    assert extra_report.metrics.normalized_character_error_rate > 0
    assert extra_report.metrics.table_detection_precision == 0
    assert missing_report.metrics.normalized_character_error_rate > 0
    assert missing_report.metrics.table_detection_recall == 0


def test_non_table_character_error_projection_and_mixed_denominator_are_stable() -> None:
    normalized_reference = paragraph("normalized-reference", "A B", 0, 0)
    normalized_candidate = paragraph("normalized-candidate", "Ａ\nB", 0, 0)
    normalization_report = evaluate_document(as_candidate([normalized_candidate]), [normalized_reference])
    changed_report = evaluate_document(
        as_candidate([paragraph("changed-candidate", "AX", 0, 0)]),
        [paragraph("changed-reference", "AB", 0, 0)],
    )

    reference_table = table("reference-table", "html").model_copy(update={"order": 1})
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    reference_structure = reference_structure.model_copy(
        update={
            "cells": [
                reference_structure.cells[0].model_copy(update={"text": "C"}),
                reference_structure.cells[1].model_copy(update={"text": "D"}),
            ],
            "row_count": 1,
            "column_count": 2,
            "header_row_count": 1,
        }
    )
    reference_table = reference_table.model_copy(
        update={
            "content": render_table(reference_structure),
            "structure": ElementStructure(table=reference_structure),
        }
    )
    candidate_structure = reference_structure.model_copy(
        update={
            "cells": [
                reference_structure.cells[0],
                reference_structure.cells[1].model_copy(update={"text": "E"}),
            ]
        }
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate-table",
            "content": render_table(candidate_structure),
            "structure": ElementStructure(table=candidate_structure),
        }
    )
    reference = [paragraph("reference-body", "AB", 0, 0), reference_table]
    candidate = as_candidate([paragraph("candidate-body", "AX", 0, 0), candidate_table])

    mixed_report = evaluate_document(candidate, reference)

    assert normalization_report.metrics.normalized_character_error_rate == 0
    assert changed_report.metrics.normalized_character_error_rate == pytest.approx(1 / 2)
    assert mixed_report.metrics.normalized_character_error_rate == pytest.approx(2 / 7)


def test_semantic_role_and_detection_metrics_are_reported() -> None:
    heading = paragraph("heading", "Section heading", 0, 0).model_copy(
        update={
            "element_type": "heading",
            "structure": ElementStructure(paragraph=ParagraphStructure(role="heading", heading_level=2)),
        }
    )
    caption = paragraph("caption", "Figure 1. Result", 1, 20).model_copy(
        update={
            "element_type": "caption",
            "structure": ElementStructure(paragraph=ParagraphStructure(role="caption")),
        }
    )
    candidate_caption = caption.model_copy(
        update={
            "element_type": "paragraph",
            "structure": ElementStructure(paragraph=ParagraphStructure(role="body")),
        }
    )

    report = evaluate_document(as_candidate([heading, candidate_caption]), [heading, caption])

    assert report.metrics.semantic_role_accuracy == 0.5
    assert report.metrics.semantic_role_conditional_accuracy == 0.5
    assert report.metrics.heading_detection_precision == 1
    assert report.metrics.heading_detection_recall == 1
    assert report.metrics.caption_detection_precision is None
    assert report.metrics.caption_detection_recall == 0


def test_semantic_role_accuracy_counts_missing_reference_roles_as_wrong() -> None:
    body = paragraph("body", "Body text", 0, 0)
    missing_heading = paragraph("heading", "Missing heading", 1, 20).model_copy(
        update={
            "element_type": "heading",
            "structure": ElementStructure(paragraph=ParagraphStructure(role="heading", heading_level=2)),
        }
    )

    report = evaluate_document(as_candidate([body]), [body, missing_heading])

    assert report.metrics.semantic_role_accuracy == 0.5
    assert report.metrics.semantic_role_conditional_accuracy == 1


def test_blank_figure_aligns_from_same_page_geometry_and_figure_reference() -> None:
    reference = [
        figure(
            "reference-figure",
            5,
            content="Figure 4: manually described chart",
            bbox=(10, 20, 90, 70),
            asset_path="data/bronze/page-0005.png",
        )
    ]
    candidate = as_candidate([
        figure("candidate-figure", 5, bbox=(11, 19, 89, 71)),
    ])

    report = evaluate_document(candidate, reference)

    assert [(alignment.candidate_index, alignment.reference_index) for alignment in report.alignments] == [
        (0, 0)
    ]
    assert report.metrics.figure_detection_precision == 1
    assert report.metrics.figure_detection_recall == 1


def test_nearby_blank_figures_align_by_geometry_without_cross_pairing() -> None:
    left = figure("reference-left", 1, bbox=(5, 10, 45, 60))
    right = figure("reference-right", 1, bbox=(55, 10, 95, 60))
    candidate = as_candidate([
        figure("candidate-right", 1, bbox=(57, 11, 94, 59)),
        figure("candidate-left", 1, bbox=(6, 9, 43, 61)),
    ])

    report = evaluate_document(candidate, [left, right])
    pairs = {
        (candidate[alignment.candidate_index].element_id, [left, right][alignment.reference_index].element_id)
        for alignment in report.alignments
    }

    assert pairs == {
        ("candidate-left", "reference-left"),
        ("candidate-right", "reference-right"),
    }


def test_linked_caption_similarity_disambiguates_identical_figure_geometry() -> None:
    reference = [
        figure("reference-alpha", 1, bbox=(10, 10, 90, 70), caption_element_id="alpha-caption"),
        figure("reference-beta", 1, bbox=(10, 10, 90, 70), caption_element_id="beta-caption"),
        caption("alpha-caption", "Figure 1. Alpha result", 2),
        caption("beta-caption", "Figure 2. Beta result", 3, 90),
    ]
    candidate = as_candidate([
        figure("candidate-beta", 1, bbox=(10, 10, 90, 70), caption_element_id="candidate-beta-caption"),
        figure(
            "candidate-alpha",
            1,
            bbox=(10, 10, 90, 70),
            caption_element_id="candidate-alpha-caption",
        ),
        caption("candidate-beta-caption", "Figure 2. Beta result", 2, 90),
        caption("candidate-alpha-caption", "Figure 1. Alpha result", 3),
    ])

    report = evaluate_document(candidate, reference)
    figure_pairs = {
        (candidate[alignment.candidate_index].element_id, reference[alignment.reference_index].element_id)
        for alignment in report.alignments
        if candidate[alignment.candidate_index].element_type == "figure"
    }

    assert figure_pairs == {
        ("candidate-alpha", "reference-alpha"),
        ("candidate-beta", "reference-beta"),
    }


def test_duplicate_blank_figure_is_not_matched_twice() -> None:
    reference = [figure("reference", 1, bbox=(10, 10, 90, 70))]
    candidate = as_candidate([
        figure("candidate-original", 1, bbox=(10, 10, 90, 70)),
        figure("candidate-duplicate", 1, bbox=(10, 10, 90, 70)),
    ])

    report = evaluate_document(candidate, reference)

    assert len(report.alignments) == 1
    assert report.metrics.figure_detection_precision == 0.5
    assert report.metrics.figure_detection_recall == 1


def test_blank_figures_require_geometry_and_reject_page_backgrounds_and_unrelated_empty() -> None:
    reference = [figure("reference", 1, bbox=(10, 10, 90, 70))]
    missing_geometry = figure("missing-geometry", 1, bbox=(10, 10, 90, 70)).model_copy(
        update={"fragments": []}
    )
    page_background = figure("page-background", 1, bbox=(0, 0, 100, 100))
    unrelated_figure = figure("unrelated-figure", 1, bbox=(0, 80, 10, 90))
    unrelated_empty = paragraph("empty", "", 0, 10)

    report = evaluate_document(
        as_candidate([missing_geometry, page_background, unrelated_figure, unrelated_empty]),
        reference,
    )

    assert report.alignments == []
    assert report.metrics.figure_detection_precision == 0
    assert report.metrics.figure_detection_recall == 0


def test_figure_and_table_never_align_even_with_blank_content_and_identical_geometry() -> None:
    reference = [figure("reference-figure", 1, bbox=(0, 0, 100, 20))]
    candidate_table = table("candidate-table", "html").model_copy(update={"content": ""})

    report = evaluate_document(as_candidate([candidate_table]), reference)

    assert report.alignments == []
    assert report.metrics.figure_detection_recall == 0
    assert report.metrics.table_detection_precision == 0


def test_blank_cross_page_figures_and_tables_do_not_match() -> None:
    figure_report = evaluate_document(
        as_candidate([figure("candidate-figure", 2)]),
        [figure("reference-figure", 1)],
    )

    reference_table = table("reference-table", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    blank_structure = reference_structure.model_copy(
        update={"cells": [cell.model_copy(update={"text": ""}) for cell in reference_structure.cells]}
    )
    reference_table = reference_table.model_copy(
        update={"content": "", "structure": ElementStructure(table=blank_structure)}
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate-table",
            "fragments": [reference_table.fragments[0].model_copy(update={"page_number": 2})],
        }
    )
    table_report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert figure_report.alignments == []
    assert figure_report.metrics.figure_detection_precision == 0
    assert figure_report.metrics.figure_detection_recall == 0
    assert table_report.metrics.table_detection_precision == 0
    assert table_report.metrics.table_detection_recall == 0


def test_detection_absent_class_semantics_are_consistent() -> None:
    reference_heading = paragraph("reference-heading", "Reference", 0, 0).model_copy(
        update={
            "element_type": "heading",
            "structure": ElementStructure(paragraph=ParagraphStructure(role="heading", heading_level=1)),
        }
    )
    candidate_heading = paragraph("candidate-heading", "Candidate", 0, 0).model_copy(
        update={
            "element_type": "heading",
            "structure": ElementStructure(paragraph=ParagraphStructure(role="heading", heading_level=1)),
        }
    )
    candidate_absent = evaluate_document(
        as_candidate([paragraph("candidate-body", "Candidate", 0, 0)]),
        [reference_heading, table("reference-table", "html")],
    )
    reference_absent = evaluate_document(
        as_candidate([candidate_heading, table("candidate-table", "html")]),
        [paragraph("reference-body", "Reference", 0, 0)],
    )

    assert candidate_absent.metrics.heading_detection_precision is None
    assert candidate_absent.metrics.heading_detection_recall == 0
    assert candidate_absent.metrics.table_detection_precision is None
    assert candidate_absent.metrics.table_detection_recall == 0
    assert candidate_absent.metrics.table_cell_role_precision is None
    assert candidate_absent.metrics.table_cell_role_recall == 0
    assert candidate_absent.metrics.table_cell_content_precision is None
    assert candidate_absent.metrics.table_cell_content_recall == 0
    assert candidate_absent.metrics.table_cell_content_f1 == 0
    assert candidate_absent.metrics.table_cell_token_f1 == 0
    assert reference_absent.metrics.heading_detection_precision == 0
    assert reference_absent.metrics.heading_detection_recall is None
    assert reference_absent.metrics.table_detection_precision == 0
    assert reference_absent.metrics.table_detection_recall is None
    assert reference_absent.metrics.table_cell_role_precision == 0
    assert reference_absent.metrics.table_cell_role_recall is None
    assert reference_absent.metrics.table_cell_content_precision == 0
    assert reference_absent.metrics.table_cell_content_recall is None
    assert reference_absent.metrics.table_cell_content_f1 == 0
    assert reference_absent.metrics.table_cell_token_f1 == 0


def test_candidate_cell_metrics_penalize_extra_cells_without_changing_accuracy() -> None:
    reference_table = table("reference", "html")
    candidate_structure = reference_table.structure.table
    assert candidate_structure is not None
    extra_cells = [
        TableCell(
            row_index=2,
            column_index=column,
            role="body",
            text=("South", "20")[column],
            fragments=[
                PageFragment(
                    page_number=1,
                    page_width=100,
                    page_height=100,
                    bbox=BoundingBox(
                        x0=column * 50,
                        y0=20,
                        x1=(column + 1) * 50,
                        y1=30,
                    ),
                )
            ],
        )
        for column in range(2)
    ]
    candidate_structure = candidate_structure.model_copy(
        update={"row_count": 3, "cells": [*candidate_structure.cells, *extra_cells]}
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(candidate_structure),
            "structure": ElementStructure(table=candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.table_cell_role_accuracy == 1
    assert report.metrics.table_cell_content_accuracy == 1
    assert report.metrics.table_cell_role_precision == pytest.approx(2 / 3)
    assert report.metrics.table_cell_role_recall == 1
    assert report.metrics.table_cell_role_f1 == pytest.approx(0.8)
    assert report.metrics.table_cell_content_precision == pytest.approx(2 / 3)
    assert report.metrics.table_cell_content_recall == 1
    assert report.metrics.table_cell_content_f1 == pytest.approx(0.8)
    assert report.metrics.table_cell_token_precision == pytest.approx(2 / 3)
    assert report.metrics.table_cell_token_recall == 1
    assert report.metrics.table_cell_token_f1 == pytest.approx(0.8)

    missing_report = evaluate_document(as_candidate([reference_table]), [candidate_table])
    assert missing_report.metrics.table_cell_token_precision == 1
    assert missing_report.metrics.table_cell_token_recall == pytest.approx(2 / 3)
    assert missing_report.metrics.table_cell_token_f1 == pytest.approx(0.8)


def test_cell_content_is_independent_of_grid_position() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    shifted_structure = reference_structure.model_copy(
        update={
            "row_count": 3,
            "header_row_count": 0,
            "cells": [
                cell.model_copy(update={"row_index": cell.row_index + 1})
                for cell in reference_structure.cells
            ],
        }
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(shifted_structure),
            "structure": ElementStructure(table=shifted_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.table_topology_accuracy == 0
    assert report.metrics.table_cell_role_accuracy == 0
    assert report.metrics.table_cell_content_accuracy == 1
    assert report.metrics.table_cell_content_precision == 1
    assert report.metrics.table_cell_content_recall == 1
    assert report.metrics.table_cell_content_f1 == 1


def test_cell_content_matching_preserves_duplicate_multiplicity() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    candidate_cells = [
        cell.model_copy(update={"text": "North" if cell.text == "10" else cell.text})
        for cell in reference_structure.cells
    ]
    candidate_structure = reference_structure.model_copy(update={"cells": candidate_cells})
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(candidate_structure),
            "structure": ElementStructure(table=candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.table_cell_content_accuracy == pytest.approx(3 / 4)
    assert report.metrics.table_cell_content_precision == pytest.approx(3 / 4)
    assert report.metrics.table_cell_content_recall == pytest.approx(3 / 4)


def test_cell_token_content_handles_merged_and_split_text_without_rewarding_cell_strings() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    candidate_cells = [
        reference_structure.cells[0].model_copy(update={"text": "Name Value"}),
        reference_structure.cells[1].model_copy(update={"text": ""}),
        *reference_structure.cells[2:],
    ]
    candidate_structure = reference_structure.model_copy(update={"cells": candidate_cells})
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(candidate_structure),
            "structure": ElementStructure(table=candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.table_cell_content_f1 == pytest.approx(0.5)
    assert report.metrics.table_cell_token_precision == 1
    assert report.metrics.table_cell_token_recall == 1
    assert report.metrics.table_cell_token_f1 == 1
    assert report.metrics.table_cell_content_assignment_accuracy == pytest.approx(0.5)
    assert report.metrics.table_grid_topology_accuracy == 1

    split_report = evaluate_document(
        as_candidate([reference_table.model_copy(update={"element_id": "candidate-split"})]),
        [candidate_table],
    )
    assert split_report.metrics.table_cell_content_f1 == pytest.approx(0.5)
    assert split_report.metrics.table_cell_token_f1 == 1
    assert split_report.metrics.table_cell_content_assignment_accuracy == pytest.approx(0.5)


def test_cell_token_content_preserves_multiplicity_and_ignores_empty_cells() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    reference_cells = [
        cell.model_copy(update={"text": text})
        for cell, text in zip(
            reference_structure.cells,
            ("Repeated", "Repeated", "", ""),
            strict=True,
        )
    ]
    candidate_cells = [
        cell.model_copy(update={"text": text})
        for cell, text in zip(
            reference_structure.cells,
            ("Repeated", "", "", "Extra"),
            strict=True,
        )
    ]
    reference_structure = reference_structure.model_copy(update={"cells": reference_cells})
    candidate_structure = reference_structure.model_copy(update={"cells": candidate_cells})
    reference_table = reference_table.model_copy(
        update={
            "content": render_table(reference_structure),
            "structure": ElementStructure(table=reference_structure),
        }
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(candidate_structure),
            "structure": ElementStructure(table=candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.table_cell_token_precision == pytest.approx(0.5)
    assert report.metrics.table_cell_token_recall == pytest.approx(0.5)
    assert report.metrics.table_cell_token_f1 == pytest.approx(0.5)
    assert report.metrics.table_cell_content_assignment_accuracy == pytest.approx(1 / 3)


def test_cell_content_inventory_does_not_hide_wrong_cell_assignment() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    swapped_cells = [
        cell.model_copy(update={"text": reference_structure.cells[3 - index].text})
        for index, cell in enumerate(reference_structure.cells)
    ]
    candidate_structure = reference_structure.model_copy(update={"cells": swapped_cells})
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(candidate_structure),
            "structure": ElementStructure(table=candidate_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.table_cell_content_f1 == 1
    assert report.metrics.table_cell_token_f1 == 1
    assert report.metrics.table_cell_content_assignment_accuracy == 0
    assert report.metrics.table_grid_topology_accuracy == 1
    assert report.metrics.table_cell_role_accuracy == 1


def test_cell_content_assignment_is_conditional_on_shared_topology() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    shifted_structure = reference_structure.model_copy(
        update={
            "column_count": 4,
            "cells": [
                cell.model_copy(update={"column_index": cell.column_index + 2})
                for cell in reference_structure.cells
            ],
        }
    )
    candidate_table = reference_table.model_copy(
        update={
            "element_id": "candidate",
            "content": render_table(shifted_structure),
            "structure": ElementStructure(table=shifted_structure),
        }
    )

    report = evaluate_document(as_candidate([candidate_table]), [reference_table])

    assert report.metrics.table_cell_content_f1 == 1
    assert report.metrics.table_cell_token_f1 == 1
    assert report.metrics.table_cell_content_assignment_accuracy == 0
    assert report.metrics.table_grid_topology_accuracy == 0


def test_blank_cell_content_has_no_token_or_assignment_claim() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    blank_structure = reference_structure.model_copy(
        update={"cells": [cell.model_copy(update={"text": ""}) for cell in reference_structure.cells]}
    )
    reference_table = reference_table.model_copy(
        update={"content": "", "structure": ElementStructure(table=blank_structure)}
    )

    report = evaluate_document(as_candidate([reference_table]), [reference_table])

    assert report.metrics.table_cell_content_f1 == 1
    assert report.metrics.table_cell_token_precision is None
    assert report.metrics.table_cell_token_recall is None
    assert report.metrics.table_cell_token_f1 is None
    assert report.metrics.table_cell_content_assignment_accuracy is None


def test_table_pair_alignment_uses_content_inventory_not_cell_list_order() -> None:
    first = table("first", "html")
    first_structure = first.structure.table
    assert first_structure is not None
    second_cells = [
        cell.model_copy(update={"text": text})
        for cell, text in zip(
            first_structure.cells,
            ("Region", "Amount", "South", "20"),
            strict=True,
        )
    ]
    second_structure = first_structure.model_copy(update={"cells": second_cells})
    second = first.model_copy(
        update={
            "element_id": "second",
            "content": render_table(second_structure),
            "structure": ElementStructure(table=second_structure),
        }
    )
    reversed_first_structure = first_structure.model_copy(update={"cells": first_structure.cells[::-1]})
    reversed_second_structure = second_structure.model_copy(update={"cells": second_structure.cells[::-1]})
    candidates = as_candidate([
        second.model_copy(
            update={
                "content": render_table(reversed_second_structure),
                "structure": ElementStructure(table=reversed_second_structure),
            }
        ),
        first.model_copy(
            update={
                "content": render_table(reversed_first_structure),
                "structure": ElementStructure(table=reversed_first_structure),
            }
        ),
    ])

    report = evaluate_document(candidates, [first, second])

    assert report.metrics.table_detection_precision == 1
    assert report.metrics.table_detection_recall == 1
    assert report.metrics.table_cell_content_f1 == 1
    assert report.metrics.table_cell_token_f1 == 1
    assert report.metrics.table_cell_content_assignment_accuracy == 1


def test_cross_page_table_is_one_aligned_pair() -> None:
    reference_table = table("reference", "html")
    second_fragment = reference_table.fragments[0].model_copy(update={"page_number": 2})
    reference_table = reference_table.model_copy(
        update={"fragments": [*reference_table.fragments, second_fragment]}
    )

    report = evaluate_document(as_candidate([reference_table]), [reference_table])

    assert report.metrics.table_detection_precision == 1
    assert report.metrics.table_detection_recall == 1
    assert report.metrics.table_cell_content_f1 == 1
    assert report.metrics.table_cell_token_f1 == 1
    assert report.metrics.cross_page_continuity_accuracy == 1


def test_candidate_cannot_be_used_as_reference() -> None:
    candidate = as_candidate([paragraph("element", "text", 0, 0)])

    with pytest.raises(ValueError, match="reference elements must have silver or golden stage"):
        evaluate_document(candidate, candidate)


def test_reference_churn_reports_changed_and_candidate_identical_elements() -> None:
    previous = [paragraph("one", "old text", 0, 0), paragraph("two", "stable", 1, 20)]
    current = [paragraph("one", "new text", 0, 0), paragraph("two", "stable", 1, 20)]
    candidate = as_candidate([paragraph("candidate", "new text", 0, 0)])

    report = evaluate_reference_churn(previous, current, candidate=candidate)

    assert report.changed_count == 1
    assert report.unchanged_count == 1
    assert report.churn_rate == 0.5
    assert report.candidate_identity_rate == 0.5
    assert report.candidate_content_identity_rate == 0.5


def test_candidate_churn_reports_ipcc_like_numeric_fragmentation_and_merge() -> None:
    combined = paragraph("combined", "0.3–0.6 1.1–2.2 3.3–4.4", 0, 0)
    split = [
        paragraph("low", "0.3–0.6", 0, 40),
        paragraph("middle", "1.1–2.2", 1, 60),
        paragraph("high", "3.3–4.4", 2, 80),
    ]

    fragmentation = evaluate_candidate_churn(as_candidate([combined]), as_candidate(split))
    merge = evaluate_candidate_churn(as_candidate(split), as_candidate([combined]))

    assert fragmentation.content_fragmentation_hits == 1
    assert fragmentation.content_fragmentation_rate == 1
    assert fragmentation.content_merge_hits == 0
    assert fragmentation.added_normalized_content_count == 3
    assert fragmentation.removed_normalized_content_count == 1
    assert merge.content_merge_hits == 1
    assert merge.content_merge_rate == pytest.approx(1 / 3)
    assert merge.content_fragmentation_hits == 0


def test_candidate_churn_counts_duplicate_normalized_content_as_a_multiset() -> None:
    previous = as_candidate([
        paragraph("first", "Alpha", 0, 0),
        paragraph("second", "Alpha", 1, 20),
    ])
    current = as_candidate([paragraph("remaining", "  Alpha  ", 0, 0)])

    report = evaluate_candidate_churn(previous, current)

    assert report.exact_content_match_count == 1
    assert report.removed_normalized_content_count == 1
    assert report.added_normalized_content_count == 0
    assert report.exact_content_identity_rate == 0.5


def test_candidate_churn_duplicate_components_cannot_manufacture_fragmentation() -> None:
    previous = as_candidate([paragraph("whole", "Alpha Beta", 0, 0)])
    current = as_candidate([
        paragraph("alpha-one", "Alpha", 0, 20),
        paragraph("alpha-two", "Alpha", 1, 40),
    ])

    report = evaluate_candidate_churn(previous, current)

    assert report.content_fragmentation_hits == 0
    assert report.content_fragmentation_rate == 0


def test_candidate_churn_merge_rate_is_stable_when_unrelated_extras_change() -> None:
    previous = as_candidate([
        paragraph("alpha", "Alpha", 0, 0),
        paragraph("beta", "Beta", 1, 20),
        paragraph("stable", "Stable", 2, 40),
    ])
    compact = as_candidate([
        paragraph("merged", "Alpha Beta", 0, 0),
        paragraph("stable-current", "Stable", 1, 40),
    ])
    padded = as_candidate([
        *compact,
        paragraph("noise-one", "Unmatched noise one", 2, 60),
        paragraph("noise-two", "Unmatched noise two", 3, 80),
    ])

    compact_report = evaluate_candidate_churn(previous, compact)
    padded_report = evaluate_candidate_churn(previous, padded)

    assert compact_report.content_merge_hits == 1
    assert padded_report.content_merge_hits == 1
    assert compact_report.content_merge_rate == pytest.approx(1 / 3)
    assert padded_report.content_merge_rate == compact_report.content_merge_rate


def test_candidate_churn_exact_counterpart_suppresses_fragmentation() -> None:
    previous = as_candidate([paragraph("whole", "Alpha Beta", 0, 0)])
    current = as_candidate([
        paragraph("exact", "Alpha Beta", 0, 0),
        paragraph("alpha", "Alpha", 1, 20),
        paragraph("beta", "Beta", 2, 40),
    ])

    report = evaluate_candidate_churn(previous, current)

    assert report.content_fragmentation_hits == 0
    assert report.exact_content_match_count == 1
    assert report.exact_content_type_order_match_count == 1


def test_candidate_churn_rejects_cross_document_and_mixed_stage_revisions() -> None:
    previous = as_candidate([paragraph("previous", "Alpha", 0, 0)])
    other_document = [
        element.model_copy(update={"document_id": "other"})
        for element in as_candidate([paragraph("current", "Alpha", 0, 0)])
    ]

    with pytest.raises(ValueError, match="candidate revisions must belong to one document"):
        evaluate_candidate_churn(previous, other_document)
    with pytest.raises(ValueError, match="previous candidate elements must have candidate stage"):
        evaluate_candidate_churn([paragraph("silver", "Alpha", 0, 0)], previous)
    with pytest.raises(ValueError, match="current candidate elements must have candidate stage"):
        evaluate_candidate_churn(previous, [paragraph("silver", "Alpha", 0, 0)])


def test_candidate_churn_no_op_has_exact_identity_and_frozen_report() -> None:
    revision = as_candidate([
        paragraph("one", "First paragraph", 0, 0),
        paragraph("two", "Second paragraph", 1, 20),
    ])

    report = evaluate_candidate_churn(revision, revision)

    assert isinstance(report, CandidateChurnReport)
    assert report.previous_stage == "candidate"
    assert report.current_stage == "candidate"
    assert report.previous_count == 2
    assert report.current_count == 2
    assert report.added_normalized_content_count == 0
    assert report.removed_normalized_content_count == 0
    assert report.content_fragmentation_hits == 0
    assert report.content_merge_hits == 0
    assert report.aligned_count == 2
    assert report.exact_content_identity_rate == 1
    assert report.exact_type_identity_rate == 1
    assert report.exact_order_identity_rate == 1
    assert report.exact_content_type_order_identity_rate == 1
    assert report.model_config.get("frozen") is True


def test_alignment_ignores_rejected_edges_during_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = as_candidate([
        paragraph("candidate-zero", "Zero", 0, 0),
        paragraph("candidate-one", "One", 1, 20),
    ])
    reference = [
        paragraph("reference-zero", "Zero", 0, 0),
        paragraph("reference-one", "One", 1, 20),
    ]
    scores = [[0.6, 0.34], [0.34, 0.0]]

    def synthetic_score(
        candidate_element: DocumentElement,
        reference_element: DocumentElement,
        *,
        candidate_by_id: dict[str, DocumentElement],
        reference_by_id: dict[str, DocumentElement],
    ) -> float:
        del candidate_by_id, reference_by_id
        candidate_index = candidate.index(candidate_element)
        reference_index = reference.index(reference_element)
        return scores[candidate_index][reference_index]

    monkeypatch.setattr("app.pdf2md.evaluation._element_similarity", synthetic_score)

    alignments = align_elements(candidate, reference)

    assert [(alignment.candidate_index, alignment.reference_index) for alignment in alignments] == [(0, 0)]


def test_metrics_are_invariant_to_input_list_permutation() -> None:
    reference = [
        paragraph("one", "First paragraph", 0, 0),
        paragraph("two", "Second paragraph", 1, 20),
    ]
    candidate = as_candidate(reference)

    canonical = evaluate_document(candidate, reference)
    permuted = evaluate_document(candidate[::-1], reference[::-1])

    assert permuted.metrics == canonical.metrics
    assert permuted.metrics.reading_order_accuracy == 1
    assert permuted.metrics.normalized_character_error_rate == 0


def test_text_elements_on_disjoint_pages_do_not_align() -> None:
    reference = paragraph("reference", "Repeated running heading", 0, 0)
    candidate = paragraph("candidate", "Repeated running heading", 0, 0)
    candidate = candidate.model_copy(
        update={"fragments": [candidate.fragments[0].model_copy(update={"page_number": 2})]}
    )

    report = evaluate_document(as_candidate([candidate]), [reference])

    assert report.alignments == []
    assert report.metrics.element_f1 == 0


def test_multipage_geometry_penalizes_missing_page_and_preserves_all_fragments() -> None:
    reference = paragraph("reference", "Cross-page paragraph", 0, 0)
    page_two = reference.fragments[0].model_copy(update={"page_number": 2})
    reference = reference.model_copy(update={"fragments": [reference.fragments[0], page_two]})
    candidate = as_candidate([
        reference.model_copy(update={"element_id": "candidate", "fragments": [reference.fragments[0]]})
    ])

    missing_page = evaluate_document(candidate, [reference])
    reordered_fragments = evaluate_document(
        as_candidate([reference.model_copy(update={"fragments": reference.fragments[::-1]})]),
        [reference],
    )

    assert missing_page.metrics.mean_geometry_iou == pytest.approx(0.5)
    assert missing_page.metrics.cross_page_continuity_accuracy == 0
    assert reordered_fragments.metrics.mean_geometry_iou == 1


def test_assignment_accuracy_penalizes_missing_extra_and_unmatched_cells() -> None:
    reference_table = table("reference", "html")
    reference_structure = reference_table.structure.table
    assert reference_structure is not None
    missing_structure = reference_structure.model_copy(update={"cells": reference_structure.cells[:-1]})
    missing_table = reference_table.model_copy(
        update={
            "element_id": "missing",
            "content": render_table(missing_structure),
            "structure": ElementStructure(table=missing_structure),
        }
    )
    extra_cell = reference_structure.cells[-1].model_copy(update={"row_index": 2, "text": "Extra assignment"})
    extra_structure = reference_structure.model_copy(
        update={"row_count": 3, "cells": [*reference_structure.cells, extra_cell]}
    )
    extra_table = reference_table.model_copy(
        update={
            "element_id": "extra",
            "content": render_table(extra_structure),
            "structure": ElementStructure(table=extra_structure),
        }
    )

    missing = evaluate_document(as_candidate([missing_table]), [reference_table])
    extra = evaluate_document(as_candidate([extra_table]), [reference_table])
    unmatched = evaluate_document(as_candidate([paragraph("body", "Body", 0, 50)]), [reference_table])

    assert missing.metrics.table_cell_content_assignment_accuracy == pytest.approx(3 / 4)
    assert extra.metrics.table_cell_content_assignment_accuracy == pytest.approx(4 / 5)
    assert unmatched.metrics.table_cell_content_assignment_accuracy == 0


def test_reference_identity_metrics_preserve_duplicate_multiplicity() -> None:
    reference = [
        paragraph("first", "Repeated", 0, 0),
        paragraph("second", "Repeated", 1, 0),
    ]
    candidate = as_candidate([reference[0]])

    report = evaluate_document(candidate, reference)

    assert report.metrics.candidate_reference_identity_rate == 0.5
    assert report.metrics.candidate_reference_content_identity_rate == 0.5


def test_identity_ignores_provenance_only_structure_properties() -> None:
    candidate = paragraph("candidate", "Body", 0, 0)
    reference = candidate.model_copy(
        update={
            "element_id": "reference",
            "structure": candidate.structure.model_copy(
                update={
                    "properties": [
                        StructureProperty(key="source_note", value="native page 1 block 2"),
                        StructureProperty(
                            key="inline_footnote_markers",
                            value='{"marker":"a","target":"footnote-a"}',
                        ),
                    ]
                }
            ),
        }
    )

    report = evaluate_document(as_candidate([candidate]), [reference])

    assert report.metrics.candidate_reference_identity_rate == 1


def test_identity_preserves_semantic_structure_properties() -> None:
    candidate = paragraph("candidate", "Body", 0, 0)
    reference = candidate.model_copy(
        update={
            "element_id": "reference",
            "structure": candidate.structure.model_copy(
                update={"properties": [StructureProperty(key="semantic_key", value="meaningful")]}
            ),
        }
    )

    report = evaluate_document(as_candidate([candidate]), [reference])

    assert report.metrics.candidate_reference_identity_rate == 0


def test_merge_rates_remain_bounded_with_duplicate_merged_candidates() -> None:
    reference = [paragraph("alpha", "Alpha", 0, 0), paragraph("beta", "Beta", 1, 5)]
    merged = [paragraph(f"merged-{index}", "Alpha Beta", index, 0) for index in range(3)]

    report = evaluate_document(as_candidate(merged), reference)

    assert report.metrics.merge_rate == 1
    assert report.metrics.content_merge_rate == 1
