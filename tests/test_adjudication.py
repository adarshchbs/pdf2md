import pytest
from pydantic import ValidationError

from app.pdf2md.adjudication import (
    AdjudicationDecision,
    apply_adjudication_decisions,
    build_blind_adjudication_bundle,
    promote_silver_to_golden,
    select_human_audit_elements,
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
)


def paragraph(
    element_id: str,
    content: str,
    order: int = 0,
    *,
    confidence: float = 1,
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
                bbox=BoundingBox(x0=0, y0=order + 1, x1=100, y1=order + 2),
            )
        ],
        structure=ElementStructure(paragraph=ParagraphStructure(role="body")),
        annotation=AnnotationMetadata(
            stage="silver",
            revision=1,
            annotator="test",
            confidence=confidence,
            adjudication_status="unreviewed",
        ),
    )


def as_candidate(element: DocumentElement) -> DocumentElement:
    return element.model_copy(
        update={"annotation": element.annotation.model_copy(update={"stage": "candidate"})}
    )


def test_candidate_choice_cannot_be_imported_into_silver() -> None:
    reference = paragraph("stable-id", "silver text")
    candidate = as_candidate(paragraph("candidate-id", "candidate text"))
    bundle = build_blind_adjudication_bundle([candidate], [reference])
    key = bundle.keys[0]
    choice = "A" if key.source_a == "candidate" else "B"
    decision = AdjudicationDecision(
        packet_id=key.packet_id,
        choice=choice,
        reason_code="candidate_matches_image",
        rationale="The page image contains the candidate wording.",
        confidence=0.95,
    )

    with pytest.raises(ValueError, match="cannot be imported into silver"):
        apply_adjudication_decisions(
            [candidate],
            [reference],
            bundle,
            [decision],
            adjudicator="judge",
        )


def test_independent_choice_c_correction_produces_silver_revision() -> None:
    reference = paragraph("stable-id", "silver text")
    candidate = as_candidate(paragraph("candidate-id", "candidate text"))
    corrected = paragraph("correction-id", "text independently read from bronze")
    bundle = build_blind_adjudication_bundle([candidate], [reference])
    decision = AdjudicationDecision(
        packet_id=bundle.keys[0].packet_id,
        choice="C",
        reason_code="independent_reannotation",
        rationale="The element was reconstructed from the page image.",
        confidence=0.95,
        corrected_element=corrected,
    )

    result = apply_adjudication_decisions([candidate], [reference], bundle, [decision], adjudicator="judge")

    assert result[0].content == "text independently read from bronze"
    assert result[0].element_id == "stable-id"
    assert result[0].annotation.stage == "silver"
    assert result[0].annotation.revision == 2
    assert result[0].annotation.parent_revision == 1


def test_independent_correction_cannot_drop_existing_provenance() -> None:
    reference = paragraph("stable-id", "silver text")
    reference = reference.model_copy(
        update={
            "fragments": [reference.fragments[0].model_copy(update={"source_item_ids": ["liteparse:item-1"]})]
        }
    )
    candidate = as_candidate(paragraph("candidate-id", "candidate text"))
    corrected = paragraph("correction-id", "independent correction")
    bundle = build_blind_adjudication_bundle([candidate], [reference])
    decision = AdjudicationDecision(
        packet_id=bundle.keys[0].packet_id,
        choice="C",
        reason_code="independent_reannotation",
        rationale="The correction was read from the page image.",
        confidence=0.9,
        corrected_element=corrected,
    )

    with pytest.raises(ValueError, match="cannot discard reference source_item_ids"):
        apply_adjudication_decisions([candidate], [reference], bundle, [decision], adjudicator="judge")


def test_blind_packets_remove_provenance_and_round_geometry() -> None:
    reference = paragraph("reference", "same text")
    reference = reference.model_copy(
        update={
            "fragments": [reference.fragments[0].model_copy(update={"source_item_ids": ["liteparse:item-1"]})]
        }
    )
    raw_candidate = paragraph("candidate", "same text")
    candidate_fragment = raw_candidate.fragments[0]
    candidate = as_candidate(
        raw_candidate.model_copy(
            update={
                "fragments": [
                    candidate_fragment.model_copy(
                        update={"bbox": candidate_fragment.bbox.model_copy(update={"x1": 99.9999999})}
                    )
                ]
            }
        )
    )

    packet = build_blind_adjudication_bundle([candidate], [reference]).packets[0]

    options = [option for option in (packet.option_a, packet.option_b) if option is not None]
    assert all("source_item_ids" not in fragment for option in options for fragment in option.fragments)
    assert all(
        fragment["bbox"]["x1"] == round(fragment["bbox"]["x1"], 3)
        for option in options
        for fragment in option.fragments
        if isinstance(fragment["bbox"], dict) and isinstance(fragment["bbox"]["x1"], (int, float))
    )


def test_blind_options_defeat_nested_identity_and_provenance_classifiers() -> None:
    def figure(source: str, digest: str) -> DocumentElement:
        base = paragraph(f"{source}-figure-id", "Adjudicable figure", confidence=1)
        return base.model_copy(
            update={
                "element_type": "figure",
                "structure": ElementStructure(
                    figure=FigureStructure(
                        asset_path=f"data/{source}/page-1-{digest}.png",
                        sha256=digest,
                        width=640,
                        height=480,
                        caption_element_id=f"{source}-caption-7",
                    ),
                    footnote=FootnoteStructure(
                        label="1",
                        reference_element_ids=[f"{source}-reference-3"],
                        association_confident=True,
                    ),
                    linked_element_ids=[f"{source}-linked-2"],
                    properties=[
                        StructureProperty(key="source_note", value=f"{source} item {source}-item-9"),
                        StructureProperty(key="panels", value="a) input; b) output"),
                    ],
                ),
            }
        )

    candidate = as_candidate(figure("candidate", "a" * 64))
    reference = figure("reference", "b" * 64)
    packet = build_blind_adjudication_bundle([candidate], [reference]).packets[0]
    options = [option.model_dump(mode="json") for option in (packet.option_a, packet.option_b) if option]
    forbidden_names = {
        "asset_path",
        "caption_element_id",
        "linked_element_ids",
        "reference_element_ids",
        "sha256",
        "source_item_ids",
    }
    identity_tokens = {
        candidate.element_id,
        reference.element_id,
        "candidate-caption-7",
        "reference-caption-7",
        "candidate-reference-3",
        "reference-reference-3",
        "candidate-linked-2",
        "reference-linked-2",
        "a" * 64,
        "b" * 64,
    }

    def attack_every_nested_field(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                assert key not in forbidden_names
                assert not key.endswith(("_element_id", "_element_ids"))
                attack_every_nested_field(nested)
        elif isinstance(value, list):
            for nested in value:
                attack_every_nested_field(nested)
        elif isinstance(value, str):
            assert all(token not in value for token in identity_tokens)
            assert "source_note" not in value

    for option in options:
        attack_every_nested_field(option)
        structure = option["structure"]
        assert isinstance(structure, dict)
        figure_payload = structure["figure"]
        assert isinstance(figure_payload, dict)
        assert figure_payload == {"height": 480, "width": 640}
        assert structure["properties"] == [{"key": "panels", "value": "a) input; b) output"}]


def test_choice_c_requires_corrected_element() -> None:
    with pytest.raises(ValidationError, match="requires corrected_element"):
        AdjudicationDecision(
            packet_id="packet",
            choice="C",
            reason_code="both_wrong",
            rationale="Neither candidate matches the image.",
            confidence=0.8,
        )


def test_human_audit_promotes_silver_to_golden() -> None:
    elements = [paragraph(f"element-{index}", f"text {index}", index) for index in range(100)]

    golden = promote_silver_to_golden(
        elements,
        {"element-50"},
        human_reviewer="human",
    )

    assert all(element.annotation.stage == "golden" for element in golden)
    assert golden[50].annotation.adjudication_status == "human_reviewed"
    assert golden[50].annotation.annotator == "human"
    assert golden[0].annotation.adjudication_status == "accepted"


def test_audit_is_stratified_and_prioritizes_low_confidence() -> None:
    elements = [paragraph(f"element-{index}", f"text {index}", index) for index in range(100)]
    elements[50] = paragraph("priority", "uncertain", 50, confidence=0.5)

    selected = select_human_audit_elements(elements, fraction=0.02, seed=42)

    assert "priority" in selected
    assert len(selected) == 2


def test_audit_prioritizes_missing_item_level_provenance() -> None:
    elements = [paragraph(f"element-{index}", f"text {index}", index) for index in range(100)]
    elements = [
        element.model_copy(
            update={
                "fragments": [
                    element.fragments[0].model_copy(
                        update={"source_item_ids": [f"pymupdf:p000001:span:item-{index}"]}
                    )
                ]
            }
        )
        for index, element in enumerate(elements)
    ]
    elements[50] = paragraph("missing-provenance", "uncertain source", 50)

    selected = select_human_audit_elements(elements, fraction=0.01, seed=42)

    assert selected == ["missing-provenance"]


def test_audit_never_exceeds_requested_fraction_when_many_elements_are_risky() -> None:
    elements = [paragraph(f"element-{index}", f"text {index}", index, confidence=0.5) for index in range(100)]

    selected = select_human_audit_elements(elements, fraction=0.02, seed=42)

    assert len(selected) == 2
