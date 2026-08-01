from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from app.pdf2md.evaluation import EvaluationReport, evaluate_document
from app.pdf2md.schema import (
    AnnotationMetadata,
    DocumentElement,
    SchemaModel,
)

Source = Literal["candidate", "reference", "missing"]
DecisionChoice = Literal["A", "B", "C"]


class BlindElement(SchemaModel):
    element_type: str
    content: str
    format: str
    include_in_output: bool
    fragments: list[dict[str, JsonValue]]
    structure: dict[str, JsonValue]


class BlindAdjudicationPacket(SchemaModel):
    packet_id: str
    document_id: str
    page_numbers: list[int]
    page_images: list[str]
    option_a: BlindElement | None
    option_b: BlindElement | None
    issue_codes: list[str]


class AdjudicationKey(SchemaModel):
    packet_id: str
    candidate_index: int | None = Field(default=None, ge=0)
    reference_index: int | None = Field(default=None, ge=0)
    source_a: Source
    source_b: Source


class AdjudicationDecision(SchemaModel):
    packet_id: str
    choice: DecisionChoice
    reason_code: str
    rationale: str
    confidence: float = Field(ge=0, le=1)
    corrected_element: DocumentElement | None = None

    @model_validator(mode="after")
    def validate_correction(self) -> AdjudicationDecision:
        if self.choice == "C" and self.corrected_element is None:
            raise ValueError("choice C requires corrected_element")
        if self.choice != "C" and self.corrected_element is not None:
            raise ValueError("corrected_element is only valid for choice C")
        return self


class AdjudicationBundle(SchemaModel):
    report: EvaluationReport
    packets: list[BlindAdjudicationPacket]
    keys: list[AdjudicationKey]


def build_blind_adjudication_bundle(
    candidate: list[DocumentElement],
    reference: list[DocumentElement],
    *,
    page_images: dict[int, Path] | None = None,
) -> AdjudicationBundle:
    report = evaluate_document(candidate, reference)
    images = page_images or {}
    packets: list[BlindAdjudicationPacket] = []
    keys: list[AdjudicationKey] = []

    for alignment in report.alignments:
        candidate_element = candidate[alignment.candidate_index]
        reference_element = reference[alignment.reference_index]
        issues = _issue_codes(candidate_element, reference_element)
        if not issues:
            continue
        packet, key = _packet(
            candidate_element,
            reference_element,
            alignment.candidate_index,
            alignment.reference_index,
            issues,
            images,
        )
        packets.append(packet)
        keys.append(key)

    for index in report.unmatched_candidate_indices:
        packet, key = _packet(
            candidate[index],
            None,
            index,
            None,
            ["candidate_only_element"],
            images,
        )
        packets.append(packet)
        keys.append(key)
    for index in report.unmatched_reference_indices:
        packet, key = _packet(
            None,
            reference[index],
            None,
            index,
            ["reference_only_element"],
            images,
        )
        packets.append(packet)
        keys.append(key)

    return AdjudicationBundle(report=report, packets=packets, keys=keys)


def apply_adjudication_decisions(
    candidate: list[DocumentElement],
    reference: list[DocumentElement],
    bundle: AdjudicationBundle,
    decisions: list[AdjudicationDecision],
    *,
    adjudicator: str,
) -> list[DocumentElement]:
    decision_by_packet = {decision.packet_id: decision for decision in decisions}
    if len(decision_by_packet) != len(decisions):
        raise ValueError("adjudication decisions contain duplicate packet_id values")
    expected_ids = {key.packet_id for key in bundle.keys}
    if set(decision_by_packet) != expected_ids:
        missing = sorted(expected_ids - set(decision_by_packet))
        extra = sorted(set(decision_by_packet) - expected_ids)
        raise ValueError(f"decisions must cover every packet; missing={missing}, extra={extra}")

    result: dict[int, DocumentElement] = {index: element for index, element in enumerate(reference)}
    additions: list[DocumentElement] = []
    for key in bundle.keys:
        decision = decision_by_packet[key.packet_id]
        selected_source = _selected_source(key, decision.choice)
        if selected_source == "candidate":
            raise ValueError(
                "candidate selections cannot be imported into silver; re-annotate from bronze as choice C"
            )
        selected = _selected_element(
            selected_source,
            key,
            candidate,
            reference,
            decision.corrected_element,
        )
        if key.reference_index is not None:
            if selected is None:
                result.pop(key.reference_index)
                continue
            parent = reference[key.reference_index]
            _validate_independent_correction(parent, selected, decision)
            stable_id = parent.element_id
            result[key.reference_index] = _promote_to_silver(
                selected,
                stable_id,
                decision,
                adjudicator,
                parent_revision=reference[key.reference_index].annotation.revision,
            )
        elif selected is not None:
            additions.append(
                _promote_to_silver(
                    selected,
                    selected.element_id,
                    decision,
                    adjudicator,
                    parent_revision=None,
                )
            )

    ordered = [*result.values(), *additions]
    ordered.sort(key=_document_position)
    return [_with_order(element, order) for order, element in enumerate(ordered)]


def promote_silver_to_golden(
    elements: list[DocumentElement],
    audited_element_ids: set[str],
    *,
    human_reviewer: str,
) -> list[DocumentElement]:
    if not elements:
        raise ValueError("golden promotion requires at least one silver element")
    if any(element.annotation.stage != "silver" for element in elements):
        raise ValueError("only silver elements can be promoted to golden")
    known_ids = {element.element_id for element in elements}
    unknown_ids = audited_element_ids - known_ids
    if unknown_ids:
        raise ValueError(f"audit contains unknown element IDs: {sorted(unknown_ids)}")
    minimum_audit_count = max(1, math.ceil(len(elements) * 0.01))
    if len(audited_element_ids) < minimum_audit_count:
        raise ValueError(f"golden promotion requires at least {minimum_audit_count} human-audited elements")

    golden: list[DocumentElement] = []
    for element in elements:
        payload = element.model_dump(mode="json")
        payload["annotation"] = AnnotationMetadata(
            stage="golden",
            revision=element.annotation.revision + 1,
            annotator=(
                human_reviewer if element.element_id in audited_element_ids else element.annotation.annotator
            ),
            confidence=(1.0 if element.element_id in audited_element_ids else element.annotation.confidence),
            parent_revision=element.annotation.revision,
            adjudication_status=(
                "human_reviewed" if element.element_id in audited_element_ids else "accepted"
            ),
        ).model_dump(mode="json")
        golden.append(DocumentElement.model_validate(payload))
    return golden


def select_human_audit_elements(
    elements: list[DocumentElement],
    *,
    fraction: float = 0.02,
    seed: int = 0,
) -> list[str]:
    if not 0.01 <= fraction <= 0.02:
        raise ValueError("human audit fraction must be between 1% and 2%")
    if not elements:
        return []

    target = max(1, math.ceil(len(elements) * fraction))
    priority_strata: dict[str, list[DocumentElement]] = {}
    ordinary_strata: dict[str, list[DocumentElement]] = {}
    for element in elements:
        risk_codes = _audit_risk_codes(element)
        strata = priority_strata if risk_codes else ordinary_strata
        key = ":".join(risk_codes) if risk_codes else _audit_stratum(element)
        strata.setdefault(key, []).append(element)

    generator = random.Random(seed)
    for strata in (priority_strata, ordinary_strata):
        for values in strata.values():
            generator.shuffle(values)

    selected: list[str] = []
    for strata in (priority_strata, ordinary_strata):
        while len(selected) < target and any(strata.values()):
            for key in sorted(strata):
                if strata[key] and len(selected) < target:
                    selected.append(strata[key].pop().element_id)
    return sorted(selected)


def _packet(
    candidate: DocumentElement | None,
    reference: DocumentElement | None,
    candidate_index: int | None,
    reference_index: int | None,
    issues: list[str],
    page_images: dict[int, Path],
) -> tuple[BlindAdjudicationPacket, AdjudicationKey]:
    element = candidate or reference
    if element is None:
        raise ValueError("adjudication packet requires at least one element")
    packet_id = _packet_id(element.document_id, candidate_index, reference_index)
    candidate_first = int(packet_id[-1], 16) % 2 == 0
    candidate_blind = _blind(candidate)
    reference_blind = _blind(reference)
    candidate_source: Source = "candidate" if candidate is not None else "missing"
    reference_source: Source = "reference" if reference is not None else "missing"
    if candidate_first:
        option_a, source_a = candidate_blind, candidate_source
        option_b, source_b = reference_blind, reference_source
    else:
        option_a, source_a = reference_blind, reference_source
        option_b, source_b = candidate_blind, candidate_source
    pages = sorted({
        fragment.page_number
        for value in (candidate, reference)
        if value is not None
        for fragment in value.fragments
    })
    return (
        BlindAdjudicationPacket(
            packet_id=packet_id,
            document_id=element.document_id,
            page_numbers=pages,
            page_images=[str(page_images[page]) for page in pages if page in page_images],
            option_a=option_a,
            option_b=option_b,
            issue_codes=issues,
        ),
        AdjudicationKey(
            packet_id=packet_id,
            candidate_index=candidate_index,
            reference_index=reference_index,
            source_a=source_a,
            source_b=source_b,
        ),
    )


def _issue_codes(candidate: DocumentElement, reference: DocumentElement) -> list[str]:
    issues: list[str] = []
    if candidate.element_type != reference.element_type:
        issues.append("element_type")
    if candidate.content != reference.content:
        issues.append("content")
    if candidate.format != reference.format:
        issues.append("format")
    if candidate.include_in_output != reference.include_in_output:
        issues.append("output_inclusion")
    if candidate.structure != reference.structure:
        issues.append("structure")
    if candidate.fragments != reference.fragments:
        issues.append("geometry")
    if _fragment_pages(candidate) != _fragment_pages(reference):
        issues.append("cross_page_continuity")
    return issues


_IDENTITY_FIELDS = {
    "asset_path",
    "caption_element_id",
    "linked_element_ids",
    "reference_element_ids",
    "sha256",
    "source_item_ids",
}
_PROVENANCE_PROPERTY_MARKERS = (
    "asset",
    "classification_note",
    "diagnostic",
    "joined_fragment",
    "parent_item",
    "source",
    "suppression",
    "traceability",
)
_PROVENANCE_VALUE_MARKERS = ("bronze", "liteparse", "native pdf", "pymupdf")
_GEOMETRY_FIELDS = {"page_width", "page_height", "x0", "y0", "x1", "y1"}


def blind_element_payload(element: DocumentElement | None) -> BlindElement | None:
    """Return adjudicable content with all nested identity and provenance fields removed."""
    if element is None:
        return None
    payload = element.model_dump(
        mode="json",
        include={
            "element_type",
            "content",
            "format",
            "include_in_output",
            "fragments",
            "structure",
        },
    )
    sanitized = _sanitize_blind_value(payload)
    if not isinstance(sanitized, dict):
        raise TypeError("blind element sanitizer must produce an object")
    return BlindElement.model_validate(sanitized)


def _blind(element: DocumentElement | None) -> BlindElement | None:
    return blind_element_payload(element)


def _sanitize_blind_value(value: JsonValue, *, field_name: str | None = None) -> JsonValue:
    if isinstance(value, dict):
        sanitized: dict[str, JsonValue] = {}
        for key, nested in value.items():
            lowered = key.lower()
            if lowered in _IDENTITY_FIELDS or lowered.endswith(("_element_id", "_element_ids")):
                continue
            if key == "properties" and isinstance(nested, list):
                sanitized[key] = _sanitize_properties(nested)
                continue
            sanitized[key] = _sanitize_blind_value(nested, field_name=key)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_blind_value(nested, field_name=field_name) for nested in value]
    if field_name in _GEOMETRY_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(value, 3)
    return value


def _sanitize_properties(properties: list[JsonValue]) -> list[JsonValue]:
    sanitized: list[JsonValue] = []
    for prop in properties:
        if not isinstance(prop, dict):
            raise TypeError("structure properties must be objects")
        key = prop.get("key")
        value = prop.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("structure properties require string key and value")
        lowered_key = key.lower()
        lowered_value = value.lower()
        if any(marker in lowered_key for marker in _PROVENANCE_PROPERTY_MARKERS):
            continue
        if any(marker in lowered_value for marker in _PROVENANCE_VALUE_MARKERS):
            continue
        sanitized.append({"key": key, "value": value})
    return sanitized


def _selected_source(key: AdjudicationKey, choice: DecisionChoice) -> Source:
    if choice == "A":
        return key.source_a
    if choice == "B":
        return key.source_b
    return "missing"


def _selected_element(
    source: Source,
    key: AdjudicationKey,
    candidate: list[DocumentElement],
    reference: list[DocumentElement],
    corrected: DocumentElement | None,
) -> DocumentElement | None:
    if corrected is not None:
        return corrected
    if source == "candidate":
        if key.candidate_index is None:
            raise ValueError("candidate source has no candidate index")
        return candidate[key.candidate_index]
    if source == "reference":
        if key.reference_index is None:
            raise ValueError("reference source has no reference index")
        return reference[key.reference_index]
    return None


def _validate_independent_correction(
    parent: DocumentElement,
    selected: DocumentElement,
    decision: AdjudicationDecision,
) -> None:
    if selected.document_id != parent.document_id:
        raise ValueError("corrected element must retain the reference document_id")
    if decision.choice != "C":
        return
    parent_has_provenance = any(fragment.source_item_ids for fragment in parent.fragments)
    corrected_has_provenance = any(fragment.source_item_ids for fragment in selected.fragments)
    if parent_has_provenance and not corrected_has_provenance:
        raise ValueError("independent correction cannot discard reference source_item_ids")


def _promote_to_silver(
    element: DocumentElement,
    element_id: str,
    decision: AdjudicationDecision,
    adjudicator: str,
    *,
    parent_revision: int | None,
) -> DocumentElement:
    payload = element.model_dump(mode="json")
    payload["element_id"] = element_id
    payload["annotation"] = AnnotationMetadata(
        stage="silver",
        revision=(parent_revision or 0) + 1,
        annotator=adjudicator,
        confidence=decision.confidence,
        parent_revision=parent_revision,
        adjudication_status="accepted" if decision.choice in {"A", "B"} else "corrected",
    ).model_dump(mode="json")
    return DocumentElement.model_validate(payload)


def _document_position(element: DocumentElement) -> tuple[int, float, float, int]:
    fragment = element.fragments[0]
    return fragment.page_number, fragment.bbox.y0, fragment.bbox.x0, element.order


def _audit_risk_codes(element: DocumentElement) -> list[str]:
    codes: list[str] = []
    if element.annotation.confidence < 0.8:
        codes.append("low-confidence")
    if element.annotation.adjudication_status == "corrected":
        codes.append("corrected")
    if len(element.fragments) > 1:
        codes.append("cross-page")
    if any(not fragment.source_item_ids for fragment in element.fragments):
        codes.append("missing-element-provenance")
    table = element.structure.table
    if table is not None and any(
        not fragment.source_item_ids for cell in table.cells for fragment in cell.fragments
    ):
        codes.append("missing-table-cell-provenance")
    if table is not None and table.representation == "html":
        codes.append("html-table")
    return codes


def _audit_stratum(element: DocumentElement) -> str:
    table = element.structure.table
    if table is not None:
        return f"table:{table.representation}"
    return element.element_type


def _packet_id(document_id: str, candidate_index: int | None, reference_index: int | None) -> str:
    value = f"{document_id}:{candidate_index}:{reference_index}"
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def _fragment_pages(element: DocumentElement) -> set[int]:
    return {fragment.page_number for fragment in element.fragments}


def _with_order(element: DocumentElement, order: int) -> DocumentElement:
    payload = element.model_dump(mode="json")
    payload["order"] = order
    return DocumentElement.model_validate(payload)
