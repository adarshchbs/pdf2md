from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pdf2md.schema import DocumentElement, PageFragment, TableCell, write_document_elements
from app.pdf2md.table_quality import normalize_table_text

SOURCE_CATALOG_SCHEMA_VERSION = "2.0.0"
LEGACY_SOURCE_CATALOG_SCHEMA_VERSION = "1.0.0"
SOURCE_ID_STABILITY_SCOPE = (
    "Stable for identical document bytes and canonical PyMuPDF native item geometry/content/style; "
    "extractor upgrades or rewrites that change those native observations intentionally change IDs."
)
SourceItemKind = Literal["span", "word", "rule"]
CoordinateFrame = Literal["source_page", "detector_page"]
IdScheme = Literal["native-content-v1", "legacy-ordinal-v1"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _document_hash(document_id: str) -> str:
    return (
        document_id
        if len(document_id) == 64 and all(char in "0123456789abcdef" for char in document_id)
        else _sha256_text(document_id)
    )


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("source item coordinates must be finite")
    return float(0.0 if value == 0 else value).hex()


def source_item_identity_sha256(
    *,
    document_id: str,
    page_number: int,
    kind: SourceItemKind,
    canonical_coordinates: tuple[float, float, float, float],
    text: str | None,
    style: object | None = None,
) -> tuple[str, str, str]:
    """Return the identity, content, and style hashes for one native item.

    Geometry is serialized with exact finite IEEE-754 hexadecimal values. Style is a
    caller-supplied canonical JSON value; ``None`` means that the native API exposes
    no independent style for that item kind.
    """
    content_sha256 = _sha256_text(text or "")
    style_bytes = json.dumps(style, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    style_sha256 = hashlib.sha256(style_bytes).hexdigest()
    payload = {
        "document_sha256": _document_hash(document_id),
        "page_number": page_number,
        "kind": kind,
        "canonical_native_geometry": [_canonical_float(value) for value in canonical_coordinates],
        "content_sha256": content_sha256,
        "style_sha256": style_sha256,
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return identity_sha256, content_sha256, style_sha256


def content_addressed_source_item_id(
    *,
    document_id: str,
    page_number: int,
    kind: SourceItemKind,
    canonical_coordinates: tuple[float, float, float, float],
    text: str | None,
    style: object | None = None,
    duplicate_index: int = 1,
) -> str:
    if duplicate_index < 1:
        raise ValueError("duplicate_index must be positive")
    identity_sha256, _, _ = source_item_identity_sha256(
        document_id=document_id,
        page_number=page_number,
        kind=kind,
        canonical_coordinates=canonical_coordinates,
        text=text,
        style=style,
    )
    return (
        f"pymupdf:d{_document_hash(document_id)}:p{page_number:06d}:{kind}:"
        f"h{identity_sha256}:d{duplicate_index:06d}"
    )


class SourceItem(_FrozenModel):
    source_item_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    kind: SourceItemKind
    coordinate_frame: CoordinateFrame
    x0: float
    y0: float
    x1: float
    y1: float
    text: str | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    style_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_x0: float
    canonical_y0: float
    canonical_x1: float
    canonical_y1: float
    duplicate_index: int = Field(ge=1)
    id_scheme: IdScheme

    @model_validator(mode="after")
    def validate_item(self) -> Self:
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        canonical = (self.canonical_x0, self.canonical_y0, self.canonical_x1, self.canonical_y1)
        if not all(math.isfinite(value) for value in (*coordinates, *canonical)):
            raise ValueError("source item coordinates must be finite")
        if self.x1 < self.x0 or self.canonical_x1 < self.canonical_x0:
            raise ValueError("source items cannot have negative width")
        if self.kind == "rule":
            if self.x1 == self.x0 or self.y1 != self.y0 or self.text is not None:
                raise ValueError("rule source items require zero height and null text")
            if self.canonical_x1 == self.canonical_x0 and self.canonical_y1 == self.canonical_y0:
                raise ValueError("canonical rule geometry must not be a point")
        elif self.y1 < self.y0 or (self.x1 == self.x0 and self.y1 == self.y0):
            raise ValueError("text source items require non-negative, non-point geometry")
        if self.content_sha256 != _sha256_text(self.text or ""):
            raise ValueError("source item content_sha256 does not match text")
        if self.id_scheme == "native-content-v1":
            expected_prefix = (
                f"pymupdf:d{_document_hash(self.document_id)}:p{self.page_number:06d}:"
                f"{self.kind}:h{self.identity_sha256}:d{self.duplicate_index:06d}"
            )
            if self.source_item_id != expected_prefix:
                raise ValueError("source item ID does not match its document-scoped native identity")
        elif not self.source_item_id.startswith(f"pymupdf:p{self.page_number:06d}:{self.kind}:"):
            raise ValueError("legacy source item ID is not page- and kind-scoped")
        return self


class SourceCatalogReferenceCounts(_FrozenModel):
    fragment_reference_count: int = Field(ge=0)
    unique_fragment_source_item_count: int = Field(ge=0)
    table_structural_reference_count: int = Field(ge=0)
    unique_table_structural_source_item_count: int = Field(ge=0)
    table_suppression_ledger_count: int = Field(ge=0)
    table_supported_span_reference_count: int = Field(ge=0)
    unique_table_supported_span_count: int = Field(ge=0)
    table_suppressed_span_reference_count: int = Field(ge=0)
    unique_table_suppressed_span_count: int = Field(ge=0)
    table_retained_overlap_span_reference_count: int = Field(ge=0)
    unique_table_retained_overlap_span_count: int = Field(ge=0)
    table_suppression_span_reference_count: int = Field(ge=0)
    unique_table_suppression_span_count: int = Field(ge=0)
    total_reference_count: int = Field(ge=0)
    total_unique_source_item_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _TableSpanLedgerRecord:
    table_id: str
    page_number: int
    supported_span_ids: tuple[str, ...]
    suppressed_span_ids: tuple[str, ...]
    retained_overlap_span_ids: tuple[str, ...]

    @property
    def referenced_span_ids(self) -> tuple[str, ...]:
        return (
            *self.supported_span_ids,
            *self.suppressed_span_ids,
            *self.retained_overlap_span_ids,
        )


class SourceCatalog(_FrozenModel):
    schema_version: Literal["2.0.0"] = SOURCE_CATALOG_SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    extractor: Literal["pymupdf"] = "pymupdf"
    id_stability_scope: Literal[
        "Stable for identical document bytes and canonical PyMuPDF native item geometry/content/style; extractor upgrades or rewrites that change those native observations intentionally change IDs."
    ] = SOURCE_ID_STABILITY_SCOPE
    migrated_from_schema_version: Literal["1.0.0"] | None = None
    items: tuple[SourceItem, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        ids = [item.source_item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("source catalog item IDs must be unique")
        if any(item.document_id != self.document_id for item in self.items):
            raise ValueError("source catalog items must match the catalog document_id")
        if ids != sorted(ids):
            raise ValueError("source catalog items must be stored in source_item_id order")
        return self

    @property
    def sha256(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


SOURCE_CATALOG_SCHEMA = pa.schema(
    [
        pa.field("source_item_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("page_number", pa.int32(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("coordinate_frame", pa.string(), nullable=False),
        pa.field("x0", pa.float64(), nullable=False),
        pa.field("y0", pa.float64(), nullable=False),
        pa.field("x1", pa.float64(), nullable=False),
        pa.field("y1", pa.float64(), nullable=False),
        pa.field("text", pa.string()),
        pa.field("content_sha256", pa.string(), nullable=False),
        pa.field("style_sha256", pa.string(), nullable=False),
        pa.field("identity_sha256", pa.string(), nullable=False),
        pa.field("canonical_x0", pa.float64(), nullable=False),
        pa.field("canonical_y0", pa.float64(), nullable=False),
        pa.field("canonical_x1", pa.float64(), nullable=False),
        pa.field("canonical_y1", pa.float64(), nullable=False),
        pa.field("duplicate_index", pa.int32(), nullable=False),
        pa.field("id_scheme", pa.string(), nullable=False),
    ],
    metadata={b"pdf2md.source_catalog_schema_version": SOURCE_CATALOG_SCHEMA_VERSION.encode()},
)

_LEGACY_SOURCE_CATALOG_SCHEMA = pa.schema(
    [
        pa.field("source_item_id", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("page_number", pa.int32(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("coordinate_frame", pa.string(), nullable=False),
        pa.field("x0", pa.float64(), nullable=False),
        pa.field("y0", pa.float64(), nullable=False),
        pa.field("x1", pa.float64(), nullable=False),
        pa.field("y1", pa.float64(), nullable=False),
        pa.field("text", pa.string()),
        pa.field("content_sha256", pa.string(), nullable=False),
    ],
    metadata={b"pdf2md.source_catalog_schema_version": LEGACY_SOURCE_CATALOG_SCHEMA_VERSION.encode()},
)


def make_source_item(
    *,
    source_item_id: str | None = None,
    document_id: str,
    page_number: int,
    kind: SourceItemKind,
    coordinate_frame: CoordinateFrame,
    coordinates: tuple[float, float, float, float],
    text: str | None,
    canonical_coordinates: tuple[float, float, float, float] | None = None,
    style: object | None = None,
    duplicate_index: int = 1,
) -> SourceItem:
    canonical = canonical_coordinates or coordinates
    identity_sha256, content_sha256, style_sha256 = source_item_identity_sha256(
        document_id=document_id,
        page_number=page_number,
        kind=kind,
        canonical_coordinates=canonical,
        text=text,
        style=style,
    )
    generated_id = content_addressed_source_item_id(
        document_id=document_id,
        page_number=page_number,
        kind=kind,
        canonical_coordinates=canonical,
        text=text,
        style=style,
        duplicate_index=duplicate_index,
    )
    actual_id = source_item_id or generated_id
    id_scheme: IdScheme = "native-content-v1" if actual_id == generated_id else "legacy-ordinal-v1"
    return SourceItem(
        source_item_id=actual_id,
        document_id=document_id,
        page_number=page_number,
        kind=kind,
        coordinate_frame=coordinate_frame,
        x0=coordinates[0],
        y0=coordinates[1],
        x1=coordinates[2],
        y1=coordinates[3],
        text=text,
        content_sha256=content_sha256,
        style_sha256=style_sha256,
        identity_sha256=identity_sha256,
        canonical_x0=canonical[0],
        canonical_y0=canonical[1],
        canonical_x1=canonical[2],
        canonical_y1=canonical[3],
        duplicate_index=duplicate_index,
        id_scheme=id_scheme,
    )


def build_source_catalog(document_id: str, items: list[SourceItem]) -> SourceCatalog:
    by_id: dict[str, SourceItem] = {}
    for item in items:
        previous = by_id.setdefault(item.source_item_id, item)
        if previous != item:
            raise ValueError(f"conflicting source catalog records for {item.source_item_id}")
    return SourceCatalog(document_id=document_id, items=tuple(by_id[key] for key in sorted(by_id)))


def reconstruct_native_word_text(words: Sequence[SourceItem]) -> str:
    """Reconstruct exact normalized text from canonical source-page word geometry."""
    if any(word.kind != "word" or word.coordinate_frame != "source_page" for word in words):
        raise ValueError("native table cell reconstruction requires source_page words")
    geometry_keys = [
        (
            word.page_number,
            word.canonical_y0,
            word.canonical_x0,
            word.canonical_y1,
            word.canonical_x1,
        )
        for word in words
    ]
    if len(geometry_keys) != len(set(geometry_keys)):
        raise ValueError("ambiguous native table cell word geometry")
    ordered: list[SourceItem] = []
    for page_number in sorted({word.page_number for word in words}):
        page_words = [word for word in words if word.page_number == page_number]
        lines: list[list[SourceItem]] = []
        for word in sorted(
            page_words,
            key=lambda item: (
                (item.canonical_y0 + item.canonical_y1) / 2,
                item.canonical_x0,
                item.source_item_id,
            ),
        ):
            word_height = word.canonical_y1 - word.canonical_y0
            candidates = []
            for line in lines:
                line_y0 = min(item.canonical_y0 for item in line)
                line_y1 = max(item.canonical_y1 for item in line)
                overlap = min(word.canonical_y1, line_y1) - max(word.canonical_y0, line_y0)
                if overlap > 0 and overlap / min(word_height, line_y1 - line_y0) >= 0.5:
                    candidates.append(line)
            if len(candidates) > 1:
                raise ValueError("ambiguous native table cell line assignment")
            if candidates:
                candidates[0].append(word)
            else:
                lines.append([word])
        lines.sort(
            key=lambda line: sum((item.canonical_y0 + item.canonical_y1) / 2 for item in line) / len(line)
        )
        for line in lines:
            x_centers = [(item.canonical_x0 + item.canonical_x1) / 2 for item in line]
            if len(x_centers) != len(set(x_centers)):
                raise ValueError("ambiguous native table cell horizontal word order")
            ordered.extend(
                sorted(
                    line,
                    key=lambda item: (
                        item.canonical_x0,
                        item.canonical_y0,
                        item.canonical_x1,
                        item.canonical_y1,
                        item.source_item_id,
                    ),
                )
            )
    return normalize_table_text(" ".join(word.text or "" for word in ordered))


def validate_table_cell_word_provenance(
    cell: TableCell,
    items_by_id: Mapping[str, SourceItem],
    *,
    document_id: str,
) -> tuple[str, ...]:
    """Validate and return exclusive cell word IDs after complete-text reconstruction."""
    source_ids = tuple(source_id for fragment in cell.fragments for source_id in fragment.source_item_ids)
    normalized_cell_text = normalize_table_text(cell.text)
    if not normalized_cell_text:
        if source_ids:
            raise ValueError("empty table cells cannot cite native source items")
        return ()
    if not source_ids:
        raise ValueError("nonempty table cells require native word source items")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("native table cell source item IDs must be unique")
    words: list[SourceItem] = []
    for fragment in cell.fragments:
        for source_id in fragment.source_item_ids:
            item = items_by_id.get(source_id)
            if item is None:
                raise ValueError(f"dangling table cell source_item_id: {source_id}")
            if item.kind != "word":
                raise ValueError(f"table cell content provenance must reference native words: {source_id}")
            if item.coordinate_frame != "source_page":
                raise ValueError(f"table cell content provenance must use source_page words: {source_id}")
            if item.document_id != document_id:
                raise ValueError(f"table cell source_item_id document mismatch: {source_id}")
            if item.page_number != fragment.page_number:
                raise ValueError(f"table cell source_item_id page mismatch: {source_id}")
            words.append(item)
    reconstructed = reconstruct_native_word_text(words)
    if reconstructed != normalized_cell_text:
        raise ValueError(
            "table cell native words do not reconstruct complete normalized cell text: "
            f"expected={normalized_cell_text!r}, actual={reconstructed!r}"
        )
    return source_ids


def validate_source_catalog(elements: list[DocumentElement], catalog: SourceCatalog) -> None:
    document_ids = {element.document_id for element in elements}
    if elements and document_ids != {catalog.document_id}:
        raise ValueError("elements and source catalog must have one matching document_id")
    items_by_id = {item.source_item_id: item for item in catalog.items}
    cell_references: list[str] = []
    for element in elements:
        table = element.structure.table
        if table is None:
            continue
        for cell in table.cells:
            cell_references.extend(
                validate_table_cell_word_provenance(
                    cell,
                    items_by_id,
                    document_id=element.document_id,
                )
            )
    if len(cell_references) != len(set(cell_references)):
        raise ValueError("a native word cannot be cited by more than one logical table cell")
    references = [
        (source_id, fragment.page_number)
        for element in elements
        for fragment in element.fragments
        for source_id in fragment.source_item_ids
    ]
    references.extend(
        (source_id, fragment.page_number)
        for element in elements
        if element.structure.table is not None
        for cell in element.structure.table.cells
        for fragment in cell.fragments
        for source_id in fragment.source_item_ids
    )
    for source_id, page_number in references:
        item = items_by_id.get(source_id)
        if item is None:
            raise ValueError(f"dangling source_item_id: {source_id}")
        if item.page_number != page_number:
            raise ValueError(f"source_item_id page mismatch: {source_id}")
    for element in elements:
        for prop in element.structure.properties:
            if prop.key != "table_structural_source_item_ids":
                continue
            if element.structure.table is None:
                raise ValueError("table structural source item IDs require a table consumer")
            raw_ids = json.loads(prop.value)
            if not isinstance(raw_ids, list) or any(not isinstance(value, str) for value in raw_ids):
                raise ValueError("table structural source item IDs must be a JSON string array")
            if raw_ids != sorted(set(raw_ids)):
                raise ValueError("table structural source item IDs must be sorted and unique")
            for source_id in raw_ids:
                item = items_by_id.get(source_id)
                if item is None:
                    raise ValueError(f"dangling structural source_item_id: {source_id}")
                if item.kind != "rule":
                    raise ValueError(f"table structural support must reference rules: {source_id}")
                if item.page_number not in {fragment.page_number for fragment in element.fragments}:
                    raise ValueError(f"structural source_item_id page mismatch: {source_id}")
    ledgers = _validate_table_span_suppression_ledgers(elements, items_by_id)
    if catalog.migrated_from_schema_version is None:
        _validate_current_candidate_table_accounting(elements, items_by_id, ledgers)


def _validate_table_span_suppression_ledgers(
    elements: Sequence[DocumentElement],
    items_by_id: Mapping[str, SourceItem],
) -> tuple[_TableSpanLedgerRecord, ...]:
    records: list[_TableSpanLedgerRecord] = []
    for element in elements:
        ledger_pages: set[int] = set()
        for prop in element.structure.properties:
            if prop.key != "table_span_suppression_v1":
                continue
            if element.structure.table is None:
                raise ValueError("table span suppression ledger requires a table consumer")
            payload = json.loads(prop.value)
            if not isinstance(payload, dict):
                raise ValueError("table span suppression ledger must be a JSON object")
            expected_keys = {
                "counts",
                "disposition",
                "page_number",
                "retained_overlap_span_ids",
                "supported_span_ids",
                "suppressed_span_ids",
                "table_id",
            }
            if set(payload) != expected_keys:
                raise ValueError("table span suppression ledger keys do not match v1")
            if payload["table_id"] != element.element_id:
                raise ValueError("table span suppression consumer table mismatch")
            page_number = payload["page_number"]
            if type(page_number) is not int or page_number < 1:
                raise ValueError("table span suppression page_number must be positive")
            if page_number in ledger_pages:
                raise ValueError("table span suppression ledgers must be unique by page")
            ledger_pages.add(page_number)
            if page_number not in {fragment.page_number for fragment in element.fragments}:
                raise ValueError("table span suppression ledger page mismatch")
            id_lists: list[list[str]] = []
            for key in (
                "supported_span_ids",
                "suppressed_span_ids",
                "retained_overlap_span_ids",
            ):
                values = payload[key]
                if not isinstance(values, list) or any(type(value) is not str for value in values):
                    raise ValueError(f"{key} must be a JSON string array")
                if values != sorted(set(values)):
                    raise ValueError(f"{key} must be sorted and unique")
                id_lists.append(values)
            flattened = [source_id for values in id_lists for source_id in values]
            if len(flattened) != len(set(flattened)):
                raise ValueError("table span suppression ledger ID lists must be disjoint")
            for source_id in flattened:
                item = items_by_id.get(source_id)
                if item is None:
                    raise ValueError(f"dangling table suppression span source_item_id: {source_id}")
                if item.kind != "span":
                    raise ValueError(f"table suppression ledger must reference spans: {source_id}")
                if item.coordinate_frame != "source_page":
                    raise ValueError(f"table suppression span coordinate frame mismatch: {source_id}")
                if item.document_id != element.document_id or item.page_number != page_number:
                    raise ValueError(f"table suppression span document/page mismatch: {source_id}")
            counts = payload["counts"]
            count_keys = {"retained_overlap", "supported", "suppressed"}
            if type(counts) is not dict or set(counts) != count_keys:
                raise ValueError(
                    "table span suppression counts must contain exactly non-negative integer values"
                )
            if any(type(value) is not int or value < 0 for value in counts.values()):
                raise ValueError(
                    "table span suppression counts must contain exactly non-negative integer values"
                )
            expected_counts = {
                "retained_overlap": len(id_lists[2]),
                "supported": len(id_lists[0]),
                "suppressed": len(id_lists[1]),
            }
            if counts != expected_counts:
                raise ValueError("table span suppression counts do not match ID lists")
            expected_disposition = (
                "suppressed_supported_spans"
                if id_lists[1]
                else "retained_ambiguous_support"
                if id_lists[0]
                else "retained_unsupported_overlap"
                if id_lists[2]
                else "no_overlap"
            )
            if payload["disposition"] != expected_disposition:
                raise ValueError("table span suppression disposition does not match ID lists")
            records.append(
                _TableSpanLedgerRecord(
                    table_id=element.element_id,
                    page_number=page_number,
                    supported_span_ids=tuple(id_lists[0]),
                    suppressed_span_ids=tuple(id_lists[1]),
                    retained_overlap_span_ids=tuple(id_lists[2]),
                )
            )
    return tuple(records)


def _items_intersect(first: SourceItem, second: SourceItem) -> bool:
    return min(first.x1, second.x1) > max(first.x0, second.x0) and min(first.y1, second.y1) > max(
        first.y0, second.y0
    )


def _item_intersects_fragment(item: SourceItem, fragment: PageFragment) -> bool:
    bbox = fragment.bbox
    return min(item.x1, bbox.x1) > max(item.x0, bbox.x0) and min(item.y1, bbox.y1) > max(item.y0, bbox.y0)


def _validate_current_candidate_table_accounting(
    elements: Sequence[DocumentElement],
    items_by_id: Mapping[str, SourceItem],
    ledgers: Sequence[_TableSpanLedgerRecord],
) -> None:
    ledgers_by_consumer = {(ledger.table_id, ledger.page_number): ledger for ledger in ledgers}
    candidate_tables = [
        element
        for element in elements
        if element.structure.table is not None and element.annotation.stage == "candidate"
    ]
    expected_consumers = {
        (element.element_id, fragment.page_number)
        for element in candidate_tables
        for fragment in element.fragments
    }
    actual_consumers = {
        (ledger.table_id, ledger.page_number)
        for ledger in ledgers
        if any(element.element_id == ledger.table_id for element in candidate_tables)
    }
    if actual_consumers != expected_consumers:
        raise ValueError(
            "current candidate tables require exactly one table_span_suppression_v1 ledger per page: "
            f"missing={sorted(expected_consumers - actual_consumers)}, "
            f"extra={sorted(actual_consumers - expected_consumers)}"
        )

    all_spans = [
        item
        for item in items_by_id.values()
        if item.kind == "span"
        and item.coordinate_frame == "source_page"
        and normalize_table_text(item.text or "")
    ]
    for table_element in candidate_tables:
        table = table_element.structure.table
        if table is None:
            raise AssertionError("candidate table selection included a non-table")
        for page_number in sorted({fragment.page_number for fragment in table_element.fragments}):
            page_fragments = [
                fragment for fragment in table_element.fragments if fragment.page_number == page_number
            ]
            words = [
                items_by_id[source_id]
                for cell in table.cells
                for fragment in cell.fragments
                if fragment.page_number == page_number
                for source_id in fragment.source_item_ids
            ]
            relevant = {
                span.source_item_id
                for span in all_spans
                if span.page_number == page_number and any(_items_intersect(span, word) for word in words)
            }
            ledger = ledgers_by_consumer[(table_element.element_id, page_number)]
            accounted = set(ledger.referenced_span_ids)
            fragment_supported = {
                span.source_item_id
                for span in all_spans
                if span.page_number == page_number
                and any(_item_intersects_fragment(span, fragment) for fragment in page_fragments)
            }
            missing = relevant - accounted
            extra = accounted - relevant - fragment_supported
            if missing or extra:
                raise ValueError(
                    "table span suppression ledger does not completely account for relevant native spans: "
                    f"table={table_element.element_id}, page={page_number}, "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )

    candidate_ids = {element.element_id for element in candidate_tables}
    candidate_ledgers = [ledger for ledger in ledgers if ledger.table_id in candidate_ids]
    disposition_ids = {
        "supported": {source_id for ledger in candidate_ledgers for source_id in ledger.supported_span_ids},
        "suppressed": {source_id for ledger in candidate_ledgers for source_id in ledger.suppressed_span_ids},
        "retained": {
            source_id for ledger in candidate_ledgers for source_id in ledger.retained_overlap_span_ids
        },
    }
    disposition_conflicts = {
        source_id
        for source_id in set().union(*disposition_ids.values())
        if sum(source_id in values for values in disposition_ids.values()) != 1
    }
    if disposition_conflicts:
        raise ValueError(
            "table spans must have exactly one supported/suppressed/retained disposition: "
            f"{sorted(disposition_conflicts)}"
        )
    suppressed_references = [
        source_id for ledger in candidate_ledgers for source_id in ledger.suppressed_span_ids
    ]
    other_references = {
        source_id
        for ledger in candidate_ledgers
        for source_id in (*ledger.supported_span_ids, *ledger.retained_overlap_span_ids)
    }
    double_consumed = {
        source_id
        for source_id, count in _value_counts(suppressed_references).items()
        if count > 1 or source_id in other_references
    }
    if double_consumed:
        raise ValueError(f"table suppression spans are double-consumed: {sorted(double_consumed)}")


def _value_counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def source_catalog_reference_counts(elements: Sequence[DocumentElement]) -> SourceCatalogReferenceCounts:
    fragment_references = [
        source_id
        for element in elements
        for fragment in element.fragments
        for source_id in fragment.source_item_ids
    ]
    fragment_references.extend(
        source_id
        for element in elements
        if element.structure.table is not None
        for cell in element.structure.table.cells
        for fragment in cell.fragments
        for source_id in fragment.source_item_ids
    )
    structural_references: list[str] = []
    ledger_references_by_key: dict[str, list[str]] = {
        "supported_span_ids": [],
        "suppressed_span_ids": [],
        "retained_overlap_span_ids": [],
    }
    ledger_count = 0
    for element in elements:
        for prop in element.structure.properties:
            if prop.key == "table_structural_source_item_ids":
                values = json.loads(prop.value)
                if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                    raise ValueError("table structural source item IDs must be a JSON string array")
                structural_references.extend(values)
            elif prop.key == "table_span_suppression_v1":
                payload = json.loads(prop.value)
                if not isinstance(payload, dict):
                    raise ValueError("table span suppression ledger must be a JSON object")
                ledger_count += 1
                for key in (
                    "supported_span_ids",
                    "suppressed_span_ids",
                    "retained_overlap_span_ids",
                ):
                    values = payload.get(key)
                    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                        raise ValueError(f"{key} must be a JSON string array")
                    ledger_references_by_key[key].extend(values)
    supported_references = ledger_references_by_key["supported_span_ids"]
    suppressed_references = ledger_references_by_key["suppressed_span_ids"]
    retained_references = ledger_references_by_key["retained_overlap_span_ids"]
    ledger_references = [*supported_references, *suppressed_references, *retained_references]
    all_references = [*fragment_references, *structural_references, *ledger_references]
    return SourceCatalogReferenceCounts(
        fragment_reference_count=len(fragment_references),
        unique_fragment_source_item_count=len(set(fragment_references)),
        table_structural_reference_count=len(structural_references),
        unique_table_structural_source_item_count=len(set(structural_references)),
        table_suppression_ledger_count=ledger_count,
        table_supported_span_reference_count=len(supported_references),
        unique_table_supported_span_count=len(set(supported_references)),
        table_suppressed_span_reference_count=len(suppressed_references),
        unique_table_suppressed_span_count=len(set(suppressed_references)),
        table_retained_overlap_span_reference_count=len(retained_references),
        unique_table_retained_overlap_span_count=len(set(retained_references)),
        table_suppression_span_reference_count=len(ledger_references),
        unique_table_suppression_span_count=len(set(ledger_references)),
        total_reference_count=len(all_references),
        total_unique_source_item_count=len(set(all_references)),
    )


def source_catalog_path(document_path: Path) -> Path:
    if document_path.suffix != ".parquet":
        raise ValueError("document path must use the .parquet suffix")
    return document_path.with_name(f"{document_path.stem}.source-items.parquet")


def write_source_catalog(catalog: SourceCatalog, output_path: Path, *, overwrite: bool = False) -> None:
    if output_path.suffix != ".parquet":
        raise ValueError("source catalog output path must use the .parquet suffix")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = SOURCE_CATALOG_SCHEMA.with_metadata({
        **(SOURCE_CATALOG_SCHEMA.metadata or {}),
        b"pdf2md.document_id": catalog.document_id.encode(),
        b"pdf2md.source_catalog_sha256": catalog.sha256.encode(),
    })
    records = [item.model_dump(mode="json") for item in catalog.items]
    pq.write_table(pa.Table.from_pylist(records, schema=schema), output_path, compression="zstd")


def write_document_with_source_catalog(
    elements: list[DocumentElement],
    catalog: SourceCatalog,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    validate_source_catalog(elements, catalog)
    catalog_path = source_catalog_path(output_path)
    if not overwrite:
        existing = [path for path in (output_path, catalog_path) if path.exists()]
        if existing:
            raise FileExistsError(existing[0])
    write_source_catalog(catalog, catalog_path, overwrite=overwrite)
    write_document_elements(elements, output_path, overwrite=overwrite)
    return catalog_path


def read_document_with_source_catalog(input_path: Path) -> tuple[list[DocumentElement], SourceCatalog]:
    from app.pdf2md.schema import read_document_elements

    elements = read_document_elements(input_path)
    catalog = read_source_catalog(source_catalog_path(input_path))
    validate_source_catalog(elements, catalog)
    return elements, catalog


def _legacy_catalog_sha256(document_id: str, records: Sequence[Mapping[str, object]]) -> str:
    payload = {
        "schema_version": LEGACY_SOURCE_CATALOG_SCHEMA_VERSION,
        "document_id": document_id,
        "extractor": "pymupdf",
        "items": records,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _migrate_legacy_catalog(document_id: str, records: Sequence[Mapping[str, object]]) -> SourceCatalog:
    empty_style_sha256 = hashlib.sha256(b"null").hexdigest()
    items = []
    for record in records:
        coordinates = tuple(record[name] for name in ("x0", "y0", "x1", "y1"))
        source_item_id = str(record["source_item_id"])
        items.append(
            SourceItem.model_validate({
                **record,
                "style_sha256": empty_style_sha256,
                "identity_sha256": hashlib.sha256(source_item_id.encode()).hexdigest(),
                "canonical_x0": coordinates[0],
                "canonical_y0": coordinates[1],
                "canonical_x1": coordinates[2],
                "canonical_y1": coordinates[3],
                "duplicate_index": 1,
                "id_scheme": "legacy-ordinal-v1",
            })
        )
    return SourceCatalog(
        document_id=document_id,
        migrated_from_schema_version=LEGACY_SOURCE_CATALOG_SCHEMA_VERSION,
        items=tuple(items),
    )


def read_source_catalog(input_path: Path) -> SourceCatalog:
    stored_schema = pq.read_schema(input_path)
    metadata = stored_schema.metadata or {}
    expected_metadata_keys = {
        b"pdf2md.source_catalog_schema_version",
        b"pdf2md.document_id",
        b"pdf2md.source_catalog_sha256",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError("source catalog metadata does not exactly match the catalog contract")
    version = metadata.get(b"pdf2md.source_catalog_schema_version")
    if version not in {
        SOURCE_CATALOG_SCHEMA_VERSION.encode(),
        LEGACY_SOURCE_CATALOG_SCHEMA_VERSION.encode(),
    }:
        raise ValueError("unsupported or missing source catalog schema version")
    base_schema = (
        SOURCE_CATALOG_SCHEMA
        if version == SOURCE_CATALOG_SCHEMA_VERSION.encode()
        else _LEGACY_SOURCE_CATALOG_SCHEMA
    )
    if not stored_schema.equals(base_schema.with_metadata(metadata), check_metadata=False):
        raise ValueError("source catalog physical schema does not exactly match its declared schema")
    raw_document_id = metadata.get(b"pdf2md.document_id")
    stored_hash = metadata.get(b"pdf2md.source_catalog_sha256")
    if raw_document_id is None or stored_hash is None:
        raise ValueError("source catalog metadata is incomplete")
    document_id = raw_document_id.decode()
    records = pq.read_table(input_path).to_pylist()
    if version == LEGACY_SOURCE_CATALOG_SCHEMA_VERSION.encode():
        if _legacy_catalog_sha256(document_id, records).encode() != stored_hash:
            raise ValueError("source catalog hash mismatch")
        return _migrate_legacy_catalog(document_id, records)
    items = tuple(SourceItem.model_validate(record) for record in records)
    catalog = SourceCatalog(document_id=document_id, items=items)
    if catalog.sha256.encode() != stored_hash:
        raise ValueError("source catalog hash mismatch")
    return catalog
