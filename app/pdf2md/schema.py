from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

SCHEMA_VERSION = "1.2.0"
PREVIOUS_SCHEMA_VERSION = "1.1.0"
LEGACY_SCHEMA_VERSION = "1.0.0"
ContentFormat = Literal["text", "markdown", "html"]
AnnotationStage = Literal["candidate", "silver", "golden"]
TableRepresentation = Literal["markdown", "html"]
CellRole = Literal["header", "row_header", "body"]
_ELEMENT_TYPES = frozenset({
    "paragraph",
    "heading",
    "caption",
    "footnote",
    "code",
    "table",
    "figure",
    "header",
    "footer",
    "note",
})
_LEGACY_ROLE_MAP = {
    "statement_title": "heading",
    "language_marker": "running_header",
    "annex_label": "heading",
    "annex_title": "heading",
    "lead_in": "body",
    "eli_footer": "running_footer",
    "code_block": "code",
    "section_heading": "heading",
    "figure_banner_title": "heading",
    "figure_panel_subtitle": "figure_text",
    "figure_axis_unit": "figure_text",
    "figure_axis_tick_labels": "figure_text",
    "figure_series_label": "figure_text",
    "figure_callout": "figure_text",
    "figure_legend": "figure_text",
    "figure_axis_category_labels": "figure_text",
    "figure_axis_group_label": "figure_text",
    "figure_axis_label": "figure_text",
    "figure_caption": "caption",
    "section_tab": "running_header",
    "part_heading": "heading",
    "form_checkbox": "body",
    "instruction": "body",
    "chapter_label": "heading",
    "chapter_title": "heading",
    "table_caption": "caption",
    "label": "body",
}
_PARAGRAPH_ROLES_BY_TYPE: dict[str, frozenset[str]] = {
    "paragraph": frozenset({
        "body",
        "subtitle",
        "list_item",
        "list_item_continuation",
        "figure_text",
    }),
    "heading": frozenset({"heading", "figure_panel_heading"}),
    "caption": frozenset({"caption"}),
    "footnote": frozenset({"footnote"}),
    "code": frozenset({"code"}),
    "header": frozenset({"running_header", "page_number"}),
    "footer": frozenset({"running_footer", "page_number"}),
    "note": frozenset({"table_continuation_marker"}),
}


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BoundingBox(SchemaModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("bounding box coordinates must be finite")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive width and height")
        return self


class PageFragment(SchemaModel):
    page_number: int = Field(ge=1)
    page_width: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    page_height: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    bbox: BoundingBox
    source_item_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provenance(self) -> Self:
        if (self.page_width is None) != (self.page_height is None):
            raise ValueError("page_width and page_height must either both be present or both be null")
        if any(not source_item_id for source_item_id in self.source_item_ids):
            raise ValueError("source_item_ids cannot contain empty values")
        if len(self.source_item_ids) != len(set(self.source_item_ids)):
            raise ValueError("source_item_ids must be unique within a fragment")
        return self


class ParagraphStructure(SchemaModel):
    role: str
    heading_level: int | None = Field(default=None, ge=1, le=6)
    list_depth: int | None = Field(default=None, ge=0)
    list_label: str | None = None
    title_evidence: list[str] = Field(default_factory=list)
    heading_evidence: list[str] = Field(default_factory=list)
    list_evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_semantic_evidence(self) -> Self:
        evidence_groups = (self.title_evidence, self.heading_evidence, self.list_evidence)
        if any(not evidence for group in evidence_groups for evidence in group):
            raise ValueError("paragraph evidence values cannot be empty")
        if any(len(group) != len(set(group)) for group in evidence_groups):
            raise ValueError("paragraph evidence values must be unique")
        if self.title_evidence and self.heading_level != 1:
            raise ValueError("title evidence requires heading level one")
        if self.heading_evidence and self.heading_level is None:
            raise ValueError("heading evidence requires a heading level")
        if (self.list_depth is None) != (self.list_label is None):
            raise ValueError("list depth and label must be provided together")
        if self.role == "list_item":
            if self.list_depth is None or self.list_label is None:
                raise ValueError("list_item requires list depth and label")
        elif self.list_depth is not None or self.list_label is not None:
            raise ValueError("list metadata is only valid for list_item")
        if self.list_evidence and self.role != "list_item":
            raise ValueError("list evidence requires the list_item role")
        return self


class TextStyleRun(SchemaModel):
    """A half-open character range over the element's unchanged plain content."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    font_family: str | None = None
    font_size: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("style run end must be greater than start")
        if self.font_family == "":
            raise ValueError("style run font_family cannot be empty")
        return self


class TableCell(SchemaModel):
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    role: CellRole
    text: str
    fragments: list[PageFragment] = Field(min_length=1)


class TableStructure(SchemaModel):
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    header_row_count: int = Field(ge=0)
    representation: TableRepresentation
    classification_reasons: list[str] = Field(default_factory=list)
    cells: list[TableCell] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        if self.header_row_count > self.row_count:
            raise ValueError("header_row_count cannot exceed row_count")

        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if cell.row_index + cell.rowspan > self.row_count:
                raise ValueError("cell rowspan exceeds table row_count")
            if cell.column_index + cell.colspan > self.column_count:
                raise ValueError("cell colspan exceeds table column_count")
            positions = {
                (row, column)
                for row in range(cell.row_index, cell.row_index + cell.rowspan)
                for column in range(cell.column_index, cell.column_index + cell.colspan)
            }
            if occupied.intersection(positions):
                raise ValueError("table cells overlap")
            if cell.row_index < self.header_row_count and cell.role != "header":
                raise ValueError("cells starting in header rows require the header role")
            occupied.update(positions)

        if self.representation == "markdown":
            expected = {(row, column) for row in range(self.row_count) for column in range(self.column_count)}
            if self.header_row_count != 1:
                raise ValueError("Markdown tables require exactly one header row")
            if any(cell.rowspan != 1 or cell.colspan != 1 for cell in self.cells):
                raise ValueError("Markdown tables cannot contain spanning cells")
            if occupied != expected:
                raise ValueError("Markdown tables require a complete rectangular grid")
            if self.classification_reasons:
                raise ValueError("Markdown tables cannot have HTML classification reasons")
        elif not self.classification_reasons:
            raise ValueError("HTML tables require at least one classification reason")

        return self


class FigureStructure(SchemaModel):
    asset_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    caption_element_id: str | None = None


_SAFE_FOOTNOTE_LABEL_RE = re.compile(r"[^\s\[\]\^\\]+")


def _is_safe_footnote_label(label: object) -> bool:
    return bool(
        isinstance(label, str)
        and label.isprintable()
        and _SAFE_FOOTNOTE_LABEL_RE.fullmatch(label) is not None
    )


class FootnoteStructure(SchemaModel):
    label: str | None = None
    reference_element_ids: list[str] = Field(default_factory=list)
    association_confident: bool

    @model_validator(mode="after")
    def validate_association(self) -> Self:
        if self.label is not None and not _is_safe_footnote_label(self.label):
            raise ValueError("footnote label is unsafe for generated Markdown syntax")
        if any(not reference_id for reference_id in self.reference_element_ids):
            raise ValueError("footnote reference_element_ids cannot contain empty values")
        if len(self.reference_element_ids) != len(set(self.reference_element_ids)):
            raise ValueError("footnote reference_element_ids must be unique")
        if self.association_confident:
            if self.label is None:
                raise ValueError("a confident footnote association requires a label")
            if not self.reference_element_ids:
                raise ValueError("a confident footnote association requires references")
        elif self.reference_element_ids:
            raise ValueError("footnote references require a confident association")
        return self


class StructureProperty(SchemaModel):
    key: str
    value: str


class InlineFootnoteMarkerRange(SchemaModel):
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("inline footnote marker end must be greater than start")
        if not _is_safe_footnote_label(self.label):
            raise ValueError("inline footnote marker label is unsafe for generated Markdown syntax")
        return self


class ElementStructure(SchemaModel):
    paragraph: ParagraphStructure | None = None
    table: TableStructure | None = None
    figure: FigureStructure | None = None
    footnote: FootnoteStructure | None = None
    style_runs: list[TextStyleRun] = Field(default_factory=list)
    linked_element_ids: list[str] = Field(default_factory=list)
    properties: list[StructureProperty] = Field(default_factory=list)


def _json_object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key in inline footnote marker metadata: {key!r}")
        result[key] = value
    return result


def _parse_inline_footnote_marker_ranges(
    content: str,
    properties: Sequence[StructureProperty],
) -> tuple[InlineFootnoteMarkerRange, ...]:
    marker_properties = [prop for prop in properties if prop.key == "inline_footnote_markers"]
    if not marker_properties:
        return ()
    if len(marker_properties) != 1:
        raise ValueError("an element cannot contain multiple inline_footnote_markers properties")
    try:
        payload = json.loads(
            marker_properties[0].value,
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError("inline footnote marker metadata must be valid JSON") from error
    if not isinstance(payload, list):
        raise ValueError("inline footnote marker metadata must be a JSON array")
    if all(isinstance(item, str) for item in payload):
        return ()
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError("inline footnote marker metadata must contain only range objects")
    ranges = tuple(InlineFootnoteMarkerRange.model_validate(item) for item in payload)
    if list(ranges) != sorted(ranges, key=lambda item: (item.start, item.end, item.label)):
        raise ValueError("inline footnote marker ranges must be stored in source order")
    previous_end = 0
    for marker in ranges:
        if marker.start < previous_end:
            raise ValueError("inline footnote marker ranges must not overlap")
        if marker.end > len(content):
            raise ValueError("inline footnote marker range exceeds element content")
        if content[marker.start : marker.end] != f"[^{marker.label}]":
            raise ValueError("inline footnote marker range does not match generated syntax")
        previous_end = marker.end
    return ranges


class AnnotationMetadata(SchemaModel):
    stage: AnnotationStage
    revision: int = Field(ge=1)
    annotator: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    parent_revision: int | None = Field(default=None, ge=1)
    adjudication_status: Literal["unreviewed", "accepted", "corrected", "human_reviewed"]

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        if self.revision == 1 and self.parent_revision is not None:
            raise ValueError("the first annotation revision cannot have a parent_revision")
        if self.parent_revision is not None and self.parent_revision >= self.revision:
            raise ValueError("parent_revision must be less than revision")
        return self


class DocumentElement(SchemaModel):
    schema_version: Literal["1.2.0"] = SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    element_type: str = Field(min_length=1)
    content: str
    format: ContentFormat
    include_in_output: bool = True
    fragments: list[PageFragment] = Field(min_length=1)
    structure: ElementStructure
    annotation: AnnotationMetadata

    @model_validator(mode="after")
    def validate_element_structure(self) -> Self:
        if self.element_type not in _ELEMENT_TYPES:
            raise ValueError(f"unsupported element_type: {self.element_type}")
        table = self.structure.table
        paragraph = self.structure.paragraph
        if self.element_type == "table":
            if paragraph is not None:
                raise ValueError("table elements cannot carry paragraph structure")
            if self.format not in {"markdown", "html"}:
                raise ValueError("table elements require markdown or html format")
            if table is None:
                raise ValueError("table elements require table structure")
            if self.format != table.representation:
                raise ValueError("table format must match its selected representation")
        elif table is not None:
            raise ValueError("table structure is only valid for table elements")
        elif self.format != "text":
            raise ValueError("non-table elements require text format")

        figure = self.structure.figure
        if self.element_type == "figure":
            if figure is None:
                raise ValueError("figure elements require figure structure")
            if paragraph is not None:
                raise ValueError("figure elements cannot carry paragraph structure")
        elif figure is not None:
            raise ValueError("figure structure is only valid for figure elements")

        if self.element_type not in {"table", "figure"}:
            if paragraph is None:
                raise ValueError("non-table, non-figure elements require paragraph structure")
            allowed_roles = _PARAGRAPH_ROLES_BY_TYPE[self.element_type]
            if paragraph.role not in allowed_roles:
                raise ValueError(f"paragraph role {paragraph.role!r} is invalid for {self.element_type!r}")
            if self.element_type == "heading":
                if paragraph.heading_level is None:
                    raise ValueError("heading elements require a heading level")
            elif paragraph.heading_level is not None:
                raise ValueError("heading levels are only valid for heading elements")
            if paragraph.title_evidence and (self.element_type != "heading" or paragraph.role != "heading"):
                raise ValueError("title evidence is only valid for document headings")

        footnote = self.structure.footnote
        if self.element_type == "footnote":
            if footnote is None:
                raise ValueError("footnote elements require footnote structure")
        elif footnote is not None:
            raise ValueError("footnote structure is only valid for footnote elements")

        style_runs = self.structure.style_runs
        if table is not None and style_runs:
            raise ValueError("table elements cannot carry text style runs")
        previous_end = 0
        for run in style_runs:
            if run.end > len(self.content):
                raise ValueError("style run exceeds element content length")
            if run.start < previous_end:
                raise ValueError("style runs must be sorted and non-overlapping")
            previous_end = run.end

        if table is not None:
            element_fragments_by_page: dict[int, list[PageFragment]] = {}
            for fragment in self.fragments:
                element_fragments_by_page.setdefault(fragment.page_number, []).append(fragment)
            for cell in table.cells:
                for fragment in cell.fragments:
                    page_fragments = element_fragments_by_page.get(fragment.page_number)
                    if page_fragments is None:
                        raise ValueError(
                            "table cell fragments must belong to a page represented by the table element"
                        )
                    element_dimensions = {
                        (page_fragment.page_width, page_fragment.page_height)
                        for page_fragment in page_fragments
                        if page_fragment.page_width is not None
                    }
                    if fragment.page_width is not None and element_dimensions != {
                        (fragment.page_width, fragment.page_height)
                    }:
                        raise ValueError("table cell fragment page dimensions must match the table element")

        _parse_inline_footnote_marker_ranges(self.content, self.structure.properties)
        return self


def inline_footnote_marker_ranges(
    element: DocumentElement,
) -> tuple[InlineFootnoteMarkerRange, ...]:
    """Return exact generated marker ranges; legacy label-only metadata claims nothing."""
    return _parse_inline_footnote_marker_ranges(element.content, element.structure.properties)


def inline_footnote_marker_property(
    ranges: Sequence[InlineFootnoteMarkerRange],
) -> StructureProperty:
    ordered = sorted(ranges, key=lambda item: (item.start, item.end, item.label))
    if list(ranges) != ordered:
        raise ValueError("inline footnote marker ranges must be supplied in source order")
    if any(current.start < previous.end for previous, current in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("inline footnote marker ranges must not overlap")
    return StructureProperty(
        key="inline_footnote_markers",
        value=json.dumps(
            [item.model_dump(mode="json") for item in ordered],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


_BBOX_TYPE = pa.struct([
    pa.field("x0", pa.float64(), nullable=False),
    pa.field("y0", pa.float64(), nullable=False),
    pa.field("x1", pa.float64(), nullable=False),
    pa.field("y1", pa.float64(), nullable=False),
])
_FRAGMENT_TYPE = pa.struct([
    pa.field("page_number", pa.int32(), nullable=False),
    pa.field("page_width", pa.float64()),
    pa.field("page_height", pa.float64()),
    pa.field("bbox", _BBOX_TYPE, nullable=False),
    pa.field("source_item_ids", pa.list_(pa.string()), nullable=False),
])
_PARAGRAPH_TYPE = pa.struct([
    pa.field("role", pa.string(), nullable=False),
    pa.field("heading_level", pa.int8()),
    pa.field("list_depth", pa.int16()),
    pa.field("list_label", pa.string()),
    pa.field("title_evidence", pa.list_(pa.string()), nullable=False),
    pa.field("heading_evidence", pa.list_(pa.string()), nullable=False),
    pa.field("list_evidence", pa.list_(pa.string()), nullable=False),
])
_STYLE_RUN_TYPE = pa.struct([
    pa.field("start", pa.int32(), nullable=False),
    pa.field("end", pa.int32(), nullable=False),
    pa.field("font_family", pa.string()),
    pa.field("font_size", pa.float64()),
    pa.field("bold", pa.bool_(), nullable=False),
    pa.field("italic", pa.bool_(), nullable=False),
    pa.field("underline", pa.bool_(), nullable=False),
    pa.field("strikeout", pa.bool_(), nullable=False),
])
_CELL_TYPE = pa.struct([
    pa.field("row_index", pa.int32(), nullable=False),
    pa.field("column_index", pa.int32(), nullable=False),
    pa.field("rowspan", pa.int32(), nullable=False),
    pa.field("colspan", pa.int32(), nullable=False),
    pa.field("role", pa.string(), nullable=False),
    pa.field("text", pa.string(), nullable=False),
    pa.field("fragments", pa.list_(_FRAGMENT_TYPE), nullable=False),
])
_TABLE_TYPE = pa.struct([
    pa.field("row_count", pa.int32(), nullable=False),
    pa.field("column_count", pa.int32(), nullable=False),
    pa.field("header_row_count", pa.int32(), nullable=False),
    pa.field("representation", pa.string(), nullable=False),
    pa.field("classification_reasons", pa.list_(pa.string()), nullable=False),
    pa.field("cells", pa.list_(_CELL_TYPE), nullable=False),
])
_FIGURE_TYPE = pa.struct([
    pa.field("asset_path", pa.string(), nullable=False),
    pa.field("sha256", pa.string(), nullable=False),
    pa.field("width", pa.int32(), nullable=False),
    pa.field("height", pa.int32(), nullable=False),
    pa.field("caption_element_id", pa.string()),
])
_FOOTNOTE_TYPE = pa.struct([
    pa.field("label", pa.string()),
    pa.field("reference_element_ids", pa.list_(pa.string()), nullable=False),
    pa.field("association_confident", pa.bool_(), nullable=False),
])
_STRUCTURE_TYPE = pa.struct([
    pa.field("paragraph", _PARAGRAPH_TYPE),
    pa.field("table", _TABLE_TYPE),
    pa.field("figure", _FIGURE_TYPE),
    pa.field("footnote", _FOOTNOTE_TYPE),
    pa.field("style_runs", pa.list_(_STYLE_RUN_TYPE), nullable=False),
    pa.field("linked_element_ids", pa.list_(pa.string()), nullable=False),
    pa.field(
        "properties",
        pa.list_(
            pa.struct([
                pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.string(), nullable=False),
            ])
        ),
        nullable=False,
    ),
])
_ANNOTATION_TYPE = pa.struct([
    pa.field("stage", pa.string(), nullable=False),
    pa.field("revision", pa.int32(), nullable=False),
    pa.field("annotator", pa.string(), nullable=False),
    pa.field("confidence", pa.float64(), nullable=False),
    pa.field("parent_revision", pa.int32()),
    pa.field("adjudication_status", pa.string(), nullable=False),
])
DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("element_id", pa.string(), nullable=False),
        pa.field("order", pa.int32(), nullable=False),
        pa.field("element_type", pa.string(), nullable=False),
        pa.field("content", pa.string(), nullable=False),
        pa.field("format", pa.string(), nullable=False),
        pa.field("include_in_output", pa.bool_(), nullable=False),
        pa.field("fragments", pa.list_(_FRAGMENT_TYPE), nullable=False),
        pa.field("structure", _STRUCTURE_TYPE, nullable=False),
        pa.field("annotation", _ANNOTATION_TYPE, nullable=False),
    ],
    metadata={b"pdf2md.schema_version": SCHEMA_VERSION.encode()},
)

PREVIOUS_DOCUMENT_SCHEMA = DOCUMENT_SCHEMA.with_metadata({
    b"pdf2md.schema_version": PREVIOUS_SCHEMA_VERSION.encode()
})

_LEGACY_PARAGRAPH_TYPE = pa.struct(list(_PARAGRAPH_TYPE)[:4])
_LEGACY_STRUCTURE_TYPE = pa.struct([
    pa.field("paragraph", _LEGACY_PARAGRAPH_TYPE),
    pa.field("table", _TABLE_TYPE),
    pa.field("figure", _FIGURE_TYPE),
    pa.field("footnote", _FOOTNOTE_TYPE),
    pa.field("linked_element_ids", pa.list_(pa.string()), nullable=False),
    pa.field(
        "properties",
        pa.list_(
            pa.struct([
                pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.string(), nullable=False),
            ])
        ),
        nullable=False,
    ),
])
LEGACY_DOCUMENT_SCHEMA = pa.schema(
    [
        field if field.name != "structure" else pa.field("structure", _LEGACY_STRUCTURE_TYPE, nullable=False)
        for field in DOCUMENT_SCHEMA
    ],
    metadata={b"pdf2md.schema_version": LEGACY_SCHEMA_VERSION.encode()},
)


def _reject_undeclared_model_attributes(value: object) -> None:
    """Reject unchecked model_copy additions while ignoring Pydantic private state."""
    if isinstance(value, BaseModel):
        declared_fields = set(type(value).model_fields)
        undeclared = (set(value.__dict__) | value.model_fields_set) - declared_fields
        if value.__pydantic_extra__:
            undeclared.update(value.__pydantic_extra__)
        if undeclared:
            names = ", ".join(sorted(undeclared))
            raise ValueError(f"model contains undeclared model attributes: {names}")
        for field_name in declared_fields:
            _reject_undeclared_model_attributes(getattr(value, field_name))
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_undeclared_model_attributes(item)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _reject_undeclared_model_attributes(item)


def _validate_document_collection(elements: list[DocumentElement], *, require_canonical_order: bool) -> None:
    if not elements:
        raise ValueError("a document Parquet file must contain at least one element")

    document_ids = {element.document_id for element in elements}
    if len(document_ids) != 1:
        raise ValueError("one Parquet file must contain exactly one document_id")
    element_ids = [element.element_id for element in elements]
    if len(element_ids) != len(set(element_ids)):
        raise ValueError("element_id values must be unique within a document")
    orders = [element.order for element in elements]
    if sorted(orders) != list(range(len(elements))):
        raise ValueError("element order must be unique and contiguous from zero")
    if require_canonical_order and orders != list(range(len(elements))):
        raise ValueError("Parquet rows must be stored in canonical element order")
    stages = {element.annotation.stage for element in elements}
    if len(stages) != 1:
        raise ValueError("one document Parquet file must contain exactly one annotation stage")


def write_document_elements(
    elements: list[DocumentElement], output_path: Path, *, overwrite: bool = False
) -> None:
    if output_path.suffix != ".parquet":
        raise ValueError("document output path must use the .parquet suffix")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    for element in elements:
        _reject_undeclared_model_attributes(element)
    validated_elements = [
        DocumentElement.model_validate(element.model_dump(mode="json")) for element in elements
    ]
    _validate_document_collection(validated_elements, require_canonical_order=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        element.model_dump(mode="json") for element in sorted(validated_elements, key=lambda item: item.order)
    ]
    table = pa.Table.from_pylist(records, schema=DOCUMENT_SCHEMA)
    pq.write_table(table, output_path, compression="zstd")


def _migrate_legacy_inline_marker_properties(record: dict[str, object]) -> None:
    """Drop only legacy marker claims that cannot satisfy the current exact-range contract."""
    structure = record["structure"]
    if not isinstance(structure, dict):
        raise TypeError("legacy document structure must be a mapping")
    properties = structure.get("properties")
    if not isinstance(properties, list):
        raise TypeError("legacy document structure properties must be a list")

    marker_records = [
        prop for prop in properties if isinstance(prop, dict) and prop.get("key") == "inline_footnote_markers"
    ]
    if not marker_records:
        return
    content = record.get("content")
    if not isinstance(content, str):
        raise TypeError("legacy document content must be a string")
    try:
        marker_properties = [StructureProperty.model_validate(prop) for prop in marker_records]
        _parse_inline_footnote_marker_ranges(content, marker_properties)
    except ValueError:
        structure["properties"] = [
            prop
            for prop in properties
            if not (isinstance(prop, dict) and prop.get("key") == "inline_footnote_markers")
        ]


def _migrate_legacy_footnote_structure(record: dict[str, object]) -> None:
    """Conservatively repair pre-1.2 footnote states before strict validation."""
    structure = record["structure"]
    if not isinstance(structure, dict):
        raise TypeError("legacy document structure must be a mapping")
    footnote = structure.get("footnote")
    if footnote is None:
        return
    if not isinstance(footnote, dict):
        raise TypeError("legacy footnote structure must be a mapping")

    label = footnote.get("label")
    raw_reference_ids = footnote.get("reference_element_ids")
    if not isinstance(raw_reference_ids, list):
        raise TypeError("legacy footnote reference_element_ids must be a list")
    reference_ids = list(
        dict.fromkeys(
            reference_id
            for reference_id in raw_reference_ids
            if isinstance(reference_id, str) and reference_id
        )
    )
    confident = footnote.get("association_confident") is True
    if not _is_safe_footnote_label(label) or not confident or not reference_ids:
        footnote.update(label=None if not _is_safe_footnote_label(label) else label)
        footnote.update(reference_element_ids=[], association_confident=False)
        return
    footnote.update(reference_element_ids=reference_ids, association_confident=True)


def read_document_elements(input_path: Path) -> list[DocumentElement]:
    if input_path.suffix != ".parquet":
        raise ValueError("document input path must use the .parquet suffix")
    stored_schema = pq.read_schema(input_path)
    stored_version = stored_schema.metadata.get(b"pdf2md.schema_version") if stored_schema.metadata else None
    schemas_by_version = {
        SCHEMA_VERSION.encode(): DOCUMENT_SCHEMA,
        PREVIOUS_SCHEMA_VERSION.encode(): PREVIOUS_DOCUMENT_SCHEMA,
        LEGACY_SCHEMA_VERSION.encode(): LEGACY_DOCUMENT_SCHEMA,
    }
    expected_schema = None if stored_version is None else schemas_by_version.get(stored_version)
    if expected_schema is None:
        raise ValueError(
            f"unsupported or missing Parquet schema version: {stored_version!r}; "
            f"supported={sorted(version.decode() for version in schemas_by_version)}"
        )
    if stored_schema.metadata != expected_schema.metadata:
        raise ValueError("Parquet metadata does not exactly match its declared DOCUMENT_SCHEMA")
    if not stored_schema.equals(expected_schema, check_metadata=False):
        raise ValueError("Parquet physical schema does not exactly match its declared DOCUMENT_SCHEMA")

    records = pq.read_table(input_path).to_pylist()
    if stored_version in {PREVIOUS_SCHEMA_VERSION.encode(), LEGACY_SCHEMA_VERSION.encode()}:
        for record in records:
            record["schema_version"] = SCHEMA_VERSION
            _migrate_legacy_inline_marker_properties(record)
            _migrate_legacy_footnote_structure(record)
            if stored_version == LEGACY_SCHEMA_VERSION.encode():
                paragraph = record["structure"]["paragraph"]
                if paragraph is not None:
                    paragraph.update(title_evidence=[], heading_evidence=[], list_evidence=[])
                record["structure"]["style_runs"] = []
            paragraph = record["structure"]["paragraph"]
            if paragraph is None and record["element_type"] == "footnote":
                paragraph = {
                    "role": "footnote",
                    "heading_level": None,
                    "list_depth": None,
                    "list_label": None,
                    "title_evidence": [],
                    "heading_evidence": [],
                    "list_evidence": [],
                }
                record["structure"]["paragraph"] = paragraph
            if paragraph is not None:
                role = paragraph["role"]
                if role in {"continuation_notice", "continuation_caption"}:
                    paragraph["role"] = "table_continuation_marker"
                    record["element_type"] = "note"
                    record["include_in_output"] = False
                elif role in _LEGACY_ROLE_MAP:
                    paragraph["role"] = _LEGACY_ROLE_MAP[role]
    elements = [DocumentElement.model_validate(record) for record in records]
    _validate_document_collection(elements, require_canonical_order=True)
    return elements
