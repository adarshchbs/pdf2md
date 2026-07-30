from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

SCHEMA_VERSION = "1.0"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
type BoundingBox = tuple[float, float, float, float]
type ElementType = Literal[
    "paragraph",
    "heading",
    "list_item",
    "table",
    "figure",
    "caption",
    "footnote",
    "header",
    "footer",
    "other",
]


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PageSize(CanonicalModel):
    width: float = Field(gt=0, allow_inf_nan=False)
    height: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["point", "pixel"]


class CanonicalCell(CanonicalModel):
    id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    text: str
    bbox: BoundingBox | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BoundingBox | None) -> BoundingBox | None:
        return _validate_bbox(bbox)


class CanonicalTable(CanonicalModel):
    id: str = Field(min_length=1)
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    cells: list[CanonicalCell] = Field(min_length=1)
    bbox: BoundingBox | None = None
    markdown: str | None = None
    html: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BoundingBox | None) -> BoundingBox | None:
        return _validate_bbox(bbox)

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        if (self.markdown is None) == (self.html is None):
            raise ValueError("a table must have exactly one of markdown or html")
        cell_ids = [cell.id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("cell ids must be unique within a table")

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
            occupied.update(positions)
        return self


class CanonicalFigure(CanonicalModel):
    id: str = Field(min_length=1)
    bbox: BoundingBox | None = None
    caption: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BoundingBox | None) -> BoundingBox | None:
        return _validate_bbox(bbox)


class CanonicalElement(CanonicalModel):
    id: str = Field(min_length=1)
    element_type: ElementType
    reading_order: int = Field(ge=0)
    text: str
    markdown: str
    bbox: BoundingBox | None = None
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    include_in_output: bool = True
    table_id: str | None = Field(default=None, min_length=1)
    figure_id: str | None = Field(default=None, min_length=1)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: BoundingBox | None) -> BoundingBox | None:
        return _validate_bbox(bbox)

    @model_validator(mode="after")
    def validate_typed_reference(self) -> Self:
        if (self.element_type == "table") != (self.table_id is not None):
            raise ValueError("exactly table elements require table_id")
        if (self.element_type == "figure") != (self.figure_id is not None):
            raise ValueError("exactly figure elements require figure_id")
        return self


class RuntimeStats(CanonicalModel):
    wall_seconds: float = Field(ge=0, allow_inf_nan=False)
    peak_rss_bytes: int | None = Field(default=None, ge=0)


class CanonicalProvenance(CanonicalModel):
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    config_sha256: str = Field(pattern=_SHA256_PATTERN)


class CanonicalPage(CanonicalModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    document_id: str = Field(min_length=1)
    page_index: int = Field(ge=0)
    page_size: PageSize
    bbox_origin: Literal["top-left"] = "top-left"
    markdown: str
    elements: list[CanonicalElement]
    tables: list[CanonicalTable]
    figures: list[CanonicalFigure]
    runtime: RuntimeStats
    provenance: CanonicalProvenance
    output_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        for label, bbox in (
            *[(f"element {element.id!r}", element.bbox) for element in self.elements],
            *[(f"table {table.id!r}", table.bbox) for table in self.tables],
            *[
                (f"cell {cell.id!r} in table {table.id!r}", cell.bbox)
                for table in self.tables
                for cell in table.cells
            ],
            *[(f"figure {figure.id!r}", figure.bbox) for figure in self.figures],
        ):
            _validate_page_bbox(label, bbox, self.page_size)

        _require_unique_ids("element", [element.id for element in self.elements])
        _require_unique_ids("table", [table.id for table in self.tables])
        _require_unique_ids("figure", [figure.id for figure in self.figures])

        reading_order = [element.reading_order for element in self.elements]
        if sorted(reading_order) != list(range(len(self.elements))):
            raise ValueError("element reading_order must be unique and contiguous from zero")
        if reading_order != list(range(len(self.elements))):
            raise ValueError("elements must be stored in reading order")

        table_ids = {table.id for table in self.tables}
        table_elements = {
            element.table_id: element for element in self.elements if element.table_id is not None
        }
        referenced_table_ids = [element.table_id for element in self.elements if element.table_id is not None]
        if set(referenced_table_ids) != table_ids or len(referenced_table_ids) != len(table_ids):
            raise ValueError("every table must be referenced by exactly one table element")
        for table in self.tables:
            if table.markdown is not None:
                representation = table.markdown
            elif table.html is not None:
                representation = table.html
            else:
                raise ValueError("a table must contain a representation")
            table_element = table_elements[table.id]
            if table_element.markdown != representation or representation not in self.markdown:
                raise ValueError("table representation must be retained in its element and page markdown")

        figure_ids = {figure.id for figure in self.figures}
        referenced_figure_ids = [
            element.figure_id for element in self.elements if element.figure_id is not None
        ]
        if set(referenced_figure_ids) != figure_ids or len(referenced_figure_ids) != len(figure_ids):
            raise ValueError("every figure must be referenced by exactly one figure element")
        return self


class AdapterRun(CanonicalModel):
    status: Literal["success", "failed"]
    pages: list[CanonicalPage]
    raw_records: list[JsonValue]
    runtime: RuntimeStats
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "success":
            if self.error is not None:
                raise ValueError("a successful adapter run cannot contain an error")
            validate_page_collection(self.pages)
        elif self.error is None:
            raise ValueError("a failed adapter run requires an error")
        return self


def validate_page_collection(pages: list[CanonicalPage], *, require_canonical_order: bool = True) -> None:
    if not pages:
        raise ValueError("a canonical page collection must contain at least one page")

    keys = [(page.document_id, page.page_index) for page in pages]
    if len(keys) != len(set(keys)):
        raise ValueError("(document_id, page_index) keys must be unique")

    for document_id in {page.document_id for page in pages}:
        document_pages = [page for page in pages if page.document_id == document_id]
        indexes = sorted(page.page_index for page in document_pages)
        if indexes != list(range(len(indexes))):
            raise ValueError(f"page indexes for {document_id!r} must be contiguous from zero")
        provenance = document_pages[0].provenance
        if any(page.provenance != provenance for page in document_pages[1:]):
            raise ValueError(f"provenance for {document_id!r} must be consistent across pages")

    canonical_keys = sorted(keys)
    if require_canonical_order and keys != canonical_keys:
        raise ValueError("pages must be stored in canonical document and page order")


def canonical_json(value: CanonicalModel | JsonValue) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def write_pages_jsonl(pages: list[CanonicalPage], output_path: Path, *, overwrite: bool = False) -> None:
    if output_path.suffix != ".jsonl":
        raise ValueError("canonical output path must use the .jsonl suffix")
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    validate_page_collection(pages, require_canonical_order=False)

    ordered_pages = sorted(pages, key=lambda page: (page.document_id, page.page_index))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{canonical_json(page)}\n" for page in ordered_pages),
        encoding="utf-8",
        newline="\n",
    )


def read_pages_jsonl(input_path: Path) -> list[CanonicalPage]:
    if input_path.suffix != ".jsonl":
        raise ValueError("canonical input path must use the .jsonl suffix")

    pages: list[CanonicalPage] = []
    with input_path.open(encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"JSONL line {line_number} is not newline-terminated")
            if not line.strip():
                raise ValueError(f"JSONL line {line_number} is blank")
            pages.append(CanonicalPage.model_validate_json(line))
    validate_page_collection(pages)
    return pages


def semantic_hash(pages: list[CanonicalPage]) -> str:
    validate_page_collection(pages, require_canonical_order=False)
    normalized = []
    for page in sorted(pages, key=lambda item: (item.document_id, item.page_index)):
        record = page.model_dump(mode="json", exclude={"runtime", "output_sha256"})
        normalized.append(record)
    payload = "".join(f"{canonical_json(record)}\n" for record in normalized).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_bbox(bbox: BoundingBox | None) -> BoundingBox | None:
    if bbox is None:
        return None
    if not all(math.isfinite(coordinate) for coordinate in bbox):
        raise ValueError("bounding box coordinates must be finite")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bounding box must have positive area")
    return bbox


def _validate_page_bbox(label: str, bbox: BoundingBox | None, page_size: PageSize) -> None:
    if bbox is None:
        return
    tolerance = 1e-6
    x0, y0, x1, y1 = bbox
    if (
        x0 < -tolerance
        or y0 < -tolerance
        or x1 > page_size.width + tolerance
        or y1 > page_size.height + tolerance
    ):
        raise ValueError(f"{label} bounding box lies outside the page")


def _require_unique_ids(kind: str, ids: list[str]) -> None:
    if len(ids) != len(set(ids)):
        raise ValueError(f"{kind} ids must be unique within a page")
