from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import pymupdf
import pytest

from app.pdf2md.pymupdf_table_detection import (
    DetectedTable,
    _native_word_tokens,  # pyright: ignore[reportPrivateUsage]
    detect_page_tables,
)
from app.pdf2md.pymupdf_table_detection import (
    _horizontal_rules as _detected_horizontal_rules,  # pyright: ignore[reportPrivateUsage]
)
from app.pdf2md.pymupdf_tables import _append_table_source_items  # pyright: ignore[reportPrivateUsage]
from app.pdf2md.source_catalog import SourceItem, content_addressed_source_item_id
from app.pdf2md.table_provenance import (
    CoordinateFrame,
    FinderProvenance,
    FinderSnapshot,
    GeometricBand,
    GeometricBandKind,
    NativeRuleId,
    NativeToken,
    NativeTokenId,
    TableDiagnosticOutcome,
    TableDiagnosticStage,
    TableProvenance,
    reconstruct_table,
)


@dataclass
class FakeTable:
    bbox: tuple[float, float, float, float]
    name: str
    rows: list[list[str | None]] = field(default_factory=lambda: [["Name", "Value"], ["A", "1"]])

    def extract(self) -> list[list[str | None]]:
        return self.rows


@dataclass
class FakeHeader:
    external: bool
    names: Sequence[str | None]


@dataclass
class FakeGeometryRow:
    cells: Sequence[tuple[float, float, float, float] | None]


@dataclass
class GeometryFakeTable:
    bbox: tuple[float, float, float, float]
    name: str
    extracted_rows: list[list[str | None]]
    geometry_rows: list[FakeGeometryRow]

    @property
    def cells(self) -> list[tuple[float, float, float, float]]:
        return [cell for row in self.geometry_rows for cell in row.cells if cell is not None]

    @property
    def rows(self) -> list[FakeGeometryRow]:
        return self.geometry_rows

    @property
    def header(self) -> FakeHeader:
        return FakeHeader(external=False, names=self.extracted_rows[0])

    def extract(self) -> list[list[str | None]]:
        return self.extracted_rows


@dataclass
class FakeFinder:
    tables: Sequence[DetectedTable]


@dataclass
class FakeRect:
    width: float = 600.0
    height: float = 800.0


class FakePage:
    def __init__(
        self,
        *,
        default: Sequence[DetectedTable] = (),
        strict: Sequence[DetectedTable] = (),
        clipped_text: Sequence[DetectedTable] = (),
        clipped_mixed: Sequence[DetectedTable] = (),
        drawings: Sequence[Mapping[str, object]] = (),
        blocks: Sequence[Sequence[object]] = (),
        words: Sequence[Sequence[object]] = (),
        width: float = 600.0,
        height: float = 800.0,
        rotation_matrix: pymupdf.Matrix | None = None,
    ) -> None:
        self.rect = FakeRect(width=width, height=height)
        self.rotation_matrix = rotation_matrix or pymupdf.Matrix(1, 0, 0, 1, 0, 0)
        self.default = default
        self.strict = strict
        self.clipped_text = clipped_text
        self.clipped_mixed = clipped_mixed
        self.drawings = drawings
        self.blocks = blocks
        self.words = words
        self.calls: list[dict[str, object]] = []

    def find_tables(self, **kwargs: object) -> FakeFinder:
        self.calls.append(kwargs)
        if not kwargs:
            return FakeFinder(self.default)
        if "clip" not in kwargs:
            return FakeFinder(self.strict)
        if kwargs["horizontal_strategy"] == "text":
            return FakeFinder(self.clipped_text)
        return FakeFinder(self.clipped_mixed)

    def get_drawings(self) -> list[Mapping[str, object]]:
        return list(self.drawings)

    def get_text(self, option: str, *, sort: bool) -> Sequence[Sequence[object]]:
        assert sort is True
        if option == "words":
            return self.words
        assert option == "blocks"
        return self.blocks


def _detect(page: FakePage) -> list[DetectedTable]:
    return detect_page_tables(cast(pymupdf.Page, page))


def _assert_snapshot(
    tables: list[DetectedTable],
    expected: FakeTable,
    finder: FinderProvenance,
) -> DetectedTable:
    assert len(tables) == 1
    snapshot = tables[0]
    assert tuple(snapshot.bbox) == expected.bbox
    assert snapshot.extract() == expected.extract()
    assert getattr(snapshot, "finder_provenance") == finder
    return snapshot


def _horizontal_rules(scale: float = 1.0) -> list[Mapping[str, object]]:
    return [
        {
            "rect": (100.0 * scale, 200.0 * scale, 500.0 * scale, 240.0 * scale),
            "items": [
                ("l", (100.0 * scale, 200.0 * scale), (500.0 * scale, 200.0 * scale)),
                ("l", (100.0 * scale, 220.0 * scale), (500.0 * scale, 220.0 * scale)),
                ("l", (100.0 * scale, 240.0 * scale), (500.0 * scale, 240.0 * scale)),
            ],
        }
    ]


def _fragmented_grid_rules() -> list[Mapping[str, object]]:
    levels = (100.0, 110.0, 120.0, 140.0, 160.0, 180.0, 200.0, 210.0, 220.0)
    return [
        {
            "rect": (100.0, levels[0], 500.0, levels[-1]),
            "color": (0.0, 0.0, 0.0),
            "items": [
                *(("l", (100.0, y), (500.0, y)) for y in levels),
                *(("l", (x, levels[0]), (x, levels[-1])) for x in (200.0, 300.0, 400.0)),
            ],
        }
    ]


@pytest.mark.parametrize(
    "rectangle",
    [
        (100.0, 200.0, 100.0, 300.0),
        (100.0, 200.0, 500.0, 200.0),
    ],
)
def test_degenerate_rectangle_items_do_not_abort_detection(
    rectangle: tuple[float, float, float, float],
) -> None:
    table = FakeTable((100.0, 200.0, 500.0, 300.0), "table")
    page = FakePage(
        default=[table],
        drawings=[
            {
                "rect": rectangle,
                "color": (0.0, 0.0, 0.0),
                "items": [("re", rectangle, 1)],
            }
        ],
    )

    detected = _detect(page)

    assert len(detected) == 1
    assert detected[0].bbox == table.bbox
    assert detected[0].extract() == table.extract()


def test_degenerate_native_word_bbox_remains_invalid() -> None:
    class WordPage:
        number = 0
        rotation_matrix = pymupdf.Matrix(1, 0, 0, 1, 0, 0)

        def get_text(self, option: str, *, sort: bool) -> list[tuple[object, ...]]:
            assert option == "words" and sort
            return [(10.0, 20.0, 10.0, 40.0, "invalid")]

    with pytest.raises(ValueError, match="native word bbox must have positive dimensions"):
        _native_word_tokens(cast(pymupdf.Page, WordPage()))


def test_native_ids_are_document_scoped_duplicate_stable_and_input_permutation_invariant() -> None:
    class WordPage:
        number = 0
        rotation_matrix = pymupdf.Matrix(1, 0, 0, 1, 0, 0)

        def __init__(self, words: Sequence[tuple[object, ...]]) -> None:
            self.words = list(words)

        def get_text(self, option: str, *, sort: bool) -> list[tuple[object, ...]]:
            assert option == "words" and sort
            return self.words

    words = [
        (10.0, 20.0, 30.0, 40.0, "same", 2, 1, 1),
        (10.0, 20.0, 30.0, 40.0, "same", 1, 1, 1),
        (50.0, 20.0, 70.0, 40.0, "other", 1, 1, 2),
    ]
    document_id = "a" * 64
    forward = _native_word_tokens(cast(pymupdf.Page, WordPage(words)), document_id=document_id)
    reversed_input = _native_word_tokens(
        cast(pymupdf.Page, WordPage(list(reversed(words)))), document_id=document_id
    )
    other_document = _native_word_tokens(cast(pymupdf.Page, WordPage(words)), document_id="b" * 64)

    forward_ids = {token.token_id.value for token in forward}
    assert forward_ids == {token.token_id.value for token in reversed_input}
    duplicate_ids = sorted(token.token_id.value for token in forward if token.text == "same")
    assert duplicate_ids[0].endswith(":d000001")
    assert duplicate_ids[1].endswith(":d000002")
    assert all(value.startswith(f"pymupdf:d{document_id}:p000001:word:") for value in forward_ids)
    assert forward_ids.isdisjoint({token.token_id.value for token in other_document})


def test_overprinted_in_table_word_never_downgrades_document_scoped_ids() -> None:
    document_id = "d" * 64
    duplicate = (140.0, 230.0, 160.0, 250.0, "7", 1, 1, 1)
    page = FakePage(
        default=[FakeTable((100.0, 200.0, 500.0, 300.0), "table")],
        words=[duplicate, duplicate],
    )

    detected = detect_page_tables(cast(pymupdf.Page, page), document_id=document_id)

    assert len(detected) == 1
    evidence = getattr(detected[0], "provenance", None)
    assert isinstance(evidence, TableProvenance)
    assert [token.duplicate_index for token in evidence.native_tokens] == [1, 2]
    source_items: list[SourceItem] = []
    _append_table_source_items(
        source_items,
        evidence,
        document_id=document_id,
        page_number=1,
    )
    assert [item.duplicate_index for item in source_items if item.kind == "word"] == [1, 2]
    assert all(item.id_scheme == "native-content-v1" for item in source_items)
    assert all(item.source_item_id.startswith(f"pymupdf:d{document_id}:p000001:") for item in source_items)


def test_table_word_source_items_use_canonical_source_page_geometry() -> None:
    document_id = "f" * 64
    canonical_bbox = (10.0, 20.0, 30.0, 40.0)
    token_id = content_addressed_source_item_id(
        document_id=document_id,
        page_number=1,
        kind="word",
        canonical_coordinates=canonical_bbox,
        text="value",
    )
    token = NativeToken(
        token_id=NativeTokenId(token_id),
        frame=CoordinateFrame.DETECTOR_PAGE,
        bbox=(260.0, 10.0, 280.0, 30.0),
        canonical_bbox=canonical_bbox,
        baseline=30,
        text="value",
    )
    evidence = TableProvenance(
        finder=FinderProvenance.DEFAULT,
        frame=CoordinateFrame.DETECTOR_PAGE,
        native_token_ids=(token.token_id,),
        native_tokens=(token,),
    )
    source_items: list[SourceItem] = []

    _append_table_source_items(
        source_items,
        evidence,
        document_id=document_id,
        page_number=1,
    )

    assert len(source_items) == 1
    assert source_items[0].coordinate_frame == "source_page"
    assert (source_items[0].x0, source_items[0].y0, source_items[0].x1, source_items[0].y1) == canonical_bbox


def test_duplicate_native_line_rules_preserve_multiset_ids_and_drawing_permutation() -> None:
    document_id = "e" * 64
    duplicate_line = ("l", (120.0, 240.0), (480.0, 240.0))
    drawings = [
        {
            "rect": (120.0, 240.0, 480.0, 260.0),
            "color": (0.0, 0.0, 0.0),
            "items": [duplicate_line, ("l", (120.0, 260.0), (480.0, 260.0))],
        },
        {
            "rect": (120.0, 240.0, 480.0, 280.0),
            "color": (0.0, 0.0, 0.0),
            "items": [duplicate_line, ("l", (120.0, 280.0), (480.0, 280.0))],
        },
    ]

    def evidence_for(source_drawings: Sequence[Mapping[str, object]]) -> TableProvenance:
        page = FakePage(
            default=[FakeTable((100.0, 200.0, 500.0, 300.0), "table")],
            drawings=source_drawings,
        )
        detected = detect_page_tables(cast(pymupdf.Page, page), document_id=document_id)
        assert len(detected) == 1
        evidence = getattr(detected[0], "provenance", None)
        assert isinstance(evidence, TableProvenance)
        return evidence

    forward = evidence_for(drawings)
    reversed_order = evidence_for(list(reversed(drawings)))
    duplicate_rules = [
        rule for rule in forward.native_rules if rule.canonical_geometry == (120.0, 240.0, 480.0, 240.0)
    ]

    assert [rule.duplicate_index for rule in duplicate_rules] == [1, 2]
    assert [rule.rule_id.value.rsplit(":", 1)[-1] for rule in duplicate_rules] == ["d000001", "d000002"]
    assert {rule.rule_id for rule in forward.native_rules} == {
        rule.rule_id for rule in reversed_order.native_rules
    }
    source_items: list[SourceItem] = []
    _append_table_source_items(
        source_items,
        forward,
        document_id=document_id,
        page_number=1,
    )
    duplicate_items = [
        item
        for item in source_items
        if item.kind == "rule"
        and (item.canonical_x0, item.canonical_y0, item.canonical_x1, item.canonical_y1)
        == (120.0, 240.0, 480.0, 240.0)
    ]
    assert [item.duplicate_index for item in duplicate_items] == [1, 2]
    assert all(item.id_scheme == "native-content-v1" for item in duplicate_items)
    assert [item.source_item_id.rsplit(":", 1)[-1] for item in duplicate_items] == [
        "d000001",
        "d000002",
    ]


def test_one_zero_height_rectangle_is_one_native_rule_and_one_topology_rule() -> None:
    document_id = "f" * 64
    rectangle_item = ("re", (120.0, 240.0, 480.0, 240.0), 1)
    drawings = [
        {
            "rect": (120.0, 240.0, 480.0, 240.0),
            "color": (0.0, 0.0, 0.0),
            "items": [rectangle_item],
        }
    ]
    page = FakePage(
        default=[FakeTable((100.0, 200.0, 500.0, 300.0), "table")],
        drawings=drawings,
    )

    detected = detect_page_tables(cast(pymupdf.Page, page), document_id=document_id)

    assert len(detected) == 1
    evidence = getattr(detected[0], "provenance", None)
    assert isinstance(evidence, TableProvenance)
    assert len(evidence.native_rules) == 1
    assert evidence.native_rules[0].duplicate_index == 1
    topology_rules = _detected_horizontal_rules(drawings, 600.0, 800.0, 1)
    assert len(topology_rules) == 1
    source_items: list[SourceItem] = []
    _append_table_source_items(
        source_items,
        evidence,
        document_id=document_id,
        page_number=1,
    )
    rule_items = [item for item in source_items if item.kind == "rule"]
    assert len(rule_items) == 1
    assert rule_items[0].id_scheme == "native-content-v1"


def test_separate_identical_rectangle_items_remain_duplicate_observations() -> None:
    document_id = "1" * 64
    rectangle_item = ("re", (120.0, 240.0, 480.0, 240.0), 1)
    drawings = [
        {
            "rect": (120.0, 240.0, 480.0, 260.0),
            "color": (0.0, 0.0, 0.0),
            "items": [rectangle_item, ("l", (120.0, 260.0), (480.0, 260.0))],
        },
        {
            "rect": (120.0, 240.0, 480.0, 280.0),
            "color": (0.0, 0.0, 0.0),
            "items": [rectangle_item, ("l", (120.0, 280.0), (480.0, 280.0))],
        },
    ]

    def evidence_for(source_drawings: Sequence[Mapping[str, object]]) -> TableProvenance:
        page = FakePage(
            default=[FakeTable((100.0, 200.0, 500.0, 300.0), "table")],
            drawings=source_drawings,
        )
        detected = detect_page_tables(cast(pymupdf.Page, page), document_id=document_id)
        assert len(detected) == 1
        evidence = getattr(detected[0], "provenance", None)
        assert isinstance(evidence, TableProvenance)
        return evidence

    forward = evidence_for(drawings)
    reversed_order = evidence_for(list(reversed(drawings)))
    duplicate_rules = [
        rule for rule in forward.native_rules if rule.canonical_geometry == (120.0, 240.0, 480.0, 240.0)
    ]

    assert [rule.duplicate_index for rule in duplicate_rules] == [1, 2]
    assert [rule.rule_id.value.rsplit(":", 1)[-1] for rule in duplicate_rules] == ["d000001", "d000002"]
    assert {rule.rule_id for rule in forward.native_rules} == {
        rule.rule_id for rule in reversed_order.native_rules
    }
    source_items: list[SourceItem] = []
    _append_table_source_items(
        source_items,
        forward,
        document_id=document_id,
        page_number=1,
    )
    duplicate_items = [
        item
        for item in source_items
        if item.kind == "rule"
        and (item.canonical_x0, item.canonical_y0, item.canonical_x1, item.canonical_y1)
        == (120.0, 240.0, 480.0, 240.0)
    ]
    assert [item.duplicate_index for item in duplicate_items] == [1, 2]
    assert all(item.id_scheme == "native-content-v1" for item in duplicate_items)


def test_native_ids_use_rotation_zero_geometry_across_quarter_turn_detection() -> None:
    document_id = "c" * 64
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 30), "Alpha")
    page.draw_line((10, 50), (190, 50))
    words_by_rotation: list[set[str]] = []
    for rotation in (0, 90, 180, 270):
        page.set_rotation(rotation)
        words_by_rotation.append({
            token.token_id.value for token in _native_word_tokens(page, document_id=document_id)
        })
    document.close()

    assert all(ids == words_by_rotation[0] for ids in words_by_rotation)


def test_always_collects_lines_strict_but_defaults_remain_authoritative() -> None:
    default = FakeTable((100.0, 400.0, 500.0, 500.0), "default")
    strict_duplicate = FakeTable((100.4, 400.4, 500.4, 500.4), "strict-duplicate")
    strict_only = FakeTable((100.0, 100.0, 500.0, 180.0), "strict-only")
    page = FakePage(default=[default], strict=[strict_duplicate, strict_only])

    diagnostics = []
    tables = detect_page_tables(cast(pymupdf.Page, page), diagnostics=diagnostics)

    _assert_snapshot(tables, default, FinderProvenance.DEFAULT)
    strict_disposition = next(
        record
        for record in diagnostics
        if record.stage == TableDiagnosticStage.BOUNDARY_DETECTION
        and record.reason == "unused_lines_strict_candidates"
    )
    assert strict_disposition.outcome == TableDiagnosticOutcome.REJECTED
    assert strict_disposition.metric("input_count") == 2
    assert strict_disposition.metric("output_count") == 0
    assert page.calls == [
        {},
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
    ]


def test_strong_rule_band_runs_consensus_recovery() -> None:
    text_table = FakeTable((100.0, 200.0, 500.0, 240.0), "text")
    mixed_duplicate = FakeTable((100.0, 200.0, 500.0, 240.0), "mixed")
    page = FakePage(
        drawings=_horizontal_rules(),
        blocks=[(100.0, 170.0, 500.0, 190.0, "Table 1. Results")],
        clipped_text=[text_table],
        clipped_mixed=[mixed_duplicate],
    )

    tables = _detect(page)

    snapshot = _assert_snapshot(tables, text_table, FinderProvenance.RULE_BAND_TEXT)
    bands = getattr(snapshot, "geometric_bands")
    assert [(band.kind, band.band_id) for band in bands] == [(GeometricBandKind.RULE_BAND, "rule-band-0001")]
    assert [rule_id.value for rule_id in bands[0].native_rule_ids] == [
        "pymupdf:p000001:rule:r000001",
        "pymupdf:p000001:rule:r000002",
        "pymupdf:p000001:rule:r000003",
    ]
    assert len(page.calls) == 4
    text_call, mixed_call = page.calls[2:]
    assert text_call["vertical_strategy"] == "text"
    assert text_call["horizontal_strategy"] == "text"
    assert mixed_call["vertical_strategy"] == "text"
    assert mixed_call["horizontal_strategy"] == "lines_strict"
    assert text_call["clip"] == mixed_call["clip"]


def test_rule_band_thresholds_and_clip_margins_scale_with_page() -> None:
    normal = FakePage(
        drawings=_horizontal_rules(),
        blocks=[(100.0, 170.0, 500.0, 190.0, "Table 1. Results")],
    )
    doubled = FakePage(
        drawings=_horizontal_rules(2.0),
        blocks=[(200.0, 340.0, 1000.0, 380.0, "Table 1. Results")],
        width=1200.0,
        height=1600.0,
    )

    _detect(normal)
    _detect(doubled)

    normal_clip = cast(tuple[float, float, float, float], normal.calls[2]["clip"])
    doubled_clip = cast(tuple[float, float, float, float], doubled.calls[2]["clip"])
    assert doubled_clip == tuple(value * 2 for value in normal_clip)


def test_rule_band_without_table_caption_does_not_run_recovery() -> None:
    page = FakePage(drawings=_horizontal_rules())

    assert _detect(page) == []
    assert page.calls == [
        {},
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
        {"vertical_strategy": "text", "horizontal_strategy": "text"},
    ]


def test_incoherent_clipped_candidate_is_not_recovered() -> None:
    incoherent = FakeTable(
        (100.0, 200.0, 500.0, 240.0),
        "incoherent",
        rows=[["Title", None], [None, None], [None, None], [None, None]],
    )
    mixed = FakeTable((100.0, 200.0, 500.0, 240.0), "mixed")
    page = FakePage(
        drawings=_horizontal_rules(),
        blocks=[(100.0, 170.0, 500.0, 190.0, "Table 1. Results")],
        clipped_text=[incoherent],
        clipped_mixed=[mixed],
    )
    diagnostics = []

    assert detect_page_tables(cast(pymupdf.Page, page), diagnostics=diagnostics) == []
    assert any(
        record.stage == TableDiagnosticStage.BOUNDARY_FILTER
        and record.outcome == TableDiagnosticOutcome.REJECTED
        and record.reason == "recovery_consensus_failed"
        for record in diagnostics
    )


def test_recovery_overlapping_authoritative_default_is_not_accumulated() -> None:
    default = FakeTable((100.0, 200.0, 500.0, 240.0), "default")
    text = FakeTable((100.0, 200.0, 500.0, 240.0), "text")
    mixed = FakeTable((100.0, 200.0, 500.0, 240.0), "mixed")
    page = FakePage(
        default=[default],
        drawings=_horizontal_rules(),
        blocks=[(100.0, 170.0, 500.0, 190.0, "Table 1. Results")],
        clipped_text=[text],
        clipped_mixed=[mixed],
    )

    diagnostics = []
    _assert_snapshot(
        detect_page_tables(cast(pymupdf.Page, page), diagnostics=diagnostics),
        default,
        FinderProvenance.DEFAULT,
    )
    assert any(
        record.stage == TableDiagnosticStage.BOUNDARY_FILTER
        and record.outcome == TableDiagnosticOutcome.REJECTED
        and record.reason == "overlaps_authoritative_boundary"
        and record.metric("input_count") == 1
        and record.metric("output_count") == 0
        for record in diagnostics
    )


def test_default_container_replaces_contained_fragment() -> None:
    fragment = FakeTable((150.0, 150.0, 250.0, 250.0), "fragment")
    container = FakeTable((100.0, 100.0, 500.0, 500.0), "container")
    page = FakePage(default=[fragment, container])

    _assert_snapshot(_detect(page), container, FinderProvenance.DEFAULT)


@pytest.mark.parametrize("reverse_candidates", [False, True])
def test_preserves_exact_three_legitimate_captionless_aligned_same_cardinality_tables(
    reverse_candidates: bool,
) -> None:
    """Identical visual grids do not prove that three independently emitted tables are fragments."""
    top = FakeTable(
        (100.0, 100.0, 500.0, 110.0),
        "quarterly-summary",
        rows=[["Region", "Q1", "Q2", "Q3"]],
    )
    core = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "operating-results",
        rows=[
            ["Division", "Q1", "Q2", "Q3"],
            ["North", "2", "3", "3"],
            ["South", "2", "4", "4"],
            ["West", "2", "6", "6"],
        ],
    )
    bottom = FakeTable(
        (100.0, 210.0, 500.0, 220.0),
        "forecast-summary",
        rows=[["Scenario", "Low", "Base", "High"]],
    )
    candidates = [top, core, bottom]
    if reverse_candidates:
        candidates.reverse()
    page = FakePage(default=candidates, drawings=_fragmented_grid_rules())

    assert [table.extract() for table in _detect(page)] == [top.extract(), core.extract(), bottom.extract()]


def test_preserves_exact_three_legitimate_tables_when_single_caption_belongs_to_top_table() -> None:
    top = FakeTable(
        (100.0, 100.0, 500.0, 110.0),
        "captioned-summary",
        rows=[["Region", "Q1", "Q2", "Q3"]],
    )
    core = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "operating-results",
        rows=[
            ["Division", "Q1", "Q2", "Q3"],
            ["North", "2", "3", "3"],
            ["South", "2", "4", "4"],
            ["West", "2", "6", "6"],
        ],
    )
    bottom = FakeTable(
        (100.0, 210.0, 500.0, 220.0),
        "forecast-summary",
        rows=[["Scenario", "Low", "Base", "High"]],
    )
    page = FakePage(
        default=[bottom, core, top],
        drawings=_fragmented_grid_rules(),
        blocks=[(100.0, 75.0, 500.0, 95.0, "Table 8. Quarterly summary")],
    )

    assert [table.extract() for table in _detect(page)] == [top.extract(), core.extract(), bottom.extract()]


def test_preserves_paired_same_cardinality_rows_when_vertical_grids_disagree() -> None:
    top = FakeTable((100.0, 100.0, 500.0, 110.0), "top", rows=[["A", "B", "C", "D"]])
    core = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "core",
        rows=[["A", "B", "C", "D"], ["1", "2", "3", "4"], ["5", "6", "7", "8"]],
    )
    bottom = FakeTable((100.0, 210.0, 500.0, 220.0), "bottom", rows=[["A", "B", "C", "D"]])
    horizontal = [("l", (100.0, y), (500.0, y)) for y in (100.0, 110.0, 120.0, 160.0, 200.0, 210.0, 220.0)]
    vertical = [
        *(("l", (x, 100.0), (x, 200.0)) for x in (200.0, 300.0, 400.0)),
        *(("l", (x, 210.0), (x, 220.0)) for x in (250.0, 350.0)),
    ]
    page = FakePage(
        default=[top, core, bottom],
        drawings=[
            {
                "rect": (100.0, 100.0, 500.0, 220.0),
                "color": (0.0, 0.0, 0.0),
                "items": [*horizontal, *vertical],
            }
        ],
    )

    assert [tuple(table.bbox) for table in _detect(page)] == [top.bbox, core.bbox, bottom.bbox]


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_detaches_shallow_sentence_banner_before_independently_ruled_grid(scale: float) -> None:
    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (
            x0 * scale,
            y0 * scale,
            x1 * scale,
            y1 * scale,
        )

    geometry = [
        FakeGeometryRow((box(100, 100, 500, 110), None, None, None)),
        FakeGeometryRow(tuple(box(x, 110, x + 100, 130) for x in (100, 200, 300, 400))),
        FakeGeometryRow(tuple(box(x, 130, x + 100, 155) for x in (100, 200, 300, 400))),
        FakeGeometryRow(tuple(box(x, 155, x + 100, 180) for x in (100, 200, 300, 400))),
    ]
    table = GeometryFakeTable(
        box(100, 100, 500, 180),
        "banner-grid",
        [
            ["Review the separate response.", None, None, None],
            ["Item", "A", "B", "C"],
            ["X", "1", "2", "3"],
            ["Y", "4", "5", "6"],
        ],
        geometry,
    )
    items = [
        *(("l", (100 * scale, y * scale), (500 * scale, y * scale)) for y in (100, 110, 130, 155, 180)),
        *(("l", (x * scale, 110 * scale), (x * scale, 180 * scale)) for x in (200, 300, 400)),
    ]
    page = FakePage(
        default=[table],
        drawings=[{"rect": box(100, 100, 500, 180), "color": (0.0, 0.0, 0.0), "items": items}],
        width=600 * scale,
        height=800 * scale,
    )

    diagnostics = []
    detected = detect_page_tables(cast(pymupdf.Page, page), diagnostics=diagnostics)

    assert len(detected) == 1
    assert tuple(detected[0].bbox) == box(100, 110, 500, 180)
    assert detected[0].extract() == table.extracted_rows[1:]
    assert any(
        record.stage == TableDiagnosticStage.BOUNDARY_FILTER
        and record.outcome == TableDiagnosticOutcome.APPLIED
        and record.reason == "external_sentence_banner_detached"
        and record.metric("affected_count") == 1
        for record in diagnostics
    )


@pytest.mark.parametrize(
    ("banner_text", "first_height", "include_body_grid"),
    [
        ("Quarterly results", 10, True),
        ("Review the separate response.", 20, True),
        ("Review the separate response.", 10, False),
    ],
)
def test_preserves_ambiguous_leading_table_banners(
    banner_text: str, first_height: int, include_body_grid: bool
) -> None:
    body_top = 100 + first_height
    geometry = [
        FakeGeometryRow(((100.0, 100.0, 500.0, float(body_top)), None, None, None)),
        FakeGeometryRow(
            tuple((float(x), float(body_top), float(x + 100), 130.0) for x in (100, 200, 300, 400))
        ),
        FakeGeometryRow(tuple((float(x), 130.0, float(x + 100), 155.0) for x in (100, 200, 300, 400))),
        FakeGeometryRow(tuple((float(x), 155.0, float(x + 100), 180.0) for x in (100, 200, 300, 400))),
    ]
    table = GeometryFakeTable(
        (100.0, 100.0, 500.0, 180.0),
        "ambiguous-banner",
        [
            [banner_text, None, None, None],
            ["Item", "A", "B", "C"],
            ["X", "1", "2", "3"],
            ["Y", "4", "5", "6"],
        ],
        geometry,
    )
    items = [
        *(("l", (100.0, float(y)), (500.0, float(y))) for y in (100, body_top, 130, 155, 180)),
        *(
            (("l", (float(x), float(body_top)), (float(x), 180.0)) for x in (200, 300, 400))
            if include_body_grid
            else ()
        ),
    ]
    page = FakePage(
        default=[table],
        drawings=[{"rect": (100.0, 100.0, 500.0, 180.0), "color": (0.0, 0.0, 0.0), "items": items}],
    )

    detected = _detect(page)
    assert len(detected) == 1
    assert tuple(detected[0].bbox) == table.bbox
    assert detected[0].extract() == table.extract()


@pytest.mark.parametrize(
    ("scale", "dx", "dy", "reverse_candidates", "reverse_evidence"),
    [
        (0.5, 17.0, -9.0, False, True),
        (1.0, 0.0, 0.0, True, False),
        (2.0, -31.0, 23.0, False, False),
    ],
)
def test_suppresses_rule_connected_lower_cardinality_external_strip_under_affine_and_permutation(
    scale: float,
    dx: float,
    dy: float,
    reverse_candidates: bool,
    reverse_evidence: bool,
) -> None:
    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)

    strip = FakeTable(box(100.0, 100.0, 500.0, 110.0), "external-strip", rows=[["Band", "Title"]])
    core = FakeTable(
        box(100.0, 120.0, 500.0, 200.0),
        "core",
        rows=[
            ["Item", "A", "B", "C"],
            ["X", "1", "2", "3"],
            ["Y", "4", "5", "6"],
        ],
    )
    candidates = [strip, core]
    if reverse_candidates:
        candidates.reverse()
    items = [
        *(
            ("l", (100.0 * scale + dx, y * scale + dy), (500.0 * scale + dx, y * scale + dy))
            for y in (100.0, 110.0, 120.0, 160.0, 200.0)
        ),
        *(
            ("l", (x * scale + dx, 120.0 * scale + dy), (x * scale + dx, 200.0 * scale + dy))
            for x in (200.0, 300.0, 400.0)
        ),
    ]
    if reverse_evidence:
        items.reverse()
    page = FakePage(
        default=candidates,
        drawings=[
            {
                "rect": box(100.0, 100.0, 500.0, 200.0),
                "color": (0.0, 0.0, 0.0),
                "items": items,
            }
        ],
        width=600.0 * scale,
        height=800.0 * scale,
    )

    _assert_snapshot(_detect(page), core, FinderProvenance.DEFAULT)


@pytest.mark.parametrize(
    ("scale", "dx", "dy", "reverse_candidates", "reverse_evidence"),
    [
        (0.5, 13.0, -7.0, False, True),
        (1.0, 0.0, 0.0, True, False),
        (2.0, -29.0, 19.0, False, False),
    ],
)
def test_preserves_captionless_independent_ruled_strip_with_its_own_vertical_grid(
    scale: float,
    dx: float,
    dy: float,
    reverse_candidates: bool,
    reverse_evidence: bool,
) -> None:
    def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * scale + dx, y0 * scale + dy, x1 * scale + dx, y1 * scale + dy)

    strip = FakeTable(box(100.0, 100.0, 500.0, 110.0), "independent", rows=[["Metric", "Value"]])
    core = FakeTable(
        box(100.0, 120.0, 500.0, 200.0),
        "core",
        rows=[["Item", "A", "B", "C"], ["X", "1", "2", "3"], ["Y", "4", "5", "6"]],
    )
    candidates = [strip, core]
    if reverse_candidates:
        candidates.reverse()
    items = [
        *(
            ("l", (100.0 * scale + dx, y * scale + dy), (500.0 * scale + dx, y * scale + dy))
            for y in (100.0, 110.0, 120.0, 160.0, 200.0)
        ),
        ("l", (300.0 * scale + dx, 100.0 * scale + dy), (300.0 * scale + dx, 110.0 * scale + dy)),
        *(
            ("l", (x * scale + dx, 120.0 * scale + dy), (x * scale + dx, 200.0 * scale + dy))
            for x in (200.0, 300.0, 400.0)
        ),
    ]
    if reverse_evidence:
        items.reverse()
    page = FakePage(
        default=candidates,
        drawings=[
            {
                "rect": box(100.0, 100.0, 500.0, 200.0),
                "color": (0.0, 0.0, 0.0),
                "items": items,
            }
        ],
        width=600.0 * scale,
        height=800.0 * scale,
    )

    tables = _detect(page)

    assert [tuple(table.bbox) for table in tables] == [strip.bbox, core.bbox]


def test_preserves_lower_cardinality_strip_without_adjacent_core_boundary_rule() -> None:
    strip = FakeTable((100.0, 100.0, 500.0, 110.0), "strip", rows=[["Band", "Title"]])
    core = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "core",
        rows=[["Item", "A", "B", "C"], ["X", "1", "2", "3"], ["Y", "4", "5", "6"]],
    )
    page = FakePage(
        default=[strip, core],
        drawings=[
            {
                "rect": (100.0, 100.0, 500.0, 200.0),
                "items": [("l", (100.0, y), (500.0, y)) for y in (100.0, 110.0, 115.0, 160.0, 200.0)],
            }
        ],
    )

    tables = _detect(page)

    assert [tuple(table.bbox) for table in tables] == [strip.bbox, core.bbox]


def test_preserves_strip_when_horizontal_support_is_fill_only() -> None:
    strip = FakeTable((100.0, 100.0, 500.0, 110.0), "strip", rows=[["Band", "Title"]])
    core = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "core",
        rows=[["Item", "A", "B", "C"], ["X", "1", "2", "3"], ["Y", "4", "5", "6"]],
    )
    page = FakePage(
        default=[strip, core],
        drawings=[
            {
                "rect": (100.0, 100.0, 500.0, 200.2),
                "color": None,
                "fill": (0.0, 0.0, 0.0),
                "items": [("re", (100.0, y, 500.0, y + 0.2)) for y in (100.0, 110.0, 120.0)],
            },
            {
                "rect": (200.0, 120.0, 400.0, 200.0),
                "color": (0.0, 0.0, 0.0),
                "items": [("l", (x, 120.0), (x, 200.0)) for x in (200.0, 300.0, 400.0)],
            },
        ],
    )

    tables = _detect(page)

    assert [tuple(table.bbox) for table in tables] == [strip.bbox, core.bbox]


@pytest.mark.parametrize(
    "separator",
    [
        (100.0, 111.0, 500.0, 119.0, "Table 2. Independent summary"),
        (100.0, 109.0, 500.0, 121.0, "Table — Independent summary"),
        (100.0, 111.0, 500.0, 119.0, "Figure: independent equation panel"),
    ],
)
def test_preserves_unrelated_lower_cardinality_ruled_table_across_caption_like_separator(
    separator: tuple[float, float, float, float, str],
) -> None:
    strip = FakeTable((100.0, 100.0, 500.0, 110.0), "independent", rows=[["Metric", "Value"]])
    core = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "nearby-grid",
        rows=[["Name", "A", "B", "C"], ["X", "1", "2", "3"], ["Y", "4", "5", "6"]],
    )
    page = FakePage(default=[strip, core], drawings=_fragmented_grid_rules(), blocks=[separator])

    tables = _detect(page)

    assert [tuple(table.bbox) for table in tables] == [strip.bbox, core.bbox]


@pytest.mark.parametrize("rotation", [90, 270])
def test_quarter_turned_aligned_tables_fail_closed_when_support_rules_are_not_horizontal(
    rotation: int,
) -> None:
    matrix = (
        pymupdf.Matrix(0.0, 1.0, -1.0, 0.0, 800.0, 0.0)
        if rotation == 90
        else pymupdf.Matrix(0.0, -1.0, 1.0, 0.0, 0.0, 600.0)
    )

    def transformed_box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        rectangle = pymupdf.Rect(x0, y0, x1, y1) * matrix
        return rectangle.x0, rectangle.y0, rectangle.x1, rectangle.y1

    strip = FakeTable(transformed_box(100.0, 100.0, 500.0, 110.0), "rotated-strip", rows=[["Band", "Title"]])
    core = FakeTable(
        transformed_box(100.0, 120.0, 500.0, 200.0),
        "rotated-core",
        rows=[["Item", "A", "B", "C"], ["X", "1", "2", "3"], ["Y", "4", "5", "6"]],
    )
    items = [
        *(("l", (100.0, y), (500.0, y)) for y in (100.0, 110.0, 120.0, 160.0, 200.0)),
        ("l", (300.0, 100.0), (300.0, 110.0)),
        *(("l", (x, 120.0), (x, 200.0)) for x in (200.0, 300.0, 400.0)),
    ]
    page = FakePage(
        default=[strip, core],
        drawings=[
            {
                "rect": (100.0, 100.0, 500.0, 200.0),
                "color": (0.0, 0.0, 0.0),
                "items": items,
            }
        ],
        width=800.0,
        height=600.0,
        rotation_matrix=matrix,
    )

    tables = _detect(page)

    assert [tuple(table.bbox) for table in tables] == sorted(
        [strip.bbox, core.bbox], key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2])
    )


def test_preserves_same_cardinality_rule_connected_strip_as_possible_grid_fragment() -> None:
    strip = FakeTable(
        (100.0, 100.0, 500.0, 110.0),
        "possible-header",
        rows=[["Item", "A", "B", "C"]],
    )
    core = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "core",
        rows=[["X", "1", "2", "3"], ["Y", "4", "5", "6"], ["Z", "7", "8", "9"]],
    )
    page = FakePage(default=[strip, core], drawings=_fragmented_grid_rules())

    tables = _detect(page)

    assert [tuple(table.bbox) for table in tables] == [strip.bbox, core.bbox]


def test_preserves_caption_separated_single_row_table() -> None:
    one_row_table = FakeTable(
        (100.0, 100.0, 500.0, 110.0),
        "one-row-table",
        rows=[["Metric", "A", "B", "C"]],
    )
    nearby_grid = FakeTable(
        (100.0, 120.0, 500.0, 200.0),
        "nearby-grid",
        rows=[["Name", "A", "B", "C"], ["X", "1", "2", "3"], ["Y", "4", "5", "6"]],
    )
    page = FakePage(
        default=[one_row_table, nearby_grid],
        drawings=_fragmented_grid_rules(),
        blocks=[(100.0, 111.0, 500.0, 119.0, "Table 2. Independent summary")],
    )

    tables = _detect(page)

    assert [tuple(table.bbox) for table in tables] == [one_row_table.bbox, nearby_grid.bbox]


def test_figure_caption_does_not_override_authoritative_default_candidate() -> None:
    table_candidate = FakeTable((60.0, 140.0, 280.0, 240.0), "table")
    figure_candidate = FakeTable((320.0, 340.0, 560.0, 440.0), "figure")
    page = FakePage(
        default=[figure_candidate, table_candidate],
        blocks=[
            (60.0, 110.0, 280.0, 130.0, "Table 2. Results"),
            (320.0, 450.0, 560.0, 470.0, "Figure 4. Process diagram"),
        ],
    )

    tables = _detect(page)
    assert [tuple(table.bbox) for table in tables] == [table_candidate.bbox, figure_candidate.bbox]
    assert [getattr(table, "finder_provenance") for table in tables] == [
        FinderProvenance.DEFAULT,
        FinderProvenance.DEFAULT,
    ]


def test_vector_complexity_does_not_override_authoritative_default_candidate() -> None:
    candidate = FakeTable((100.0, 100.0, 500.0, 500.0), "diagram")
    curves = [("c", (120.0, 120.0), (200.0, 180.0), (300.0, 220.0), (480.0, 480.0))] * 20
    page = FakePage(
        default=[candidate],
        drawings=[{"rect": (100.0, 100.0, 500.0, 500.0), "items": curves}],
    )

    _assert_snapshot(_detect(page), candidate, FinderProvenance.DEFAULT)


def test_axis_aligned_table_grid_is_not_rejected_as_vector_complex() -> None:
    candidate = FakeTable((100.0, 100.0, 500.0, 500.0), "grid")
    lines = [("l", (100.0 + index * 20.0, 100.0), (100.0 + index * 20.0, 500.0)) for index in range(20)]
    page = FakePage(
        default=[candidate],
        drawings=[{"rect": (100.0, 100.0, 500.0, 500.0), "items": lines}],
    )

    _assert_snapshot(_detect(page), candidate, FinderProvenance.DEFAULT)


def test_weak_default_is_replaced_by_containing_coherent_borderless_candidate() -> None:
    weak = FakeTable(
        (140.0, 100.0, 460.0, 300.0),
        "weak-default",
        rows=[["Heading", "A 10 B 20", "C 30", "Total 60"]],
    )
    borderless = FakeTable(
        (100.0, 101.0, 500.0, 299.0),
        "borderless",
        rows=[
            ["Item", "Q1", "Q2", "Total"],
            ["A", "10", "20", "30"],
            ["B", "5", "15", "20"],
            ["Total", "15", "35", "50"],
        ],
    )
    page = FakePage(default=[weak], strict=[borderless])
    diagnostics = []

    _assert_snapshot(
        detect_page_tables(cast(pymupdf.Page, page), diagnostics=diagnostics),
        borderless,
        FinderProvenance.BORDERLESS_TEXT,
    )
    assert any(
        record.stage == TableDiagnosticStage.BOUNDARY_FILTER
        and record.outcome == TableDiagnosticOutcome.APPLIED
        and record.reason == "weak_default_replaced"
        and record.metric("affected_count") == 1
        for record in diagnostics
    )
    assert page.calls == [
        {},
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
        {"vertical_strategy": "text", "horizontal_strategy": "text"},
    ]


def test_weak_default_is_not_replaced_without_strong_containment() -> None:
    weak = FakeTable(
        (100.0, 100.0, 300.0, 300.0),
        "weak-default",
        rows=[["Heading", "A 10 B 20", "C 30", "Total 60"]],
    )
    unrelated = FakeTable(
        (320.0, 100.0, 560.0, 300.0),
        "unrelated-borderless",
        rows=[["Item", "Value"], ["A", "10"], ["B", "20"]],
    )
    page = FakePage(default=[weak], strict=[unrelated])
    diagnostics = []

    _assert_snapshot(
        detect_page_tables(cast(pymupdf.Page, page), diagnostics=diagnostics),
        weak,
        FinderProvenance.DEFAULT,
    )
    assert any(
        record.stage == TableDiagnosticStage.BOUNDARY_FILTER
        and record.outcome == TableDiagnosticOutcome.REJECTED
        and record.reason == "existing_table_retained"
        for record in diagnostics
    )


def test_detector_owns_borderless_fallback_after_ruled_finders_are_empty() -> None:
    borderless = FakeTable(
        (100.0, 100.0, 500.0, 240.0),
        "borderless",
        rows=[["Item", "Value"], ["A", "10"], ["B", "20"]],
    )
    page = FakePage(strict=[borderless])
    diagnostics = []

    _assert_snapshot(
        detect_page_tables(cast(pymupdf.Page, page), diagnostics=diagnostics),
        borderless,
        FinderProvenance.BORDERLESS_TEXT,
    )
    borderless_dedup = next(
        record
        for record in diagnostics
        if record.stage == TableDiagnosticStage.BOUNDARY_DEDUPLICATION
        and record.reason in {"borderless_duplicate_boundary", "no_borderless_duplicate_boundary"}
    )
    assert borderless_dedup.outcome == TableDiagnosticOutcome.UNCHANGED
    assert borderless_dedup.reason == "no_borderless_duplicate_boundary"
    assert borderless_dedup.metric("input_count") == 1
    assert borderless_dedup.metric("output_count") == 1
    assert page.calls == [
        {},
        {"vertical_strategy": "lines_strict", "horizontal_strategy": "lines_strict"},
        {"vertical_strategy": "text", "horizontal_strategy": "text"},
    ]


def test_detection_snapshots_tables_before_later_finder_calls() -> None:
    @dataclass
    class Header:
        external: bool = False
        names: tuple[str, str] = ("Name", "Value")

    @dataclass
    class GeometryRow:
        cells: tuple[tuple[float, float, float, float], ...]

    class StatefulTable:
        bbox = (0.0, 0.0, 20.0, 20.0)
        cells = (
            (0.0, 0.0, 10.0, 10.0),
            (10.0, 0.0, 20.0, 10.0),
            (0.0, 10.0, 10.0, 20.0),
            (10.0, 10.0, 20.0, 20.0),
        )
        header = Header()
        rows = (GeometryRow(cells[:2]), GeometryRow(cells[2:]))

        def __init__(self, page: FakePage) -> None:
            self.page = page

        def extract(self) -> list[list[str | None]]:
            return [["Name", "Value"], ["A", "1"]] if len(self.page.calls) == 1 else []

    page = FakePage()
    table = StatefulTable(page)
    page.default = [cast(DetectedTable, table)]

    detected = _detect(page)

    assert len(detected) == 1
    assert detected[0].extract() == [["Name", "Value"], ["A", "1"]]
    assert getattr(detected[0], "finder_provenance") == FinderProvenance.DEFAULT
    bands = getattr(detected[0], "geometric_bands")
    assert [band.kind for band in bands] == [GeometricBandKind.FINDER_ROW] * 2
    assert [band.bbox for band in bands] == [(0.0, 0.0, 20.0, 10.0), (0.0, 10.0, 20.0, 20.0)]
    assert all(not isinstance(value, pymupdf.Rect) for band in bands for value in band.bbox)

    evidence = getattr(detected[0], "provenance")
    reconstruction = reconstruct_table(cast(FinderSnapshot, detected[0]), evidence)
    assert reconstruction.adapter == "finder_grid_v1"
    assert reconstruction.provenance is not None
    assert reconstruction.provenance.finder == FinderProvenance.DEFAULT
    assert reconstruction.provenance.geometric_band_ids == ("finder-row-0001", "finder-row-0002")
    assert [(cell.row_index, cell.column_index, cell.text) for cell in reconstruction.logical_cells] == [
        (0, 0, "Name"),
        (0, 1, "Value"),
        (1, 0, "A"),
        (1, 1, "1"),
    ]


def test_candidate_permutation_does_not_change_duplicate_selection() -> None:
    first = FakeTable((100.0, 100.0, 500.0, 200.0), "z", rows=[["Z", "2"], ["B", "2"]])
    second = FakeTable((100.0, 100.0, 500.0, 200.0), "a", rows=[["A", "1"], ["B", "1"]])
    forward_diagnostics = []
    reverse_diagnostics = []

    forward = detect_page_tables(
        cast(pymupdf.Page, FakePage(default=[first, second])), diagnostics=forward_diagnostics
    )
    reverse = detect_page_tables(
        cast(pymupdf.Page, FakePage(default=[second, first])), diagnostics=reverse_diagnostics
    )

    assert forward[0].extract() == reverse[0].extract() == second.extract()
    assert getattr(forward[0], "finder_provenance") == FinderProvenance.DEFAULT
    assert forward_diagnostics == reverse_diagnostics
    assert any(
        record.stage == TableDiagnosticStage.BOUNDARY_DEDUPLICATION
        and record.outcome == TableDiagnosticOutcome.REJECTED
        and record.reason == "default_duplicate_boundary"
        for record in forward_diagnostics
    )
    evidence = getattr(forward[0], "provenance")
    assert any(
        record.stage == TableDiagnosticStage.BOUNDARY_DETECTION
        and record.outcome == TableDiagnosticOutcome.ACCEPTED
        for record in evidence.diagnostics
    )


def test_provenance_rejects_nonfinite_duplicate_and_dangling_evidence() -> None:
    rule_id = NativeRuleId("native-rule-0001")
    band = GeometricBand(
        band_id="rule-band-0001",
        kind=GeometricBandKind.RULE_BAND,
        frame=CoordinateFrame.DETECTOR_PAGE,
        bbox=(0.0, 0.0, 10.0, 10.0),
        native_rule_ids=(rule_id,),
    )
    with pytest.raises(ValueError, match="unknown native rule IDs"):
        TableProvenance(
            finder=FinderProvenance.RULE_BAND_TEXT,
            frame=CoordinateFrame.DETECTOR_PAGE,
            geometric_bands=(band,),
        )
    with pytest.raises(ValueError, match="native rule IDs must be unique"):
        TableProvenance(
            finder=FinderProvenance.RULE_BAND_TEXT,
            frame=CoordinateFrame.DETECTOR_PAGE,
            native_rule_ids=(rule_id, rule_id),
        )
    with pytest.raises(ValueError, match="finite"):
        GeometricBand(
            band_id="bad-band",
            kind=GeometricBandKind.FINDER_ROW,
            frame=CoordinateFrame.DETECTOR_PAGE,
            bbox=(0.0, 0.0, float("nan"), 10.0),
        )


def test_missing_finder_fails_fast() -> None:
    page = FakePage()
    page.find_tables = lambda **_kwargs: None  # type: ignore[method-assign, assignment]

    with pytest.raises(RuntimeError, match="did not return a table finder"):
        _detect(page)
