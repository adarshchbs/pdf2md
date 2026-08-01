from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from benchmarks.digital_pdf_slice import (
    DigitalSliceConfig,
    PageEvidence,
    classify_document_evidence,
    classify_pdf,
    parsebench_test_id,
)
from benchmarks.integrations.parsebench.score_digital_pdf_slice import (
    load_dataset_records,
    macro_metric_aggregation_counts,
    metric_example_counts,
)


def test_document_evidence_accepts_native_text_and_rejects_scan_and_ocr() -> None:
    config = DigitalSliceConfig()

    digital = classify_document_evidence(
        [PageEvidence(native_characters=200, image_coverage=0.1, has_glyphless_font=False)],
        config,
    )
    scanned = classify_document_evidence(
        [PageEvidence(native_characters=0, image_coverage=1.0, has_glyphless_font=False)],
        config,
    )
    ocr_layer = classify_document_evidence(
        [PageEvidence(native_characters=200, image_coverage=1.0, has_glyphless_font=True)],
        config,
    )

    assert digital.eligible is True
    assert digital.reason == "born_digital"
    assert scanned.eligible is False
    assert scanned.reason == "scanned_raster_majority"
    assert ocr_layer.eligible is False
    assert ocr_layer.reason == "ocr_text_layer"


def test_low_text_chart_page_is_not_rejected_without_full_page_raster() -> None:
    result = classify_document_evidence(
        [PageEvidence(native_characters=2, image_coverage=0.49, has_glyphless_font=False)],
        DigitalSliceConfig(),
    )

    assert result.eligible is True
    assert result.reason == "born_digital"


def test_document_without_native_text_is_rejected_even_when_images_are_tiled() -> None:
    result = classify_document_evidence(
        [PageEvidence(native_characters=0, image_coverage=0.49, has_glyphless_font=False)],
        DigitalSliceConfig(),
    )

    assert result.eligible is False
    assert result.reason == "no_native_text"


def test_invisible_ocr_text_layer_is_rejected_without_glyphless_font() -> None:
    result = classify_document_evidence(
        [
            PageEvidence(
                native_characters=200,
                image_coverage=1.0,
                has_glyphless_font=False,
                invisible_text_ratio=1.0,
            )
        ],
        DigitalSliceConfig(),
    )

    assert result.eligible is False
    assert result.reason == "ocr_text_layer"


def test_invisible_text_without_raster_scan_evidence_is_not_rejected() -> None:
    result = classify_document_evidence(
        [
            PageEvidence(
                native_characters=200,
                image_coverage=0.1,
                has_glyphless_font=False,
                invisible_text_ratio=1.0,
            )
        ],
        DigitalSliceConfig(),
    )

    assert result.eligible is True
    assert result.reason == "born_digital"


def test_classify_pdf_reads_native_pdf_evidence(tmp_path: Path) -> None:
    path = tmp_path / "digital.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Born digital text " * 10)
    document.save(path)
    document.close()

    result = classify_pdf(path)

    assert result.eligible is True
    assert result.page_count == 1
    assert result.pages[0].native_characters >= 20


def test_parsebench_test_id_uses_shared_text_group_and_pdf_stem() -> None:
    assert parsebench_test_id("text_content", Path("docs/text/a.b.pdf")) == "text/a.b"
    assert parsebench_test_id("text_formatting", Path("docs/text/a.b.pdf")) == "text/a.b"
    assert parsebench_test_id("layout", Path("docs/layout/a.pdf")) == "layout/a"


def test_dataset_records_merge_shared_text_categories_without_reading_references(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "text" / "sample.pdf"
    source.parent.mkdir(parents=True)
    source.touch()
    row = '{"pdf":"docs/text/sample.pdf","tags":["simple"],"expected_markdown":"secret"}\n'
    (tmp_path / "text_content.jsonl").write_text(row, encoding="utf-8")
    (tmp_path / "text_formatting.jsonl").write_text(row, encoding="utf-8")

    records = load_dataset_records(tmp_path)

    assert len(records) == 1
    assert records[0].test_id == "text/sample"
    assert records[0].categories == ("text_content", "text_formatting")
    assert records[0].tags == ("simple",)


def test_dataset_records_reject_path_traversal(tmp_path: Path) -> None:
    (tmp_path / "text_content.jsonl").write_text(
        '{"pdf":"../outside.pdf","tags":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe PDF path"):
        load_dataset_records(tmp_path)


def test_metric_example_counts_reports_actual_emitting_examples() -> None:
    results = [
        SimpleNamespace(
            test_id="text/a",
            success=True,
            metrics=[
                SimpleNamespace(metric_name="content_faithfulness"),
                SimpleNamespace(metric_name="semantic_formatting"),
            ],
        ),
        SimpleNamespace(
            test_id="text/b",
            success=True,
            metrics=[SimpleNamespace(metric_name="semantic_formatting")],
        ),
        SimpleNamespace(test_id="text/c", success=False, metrics=[]),
    ]

    assert metric_example_counts(results) == {
        "content_faithfulness": 1,
        "semantic_formatting": 2,
    }


def test_metric_example_counts_rejects_duplicate_metric_names() -> None:
    result = SimpleNamespace(
        test_id="text/a",
        success=True,
        metrics=[
            SimpleNamespace(metric_name="content_faithfulness"),
            SimpleNamespace(metric_name="content_faithfulness"),
        ],
    )

    with pytest.raises(ValueError, match="emits duplicate metrics"):
        metric_example_counts([result])


def test_macro_metric_aggregation_counts_includes_official_failure_padding() -> None:
    result = SimpleNamespace(
        success=True,
        product_type="parse",
        metrics=[
            SimpleNamespace(
                metric_name="content_faithfulness",
                value=0.75,
                metadata={},
            )
        ],
    )

    assert macro_metric_aggregation_counts(
        [result],
        Counter({"parse": 2}),
        frozenset(),
    ) == {
        "content_faithfulness": {
            "emitting_examples": 1,
            "failure_padding": 2,
            "aggregate_denominator": 3,
        }
    }
