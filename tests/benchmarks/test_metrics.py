from __future__ import annotations

import pytest

from benchmarks.metrics import Metric, aggregate_metric, aggregate_metric_maps
from benchmarks.metrics.citations import citation_coverage
from benchmarks.metrics.efficiency import efficiency_metrics
from benchmarks.metrics.tables import (
    normalize_number,
    normalize_unicode,
    numeric_tokens,
    table_metrics,
)


def test_unicode_and_numeric_normalization_handles_compatibility_digits_and_separators() -> None:
    assert normalize_unicode("  ＣＡＦÉ  ١٢−３  ") == "café 12-3"
    assert normalize_number("١٬٢٠٠٫٥٠") == "1200.5"
    assert normalize_number("(１ ２００,５０)") == "-1200.5"
    assert numeric_tokens("Revenue was ١٬٢٠٠٫٥٠ and margin ２５%.") == ("1200.5", "25%")


def test_table_metrics_report_numeric_recall_exact_cells_spans_and_axes() -> None:
    reference = [
        {"row": 0, "column": 0, "text": "Header", "colspan": 2},
        {"row": 1, "column": 0, "text": "Revenue ١٬٢٠٠٫٥٠"},
        {"row": 1, "column": 1, "text": "25 %"},
        {"row": 2, "column": 0, "text": "Total 30"},
    ]
    candidate = [
        {"row": 0, "column": 0, "text": "Ｈｅａｄｅｒ", "colspan": 2},
        {"row": 1, "column": 0, "text": "Revenue 1,200.5"},
        {"row": 1, "column": 1, "text": "24 %"},
        {"row": 2, "column": 0, "text": "Total 30"},
    ]

    metrics = table_metrics(candidate, reference)

    assert metrics["numeric_recall"].value == pytest.approx(2 / 3)
    assert metrics["exact_numeric_cell_precision"].value == pytest.approx(2 / 3)
    assert metrics["exact_numeric_cell_recall"].value == pytest.approx(2 / 3)
    assert metrics["exact_numeric_cell_f1"].value == pytest.approx(2 / 3)
    assert metrics["span_structure_precision"].value == 1
    assert metrics["span_structure_recall"].value == 1
    assert metrics["span_structure_f1"].value == 1
    assert metrics["row_exact"].value == pytest.approx(2 / 3)
    assert metrics["column_exact"].value == 0


def test_table_structure_metrics_preserve_absent_claim_semantics_and_multiplicity() -> None:
    reference = [{"row": 0, "column": 0, "text": "A", "colspan": 2}]
    candidate = [{"row": 0, "column": 0, "text": "A"}]

    metrics = table_metrics(candidate, reference)

    assert metrics["span_structure_precision"].value is None
    assert metrics["span_structure_precision"].supported is True
    assert metrics["span_structure_recall"].value == 0
    assert metrics["span_structure_f1"].value == 0


def test_table_metrics_mark_unavailable_capabilities_unsupported() -> None:
    metrics = table_metrics([], [], supported_metrics={"numeric_recall"})

    assert metrics["numeric_recall"] == Metric(value=None, supported=True, n=0, eligible=0)
    assert metrics["row_exact"] == Metric.unsupported()
    assert metrics["row_exact"].value is None
    assert metrics["row_exact"].supported is False


def test_table_metrics_fail_fast_on_malformed_cells() -> None:
    with pytest.raises(ValueError, match="duplicate cell anchor"):
        table_metrics(
            [
                {"row": 0, "column": 0, "text": "first"},
                {"row": 0, "column": 0, "text": "second"},
            ],
            [],
        )
    with pytest.raises(TypeError, match="row must be an integer"):
        table_metrics([{"row": True, "column": 0, "text": "bad"}], [])
    with pytest.raises(ValueError, match="not a numeric token"):
        normalize_number("twelve")


def test_citation_coverage_is_separate_for_paragraphs_tables_and_cells() -> None:
    metrics = citation_coverage([
        {"kind": "paragraph", "text": "Claim", "citations": ["source-1"]},
        {"kind": "paragraph", "text": "Other claim", "citations": []},
        {"kind": "paragraph", "text": "   ", "citations": []},
        {"kind": "table", "text": "Table 1", "citations": ["source-2"]},
        {"kind": "cell", "text": "10", "citations": ["source-2"]},
        {"kind": "cell", "text": "20"},
    ])

    assert metrics["paragraph_citation_coverage"] == Metric(value=0.5, supported=True, n=3, eligible=2)
    assert metrics["table_citation_coverage"].value == 1
    assert metrics["cell_citation_coverage"].value == 0.5


def test_citation_capability_can_be_explicitly_unsupported() -> None:
    metrics = citation_coverage(
        [{"kind": "cell", "text": "10", "citations": []}],
        supported_kinds={"paragraph", "table"},
    )

    assert metrics["cell_citation_coverage"] == Metric.unsupported(n=1)
    assert metrics["paragraph_citation_coverage"].supported is True
    assert metrics["paragraph_citation_coverage"].value is None


def test_efficiency_rates_and_strict_repeat_determinism_include_counts() -> None:
    metrics = efficiency_metrics([
        {"case_id": "a", "status": "success", "output_digest": "same"},
        {"case_id": "a", "status": "success", "output_digest": "same"},
        {"case_id": "b", "status": "success", "output_digest": "first"},
        {"case_id": "b", "status": "empty", "output_digest": "empty"},
        {"case_id": "a", "status": "timeout"},
        {"case_id": "b", "status": "invalid"},
    ])

    assert metrics["success_rate"] == Metric(value=0.5, supported=True, n=6, eligible=6, failures=2)
    assert metrics["empty_rate"].value == pytest.approx(1 / 6)
    assert metrics["timeout_rate"].value == pytest.approx(1 / 6)
    assert metrics["invalid_rate"].value == pytest.approx(1 / 6)
    assert metrics["determinism"] == Metric(value=0.5, supported=True, n=6, eligible=2, failures=2)


def test_efficiency_can_report_determinism_as_unsupported_without_output_fields() -> None:
    metrics = efficiency_metrics(
        [{"status": "success"}],
        supported_metrics={"success_rate", "empty_rate", "timeout_rate", "invalid_rate"},
    )

    assert metrics["determinism"] == Metric.unsupported(n=1)
    with pytest.raises(TypeError, match="output_digest"):
        efficiency_metrics([{"case_id": "sample", "status": "success"}])
    crashed = efficiency_metrics([{"status": "crashed"}])
    assert crashed["crashed_rate"] == Metric(value=1.0, supported=True, n=1, eligible=1, failures=1)
    with pytest.raises(ValueError, match="run status"):
        efficiency_metrics([{"status": "aborted"}])


def test_aggregate_metrics_carry_population_eligibility_and_failures() -> None:
    aggregate = aggregate_metric([
        Metric.ratio(1, 2),
        Metric.unsupported(n=1, failures=1),
        Metric.ratio(1, 1),
    ])

    assert aggregate.value == pytest.approx(2 / 3)
    assert (aggregate.supported, aggregate.n, aggregate.eligible, aggregate.failures) == (True, 4, 3, 1)
    mapped = aggregate_metric_maps([
        {"coverage": Metric.ratio(1, 2), "optional": Metric.unsupported(n=2)},
        {"coverage": Metric.ratio(1, 1), "optional": Metric.unsupported(n=1, failures=1)},
    ])
    assert list(mapped) == ["coverage", "optional"]
    assert mapped["coverage"].value == pytest.approx(2 / 3)
    assert mapped["optional"] == Metric.unsupported(n=3, failures=1)

    with pytest.raises(ValueError, match="same metric names"):
        aggregate_metric_maps([{"a": Metric.ratio(1, 1)}, {"b": Metric.ratio(1, 1)}])
