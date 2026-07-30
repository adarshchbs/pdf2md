"""Small, schema-independent building blocks for benchmark product metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class Metric:
    """A metric value together with the population that produced it.

    ``n`` is the number of attempted observations, ``eligible`` is the number
    included in the value, and ``failures`` is the number that failed before a
    value could be produced. Unsupported metrics are represented explicitly as
    ``value=None, supported=False`` rather than being silently set to zero.
    """

    value: float | None
    supported: bool
    n: int
    eligible: int
    failures: int = 0

    def __post_init__(self) -> None:
        if self.n < 0 or self.eligible < 0 or self.failures < 0:
            raise ValueError("metric counts must be non-negative")
        if self.eligible > self.n:
            raise ValueError("eligible cannot exceed n")
        if self.failures > self.n:
            raise ValueError("failures cannot exceed n")
        if not self.supported and self.value is not None:
            raise ValueError("unsupported metrics must have a null value")
        if self.value is not None and not isfinite(self.value):
            raise ValueError("metric values must be finite")
        if self.value is not None and self.eligible == 0:
            raise ValueError("a valued metric must have an eligible observation")

    @classmethod
    def ratio(
        cls,
        numerator: int | float,
        denominator: int,
        *,
        n: int | None = None,
        failures: int = 0,
    ) -> Metric:
        if denominator < 0:
            raise ValueError("denominator must be non-negative")
        if numerator < 0 or numerator > denominator:
            raise ValueError("ratio numerator must be between zero and denominator")
        population = denominator if n is None else n
        if population < denominator:
            raise ValueError("n cannot be smaller than denominator")
        if denominator == 0:
            return cls(value=None, supported=True, n=population, eligible=0, failures=failures)
        return cls(
            value=float(numerator / denominator),
            supported=True,
            n=population,
            eligible=denominator,
            failures=failures,
        )

    @classmethod
    def unsupported(cls, *, n: int = 0, failures: int = 0) -> Metric:
        return cls(value=None, supported=False, n=n, eligible=0, failures=failures)


def aggregate_metric(metrics: Iterable[Metric]) -> Metric:
    """Return an eligible-count-weighted macro aggregate.

    Supported nulls and unsupported values remain in ``n`` but do not
    contribute to the mean. This makes the aggregate denominator auditable.
    """

    values = tuple(metrics)
    n = sum(metric.n for metric in values)
    failures = sum(metric.failures for metric in values)
    contributing = tuple(metric for metric in values if metric.supported and metric.value is not None)
    eligible = sum(metric.eligible for metric in contributing)
    if not contributing:
        return Metric(
            value=None,
            supported=any(metric.supported for metric in values),
            n=n,
            eligible=0,
            failures=failures,
        )
    value = sum(metric.value * metric.eligible for metric in contributing if metric.value is not None)
    return Metric(value=value / eligible, supported=True, n=n, eligible=eligible, failures=failures)


def aggregate_metric_maps(reports: Iterable[Mapping[str, Metric]]) -> Mapping[str, Metric]:
    """Aggregate reports that expose the same metric names."""

    materialized = tuple(reports)
    if not materialized:
        return MappingProxyType({})
    names = set(materialized[0])
    if any(set(report) != names for report in materialized[1:]):
        raise ValueError("all reports must contain the same metric names")
    return MappingProxyType({
        name: aggregate_metric(report[name] for report in materialized) for name in sorted(names)
    })


from benchmarks.metrics.citations import citation_coverage  # noqa: E402
from benchmarks.metrics.efficiency import efficiency_metrics  # noqa: E402
from benchmarks.metrics.tables import table_metrics  # noqa: E402

__all__ = [
    "Metric",
    "aggregate_metric",
    "aggregate_metric_maps",
    "citation_coverage",
    "efficiency_metrics",
    "table_metrics",
]
