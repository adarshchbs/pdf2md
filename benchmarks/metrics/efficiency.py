"""Operational benchmark rates and repeat-run determinism."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from benchmarks.metrics import Metric

_STATUSES = ("success", "empty", "failed", "timeout", "invalid", "crashed", "skipped")
_METRIC_NAMES = (*[f"{status}_rate" for status in _STATUSES], "determinism")


def efficiency_metrics(
    runs: Iterable[Mapping[str, object]],
    *,
    supported_metrics: Iterable[str] = _METRIC_NAMES,
) -> Mapping[str, Metric]:
    """Compute outcome rates and strict repeat-group determinism.

    A run requires a declared harness outcome status. For
    determinism, successful and empty runs additionally require ``case_id`` and
    ``output_digest`` strings. A repeat group is eligible when it has at least
    two completed runs, and is deterministic only when all digests agree.
    """

    materialized = tuple(runs)
    supported = frozenset(supported_metrics)
    unknown = supported.difference(_METRIC_NAMES)
    if unknown:
        raise ValueError(f"unknown supported efficiency metrics: {sorted(unknown)}")

    statuses: list[str] = []
    groups: dict[str, list[str]] = defaultdict(list)
    for run in materialized:
        status = run.get("status")
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError(f"run status must be one of {', '.join(_STATUSES)}")
        statuses.append(status)
        if status not in {"success", "empty"}:
            continue
        if "determinism" not in supported:
            continue
        case_id = run.get("case_id")
        output_digest = run.get("output_digest")
        if not isinstance(case_id, str) or not case_id:
            raise TypeError("completed runs require a non-empty string case_id")
        if not isinstance(output_digest, str) or not output_digest:
            raise TypeError("completed runs require a non-empty string output_digest")
        groups[case_id].append(output_digest)

    counts = Counter(statuses)
    failures = sum(counts[status] for status in ("failed", "timeout", "invalid", "crashed"))
    metrics: dict[str, Metric] = {
        f"{status}_rate": Metric.ratio(
            counts[status], len(materialized), n=len(materialized), failures=failures
        )
        for status in _STATUSES
    }

    if "determinism" in supported:
        eligible_groups = [digests for digests in groups.values() if len(digests) >= 2]
        deterministic_groups = sum(len(set(digests)) == 1 for digests in eligible_groups)
        metrics["determinism"] = Metric.ratio(
            deterministic_groups,
            len(eligible_groups),
            n=len(materialized),
            failures=failures,
        )
    else:
        metrics["determinism"] = Metric.unsupported(n=len(materialized), failures=failures)

    for name in _METRIC_NAMES:
        if name not in supported:
            metrics[name] = Metric.unsupported(n=metrics[name].n, failures=failures)
    return MappingProxyType(metrics)
