"""Citation coverage metrics over schema-independent content mappings."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from benchmarks.metrics import Metric

_CITATION_KINDS = ("paragraph", "table", "cell")


def citation_coverage(
    items: Iterable[Mapping[str, object]],
    *,
    supported_kinds: Iterable[str] = _CITATION_KINDS,
) -> Mapping[str, Metric]:
    """Measure non-empty content units having at least one citation.

    Each item must have ``kind`` (paragraph, table, or cell), ``text``, and may
    have a ``citations`` sequence. Coverage is reported separately because a
    table-level citation must not conceal uncited cells (or vice versa).
    """

    supported = frozenset(supported_kinds)
    unknown_supported = supported.difference(_CITATION_KINDS)
    if unknown_supported:
        raise ValueError(f"unknown supported citation kinds: {sorted(unknown_supported)}")

    totals = {kind: 0 for kind in _CITATION_KINDS}
    eligible = {kind: 0 for kind in _CITATION_KINDS}
    cited = {kind: 0 for kind in _CITATION_KINDS}
    for item in items:
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in totals:
            raise ValueError("citation item kind must be paragraph, table, or cell")
        text = item.get("text")
        if not isinstance(text, str):
            raise TypeError("citation item text must be a string")
        citations = item.get("citations", ())
        if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
            raise TypeError("citations must be a non-string sequence")
        if any(citation is None for citation in citations):
            raise ValueError("citations cannot contain null entries")

        totals[kind] += 1
        if text.strip():
            eligible[kind] += 1
            cited[kind] += bool(citations)

    metrics: dict[str, Metric] = {}
    for kind in _CITATION_KINDS:
        name = f"{kind}_citation_coverage"
        if kind not in supported:
            metrics[name] = Metric.unsupported(n=totals[kind])
        else:
            metrics[name] = Metric.ratio(cited[kind], eligible[kind], n=totals[kind])
    return MappingProxyType(metrics)
