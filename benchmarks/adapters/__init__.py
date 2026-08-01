from __future__ import annotations

from benchmarks.adapters.base import AdapterRegistry, BenchmarkAdapter


def _our_parser() -> BenchmarkAdapter:
    from benchmarks.adapters.our_parser import OurParserAdapter

    return OurParserAdapter()


def _pymupdf_text() -> BenchmarkAdapter:
    from benchmarks.adapters.pymupdf_text import PyMuPDFTextAdapter

    return PyMuPDFTextAdapter()


ADAPTER_REGISTRY = AdapterRegistry({
    "our_parser": _our_parser,
    "pymupdf_text": _pymupdf_text,
})


def get_adapter(adapter_id: str) -> BenchmarkAdapter:
    return ADAPTER_REGISTRY[adapter_id]


__all__ = ["ADAPTER_REGISTRY", "AdapterRegistry", "BenchmarkAdapter", "get_adapter"]
