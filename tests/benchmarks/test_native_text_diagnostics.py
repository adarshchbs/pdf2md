from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import pymupdf
import pytest

from benchmarks.adapters import base as adapter_base
from benchmarks.adapters.pymupdf_text import PyMuPDFTextAdapter
from benchmarks.canonical import (
    AdapterRun,
    CanonicalElement,
    CanonicalPage,
    CanonicalProvenance,
    RuntimeStats,
)
from benchmarks.native_text_diagnostics import (
    NativeTextBaselineEvidence,
    _content_characters,  # pyright: ignore[reportPrivateUsage]
    compare_native_text,
)

_RUNTIME = RuntimeStats(wall_seconds=0)


def _native_run(
    tmp_path: Path, page_texts: list[str], *, name: str = "source.pdf"
) -> tuple[Path, AdapterRun]:
    path = tmp_path / name
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page(width=300, height=500)
            for line_number, line in enumerate(text.splitlines() or [""]):
                if line:
                    page.insert_text((20, 30 + line_number * 15), line, fontsize=10)
        document.save(path)
    run = PyMuPDFTextAdapter().process(path)
    assert run.status == "success"
    return path, run


def _candidate(
    baseline: AdapterRun,
    *,
    page_elements: list[list[tuple[str, str, bool]]] | None = None,
) -> list[CanonicalPage]:
    pages: list[CanonicalPage] = []
    for page_index, baseline_page in enumerate(baseline.pages):
        specs = (
            page_elements[page_index]
            if page_elements is not None
            else [("paragraph", baseline_page.markdown, True)]
        )
        elements = [
            CanonicalElement(
                id=f"element-{index}",
                element_type=element_type,  # type: ignore[arg-type]
                reading_order=index,
                text=text,
                markdown=text,
                include_in_output=include,
            )
            for index, (element_type, text, include) in enumerate(specs)
        ]
        pages.append(
            CanonicalPage(
                document_id=baseline_page.document_id,
                page_index=baseline_page.page_index,
                page_size=baseline_page.page_size,
                markdown="unused",
                elements=elements,
                tables=[],
                figures=[],
                runtime=_RUNTIME,
                provenance=CanonicalProvenance(
                    tool_name="pdf2md",
                    tool_version="1",
                    mode="structured",
                    input_sha256=baseline_page.provenance.input_sha256,
                    config_sha256="b" * 64,
                ),
            )
        )
    return pages


def _evidence(path: Path, run: AdapterRun) -> NativeTextBaselineEvidence:
    return NativeTextBaselineEvidence(source_pdf=path, run=run)


def test_valid_locked_extraction_is_recomputed_and_accepted(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["Alpha beta."])

    report = compare_native_text(_candidate(baseline), _evidence(path, baseline))

    page = report.documents[0].pages[0]
    assert page.metrics.content_character_recall == 1
    assert page.metrics.normalized_text_correctness == 1
    assert page.losses.missing_or_filtered_characters == 0


@pytest.mark.parametrize("binary_flag", [0, 1 << 29])
def test_snapshot_open_flags_include_optional_binary_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binary_flag: int
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"exact\r\nbytes")
    real_open = os.open
    captured_flags: list[int] = []
    if binary_flag:
        monkeypatch.setattr(adapter_base.os, "O_BINARY", binary_flag, raising=False)
    else:
        monkeypatch.delattr(adapter_base.os, "O_BINARY", raising=False)

    def capture_open(path: Path, flags: int) -> int:
        captured_flags.append(flags)
        return real_open(path, flags & ~binary_flag)

    monkeypatch.setattr(adapter_base.os, "open", capture_open)

    snapshot = adapter_base.snapshot_regular_file(source)

    assert snapshot.data == b"exact\r\nbytes"
    assert captured_flags[0] & binary_flag == binary_flag


def test_symlink_source_is_rejected(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["text"])
    symlink = tmp_path / "source-link.pdf"
    symlink.symlink_to(path)

    with pytest.raises(ValueError, match="symlink"):
        compare_native_text(_candidate(baseline), _evidence(symlink, baseline))


def test_same_size_inode_replacement_during_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, baseline = _native_run(tmp_path, ["text"])
    original_read = adapter_base._read_fd_bytes  # pyright: ignore[reportPrivateUsage]

    def replace_after_read(descriptor: int) -> bytes:
        data = original_read(descriptor)
        replacement = tmp_path / "replacement.pdf"
        replacement.write_bytes(b"x" * len(data))
        os.replace(replacement, path)
        return data

    monkeypatch.setattr(adapter_base, "_read_fd_bytes", replace_after_read)
    with pytest.raises(ValueError, match="immutable snapshot read"):
        compare_native_text(_candidate(baseline), _evidence(path, baseline))


def test_in_place_mutation_during_read_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, baseline = _native_run(tmp_path, ["text"])
    original_read = adapter_base._read_fd_bytes  # pyright: ignore[reportPrivateUsage]

    def mutate_after_read(descriptor: int) -> bytes:
        data = original_read(descriptor)
        with path.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            last_byte = stream.read(1)
            stream.seek(-1, os.SEEK_END)
            stream.write(bytes([last_byte[0] ^ 1]))
        return data

    monkeypatch.setattr(adapter_base, "_read_fd_bytes", mutate_after_read)
    with pytest.raises(ValueError, match="mutated during"):
        compare_native_text(_candidate(baseline), _evidence(path, baseline))


def test_transient_path_swap_cannot_change_snapshotted_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, baseline = _native_run(tmp_path, ["trusted text"])
    wrong_path, _ = _native_run(tmp_path, ["hostile text"], name="wrong.pdf")
    original_read = adapter_base._read_fd_bytes  # pyright: ignore[reportPrivateUsage]

    def swap_away_and_back(descriptor: int) -> bytes:
        data = original_read(descriptor)
        held = tmp_path / "held.pdf"
        os.replace(path, held)
        os.replace(wrong_path, path)
        os.replace(path, wrong_path)
        os.replace(held, path)
        return data

    monkeypatch.setattr(adapter_base, "_read_fd_bytes", swap_away_and_back)

    with pytest.raises(ValueError, match="immutable snapshot read"):
        compare_native_text(_candidate(baseline), _evidence(path, baseline))


def test_canonical_pages_or_adapter_run_alone_cannot_claim_baseline_authority(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["text"])
    candidate = _candidate(baseline)

    for self_attested in (baseline, baseline.pages):
        with pytest.raises(TypeError, match="NativeTextBaselineEvidence"):
            compare_native_text(candidate, cast(NativeTextBaselineEvidence, self_attested))
    assert compare_native_text(candidate, _evidence(path, baseline)).metrics.content_character_recall == 1


def test_fabricated_canonical_page_is_rejected_against_source(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["source text"])
    fabricated_page = baseline.pages[0].model_copy(update={"markdown": "fabricated"})
    fabricated = baseline.model_copy(update={"pages": [fabricated_page]})

    with pytest.raises(ValueError, match="canonical pages"):
        compare_native_text(_candidate(baseline), _evidence(path, fabricated))


def test_contradictory_raw_records_are_rejected_against_source(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["source text"])
    contradictory = baseline.model_copy(
        update={"raw_records": [{"page_index": 0, "width": 300.0, "height": 500.0, "text": "other"}]}
    )

    with pytest.raises(ValueError, match="raw records"):
        compare_native_text(_candidate(baseline), _evidence(path, contradictory))


def test_wrong_source_bytes_and_candidate_hash_are_rejected(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["source text"])
    wrong_path, _ = _native_run(tmp_path, ["other text"], name="wrong.pdf")
    wrong_candidate_page = _candidate(baseline)[0]
    wrong_candidate_page = wrong_candidate_page.model_copy(
        update={"provenance": wrong_candidate_page.provenance.model_copy(update={"input_sha256": "c" * 64})}
    )

    with pytest.raises(ValueError, match="source SHA-256"):
        compare_native_text(_candidate(baseline), _evidence(wrong_path, baseline))
    with pytest.raises(ValueError, match="input SHA-256"):
        compare_native_text([wrong_candidate_page], _evidence(path, baseline))


def test_attributes_supported_header_dehyphenation_and_duplicate_losses(tmp_path: Path) -> None:
    path, baseline = _native_run(
        tmp_path,
        ["Report Header\ninter-\nnational\nalpha\nalpha\nBody text"],
    )
    candidate = _candidate(
        baseline,
        page_elements=[
            [
                ("header", "Report Header", False),
                ("paragraph", "international", True),
                ("paragraph", "alpha", True),
                ("paragraph", "Body text", True),
            ]
        ],
    )

    page = compare_native_text(candidate, _evidence(path, baseline)).documents[0].pages[0]

    assert page.losses.header_footer_exclusion_characters == 12
    assert page.losses.dehyphenation_join_characters == 1
    assert page.losses.duplicate_suppression_characters == 5
    assert page.losses.missing_or_filtered_characters == 0


def test_duplicate_attribution_requires_retained_exact_occurrence(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["alpha\nalpha"])
    candidate = _candidate(baseline, page_elements=[[]])

    page = compare_native_text(candidate, _evidence(path, baseline)).documents[0].pages[0]

    assert page.losses.duplicate_suppression_characters == 0
    assert page.losses.missing_or_filtered_characters == 10


def test_header_attribution_rejects_substring_match(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["Report Header extended"])
    candidate = _candidate(
        baseline,
        page_elements=[[("header", "Report Header", False), ("paragraph", "extended", True)]],
    )

    page = compare_native_text(candidate, _evidence(path, baseline)).documents[0].pages[0]

    assert page.losses.header_footer_exclusion_characters == 0
    assert page.losses.missing_or_filtered_characters == 12


def test_dehyphenation_attribution_withheld_under_duplicate_ambiguity(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["inter-\nnational\ninter-\nnational"])
    candidate = _candidate(
        baseline,
        page_elements=[[("paragraph", "international", True)]],
    )

    page = compare_native_text(candidate, _evidence(path, baseline)).documents[0].pages[0]

    assert page.losses.dehyphenation_join_characters == 0
    assert page.losses.missing_or_filtered_characters > 0


def test_reading_order_reports_pairwise_inversions(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["one two three"])
    candidate = _candidate(
        baseline,
        page_elements=[
            [
                ("paragraph", "three", True),
                ("paragraph", "one", True),
                ("paragraph", "two", True),
            ]
        ],
    )

    page = compare_native_text(candidate, _evidence(path, baseline)).documents[0].pages[0]

    assert page.metrics.reading_order_delta == pytest.approx(2 / 3)
    assert page.losses.order_inversions == 2
    assert page.losses.comparable_order_pairs == 3


def test_document_metrics_aggregate_recomputed_pages(tmp_path: Path) -> None:
    path, baseline = _native_run(tmp_path, ["abcd", "c"])
    candidate = _candidate(
        baseline,
        page_elements=[[("paragraph", "ab", True)], [("paragraph", "c", True)]],
    )

    report = compare_native_text(candidate, _evidence(path, baseline))

    assert report.documents[0].metrics.baseline_content_characters == 5
    assert report.documents[0].metrics.recalled_content_characters == 3
    assert report.documents[0].metrics.content_character_recall == pytest.approx(3 / 5)
    assert report.metrics == report.documents[0].metrics


def test_content_character_accounting_is_nfkc_consistent() -> None:
    assert _content_characters("oﬃce") == _content_characters("office")
