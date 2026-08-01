from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from benchmarks.adapters.base import snapshot_regular_file
from benchmarks.adapters.pymupdf_text import PyMuPDFTextAdapter
from benchmarks.canonical import AdapterRun, CanonicalElement, CanonicalPage, validate_page_collection

CanonicalDocument: TypeAlias = Sequence[CanonicalPage]
DiagnosticInput: TypeAlias = AdapterRun | CanonicalDocument


@dataclass(frozen=True, slots=True)
class NativeTextBaselineEvidence:
    source_pdf: Path
    run: AdapterRun


@dataclass(frozen=True, slots=True)
class NativeTextMetrics:
    baseline_content_characters: int
    recalled_content_characters: int
    content_character_loss: int
    content_character_recall: float | None
    normalized_text_correctness: float
    reading_order_delta: float | None


@dataclass(frozen=True, slots=True)
class NativeTextLossBuckets:
    missing_or_filtered_characters: int
    whitespace_normalization_edits: int | None
    dehyphenation_join_characters: int
    duplicate_suppression_characters: int
    header_footer_exclusion_characters: int
    order_inversions: int
    comparable_order_pairs: int


@dataclass(frozen=True, slots=True)
class NativeTextPageDiagnostic:
    document_id: str
    page_index: int
    metrics: NativeTextMetrics
    losses: NativeTextLossBuckets


@dataclass(frozen=True, slots=True)
class NativeTextDocumentDiagnostic:
    document_id: str
    metrics: NativeTextMetrics
    losses: NativeTextLossBuckets
    pages: tuple[NativeTextPageDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class NativeTextDiagnosticReport:
    metrics: NativeTextMetrics
    losses: NativeTextLossBuckets
    documents: tuple[NativeTextDocumentDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class _PageEvidence:
    diagnostic: NativeTextPageDiagnostic
    normalized_edit_distance: int
    normalized_length: int


def compare_native_text(
    candidate: DiagnosticInput, baseline: NativeTextBaselineEvidence
) -> NativeTextDiagnosticReport:
    """Compare parser output with a source-verified, freshly recomputed native baseline.

    Supplied baseline artifacts are evidence to verify, never authority. The locked
    adapter is rerun against the immutable source path and its output drives the report.
    """
    if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
        baseline, NativeTextBaselineEvidence
    ):
        raise TypeError("baseline must be NativeTextBaselineEvidence with a source PDF")
    candidate_pages = _pages_from_input(candidate, label="candidate")
    baseline_pages = _verify_and_recompute_baseline(baseline)

    candidate_by_key = {(page.document_id, page.page_index): page for page in candidate_pages}
    baseline_by_key = {(page.document_id, page.page_index): page for page in baseline_pages}
    if candidate_by_key.keys() != baseline_by_key.keys():
        raise ValueError("candidate and baseline must contain the same document/page keys")
    if any(
        candidate_by_key[key].provenance.input_sha256 != baseline_by_key[key].provenance.input_sha256
        for key in baseline_by_key
    ):
        raise ValueError("candidate and baseline input SHA-256 values must match")

    evidence = [_compare_page(candidate_by_key[key], baseline_by_key[key]) for key in sorted(baseline_by_key)]
    documents: list[NativeTextDocumentDiagnostic] = []
    for document_id in sorted({item.diagnostic.document_id for item in evidence}):
        document_evidence = [item for item in evidence if item.diagnostic.document_id == document_id]
        metrics, losses = _aggregate(document_evidence)
        documents.append(
            NativeTextDocumentDiagnostic(
                document_id=document_id,
                metrics=metrics,
                losses=losses,
                pages=tuple(item.diagnostic for item in document_evidence),
            )
        )
    metrics, losses = _aggregate(evidence)
    return NativeTextDiagnosticReport(metrics=metrics, losses=losses, documents=tuple(documents))


def _verify_and_recompute_baseline(evidence: NativeTextBaselineEvidence) -> list[CanonicalPage]:
    source_pdf = evidence.source_pdf
    if not isinstance(source_pdf, Path):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("baseline source_pdf must be a pathlib.Path")
    if not source_pdf.is_file():
        raise FileNotFoundError(source_pdf)
    if not isinstance(evidence.run, AdapterRun):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("baseline run must be an AdapterRun")
    if evidence.run.status != "success":
        raise ValueError("supplied native baseline run must be successful")

    snapshot = snapshot_regular_file(source_pdf)
    supplied_hashes = {page.provenance.input_sha256 for page in evidence.run.pages}
    if supplied_hashes != {snapshot.sha256}:
        raise ValueError("supplied baseline source SHA-256 does not match the source PDF")

    recomputed = PyMuPDFTextAdapter().process_snapshot(snapshot)
    if recomputed.status != "success":
        raise ValueError(f"locked PyMuPDF extraction failed: {recomputed.error}")
    if evidence.run.raw_records != recomputed.raw_records:
        raise ValueError("supplied baseline raw records do not match locked source extraction")
    supplied_pages = [page.model_dump(mode="json", exclude={"runtime"}) for page in evidence.run.pages]
    recomputed_pages = [page.model_dump(mode="json", exclude={"runtime"}) for page in recomputed.pages]
    if supplied_pages != recomputed_pages:
        raise ValueError("supplied baseline canonical pages do not match locked source extraction")
    if evidence.run.warnings != recomputed.warnings or evidence.run.error != recomputed.error:
        raise ValueError("supplied baseline outcome does not match locked source extraction")
    return list(recomputed.pages)


def _pages_from_input(value: DiagnosticInput, *, label: str) -> list[CanonicalPage]:
    if isinstance(value, AdapterRun):
        if value.status != "success":
            raise ValueError(f"{label} AdapterRun must be successful")
        pages = list(value.pages)
    else:
        pages = list(value)
    validate_page_collection(pages)
    return pages


def _compare_page(candidate: CanonicalPage, baseline: CanonicalPage) -> _PageEvidence:
    candidate_text = _output_text(candidate.elements)
    baseline_text = baseline.markdown
    baseline_content = _content_characters(baseline_text)
    candidate_content = _content_characters(candidate_text)
    recalled = sum((baseline_content & candidate_content).values())
    missing = baseline_content - candidate_content
    baseline_count = sum(baseline_content.values())

    header_footer = _attribute_header_footer(candidate.elements, baseline_text, candidate_text, missing)
    duplicate = _attribute_duplicate_suppression(baseline_text, candidate.elements, missing)
    dehyphenation = _attribute_dehyphenation(baseline_text, candidate_text, missing)
    residual = sum(missing.values())
    whitespace_edits = _whitespace_edits(baseline_text, candidate_text)

    normalized_baseline = _normalize_text(baseline_text)
    normalized_candidate = _normalize_text(candidate_text)
    edit_distance = _levenshtein(normalized_baseline, normalized_candidate)
    normalized_length = max(len(normalized_baseline), len(normalized_candidate))
    correctness = 1.0 if normalized_length == 0 else 1 - edit_distance / normalized_length
    inversions, pairs = _order_inversions(baseline_text, candidate_text)
    order_delta = inversions / pairs if pairs else None

    metrics = NativeTextMetrics(
        baseline_content_characters=baseline_count,
        recalled_content_characters=recalled,
        content_character_loss=baseline_count - recalled,
        content_character_recall=recalled / baseline_count if baseline_count else None,
        normalized_text_correctness=correctness,
        reading_order_delta=order_delta,
    )
    losses = NativeTextLossBuckets(
        missing_or_filtered_characters=residual,
        whitespace_normalization_edits=whitespace_edits,
        dehyphenation_join_characters=dehyphenation,
        duplicate_suppression_characters=duplicate,
        header_footer_exclusion_characters=header_footer,
        order_inversions=inversions,
        comparable_order_pairs=pairs,
    )
    return _PageEvidence(
        diagnostic=NativeTextPageDiagnostic(
            document_id=baseline.document_id,
            page_index=baseline.page_index,
            metrics=metrics,
            losses=losses,
        ),
        normalized_edit_distance=edit_distance,
        normalized_length=normalized_length,
    )


def _output_text(elements: Sequence[CanonicalElement]) -> str:
    return "\n".join(element.text for element in elements if element.include_in_output)


def _attribute_header_footer(
    elements: Sequence[CanonicalElement],
    baseline_text: str,
    candidate_text: str,
    missing: Counter[str],
) -> int:
    attributed = 0
    available_occurrences = Counter(_normalized_lines(baseline_text)) - Counter(
        _normalized_lines(candidate_text)
    )
    for element in elements:
        if element.include_in_output or element.element_type not in {"header", "footer"}:
            continue
        normalized = _normalize_text(element.text)
        if not normalized or available_occurrences[normalized] <= 0:
            continue
        available_occurrences[normalized] -= 1
        attributed += _allocate_missing(element.text, missing)
    return attributed


def _attribute_duplicate_suppression(
    baseline_text: str,
    elements: Sequence[CanonicalElement],
    missing: Counter[str],
) -> int:
    baseline_lines: dict[str, list[str]] = defaultdict(list)
    for line in baseline_text.splitlines():
        normalized = _normalize_text(line)
        if normalized:
            baseline_lines[normalized].append(line)
    candidate_counts = Counter(
        _normalize_text(element.text) for element in elements if element.include_in_output
    )
    attributed = 0
    for normalized, lines in sorted(baseline_lines.items()):
        retained = candidate_counts[normalized]
        suppressed = len(lines) - retained
        if len(lines) < 2 or retained < 1 or suppressed <= 0:
            continue
        for line in lines[:suppressed]:
            attributed += _allocate_missing(line, missing)
    return attributed


def _attribute_dehyphenation(baseline_text: str, candidate_text: str, missing: Counter[str]) -> int:
    matches_by_joined: dict[str, list[re.Match[str]]] = defaultdict(list)
    normalized_baseline = unicodedata.normalize("NFKC", baseline_text)
    for match in re.finditer(r"(?P<left>\w+)-\s+(?P<right>\w+)", normalized_baseline, flags=re.UNICODE):
        joined = f"{match.group('left')}{match.group('right')}".casefold()
        matches_by_joined[joined].append(match)

    candidate_tokens = Counter(_tokens(candidate_text))
    baseline_tokens = Counter(_tokens(baseline_text))
    attributed = 0
    for joined, matches in sorted(matches_by_joined.items()):
        if len(matches) != 1 or candidate_tokens[joined] != 1 or baseline_tokens[joined] != 0:
            continue
        attributed += _allocate_missing("-", missing)
    return attributed


def _allocate_missing(text: str, missing: Counter[str]) -> int:
    allocated = 0
    for character in unicodedata.normalize("NFKC", text):
        if character.isspace() or missing[character] <= 0:
            continue
        missing[character] -= 1
        allocated += 1
    return allocated


def _content_characters(text: str) -> Counter[str]:
    return Counter(character for character in unicodedata.normalize("NFKC", text) if not character.isspace())


def _normalized_lines(text: str) -> list[str]:
    return [normalized for line in text.splitlines() if (normalized := _normalize_text(line))]


def _whitespace_edits(baseline_text: str, candidate_text: str) -> int | None:
    baseline_text = unicodedata.normalize("NFKC", baseline_text)
    candidate_text = unicodedata.normalize("NFKC", candidate_text)
    baseline_content = "".join(character for character in baseline_text if not character.isspace())
    candidate_content = "".join(character for character in candidate_text if not character.isspace())
    if baseline_content != candidate_content:
        return None
    baseline_runs = re.split(r"\S", baseline_text)
    candidate_runs = re.split(r"\S", candidate_text)
    return sum(_levenshtein(left, right) for left, right in zip(baseline_runs, candidate_runs, strict=True))


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+(?:['’]\w+)*", normalized, flags=re.UNICODE)


def _order_inversions(baseline_text: str, candidate_text: str) -> tuple[int, int]:
    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, token in enumerate(_tokens(baseline_text)):
        positions[token].append(index)
    matched_positions: list[int] = []
    for token in _tokens(candidate_text):
        if positions[token]:
            matched_positions.append(positions[token].popleft())
    inversions = sum(
        left > right
        for index, left in enumerate(matched_positions)
        for right in matched_positions[index + 1 :]
    )
    pair_count = len(matched_positions) * (len(matched_positions) - 1) // 2
    return inversions, pair_count


def _levenshtein(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_character in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_character in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _aggregate(evidence: Sequence[_PageEvidence]) -> tuple[NativeTextMetrics, NativeTextLossBuckets]:
    baseline_count = sum(item.diagnostic.metrics.baseline_content_characters for item in evidence)
    recalled = sum(item.diagnostic.metrics.recalled_content_characters for item in evidence)
    edit_distance = sum(item.normalized_edit_distance for item in evidence)
    normalized_length = sum(item.normalized_length for item in evidence)
    inversions = sum(item.diagnostic.losses.order_inversions for item in evidence)
    pairs = sum(item.diagnostic.losses.comparable_order_pairs for item in evidence)
    whitespace_values = [item.diagnostic.losses.whitespace_normalization_edits for item in evidence]
    whitespace = (
        None
        if any(value is None for value in whitespace_values)
        else sum(value for value in whitespace_values if value is not None)
    )
    metrics = NativeTextMetrics(
        baseline_content_characters=baseline_count,
        recalled_content_characters=recalled,
        content_character_loss=baseline_count - recalled,
        content_character_recall=recalled / baseline_count if baseline_count else None,
        normalized_text_correctness=(
            1.0 if normalized_length == 0 else 1 - edit_distance / normalized_length
        ),
        reading_order_delta=inversions / pairs if pairs else None,
    )
    losses = NativeTextLossBuckets(
        missing_or_filtered_characters=sum(
            item.diagnostic.losses.missing_or_filtered_characters for item in evidence
        ),
        whitespace_normalization_edits=whitespace,
        dehyphenation_join_characters=sum(
            item.diagnostic.losses.dehyphenation_join_characters for item in evidence
        ),
        duplicate_suppression_characters=sum(
            item.diagnostic.losses.duplicate_suppression_characters for item in evidence
        ),
        header_footer_exclusion_characters=sum(
            item.diagnostic.losses.header_footer_exclusion_characters for item in evidence
        ),
        order_inversions=inversions,
        comparable_order_pairs=pairs,
    )
    return metrics, losses
