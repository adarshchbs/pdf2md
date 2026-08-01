from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarks.canonical import CanonicalPage, validate_page_collection


def export_olmocr_bench(
    pages: list[CanonicalPage],
    output_dir: Path,
    *,
    extension: str = ".md",
    overwrite: bool = False,
    file_mapping: Mapping[tuple[str, int], Path] | None = None,
    repeat: int = 1,
) -> list[Path]:
    """Write canonical page Markdown verbatim in deterministic per-page files."""
    if extension not in {".md", ".txt"}:
        raise ValueError("olmOCR bench output extension must be .md or .txt")
    if repeat < 1:
        raise ValueError("olmOCR bench repeat must be positive")
    if file_mapping is not None and extension != ".md":
        raise ValueError("official olmOCR bench candidate files must use .md")
    validate_page_collection(pages, require_canonical_order=False)
    ordered = sorted(pages, key=lambda page: (page.document_id, page.page_index))
    keys = {(page.document_id, page.page_index) for page in ordered}
    if file_mapping is not None and set(file_mapping) != keys:
        raise ValueError("mapping IDs do not match canonical page IDs")
    paths = [
        _official_candidate_path(output_dir, file_mapping[(page.document_id, page.page_index)], page, repeat)
        if file_mapping is not None
        else output_dir / f"{_safe_document_id(page.document_id)}__page_{page.page_index:06d}{extension}"
        for page in ordered
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("deterministic olmOCR output names collide")
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(existing[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    for page, path in zip(ordered, paths, strict=True):
        # CanonicalPage guarantees that an HTML table representation occurs exactly in page.markdown.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page.markdown, encoding="utf-8", newline="\n")
    return paths


def _official_candidate_path(
    output_dir: Path,
    relative_pdf_path: Path,
    page: CanonicalPage,
    repeat: int,
) -> Path:
    if relative_pdf_path.is_absolute() or ".." in relative_pdf_path.parts:
        raise ValueError(f"relative PDF path must stay below the corpus root: {relative_pdf_path}")
    if relative_pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"relative PDF path must use the .pdf suffix: {relative_pdf_path}")
    filename = f"{relative_pdf_path.stem}_pg{page.page_index + 1}_repeat{repeat}.md"
    return output_dir / relative_pdf_path.parent / filename


def _safe_document_id(document_id: str) -> str:
    if document_id in {".", ".."} or not document_id:
        raise ValueError("document_id cannot be used as an output basename")
    if Path(document_id).name != document_id or "\\" in document_id:
        raise ValueError(f"document_id cannot contain path separators: {document_id!r}")
    return document_id
