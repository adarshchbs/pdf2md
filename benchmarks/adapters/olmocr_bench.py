from __future__ import annotations

from pathlib import Path

from benchmarks.canonical import CanonicalPage, validate_page_collection


def export_olmocr_bench(
    pages: list[CanonicalPage],
    output_dir: Path,
    *,
    extension: str = ".md",
    overwrite: bool = False,
) -> list[Path]:
    """Write canonical page Markdown verbatim in deterministic per-page files."""
    if extension not in {".md", ".txt"}:
        raise ValueError("olmOCR bench output extension must be .md or .txt")
    validate_page_collection(pages, require_canonical_order=False)
    ordered = sorted(pages, key=lambda page: (page.document_id, page.page_index))
    paths = [
        output_dir / f"{_safe_document_id(page.document_id)}__page_{page.page_index:06d}{extension}"
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
        path.write_text(page.markdown, encoding="utf-8", newline="\n")
    return paths


def _safe_document_id(document_id: str) -> str:
    if document_id in {".", ".."} or not document_id:
        raise ValueError("document_id cannot be used as an output basename")
    if Path(document_id).name != document_id or "\\" in document_id:
        raise ValueError(f"document_id cannot contain path separators: {document_id!r}")
    return document_id
