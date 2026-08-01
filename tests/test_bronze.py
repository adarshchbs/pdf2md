from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from app.pdf2md.bronze import (
    BronzeConfig,
    PageSelection,
    generate_bronze_bundle,
    verify_bronze_bundle,
)


def create_pdf(path: Path) -> None:
    document = pymupdf.open()
    for page_number in (1, 2):
        page = document.new_page()
        page.insert_text(
            (72, 72),
            f"Page {page_number} contains enough born-digital text for deterministic bronze testing.",
        )
    document.save(path)
    document.close()


def test_page_selection_requires_complete_requested_subset() -> None:
    with pytest.raises(ValidationError, match="contained in annotated_pages"):
        PageSelection(requested_pages=[1, 2], annotated_pages=[1])


def test_generate_and_verify_bronze_bundle(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    output_dir = tmp_path / "bronze"
    create_pdf(pdf_path)
    selection = PageSelection(
        requested_pages=[1],
        annotated_pages=[1],
        context_pages=[2],
    )

    manifest = generate_bronze_bundle(
        pdf_path,
        output_dir,
        selection,
        config=BronzeConfig(render_dpi=72),
    )

    assert manifest.selection.rendered_pages == [1, 2]
    assert {artifact.path for artifact in manifest.artifacts} == {
        "pages/page-0001.png",
        "pages/page-0002.png",
        "liteparse.json",
        "liteparse.md",
        "liteparse.txt",
    }
    assert verify_bronze_bundle(output_dir) == manifest


def test_verify_bronze_detects_tampering(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    output_dir = tmp_path / "bronze"
    create_pdf(pdf_path)
    selection = PageSelection(requested_pages=[1], annotated_pages=[1])
    generate_bronze_bundle(pdf_path, output_dir, selection)
    (output_dir / "liteparse.txt").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact (size|hash) mismatch"):
        verify_bronze_bundle(output_dir)
