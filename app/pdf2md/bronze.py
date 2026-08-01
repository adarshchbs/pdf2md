from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

import pymupdf
from pydantic import Field, model_validator

from app.pdf2md.pymupdf_runtime import open_document
from app.pdf2md.schema import SchemaModel

BRONZE_SCHEMA_VERSION = "1.0.0"


class PageSelection(SchemaModel):
    requested_pages: list[int] = Field(default_factory=list)
    annotated_pages: list[int] = Field(default_factory=list)
    context_pages: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pages(self) -> PageSelection:
        for name, pages in (
            ("requested_pages", self.requested_pages),
            ("annotated_pages", self.annotated_pages),
            ("context_pages", self.context_pages),
        ):
            if any(page < 1 for page in pages):
                raise ValueError(f"{name} must use one-based positive page numbers")
            if len(pages) != len(set(pages)) or pages != sorted(pages):
                raise ValueError(f"{name} must be sorted and unique")
        if not set(self.requested_pages).issubset(self.annotated_pages):
            raise ValueError("requested_pages must be contained in annotated_pages")
        if set(self.annotated_pages).intersection(self.context_pages):
            raise ValueError("annotated_pages and context_pages cannot overlap")
        return self

    @property
    def rendered_pages(self) -> list[int]:
        return sorted(set(self.annotated_pages) | set(self.context_pages))


class BronzeConfig(SchemaModel):
    render_dpi: int = Field(default=144, ge=72, le=600)
    liteparse_dpi: int = Field(default=150, ge=72, le=600)
    min_native_characters_per_page: int = Field(default=20, ge=1)
    liteparse_executable: str = "lit"


class BronzeArtifact(SchemaModel):
    path: str
    sha256: str
    size_bytes: int = Field(ge=0)


class BronzeManifest(SchemaModel):
    schema_version: str = BRONZE_SCHEMA_VERSION
    source_name: str
    source_path: str
    source_sha256: str
    source_size_bytes: int = Field(ge=1)
    source_page_count: int = Field(ge=1)
    selection: PageSelection
    config: BronzeConfig
    pymupdf_version: str
    liteparse_version: str
    artifacts: list[BronzeArtifact]


def generate_bronze_bundle(
    pdf_path: Path,
    output_dir: Path,
    selection: PageSelection,
    *,
    config: BronzeConfig | None = None,
) -> BronzeManifest:
    config = config or BronzeConfig()
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("bronze input must be a PDF")
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if output_dir.exists():
        raise FileExistsError(f"bronze output is immutable and already exists: {output_dir}")

    with open_document(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("encrypted PDFs are not supported")
        page_count = document.page_count
        if not selection.rendered_pages:
            all_pages = list(range(1, page_count + 1))
            selection = PageSelection(
                requested_pages=all_pages,
                annotated_pages=all_pages,
            )
        pages = selection.rendered_pages
        _validate_page_numbers(pages, page_count)
        _validate_native_text(document, pages, config.min_native_characters_per_page)

        output_dir.mkdir(parents=True)
        pages_dir = output_dir / "pages"
        pages_dir.mkdir()
        for page_number in pages:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(
                dpi=config.render_dpi,
                colorspace=pymupdf.csRGB,
                alpha=False,
                annots=True,
            )
            (pages_dir / f"page-{page_number:04d}.png").write_bytes(pixmap.tobytes("png"))

    target_pages = _liteparse_page_spec(pages)
    common_args = [
        config.liteparse_executable,
        "parse",
        str(pdf_path),
        "--no-ocr",
        "--target-pages",
        target_pages,
        "--dpi",
        str(config.liteparse_dpi),
        "--keep-headers-footers",
        "--quiet",
    ]
    raw_json_path = output_dir / "liteparse.json"
    _run_liteparse([
        *common_args,
        "--format",
        "json",
        "--extract-content-bounds",
        "--complexity",
        "--extract-text-metadata",
        "--extract-vector-graphics",
        "--extract-annotations",
        "--extract-form-fields",
        "--extract-structure-tree",
        "--output",
        str(raw_json_path),
    ])
    _validate_liteparse_json(raw_json_path, pages)

    markdown_path = output_dir / "liteparse.md"
    _run_liteparse([
        *common_args,
        "--format",
        "markdown",
        "--image-mode",
        "placeholder",
        "--output",
        str(markdown_path),
    ])
    text_path = output_dir / "liteparse.txt"
    _run_liteparse([
        *common_args,
        "--format",
        "text",
        "--output",
        str(text_path),
    ])

    version = _command_output([config.liteparse_executable, "--version"])
    artifact_paths = [*sorted(pages_dir.glob("*.png")), raw_json_path, markdown_path, text_path]
    manifest = BronzeManifest(
        source_name=pdf_path.name,
        source_path=str(pdf_path),
        source_sha256=_sha256(pdf_path),
        source_size_bytes=pdf_path.stat().st_size,
        source_page_count=page_count,
        selection=selection,
        config=config,
        pymupdf_version=pymupdf.VersionBind,
        liteparse_version=version,
        artifacts=[
            BronzeArtifact(
                path=str(path.relative_to(output_dir)),
                sha256=_sha256(path),
                size_bytes=path.stat().st_size,
            )
            for path in artifact_paths
        ],
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_bronze_bundle(output_dir: Path) -> BronzeManifest:
    manifest_path = output_dir / "manifest.json"
    manifest = BronzeManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest.artifacts:
        path = output_dir / artifact.path
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact.size_bytes:
            raise ValueError(f"artifact size mismatch: {artifact.path}")
        if _sha256(path) != artifact.sha256:
            raise ValueError(f"artifact hash mismatch: {artifact.path}")
    return manifest


def _validate_page_numbers(pages: list[int], page_count: int) -> None:
    invalid = [page for page in pages if page > page_count]
    if invalid:
        raise ValueError(f"selected pages exceed PDF page count {page_count}: {invalid}")


def _validate_native_text(document: pymupdf.Document, pages: list[int], min_characters: int) -> None:
    for page_number in pages:
        text = cast(str, document[page_number - 1].get_text("text")).strip()
        if len(text) < min_characters:
            raise ValueError(
                f"page {page_number} has only {len(text)} native text characters; "
                "scanned and image-only pages are excluded"
            )


def _liteparse_page_spec(pages: list[int]) -> str:
    if not pages:
        raise ValueError("at least one page must be selected")
    ranges: list[str] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _run_liteparse(args: list[str]) -> None:
    subprocess.run(args, check=True)


def _command_output(args: list[str]) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip() or result.stderr.strip()


def _validate_liteparse_json(path: Path, expected_pages: list[int]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_pages = [page["page"] for page in payload["pages"]]
    if actual_pages != expected_pages:
        raise ValueError(f"LiteParse returned pages {actual_pages}, expected {expected_pages}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
