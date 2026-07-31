from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

import app.comparison_router as comparison_router
from app.comparison_router import (
    WORK_LIMITER,
    catalog,
    document,
    documents,
    elements,
    evaluation,
    metadata,
    source_pdf,
)
from app.main import app
from app.pdf2md.comparison import ComparisonCatalog, ResolvedArtifact, roots_from_environment


def _partition(root: Path) -> None:
    path = root / "data/corpus"
    path.mkdir(parents=True)
    (path / "partition-v7.json").write_text(
        json.dumps({"documents": [{"id": "doc-1", "split": "train", "local_path": "pdfs/doc-1.pdf"}]}),
        encoding="utf-8",
    )


def test_catalog_enumerates_splits_and_reports_missing_artifacts(tmp_path: Path) -> None:
    _partition(tmp_path)
    catalog_instance = ComparisonCatalog(tmp_path)
    assert [document.document_id for document in catalog_instance.documents("train")] == ["doc-1"]
    comparison = catalog_instance.comparison("doc-1")
    assert comparison["gold"] is None
    assert comparison["candidate"]["status"] == "absent"  # type: ignore[index]


def test_catalog_rejects_paths_that_escape_artifact_root(tmp_path: Path) -> None:
    _partition(tmp_path)
    (tmp_path / "data/corpus/partition-v7.json").write_text(
        json.dumps({"documents": [{"id": "doc-1", "split": "train", "local_path": "../outside.pdf"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe local_path"):
        ComparisonCatalog(tmp_path)


def test_catalog_rejects_duplicate_ids_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "data/corpus"
    path.mkdir(parents=True)
    for partition_documents in (
        [
            {"id": "same", "split": "train", "local_path": "a.pdf"},
            {"id": "same", "split": "validation", "local_path": "b.pdf"},
        ],
        [
            {"id": "a", "split": "train", "local_path": "same.pdf"},
            {"id": "b", "split": "holdout", "local_path": "same.pdf"},
        ],
    ):
        (path / "partition-v7.json").write_text(
            json.dumps({"documents": partition_documents}), encoding="utf-8"
        )
        with pytest.raises(ValueError):
            ComparisonCatalog(tmp_path)


def test_symlink_roots_and_recursive_symlinks_are_rejected(tmp_path: Path) -> None:
    _partition(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(FileNotFoundError, match="symlink"):
        ComparisonCatalog(tmp_path, candidate_roots=(link,))
    (real / "nested-link").symlink_to(tmp_path, target_is_directory=True)
    assert (
        ComparisonCatalog(tmp_path, candidate_roots=(real,)).artifact("doc-1", "candidate", "elements").status
        == "present-invalid"
    )


def test_exact_metacharacter_document_id_does_not_glob(tmp_path: Path) -> None:
    path = tmp_path / "data/corpus"
    path.mkdir(parents=True)
    (path / "partition-v7.json").write_text(
        json.dumps({"documents": [{"id": "a[*]", "split": "train", "local_path": "a.pdf"}]}), encoding="utf-8"
    )
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "aX.parquet").write_bytes(b"not parquet")
    assert (
        ComparisonCatalog(tmp_path, candidate_roots=(root,)).artifact("a[*]", "candidate", "elements").status
        == "absent"
    )


def test_original_source_endpoint_serves_catalog_pdf_and_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _partition(tmp_path)
    source = tmp_path / "pdfs"
    source.mkdir()
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(source / "doc-1.pdf")
    partition = json.loads((tmp_path / "data/corpus/partition-v7.json").read_text(encoding="utf-8"))
    partition["documents"][0]["sha256"] = hashlib.sha256((source / "doc-1.pdf").read_bytes()).hexdigest()
    (tmp_path / "data/corpus/partition-v7.json").write_text(json.dumps(partition), encoding="utf-8")
    monkeypatch.setenv("PDF2MD_PROJECT_ROOT", str(tmp_path))
    client = TestClient(app)
    response = client.get("/comparison/documents/doc-1/source/original")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    full = response.content
    ranged = client.get("/comparison/documents/doc-1/source/original", headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert ranged.content == full[:10]
    assert ranged.headers["content-range"] == f"bytes 0-9/{len(full)}"
    assert (
        client.get(
            "/comparison/documents/doc-1/source/original", headers={"Range": "bytes=999999999-"}
        ).status_code
        == 416
    )
    head = client.head("/comparison/documents/doc-1/source/original")
    assert head.status_code == 200
    assert int(head.headers["content-length"]) == len(full)
    payload = client.get("/comparison/documents/doc-1").json()
    assert payload["original_source"]["status"] == "available"
    (source / "doc-1.pdf").unlink()
    assert client.get("/comparison/documents/doc-1/source/original").status_code == 404


def testparse_range_parser_rejects_non_ascii_digits_signs_and_whitespace() -> None:
    from app.comparison_router import parse_range

    for header in ("bytes=+1-2", "bytes= 1-2", "bytes=1 -2", "bytes=١-٢", "bytes=1-2,3-4", "bytes=1"):
        with pytest.raises(Exception) as caught:
            parse_range(header, 10)
        assert getattr(caught.value, "headers", {}).get("Content-Range") == "bytes */10"


def test_invalid_pdf_is_present_invalid(tmp_path: Path) -> None:
    _partition(tmp_path)
    root = tmp_path / "candidate"
    (root / "pdfs").mkdir(parents=True)
    (root / "pdfs/doc-1.pdf").write_bytes(b"not a PDF")
    assert (
        ComparisonCatalog(tmp_path, candidate_roots=(root,)).artifact("doc-1", "candidate", "pdf").status
        == "present-invalid"
    )


def test_candidate_pdf_may_differ_from_original_source_hash(tmp_path: Path) -> None:
    _partition(tmp_path)
    source = tmp_path / "pdfs"
    candidate = tmp_path / "candidate/pdfs"
    source.mkdir()
    candidate.mkdir(parents=True)
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(source / "doc-1.pdf")
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.new_page()
        pdf.save(candidate / "doc-1.pdf")
    partition_path = tmp_path / "data/corpus/partition-v7.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    partition["documents"][0]["sha256"] = hashlib.sha256((source / "doc-1.pdf").read_bytes()).hexdigest()
    partition_path.write_text(json.dumps(partition), encoding="utf-8")

    artifact = ComparisonCatalog(
        tmp_path,
        candidate_roots=(tmp_path / "candidate",),
    ).artifact("doc-1", "candidate", "pdf")
    assert artifact.status == "available"


def test_fast_summary_marks_presence_unvalidated_without_exposing_paths(tmp_path: Path) -> None:
    _partition(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "doc-1.parquet").write_bytes(b"validation happens on selection")
    payload = ComparisonCatalog(
        tmp_path,
        candidate_roots=(candidate,),
    ).comparison_summaries("train")[0]

    assert payload["candidate"]["status"] == "present-unvalidated"  # type: ignore[index]
    assert payload["candidate"]["available"] is True  # type: ignore[index]
    assert "path" not in payload["candidate"]  # type: ignore[operator]


def test_comparison_metadata_is_explicitly_non_blind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _partition(tmp_path)
    monkeypatch.setenv("PDF2MD_PROJECT_ROOT", str(tmp_path))
    response = TestClient(app).get("/comparison/metadata")
    assert response.status_code == 200
    assert response.json()["mode"] == "non-blind engineering comparison"
    assert response.json()["gold"] is None


def test_documents_returns_clear_404_when_partition_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PDF2MD_PROJECT_ROOT", str(tmp_path))
    response = TestClient(app).get("/comparison/documents")
    assert response.status_code == 404
    assert "partition-v7" in response.json()["detail"]


def test_stream_generator_closes_on_read_failure_and_early_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.pdf"
    path.write_bytes(b"0123456789")
    artifact = ResolvedArtifact(path, "available", None, root=tmp_path)
    expected = path.stat()
    original_read = comparison_router.os.read
    original_close = comparison_router.os.close
    closed: list[int] = []

    def tracking_close(fd: int) -> None:
        closed.append(fd)
        original_close(fd)

    def failing_read(fd: int, size: int) -> bytes:
        raise OSError("injected read failure")

    monkeypatch.setattr(comparison_router.os, "close", tracking_close)
    monkeypatch.setattr(comparison_router.os, "read", failing_read)

    async def consume_failure() -> None:
        generator = comparison_router.stream_artifact(artifact, expected, 0, 5)
        with pytest.raises(OSError, match="injected read failure"):
            await generator.__anext__()

    asyncio.run(consume_failure())
    assert closed
    monkeypatch.setattr(comparison_router.os, "read", original_read)

    async def consume_early_close() -> None:
        generator = comparison_router.stream_artifact(artifact, expected, 0, 5)
        assert await generator.__anext__() == b"01234"
        await generator.aclose()

    asyncio.run(consume_early_close())
    assert len(closed) >= 2


def test_frontend_root_and_health_api_have_distinct_precedence() -> None:
    client = TestClient(app)
    root_response = client.get("/")
    assert root_response.status_code == 200
    assert "<html" in root_response.text.lower()
    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    assert health_response.json()["message"] == "PDF Parser v4 is online"


def test_fastapi_handlers_are_async_and_work_is_bounded() -> None:
    assert all(
        inspect.iscoroutinefunction(handler)
        for handler in (catalog, documents, document, elements, evaluation, metadata, source_pdf)
    )
    assert WORK_LIMITER.total_tokens == 4


def test_default_roots_select_one_current_bundle_without_ignored_data(tmp_path: Path) -> None:
    (tmp_path / "data/benchmark-batch-semantic16/candidates").mkdir(parents=True)
    (tmp_path / "data/benchmark-batch-semantic15/candidates").mkdir(parents=True)
    (tmp_path / "data/silver-cycle2").mkdir(parents=True)
    assert roots_from_environment("PDF2MD_CANDIDATE_ROOTS", root=tmp_path) == (
        tmp_path / "data/benchmark-batch-semantic16/candidates",
    )
    assert roots_from_environment("PDF2MD_REFERENCE_ROOTS", root=tmp_path) == (
        tmp_path / "data/silver-cycle2",
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    assert roots_from_environment("PDF2MD_CANDIDATE_ROOTS", root=empty) == ()


def test_environment_roots_override_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "override"
    override.mkdir()
    monkeypatch.setenv("PDF2MD_CANDIDATE_ROOTS", str(override))
    assert roots_from_environment("PDF2MD_CANDIDATE_ROOTS", root=tmp_path) == (override,)


def test_partition_id_and_source_sha_aliases_resolve_exact_stems(tmp_path: Path) -> None:
    sha = "a" * 64
    partition = tmp_path / "data/corpus"
    partition.mkdir(parents=True)
    (partition / "partition-v7.json").write_text(
        json.dumps({
            "documents": [
                {"id": "partition-id", "split": "train", "local_path": "pdfs/source-name.pdf", "sha256": sha}
            ]
        }),
        encoding="utf-8",
    )
    roots = tmp_path / "artifacts"
    roots.mkdir()
    (roots / f"{sha}.parquet").write_bytes(b"not parquet")
    assert (
        ComparisonCatalog(tmp_path, candidate_roots=(roots,))
        .artifact("partition-id", "candidate", "elements")
        .status
        == "present-invalid"
    )


def test_default_roots_do_not_read_outside_project(tmp_path: Path) -> None:
    # Defaults are resolved beneath the supplied project root, never cwd or a
    # sibling checkout containing ignored benchmark data.
    assert roots_from_environment("PDF2MD_CANDIDATE_ROOTS", root=tmp_path) == ()
