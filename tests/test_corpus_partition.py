import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.pdf2md.corpus_partition import (
    RATIOS,
    _read_records,  # pyright: ignore[reportPrivateUsage]
    _validate_checkpoint,  # pyright: ignore[reportPrivateUsage]
    build_corpus_partition,
    document_targets,
)


def _sha(index: int) -> str:
    return hashlib.sha256(f"document-{index}".encode()).hexdigest()


def _normalize(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def _refresh_deduplication_keys(document: dict[str, object]) -> None:
    issuer = document.get("issuer")
    producer = str(issuer or document.get("producer"))
    document["producer"] = producer
    normalized_title = _normalize(document["title"])
    title_family = _normalize(re.sub(r"\b(?:19|20)\d{2}\b", "", normalized_title))
    template_family = f"publisher-template:{_normalize(producer)}|{title_family}"
    document["template_family"] = template_family
    document["deduplication_keys"] = {
        "sha256": document["sha256"],
        "normalized_url": str(document["source_url"]).split("#", 1)[0].rstrip("/"),
        "normalized_title": normalized_title,
        "normalized_publisher": _normalize(producer),
        "template_family": template_family,
    }


def _document(index: int, *, title: str | None = None, issuer: str | None = None) -> dict[str, object]:
    document: dict[str, object] = {
        "id": f"doc-{index}",
        "title": title or f"Document {index}",
        "local_path": f"staging/doc-{index}.pdf",
        "sha256": _sha(index),
        "page_count": index + 1,
        "size_bytes": 1_000 + index,
        "category": "report" if index % 2 else "form",
        "status": "accepted",
        "provenance": "Synthetic partition-test fixture.",
        "producer": issuer or f"producer {index} example",
        "region": "test-region",
        "year": 2026,
        "language": "en",
        "layout_signals": {"synthetic": True},
        "source_url": f"https://producer-{index}.example/doc.pdf",
        "accuracy_inspection_status": "uninspected",
    }
    if issuer is not None:
        document["issuer"] = issuer
    _refresh_deduplication_keys(document)
    return document


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    corpus_path = root / "data/corpus/manifest.json"
    partition_path = root / "data/corpus/partition-v1.json"
    staging_path = root / "staging/accepted.json"
    corpus_documents = [_document(index) for index in range(8)]
    corpus_documents.extend([
        _document(100, title="LayoutLM: Pre-training of Text and Layout"),
        _document(101, title="LayoutLMv2: Multi-modal Pre-training"),
    ])
    _write_json(corpus_path, {"documents": corpus_documents})
    _write_json(
        partition_path,
        {
            "documents": [
                {**corpus_documents[0], "split": "dev"},
                {**corpus_documents[1], "split": "validation"},
                {**corpus_documents[2], "split": "holdout"},
            ]
        },
    )
    new_documents = [_document(index, issuer=f"Issuer {index}") for index in range(20, 32)]
    _write_json(staging_path, {"documents": new_documents})
    return corpus_path, partition_path, staging_path


def _difficult_checkpoint(root: Path, staging_path: Path) -> Path:
    stage = root / "data/corpus-expansion/v3-difficult"
    source_path = stage / "source_manifest.jsonl"
    accepted_path = stage / "accepted_manifest.jsonl"
    rejected_path = stage / "rejected_manifest.jsonl"
    summary_path = stage / "summary.json"
    documents = json.loads(staging_path.read_text(encoding="utf-8"))["documents"]
    sources: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for document in documents:
        source = {
            "slug": document["id"],
            "title": document["title"],
            "producer": document["producer"],
            "edition": document["id"],
            "source_url": document["source_url"],
            "source_kind": "test",
            "stratum": document["category"],
            "template_family": document["template_family"],
            "assignment": "unassigned",
        }
        sources.append(source)
        accepted.append({
            **source,
            "status": "accepted",
            "local_path": document["local_path"],
            "sha256": document["sha256"],
            "size_bytes": document["size_bytes"],
            "page_count": document["page_count"],
            "native_text_reliable": True,
            "useful_structural_signals": True,
            "validation_note": "Metadata-only uninspected checkpoint fixture.",
            "sampled_detected_table_count": 0,
            "rotated_page_count": 0,
            "landscape_page_count": 0,
            "form_field_count": 0,
        })
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8")
    accepted_path.write_text("".join(json.dumps(row) + "\n" for row in accepted), encoding="utf-8")
    rejected_path.write_text("", encoding="utf-8")
    _write_json(
        summary_path,
        {
            "task": 43,
            "source_count": len(sources),
            "accepted_count": len(accepted),
            "rejected_count": 0,
            "accepted_page_count": sum(int(row["page_count"]) for row in accepted),
            "accepted_size_bytes": sum(int(row["size_bytes"]) for row in accepted),
            "scope_guard": "No accuracy, bronze, silver, validation, or holdout artifacts were inspected.",
        },
    )
    return accepted_path


def test_builder_is_exact_family_safe_and_preserves_v1(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    partition_before = partition_v1.read_bytes()
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "data/corpus/manifest-v3.json",
        output_partition_path=tmp_path / "data/corpus/partition-v3.json",
        status="provisional",
    )

    assert result.document_counts == {"train": 12, "validation": 2, "holdout": 6}
    assert partition_v1.read_bytes() == partition_before
    output = json.loads(result.partition_path.read_text(encoding="utf-8"))
    by_id = {document["id"]: document for document in output["documents"]}
    assert by_id["doc-0"]["split"] == "train"
    assert by_id["doc-1"]["split"] == "validation"
    assert by_id["doc-2"]["split"] == "holdout"
    assert "doc-100" not in by_id
    assert "doc-101" not in by_id
    assert output["ideal_targets_family_reachable"] is True
    assert len(output["excluded_documents"]) == 2
    assert set(output["strata_summary"]) >= {"category", "page_count_band", "source_producer"}

    repeated = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "repeat/manifest-v3.json",
        output_partition_path=tmp_path / "repeat/partition-v3.json",
        status="provisional",
    )
    repeated_output = json.loads(repeated.partition_path.read_text(encoding="utf-8"))
    assert repeated_output["documents"] == output["documents"]
    assert repeated_output["summary"] == output["summary"]
    assert repeated_output["selected_document_targets"] == output["selected_document_targets"]


def test_builder_keeps_publisher_family_atomic(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0].update({
        "issuer": "Shared Publisher",
        "title": "Shared Publisher Annual Report 2023",
    })
    staging["documents"][1].update({
        "issuer": "Shared Publisher",
        "title": "Shared Publisher Annual Report 2024",
    })
    _refresh_deduplication_keys(staging["documents"][0])
    _refresh_deduplication_keys(staging["documents"][1])
    _write_json(staging_path, staging)
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        status="provisional",
    )
    output = json.loads(result.partition_path.read_text(encoding="utf-8"))
    shared = [
        document for document in output["documents"] if document["source_producer"] == "shared publisher"
    ]
    assert len(shared) == 2
    assert len({document["split"] for document in shared}) == 1


def test_builder_forces_new_member_of_existing_train_family_to_train(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0].update({"issuer": "producer 0 example", "title": "Document 0 2026"})
    _refresh_deduplication_keys(staging["documents"][0])
    _write_json(staging_path, staging)
    result = build_corpus_partition(
        root_dir=tmp_path,
        corpus_manifest_path=corpus_path,
        previous_partition_path=partition_v1,
        new_manifest_paths=[staging_path],
        output_manifest_path=tmp_path / "manifest-v3.json",
        output_partition_path=tmp_path / "partition-v3.json",
        status="provisional",
    )
    output = json.loads(result.partition_path.read_text(encoding="utf-8"))
    member = next(document for document in output["documents"] if document["id"] == "doc-20")
    assert member["split"] == "train"
    assert member["assignment_origin"] == "new-v3-existing-family"
    assert output["new_documents_forced_by_existing_family"] == 1


def test_builder_rejects_inspected_new_candidate(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0]["accuracy_inspected"] = True
    _write_json(staging_path, staging)
    with pytest.raises(ValueError, match="forbidden evidence flags must be boolean false"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_never_overwrites_versioned_outputs(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    output = tmp_path / "partition-v3.json"
    output.write_text("sealed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must not already exist"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=output,
            status="provisional",
        )


def test_builder_rejects_stale_checkpoint(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    checkpoint.with_name("source_manifest.jsonl").touch()

    with pytest.raises(ValueError, match="stale relative to source"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[checkpoint],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    checkpoint = _difficult_checkpoint(tmp_path, staging_path)
    summary_path = checkpoint.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["accepted_count"] += 1
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="completeness counts"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[checkpoint],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_final_requires_expanded_inputs(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="final status requires"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="final",
        )


def test_target_ratios_are_60_10_30_at_816_documents() -> None:
    assert RATIOS == {"train": 0.60, "validation": 0.10, "holdout": 0.30}
    assert document_targets(816) == {"train": 490, "validation": 81, "holdout": 245}
    assert document_targets(734) == {"train": 441, "validation": 73, "holdout": 220}


def test_builder_rejects_incomplete_new_metadata(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    del staging["documents"][0]["provenance"]
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="missing provenance"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_duplicate_normalized_url(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][1]["source_url"] = staging["documents"][0]["source_url"] + "#copy"
    _refresh_deduplication_keys(staging["documents"][1])
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="duplicate URL"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_duplicate_title_and_publisher(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][1].update({
        "title": staging["documents"][0]["title"],
        "issuer": staging["documents"][0]["issuer"],
    })
    _refresh_deduplication_keys(staging["documents"][1])
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="duplicate normalized title and publisher"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_string_inspection_flag(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0]["accuracy_inspected"] = "true"
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="forbidden evidence flags must be boolean false"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_final_v3_artifacts_are_fresh_complete_exact_and_immutable() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/corpus/manifest-v3.json"
    partition_path = root / "data/corpus/partition-v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    previous = json.loads((root / "data/corpus/partition-v2.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "21a6e8d797ae748beb67ea50c896c983d1b07d504e11f17ccf5d70bbb66c8cf7"
    )
    assert hashlib.sha256(partition_path.read_bytes()).hexdigest() == (
        "998f26f8e03ce6769a389cdfb13b8968f672958c8d93f262ce4c4d616497cce7"
    )
    assert manifest["status"] == partition["status"] == "final"
    assert manifest["documents"] == partition["documents"]
    assert len(partition["documents"]) == 787
    assert (
        partition["selected_document_targets"]
        == document_targets(787)
        == {
            "train": 472,
            "validation": 79,
            "holdout": 236,
        }
    )
    assert manifest["selection"] == {
        "previously_active_documents": 408,
        "new_accuracy_uninspected_documents": 379,
        "retained_unique_documents": 787,
    }

    documents = partition["documents"]
    assert len({document["id"] for document in documents}) == len(documents)
    assert len({document["sha256"] for document in documents}) == len(documents)
    assert not any(re.search(r"layout\s*lm(?:v?2)?", json.dumps(document), re.I) for document in documents)
    by_sha = {document["sha256"]: document for document in documents}
    assert all(by_sha[document["sha256"]]["split"] == document["split"] for document in previous["documents"])

    new_documents = [
        document for document in documents if str(document["assignment_origin"]).startswith("new-v3")
    ]
    family_splits: dict[str, set[str]] = {}
    for document in new_documents:
        family_splits.setdefault(str(document["family_id"]), set()).add(str(document["split"]))
    assert all(len(splits) == 1 for splits in family_splits.values())


def test_final_v4_artifacts_consolidate_all_finalized_inputs_without_leakage() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/corpus/manifest-v4.json"
    partition_path = root / "data/corpus/partition-v4.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    previous = json.loads((root / "data/corpus/partition-v2.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "5b4bad443480cf12cc6362bb96256965cba0823fad6d7dc838d03c493c184de2"
    )
    assert hashlib.sha256(partition_path.read_bytes()).hexdigest() == (
        "af7b196bdeb61cfdf8613baba8c64b63a31bed57c02f27d3d2de8897148e6f54"
    )
    assert manifest["schema_version"] == partition["schema_version"] == "4.0"
    assert manifest["status"] == partition["status"] == "final"
    assert manifest["documents"] == partition["documents"]
    assert (
        partition["selected_document_targets"]
        == document_targets(851)
        == {
            "train": 511,
            "validation": 85,
            "holdout": 255,
        }
    )
    assert manifest["selection"] == {
        "previously_active_documents": 408,
        "new_accuracy_uninspected_documents": 443,
        "retained_unique_documents": 851,
    }
    assert {descriptor["path"]: descriptor["sha256"] for descriptor in partition["inputs"]} == {
        "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl": (
            "ef775c528961ee1a95dbc8de84cbaa04e5a2f3e921cfba4eb45512b89cd707a8"
        ),
        "data/corpus-expansion/v3-financial/accepted_manifest.jsonl": (
            "4a258db265198b63b5668be1d54e7155ebfbd5921a0fba8c3da28e68cb8d9f8f"
        ),
        "data/corpus-expansion/v3-gap/accepted_manifest.jsonl": (
            "7ded8c2545fb4932e66d7e0152ae25d7f7da3ed6c374e643c7c0a557854f6372"
        ),
    }

    documents = partition["documents"]
    assert len(documents) == 851 >= 816
    assert len({document["id"] for document in documents}) == len(documents)
    assert len({document["sha256"] for document in documents}) == len(documents)
    assert sum(int(document["page_count"]) for document in documents) == 101_187
    assert not any(re.search(r"layout\s*lm(?:v?2)?", json.dumps(document), re.I) for document in documents)
    by_sha = {document["sha256"]: document for document in documents}
    assert all(by_sha[document["sha256"]]["split"] == document["split"] for document in previous["documents"])

    new_documents = [
        document for document in documents if str(document["assignment_origin"]).startswith("new-v4")
    ]
    assert len(new_documents) == 443
    assert len({document["deduplication_keys"]["normalized_url"] for document in new_documents}) == 443
    assert len({document["deduplication_keys"]["normalized_title"] for document in new_documents}) == 443
    family_splits: dict[str, set[str]] = {}
    for document in new_documents:
        family_splits.setdefault(str(document["family_id"]), set()).add(str(document["split"]))
    assert all(len(splits) == 1 for splits in family_splits.values())

    assert hashlib.sha256((root / "data/corpus/manifest-v3.json").read_bytes()).hexdigest() == (
        "21a6e8d797ae748beb67ea50c896c983d1b07d504e11f17ccf5d70bbb66c8cf7"
    )
    assert hashlib.sha256((root / "data/corpus/partition-v3.json").read_bytes()).hexdigest() == (
        "998f26f8e03ce6769a389cdfb13b8968f672958c8d93f262ce4c4d616497cce7"
    )


def test_final_v5_artifacts_use_corrected_financial_ledger_without_leakage() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/corpus/manifest-v5.json"
    partition_path = root / "data/corpus/partition-v5.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    previous = json.loads((root / "data/corpus/partition-v2.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "658f5591674b1ba50c6aa5a261ee9cb8ece1a17c8bc39ccb3fd30f99f980f3f2"
    )
    assert hashlib.sha256(partition_path.read_bytes()).hexdigest() == (
        "b576123750f6f02de53e14633012e337111ed05f75b16af6c6f7acf221d4b9eb"
    )
    assert manifest["schema_version"] == partition["schema_version"] == "5.0"
    assert manifest["status"] == partition["status"] == "final"
    assert manifest["documents"] == partition["documents"]
    assert len(partition["documents"]) == 851 >= 816
    assert (
        partition["selected_document_targets"]
        == document_targets(851)
        == {
            "train": 511,
            "validation": 85,
            "holdout": 255,
        }
    )
    assert manifest["selection"] == {
        "previously_active_documents": 408,
        "new_accuracy_uninspected_documents": 443,
        "retained_unique_documents": 851,
    }
    assert {descriptor["path"]: descriptor["sha256"] for descriptor in partition["inputs"]} == {
        "data/corpus-expansion/v3-difficult/accepted_manifest.jsonl": (
            "ef775c528961ee1a95dbc8de84cbaa04e5a2f3e921cfba4eb45512b89cd707a8"
        ),
        "data/corpus-expansion/v3-financial-r2/accepted_manifest.jsonl": (
            "4a258db265198b63b5668be1d54e7155ebfbd5921a0fba8c3da28e68cb8d9f8f"
        ),
        "data/corpus-expansion/v3-gap/accepted_manifest.jsonl": (
            "7ded8c2545fb4932e66d7e0152ae25d7f7da3ed6c374e643c7c0a557854f6372"
        ),
    }

    documents = partition["documents"]
    by_sha = {document["sha256"]: document for document in documents}
    assert all(by_sha[document["sha256"]]["split"] == document["split"] for document in previous["documents"])
    new_documents = [
        document for document in documents if str(document["assignment_origin"]).startswith("new-v5")
    ]
    family_splits: dict[str, set[str]] = {}
    for document in new_documents:
        family_splits.setdefault(str(document["family_id"]), set()).add(str(document["split"]))
    assert len(new_documents) == 443
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert hashlib.sha256((root / "data/corpus/manifest-v4.json").read_bytes()).hexdigest() == (
        "5b4bad443480cf12cc6362bb96256965cba0823fad6d7dc838d03c493c184de2"
    )
    assert hashlib.sha256((root / "data/corpus/partition-v4.json").read_bytes()).hexdigest() == (
        "af7b196bdeb61cfdf8613baba8c64b63a31bed57c02f27d3d2de8897148e6f54"
    )


def test_builder_rejects_repeated_sha_even_when_path_matches(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    duplicate = dict(staging["documents"][0])
    duplicate["id"] = "duplicate-id"
    staging["documents"].append(duplicate)
    _write_json(staging_path, staging)

    with pytest.raises(ValueError, match="duplicate SHA-256"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_builder_rejects_unknown_output_revision(tmp_path: Path) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="explicitly supported v3, v4, or v5"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v6.json",
            output_partition_path=tmp_path / "partition-v6.json",
            status="provisional",
        )


def test_json_array_records_must_be_objects(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_json(path, {"documents": [{"id": "valid"}, "not-an-object"]})
    with pytest.raises(ValueError, match=r"documents\[1\]"):
        _read_records(path)


@pytest.mark.parametrize("invalid_false", [0, "false", None])
def test_forbidden_evidence_flags_require_boolean_false(tmp_path: Path, invalid_false: object) -> None:
    corpus_path, partition_v1, staging_path = _fixture(tmp_path)
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    staging["documents"][0]["accuracy_inspected"] = invalid_false
    _write_json(staging_path, staging)
    with pytest.raises(ValueError, match="forbidden evidence flags must be boolean false"):
        build_corpus_partition(
            root_dir=tmp_path,
            corpus_manifest_path=corpus_path,
            previous_partition_path=partition_v1,
            new_manifest_paths=[staging_path],
            output_manifest_path=tmp_path / "manifest-v3.json",
            output_partition_path=tmp_path / "partition-v3.json",
            status="provisional",
        )


def test_corrected_financial_checkpoint_has_exact_source_outcome_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    defective = root / "data/corpus-expansion/v3-financial/accepted_manifest.jsonl"
    corrected = root / "data/corpus-expansion/v3-financial-r2/accepted_manifest.jsonl"

    with pytest.raises(ValueError, match="required source/accepted/rejected counts"):
        _validate_checkpoint(defective, _read_records(defective))

    descriptor = _validate_checkpoint(corrected, _read_records(corrected))
    assert descriptor is not None
    assert descriptor["kind"] == "financial"
    assert corrected.read_bytes() == defective.read_bytes()
