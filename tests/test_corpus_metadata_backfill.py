import hashlib
import json
from pathlib import Path
from shutil import copy2

import pytest

from app.pdf2md.corpus_metadata_backfill import (
    EXPECTED_INPUT_SHA256,
    PROTECTED_FIELDS,
    apply_proposal,
    build_proposal,
    exact_language_evidence,
    exact_url_evidence,
    protected_membership_digest,
    render_report,
    validate_proposal,
)


def _document(*, source_url: str, title: str = "Document") -> dict[str, object]:
    return {
        "id": "doc",
        "sha256": "0" * 64,
        "split": "train",
        "family_id": "family",
        "source_url": source_url,
        "title": title,
    }


def _copy_v6_contract(target: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    for relative in (*EXPECTED_INPUT_SHA256, "data/corpus-metadata-backfill-v5-task57.json"):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source / relative, destination)


def _apply_v6(root: Path) -> tuple[Path, Path]:
    return apply_proposal(
        root=root,
        proposal_path=root / "data/corpus-metadata-backfill-v5-task57.json",
        output_manifest_path=root / "data/corpus/manifest-v6.json",
        output_partition_path=root / "data/corpus/partition-v6.json",
        revision="v6",
    )


def test_restricted_government_domains_are_exact_but_generic_cctlds_are_not() -> None:
    us = exact_url_evidence(_document(source_url="https://nvlpubs.nist.gov/report.pdf"))
    uk = exact_url_evidence(_document(source_url="https://assets.publishing.service.gov.uk/report.pdf"))
    commercial = exact_url_evidence(_document(source_url="https://publisher.co.uk/report.pdf"))

    assert {(item.field, item.value) for item in us} == {
        ("country", "United States"),
        ("region", "North America"),
    }
    assert {(item.field, item.value) for item in uk} == {
        ("country", "United Kingdom"),
        ("region", "Europe"),
    }
    assert commercial == []


def test_language_requires_concordant_declared_title_and_url() -> None:
    exact = exact_language_evidence(
        _document(
            source_url="https://example.org/report-English.pdf",
            title="Annual Report English",
        )
    )
    title_only = exact_language_evidence(
        _document(source_url="https://example.org/report.pdf", title="Annual Report English")
    )
    ambiguous_subject = exact_language_evidence(
        _document(source_url="https://example.org/arabic-chart.pdf", title="Arabic code chart")
    )

    assert [(item.field, item.value) for item in exact] == [("language_mix", "English")]
    assert title_only == []
    assert ambiguous_subject == []


def test_v5_proposal_is_exact_deterministic_and_preserves_membership() -> None:
    root = Path(__file__).resolve().parents[1]
    first = build_proposal(root)
    second = build_proposal(root)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["counts"] == {
        "documents": 851,
        "proposed_field_updates": 225,
        "proposed_documents": 130,
        "generic_before": {"country": 556, "region": 513, "language_mix": 796},
        "generic_after_proposal": {"country": 439, "region": 415, "language_mix": 786},
        "generic_reduction": {"country": 117, "region": 98, "language_mix": 10},
        "unresolved_fields": 1640,
    }
    assert first["conflicts"] == []
    assert len(validate_proposal(root, first)) == 851
    assert all(row["confidence"] == "exact" for row in first["exact_mappings"])
    assert all(row["field"] not in PROTECTED_FIELDS for row in first["exact_mappings"])
    assert len({row["sha256"] for row in first["exact_mappings"]}) == 130
    assert "No immutable manifest/partition revision is emitted" in render_report(first)


def test_input_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/corpus/manifest-v5.json"
    target = tmp_path / "data/corpus/manifest-v5.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="metadata input SHA-256 mismatch"):
        build_proposal(tmp_path)


def test_v6_rejects_unknown_revision(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="unsupported metadata backfill revision"):
        apply_proposal(
            root=root,
            proposal_path=root / "data/corpus-metadata-backfill-v5-task57.json",
            output_manifest_path=tmp_path / "manifest-v6.json",
            output_partition_path=tmp_path / "partition-v6.json",
            revision="v7",
        )


def test_v6_rejects_stale_proposal_and_input(tmp_path: Path) -> None:
    root = tmp_path / "contract"
    _copy_v6_contract(root)
    proposal_path = root / "data/corpus-metadata-backfill-v5-task57.json"
    proposal_path.write_bytes(proposal_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="proposal SHA-256 is stale"):
        _apply_v6(root)

    root = tmp_path / "stale-input"
    _copy_v6_contract(root)
    source_path = root / "data/corpus-expansion/v3-gap/source_manifest.jsonl"
    source_path.write_bytes(source_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="proposal is stale relative"):
        _apply_v6(root)


def test_v6_rejects_unsupported_evidence_and_conflicts() -> None:
    root = Path(__file__).resolve().parents[1]
    proposal = json.loads((root / "data/corpus-metadata-backfill-v5-task57.json").read_text())
    proposal["conflicts"] = [{"field": "country"}]
    with pytest.raises(ValueError, match="unsupported evidence conflicts"):
        validate_proposal(root, proposal)

    proposal = json.loads((root / "data/corpus-metadata-backfill-v5-task57.json").read_text())
    proposal["exact_mappings"][0]["rule_id"] = "organization-name-guess"
    with pytest.raises(ValueError, match="unsupported metadata evidence rule"):
        validate_proposal(root, proposal)


def test_v6_application_is_byte_deterministic_and_membership_immutable(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    outputs: list[tuple[bytes, bytes]] = []
    for root in roots:
        _copy_v6_contract(root)
        manifest_path, partition_path = _apply_v6(root)
        outputs.append((manifest_path.read_bytes(), partition_path.read_bytes()))
    assert outputs[0] == outputs[1]

    manifest_v5 = json.loads((roots[0] / "data/corpus/manifest-v5.json").read_text())
    manifest_v6 = json.loads(outputs[0][0])
    old_documents = manifest_v5["documents"]
    new_documents = manifest_v6["documents"]
    assert protected_membership_digest(old_documents) == protected_membership_digest(new_documents)
    assert all(
        all(old[field] == new[field] for field in PROTECTED_FIELDS)
        for old, new in zip(old_documents, new_documents, strict=True)
    )
    changes: list[tuple[str, str]] = []
    for old, new in zip(old_documents, new_documents, strict=True):
        for field in set(old) | set(new):
            if old.get(field) != new.get(field):
                changes.append((str(old["sha256"]), field))
    assert len(changes) == 225
    assert {field for _sha256, field in changes} == {"country", "region", "language_mix"}
    assert hashlib.sha256(outputs[0][0]).hexdigest() == (
        "7cf1941e089e8fad73a953cfd467bf616a989b824ae5a5bc6069cef40098dab4"
    )
    assert hashlib.sha256(outputs[0][1]).hexdigest() == (
        "a1d2423c5a34f720132835d886bd9fa32211db8646a7469441dda84ad29b612b"
    )
    source_root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((source_root / "data/corpus/manifest-v5.json").read_bytes()).hexdigest() == (
        "658f5591674b1ba50c6aa5a261ee9cb8ece1a17c8bc39ccb3fd30f99f980f3f2"
    )
    assert hashlib.sha256((source_root / "data/corpus/partition-v5.json").read_bytes()).hexdigest() == (
        "b576123750f6f02de53e14633012e337111ed05f75b16af6c6f7acf221d4b9eb"
    )
