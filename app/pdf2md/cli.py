from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import click
from pydantic import TypeAdapter

from app.pdf2md.adjudication import (
    AdjudicationBundle,
    AdjudicationDecision,
    AdjudicationKey,
    BlindAdjudicationPacket,
    apply_adjudication_decisions,
    build_blind_adjudication_bundle,
)
from app.pdf2md.adjudication_material import reproduce_semantic11_blind_material
from app.pdf2md.benchmark_batch import run_benchmark_batch, verify_benchmark_run
from app.pdf2md.blind_v3 import build_semantic12_blind_v3, read_protected_seed
from app.pdf2md.blind_v4 import build_blind_v4
from app.pdf2md.blind_v4 import read_protected_seed as read_v4_protected_seed
from app.pdf2md.blind_v5 import build_blind_v5
from app.pdf2md.blind_v5 import read_protected_seed as read_v5_protected_seed
from app.pdf2md.blind_v6 import build_blind_v6
from app.pdf2md.blind_v6 import read_protected_seed as read_v6_protected_seed
from app.pdf2md.blind_v7 import build_blind_v7
from app.pdf2md.blind_v7 import read_protected_seed as read_v7_protected_seed
from app.pdf2md.bronze import (
    BronzeConfig,
    PageSelection,
    generate_bronze_bundle,
    verify_bronze_bundle,
)
from app.pdf2md.corpus_partition import build_corpus_partition, verify_corpus_partition_descriptor
from app.pdf2md.engine import extract_document_with_catalog, render_document
from app.pdf2md.evaluation import EvaluationReport, evaluate_document, evaluate_reference_churn
from app.pdf2md.pymupdf_tables import extract_document_table_elements
from app.pdf2md.schema import read_document_elements, write_document_elements
from app.pdf2md.source_catalog import write_document_with_source_catalog


@click.group()
def cli() -> None:
    """Build bronze data and extract structure-aware PDF tables."""


@cli.command("build-corpus-partition")
@click.option(
    "--root-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
)
@click.option(
    "--corpus-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/corpus/manifest-v2.json"),
    show_default=True,
)
@click.option(
    "--previous-partition",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/corpus/partition-v2.json"),
    show_default=True,
)
@click.option(
    "--new-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
    help="Accuracy-uninspected staging manifest; repeat in stable desired input order.",
)
@click.option(
    "--revision-sidecar",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "Immutable checkpoint revision sidecar for each new manifest, in matching order; "
        "required for final builds."
    ),
)
@click.option(
    "--expected-revision-sidecar-sha256",
    multiple=True,
    help="Externally trusted SHA-256 for each immutable checkpoint sidecar, in matching order.",
)
@click.option("--output-manifest", type=click.Path(path_type=Path), required=True)
@click.option("--output-partition", type=click.Path(path_type=Path), required=True)
@click.option(
    "--output-descriptor",
    type=click.Path(path_type=Path),
    help="Immutable commit descriptor, published last; required for final builds.",
)
@click.option("--revision", type=click.Choice(("v3", "v4", "v5")))
@click.option("--status", type=click.Choice(("provisional", "final")), required=True)
def build_corpus_partition_command(
    root_dir: Path,
    corpus_manifest: Path,
    previous_partition: Path,
    new_manifest: tuple[Path, ...],
    revision_sidecar: tuple[Path, ...],
    expected_revision_sidecar_sha256: tuple[str, ...],
    output_manifest: Path,
    output_partition: Path,
    output_descriptor: Path | None,
    revision: str | None,
    status: str,
) -> None:
    """Build an immutable, family-safe corpus partition from metadata manifests."""
    authenticate_inputs = bool(revision_sidecar or expected_revision_sidecar_sha256) or status == "final"
    if status == "final" and output_descriptor is None:
        raise click.ClickException("final status requires an immutable output descriptor")
    if status == "final" and not revision_sidecar:
        raise click.ClickException(
            "final status requires externally SHA-pinned immutable checkpoint sidecars"
        )
    if authenticate_inputs and len(revision_sidecar) != len(new_manifest):
        raise click.ClickException("every new manifest requires exactly one revision sidecar")
    if authenticate_inputs and len(expected_revision_sidecar_sha256) != len(revision_sidecar):
        raise click.ClickException("every revision sidecar requires an externally supplied SHA-256")
    try:
        result = build_corpus_partition(
            root_dir=root_dir.resolve(),
            corpus_manifest_path=corpus_manifest.resolve(),
            previous_partition_path=previous_partition.resolve(),
            new_manifest_paths=[path.resolve() for path in new_manifest],
            output_manifest_path=output_manifest.resolve(),
            output_partition_path=output_partition.resolve(),
            output_descriptor_path=(output_descriptor.resolve() if output_descriptor is not None else None),
            revision=(cast(Literal["v3", "v4", "v5"], revision) if revision is not None else None),
            status=cast(Literal["provisional", "final"], status),
            revision_sidecar_paths=[path.resolve() for path in revision_sidecar],
            expected_revision_sidecar_sha256=list(expected_revision_sidecar_sha256),
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    counts = ", ".join(f"{split}={count}" for split, count in result.document_counts.items())
    click.echo(f"wrote {result.status} partition: {counts}")


@cli.command("verify-corpus-partition")
@click.argument("descriptor_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--expected-descriptor-sha256",
    required=True,
    help="Externally trusted SHA-256 of the immutable output descriptor.",
)
@click.option(
    "--root-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
)
def verify_corpus_partition_command(
    descriptor_path: Path,
    expected_descriptor_sha256: str,
    root_dir: Path,
) -> None:
    """Verify a corpus descriptor, its parents, inputs, and outputs."""
    try:
        descriptor = verify_corpus_partition_descriptor(
            descriptor_path.resolve(),
            root_dir=root_dir.resolve(),
            expected_descriptor_sha256=expected_descriptor_sha256,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"verified corpus partition {descriptor['revision_id']}")


@cli.command("bronze")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--requested-page", type=int, multiple=True)
@click.option("--annotated-page", type=int, multiple=True)
@click.option("--context-page", type=int, multiple=True)
@click.option("--dpi", type=click.IntRange(72, 600), default=144, show_default=True)
def bronze_command(
    pdf_path: Path,
    output_dir: Path,
    requested_page: tuple[int, ...],
    annotated_page: tuple[int, ...],
    context_page: tuple[int, ...],
    dpi: int,
) -> None:
    """Create an immutable image and LiteParse bronze bundle."""
    selection = PageSelection(
        requested_pages=sorted(requested_page),
        annotated_pages=sorted(annotated_page),
        context_pages=sorted(context_page),
    )
    manifest = generate_bronze_bundle(
        pdf_path,
        output_dir,
        selection,
        config=BronzeConfig(render_dpi=dpi),
    )
    click.echo(f"created {len(manifest.artifacts)} artifacts for {manifest.source_name}")


@cli.command("verify-bronze")
@click.argument("bundle_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def verify_bronze_command(bundle_dir: Path) -> None:
    """Verify every artifact against an immutable bronze manifest."""
    manifest = verify_bronze_bundle(bundle_dir)
    click.echo(f"verified {len(manifest.artifacts)} artifacts")


@cli.command("benchmark-batch")
@click.argument("manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("baseline_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--corpus-partition",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Trusted corpus partition used to prove every selection is TRAIN.",
)
@click.option(
    "--expected-corpus-partition-sha256",
    required=True,
    help="Externally trusted SHA-256 of the corpus partition.",
)
@click.option(
    "--root-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
)
def benchmark_batch_command(
    manifest_path: Path,
    reference_dir: Path,
    baseline_candidate_dir: Path,
    output_dir: Path,
    corpus_partition: Path,
    expected_corpus_partition_sha256: str,
    root_dir: Path,
) -> None:
    """Generate and gate a verified TRAIN-only benchmark batch."""
    try:
        result = run_benchmark_batch(
            manifest_path,
            reference_dir,
            baseline_candidate_dir,
            output_dir,
            root_dir=root_dir,
            corpus_partition_path=corpus_partition,
            expected_corpus_partition_sha256=expected_corpus_partition_sha256,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    summary = (
        f"wrote {result.aggregate.document_count} documents to {result.output_dir}; "
        f"regression gate {result.regression_gate.status}"
    )
    if result.regression_gate.status == "blocked":
        raise click.ClickException(summary)
    click.echo(summary)


@cli.command("verify-benchmark-run")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--expected-run-id", help="Trusted externally supplied run ID.")
@click.option(
    "--expected-anchor",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Trusted externally published anchor record.",
)
def verify_benchmark_run_command(
    run_dir: Path,
    expected_run_id: str | None,
    expected_anchor: Path | None,
) -> None:
    """Verify an archive, optionally against an externally trusted anchor."""
    try:
        provenance = verify_benchmark_run(
            run_dir,
            expected_run_id=expected_run_id,
            expected_anchor=expected_anchor,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"verified benchmark run {provenance.run_id}")


@cli.command("evaluate")
@click.argument("candidate_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reference_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def evaluate_command(candidate_path: Path, reference_path: Path, output_path: Path) -> None:
    """Evaluate candidate Parquet against silver or golden Parquet."""
    if output_path.exists():
        raise click.ClickException(f"output already exists: {output_path}")
    report = evaluate_document(
        read_document_elements(candidate_path),
        read_document_elements(reference_path),
    )
    _write_json(output_path, report.model_dump(mode="json"))
    click.echo(f"wrote evaluation report to {output_path}")


@cli.command("reference-churn")
@click.argument("previous_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("current_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--candidate-path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def reference_churn_command(
    previous_path: Path,
    current_path: Path,
    output_path: Path,
    candidate_path: Path | None,
) -> None:
    """Report reference drift and candidate identity between revisions."""
    if output_path.exists():
        raise click.ClickException(f"output already exists: {output_path}")
    report = evaluate_reference_churn(
        read_document_elements(previous_path),
        read_document_elements(current_path),
        candidate=(read_document_elements(candidate_path) if candidate_path is not None else None),
    )
    _write_json(output_path, report.model_dump(mode="json"))
    click.echo(f"wrote reference churn report to {output_path}")


@cli.command("build-adjudication")
@click.argument("candidate_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reference_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
def build_adjudication_command(
    candidate_path: Path,
    reference_path: Path,
    bronze_dir: Path,
    output_dir: Path,
) -> None:
    """Build normalized blind A/B/C packets and a separate source map."""
    if output_dir.exists():
        raise click.ClickException(f"output already exists: {output_dir}")
    page_images = {
        int(path.stem.removeprefix("page-")): path for path in (bronze_dir / "pages").glob("page-*.png")
    }
    bundle = build_blind_adjudication_bundle(
        read_document_elements(candidate_path),
        read_document_elements(reference_path),
        page_images=page_images,
    )
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "packets.json",
        [packet.model_dump(mode="json") for packet in bundle.packets],
    )
    _write_json(
        output_dir / "source-map.json",
        [key.model_dump(mode="json") for key in bundle.keys],
    )
    _write_json(output_dir / "report.json", bundle.report.model_dump(mode="json"))
    click.echo(f"wrote {len(bundle.packets)} blind packets to {output_dir}")


@cli.command("reproduce-semantic11-blind-material")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("semantic12_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def reproduce_semantic11_blind_material_command(
    source_dir: Path,
    semantic12_candidate_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    root_dir: Path,
) -> None:
    """Reproduce fixed semantic11 selections from semantic12-identical candidates."""
    reproduce_semantic11_blind_material(
        source_dir,
        semantic12_candidate_dir,
        reference_dir,
        output_dir,
        root_dir=root_dir,
    )
    click.echo(f"wrote fixed blind material to {output_dir}")


@cli.command("build-semantic12-blind-v3")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def build_semantic12_blind_v3_command(
    candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Export reviewer-only v3 files and a separately protected source map."""
    build_semantic12_blind_v3(
        candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
    )
    click.echo(f"wrote reviewer bundle to {reviewer_dir}")
    click.echo(f"wrote protected source map to {protected_dir}")


@cli.command("reproduce-semantic12-blind-v3")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("protected_source_map", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def reproduce_semantic12_blind_v3_command(
    candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    protected_source_map: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Exactly regenerate v3 using a separately supplied protected seed."""
    build_semantic12_blind_v3(
        candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
        seed=read_protected_seed(protected_source_map),
    )
    click.echo(f"reproduced reviewer bundle to {reviewer_dir}")
    click.echo(f"reproduced protected source map to {protected_dir}")


@cli.command("build-blind-v4")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def build_blind_v4_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Build and adversarially gate fresh semantic13 v4 material."""
    build_blind_v4(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
    )
    click.echo(f"wrote reviewer bundle to {reviewer_dir}")
    click.echo(f"wrote protected metadata to {protected_dir}")


@cli.command("reproduce-blind-v4")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("protected_seed", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def reproduce_blind_v4_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    protected_seed: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Exactly regenerate v4 with its separately protected seed file."""
    build_blind_v4(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
        seed=read_v4_protected_seed(protected_seed),
    )
    click.echo(f"reproduced reviewer bundle to {reviewer_dir}")
    click.echo(f"reproduced protected metadata to {protected_dir}")


@cli.command("build-blind-v5")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def build_blind_v5_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Build the gated v5 reviewer audit sample and protected full ledger."""
    build_blind_v5(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
    )
    click.echo(f"wrote reviewer audit sample to {reviewer_dir}")
    click.echo(f"wrote protected full ledger to {protected_dir}")


@cli.command("reproduce-blind-v5")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("protected_seed", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def reproduce_blind_v5_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    protected_seed: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Exactly regenerate v5 from its separately protected seed."""
    build_blind_v5(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
        seed=read_v5_protected_seed(protected_seed),
    )
    click.echo(f"reproduced reviewer audit sample to {reviewer_dir}")
    click.echo(f"reproduced protected full ledger to {protected_dir}")


@cli.command("build-blind-v6")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def build_blind_v6_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Build fresh v6 reviewer and protected artifacts after deterministic gates."""
    build_blind_v6(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
    )
    click.echo(f"wrote reviewer audit sample to {reviewer_dir}")
    click.echo(f"wrote protected full ledger to {protected_dir}")


@cli.command("reproduce-blind-v6")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("protected_seed", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def reproduce_blind_v6_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    protected_seed: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Exactly regenerate v6 from supplied bronze paths and its protected seed."""
    build_blind_v6(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
        seed=read_v6_protected_seed(protected_seed),
    )
    click.echo(f"reproduced reviewer audit sample to {reviewer_dir}")
    click.echo(f"reproduced protected full ledger to {protected_dir}")


@cli.command("build-blind-v7")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def build_blind_v7_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Build isolated standalone v7 reviewer and protected artifacts."""
    build_blind_v7(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
    )
    click.echo(f"wrote standalone reviewer audit sample to {reviewer_dir}")
    click.echo(f"wrote protected v7 closure to {protected_dir}")


@cli.command("reproduce-blind-v7")
@click.argument("candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("deterministic_candidate_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("reference_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("bronze_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("protected_seed", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reviewer_dir", type=click.Path(path_type=Path))
@click.argument("protected_dir", type=click.Path(path_type=Path))
@click.option("--root-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path("."))
def reproduce_blind_v7_command(
    candidate_dir: Path,
    deterministic_candidate_dir: Path,
    reference_dir: Path,
    bronze_dir: Path,
    protected_seed: Path,
    reviewer_dir: Path,
    protected_dir: Path,
    root_dir: Path,
) -> None:
    """Exactly regenerate v7 from supplied inputs and its protected seed."""
    build_blind_v7(
        candidate_dir,
        deterministic_candidate_dir,
        reference_dir,
        bronze_dir,
        reviewer_dir,
        protected_dir,
        root_dir=root_dir,
        seed=read_v7_protected_seed(protected_seed),
    )
    click.echo(f"reproduced standalone reviewer audit sample to {reviewer_dir}")
    click.echo(f"reproduced protected v7 closure to {protected_dir}")


@cli.command("apply-adjudication")
@click.argument("candidate_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reference_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("adjudication_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--adjudicator", required=True)
def apply_adjudication_command(
    candidate_path: Path,
    reference_path: Path,
    adjudication_dir: Path,
    output_path: Path,
    adjudicator: str,
) -> None:
    """Apply complete blind decisions to produce the next silver revision."""
    bundle = AdjudicationBundle(
        report=EvaluationReport.model_validate_json(
            (adjudication_dir / "report.json").read_text(encoding="utf-8")
        ),
        packets=TypeAdapter(list[BlindAdjudicationPacket]).validate_json(
            (adjudication_dir / "packets.json").read_text(encoding="utf-8")
        ),
        keys=TypeAdapter(list[AdjudicationKey]).validate_json(
            (adjudication_dir / "source-map.json").read_text(encoding="utf-8")
        ),
    )
    decisions = TypeAdapter(list[AdjudicationDecision]).validate_json(
        (adjudication_dir / "decisions.json").read_text(encoding="utf-8")
    )
    elements = apply_adjudication_decisions(
        read_document_elements(candidate_path),
        read_document_elements(reference_path),
        bundle,
        decisions,
        adjudicator=adjudicator,
    )
    write_document_elements(elements, output_path)
    click.echo(f"wrote silver revision with {len(elements)} elements to {output_path}")


@cli.command("extract")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("markdown_path", type=click.Path(path_type=Path))
@click.option("--parquet-path", type=click.Path(path_type=Path))
@click.option("--page", "pages", type=int, multiple=True)
def extract_command(
    pdf_path: Path,
    markdown_path: Path,
    parquet_path: Path | None,
    pages: tuple[int, ...],
) -> None:
    """Extract ordered document elements and render canonical Markdown."""
    if markdown_path.exists():
        raise click.ClickException(f"output already exists: {markdown_path}")
    extracted = extract_document_with_catalog(pdf_path, pages=sorted(pages) or None)
    elements = list(extracted.elements)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_document(elements) + "\n", encoding="utf-8")
    if parquet_path is not None:
        write_document_with_source_catalog(elements, extracted.source_catalog, parquet_path)
    click.echo(f"wrote {len(elements)} elements to {markdown_path}")


@cli.command("extract-tables")
@click.argument("pdf_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--page", "pages", type=int, multiple=True)
def extract_tables_command(pdf_path: Path, output_path: Path, pages: tuple[int, ...]) -> None:
    """Extract detected tables to one candidate Parquet file."""
    document_id = _sha256(pdf_path)
    elements = extract_document_table_elements(
        pdf_path,
        document_id,
        pages=sorted(pages) or None,
        annotator="pymupdf-1.28",
    )
    if not elements:
        raise click.ClickException("no tables were detected on the selected pages")
    write_document_elements(elements, output_path)
    click.echo(f"wrote {len(elements)} tables to {output_path}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    cli()
