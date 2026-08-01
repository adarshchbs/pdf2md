from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal

import pyarrow as pa

from app.pdf2md.engine import render_document_elements
from app.pdf2md.evaluation import EvaluationReport, evaluate_document
from app.pdf2md.pymupdf_runtime import open_document
from app.pdf2md.schema import DocumentElement, read_document_elements

Split = Literal["train", "validation", "holdout"]
Source = Literal["candidate", "reference"]
ArtifactKind = Literal["pdf", "elements"]
ArtifactStatus = Literal["absent", "present-invalid", "present-unvalidated", "available"]


@dataclass(frozen=True, slots=True)
class CatalogDocument:
    document_id: str
    split: Split
    local_path: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    path: Path | None
    status: ArtifactStatus
    reason: str | None
    stage: Literal["candidate", "silver", "golden"] | None = None
    root: Path | None = None
    sha256: str | None = None

    @property
    def available(self) -> bool:
        return self.status in {"available", "present-unvalidated"}


class ComparisonCatalog:
    """Root-confined local comparison catalog.

    This module is intended for a local workstation, not an internet-facing file server.
    Every artifact is resolved from configured roots and validated before it is exposed.
    """

    _MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
    _MAX_ELEMENTS = 100_000
    _CACHE_SIZE = 8

    def __init__(
        self, root: Path, *, candidate_roots: tuple[Path, ...] = (), reference_roots: tuple[Path, ...] = ()
    ):
        if root.is_symlink():
            raise FileNotFoundError(f"project root is a symlink: {root}")
        self.root = root.resolve()
        self.root_roots = (self.root,)
        self.partition_path = self.root / "data/corpus/partition-v7.json"
        self.candidate_roots = _safe_roots(candidate_roots)
        self.reference_roots = _safe_roots(reference_roots)
        self._documents = self._load_partition()
        self._element_cache: OrderedDict[
            tuple[str, Source, Path, tuple[int, int, int, int]], tuple[DocumentElement, ...]
        ] = OrderedDict()
        self._hash_cache: dict[tuple[Path, tuple[int, int, int, int]], str] = {}
        self._element_tree_index: dict[Path, tuple[dict[str, tuple[Path, ...]], bool]] = {}
        self._cache_lock = RLock()

    def _load_partition(self) -> tuple[CatalogDocument, ...]:
        if not self.partition_path.is_file() or self.partition_path.is_symlink():
            raise FileNotFoundError(f"partition-v7 is unavailable: {self.partition_path}")
        payload = json.loads(self.partition_path.read_text(encoding="utf-8"))
        raw = payload.get("documents")
        if not isinstance(raw, list):
            raise ValueError("partition-v7 documents must be an array")
        documents: list[CatalogDocument] = []
        ids: set[str] = set()
        paths: dict[str, tuple[str, Split]] = {}
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("partition-v7 contains a document without an id")
            document_id = item["id"]
            split = item.get("split")
            local_path = item.get("local_path")
            if document_id in ids:
                raise ValueError(f"partition-v7 contains duplicate document id: {document_id}")
            if split not in {"train", "validation", "holdout"} or not isinstance(local_path, str):
                raise ValueError(f"invalid partition-v7 document metadata: {document_id!r}")
            if not local_path or Path(local_path).is_absolute() or ".." in Path(local_path).parts:
                raise ValueError(f"unsafe local_path for {document_id}: {local_path!r}")
            previous = paths.get(local_path)
            if previous is not None:
                raise ValueError(
                    f"partition-v7 local_path is ambiguous: {local_path!r} maps to {previous[0]} and {document_id}"
                )
            ids.add(document_id)
            paths[local_path] = (document_id, split)
            documents.append(CatalogDocument(document_id, split, local_path, dict(item)))
        return tuple(sorted(documents, key=lambda document: document.document_id))

    def documents(self, split: Split | None = None) -> tuple[CatalogDocument, ...]:
        return tuple(document for document in self._documents if split is None or document.split == split)

    def document(self, document_id: str) -> CatalogDocument:
        for document in self._documents:
            if document.document_id == document_id:
                return document
        raise KeyError(f"document is not in partition-v7: {document_id}")

    def _indexed_element_paths(
        self,
        roots: tuple[Path, ...],
        artifact_names: set[str],
    ) -> tuple[list[Path], bool]:
        matches: list[Path] = []
        contains_symlink = False
        for root in roots:
            with self._cache_lock:
                cached = self._element_tree_index.get(root)
            if cached is None:
                by_name: dict[str, list[Path]] = {}
                root_contains_symlink = False
                for path in root.rglob("*"):
                    if path.is_symlink():
                        root_contains_symlink = True
                        continue
                    if path.is_file():
                        by_name.setdefault(path.name, []).append(path)
                cached = (
                    {name: tuple(paths) for name, paths in by_name.items()},
                    root_contains_symlink,
                )
                with self._cache_lock:
                    self._element_tree_index[root] = cached
            index, root_contains_symlink = cached
            contains_symlink = contains_symlink or root_contains_symlink
            for name in artifact_names:
                matches.extend(index.get(name, ()))
        return matches, contains_symlink

    def artifact(
        self, document_id: str, source: Source | Literal["original"], kind: ArtifactKind
    ) -> ResolvedArtifact:
        document = self.document(document_id)
        roots = (
            self.root_roots
            if source == "original"
            else (self.candidate_roots if source == "candidate" else self.reference_roots)
        )
        if not roots:
            return ResolvedArtifact(None, "absent", f"no {source} roots configured")
        relative = Path(document.local_path)
        if source == "original":
            names = (relative,)
        else:
            aliases = _artifact_aliases(document)
            suffix = ".pdf" if kind == "pdf" else ".parquet"
            names = tuple(
                dict.fromkeys((
                    relative.with_suffix(suffix),
                    *(Path(f"{alias}{suffix}") for alias in aliases),
                ))
            )
        if any(_contains_symlink(root, name) for root in roots for name in names):
            return ResolvedArtifact(None, "present-invalid", f"{source} {kind} artifact uses a symlink")
        paths = tuple(_confined(root, name) for root in roots for name in names)
        recursive_symlink = False
        if kind == "elements":
            indexed_paths, recursive_symlink = self._indexed_element_paths(
                roots,
                {name.name for name in names},
            )
            recursive_paths = [
                path.resolve()
                for root in roots
                for path in indexed_paths
                if _confined_existing(root, path) is not None
            ]
            paths += tuple(recursive_paths)
        if recursive_symlink:
            return ResolvedArtifact(None, "present-invalid", f"{source} elements tree contains symlinks")
        unique_paths = tuple(dict.fromkeys(path for path in paths if path is not None))
        files = tuple(path for path in unique_paths if path.is_file() and not path.is_symlink())
        if len(files) > 1:
            return ResolvedArtifact(
                None, "present-invalid", f"ambiguous {source} {kind} artifacts for {document_id}"
            )
        if not files:
            return ResolvedArtifact(None, "absent", f"{source} {kind} is not available for {document_id}")
        path = files[0]
        if path.stat().st_size > self._MAX_ARTIFACT_BYTES:
            return ResolvedArtifact(path, "present-invalid", f"{source} artifact exceeds size limit")
        if kind == "pdf":
            if path.suffix.casefold() != ".pdf":
                return ResolvedArtifact(path, "present-invalid", "source artifact is not a PDF")
            try:
                with open_document(path) as pdf:
                    page_count = pdf.page_count
                    if page_count < 1:
                        raise ValueError("PDF has no pages")
                actual_sha = self._sha256(path)
                if source == "original":
                    expected_sha = document.metadata.get("sha256")
                    if (
                        not isinstance(expected_sha, str)
                        or expected_sha != expected_sha.lower()
                        or len(expected_sha) != 64
                    ):
                        return ResolvedArtifact(
                            path, "present-invalid", "catalog source sha256 is unavailable or not lowercase"
                        )
                    if actual_sha != expected_sha:
                        return ResolvedArtifact(
                            path, "present-invalid", "source PDF sha256 does not match catalog"
                        )
            except (OSError, RuntimeError, ValueError) as error:
                return ResolvedArtifact(path, "present-invalid", f"invalid PDF: {error}")
            artifact_root = next(root for root in roots if root == path or root in path.parents)
            return ResolvedArtifact(path, "available", None, None, artifact_root, actual_sha)
        if kind == "elements":
            if source == "original":
                return ResolvedArtifact(
                    path, "present-invalid", "original source does not provide structured elements"
                )
            return self._validated_elements_artifact(document_id, source, path)
        artifact_root = next(root for root in roots if root == path or root in path.parents)
        return ResolvedArtifact(path, "available", None, None, artifact_root)

    def _sha256(self, path: Path) -> str:
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        key = (path.resolve(), identity)
        with self._cache_lock:
            cached = self._hash_cache.get(key)
            if cached is not None:
                return cached
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        result = digest.hexdigest()
        with self._cache_lock:
            self._hash_cache[key] = result
        return result

    def _validated_elements_artifact(self, document_id: str, source: Source, path: Path) -> ResolvedArtifact:
        try:
            elements = self._read_cached(document_id, source, path)
        except (OSError, ValueError, pa.ArrowException) as error:
            return ResolvedArtifact(path, "present-invalid", f"invalid structured artifact: {error}")
        stages = {element.annotation.stage for element in elements}
        expected = {"candidate"} if source == "candidate" else {"silver", "golden"}
        if not stages or not stages.issubset(expected) or (source == "reference" and len(stages) != 1):
            return ResolvedArtifact(path, "present-invalid", f"invalid {source} stage: {sorted(stages)}")
        stage = next(iter(stages))
        return ResolvedArtifact(path, "available", None, stage)  # type: ignore[arg-type]

    def _read_cached(self, document_id: str, source: Source, path: Path) -> tuple[DocumentElement, ...]:
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        key = (document_id, source, path.resolve(), identity)
        with self._cache_lock:
            cached = self._element_cache.get(key)
            if cached is not None:
                self._element_cache.move_to_end(key)
                return cached
        elements = tuple(read_document_elements(path))
        if len(elements) > self._MAX_ELEMENTS:
            raise ValueError("structured artifact exceeds element count limit")
        expected_document_id = self.document(document_id).metadata.get("sha256", document_id)
        if not isinstance(expected_document_id, str) or any(
            element.document_id != expected_document_id for element in elements
        ):
            raise ValueError(f"{source} elements do not belong to catalog document {document_id}")
        with self._cache_lock:
            self._element_cache[key] = elements
            self._element_cache.move_to_end(key)
            while len(self._element_cache) > self._CACHE_SIZE:
                self._element_cache.popitem(last=False)
        return elements

    def _artifact_presence(
        self,
        document: CatalogDocument,
        source: Source | Literal["original"],
    ) -> ResolvedArtifact:
        if source == "original":
            relative = Path(document.local_path)
            if _contains_symlink(self.root, relative):
                return ResolvedArtifact(None, "present-invalid", "original PDF uses a symlink")
            path = _confined(self.root, relative)
            if path is None or not path.is_file() or path.is_symlink():
                return ResolvedArtifact(None, "absent", "original PDF is unavailable")
            return ResolvedArtifact(path, "present-unvalidated", "validation deferred until selection")

        roots = self.candidate_roots if source == "candidate" else self.reference_roots
        if not roots:
            return ResolvedArtifact(None, "absent", f"no {source} roots configured")
        names = tuple(Path(f"{alias}.parquet") for alias in _artifact_aliases(document))
        if any(_contains_symlink(root, name) for root in roots for name in names):
            return ResolvedArtifact(None, "present-invalid", f"{source} elements artifact uses a symlink")
        direct = tuple(_confined(root, name) for root in roots for name in names)
        indexed, recursive_symlink = self._indexed_element_paths(
            roots,
            {name.name for name in names},
        )
        if recursive_symlink:
            return ResolvedArtifact(None, "present-invalid", f"{source} elements tree contains symlinks")
        recursive = tuple(
            path.resolve() for root in roots for path in indexed if _confined_existing(root, path) is not None
        )
        files = tuple(
            path
            for path in dict.fromkeys((*direct, *recursive))
            if path is not None and path.is_file() and not path.is_symlink()
        )
        if len(files) > 1:
            return ResolvedArtifact(None, "present-invalid", f"ambiguous {source} elements artifacts")
        if not files:
            return ResolvedArtifact(None, "absent", f"{source} elements are unavailable")
        return ResolvedArtifact(
            files[0],
            "present-unvalidated",
            "validation deferred until selection",
        )

    @staticmethod
    def _comparison_payload(
        document: CatalogDocument,
        original: ResolvedArtifact,
        candidate: ResolvedArtifact,
        reference: ResolvedArtifact,
    ) -> dict[str, object]:
        return {
            "document": {
                "id": document.document_id,
                "split": document.split,
                "local_path": document.local_path,
                "metadata": document.metadata,
            },
            "original_source": _availability(original),
            "candidate": _availability(candidate),
            "reference": _availability(reference),
            "gold": None,
        }

    def comparison_summaries(self, split: Split | None = None) -> list[dict[str, object]]:
        return [
            self._comparison_payload(
                document,
                self._artifact_presence(document, "original"),
                self._artifact_presence(document, "candidate"),
                self._artifact_presence(document, "reference"),
            )
            for document in self.documents(split)
        ]

    def comparison(self, document_id: str) -> dict[str, object]:
        document = self.document(document_id)
        candidate = self.artifact(document_id, "candidate", "elements")
        reference = self.artifact(document_id, "reference", "elements")
        original = self.artifact(document_id, "original", "pdf")
        return self._comparison_payload(document, original, candidate, reference)

    def elements(self, document_id: str, source: Source) -> dict[str, object]:
        artifact = self.artifact(document_id, source, "elements")
        if not artifact.available or artifact.path is None:
            raise FileNotFoundError(artifact.reason or "structured elements are unavailable")
        elements = self._read_cached(document_id, source, artifact.path)
        rendered = {item.element_id: item.markdown for item in render_document_elements(elements)}
        return {
            "source": source,
            "stage": artifact.stage,
            "gold": None,
            "elements": [
                {
                    "element": element.model_dump(mode="json"),
                    "canonical_rendering": rendered.get(element.element_id),
                }
                for element in elements
            ],
        }

    def evaluation(self, document_id: str) -> EvaluationReport:
        candidate = self._validated_read(document_id, "candidate")
        reference = self._validated_read(document_id, "reference")
        return evaluate_document(list(candidate), list(reference))

    def _validated_read(self, document_id: str, source: Source) -> tuple[DocumentElement, ...]:
        artifact = self.artifact(document_id, source, "elements")
        if not artifact.available or artifact.path is None:
            raise FileNotFoundError(artifact.reason or "structured elements are unavailable")
        return self._read_cached(document_id, source, artifact.path)


def _artifact_aliases(document: CatalogDocument) -> tuple[str, ...]:
    # Partition IDs, source slugs, local stems, and source hashes are all
    # established identifiers.  Never derive an alias from arbitrary metadata.
    deduplication_keys = document.metadata.get("deduplication_keys")
    deduplication_sha = deduplication_keys.get("sha256") if isinstance(deduplication_keys, dict) else None
    raw_aliases = (
        document.document_id,
        document.metadata.get("slug"),
        Path(document.local_path).stem,
        document.metadata.get("sha256"),
        deduplication_sha,
    )
    aliases: list[str] = []
    for raw in raw_aliases:
        if not isinstance(raw, str) or not raw or Path(raw).name != raw or raw in {".", ".."}:
            continue
        aliases.append(raw)
    return tuple(dict.fromkeys(aliases))


def _safe_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    result: list[Path] = []
    for original in roots:
        if original.is_symlink() or not original.is_dir():
            raise FileNotFoundError(f"artifact root does not exist or is a symlink: {original}")
        result.append(original.resolve())
    return tuple(result)


def _contains_symlink(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _confined(root: Path, relative: Path) -> Path | None:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return _confined_existing(root, root / relative)


def _confined_existing(root: Path, path: Path) -> Path | None:
    resolved = path.resolve()
    return resolved if resolved != root and root in resolved.parents else None


def _availability(artifact: ResolvedArtifact) -> dict[str, object]:
    return {
        "status": artifact.status,
        "available": artifact.available,
        "reason": artifact.reason,
        "stage": artifact.stage,
    }


_DEFAULT_CANDIDATE_RELATIVE_ROOTS = (
    Path("data/dashboard-candidates-v8-20260801T"),
    Path("data/benchmark-batch-semantic16/candidates"),
    Path("data/benchmark-batch-semantic15/candidates"),
    Path("data/candidate-latest"),
)
_DEFAULT_REFERENCE_RELATIVE_ROOTS = (Path("data/silver-cycle2"),)


def _default_roots(name: str, root: Path) -> tuple[Path, ...]:
    if name == "PDF2MD_CANDIDATE_ROOTS":
        relative_roots = _DEFAULT_CANDIDATE_RELATIVE_ROOTS
    elif name == "PDF2MD_REFERENCE_ROOTS":
        relative_roots = _DEFAULT_REFERENCE_RELATIVE_ROOTS
    else:
        return ()
    # Select one deterministic root, rather than combining similarly named
    # historical bundles and making every overlapping document ambiguous.
    if name == "PDF2MD_CANDIDATE_ROOTS":
        dynamic = sorted(root.glob("data/dashboard-candidates-v8-20260801T*Z"), reverse=True)
        for candidate in dynamic:
            if candidate.is_dir() and not candidate.is_symlink():
                return (candidate,)
    for relative in relative_roots[1:] if name == "PDF2MD_CANDIDATE_ROOTS" else relative_roots:
        candidate = root / relative
        if candidate.is_dir() and not candidate.is_symlink():
            return (candidate,)
    return ()


def roots_from_environment(name: str, *, root: Path | None = None) -> tuple[Path, ...]:
    value = os.environ.get(name)
    if value is not None:
        return tuple(Path(item) for item in value.split(os.pathsep) if item)
    if root is None:
        return ()
    return _default_roots(name, root.resolve())
