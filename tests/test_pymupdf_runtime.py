from __future__ import annotations

import ast
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pymupdf
import pytest

from app.pdf2md.engine import ExtractedDocument, extract_document_with_catalog, render_document
from app.pdf2md.pymupdf_geometry import unrotated_page_extent
from app.pdf2md.pymupdf_runtime import open_document, pymupdf_session

_A_BODY = "BODY_A_7f3c91e6b8424d09a1c5"
_A_TABLE = "TABLEAc8e2f4197d635ab0"
_B_BODY = "BODY_B_4a9d70c1e6385f2b"
_B_TABLE = "TABLEB91b6e3a7c4058d2f"


def _make_pdf(path: Path, body_nonce: str, table_nonce: str, *, rotation: int, crop: bool) -> None:
    document = pymupdf.open()
    page = document.new_page(width=640, height=820)
    page.insert_text((72, 82), f"Synthetic body evidence {body_nonce}", fontsize=12)
    left, top, cell_width, row_height = 72.0, 160.0, 180.0, 42.0
    rows = (("Nonce", "Value"), (table_nonce, "314159"), ("stable", "271828"))
    for column in range(3):
        x = left + column * cell_width
        page.draw_line((x, top), (x, top + len(rows) * row_height), width=1)
    for row in range(len(rows) + 1):
        y = top + row * row_height
        page.draw_line((left, y), (left + 2 * cell_width, y), width=1)
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            page.insert_text(
                (left + column_index * cell_width + 8, top + row_index * row_height + 25),
                value,
                fontsize=10,
            )
    if crop:
        page.set_cropbox(pymupdf.Rect(36, 36, 604, 784))
    page.set_rotation(rotation)
    document.save(path, garbage=4, deflate=True)
    document.close()


@pytest.fixture(scope="module")
def nonce_pdfs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, str, str]]:
    root = tmp_path_factory.mktemp("pymupdf-runtime")
    first = root / "nonce-a.pdf"
    second = root / "nonce-b.pdf"
    _make_pdf(first, _A_BODY, _A_TABLE, rotation=0, crop=False)
    _make_pdf(second, _B_BODY, _B_TABLE, rotation=90, crop=True)
    return {
        "A": (first, _A_BODY, _A_TABLE),
        "B": (second, _B_BODY, _B_TABLE),
    }


def _hashes(extracted: ExtractedDocument) -> tuple[str, str, str]:
    elements = json.dumps(
        [element.model_dump(mode="json") for element in extracted.elements],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    catalog = json.dumps(
        extracted.source_catalog.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    rendered = render_document(list(extracted.elements)).encode()
    return (
        hashlib.sha256(elements).hexdigest(),
        hashlib.sha256(catalog).hexdigest(),
        hashlib.sha256(rendered).hexdigest(),
    )


def _extract_checked(
    key: str,
    fixtures: dict[str, tuple[Path, str, str]],
) -> tuple[str, tuple[str, str, str]]:
    path, body_nonce, table_nonce = fixtures[key]
    extracted = extract_document_with_catalog(path)
    payload = json.dumps(
        {
            "elements": [element.model_dump(mode="json") for element in extracted.elements],
            "catalog": extracted.source_catalog.model_dump(mode="json"),
            "render": render_document(list(extracted.elements)),
        },
        sort_keys=True,
    )
    other_nonces = (_B_BODY, _B_TABLE) if key == "A" else (_A_BODY, _A_TABLE)
    assert body_nonce in payload
    assert all(nonce not in payload for nonce in other_nonces)

    catalog_by_id = {item.source_item_id: item for item in extracted.source_catalog.items}
    accepted_tables = [
        element
        for element in extracted.elements
        if element.element_type == "table" and element.structure.table is not None
    ]
    nonce_cells = [
        cell
        for element in accepted_tables
        for cell in element.structure.table.cells  # type: ignore[union-attr]
        if table_nonce in cell.text
    ]
    assert nonce_cells, "the table nonce must occur in accepted table cell text"
    nonempty_cells = [
        cell
        for element in accepted_tables
        for cell in element.structure.table.cells  # type: ignore[union-attr]
        if cell.text.strip()
    ]
    assert nonempty_cells
    for cell in nonempty_cells:
        assert any(fragment.source_item_ids for fragment in cell.fragments)
        for fragment in cell.fragments:
            for source_item_id in fragment.source_item_ids:
                source_item = catalog_by_id[source_item_id]
                assert source_item.kind == "word"
                assert source_item.document_id == extracted.source_catalog.document_id
                assert source_item.page_number == fragment.page_number
    return key, _hashes(extracted)


def test_sequential_same_process_first_and_warm_prefix_history_is_invariant(
    nonce_pdfs: dict[str, tuple[Path, str, str]],
) -> None:
    baseline = dict(_extract_checked(key, nonce_pdfs) for key in ("A", "B"))
    for history in (("A", "B"), ("B", "A"), ("A", "B", "A"), ("A",) * 5, ("B",) * 5):
        for key in history:
            actual_key, hashes = _extract_checked(key, nonce_pdfs)
            assert hashes == baseline[actual_key]


@pytest.mark.parametrize("worker_count", [2, 4, 8])
def test_threaded_prefix_history_is_invariant_and_has_no_nonce_leakage(
    nonce_pdfs: dict[str, tuple[Path, str, str]], worker_count: int
) -> None:
    baseline = dict(_extract_checked(key, nonce_pdfs) for key in ("A", "B"))
    history = tuple("ABBAABAB"[index % 8] for index in range(worker_count * 2))
    start = threading.Barrier(worker_count)

    def extract_after_start_barrier(key: str) -> tuple[str, tuple[str, str, str]]:
        start.wait(timeout=5)
        return _extract_checked(key, nonce_pdfs)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(extract_after_start_barrier, history))
    for key, hashes in results:
        assert hashes == baseline[key]


def test_open_document_serializes_complete_document_lifetimes(
    nonce_pdfs: dict[str, tuple[Path, str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.pdf2md.pymupdf_runtime as runtime

    real_open = pymupdf.open
    state_lock = threading.Lock()
    active = 0
    maximum = 0

    class TrackedDocument:
        def __init__(
            self,
            path: Path,
            *,
            stream: bytes | bytearray | memoryview | None = None,
            filetype: str | None = None,
        ) -> None:
            nonlocal active, maximum
            self.document = real_open(path, stream=stream, filetype=filetype)
            with state_lock:
                active += 1
                maximum = max(maximum, active)

        @property
        def page_count(self) -> int:
            return self.document.page_count

        def close(self) -> None:
            nonlocal active
            self.document.close()
            with state_lock:
                active -= 1

    monkeypatch.setattr(runtime.pymupdf, "open", TrackedDocument)
    paths = [nonce_pdfs[key][0] for key in "ABABABAB"]

    def hold_document(path: Path) -> None:
        with open_document(path) as document:
            assert document.page_count == 1
            time.sleep(0.01)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(hold_document, paths))
    assert maximum == 1
    assert active == 0


def test_unrotated_page_extent_waits_for_runtime_guard() -> None:
    entered = threading.Event()
    completed = threading.Event()

    class Page:
        @property
        def cropbox(self) -> tuple[float, float, float, float]:
            entered.set()
            return 0.0, 0.0, 20.0, 30.0

    def read_extent() -> None:
        assert unrotated_page_extent(cast(pymupdf.Page, Page())) == (20.0, 30.0)
        completed.set()

    with pymupdf_session():
        thread = threading.Thread(target=read_extent)
        thread.start()
        assert not entered.wait(timeout=0.05)
    thread.join(timeout=2)
    assert entered.is_set()
    assert completed.is_set()


def test_session_is_reentrant() -> None:
    with pymupdf_session():
        with pymupdf_session():
            with pymupdf_session():
                pass


def test_nested_reentrant_open_and_extraction_use_distinct_documents(
    nonce_pdfs: dict[str, tuple[Path, str, str]],
) -> None:
    first_path = nonce_pdfs["A"][0]
    second_path = nonce_pdfs["B"][0]
    with open_document(first_path) as first_document:
        first_text_before = str(first_document[0].get_text("text"))
        assert _A_BODY in first_text_before
        assert _B_BODY not in first_text_before

        assert _extract_checked("B", nonce_pdfs)[0] == "B"
        with open_document(second_path) as second_document:
            second_text = str(second_document[0].get_text("text"))
            assert _B_BODY in second_text
            assert _A_BODY not in second_text
            assert _extract_checked("A", nonce_pdfs)[0] == "A"

        first_text_after = str(first_document[0].get_text("text"))
        assert first_text_after == first_text_before


def test_open_and_body_exceptions_release_lock(
    nonce_pdfs: dict[str, tuple[Path, str, str]], tmp_path: Path
) -> None:
    with pytest.raises(pymupdf.FileNotFoundError):
        with open_document(tmp_path / "missing.pdf"):
            pass
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with open_document(nonce_pdfs["A"][0]):
            raise RuntimeError("synthetic failure")

    completed = threading.Event()

    def acquire_after_exceptions() -> None:
        with open_document(nonce_pdfs["B"][0]) as document:
            assert document.page_count == 1
        completed.set()

    thread = threading.Thread(target=acquire_after_exceptions)
    thread.start()
    thread.join(timeout=2)
    assert completed.is_set()


def test_rotation_and_crop_are_stable_across_interleavings(
    nonce_pdfs: dict[str, tuple[Path, str, str]],
) -> None:
    rotated_path = nonce_pdfs["B"][0]
    with open_document(rotated_path) as document:
        page = document[0]
        before = (int(page.rotation), tuple(cast(tuple[float, float, float, float], page.cropbox)))
        _extract_checked("B", nonce_pdfs)
        after = (int(page.rotation), tuple(cast(tuple[float, float, float, float], page.cropbox)))
    assert before == after
    assert _extract_checked("B", nonce_pdfs)[1] == _extract_checked("B", nonce_pdfs)[1]


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _unguarded_pymupdf_open_lines(source: str) -> list[int]:
    tree = ast.parse(source)
    module_aliases = {"pymupdf", "fitz"}
    open_aliases: set[str] = set()
    assignments: list[tuple[tuple[str, ...], ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name in {"pymupdf", "fitz"}
            )
        elif isinstance(node, ast.ImportFrom) and node.module in {"pymupdf", "fitz"}:
            open_aliases.update(
                alias.asname or alias.name for alias in node.names if alias.name in {"open", "*"}
            )
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            names = _assignment_names(node)
            if value is not None and names:
                assignments.append((names, value))

    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            is_module_alias = isinstance(value, ast.Name) and value.id in module_aliases
            is_open_alias = (
                isinstance(value, ast.Name)
                and value.id in open_aliases
                or isinstance(value, ast.Attribute)
                and value.attr == "open"
                and isinstance(value.value, ast.Name)
                and value.value.id in module_aliases
            )
            destination = module_aliases if is_module_alias else open_aliases if is_open_alias else None
            if destination is not None:
                previous_size = len(destination)
                destination.update(names)
                changed = changed or len(destination) != previous_size

    findings: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in open_aliases:
            findings.append(node.lineno)
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in module_aliases
        ):
            findings.append(node.lineno)
    return sorted(set(findings))


@pytest.mark.parametrize(
    ("source", "expected_lines"),
    [
        ("import pymupdf\npymupdf.open('a.pdf')\n", [2]),
        ("import pymupdf\npm = pymupdf\npm2 = pm\npm2.open('a.pdf')\n", [4]),
        (
            "import pymupdf as pm\nopen_pdf = pm.open\nopen_alias = open_pdf\nopen_alias('a.pdf')\n",
            [4],
        ),
        ("from fitz import open as open_pdf\nopen_pdf('a.pdf')\n", [2]),
    ],
)
def test_ast_guard_detects_direct_module_and_callable_alias_opens(
    source: str, expected_lines: list[int]
) -> None:
    assert _unguarded_pymupdf_open_lines(source) == expected_lines


@pytest.mark.parametrize(
    "source",
    [
        "from app.pdf2md.pymupdf_runtime import open_document\nopen_document('a.pdf')\n",
        "from pathlib import Path\nPath('a.pdf').open('rb')\n",
        "import unrelated as pm\npm.open('a.pdf')\n",
    ],
)
def test_ast_guard_ignores_guarded_and_unrelated_opens(source: str) -> None:
    assert _unguarded_pymupdf_open_lines(source) == []


def test_no_unguarded_modern_pymupdf_open_sites() -> None:
    root = Path(__file__).resolve().parents[1]
    audited = sorted((root / "app/pdf2md").rglob("*.py")) + sorted(
        path
        for path in (root / "benchmarks").rglob("*.py")
        if not {"datasets", "downloads", "__pycache__"}.intersection(path.parts)
    )
    runtime_path = root / "app/pdf2md/pymupdf_runtime.py"
    failures = [
        f"{path.relative_to(root)}:{line}"
        for path in audited
        if path != runtime_path
        for line in _unguarded_pymupdf_open_lines(path.read_text(encoding="utf-8"))
    ]
    assert not failures, f"unguarded modern PyMuPDF opens: {failures}"
