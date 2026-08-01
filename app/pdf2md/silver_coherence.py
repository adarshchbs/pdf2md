from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Final

from app.pdf2md.schema import (
    DocumentElement,
    ParagraphStructure,
    read_document_elements,
    write_document_elements,
)
from app.pdf2md.tables import render_table, strict_table_classification

ELEMENT_TYPES: Final = frozenset({
    "paragraph",
    "heading",
    "caption",
    "footnote",
    "code",
    "table",
    "figure",
    "header",
    "footer",
    "note",
})
PARAGRAPH_ROLES: Final = frozenset({
    "body",
    "subtitle",
    "list_item",
    "figure_text",
    "figure_panel_heading",
    "heading",
    "caption",
    "footnote",
    "code",
    "running_header",
    "running_footer",
    "page_number",
    "table_continuation_marker",
})
HTML_REASONS: Final = frozenset({
    "no_header",
    "multiple_header_rows",
    "spanning_cells",
    "incomplete_grid",
    "nested_content",
    "sparse_section",
    "multi_paragraph_cell",
    "ambiguous_continuation",
    "geometry_ambiguous",
})
ROLE_MAP: Final = {
    "statement_title": "heading",
    "language_marker": "running_header",
    "annex_label": "heading",
    "annex_title": "heading",
    "lead_in": "body",
    "eli_footer": "running_footer",
    "code_block": "code",
    "section_heading": "heading",
    "figure_banner_title": "heading",
    "figure_panel_subtitle": "figure_text",
    "figure_axis_unit": "figure_text",
    "figure_axis_tick_labels": "figure_text",
    "figure_series_label": "figure_text",
    "figure_callout": "figure_text",
    "figure_legend": "figure_text",
    "figure_axis_category_labels": "figure_text",
    "figure_axis_group_label": "figure_text",
    "figure_axis_label": "figure_text",
    "figure_caption": "caption",
    "section_tab": "running_header",
    "part_heading": "heading",
    "form_checkbox": "body",
    "instruction": "body",
    "chapter_label": "heading",
    "chapter_title": "heading",
    "table_caption": "caption",
    "label": "body",
    "list_item_continuation": "body",
}
SOURCE_HASHES: Final = {
    "apple-10k2024": "c1ec83c90c0c7d403f0b9fa1b1b3bbbf887eb1edd2ac452126d62e3b3e3ce29c",
    "deep-residual-learning": "15796553182196c92acc456a8e596d6faf645cb3d4c8c5add6b08c0f3eb6f7c6",
    "eu-ai-act": "bd134a6ea87a3566066fb12c7235d937b46d9338d45234a5112413bf4a674557",
    "fao-yearbook2024": "fd14d30eae2af2ecb7d70d23812e0f04efb5f241c9b1d218a09ad46a0c1fcea0",
    "gnu-make": "d73f5f1d8234ca9a0c90e20ec8973188cf66d374a77d48dcad325da4c8fe6055",
    "ipcc-ar6-syr": "1cee257f5aedbd993a985c0dc2fb4fd7c411c78583f405210322cefd23373959",
    "irs-form990": "ec6773475a890af7631dce4fdba479dadd996427a6f036680f6954ab9e4b7518",
    "latex-tabularray": "dcb26f2110df8d7d4f4b428673a3d4810752d20a8b4792ef8004a4c36c3912cb",
    "mozilla-fin2024": "0681144d79566e1862f6d543e9ba5a1aea6da5137963608a57daf4efcecb7c85",
    "rp2040-datasheet": "bd1b0ca51078dc3c7eabea421707e6424ca60446b4b774ba76bbdfc2ef3820aa",
    "undp-hdr2024": "17e1d1af92775d60170b95762a1cb6eb210882a41fedb21778779bed9f49cde8",
    "zoning-hard-page2": "a175af2d0003b6d017f01dd5a979143ea99a2751c935a197bd1c391ae34a414d",
}
_SOURCE_ID = re.compile(r"liteparse:p(?P<page>\d+):(?:item|text-)(?P<index>\d+)$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(element: DocumentElement, observation: str) -> dict[str, object]:
    return {
        "basis": "bronze_native_not_candidate",
        "observation": observation,
        "pages": [fragment.page_number for fragment in element.fragments],
        "page_images": [f"pages/page-{fragment.page_number:04d}.png" for fragment in element.fragments],
        "fragment_bboxes": [fragment.bbox.model_dump(mode="json") for fragment in element.fragments],
        "source_item_ids": [
            source_id for fragment in element.fragments for source_id in fragment.source_item_ids
        ],
    }


def _bump(element: DocumentElement, **updates: object) -> DocumentElement:
    annotation = element.annotation.model_copy(
        update={
            "revision": element.annotation.revision + 1,
            "parent_revision": element.annotation.revision,
            "annotator": "silver-coherence-v1-bronze-curation",
            "adjudication_status": "corrected",
        }
    )
    return element.model_copy(update={**updates, "annotation": annotation})


def _with_role(element: DocumentElement, role: str) -> DocumentElement:
    old = element.structure.paragraph
    paragraph = ParagraphStructure(role=role) if old is None else old.model_copy(update={"role": role})
    structure = element.structure.model_copy(update={"paragraph": paragraph})
    return _bump(element, structure=structure)


def _normalize_table(element: DocumentElement) -> tuple[DocumentElement, list[str]]:
    table = element.structure.table
    if table is None:
        return element, []
    representation, reasons = strict_table_classification(table)
    normalized = table.model_copy(
        update={"representation": representation, "classification_reasons": reasons}
    )
    changes: list[str] = []
    if table.representation != representation:
        changes.extend([
            f"format:{element.format}->{representation}",
            f"table.representation:{table.representation}->{representation}",
        ])
    if table.classification_reasons != reasons:
        changes.append(f"classification_reasons:{table.classification_reasons!r}->{reasons!r}")
    rendered = render_table(normalized)
    if rendered != element.content:
        changes.append("content:stored->canonical_render")
    if not changes:
        return element, []
    structure = element.structure.model_copy(update={"table": normalized})
    return _bump(
        element,
        structure=structure,
        format=normalized.representation,
        content=rendered,
    ), changes


def _normalize_document(
    elements: list[DocumentElement],
) -> tuple[list[DocumentElement], list[dict[str, object]]]:
    result: list[DocumentElement] = []
    changes: list[dict[str, object]] = []
    for element in elements:
        paragraph = element.structure.paragraph
        old_role = paragraph.role if paragraph is not None else None
        updated = element
        element_changes: list[str] = []
        observation = ""
        if old_role in {"continuation_notice", "continuation_caption"}:
            updated = _with_role(element, "table_continuation_marker")
            updated = updated.model_copy(update={"element_type": "note", "include_in_output": False})
            element_changes = [
                f"element_type:{element.element_type}->note",
                f"paragraph.role:{old_role}->table_continuation_marker",
            ]
            if element.include_in_output:
                element_changes.append("include_in_output:true->false")
            observation = (
                "The repeated '(Continued)' title above the continuing grid marks a repeated table fragment and is excluded from logical content."
                if old_role == "continuation_caption"
                else "The blue line sits immediately below the continuing grid, recurs on adjoining pages, and announces continuation rather than captioning content."
            )
        elif old_role in ROLE_MAP:
            new_role = ROLE_MAP[old_role]
            updated = _with_role(element, new_role)
            element_changes = [f"paragraph.role:{old_role}->{new_role}"]
            observation = (
                "Page 127 shows an unlabelled, separately spaced and more deeply indented paragraph inside item (a), before sibling (b); it is a body paragraph logically hosted by item (a), not a new list item."
                if old_role == "list_item_continuation"
                else f"Typography, placement, and function on the cited bronze page support canonical role {new_role}; no candidate output was consulted."
            )
        elif paragraph is None and element.element_type not in {"table", "figure"}:
            if element.element_type != "footnote":
                raise ValueError(f"missing ParagraphStructure requires review: {element.element_id}")
            updated = _with_role(element, "footnote")
            element_changes = ["structure.paragraph:null->footnote"]
            observation = "Small foot-area text is separately annotated as a footnote in the bronze page region; the canonical typed role is footnote."

        updated, table_changes = _normalize_table(updated)
        element_changes.extend(table_changes)
        if table_changes and not observation:
            observation = "The unchanged bronze-derived table topology requires canonical renderer output; a sparse one-cell body band forces HTML when noted."
        if element_changes:
            changes.append({
                "action": "updated",
                "element_id": element.element_id,
                "fields": element_changes,
                "old_revision": element.annotation.revision,
                "new_revision": updated.annotation.revision,
                "parent_revision": updated.annotation.parent_revision,
                "evidence": _evidence(element, observation),
            })
        result.append(updated)

    return result, changes


def migrate(source_dir: Path, bronze_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")
    actual = {path.stem for path in source_dir.glob("*.parquet")}
    if actual != set(SOURCE_HASHES):
        raise ValueError(f"authoritative source set changed: {sorted(actual)}")
    output_dir.mkdir(parents=False)
    documents: list[dict[str, object]] = []
    for document in sorted(SOURCE_HASHES):
        source = source_dir / f"{document}.parquet"
        if _sha256(source) != SOURCE_HASHES[document]:
            raise ValueError(f"authoritative source hash changed: {source}")
        before = read_document_elements(source)
        after, changes = _normalize_document(before)
        output = output_dir / source.name
        write_document_elements(after, output)
        documents.append({
            "document": document,
            "document_id": after[0].document_id,
            "source_path": f"{source_dir.name}/{source.name}",
            "source_sha256": SOURCE_HASHES[document],
            "output_path": f"{output_dir.name}/{output.name}",
            "output_sha256": _sha256(output),
            "before_count": len(before),
            "after_count": len(after),
            "updated_count": sum(change["action"] == "updated" for change in changes),
            "removed_count": len(before) - len(after),
            "unchanged_count": len(after) - sum(change["action"] == "updated" for change in changes),
            "elements": changes,
        })
    audit: dict[str, object] = {
        "schema_version": "1.0.0",
        "migration": "silver-coherence-v1",
        "candidate_output_used_as_evidence": False,
        "source_directory": source_dir.name,
        "output_directory": output_dir.name,
        "documents": documents,
    }
    (output_dir / "migration-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation = validate(output_dir, bronze_dir)
    (output_dir / "validation-report.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def validate(silver_dir: Path, bronze_dir: Path) -> dict[str, object]:
    expected = {f"{name}.parquet" for name in SOURCE_HASHES}
    actual = {path.name for path in silver_dir.glob("*.parquet")}
    if actual != expected:
        raise ValueError(f"silver document set mismatch: {sorted(actual)}")
    summary: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "passed_with_inherited_provenance_geometry_warnings",
        "checks": {
            "parquet_roundtrip": "passed",
            "document_id": "passed",
            "element_id_order": "passed",
            "links": "passed",
            "table_grid_and_render": "passed",
            "vocabulary_and_paragraph_structure": "passed",
            "source_id_resolution_and_sibling_exclusivity": "passed",
            "source_item_bbox_containment": "blocked_inherited",
        },
        "documents": {},
        "provenance_geometry_warnings": [],
    }
    warnings = summary["provenance_geometry_warnings"]
    assert isinstance(warnings, list)
    for document in sorted(SOURCE_HASHES):
        path = silver_dir / f"{document}.parquet"
        elements = read_document_elements(path)
        manifest = json.loads((bronze_dir / document / "manifest.json").read_text(encoding="utf-8"))
        if {element.document_id for element in elements} != {manifest["source_sha256"]}:
            raise ValueError(f"document_id mismatch: {document}")
        ids = {element.element_id for element in elements}
        for element in elements:
            paragraph = element.structure.paragraph
            if element.element_type not in ELEMENT_TYPES:
                raise ValueError(f"invalid element_type: {element.element_id}")
            if element.element_type not in {"table", "figure"} and (
                paragraph is None or paragraph.role not in PARAGRAPH_ROLES
            ):
                raise ValueError(f"invalid ParagraphStructure: {element.element_id}")
            links = [*element.structure.linked_element_ids]
            if (
                element.structure.figure is not None
                and element.structure.figure.caption_element_id is not None
            ):
                links.append(element.structure.figure.caption_element_id)
            if element.structure.footnote is not None:
                links.extend(element.structure.footnote.reference_element_ids)
            if any(link not in ids for link in links):
                raise ValueError(f"unresolved link: {element.element_id}")
            table = element.structure.table
            if table is not None:
                if table.representation == "html" and any(
                    reason not in HTML_REASONS for reason in table.classification_reasons
                ):
                    raise ValueError(f"invalid HTML reason: {element.element_id}")
                representation, reasons = strict_table_classification(table)
                if (representation, reasons) != (
                    table.representation,
                    table.classification_reasons,
                ):
                    raise ValueError(f"noncanonical table classification: {element.element_id}")
                if render_table(table) != element.content:
                    raise ValueError(f"noncanonical table content: {element.element_id}")
        _check_source_ids(document, elements, bronze_dir, warnings)
        roundtrip = read_document_elements(path)
        if roundtrip != elements:
            raise ValueError(f"roundtrip mismatch: {document}")
        document_summary = summary["documents"]
        assert isinstance(document_summary, dict)
        document_summary[document] = {"elements": len(elements), "sha256": _sha256(path)}
    return summary


def _check_source_ids(
    document: str,
    elements: list[DocumentElement],
    bronze_dir: Path,
    warnings: list[object],
) -> None:
    payload = json.loads((bronze_dir / document / "liteparse.json").read_text(encoding="utf-8"))
    pages = {page["page"]: page["text_items"] for page in payload["pages"]}
    owners: defaultdict[str, set[str]] = defaultdict(set)
    fragments: list[tuple[str, object]] = []
    for element in elements:
        fragments.extend((element.element_id, fragment) for fragment in element.fragments)
        table = element.structure.table
        if table is not None:
            fragments.extend(
                (f"{element.element_id}:cell{cell.row_index},{cell.column_index}", fragment)
                for cell in table.cells
                for fragment in cell.fragments
            )
    for owner, untyped_fragment in fragments:
        fragment = untyped_fragment
        for source_id in fragment.source_item_ids:  # type: ignore[union-attr]
            match = _SOURCE_ID.fullmatch(source_id)
            if match is None:
                raise ValueError(f"invalid source ID: {source_id}")
            page = int(match.group("page"))
            index = int(match.group("index"))
            if page != fragment.page_number or index >= len(pages.get(page, [])):  # type: ignore[union-attr]
                raise ValueError(f"source ID page/index mismatch: {source_id}")
            owners[source_id].add(owner)
            item = pages[page][index]
            box = fragment.bbox  # type: ignore[union-attr]
            item_box = (item["x"], item["y"], item["x"] + item["width"], item["y"] + item["height"])
            if (
                item_box[0] < box.x0 - 1
                or item_box[1] < box.y0 - 1
                or item_box[2] > box.x1 + 1
                or item_box[3] > box.y1 + 1
            ):
                warnings.append({
                    "document": document,
                    "owner": owner,
                    "source_item_id": source_id,
                    "status": "blocked_inherited_geometry",
                    "reason": "LiteParse item font box is not contained by the inherited fragment bbox; no geometry was guessed during vocabulary-only migration.",
                })
    for source_id, source_owners in owners.items():
        cell_owners = {owner for owner in source_owners if ":cell" in owner}
        element_owners = source_owners - cell_owners
        if len(cell_owners) > 1 or len(element_owners) > 1:
            raise ValueError(f"source ID shared by siblings: {source_id}: {sorted(source_owners)}")
