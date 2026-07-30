from __future__ import annotations

from benchmarks.canonical import CanonicalPage, validate_page_collection


class UnsupportedTableRepresentationError(ValueError):
    """Raised when RD TableBench cannot consume a canonical table representation."""


def export_rd_tablebench(pages: list[CanonicalPage]) -> list[dict[str, object]]:
    """Return exact HTML tables with stable source-derived IDs.

    Markdown-only tables are rejected rather than converted because conversion would not
    preserve the parser's exact table output.
    """
    validate_page_collection(pages, require_canonical_order=False)
    records: list[dict[str, object]] = []
    emitted: dict[tuple[str, str], str] = {}
    for page in sorted(pages, key=lambda item: (item.document_id, item.page_index)):
        for table in page.tables:
            if table.html is None:
                raise UnsupportedTableRepresentationError(
                    f"RD TableBench export unsupported for markdown-only table "
                    f"{table.id!r} on {(page.document_id, page.page_index)!r}"
                )
            key = (page.document_id, table.id)
            previous_html = emitted.get(key)
            if previous_html is not None:
                if previous_html != table.html:
                    raise ValueError(f"logical table {table.id!r} has inconsistent HTML projections")
                continue
            emitted[key] = table.html
            records.append({
                "id": _table_id(page.document_id, table.id),
                "document_id": page.document_id,
                "first_page_index": page.page_index,
                "source_table_id": table.id,
                "html": table.html,
            })
    return records


def _table_id(document_id: str, table_id: str) -> str:
    return f"{document_id}::table-{table_id}"
