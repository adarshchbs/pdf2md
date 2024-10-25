from typing import List, Tuple


class TableCell:
    def __init__(self, text: str, bbox: Tuple[int, int, int, int]):
        self.text = text
        self.bbox = bbox


class Table:
    def __init__(self, bbox: Tuple[int, int, int, int], cells: List[TableCell]):
        self.bbox = bbox
        self.cells = cells


def sort_cells(cells: List[TableCell]) -> List[TableCell]:
    # Possible cases for complex tables:
    # 1. First column has two rows merged but second column is normal.
    # 2. Last column has three rows merged but other columns are normal.
    # 3. Multiple columns have merged rows.
    # 4. Multiple rows have merged columns.
    # 5. Rows and columns have merged cells.
    sorted_cells = sorted(cells, key=lambda cell: cell.bbox[0])
    return sorted_cells


def generate_markdown_table(table: Table) -> str:
    markdown_table = ""
    sorted_cells = sort_cells(table.cells)

    # Generate table header
    markdown_table += "|"
    for cell in sorted_cells:
        markdown_table += f" {cell.text} |"
    markdown_table += "\n"

    # Generate table separator
    markdown_table += "|"
    for _ in sorted_cells:
        markdown_table += " --- |"
    markdown_table += "\n"

    # Generate table rows
    for _ in range(len(sorted_cells)):
        markdown_table += "|"
        for cell in sorted_cells:
            markdown_table += f" {cell.text} |"
        markdown_table += "\n"

    return markdown_table
