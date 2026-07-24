"""
Table normalisation.

Financial tables are the highest-value content in a filing and the easiest to
destroy. Flattening an income statement to running text produces sequences like

    Revenue 60,922 26,974 Cost of revenue 16,621 11,618

where no relationship survives between a label, a column, and a fiscal year.
A model reading that will confidently attribute the wrong number to the wrong
year — and it will do so fluently, which is worse than failing.

Rendering to markdown preserves the grid, and markdown is the tabular format
language models handle most reliably.
"""

from __future__ import annotations

import re

# A "table" with one row or one column is almost always a false positive from
# the layout detector (a boxed callout, a header rule, a page border).
MIN_ROWS = 2
MIN_COLS = 2

# Guards against a detector returning a whole page as one enormous cell grid.
MAX_COLS = 24


def _clean_cell(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("|", "/")
    return re.sub(r"\s+", " ", text).strip()


# Filings put the currency symbol and the percent sign in their own table
# cells, so a row arrives as  | Revenue | $ | 215,938 |  rather than
# | Revenue | $215,938 |. Folding those marker columns back into their value
# keeps the figure intact as a single searchable token.
_SYMBOL_PREFIX = {"$", "₹", "€", "£", "Rs", "Rs.", "INR", "USD"}
_SYMBOL_SUFFIX = {"%", "pts", "bps"}


def _merge_symbol_columns(rows: list[list[str]], width: int) -> None:
    """Fold marker cells into their adjacent value cell, in place.

    The merge moves text *within* a row and never removes a column, so every
    row keeps the same width and the grid stays aligned. Columns left entirely
    empty are dropped by the caller's existing empty-column pass.

    A column-wise rule was tried first and does not work on real filings: the
    currency column in an EDGAR statement also carries date headers and the
    occasional bare percentage, so a test of "is every value in this column a
    symbol" is never satisfied and nothing merges.
    """
    if width < 2:
        return

    for row in rows:
        for index in range(width):
            value = row[index]
            if not value:
                continue

            if value in _SYMBOL_PREFIX and index + 1 < width and row[index + 1]:
                row[index + 1] = f"{value}{row[index + 1]}"
                row[index] = ""

            elif value in _SYMBOL_SUFFIX and index > 0 and row[index - 1]:
                joiner = "" if value == "%" else " "
                row[index - 1] = f"{row[index - 1]}{joiner}{value}"
                row[index] = ""


def rows_to_markdown(rows: list[list]) -> str | None:
    """Render extracted table rows as a markdown table.

    Returns ``None`` when the input does not look like a real table, so callers
    can fall back to treating the region as ordinary prose.
    """
    if not rows:
        return None

    cleaned = [[_clean_cell(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]

    if len(cleaned) < MIN_ROWS:
        return None

    width = max(len(row) for row in cleaned)
    if width < MIN_COLS or width > MAX_COLS:
        return None

    # Pad ragged rows so the markdown grid stays rectangular.
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    _merge_symbol_columns(cleaned, width)

    # Drop columns that are empty everywhere; PDF table detection frequently
    # emits spacer columns for the gutters between figures.
    keep = [i for i in range(width) if any(row[i] for row in cleaned)]
    if len(keep) < MIN_COLS:
        return None
    cleaned = [[row[i] for i in keep] for row in cleaned]
    width = len(keep)

    # A table whose cells are nearly all empty is a layout artefact.
    filled = sum(1 for row in cleaned for cell in row if cell)
    if filled / (len(cleaned) * width) < 0.25:
        return None

    header, *body = cleaned

    # Financial statements often have a blank top-left cell ("" | 2026 | 2025).
    # Give it a name so the column is addressable in a question.
    if not header[0]:
        header = ["Line item", *header[1:]]

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)

    return "\n".join(lines)


_NUMERIC_CELL = re.compile(r"^\(?[$₹€£]?\s*-?[\d,]+(\.\d+)?\)?\s*(%|pts|bps)?$")


def numeric_cell_ratio(markdown: str) -> float:
    """Share of non-empty cells that are pure figures.

    Counting *numbers* is not enough to tell a financial table from a page of
    prose: narrative text is full of years, percentages and dollar amounts. A
    real statement is distinguished by most of its **cells** being nothing but
    a figure, which is what this measures.
    """
    cells: list[str] = []
    for line in markdown.split("\n"):
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells.extend(cell.strip() for cell in line.strip("|").split("|"))

    filled = [cell for cell in cells if cell]
    if not filled:
        return 0.0

    numeric = sum(1 for cell in filled if _NUMERIC_CELL.match(cell))
    return numeric / len(filled)


def looks_numeric(markdown: str) -> bool:
    """Whether a rendered table actually carries figures.

    Used to prioritise financial tables over layout tables (navigation grids,
    officer lists) when building the index.
    """
    numbers = re.findall(r"\(?\$?-?[\d,]+\.?\d*\)?%?", markdown)
    substantial = [n for n in numbers if len(re.sub(r"\D", "", n)) >= 2]
    return len(substantial) >= 4
