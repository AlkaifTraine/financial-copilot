"""
HTML extraction for EDGAR filings.

EDGAR serves 10-K, 10-Q and 8-K documents as HTML, which is a considerably
better starting point than PDF: a ``<table>`` states its own row and column
structure, so the income statement grid is read rather than reconstructed from
glyph coordinates.

Two EDGAR-specific quirks are handled here:

  * Filings use nested tables for page layout, not just for data. Only the
    innermost tables carry real content, so wrappers are skipped.
  * HTML has no intrinsic pagination. Filings mark page boundaries with
    explicit page-break rules, which are counted to recover page numbers that
    correspond to the printed document.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .models import BLOCK_HEADING, BLOCK_TABLE, BLOCK_TEXT, Block, ParsedDocument
from .sections import identify, looks_like_heading, match_sec_item
from .tables import rows_to_markdown

log = logging.getLogger(__name__)

_PAGE_BREAK = re.compile(r"page-break-(after|before)\s*:\s*always", re.I)

# Elements that introduce a paragraph boundary in a filing's flow.
_BLOCK_TAGS = {"p", "div", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}


def _is_layout_wrapper(table) -> bool:
    """Whether a table exists purely to position other tables."""
    return table.find("table") is not None


def _table_rows(table) -> list[list[str]]:
    rows = []
    for row in table.find_all("tr", recursive=True):
        # Skip rows belonging to a nested table; those are handled separately.
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all(["td", "th"], recursive=False)
        rows.append([cell.get_text(" ", strip=True) for cell in cells])
    return rows


def _is_bold(element) -> bool:
    if element.find(["b", "strong"]):
        return True
    style = (element.get("style") or "").lower()
    return "font-weight:bold" in style.replace(" ", "") or "font-weight:700" in style.replace(" ", "")


def parse_html(
    path: str | Path,
    *,
    doc_id: str,
    title: str,
    doc_type: str,
    source_url: str,
    origin: str,
    fiscal_year: int | None = None,
    fiscal_period: str | None = None,
    form_type: str | None = None,
) -> ParsedDocument:
    """Extract an EDGAR HTML filing into page- and section-tagged blocks."""
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    # EDGAR documents carry an XML declaration but are HTML; the warning is
    # noise here and the HTML parser is the correct choice for them.
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    path = Path(path)
    parsed = ParsedDocument(
        doc_id=doc_id,
        title=title,
        doc_type=doc_type,
        source_url=source_url,
        origin=origin,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        local_path=str(path),
    )

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    body = soup.body or soup
    page = 1
    section: str | None = None
    consumed_tables: set[int] = set()
    is_sec_form = bool(form_type)

    for element in body.find_all(True):
        # -- pagination ---------------------------------------------------
        if element.name == "hr" or _PAGE_BREAK.search(element.get("style") or ""):
            page += 1
            continue

        # -- tables -------------------------------------------------------
        if element.name == "table":
            if id(element) in consumed_tables or _is_layout_wrapper(element):
                continue
            consumed_tables.add(id(element))

            markdown = rows_to_markdown(_table_rows(element))
            if markdown:
                parsed.blocks.append(Block(BLOCK_TABLE, markdown, page, section))
            else:
                # Not a real data table — it was being used for layout. Keep
                # its text so the prose inside is not lost.
                text = element.get_text(" ", strip=True)
                if len(text) > 40:
                    parsed.blocks.append(Block(BLOCK_TEXT, text, page, section))
            continue

        if element.name not in _BLOCK_TAGS:
            continue

        # Anything inside a table has already been captured above.
        if element.find_parent("table") is not None:
            continue

        # Only take elements whose text is their own, to avoid emitting the
        # same words once per level of div nesting.
        if element.find(_BLOCK_TAGS):
            continue

        text = element.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue

        # In an SEC filing the Item structure is the authoritative taxonomy, so
        # only an Item heading may open a new section. Allowing the generic
        # narrative headings to do it as well caused a bold "Business Overview"
        # inside Item 1A to capture the rest of Risk Factors, and citations for
        # risk disclosures then read "Business Overview" — wrong, and exactly
        # the kind of detail that destroys trust in a citation.
        named = match_sec_item(text, form_type) if is_sec_form else identify(text, form_type)

        if named or (_is_bold(element) and looks_like_heading(text)):
            if named:
                section = named
            elif not is_sec_form:
                section = text
            parsed.blocks.append(Block(BLOCK_HEADING, text, page, section))
        else:
            parsed.blocks.append(Block(BLOCK_TEXT, text, page, section))

    parsed.page_count = page

    log.info(
        "parsed %s: ~%d pages, %d blocks (%d tables)",
        title,
        parsed.page_count,
        len(parsed.blocks),
        len(parsed.tables),
    )
    return parsed
