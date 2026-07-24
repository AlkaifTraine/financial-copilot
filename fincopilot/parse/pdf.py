"""
PDF extraction with table preservation.

The previous implementation was a single call per page::

    text = page.get_text()

which returns the page as a flat string. Everything that made the page
readable — the table grid, the heading hierarchy, the page boundary — is gone,
and the numbers in an income statement end up as an unlabelled run of digits.

This module instead:

  1. detects table regions and renders each as a markdown grid,
  2. extracts the remaining prose, *excluding* those regions so table contents
     are not emitted twice in mangled form,
  3. infers headings from font size relative to the document's body text,
  4. tags every block with its page number and enclosing section.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from .models import BLOCK_HEADING, BLOCK_TABLE, BLOCK_TEXT, Block, ParsedDocument
from .sections import identify, looks_like_heading
from .tables import numeric_cell_ratio, rows_to_markdown

# Minimum share of cells that must be bare figures for a borderline-detected
# region to be treated as a table rather than prose.
#
# Chosen from the measured distribution rather than by intuition. On NVIDIA's
# FY2026 earnings release (which is entirely tables) detected regions score
# 0.15-0.57; on a 215-page designed annual report, where the same detector fires
# on prose, they top out at 0.14. 0.15 separates the two cleanly.
_MIN_NUMERIC_CELL_RATIO = 0.15

log = logging.getLogger(__name__)

# A prose block is discarded when this much of its area sits inside a detected
# table, meaning the table renderer has already captured the content.
_TABLE_OVERLAP_THRESHOLD = 0.5

# A line must be this much larger than body text to count as a heading.
_HEADING_SIZE_RATIO = 1.15

# Table detection is the expensive part of parsing. Very long documents get it
# only on pages likely to contain financial tables.
_TABLE_DETECTION_PAGE_BUDGET = 400


def _body_font_size(document) -> float:
    """The document's dominant font size, used as the heading baseline.

    Absolute point sizes vary wildly between a dense 10-K and a designed annual
    report, so headings are judged relative to each document's own body text.
    """
    sizes: Counter = Counter()
    for page_number in range(min(document.page_count, 25)):
        page = document.load_page(page_number)
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        sizes[round(span.get("size", 0), 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _find_with_strategy(page, strategy: str):
    try:
        return list(getattr(page.find_tables(strategy=strategy), "tables", []))
    except Exception as exc:
        log.debug("table detection (%s) failed on page %d: %s", strategy, page.number + 1, exc)
        return []


def _extract_tables(page) -> list[tuple[float, Block, object]]:
    """Return (y_position, table Block, bounding rect) for each table found.

    PyMuPDF's default ``"lines"`` strategy locates tables by their ruled
    borders. Financial tables are overwhelmingly *borderless* — alignment is
    done with whitespace — so that strategy finds nothing in exactly the
    documents that matter most. On NVIDIA's FY2026 earnings release, which is
    almost entirely tables, ``"lines"`` detects 0 and ``"text"`` detects 13.

    ``"text"`` is far more aggressive and will occasionally read a multi-column
    prose layout as a table, so it is used only as a fallback when the precise
    strategy comes up empty. Everything it returns still has to survive the
    structural checks in :func:`rows_to_markdown`.
    """
    import fitz

    tables = _find_with_strategy(page, "lines")
    ruled = bool(tables)
    if not tables:
        tables = _find_with_strategy(page, "text")

    results = []
    for table in tables:
        try:
            markdown = rows_to_markdown(table.extract())
        except Exception:
            continue
        if not markdown:
            continue

        # The text strategy infers columns from whitespace alignment, so it
        # also "finds" tables in justified prose and multi-column layouts. On a
        # 215-page annual report it classified 387 of 392 passages as tables.
        # A table found without ruling lines therefore has to prove itself by
        # actually containing figures; anything else falls through to prose.
        if not ruled and numeric_cell_ratio(markdown) < _MIN_NUMERIC_CELL_RATIO:
            continue
        rect = fitz.Rect(table.bbox)
        results.append(
            (rect.y0, Block(BLOCK_TABLE, markdown, page.number + 1), rect)
        )
    return results


def _overlaps_table(rect, table_rects) -> bool:
    area = rect.get_area()
    if area <= 0:
        return False
    for table_rect in table_rects:
        intersection = rect & table_rect
        if not intersection.is_empty and intersection.get_area() / area > _TABLE_OVERLAP_THRESHOLD:
            return True
    return False


def parse_pdf(
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
    """Extract a PDF into page- and section-tagged blocks."""
    import fitz

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

    with fitz.open(path) as document:
        parsed.page_count = document.page_count
        body_size = _body_font_size(document)
        detect_tables = document.page_count <= _TABLE_DETECTION_PAGE_BUDGET
        current_section: str | None = None
        is_sec_form = bool(form_type)

        for page_number in range(document.page_count):
            page = document.load_page(page_number)

            tables = _extract_tables(page) if detect_tables else []
            table_rects = [rect for _y, _block, rect in tables]

            # (y position, Block) so tables and prose can be re-interleaved in
            # reading order at the end of the page.
            page_items: list[tuple[float, Block]] = [
                (y, block) for y, block, _rect in tables
            ]

            for raw_block in page.get_text("dict").get("blocks", []):
                if raw_block.get("type") != 0:  # skip images
                    continue

                rect = fitz.Rect(raw_block["bbox"])
                if _overlaps_table(rect, table_rects):
                    continue

                for line in raw_block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(span.get("text", "") for span in spans).strip()
                    if not text:
                        continue

                    max_size = max((span.get("size", 0) for span in spans), default=0)
                    is_bold = any("bold" in (s.get("font") or "").lower() for s in spans)

                    # See parse/html.py: in an SEC filing only Item headings
                    # may open a section, so a narrative heading cannot
                    # mislabel the passages that follow it.
                    named = (
                        match_sec_item(text, form_type)
                        if is_sec_form
                        else identify(text, form_type)
                    )
                    typographic = (
                        max_size >= body_size * _HEADING_SIZE_RATIO or is_bold
                    ) and looks_like_heading(text)

                    if named or typographic:
                        kind = BLOCK_HEADING
                        if named:
                            current_section = named
                        elif not is_sec_form:
                            current_section = text.strip()
                    else:
                        kind = BLOCK_TEXT

                    page_items.append(
                        (
                            fitz.Rect(line["bbox"]).y0,
                            Block(kind, text, page_number + 1),
                        )
                    )

            page_items.sort(key=lambda item: item[0])

            # Assign sections in reading order, so a heading applies to the
            # blocks that follow it on the page rather than preceding ones.
            section = current_section
            for _y, block in page_items:
                if block.kind == BLOCK_HEADING:
                    section = identify(block.text, form_type) or block.text.strip()
                block.section = section
                parsed.blocks.append(block)
            current_section = section

    log.info(
        "parsed %s: %d pages, %d blocks (%d tables)",
        title,
        parsed.page_count,
        len(parsed.blocks),
        len(parsed.tables),
    )
    return parsed
