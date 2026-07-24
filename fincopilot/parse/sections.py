"""
Section identification.

Knowing which section a passage came from is what makes targeted retrieval
possible. "What are the main risks?" should search Item 1A, not the whole
filing; the report's Financial Performance section should draw on MD&A rather
than the compensation tables.

Two mechanisms, because the corpus has two very different document shapes:

  * SEC filings have a legally mandated structure ("Item 1A. Risk Factors"),
    which is matched exactly.
  * Designed annual reports and slide decks have no such structure, so
    headings are inferred typographically from font size and weight.
"""

from __future__ import annotations

import re

# "Item 7." / "ITEM 1A -" / "Item 1A:" at the start of a line.
ITEM_PATTERN = re.compile(
    r"^\s*item\s+(\d{1,2}[A-C]?)\s*[.\-–—:)]?\s*(.{0,90})",
    re.IGNORECASE,
)

# Canonical names, so that "Item 1A", "ITEM 1A." and "Item 1A - Risk Factors"
# all collapse to one section label.
_ITEMS_10K = {
    "1": "Item 1 -Business",
    "1A": "Item 1A -Risk Factors",
    "1B": "Item 1B -Unresolved Staff Comments",
    "1C": "Item 1C -Cybersecurity",
    "2": "Item 2 -Properties",
    "3": "Item 3 -Legal Proceedings",
    "4": "Item 4 -Mine Safety Disclosures",
    "5": "Item 5 -Market for Common Equity",
    "6": "Item 6 -Selected Financial Data",
    "7": "Item 7 -Management's Discussion and Analysis",
    "7A": "Item 7A -Market Risk Disclosures",
    "8": "Item 8 -Financial Statements",
    "9": "Item 9 -Changes in and Disagreements with Accountants",
    "9A": "Item 9A -Controls and Procedures",
    "10": "Item 10 -Directors and Corporate Governance",
    "11": "Item 11 -Executive Compensation",
    "12": "Item 12 -Security Ownership",
    "13": "Item 13 -Related Party Transactions",
    "14": "Item 14 -Principal Accountant Fees",
    "15": "Item 15 -Exhibits and Schedules",
}

_ITEMS_10Q = {
    "1": "Item 1 -Financial Statements",
    "1A": "Item 1A -Risk Factors",
    "2": "Item 2 -Management's Discussion and Analysis",
    "3": "Item 3 -Market Risk Disclosures",
    "4": "Item 4 -Controls and Procedures",
    "5": "Item 5 -Other Information",
    "6": "Item 6 -Exhibits",
}

# Headings that recur in designed annual reports and Ind-AS filings, which have
# no Item numbering at all.
_NARRATIVE_HEADINGS = (
    (re.compile(r"risk\s+(factors|management)", re.I), "Risk Factors"),
    (re.compile(r"management.s\s+discussion", re.I), "Management's Discussion and Analysis"),
    # Must actually name a statement. An earlier version made every group
    # optional, so this pattern matched a bare "operations" anywhere in the
    # text and spuriously opened an "Income Statement" section in the middle
    # of Item 1A.
    (re.compile(r"(consolidated\s+)?statements?\s+of\s+(income|operations|profit)", re.I),
     "Income Statement"),
    (re.compile(r"balance\s+sheets?|statements?\s+of\s+financial\s+position", re.I),
     "Balance Sheet"),
    (re.compile(r"statements?\s+of\s+cash\s+flows?", re.I), "Cash Flow Statement"),
    (re.compile(r"notes\s+to\s+.*financial\s+statements", re.I), "Notes to Financial Statements"),
    (re.compile(r"auditor.s\s+report|independent\s+auditor", re.I), "Auditor's Report"),
    (re.compile(r"corporate\s+governance", re.I), "Corporate Governance"),
    (re.compile(r"(chairman|ceo|chief\s+executive).s?\s+(letter|message|statement)", re.I),
     "Leadership Letter"),
    (re.compile(r"business\s+(overview|responsibility)|our\s+business", re.I), "Business Overview"),
    (re.compile(r"segment\s+(information|results|reporting)", re.I), "Segment Information"),
)


def match_sec_item(line: str, form_type: str | None) -> str | None:
    """Return a canonical section name if ``line`` is an SEC Item heading."""
    match = ITEM_PATTERN.match(line.strip())
    if not match:
        return None

    number = match.group(1).upper()
    table = _ITEMS_10Q if (form_type or "").upper().startswith("10-Q") else _ITEMS_10K
    canonical = table.get(number)
    if canonical:
        return canonical

    trailing = match.group(2).strip(" .:-–—")
    return f"Item {number}" + (f" -{trailing}" if trailing else "")


def match_narrative(line: str) -> str | None:
    """Return a canonical section name for a common narrative heading."""
    stripped = line.strip()
    # Real headings are short. A long line matching "risk factors" is prose
    # that merely mentions risk factors, not a section break.
    if not (3 <= len(stripped) <= 90):
        return None
    for pattern, name in _NARRATIVE_HEADINGS:
        if pattern.search(stripped):
            return name
    return None


def identify(line: str, form_type: str | None = None) -> str | None:
    """Best-effort canonical section name for a candidate heading line."""
    return match_sec_item(line, form_type) or match_narrative(line)


def looks_like_heading(text: str) -> bool:
    """Typographic fallback for documents with no recognisable structure."""
    stripped = text.strip()
    if not (3 <= len(stripped) <= 90):
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    # Mostly digits means it is a table row or a page number, not a heading.
    letters = sum(c.isalpha() for c in stripped)
    return letters >= max(3, len(stripped) * 0.5)
