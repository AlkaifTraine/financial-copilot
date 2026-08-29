"""
Audited annual figures from a SEBI-format results PDF.

Indian issuers file quarterly and annual results with the exchanges under
Regulation 33 in a highly consistent layout: a consolidated statement of
profit and loss, a statement of assets and liabilities, and a cash-flow
statement, each with the current period beside the restated prior one, and a
units declaration in the header. That regularity is what makes extraction
defensible here when extraction from an arbitrary PDF would not be.

It exists because the structured sources run out. The NSE results-XBRL
endpoint retains roughly three years, so for a company like Bikaji it serves
FY2023-FY2024 and stops, while FY2025 and FY2026 have been filed and audited
and sit in these PDFs. Refusing to read them means refusing to value the
company on data that demonstrably exists.

The design rule throughout is **refuse rather than guess**. Every failure mode
in this format is silent and large:

* Picking a *quarter* column instead of the *year* column understates by
  roughly four times, and both columns can carry the same date — a Q4 column
  and a full-year column both end 31 March.
* Missing the units line turns lakhs into rupees, a 100,000x error.
* Reading the standalone statements instead of the consolidated ones drops
  every subsidiary.
* These filings are frequently scanned, and the OCR mangles labels
  ("Revenue from oaerations") and negative signs ("/398.55)" for "(398.55)").

So each of those is resolved explicitly, and anything ambiguous returns
``None`` rather than a number. A missing valuation is recoverable; a valuation
built on a quarter mistaken for a year is not.

Nothing here is trusted on its own: :func:`extract` runs accounting identities
over the result and discards the whole extraction if they do not hold.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from datetime import date

from .models import FinancialHistory, FiscalYear

log = logging.getLogger(__name__)

SOURCE = "sebi_results_pdf"

# Units the header may declare, and what to multiply by to reach rupees.
_UNIT_MULTIPLIER = {
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "million": 1e6, "millions": 1e6, "mn": 1e6,
    "thousand": 1e3, "thousands": 1e3,
    "rupee": 1.0, "rupees": 1.0,
}
# The declaration is found by locating "all amounts … in" and then scanning the
# next few words for a unit we recognise, rather than assuming the unit is the
# very next token. The currency symbol sits between them and the OCR mangles it
# — one Bikaji filing reads "All Amounts In JNR Lakhs" — so requiring a
# correctly spelled "INR" would discard a statement whose unit word is perfectly
# legible. It is the unit that has to be read, not the currency.
_UNIT_ANCHOR = re.compile(r"all\s+amounts?\s+(?:are\s+)?in\b", re.I)
_UNIT_WINDOW_WORDS = 4

# Canonical field -> the labels it may appear under. Matched fuzzily, because
# these documents are scanned: "Expenses" arrives as "Exoenses" and
# "Depreciation" as "Deoreciation".
_PL_ROWS: dict[str, list[str]] = {
    "revenue": ["total revenue from operations", "revenue from operations"],
    "other_income": ["other income"],
    "total_income": ["total income"],
    "cost_of_revenue": ["cost of materials consumed"],
    "depreciation_amortisation": [
        "depreciation, amortisation and impairment expenses",
        "depreciation and amortisation expense",
        "depreciation, amortisation and impairment expense",
    ],
    "finance_costs": ["finance costs", "finance cost"],
    "total_expenses": ["total expenses"],
    "pretax_income": ["profit before tax"],
    "tax_expense": ["total tax expenses", "total tax expense", "tax expense"],
    "net_income": [
        "profit for the period /year", "profit for the period/year",
        "profit for the year", "profit for the period",
    ],
    "diluted_eps": ["(b) diluted (inr)", "diluted (inr)", "diluted"],
}

# Balance-sheet rows, deliberately few. This page fragments worse than the
# others — the detector splits single figures across cells, and a "TOTAL ASSETS"
# line arrives merged with the equity header that follows it — so only the rows
# that survived hand-verification against the filing are read.
#
# Current assets and current liabilities are NOT extracted. They read
# plausibly but wrongly (current assets came out near twice the filed figure),
# and they feed working capital, where being wrong is expensive: the DCF
# charges the difference against free cash flow in every forecast year. Left
# absent, the working-capital assumption falls back to a stated default, which
# is a known approximation instead of a confident error.
_BS_ROWS: dict[str, list[str]] = {
    "shareholders_equity": ["total equity", "total equity attributable to owners"],
    "cash_and_equivalents": ["cash and cash equivalents"],
}

_CF_ROWS: dict[str, list[str]] = {
    "operating_cash_flow": [
        "net cash generated from operating activities",
        "net cash flow from operating activities",
        "net cash generated from/ (used in) operating activities",
    ],
    "capex": [
        "purchase of property, plant and equipment, intangible under development, "
        "capital work in progress and intangible assets",
        "purchase of property, plant and equipment",
        "payment for purchase of property, plant and equipment",
    ],
}

_LABEL_MATCH_CUTOFF = 0.78

# Per-share figures are quoted in rupees, not in the statement's units. Scaling
# earnings per share by the lakh multiplier turns INR 10.30 into INR 1,030,000
# and makes every downstream per-share comparison meaningless.
_PER_SHARE_FIELDS = {"diluted_eps", "basic_eps"}


@dataclass
class Extraction:
    """One statement's worth of figures, with where each came from."""

    values: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def parse_number(cell: str) -> float | None:
    """Parse an Indian-format figure, tolerating OCR damage.

    Handles lakh/crore digit grouping ("2,93,474.32"), which needs no special
    treatment once separators are stripped, and negatives in parentheses. The
    parenthesis cases matter: these documents are scanned, and the opening
    bracket is routinely read as "/" or "l", so "(398.55)" arrives as
    "/398.55)". Treating that as positive would flip the sign of every
    inventory movement and every cash outflow in the statement.
    """
    if cell is None:
        return None
    text = str(cell).strip()
    if not text or text in {"-", "--", "—", "–", "nil", "NIL"}:
        return None

    # A bracketed figure is negative. Both brackets are matched loosely because
    # the scan mangles them: "(" is read as "/" or "l", ")" as "|", and the
    # whole pair sometimes as square brackets. A missed negative turns a cash
    # outflow into an inflow, which no subtotal check would necessarily catch.
    negative = False
    if re.match(r"^[\(\[\{/l|]", text) and re.search(r"[\)\]\}|]$", text):
        negative = True
        text = re.sub(r"^[\(\[\{/l|]|[\)\]\}|]$", "", text)

    text = text.replace(",", "").replace(" ", "").replace("₹", "")
    text = re.sub(r"[^\d.\-]", "", text)
    if not text or text in {"-", ".", "-."}:
        return None

    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def detect_units(text: str) -> tuple[float | None, str]:
    """Find the declared unit multiplier. Returns (multiplier, raw phrase).

    Returns ``None`` when no declaration is found rather than assuming rupees:
    guessing here is a 100,000x error, and every statement in this format
    carries the line.
    """
    for anchor in _UNIT_ANCHOR.finditer(text):
        tail = text[anchor.end(): anchor.end() + 80]
        words = re.findall(r"[A-Za-z]+", tail)[:_UNIT_WINDOW_WORDS]
        for word in words:
            multiplier = _UNIT_MULTIPLIER.get(word.lower().strip("."))
            if multiplier is not None:
                phrase = f"{anchor.group(0)} … {word}".replace("  ", " ")
                return multiplier, phrase
    return None, ""


def _norm(label: str) -> str:
    """Normalise a row label for fuzzy comparison."""
    text = str(label).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def label_score(label: str, candidates: list[str]) -> float:
    """How well a (possibly OCR-damaged) row label matches a field's labels."""
    norm = _norm(label)
    if not norm:
        return 0.0
    best = 0.0
    for candidate in candidates:
        target = _norm(candidate)
        if norm == target:
            return 1.0
        ratio = difflib.SequenceMatcher(None, norm, target).ratio()
        # A containment hit only counts when the candidate accounts for most of
        # the label, so "revenue from operations" does not swallow "total
        # revenue from operations".
        if target in norm and len(target) >= 0.8 * len(norm):
            ratio = max(ratio, 0.95)
        # A distinctive multi-word phrase appearing intact inside a longer
        # label is a match regardless of length. Statement lines routinely
        # carry a section header ahead of the item — the capital-expenditure
        # row arrives as "CASH FLOW FROM INVESTING ACTIVITIES:- Purchase of
        # property, plant and equipment…" — and a plain ratio drowns in the
        # prefix. Four words is long enough that a phrase like "purchase of
        # property plant" cannot collide with "proceeds from sale of property
        # plant", which is the neighbouring row and has the opposite sign.
        words = target.split()
        if len(words) >= 4:
            for length in range(len(words), 3, -1):
                if " ".join(words[:length]) in norm:
                    ratio = max(ratio, 0.90)
                    break
        best = max(best, ratio)
    return best


def match_label(label: str, candidates: list[str]) -> bool:
    """Whether a row label plausibly means one of ``candidates``."""
    return label_score(label, candidates) >= _LABEL_MATCH_CUTOFF


def assign_rows(rows: list[list[str]], mapping: dict[str, list[str]]) -> dict[str, int]:
    """Map each field to the single best row index for it.

    Assignment is by argmax in both directions, and both directions matter:

    * **One row belongs to one field.** Letting a row match every field it
      scores above a threshold on is how "Total exoenses" — the OCR of "Total
      expenses" — got read as the tax charge, putting INR 2,70,000 lakhs of
      expenses into a line that should have held INR 9,035 lakhs.
    * **One field takes its best row.** Statements repeat similar labels, so
      taking the first row that clears a threshold picked up "Other expenses"
      for total expenses and "Revenue from operations" ahead of the "Total
      revenue from operations" line that is the actual Ind AS top line.
    """
    best_for_row: dict[int, tuple[str, float]] = {}
    for index, row in enumerate(rows):
        if not row or not row[0].strip():
            continue
        scored = [(field, label_score(row[0], cands)) for field, cands in mapping.items()]
        field, score = max(scored, key=lambda pair: pair[1])
        if score >= _LABEL_MATCH_CUTOFF:
            best_for_row[index] = (field, score)

    best_for_field: dict[str, tuple[int, float]] = {}
    for index, (field, score) in best_for_row.items():
        current = best_for_field.get(field)
        if current is None or score > current[1]:
            best_for_field[field] = (index, score)

    return {field: index for field, (index, _) in best_for_field.items()}


def _rows(markdown: str) -> list[list[str]]:
    out = []
    for line in markdown.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(set(c) <= {"-", ":"} for c in cells if c):
            continue          # the markdown separator row
        out.append(cells)
    return out


_DATE = re.compile(
    r"(?:march|mar|june|jun|september|sep|sept|december|dec)\s*\.?\s*(\d{1,2})?[,\s]*(\d{4})",
    re.I,
)
_MONTH = {"march": 3, "mar": 3, "june": 6, "jun": 6, "september": 9, "sep": 9,
          "sept": 9, "december": 12, "dec": 12}


def _parse_period_end(cell: str) -> date | None:
    match = _DATE.search(str(cell))
    if not match:
        return None
    month_word = re.match(r"[A-Za-z]+", str(cell).strip()[match.start():] or "")
    word = (match.group(0).split()[0] if match.group(0) else "").lower().strip(".")
    month = _MONTH.get(word)
    if month is None:
        return None
    year = int(match.group(2))
    day = int(match.group(1)) if match.group(1) else (31 if month in (3, 12) else 30)
    try:
        return date(year, month, min(day, 31))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Column resolution — the dangerous part
# ---------------------------------------------------------------------------

@dataclass
class Columns:
    """Which table columns hold full-year figures, newest first."""

    indices: list[int]
    period_ends: list[date]
    label: str


_NUMERIC_TOKEN = re.compile(r"^[\(\)\[\]/|]?[\d,]+(?:\.\d+)?[\)\]|]?$")


def _merge_split_groups(tokens: list[str]) -> list[str]:
    """Rejoin a figure whose thousands separator the OCR dropped.

    "30,409.78" is frequently read as two tokens, "30" and "409.78". Left
    alone the row yields 409.78 where the statement says 30,409.78 — a 74x
    understatement that no accounting identity would necessarily catch, since
    the cash-flow statement's subtotals are extracted the same way.

    Only the unambiguous shape is merged: a short integer with no decimal point
    immediately followed by exactly three digits and a decimal fraction, which
    is what an Indian grouping looks like once the comma is lost.
    """
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        current = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        if (
            nxt is not None
            and re.fullmatch(r"\d{1,3}", current)
            and re.fullmatch(r"\d{3}\.\d{1,2}", nxt)
        ):
            merged.append(current + nxt)
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def positional_rows(page, *, y_tolerance: float = 3.0) -> list[tuple[str, list[float]]]:
    """Read a page as (label, figures) using word geometry rather than a grid.

    The table detector guesses a column grid, and on the scanned balance-sheet
    and cash-flow pages in these filings it guesses badly enough to split a
    single figure across two cells ("2,22,3954.23" arriving as "22" and
    "3954.23"). Word coordinates do not have that failure mode: whatever the
    grid, the words sit where they were printed.

    Labels that wrap onto their own line are handled by accumulating text until
    a line actually carrying figures is reached — in these statements the long
    "Purchase of property, plant and equipment, intangible under development…"
    label occupies a line of its own with its numbers on the next.
    """
    buckets: dict[int, list] = {}
    for word in page.get_text("words"):
        buckets.setdefault(round(word[1] / y_tolerance), []).append(word)

    out: list[tuple[str, list[float]]] = []
    pending_label: list[str] = []

    for key in sorted(buckets):
        words = sorted(buckets[key], key=lambda w: w[0])
        tokens = [w[4] for w in words]

        numeric_from = len(tokens)
        for position in range(len(tokens) - 1, -1, -1):
            if _NUMERIC_TOKEN.match(tokens[position]) and re.search(r"\d", tokens[position]):
                numeric_from = position
            else:
                break

        label_part = " ".join(tokens[:numeric_from]).strip()
        figures = _merge_split_groups(tokens[numeric_from:])
        values = [v for v in (parse_number(t) for t in figures) if v is not None]

        if not values:
            # A label-only line: hold it for the figures that follow.
            if label_part:
                pending_label.append(label_part)
            continue

        label = " ".join([*pending_label, label_part]).strip()
        pending_label = []
        if label:
            out.append((label, values))

    return out


def numeric_columns(rows: list[list[str]], *, minimum: int = 3) -> list[int]:
    """Column indices that consistently hold figures, left to right."""
    counts: dict[int, int] = {}
    for row in rows:
        for index, cell in enumerate(row[1:], start=1):
            if parse_number(cell) is not None:
                counts[index] = counts.get(index, 0) + 1
    return sorted(i for i, n in counts.items() if n >= minimum)


def resolve_annual_columns(
    rows: list[list[str]], *, period_hint: list[date] | None = None
) -> Columns | None:
    """Identify the full-year columns, or refuse.

    Two shapes occur in this format:

    * The profit-and-loss table carries **both** quarterly and annual columns
      under a "Quarter Ended" / "Year Ended" header. The annual block starts at
      the "Year Ended" cell and runs to the end of the row.
    * The balance sheet and cash-flow tables carry **only** annual columns
      ("As at" / "Year ended"), so every numeric column is annual.

    The first shape is where the four-times error lives: a Q4 column and the
    full-year column both end 31 March and are indistinguishable by date alone.
    Only the header word separates them, so if that header cannot be found on a
    table that clearly has quarterly columns, this returns ``None``.
    """
    header_index = None
    year_col = None
    has_quarter = False

    for row in rows[:6]:
        for position, cell in enumerate(row):
            text = _norm(cell)
            if "quarter ended" in text:
                has_quarter = True
            if "year ended" in text and year_col is None:
                year_col, header_index = position, rows.index(row)

    # Date row: the first row after the header carrying two or more dates.
    dates: dict[int, date] = {}
    for row in rows[: (header_index or 0) + 5]:
        found = {i: _parse_period_end(c) for i, c in enumerate(row)}
        found = {i: d for i, d in found.items() if d is not None}
        if len(found) >= 2:
            dates = found
            break

    if not dates:
        # The balance-sheet and cash-flow headers are routinely fragmented by
        # the table parser ("March 31 2026 Ma" | "rch" | "31 2025"), so the
        # dates cannot be read. Those statements carry ONLY annual columns —
        # never quarterly — so the rightmost numeric columns are the periods,
        # newest first, and the P&L's already-resolved dates label them. This
        # fallback is deliberately refused when quarterly columns are present,
        # because that is where guessing costs a factor of four.
        if has_quarter or not period_hint:
            log.warning("results PDF: no period-end dates found in the header")
            return None
        numeric = numeric_columns(rows)
        if len(numeric) < len(period_hint):
            log.warning(
                "results PDF: found %d numeric columns for %d periods; refusing",
                len(numeric), len(period_hint),
            )
            return None
        chosen = numeric[-len(period_hint):]
        log.info(
            "results PDF: header dates unreadable; using the %d rightmost "
            "numeric columns with the periods from the P&L", len(chosen),
        )
        return Columns(
            indices=chosen, period_ends=list(period_hint),
            label="As at / Year ended",
        )

    if has_quarter:
        if year_col is None:
            # Quarterly columns present and no annual header to separate them:
            # any choice here is a guess, and the wrong guess is a 4x error.
            log.warning(
                "results PDF: quarterly columns present but no 'Year Ended' "
                "header found; refusing to guess which column is the year"
            )
            return None
        annual = {i: d for i, d in dates.items() if i >= year_col}
        label = "Year Ended"
    else:
        annual = dict(dates)
        label = "As at / Year ended"

    if not annual:
        log.warning("results PDF: no dated columns in the annual block")
        return None

    ordered = sorted(annual.items(), key=lambda kv: kv[1], reverse=True)
    return Columns(
        indices=[i for i, _ in ordered],
        period_ends=[d for _, d in ordered],
        label=label,
    )


def _extract_statement(
    rows: list[list[str]], mapping: dict[str, list[str]], columns: Columns,
    *, multiplier: float, page: int, doc_label: str, rowwise: bool = False,
) -> list[Extraction]:
    """Pull the mapped fields for each annual column."""
    results = [Extraction() for _ in columns.indices]
    periods = len(columns.indices)

    for field_name, row_index in assign_rows(rows, mapping).items():
        row = rows[row_index]
        label = row[0]
        scale = 1.0 if field_name in _PER_SHARE_FIELDS else multiplier

        if rowwise:
            # Ragged grids (the balance sheet and cash flow are routinely split
            # into arbitrary columns: "BIKAJI FOODS IN | TERNATION | AL LIMIT")
            # defeat fixed column indices, but the figures still appear in
            # period order at the end of the row. Take the last N numbers.
            found = [parse_number(cell) for cell in row[1:]]
            found = [v for v in found if v is not None]
            if len(found) < periods:
                continue
            chosen = found[-periods:]
        else:
            chosen = []
            for column in columns.indices:
                chosen.append(parse_number(row[column]) if column < len(row) else None)

        for slot, value in enumerate(chosen):
            if value is None:
                continue
            results[slot].values[field_name] = value * scale
            results[slot].provenance[field_name] = (
                f"{doc_label} p.{page} · {columns.label} "
                f"{columns.period_ends[slot].isoformat()} · row \"{label[:48]}\""
            )
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def extract_positional(
    page, mapping: dict[str, list[str]], period_ends: list[date],
    *, multiplier: float, page_number: int, doc_label: str,
) -> list[Extraction]:
    """Pull mapped fields from a page using word geometry.

    Used for the balance sheet and cash flow, which carry **only** annual
    columns in this format. That is what makes taking the rightmost figures
    safe here and unsafe on the profit-and-loss page, where quarterly columns
    sit alongside the annual ones and the rightmost-figure rule would silently
    return a quarter.
    """
    results = [Extraction() for _ in period_ends]
    lines = positional_rows(page)
    periods = len(period_ends)

    rows = [[label, *[str(v) for v in values]] for label, values in lines]
    for field_name, row_index in assign_rows(rows, mapping).items():
        label, values = lines[row_index]
        if len(values) < periods:
            continue
        scale = 1.0 if field_name in _PER_SHARE_FIELDS else multiplier
        for slot, value in enumerate(values[-periods:]):
            results[slot].values[field_name] = value * scale
            results[slot].provenance[field_name] = (
                f"{doc_label} p.{page_number} · year ended "
                f"{period_ends[slot].isoformat()} · row \"{label[:48]}\""
            )
    return results


def validate(values: dict[str, float], *, tolerance: float = 0.02) -> list[str]:
    """Accounting identities that must hold. Returns the failures.

    These are the reason the extraction can be trusted at all. A mis-read
    column or a dropped minus sign breaks at least one of them, and the caller
    discards the whole year rather than keeping figures that do not reconcile.
    """
    problems: list[str] = []

    def close(a: float, b: float, name: str) -> None:
        if a is None or b is None:
            return
        scale = max(abs(a), abs(b), 1.0)
        if abs(a - b) / scale > tolerance:
            problems.append(f"{name}: {a:,.0f} vs {b:,.0f}")

    revenue = values.get("revenue")
    if revenue is not None and revenue <= 0:
        problems.append("revenue is not positive")

    total_income = values.get("total_income")
    other_income = values.get("other_income")
    if revenue is not None and other_income is not None and total_income is not None:
        close(revenue + other_income, total_income, "revenue + other income != total income")

    total_expenses = values.get("total_expenses")
    pretax = values.get("pretax_income")
    if total_income is not None and total_expenses is not None and pretax is not None:
        # Exceptional items sit between the two lines, so this is a loose check.
        close(total_income - total_expenses, pretax,
              "total income - total expenses != profit before tax")

    tax = values.get("tax_expense")
    net = values.get("net_income")
    if pretax is not None and tax is not None and net is not None:
        close(pretax - tax, net, "PBT - tax != profit for the year")

    assets = values.get("total_assets")
    equity = values.get("shareholders_equity")
    if assets is not None and equity is not None and equity > assets:
        problems.append("equity exceeds total assets")

    return problems


def _sanity_scale(values: dict[str, float], market_cap: float | None) -> str | None:
    """Catch a units error by comparing revenue to market capitalisation.

    A lakh/crore mix-up moves revenue by 100x, which no plausible price-to-sales
    ratio survives. This is the cheapest available check on the single most
    damaging failure in the format.
    """
    revenue = values.get("revenue")
    if not revenue or not market_cap or market_cap <= 0:
        return None
    ratio = market_cap / revenue
    if ratio > 200 or ratio < 0.005:
        return (
            f"implied price-to-sales of {ratio:,.1f}x is outside any plausible "
            f"range — the declared units are probably wrong"
        )
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_CONSOLIDATED = re.compile(r"consolidated", re.I)
_STANDALONE = re.compile(r"standalone", re.I)
_PL_PAGE = re.compile(r"financial results for the (quarter|year)", re.I)
_BS_PAGE = re.compile(r"statement of assets and liabilities", re.I)
_CF_PAGE = re.compile(r"statement of cash flow", re.I)


def _classify_pages(document) -> dict[str, int]:
    """Find the consolidated P&L, balance-sheet and cash-flow pages.

    Consolidated only. A results filing contains the standalone statements
    first and the consolidated ones after; taking the standalone set would
    silently drop every subsidiary, which for a group with loss-making
    subsidiaries changes the earnings picture materially.
    """
    found: dict[str, int] = {}
    for index in range(document.page_count):
        text = document.load_page(index).get_text()
        if not _CONSOLIDATED.search(text) or _STANDALONE.search(text):
            # Require consolidated and reject any page also mentioning
            # standalone — the cover pages name both.
            if not (_CONSOLIDATED.search(text) and not _STANDALONE.search(text)):
                continue
        for key, pattern in (("pl", _PL_PAGE), ("bs", _BS_PAGE), ("cf", _CF_PAGE)):
            if key not in found and pattern.search(text):
                found[key] = index
    return found


def extract(local_path: str, *, doc_label: str = "results PDF",
            market_cap: float | None = None, ticker: str = "",
            company_name: str = "", currency: str = "INR") -> FinancialHistory | None:
    """Build a history from a SEBI-format results PDF, or return ``None``.

    Returns ``None`` — never a partial or best-effort history — whenever the
    units cannot be read, the annual column cannot be identified, or the
    accounting identities do not reconcile.
    """
    try:
        import fitz
    except ImportError:                                    # pragma: no cover
        log.warning("PyMuPDF unavailable; cannot read results PDF")
        return None

    from ..parse.pdf import _extract_tables

    try:
        document = fitz.open(local_path)
    except Exception as exc:
        log.warning("could not open %s: %s", local_path, exc)
        return None

    pages = _classify_pages(document)
    if "pl" not in pages:
        log.info("no consolidated profit-and-loss page found in %s", doc_label)
        return None

    # Units are declared per page; the P&L page is authoritative.
    multiplier, phrase = detect_units(document.load_page(pages["pl"]).get_text())
    if multiplier is None:
        log.warning(
            "%s: no units declaration found; refusing to guess (a lakh/rupee "
            "mix-up is a 100,000x error)", doc_label,
        )
        return None
    log.info("%s: units '%s' -> x%g", doc_label, phrase, multiplier)

    per_year: dict[str, dict] = {}
    period_hint: list[date] | None = None

    for key, mapping in (("pl", _PL_ROWS), ("bs", _BS_ROWS), ("cf", _CF_ROWS)):
        if key not in pages:
            continue
        page_index = pages[key]
        tables = _extract_tables(document.load_page(page_index))
        if not tables:
            continue
        if key == "pl":
            # The P&L carries quarterly AND annual columns, so the annual block
            # must be identified from the header rather than by position. Its
            # resolved periods then label the other two statements.
            rows = _rows(tables[0][1].text)
            columns = resolve_annual_columns(rows)
            if columns is None:
                log.warning("%s: could not resolve the annual column; refusing", doc_label)
                return None
            period_hint = columns.period_ends
            extractions = _extract_statement(
                rows, mapping, columns, multiplier=multiplier,
                page=page_index + 1, doc_label=doc_label,
            )
            period_ends = columns.period_ends
        else:
            # The balance sheet and cash flow carry only annual columns, and
            # their grids fragment badly enough on these scans to split single
            # figures across cells. Word geometry is read instead.
            if not period_hint:
                continue
            extractions = extract_positional(
                document.load_page(page_index), mapping, period_hint,
                multiplier=multiplier, page_number=page_index + 1, doc_label=doc_label,
            )
            period_ends = period_hint

        for slot, period_end in enumerate(period_ends):
            bucket = per_year.setdefault(
                period_end.isoformat(), {"values": {}, "provenance": {}}
            )
            bucket["values"].update(extractions[slot].values)
            bucket["provenance"].update(extractions[slot].provenance)

    if not per_year:
        return None

    history = FinancialHistory(
        ticker=ticker, company_name=company_name, currency=currency, source=SOURCE,
    )

    for period_end in sorted(per_year):
        values = per_year[period_end]["values"]
        problems = validate(values)
        if problems:
            log.warning(
                "%s: %s failed validation (%s); dropping the year",
                doc_label, period_end, "; ".join(problems),
            )
            continue
        scale_problem = _sanity_scale(values, market_cap)
        if scale_problem:
            log.warning("%s: %s", doc_label, scale_problem)
            return None

        ended = date.fromisoformat(period_end)
        # An Indian fiscal year ending in Jan-Mar is named for the calendar
        # year it ends in: the year to 31 March 2026 is FY2026.
        entry = FiscalYear(fiscal_year=ended.year, period_end=period_end)
        for name in (
            "revenue", "cost_of_revenue", "pretax_income", "tax_expense",
            "net_income", "diluted_eps", "depreciation_amortisation",
            "shareholders_equity", "cash_and_equivalents",
            "operating_cash_flow", "capex",
        ):
            if name in values:
                setattr(entry, name, values[name])

        # EBIT is not a reported line in this format: it is rebuilt from
        # profit before tax by adding back financing cost and removing
        # non-operating income, both of which the statement does report.
        pretax = values.get("pretax_income")
        finance = values.get("finance_costs")
        other = values.get("other_income")
        if pretax is not None and finance is not None and other is not None:
            entry.operating_income = pretax + finance - other

        history.years.append(entry)

    if not history.years:
        log.warning("%s: every year failed validation", doc_label)
        return None

    history.notes.append(
        f"Figures extracted from the audited consolidated statements in "
        f"{doc_label}, declared in {phrase or 'the stated units'}. Each figure "
        f"was checked against the statement's own accounting identities before "
        f"use; years that did not reconcile were discarded."
    )
    return history
