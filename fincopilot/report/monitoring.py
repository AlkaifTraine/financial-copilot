"""
Forward view: upcoming catalysts and a monitoring dashboard.

A thesis is only useful if a reader knows what to watch and when. Two exhibits
serve that: the catalysts (the dated events that could move the stock, and which
way) and the monitoring dashboard (the handful of metrics whose trend confirms or
breaks the thesis between those events). Both are the "what to do on Monday"
half of research that a static valuation leaves out.

Grounded, like the thesis and risks, in the model's own numbers: the metrics
worth watching are the ones our forecast is most exposed to, so the dashboard is
handed our assumptions and the reverse-DCF gaps and told to watch the levers that
actually move the valuation, not a generic KPI list.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company

log = logging.getLogger(__name__)

_MONTH3 = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]


def _year_month(text: str) -> tuple[int, int] | None:
    """Best-effort (year, month) parsed from a date or timing string; None if unclear."""
    if not text:
        return None
    iso = re.search(r"(20\d{2})-(\d{2})", text)
    if iso:
        return int(iso.group(1)), int(iso.group(2))
    named = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(20\d{2})",
        text, re.I,
    )
    if named:
        return int(named.group(2)), _MONTH3[named.group(1)[:3].lower()]
    return None


def classify_catalysts(catalysts: list["Catalyst"], as_of: str | None):
    """Split catalysts into (upcoming, passed) by the report's as-of month.

    A catalyst whose parseable date is strictly before the as-of month has already
    happened and must not appear in an "upcoming" section. Events whose timing
    cannot be parsed to a month are kept as upcoming (the prompt is told the date,
    so this is a backstop, not the primary guard).
    """
    reference = _year_month(as_of or "")
    if reference is None:
        return catalysts, []
    upcoming, passed = [], []
    for catalyst in catalysts:
        moment = _year_month(catalyst.timing)
        if moment is not None and moment < reference:
            catalyst.status = "passed"
            passed.append(catalyst)
        else:
            upcoming.append(catalyst)
    return upcoming, passed


def _fiscal_year_end_month(history: FinancialHistory) -> int | None:
    """The company's fiscal year-end month, from its latest reported period end."""
    for year in reversed(getattr(history, "years", []) or []):
        moment = _year_month(getattr(year, "period_end", "") or "")
        if moment is not None:
            return moment[1]
    return None


def correct_fiscal_timing(catalysts: list["Catalyst"], history: FinancialHistory) -> int:
    """Rewrite catalyst timings whose fiscal-quarter label maps to the wrong calendar.

    LLMs are unreliable at fiscal-quarter arithmetic — NVIDIA's Q1 FY2027 ends in
    ~April 2026, not 2027, because its fiscal year is offset. When a timing names a
    fiscal quarter ("Q1 FY2027"), the correct earnings window is DERIVED from the
    company's fiscal year-end and substituted if the stated calendar date is off by
    more than a quarter. Returns the number of corrections made.
    """
    fye_month = _fiscal_year_end_month(history)
    if fye_month is None:
        return 0
    corrected = 0
    for catalyst in catalysts:
        match = re.search(r"\bQ([1-4])\s*FY\s?['’]?(\d{2,4})\b", catalyst.timing, re.I)
        if not match:
            continue
        quarter = int(match.group(1))
        fiscal_year = int(match.group(2))
        if fiscal_year < 100:
            fiscal_year += 2000
        # Quarter-end: Q4 lands on the fiscal year-end month; each earlier quarter is
        # three months before, wrapping into the prior calendar year as needed.
        end_month = fye_month - 3 * (4 - quarter)
        end_year = fiscal_year
        while end_month <= 0:
            end_month += 12
            end_year -= 1
        # Results are reported ~a month after the quarter closes.
        report_month = end_month + 1
        report_year = end_year
        if report_month > 12:
            report_month -= 12
            report_year += 1
        stated = _year_month(catalyst.timing)
        computed = report_year * 12 + report_month
        if stated is None or abs((stated[0] * 12 + stated[1]) - computed) > 2:
            catalyst.timing = (
                f"Q{quarter} FY{fiscal_year} earnings, "
                f"~{_MONTH_NAMES[report_month]} {report_year}"
            )
            corrected += 1
    return corrected


@dataclass
class Catalyst:
    """A dated event that could move the stock, and the direction it points."""

    event: str
    timing: str                  # e.g. "Q4 FY2026 earnings, ~Feb 2026"
    direction: str               # Positive / Negative / Two-sided
    metric: str                  # the figure it would move
    status: str = "upcoming"     # "upcoming" or "passed" (set by classify_catalysts)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WatchItem:
    """One line of the monitoring dashboard, tied to a specific thesis assumption."""

    metric: str
    assumption: str              # the model assumption this metric tests
    current: str                 # latest reading
    trend: str                   # direction of travel
    expected: str                # what our thesis/assumption expects it to do
    bull_bear: str               # the value/trend that would make us more bullish or bearish

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForwardView:
    """Catalysts plus the monitoring dashboard."""

    catalysts: list[Catalyst] = field(default_factory=list)
    watch_items: list[WatchItem] = field(default_factory=list)
    generated: bool = False

    def to_dict(self) -> dict:
        return {
            "catalysts": [c.to_dict() for c in self.catalysts],
            "watch_items": [w.to_dict() for w in self.watch_items],
            "generated": self.generated,
        }


def _facts(history: FinancialHistory, valuation) -> str:
    cur = valuation.currency
    lines: list[str] = []
    latest = history.latest
    if latest:
        lines.append(f"Most recent fiscal year on file: FY{latest.fiscal_year}")
    if valuation.share_price:
        lines.append(f"Current price: {cur} {valuation.share_price:,.2f}")
    if valuation.fair_value is not None:
        lines.append(f"Our fair value: {cur} {valuation.fair_value:,.2f} (rating {valuation.rating})")

    pi = valuation.priced_in
    if pi and pi.rows:
        for row in pi.rows:
            if row.implied_value is not None:
                lines.append(
                    f"Price implies {row.label} of {row.implied_display} vs our {row.base_display}"
                )

    for key, label in (
        ("year_one_growth", "Our year-1 revenue growth"),
        ("terminal_margin", "Our terminal operating margin"),
    ):
        assumption = valuation.assumptions.get(key)
        if assumption:
            lines.append(f"{label}: {assumption.display}")

    growth = history.growth_rates("revenue")
    if growth:
        recent = ", ".join(f"FY{y}: {g * 100:+.0f}%" for y, g in growth[-3:])
        lines.append(f"Recent revenue growth — {recent}")
    return "\n".join(lines)


_SYSTEM = """You are an equity analyst writing the forward-looking watch list for a research report.

Rules:
- Catalysts must be DATED or timeable events (earnings, product launches, regulatory decisions, contract renewals), each with the direction it points and the single metric it would move. No vague "market conditions".
- Guidance discipline: any guidance figure must carry its period (a named quarter, a full fiscal year, a calendar year, or multi-year). Many companies (NVIDIA included) guide only ONE QUARTER ahead — never present single-quarter guidance as annual/full-year.
- The monitoring dashboard tracks the few metrics the VALUATION is most exposed to — the levers the price is leaning on — not a generic KPI dump.
- Every watch item MUST map to a specific model ASSUMPTION it tests (e.g. our year-1 growth, our terminal margin, our WACC), and state the value/trend that would make us more BULLISH or more BEARISH — so the dashboard is a falsifiability check on our own thesis, not a KPI list.
- PERIOD/COMPARISON MUST MATCH. A metric's label must state its true period and comparison and match its reading: do not label an annual, year-over-year figure "quarterly" or "sequential", and do not pair a quarterly label with a full-year number. If our assumption is an annual YoY growth rate, the metric is "revenue growth (YoY)", never "quarterly revenue growth".
- Specific and quantitative. Tie each item to a figure where the data allows."""

_PROMPT = """Company: {company}

Our computed valuation:
{facts}

Filings context (guidance, product roadmap, upcoming events):
{context}

The report's as-of date is {as_of}. Everything below is forward-looking from THAT date.

Produce two things.

1. catalysts: 3-5 FUTURE events that could move the stock — every one must fall AFTER {as_of}. Do NOT list an earnings date or event that has already happened by then (e.g. a quarter that reported before {as_of}). Each: event, timing, direction (Positive / Negative / Two-sided), metric (what it moves).
   For `timing`, give the CALENDAR month and year the event occurs (for an earnings release, the month it will be REPORTED), e.g. "February 2027". If you also cite a fiscal quarter, write it in full — "Q4 FY2027", never a run-together form like "Q42027" — and make the calendar month CONSISTENT with it. Fiscal years are often offset: a January fiscal year-end means Q1 ends ~April of the calendar year BEFORE the fiscal-year name (Q1 FY2027 ends ~April 2026, reports ~May 2026). Never let the fiscal label and the calendar month disagree.
2. watch_items: 4-6 metrics to monitor, each tied to a SPECIFIC model assumption it tests. Each: metric, assumption (the model assumption it checks — quote our number), current (latest reading), trend (direction of travel), expected (what our assumption implies it should do), bull_bear (the value/trend that would make us more bullish, and the one that would make us more bearish).

Return JSON:
{{"catalysts": [{{"event": "...", "timing": "...", "direction": "...", "metric": "..."}}],
  "watch_items": [{{"metric": "...", "assumption": "...", "current": "...", "trend": "...", "expected": "...", "bull_bear": "..."}}]}}"""


def _fallback(history: FinancialHistory, valuation) -> ForwardView:
    """A deterministic forward view from the numbers, so the report holds."""
    view = ForwardView(generated=False)
    latest = history.latest
    if latest:
        view.catalysts.append(Catalyst(
            event=f"Next quarterly results (after FY{latest.fiscal_year})",
            timing="Next earnings date",
            direction="Two-sided",
            metric="Revenue growth and operating margin versus our forecast",
        ))
    pi = valuation.priced_in
    growth_row = next((r for r in pi.rows if r.key == "revenue_cagr"), None) if pi else None
    if growth_row and growth_row.implied_value is not None:
        view.watch_items.append(WatchItem(
            metric="Revenue growth",
            assumption=f"Our base-case CAGR of {growth_row.base_display}",
            current="Latest reported YoY growth",
            trend="—",
            expected=f"Decelerating toward our {growth_row.base_display} base-case CAGR",
            bull_bear=(
                f"More bullish if growth holds above the {growth_row.implied_display} the price "
                f"implies; more bearish if it falls below our {growth_row.base_display} base case."
            ),
        ))
    return view


_EARNINGS_RE = re.compile(r"\bearnings\b|quarterly results|results release|results call|\bEPS\b", re.I)


def _as_of_date(as_of: str | None):
    from datetime import date
    try:
        return date.fromisoformat((as_of or "")[:10])
    except ValueError:
        return None


def _scheduled_earnings_dates(ticker: str, as_of: str | None) -> list:
    """Real, future scheduled earnings dates from yfinance; [] if none/unavailable.

    Earnings are the one catalyst with an authoritative published date, so we look
    it up rather than trusting the model's fiscal arithmetic. Best-effort: any
    failure (offline, missing calendar) returns [] and the model's timing stands.
    """
    from datetime import date
    reference = _as_of_date(as_of)
    if not ticker or reference is None:
        return []
    try:
        import yfinance as yf
        info = yf.Ticker(ticker)
        found: set = set()
        table = getattr(info, "earnings_dates", None)
        if table is not None:
            for stamp in list(table.index):
                moment = stamp.date() if hasattr(stamp, "date") else stamp
                if isinstance(moment, date):
                    found.add(moment)
        try:
            for moment in (info.calendar or {}).get("Earnings Date") or []:
                if isinstance(moment, date):
                    found.add(moment)
        except Exception:
            pass
        return sorted(moment for moment in found if moment > reference)
    except Exception as exc:
        log.info("earnings calendar unavailable for %s: %s", ticker, exc)
        return []


def anchor_earnings_catalysts(catalysts: list["Catalyst"], dates: list) -> int:
    """Replace the model's guessed timing on earnings catalysts with real dates.

    Earnings catalysts (in list order, which is chronological) are assigned the
    successive real scheduled dates. Non-earnings catalysts — product launches,
    regulatory rulings — have no lookup-able date and are left untouched. Returns
    the number anchored.
    """
    if not dates:
        return 0
    upcoming = iter(dates)
    anchored = 0
    for catalyst in catalysts:
        if not _EARNINGS_RE.search(f"{catalyst.event} {catalyst.timing}"):
            continue
        try:
            moment = next(upcoming)
        except StopIteration:
            break
        catalyst.timing = f"{_MONTH_NAMES[moment.month]} {moment.day}, {moment.year}"
        anchored += 1
    return anchored


def generate_forward(
    company: Company,
    history: FinancialHistory,
    valuation,
    *,
    qualitative_context: str = "",
    as_of: str | None = None,
    use_model: bool = True,
) -> ForwardView:
    """Produce the catalysts and monitoring dashboard, grounded in the valuation."""
    if not use_model or valuation.dcf is None:
        view = _fallback(history, valuation)
        anchor_earnings_catalysts(view.catalysts, _scheduled_earnings_dates(company.ticker, as_of))
        correct_fiscal_timing(view.catalysts, history)
        view.catalysts, _ = classify_catalysts(view.catalysts, as_of)
        return view

    payload = complete_json(
        _PROMPT.format(
            company=f"{company.name} ({company.ticker})",
            facts=_facts(history, valuation),
            context=(qualitative_context or "None retrieved.")[:4000],
            as_of=as_of or "the report date",
        ),
        system=_SYSTEM,
        temperature=0.2,
        max_tokens=1300,
    )

    if not isinstance(payload, dict):
        log.warning("forward view generation failed; using deterministic fallback")
        return _fallback(history, valuation)

    def _str(entry: dict, key: str) -> str:
        return str(entry.get(key) or "").strip()

    catalysts = [
        Catalyst(
            event=_str(entry, "event"),
            timing=_str(entry, "timing"),
            direction=_str(entry, "direction") or "Two-sided",
            metric=_str(entry, "metric"),
        )
        for entry in (payload.get("catalysts") or [])
        if isinstance(entry, dict) and _str(entry, "event")
    ]
    watch_items = [
        WatchItem(
            metric=_str(entry, "metric"),
            assumption=_str(entry, "assumption"),
            current=_str(entry, "current"),
            trend=_str(entry, "trend"),
            expected=_str(entry, "expected"),
            bull_bear=_str(entry, "bull_bear") or _str(entry, "why"),
        )
        for entry in (payload.get("watch_items") or [])
        if isinstance(entry, dict) and _str(entry, "metric")
    ]

    if not catalysts and not watch_items:
        view = _fallback(history, valuation)
        anchor_earnings_catalysts(view.catalysts, _scheduled_earnings_dates(company.ticker, as_of))
        correct_fiscal_timing(view.catalysts, history)
        view.catalysts, _ = classify_catalysts(view.catalysts, as_of)
        return view

    # Anchor earnings catalysts to their REAL scheduled dates (authoritative), fix
    # any remaining fiscal-quarter labels by arithmetic (fallback), THEN drop events
    # already past the as-of date — so every catalyst is filtered on its true date.
    if anchor_earnings_catalysts(catalysts, _scheduled_earnings_dates(company.ticker, as_of)):
        log.info("anchored earnings catalyst(s) to real scheduled dates")
    if correct_fiscal_timing(catalysts, history):
        log.info("corrected fiscal-quarter timing on one or more catalysts")
    upcoming, passed = classify_catalysts(catalysts, as_of)
    if passed:
        log.info("dropped %d past-dated catalyst(s) before %s", len(passed), as_of)
    return ForwardView(catalysts=upcoming, watch_items=watch_items, generated=True)
