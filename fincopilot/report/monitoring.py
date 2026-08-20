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

1. catalysts: 3-5 FUTURE events that could move the stock — every one must fall AFTER {as_of}. Do NOT list an earnings date or event that has already happened by then (e.g. a quarter that reported before {as_of}). Each: event, timing (a concrete future date/month, after the as-of date), direction (Positive / Negative / Two-sided), metric (what it moves).
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
        view.catalysts, _ = classify_catalysts(view.catalysts, as_of)
        return view

    # Deterministic backstop: drop any event that has already happened by the
    # as-of date, whatever the model returned. A past event can never be a catalyst.
    upcoming, passed = classify_catalysts(catalysts, as_of)
    if passed:
        log.info("dropped %d past-dated catalyst(s) before %s", len(passed), as_of)
    return ForwardView(catalysts=upcoming, watch_items=watch_items, generated=True)
