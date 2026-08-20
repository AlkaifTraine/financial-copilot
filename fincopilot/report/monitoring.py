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
from dataclasses import asdict, dataclass, field

from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company

log = logging.getLogger(__name__)


@dataclass
class Catalyst:
    """A dated event that could move the stock, and the direction it points."""

    event: str
    timing: str                  # e.g. "Q4 FY2026 earnings, ~Feb 2026"
    direction: str               # Positive / Negative / Two-sided
    metric: str                  # the figure it would move

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WatchItem:
    """One line of the monitoring dashboard."""

    metric: str
    current: str                 # latest reading
    trend: str                   # direction of travel
    expected: str                # what our thesis expects it to do
    why: str                     # why it matters to the valuation

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
- Every "expected" is what OUR thesis implies should happen, so a reader can tell when reality diverges from our view.
- Specific and quantitative. Tie each item to a figure where the data allows."""

_PROMPT = """Company: {company}

Our computed valuation:
{facts}

Filings context (guidance, product roadmap, upcoming events):
{context}

Produce two things.

1. catalysts: 3-5 upcoming events that could move the stock. Each: event, timing (be concrete about when), direction (Positive / Negative / Two-sided), metric (what it moves).
2. watch_items: 4-6 metrics to monitor between catalysts, chosen because the valuation is exposed to them. Each: metric, current (latest reading), trend (direction of travel), expected (what our thesis expects), why (why it matters to fair value).

Return JSON:
{{"catalysts": [{{"event": "...", "timing": "...", "direction": "...", "metric": "..."}}],
  "watch_items": [{{"metric": "...", "current": "...", "trend": "...", "expected": "...", "why": "..."}}]}}"""


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
            current="Latest reported YoY growth",
            trend="—",
            expected=f"Our base case implies a {growth_row.base_display} CAGR",
            why=(
                f"The price already assumes a {growth_row.implied_display} CAGR; the gap to our "
                f"base case is the core of the thesis."
            ),
        ))
    return view


def generate_forward(
    company: Company,
    history: FinancialHistory,
    valuation,
    *,
    qualitative_context: str = "",
    use_model: bool = True,
) -> ForwardView:
    """Produce the catalysts and monitoring dashboard, grounded in the valuation."""
    if not use_model or valuation.dcf is None:
        return _fallback(history, valuation)

    payload = complete_json(
        _PROMPT.format(
            company=f"{company.name} ({company.ticker})",
            facts=_facts(history, valuation),
            context=(qualitative_context or "None retrieved.")[:4000],
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
            current=_str(entry, "current"),
            trend=_str(entry, "trend"),
            expected=_str(entry, "expected"),
            why=_str(entry, "why"),
        )
        for entry in (payload.get("watch_items") or [])
        if isinstance(entry, dict) and _str(entry, "metric")
    ]

    if not catalysts and not watch_items:
        return _fallback(history, valuation)

    return ForwardView(catalysts=catalysts, watch_items=watch_items, generated=True)
