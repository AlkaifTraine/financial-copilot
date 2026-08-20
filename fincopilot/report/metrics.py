"""
Canonical financial metrics: one identity per (metric, period, definition, unit).

A research report is only trustworthy if a number means the same thing wherever
it appears. The failure this prevents is real: a headline free cash flow of
$96.7bn (operating cash flow minus capex, full year) sitting beside a narrative
paragraph that says $34.9bn — because the language model re-derived FCF from a
quarter's cash-flow statement in the retrieved text. The two are not the same
metric, but the report presented them as if they were.

The fix is to give every headline metric a single canonical identity — value,
period, definition, unit — computed once from the audited structured data, and to
hand that identity to every prose section as the authoritative figure. The
sources are then for narrative and citations, never for re-deriving a number that
already has a canonical value. The QA pass (`report/qa.py`) uses the same identities
to check that no section contradicts them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CanonicalMetric:
    """One metric with the four things that fix its identity."""

    key: str
    label: str
    value: float
    unit: str          # "currency" | "%" | "shares"
    period: str        # e.g. "FY2026", or "current" for a live market figure
    definition: str

    def display(self, currency: str) -> str:
        if self.unit == "%":
            return f"{self.value * 100:.1f}%"
        if self.unit == "shares":
            return f"{self.value / 1e9:.2f}bn shares"
        # currency
        magnitude = abs(self.value)
        for threshold, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m")):
            if magnitude >= threshold:
                return f"{currency} {self.value / threshold:,.1f}{suffix}"
        return f"{currency} {self.value:,.0f}"


def canonical_metrics(history, valuation=None) -> list[CanonicalMetric]:
    """Every headline metric, computed once from the audited structured data."""
    latest = history.latest
    if latest is None:
        return []
    fy = f"FY{latest.fiscal_year}"
    growth = dict(history.growth_rates("revenue")).get(latest.fiscal_year)

    candidates = [
        ("revenue", "Revenue", latest.revenue, "currency", fy,
         "Total net revenue per the income statement, full fiscal year"),
        ("revenue_growth", "Revenue growth", growth, "%", fy,
         "Year-over-year growth in full-year revenue"),
        ("gross_margin", "Gross margin", latest.gross_margin, "%", fy,
         "Gross profit divided by revenue"),
        ("operating_margin", "Operating margin", latest.operating_margin, "%", fy,
         "Operating income divided by revenue (GAAP)"),
        ("net_income", "Net income", latest.net_income, "currency", fy,
         "GAAP net income, full fiscal year"),
        ("free_cash_flow", "Free cash flow", latest.free_cash_flow, "currency", fy,
         "Operating cash flow minus capital expenditure, full fiscal year"),
        ("diluted_shares", "Diluted shares", latest.diluted_shares, "shares", fy,
         "Diluted weighted-average shares outstanding"),
    ]

    if valuation is not None:
        candidates.append(
            ("share_price", "Share price", valuation.share_price, "currency", "current",
             "Latest market price per share")
        )
        if valuation.fair_value:
            candidates.append(
                ("fair_value", "Our fair value", valuation.fair_value, "currency", "our DCF",
                 "Intrinsic value per share from our discounted cash flow")
            )
        if valuation.dcf:
            candidates.append(
                ("wacc", "Discount rate (WACC)", valuation.dcf.wacc, "%", "current",
                 "Weighted average cost of capital used to discount the DCF")
            )

    return [
        CanonicalMetric(key, label, value, unit, period, definition)
        for (key, label, value, unit, period, definition) in candidates
        if value is not None
    ]


def canonical_block(history, valuation=None) -> str:
    """The canonical figures formatted as an authoritative block for a prompt."""
    metrics = canonical_metrics(history, valuation)
    if not metrics:
        return ""
    currency = history.currency
    lines = [
        "CANONICAL FIGURES — authoritative. Use these EXACT values whenever you state "
        "these metrics. Do NOT recompute them from the source passages, which may contain "
        "quarterly, trailing-twelve-month, or non-GAAP variants that are a DIFFERENT metric:"
    ]
    for metric in metrics:
        lines.append(f"- {metric.label} ({metric.period}): {metric.display(currency)}  [{metric.definition}]")
    return "\n".join(lines)
