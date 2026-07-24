"""
Assembling the report model.

Everything the renderers display is computed here and stored as plain fields.
The renderers do no arithmetic, no formatting decisions of consequence, and no
parsing — they place values. That is what keeps the HTML and PDF outputs
identical in substance and what makes the report reproducible.
"""

from __future__ import annotations

import logging

from ..fundamentals import FinancialHistory
from ..ingest import IngestResult
from ..resolve import Company
from ..valuation import Valuation
from . import charts as chart_builder
from . import sections as section_builder
from .models import KPI, ReportModel

log = logging.getLogger(__name__)


def _fmt_money(value: float | None, currency: str) -> str:
    """Compact money for a KPI tile."""
    if value is None:
        return "-"
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m")):
        if magnitude >= threshold:
            return f"{currency} {value / threshold:,.1f}{suffix}"
    return f"{currency} {value:,.0f}"


def _financial_table(history: FinancialHistory) -> list[dict]:
    """Reported figures, newest last, as display-ready rows."""
    growth = dict(history.growth_rates("revenue"))
    rows = []
    for year in history.years:
        rows.append(
            {
                "fiscal_year": f"FY{year.fiscal_year}",
                "period_end": year.period_end,
                "revenue": year.revenue,
                "revenue_growth": growth.get(year.fiscal_year),
                "operating_income": year.operating_income,
                "operating_margin": year.operating_margin,
                "net_income": year.net_income,
                "free_cash_flow": year.free_cash_flow,
            }
        )
    return rows


def _forecast_table(valuation: Valuation) -> list[dict]:
    if not valuation.dcf:
        return []
    return [
        {
            "year": f"FY{f.year}",
            "revenue": f.revenue,
            "revenue_growth": f.revenue_growth,
            "operating_margin": f.operating_margin,
            "free_cash_flow": f.free_cash_flow,
            "present_value": f.present_value,
        }
        for f in valuation.dcf.forecast
    ]


def _kpis(history: FinancialHistory, valuation: Valuation) -> list[KPI]:
    latest = history.latest
    currency = history.currency
    tiles: list[KPI] = []

    if latest and latest.revenue:
        growth = dict(history.growth_rates("revenue")).get(latest.fiscal_year)
        tiles.append(
            KPI(
                label="Revenue",
                value=_fmt_money(latest.revenue, currency),
                caption=(
                    f"FY{latest.fiscal_year}"
                    + (f", {growth * 100:+.0f}% YoY" if growth is not None else "")
                ),
                tone="positive" if (growth or 0) > 0 else "neutral",
            )
        )

    if latest and latest.operating_margin is not None:
        tiles.append(
            KPI(
                label="Operating margin",
                value=f"{latest.operating_margin * 100:.1f}%",
                caption=f"FY{latest.fiscal_year}",
            )
        )

    if latest and latest.free_cash_flow is not None:
        tiles.append(
            KPI(
                label="Free cash flow",
                value=_fmt_money(latest.free_cash_flow, currency),
                caption=f"FY{latest.fiscal_year}",
            )
        )

    if valuation.dcf:
        tiles.append(
            KPI(
                label="WACC",
                value=f"{valuation.dcf.wacc * 100:.1f}%",
                caption="Discount rate (CAPM)",
            )
        )

    if valuation.fair_value is not None:
        tiles.append(
            KPI(
                label="DCF fair value",
                value=f"{currency} {valuation.fair_value:,.2f}",
                caption=(
                    f"vs {valuation.share_price:,.2f} market"
                    if valuation.share_price
                    else ""
                ),
                tone="positive" if (valuation.upside or 0) > 0 else "negative",
            )
        )

    return tiles


def build_report(
    company: Company,
    history: FinancialHistory,
    valuation: Valuation,
    ingest: IngestResult,
    index,
    *,
    include_narrative: bool = True,
    progress=None,
) -> ReportModel:
    """Assemble the complete report model."""
    report = ReportModel(
        company_name=company.name,
        ticker=company.ticker,
        exchange=company.exchange,
        sector=company.sector,
        currency=history.currency,
        rating=valuation.rating,
        share_price=valuation.share_price,
        fair_value=valuation.fair_value,
        upside=valuation.upside,
        market_implied_growth=valuation.market_implied_growth,
        warnings=list(valuation.warnings),
    )

    report.kpis = _kpis(history, valuation)
    report.financial_table = _financial_table(history)
    report.forecast_table = _forecast_table(valuation)
    report.assumptions = valuation.assumptions.to_dict()

    if valuation.sensitivity:
        report.sensitivity = valuation.sensitivity.to_dict()
    if valuation.comps:
        report.comps = valuation.comps.to_dict()

    # The exact documents the report is grounded in, with their original URLs.
    report.sources = [
        {
            "label": document.label,
            "url": document.url,
            "origin": document.origin,
            "pages": document.page_count,
            "filed": document.filed_date,
        }
        for document in ingest.accepted
    ]

    if progress:
        progress("charts", "rendering charts")
    report.charts = chart_builder.build_all(valuation, history, slug=company.slug)

    if include_narrative and index is not None:
        report.sections = section_builder.build_all(
            index,
            company.name,
            latest_fiscal_year=history.latest.fiscal_year if history.latest else None,
            progress=progress,
        )

    # Provenance of the numbers is stated in the report, not assumed.
    if history.source == "sec_xbrl":
        report.warnings.append(
            "Financial statement figures are taken from SEC XBRL company facts "
            "(the company's own tagged filing data)."
        )
    else:
        report.warnings.append(
            f"{company.name} does not file with the SEC. Financial figures come "
            f"from a structured market data provider rather than audited XBRL "
            f"filings, and should be verified against the company's own reports."
        )

    return report
