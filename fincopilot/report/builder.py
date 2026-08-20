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

    # Lead with OUR fair value (the intrinsic DCF); show the Wall Street
    # consensus beside it as context — "what the market believes" — not as our
    # own number.
    if valuation.fair_value is not None:
        tiles.append(
            KPI(
                label="Our fair value (DCF)",
                value=f"{currency} {valuation.fair_value:,.2f}",
                caption=(
                    f"{valuation.upside * 100:+.0f}% vs market"
                    if valuation.upside is not None
                    else ""
                ),
                tone="positive" if (valuation.upside or 0) > 0 else "negative",
            )
        )

    blended = valuation.blended
    consensus = None
    if blended:
        consensus = next(
            (e for e in blended.estimates if e.key == "analyst_consensus"), None
        )
    if consensus is not None:
        tiles.append(
            KPI(
                label="Consensus target",
                value=f"{currency} {consensus.value_per_share:,.2f}",
                caption="what the market believes",
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
        # The headline recommendation is OUR intrinsic view (DCF + scenarios).
        # The analyst consensus / blended target is shown separately as "what
        # the market believes" — the counter-view the thesis argues against —
        # not folded into our own rating.
        rating=valuation.rating,
        share_price=valuation.share_price,
        fair_value=valuation.fair_value,
        dcf_fair_value=valuation.fair_value,
        upside=valuation.upside,
        market_implied_growth=valuation.market_implied_growth,
        warnings=list(valuation.warnings),
    )

    report.kpis = _kpis(history, valuation)
    report.financial_table = _financial_table(history)
    report.forecast_table = _forecast_table(valuation)
    report.assumptions = valuation.assumptions.to_dict()

    if valuation.blended and valuation.blended.blended_value:
        report.blended = valuation.blended.to_dict()
        _consensus = next(
            (e for e in valuation.blended.estimates if e.key == "analyst_consensus"), None
        )
        if _consensus is not None:
            report.consensus_target = _consensus.value_per_share
    if valuation.priced_in and valuation.priced_in.rows:
        report.priced_in = valuation.priced_in.to_dict()
    if valuation.scenarios and valuation.scenarios.cases:
        report.scenarios = valuation.scenarios.to_dict()
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

    # Investment thesis — the analytical core, rendered first. Generated from
    # the computed valuation (scenarios, blend, reverse-DCF) plus management
    # commentary retrieved from the filings, so the argument is grounded in the
    # model's own numbers rather than hardcoded.
    if include_narrative and index is not None and valuation.dcf is not None:
        if progress:
            progress("thesis", "writing the investment thesis")
        from ..retrieve import retrieve
        from .thesis import generate_thesis

        thesis_context = retrieve(
            "competitive position moat pricing power margins outlook growth guidance risks",
            index,
            top_k=8,
        ).context_block
        report.thesis = generate_thesis(
            company, history, valuation, qualitative_context=thesis_context
        ).to_dict()

        # Quantified risk assessment — its own targeted retrieval over the risk
        # factors, then an LLM pass grounded in both those filings and the
        # model's own numbers (scenario range, reverse-DCF gaps). Replaces the
        # prose risk section.
        if progress:
            progress("risks", "quantifying the key risks")
        from .risks import generate_risks

        risk_context = retrieve(
            "risk factors customer concentration supply chain regulatory export "
            "controls litigation dependence competition risks",
            index,
            top_k=8,
        ).context_block
        report.risks = generate_risks(
            company, history, valuation, qualitative_context=risk_context
        ).to_dict()["risks"]

        # Segment forecast — a bottom-up cross-check on the top-down revenue
        # growth the DCF uses. Extracts segment revenue from the segment footnote
        # (cached), forecasts each on its own trajectory, and reconciles the sum.
        if progress:
            progress("segments", "forecasting segments bottom-up")
        from ..valuation.segments import build_segments

        segment_context = retrieve(
            "revenue by reportable operating segment full fiscal year annual "
            "segment results twelve months ended segment revenue total",
            index,
            top_k=8,
        ).context_block
        segment_analysis = build_segments(
            company, history, valuation, qualitative_context=segment_context
        )
        if segment_analysis and segment_analysis.segments:
            report.segment_forecast = segment_analysis.to_dict()

        # Competitive moat + management-vs-us (#11/#13/#14).
        if progress:
            progress("competitive", "assessing the moat")
        from .competitive import generate_competitive

        competitive_context = retrieve(
            "competitive advantage moat switching costs ecosystem CUDA differentiation "
            "management guidance outlook pricing power competition market share",
            index,
            top_k=8,
        ).context_block
        competitive = generate_competitive(
            company, history, valuation, qualitative_context=competitive_context
        )
        if competitive.moat_summary or competitive.management_vs_us:
            report.competitive = competitive.to_dict()

        # Forward view: upcoming catalysts + monitoring dashboard (#19/#20).
        if progress:
            progress("forward", "mapping catalysts and what to watch")
        from .monitoring import generate_forward

        forward_context = retrieve(
            "upcoming catalysts product launch roadmap guidance next quarter capacity "
            "regulatory decision events to watch expectations timeline",
            index,
            top_k=8,
        ).context_block
        forward = generate_forward(
            company, history, valuation, qualitative_context=forward_context
        )
        if forward.catalysts or forward.watch_items:
            report.forward = forward.to_dict()

    if include_narrative and index is not None:
        from .metrics import canonical_block

        report.sections = section_builder.build_all(
            index,
            company.name,
            latest_fiscal_year=history.latest.fiscal_year if history.latest else None,
            financial_facts=canonical_block(history, valuation),
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

    # Final QA: citation grounding, internal consistency, and a canonical-metric
    # check. Runs last, over the fully assembled report. Critical failures set the
    # publication gate (report.blocked); softer findings surface in limitations.
    from .metrics import canonical_metrics
    from .qa import run_qa

    report.canonical_metrics = [
        {"key": m.key, "label": m.label, "value": m.value,
         "unit": m.unit, "period": m.period, "definition": m.definition}
        for m in canonical_metrics(history, valuation)
    ]
    run_qa(report)

    return report
