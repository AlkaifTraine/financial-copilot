"""Valuation: deterministic DCF, WACC, comparables, and sensitivity."""

from __future__ import annotations

import logging

from .. import config
from ..fundamentals import FinancialHistory
from ..resolve import Company
from .assumptions import derive_inputs
from .blend import build_blend
from .comps import build_comps
from .dcf import run_dcf
from .reverse import implied_growth
from .models import (
    Assumption,
    AssumptionLedger,
    BlendedValuation,
    CompsResult,
    DCFResult,
    ScenarioAnalysis,
    ScenarioCase,
    SensitivityGrid,
    Valuation,
    ValuationEstimate,
)
from .scenarios import build_scenarios
from .sensitivity import build_grid
from .wacc import compute_wacc

log = logging.getLogger(__name__)

__all__ = [
    "value_company",
    "Valuation",
    "Assumption",
    "AssumptionLedger",
    "DCFResult",
    "SensitivityGrid",
    "ScenarioAnalysis",
    "ScenarioCase",
    "BlendedValuation",
    "ValuationEstimate",
    "CompsResult",
    "run_dcf",
]


def value_company(
    company: Company,
    history: FinancialHistory,
    *,
    qualitative_context: str = "",
    use_model: bool = True,
) -> Valuation:
    """Produce a complete valuation for ``company``.

    Args:
        qualitative_context: management commentary retrieved from the filings,
            used to justify forward assumptions. The valuation runs without it.
        use_model: set False to derive every assumption from history alone —
            used by the tests, and as the fallback when the model is
            unavailable.
    """
    valuation = Valuation(
        ticker=company.ticker,
        company_name=company.name,
        currency=history.currency,
        share_price=history.share_price,
    )

    if not history.is_sufficient_for_dcf:
        valuation.warnings.append(
            "Insufficient financial history for a discounted cash flow model: "
            "at least two years of revenue and one year of free cash flow are "
            "required."
        )
        return valuation

    ledger = valuation.assumptions

    # -- discount rate ----------------------------------------------------
    wacc = compute_wacc(history, company, ledger)
    tax_rate = ledger.value("tax_rate", 0.21)

    # -- forward assumptions ----------------------------------------------
    try:
        inputs = derive_inputs(
            history,
            company,
            ledger,
            tax_rate=tax_rate,
            qualitative_context=qualitative_context,
            use_model=use_model,
        )
    except ValueError as exc:
        valuation.warnings.append(str(exc))
        return valuation

    # -- share count ------------------------------------------------------
    # Diluted shares from the filings are preferred over the market data feed:
    # they are audited, and they include the dilution from unvested equity that
    # a raw outstanding-share count omits.
    latest = history.latest
    shares = latest.diluted_shares or history.shares_outstanding or 0.0
    if shares <= 0:
        valuation.warnings.append("Share count unavailable; per-share value cannot be computed.")
        return valuation

    net_debt = latest.net_debt if latest.net_debt is not None else 0.0

    # -- the model --------------------------------------------------------
    try:
        valuation.dcf = run_dcf(
            base_revenue=inputs.base_revenue,
            base_year=inputs.base_year,
            growth_path=inputs.growth_path,
            margin_path=inputs.margin_path,
            tax_rate=inputs.tax_rate,
            depreciation_pct=inputs.depreciation_pct,
            capex_pct=inputs.capex_pct,
            working_capital_pct=inputs.working_capital_pct,
            wacc=wacc,
            terminal_growth=inputs.terminal_growth,
            net_debt=net_debt,
            shares_outstanding=shares,
            currency=history.currency,
        )
    except ValueError as exc:
        valuation.warnings.append(f"The discounted cash flow model could not be built: {exc}")
        return valuation

    # -- sensitivity ------------------------------------------------------
    valuation.sensitivity = build_grid(
        base_revenue=inputs.base_revenue,
        base_year=inputs.base_year,
        growth_path=inputs.growth_path,
        margin_path=inputs.margin_path,
        tax_rate=inputs.tax_rate,
        depreciation_pct=inputs.depreciation_pct,
        capex_pct=inputs.capex_pct,
        working_capital_pct=inputs.working_capital_pct,
        base_wacc=wacc,
        base_growth=inputs.terminal_growth,
        net_debt=net_debt,
        shares_outstanding=shares,
    )

    # -- scenarios (bear / base / bull) -----------------------------------
    # A single fair value is a point estimate dressed up as a fact. The scenario
    # set moves the value drivers together into three coherent worlds, sized
    # from the company's own history, and reports the range and its probability-
    # weighted centre alongside the base case.
    valuation.scenarios = build_scenarios(
        inputs=inputs,
        history=history,
        base_wacc=wacc,
        base_terminal_margin=ledger.value("terminal_margin", inputs.margin_path[-1]),
        net_debt=net_debt,
        shares_outstanding=shares,
        share_price=history.share_price,
        currency=history.currency,
    )

    # -- what the market is pricing in ------------------------------------
    if history.share_price:
        valuation.market_implied_growth = implied_growth(
            base_revenue=inputs.base_revenue,
            base_year=inputs.base_year,
            horizon=len(inputs.growth_path),
            terminal_margin=ledger.value("terminal_margin", inputs.margin_path[-1]),
            current_margin=inputs.margin_path[0],
            tax_rate=inputs.tax_rate,
            depreciation_pct=inputs.depreciation_pct,
            capex_pct=inputs.capex_pct,
            working_capital_pct=inputs.working_capital_pct,
            wacc=wacc,
            terminal_growth=inputs.terminal_growth,
            net_debt=net_debt,
            shares_outstanding=shares,
            target_price=history.share_price,
        )

    # -- comparables ------------------------------------------------------
    valuation.comps = build_comps(
        company.ticker,
        net_income=latest.net_income,
        shares_outstanding=shares,
    )

    # -- triangulation (blend) --------------------------------------------
    # Reconcile our intrinsic DCF with the analyst consensus into one headline
    # figure, weighting each source by the confidence it earns. This is what the
    # product leads with; the DCF remains visible as one input.
    valuation.blended = build_blend(
        dcf_value=valuation.dcf.fair_value_per_share,
        history=history,
        ticker=company.ticker,
        share_price=history.share_price,
        currency=history.currency,
    )

    # -- diagnostics ------------------------------------------------------
    # A wide gap between our intrinsic DCF and the street's consensus is not a
    # failure — it is the whole reason to triangulate — but the reader should be
    # told the blend is reconciling two genuinely different views.
    blended = valuation.blended
    if blended and valuation.dcf.fair_value_per_share > 0:
        consensus = next(
            (e for e in blended.estimates if e.key == "analyst_consensus"), None
        )
        if consensus is not None:
            gap = consensus.value_per_share / valuation.dcf.fair_value_per_share - 1
            if abs(gap) > config.BLEND_DIVERGENCE_FLAG:
                direction = "above" if gap > 0 else "below"
                valuation.warnings.append(
                    f"The analyst consensus ({history.currency} "
                    f"{consensus.value_per_share:,.2f}) sits {abs(gap) * 100:.0f}% "
                    f"{direction} our intrinsic DCF ({history.currency} "
                    f"{valuation.dcf.fair_value_per_share:,.2f}). The blended value "
                    f"reconciles the two; read it as a triangulation, not a precise number."
                )

    # -- diagnostics (scenarios) ------------------------------------------
    # Where the current price sits relative to the whole scenario range is a
    # sharper read than upside to a single point: a price above even the bull
    # case, or below even the bear case, is a specific and checkable claim.
    scenarios = valuation.scenarios
    if scenarios and history.share_price:
        value_range = scenarios.value_range
        if value_range:
            low, high = value_range
            if history.share_price > high:
                valuation.warnings.append(
                    f"At {history.currency} {history.share_price:,.2f} the price sits "
                    f"above even the bull case ({history.currency} {high:,.2f}); the "
                    f"market is pricing assumptions more optimistic than any of the "
                    f"three scenarios."
                )
            elif history.share_price < low:
                valuation.warnings.append(
                    f"At {history.currency} {history.share_price:,.2f} the price sits "
                    f"below even the bear case ({history.currency} {low:,.2f}); the "
                    f"market is pricing in worse than the downside scenario models."
                )
        if scenarios.dispersion is not None and scenarios.dispersion > 1.5:
            valuation.warnings.append(
                f"The bull-to-bear range spans {scenarios.dispersion * 100:.0f}% of the "
                f"base fair value, so the valuation depends heavily on which scenario "
                f"plays out and should be read as a range rather than a point."
            )

    if valuation.dcf.terminal_value_share > 0.80:
        valuation.warnings.append(
            f"{valuation.dcf.terminal_value_share:.0%} of enterprise value comes "
            f"from the terminal value, so the result is driven mainly by the "
            f"perpetuity assumption rather than the explicit forecast."
        )

    for assumption in ledger.clamped:
        valuation.warnings.append(
            f"{assumption.label} was constrained to {assumption.display} "
            f"(estimate was {assumption.raw_display})."
            if assumption.raw_value is not None
            else f"{assumption.label} was constrained to {assumption.display}."
        )

    return valuation
