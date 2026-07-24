"""Valuation: deterministic DCF, WACC, comparables, and sensitivity."""

from __future__ import annotations

import logging

from ..fundamentals import FinancialHistory
from ..resolve import Company
from .assumptions import derive_inputs
from .comps import build_comps
from .dcf import run_dcf
from .reverse import implied_growth
from .models import (
    Assumption,
    AssumptionLedger,
    CompsResult,
    DCFResult,
    SensitivityGrid,
    Valuation,
)
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

    # -- diagnostics ------------------------------------------------------
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
