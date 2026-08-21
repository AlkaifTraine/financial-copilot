"""
Sensitivity analysis across the two assumptions that dominate a DCF.

A single fair value implies a precision that no discounted cash flow model has.
The discount rate and the terminal growth rate together drive most of the
answer, and both are estimates, so the honest output is a surface rather than a
point. The grid shows how much of the conclusion survives moving them.

The same forecast cash flows are reused across the grid — only the discounting
and the perpetuity change — so this is cheap to compute and internally
consistent with the base case.
"""

from __future__ import annotations

import logging

from .. import config
from .dcf import run_dcf
from .models import CompetitionSensitivity, SensitivityGrid

log = logging.getLogger(__name__)


def build_competition_sensitivity(
    *,
    base_revenue: float,
    base_year: int,
    growth_path: list[float],
    margin_path: list[float],
    tax_rate: float,
    depreciation_pct: float,
    capex_pct: float,
    working_capital_pct: float,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
    currency: str,
    base_fair_value: float,
    haircuts: tuple[float, ...] = (0.02, 0.05, 0.10),
) -> CompetitionSensitivity:
    """Translate a competitive share/growth loss into a revenue and fair-value hit.

    Each haircut shifts every year's growth rate down by that many points (a
    persistent loss of share), floored at the terminal rate, and re-runs the DCF.
    """
    def terminal_revenue(path: list[float]) -> float:
        revenue = base_revenue
        for growth in path:
            revenue *= (1.0 + growth)
        return revenue

    base_terminal = terminal_revenue(growth_path)
    result = CompetitionSensitivity(
        base_fair_value=base_fair_value,
        base_terminal_revenue=base_terminal,
        currency=currency,
    )
    if not growth_path or base_fair_value <= 0:
        return result

    horizon = len(growth_path)
    for haircut in haircuts:
        shifted = [max(growth - haircut, terminal_growth) for growth in growth_path]
        try:
            fair_value = run_dcf(
                base_revenue=base_revenue, base_year=base_year, growth_path=shifted,
                margin_path=margin_path, tax_rate=tax_rate, depreciation_pct=depreciation_pct,
                capex_pct=capex_pct, working_capital_pct=working_capital_pct, wacc=wacc,
                terminal_growth=terminal_growth, net_debt=net_debt,
                shares_outstanding=shares_outstanding, currency=currency,
            ).fair_value_per_share
        except ValueError:
            continue
        shifted_terminal = terminal_revenue(shifted)
        result.rows.append({
            "label": f"-{haircut * 100:.0f}pp annual growth (share loss)",
            "cagr": (shifted_terminal / base_revenue) ** (1.0 / horizon) - 1.0,
            "terminal_revenue": shifted_terminal,
            "revenue_change": shifted_terminal / base_terminal - 1.0 if base_terminal else 0.0,
            "fair_value": fair_value,
            "fair_value_change": fair_value / base_fair_value - 1.0,
        })
    return result


def build_grid(
    *,
    base_revenue: float,
    base_year: int,
    growth_path: list[float],
    margin_path: list[float],
    tax_rate: float,
    depreciation_pct: float,
    capex_pct: float,
    working_capital_pct: float,
    base_wacc: float,
    base_growth: float,
    net_debt: float,
    shares_outstanding: float,
) -> SensitivityGrid:
    """Fair value per share over a WACC x terminal-growth grid."""
    wacc_steps = config.SENSITIVITY_WACC_STEPS
    growth_steps = config.SENSITIVITY_GROWTH_STEPS

    wacc_values = [
        base_wacc + (i - wacc_steps // 2) * config.SENSITIVITY_WACC_DELTA
        for i in range(wacc_steps)
    ]
    growth_values = [
        base_growth + (i - growth_steps // 2) * config.SENSITIVITY_GROWTH_DELTA
        for i in range(growth_steps)
    ]

    grid = SensitivityGrid(
        wacc_values=wacc_values,
        growth_values=growth_values,
        base_wacc=base_wacc,
        base_growth=base_growth,
    )

    for wacc in wacc_values:
        row: list[float] = []
        for growth in growth_values:
            try:
                result = run_dcf(
                    base_revenue=base_revenue,
                    base_year=base_year,
                    growth_path=growth_path,
                    margin_path=margin_path,
                    tax_rate=tax_rate,
                    depreciation_pct=depreciation_pct,
                    capex_pct=capex_pct,
                    working_capital_pct=working_capital_pct,
                    wacc=wacc,
                    terminal_growth=growth,
                    net_debt=net_debt,
                    shares_outstanding=shares_outstanding,
                )
                row.append(result.fair_value_per_share)
            except ValueError:
                # Terminal growth at or above WACC: the perpetuity does not
                # converge. Reported as absent rather than as a number.
                row.append(float("nan"))
        grid.values.append(row)

    return grid
