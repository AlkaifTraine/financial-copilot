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
from .models import SensitivityGrid

log = logging.getLogger(__name__)


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
