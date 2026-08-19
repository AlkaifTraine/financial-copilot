"""
Discounted cash flow: the arithmetic.

Deliberately a pure function. No network, no language model, no global state —
inputs in, :class:`DCFResult` out. Two consequences that matter:

* The same inputs always produce the same valuation, so a number in the report
  can be reproduced exactly.
* It is unit-testable against a hand-computed model, which is the only way to
  know the arithmetic is right.

The model is an unlevered (free-cash-flow-to-firm) DCF:

    FCFF = EBIT x (1 - tax) + D&A - capex - change in working capital

discounted at WACC to enterprise value, with a Gordon-growth terminal value.
Net debt is then removed to reach equity value, and equity value divided by
diluted shares to reach a per-share fair value.
"""

from __future__ import annotations

import logging

from .models import DCFResult, ForecastYear

log = logging.getLogger(__name__)


def run_dcf(
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
    currency: str = "USD",
) -> DCFResult:
    """Project cash flows and discount them to a per-share value.

    Args:
        growth_path: revenue growth for each forecast year, e.g. ``[0.45, 0.30]``.
        margin_path: operating margin for each forecast year; same length.
        depreciation_pct: D&A as a share of revenue.
        capex_pct: capital expenditure as a share of revenue.
        working_capital_pct: incremental working capital as a share of the
            *change* in revenue — growth consumes cash before it produces it.

    Raises:
        ValueError: if the inputs cannot produce a finite valuation.
    """
    if len(growth_path) != len(margin_path):
        raise ValueError("growth_path and margin_path must be the same length")
    if not growth_path:
        raise ValueError("forecast horizon must be at least one year")
    if base_revenue <= 0:
        raise ValueError("base revenue must be positive")

    # The Gordon growth formula diverges as g approaches WACC and goes negative
    # beyond it. A terminal growth rate at or above the discount rate asserts
    # the company outgrows its cost of capital forever, which is not a
    # valuation — it is an arithmetic error with a dollar sign attached.
    if terminal_growth >= wacc:
        raise ValueError(
            f"terminal growth ({terminal_growth:.2%}) must be below "
            f"WACC ({wacc:.2%}) for the terminal value to converge"
        )

    result = DCFResult(wacc=wacc, terminal_growth=terminal_growth, currency=currency)

    revenue = base_revenue
    previous_revenue = base_revenue

    for offset, (growth, margin) in enumerate(zip(growth_path, margin_path), start=1):
        revenue = previous_revenue * (1 + growth)

        ebit = revenue * margin
        nopat = ebit * (1 - tax_rate)
        depreciation = revenue * depreciation_pct
        capex = revenue * capex_pct
        change_in_wc = (revenue - previous_revenue) * working_capital_pct

        free_cash_flow = nopat + depreciation - capex - change_in_wc

        discount_factor = 1.0 / ((1 + wacc) ** offset)
        present_value = free_cash_flow * discount_factor

        result.forecast.append(
            ForecastYear(
                year=base_year + offset,
                revenue=revenue,
                revenue_growth=growth,
                ebit=ebit,
                operating_margin=margin,
                nopat=nopat,
                depreciation=depreciation,
                capex=capex,
                change_in_working_capital=change_in_wc,
                free_cash_flow=free_cash_flow,
                discount_factor=discount_factor,
                present_value=present_value,
            )
        )

        previous_revenue = revenue

    result.pv_forecast = sum(year.present_value for year in result.forecast)

    # Terminal value: the perpetuity beginning after the explicit forecast,
    # discounted back from the final forecast year.
    final = result.forecast[-1]
    result.terminal_value = final.free_cash_flow * (1 + terminal_growth) / (wacc - terminal_growth)
    result.pv_terminal = result.terminal_value * final.discount_factor

    result.enterprise_value = result.pv_forecast + result.pv_terminal
    result.net_debt = net_debt
    result.equity_value = result.enterprise_value - net_debt
    result.shares_outstanding = shares_outstanding

    result.fair_value_per_share = (
        result.equity_value / shares_outstanding if shares_outstanding > 0 else 0.0
    )

    return result


def decay_path(start: float, end: float, periods: int, decay: float = 0.70) -> list[float]:
    """Growth decaying geometrically from ``start`` toward ``end``.

    Used for revenue growth, where a linear fade is badly wrong over a long
    horizon. Fading NVIDIA's 65% linearly to 2.5% across ten years still
    averages 33% a year and compounds to **17x** — a forecast of $3.6 trillion
    of revenue, roughly a tenth of US GDP. The chart made that obvious
    instantly; the terminal-value diagnostics did not, because the ratio looked
    healthier the more the explicit forecast inflated.

    Geometric decay matches how high growth actually behaves: it falls fastest
    early, as the base effect bites, then flattens. At the default decay the
    same 65% start reaches about 7x cumulative growth over ten years instead of
    17x — long enough to credit a genuine multi-year compounder without
    forecasting it to swallow the economy.

    Margins keep the linear :func:`fade` — a margin genuinely does trend toward
    its steady state roughly linearly, and it is bounded anyway.
    """
    if periods <= 0:
        return []
    gap = start - end
    return [end + gap * (decay**i) for i in range(periods)]


def fade(start: float, end: float, periods: int) -> list[float]:
    """Linearly interpolate from ``start`` to ``end`` across ``periods`` years.

    Used for both growth and margin paths. A company growing 65% cannot keep
    doing so, and holding the current rate flat across the forecast is the most
    common way a DCF produces an absurd number. Fading toward a sustainable
    long-run rate is the standard treatment and makes the decay explicit rather
    than hidden in a single averaged input.
    """
    if periods <= 0:
        return []
    if periods == 1:
        return [end]
    step = (end - start) / (periods - 1)
    return [start + step * i for i in range(periods)]
