"""
Reverse DCF: what does the current share price already assume?

A forward DCF answers "what is this worth?", and on large, highly-rated
companies it frequently answers "much less than it trades for". Presented on
its own that reads as a prediction the market is wrong, which is a weak claim
for any model to make and an easy one for a reader to dismiss.

Inverting the model is more useful and more honest. Holding the discount rate,
margins and reinvestment fixed, solve for the revenue growth rate that makes
the DCF output equal today's price. The result is a statement about the market
rather than a bet against it:

    "At $383, the price implies 21% annual revenue growth for ten years."

The reader can then judge that number against the company's history, its
guidance, and its market size — which is a question they can actually answer.
Solved by bisection because fair value increases monotonically in the growth
rate, holding everything else constant.
"""

from __future__ import annotations

import logging

from .. import config
from .dcf import decay_path, fade, run_dcf

log = logging.getLogger(__name__)

_LOWER_BOUND = -0.50
_UPPER_BOUND = 3.00
_TOLERANCE = 0.0005          # half a basis point of growth
_MAX_ITERATIONS = 60


def implied_growth(
    *,
    base_revenue: float,
    base_year: int,
    horizon: int,
    terminal_margin: float,
    current_margin: float,
    tax_rate: float,
    depreciation_pct: float,
    capex_pct: float,
    working_capital_pct: float,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
    target_price: float,
) -> float | None:
    """Year-one revenue growth implied by ``target_price``.

    Returns ``None`` when the price cannot be reached anywhere in the search
    range — for instance when it sits below the value of the existing cash
    flows even at zero growth.
    """
    if target_price <= 0 or shares_outstanding <= 0 or base_revenue <= 0:
        return None

    def value_at(year_one_growth: float) -> float:
        result = run_dcf(
            base_revenue=base_revenue,
            base_year=base_year,
            growth_path=decay_path(
                year_one_growth, terminal_growth, horizon, config.DCF_GROWTH_DECAY
            ),
            margin_path=fade(current_margin, terminal_margin, horizon),
            tax_rate=tax_rate,
            depreciation_pct=depreciation_pct,
            capex_pct=capex_pct,
            working_capital_pct=working_capital_pct,
            wacc=wacc,
            terminal_growth=terminal_growth,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
        )
        return result.fair_value_per_share

    try:
        low_value = value_at(_LOWER_BOUND)
        high_value = value_at(_UPPER_BOUND)
    except ValueError as exc:
        log.info("implied growth not computable: %s", exc)
        return None

    # The target has to lie inside the achievable range for a root to exist.
    if not (low_value <= target_price <= high_value):
        log.info(
            "price %.2f outside the achievable range [%.2f, %.2f]",
            target_price,
            low_value,
            high_value,
        )
        return None

    low, high = _LOWER_BOUND, _UPPER_BOUND

    for _ in range(_MAX_ITERATIONS):
        midpoint = (low + high) / 2
        try:
            value = value_at(midpoint)
        except ValueError:
            return None

        if abs(high - low) < _TOLERANCE:
            return midpoint

        if value < target_price:
            low = midpoint
        else:
            high = midpoint

    return (low + high) / 2
