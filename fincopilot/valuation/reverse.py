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
from typing import Callable

from .. import config
from .dcf import decay_path, fade, run_dcf
from .models import PricedInComparison, PricedInRow

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
    growth_capex_per_revenue: float | None = None,
    working_capital_pct: float,
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares_outstanding: float,
    target_price: float,
    growth_decay: float = config.DCF_GROWTH_DECAY,
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
                year_one_growth, terminal_growth, horizon, growth_decay
            ),
            margin_path=fade(current_margin, terminal_margin, horizon),
            tax_rate=tax_rate,
            depreciation_pct=depreciation_pct,
            capex_pct=capex_pct,
            growth_capex_per_revenue=growth_capex_per_revenue,
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


# ---------------------------------------------------------------------------
# "What is priced in": a reverse DCF on every driver, not just growth
# ---------------------------------------------------------------------------
#
# implied_growth answers one question — what revenue growth justifies the price?
# A reader looking at a fair value below the market price has three more: what
# margin, what cash conversion, what perpetual growth would justify it instead?
#
# Each is solved the same way and with the same engine: hold every *other* driver
# at our base case and solve for the one that lifts (or lowers) the DCF to today's
# price. Every solve therefore reproduces the base fair value exactly at the base
# value of its driver, so the "our base case" and "market-implied" columns are
# strictly comparable — the only thing that changed between them is the one number
# on that row. The result is a table of testable statements ("the price needs a
# 55% mature operating margin, versus our 42%") rather than a single verdict.


def _solve(value_at: Callable[[float], float], target: float, low: float, high: float) -> float | None:
    """Bisection for the input in ``[low, high]`` whose DCF equals ``target``.

    Works whether fair value rises or falls in the input (margin rises it, capex
    lowers it): the orientation is read from the endpoints. Returns ``None`` when
    the target lies outside the achievable range — which is itself informative,
    since it means the price cannot be reached on that lever alone.
    """
    try:
        value_low = value_at(low)
        value_high = value_at(high)
    except ValueError as exc:
        log.info("priced-in solve not computable: %s", exc)
        return None

    increasing = value_high >= value_low
    reachable_low = min(value_low, value_high)
    reachable_high = max(value_low, value_high)
    if not (reachable_low <= target <= reachable_high):
        return None

    for _ in range(_MAX_ITERATIONS):
        midpoint = (low + high) / 2
        if abs(high - low) < _TOLERANCE:
            return midpoint
        try:
            value = value_at(midpoint)
        except ValueError:
            return None
        if (value < target) == increasing:
            low = midpoint
        else:
            high = midpoint

    return (low + high) / 2


def _cagr(growth_path: list[float]) -> float:
    """Compound annual growth rate implied by a per-year growth path."""
    if not growth_path:
        return 0.0
    compound = 1.0
    for growth in growth_path:
        compound *= 1 + growth
    return compound ** (1 / len(growth_path)) - 1


def build_priced_in(
    *,
    inputs,
    wacc: float,
    net_debt: float,
    shares_outstanding: float,
    terminal_margin: float,
    base_dcf,
    share_price: float,
    currency: str,
    implied_year_one_growth: float | None,
) -> PricedInComparison | None:
    """Assemble the "what is priced in" comparison for every driver.

    ``inputs`` is the :class:`ForecastInputs` behind the base DCF; ``base_dcf`` is
    that DCF's :class:`DCFResult`. ``implied_year_one_growth`` is the already-solved
    reverse-DCF growth (reused so it is not recomputed).
    """
    if not share_price or share_price <= 0 or base_dcf is None or not base_dcf.forecast:
        return None

    horizon = len(inputs.growth_path)
    year_one_growth = inputs.growth_path[0]
    current_margin = inputs.margin_path[0]
    price = share_price

    common = dict(
        base_revenue=inputs.base_revenue,
        base_year=inputs.base_year,
        tax_rate=inputs.tax_rate,
        depreciation_pct=inputs.depreciation_pct,
        working_capital_pct=inputs.working_capital_pct,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
        wacc=wacc,
    )

    rows: list[PricedInRow] = []

    # -- 1. revenue growth, expressed as the forecast-period CAGR ---------
    base_cagr = _cagr(inputs.growth_path)
    if implied_year_one_growth is not None:
        implied_path = decay_path(
            implied_year_one_growth, inputs.terminal_growth, horizon, inputs.growth_decay
        )
        rows.append(PricedInRow(
            key="revenue_cagr",
            label=f"Revenue CAGR ({horizon}-year)",
            unit="%",
            base_value=base_cagr,
            implied_value=_cagr(implied_path),
            note=(
                f"One front-loaded path, not two assumptions: ~{implied_year_one_growth * 100:.0f}% "
                f"growth in year 1 decaying toward the terminal rate, which compounds to this "
                f"{_cagr(implied_path) * 100:.1f}% {horizon}-year CAGR. The high first-year figure and "
                f"the CAGR are the same trajectory seen at two points."
            ),
        ))
    else:
        rows.append(PricedInRow(
            key="revenue_cagr",
            label=f"Revenue CAGR ({horizon}-year)",
            unit="%",
            base_value=base_cagr,
            implied_value=None,
            reachable=False,
            note="Revenue growth alone cannot reach the price within a plausible range.",
        ))

    # -- 2. mature operating margin --------------------------------------
    def margin_value(mature_margin: float) -> float:
        return run_dcf(
            growth_path=inputs.growth_path,
            margin_path=fade(current_margin, mature_margin, horizon),
            terminal_growth=inputs.terminal_growth,
            capex_pct=inputs.capex_pct,
            **common,
        ).fair_value_per_share

    # Capped at 100%: an operating margin above revenue is not a number a
    # business can post, so if the price is unreachable even there, the honest
    # statement is that margin alone cannot justify it — not some absurd figure.
    implied_margin = _solve(margin_value, price, 0.0, 1.0)
    margin_note = ""
    if implied_margin is None:
        margin_note = (
            "Even a 100%-of-revenue operating margin — the theoretical ceiling — does not "
            "reach the price; it cannot be justified on margin alone."
        )
    elif implied_margin > 0.80:
        margin_note = "An operating margin almost no company has ever sustained."
    rows.append(PricedInRow(
        key="operating_margin",
        label="Operating margin (mature)",
        unit="%",
        base_value=terminal_margin,
        implied_value=implied_margin,
        reachable=implied_margin is not None,
        note=margin_note,
    ))

    # -- 3. mature free-cash-flow margin ---------------------------------
    # Solved through capex intensity, the one input that moves the cash margin
    # one-for-one; the row reports the resulting FCF/revenue, not the capex.
    final = base_dcf.forecast[-1]
    base_fcf_margin = final.free_cash_flow / final.revenue if final.revenue else 0.0

    def capex_value(capex_pct: float) -> float:
        return run_dcf(
            growth_path=inputs.growth_path,
            margin_path=inputs.margin_path,
            terminal_growth=inputs.terminal_growth,
            capex_pct=capex_pct,
            # Deliberately NOT the growth-capex model here. This solver moves
            # capex intensity as its lever to find the cash margin the price
            # implies; deriving capex from growth instead would override the
            # very variable being solved for and the search would not move.
            growth_capex_per_revenue=None,
            **common,
        ).fair_value_per_share

    # Floor the search at zero capex: a company cannot spend less than nothing to
    # grow, so zero capex is the ceiling on cash conversion. If the price is
    # unreachable even there, no plausible cash margin justifies it.
    implied_capex = _solve(capex_value, price, 0.0, inputs.capex_pct + 1.0)
    implied_fcf_margin: float | None = None
    fcf_note = ""
    if implied_capex is not None:
        solved = run_dcf(
            growth_path=inputs.growth_path,
            margin_path=inputs.margin_path,
            terminal_growth=inputs.terminal_growth,
            capex_pct=implied_capex,
            **common,
        )
        solved_final = solved.forecast[-1]
        implied_fcf_margin = (
            solved_final.free_cash_flow / solved_final.revenue if solved_final.revenue else None
        )
        if implied_fcf_margin is not None and implied_fcf_margin > current_margin:
            fcf_note = "Exceeds the operating margin itself — it implies negative reinvestment, which no growing business sustains."
    else:
        fcf_note = (
            "Even at zero capital spending — the ceiling on cash conversion — the implied cash "
            "margin cannot reach the price."
        )
    rows.append(PricedInRow(
        key="fcf_margin",
        label="FCF margin (mature)",
        unit="%",
        base_value=base_fcf_margin,
        implied_value=implied_fcf_margin,
        reachable=implied_fcf_margin is not None,
        note=fcf_note,
    ))

    # -- 4. terminal (perpetual) growth ----------------------------------
    # Bounded strictly below WACC: at or above it the Gordon terminal value
    # diverges, so the search stops a quarter-point short.
    def terminal_growth_value(terminal_growth: float) -> float:
        return run_dcf(
            growth_path=decay_path(year_one_growth, terminal_growth, horizon, inputs.growth_decay),
            margin_path=inputs.margin_path,
            terminal_growth=terminal_growth,
            capex_pct=inputs.capex_pct,
            **common,
        ).fair_value_per_share

    implied_terminal = _solve(terminal_growth_value, price, 0.0, wacc - 0.0025)
    terminal_note = (
        "A mathematical solution holding all other assumptions fixed: the perpetual rate that "
        "alone would justify the price, not a claim that investors literally assume this rate "
        "forever. "
    )
    if implied_terminal is None:
        terminal_note = "Unreachable even at a perpetual rate just below the discount rate."
    elif implied_terminal > 0.04:
        terminal_note += "It sits above long-run GDP growth — implying the company outgrows the economy forever."
    rows.append(PricedInRow(
        key="terminal_growth",
        label="Terminal growth",
        unit="%",
        base_value=inputs.terminal_growth,
        implied_value=implied_terminal,
        reachable=implied_terminal is not None,
        note=terminal_note,
    ))

    return PricedInComparison(
        rows=rows,
        currency=currency,
        share_price=share_price,
        dcf_fair_value=base_dcf.fair_value_per_share,
        horizon=horizon,
        summary=_priced_in_summary(rows),
    )


def _priced_in_summary(rows: list[PricedInRow]) -> str:
    """Which single assumption the price leans on most — the work-done synthesis."""
    reachable = [r for r in rows if r.implied_value is not None]
    unreachable = [r for r in rows if r.implied_value is None]

    if not reachable:
        return (
            "No single driver can reach the price within a plausible range: the price assumes a "
            "combination more optimistic than any one lever on its own."
        )

    # The lever the market stretches furthest above our base is doing the most work.
    dominant = max(reachable, key=lambda r: r.implied_value - r.base_value)
    parts = [
        f"Most of the work is done by {dominant.label.lower()}: the price implies "
        f"{dominant.implied_display} against our {dominant.base_display} ({dominant.gap_display}) — "
        f"the bull case is, before anything else, a bet that this holds."
    ]
    if unreachable:
        names = ", ".join(r.label.lower() for r in unreachable)
        it = "them" if len(unreachable) > 1 else "it"
        parts.append(
            f"By contrast, {names} cannot reach the price alone at any plausible level, so the "
            f"price is not resting on {it}."
        )
    return " ".join(parts)
