"""
Scenario analysis: bear, base, and bull as coherent states of the world.

The sensitivity grid flexes two inputs one at a time and answers "how fragile is
the number?". A scenario answers a different question — "what has to be true for
this to be worth much more, or much less?" — and answers it the way an analyst
would: by moving the value drivers *together* into a single, internally
consistent story.

    Bear   growth disappoints, margins compress, and the market demands a higher
           return, all at once.
    Base   the central case — identical to the headline DCF, so nothing regresses.
    Bull   growth holds up, margins recover toward their best observed level, and
           the risk premium eases.

The size of those moves is not a fixed +/-X%. Wherever the history allows it, the
growth and margin spreads are one standard deviation of the company's *own* past
revenue growth and operating margin. A business whose results have swung wildly
earns a wide bear-to-bull range; a steady compounder earns a narrow one. Config
floors and caps stop a single anomalous year from opening an indefensible gap,
and supply a fallback when the history is too short to measure dispersion.

Everything here is deterministic and model-free: the same inputs always produce
the same three cases, so a scenario in the report can be reproduced exactly.
"""

from __future__ import annotations

import logging
import math

from .. import config
from ..fundamentals import FinancialHistory
from .assumptions import ForecastInputs
from .dcf import decay_path, fade, run_dcf
from .models import ScenarioAnalysis, ScenarioCase, ScenarioDriver

log = logging.getLogger(__name__)

# Hard ceiling on any scenario's year-one growth, mirroring the cap enforced in
# assumptions.py. A bull case is still a forecast, not a fantasy: even a company
# that just grew faster than this is not projected above it.
_GROWTH_HARD_CEILING = 0.75

# Minimum clearance kept between terminal growth and the discount rate. The
# Gordon perpetuity diverges as the two converge, so a scenario that would push
# them together is pulled back to a value that still converges.
_MIN_WACC_GROWTH_GAP = 0.005


def _stdev(values: list[float]) -> float | None:
    """Population standard deviation, or None if there is too little to measure."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _growth_spread(history: FinancialHistory) -> tuple[float, str]:
    """Size the year-one growth move from historical revenue-growth dispersion."""
    rates = [g for _year, g in history.growth_rates("revenue")]
    sigma = _stdev(rates)
    if sigma is None:
        spread = (config.SCENARIO_MIN_GROWTH_SPREAD + config.SCENARIO_MAX_GROWTH_SPREAD) / 2
        return spread, "history too short to measure growth dispersion; mid-range spread used"
    spread = min(max(sigma, config.SCENARIO_MIN_GROWTH_SPREAD), config.SCENARIO_MAX_GROWTH_SPREAD)
    return spread, f"one standard deviation of {len(rates)} years of revenue growth"


def _margin_spread(history: FinancialHistory) -> tuple[float, str, tuple[float, float] | None]:
    """Size the terminal-margin move from historical operating-margin dispersion.

    Returns the spread, its derivation, and the observed (min, max) margin band
    that caps how far the bear and bull cases may travel.
    """
    margins = [y.operating_margin for y in history.years if y.operating_margin is not None]
    band = (min(margins), max(margins)) if margins else None
    sigma = _stdev(margins)
    if sigma is None:
        spread = (config.SCENARIO_MIN_MARGIN_SPREAD + config.SCENARIO_MAX_MARGIN_SPREAD) / 2
        return spread, "history too short to measure margin dispersion; mid-range spread used", band
    spread = min(max(sigma, config.SCENARIO_MIN_MARGIN_SPREAD), config.SCENARIO_MAX_MARGIN_SPREAD)
    return spread, f"one standard deviation of {len(margins)} years of operating margin", band


def _run_case(
    *,
    key: str,
    label: str,
    probability: float,
    narrative: str,
    inputs: ForecastInputs,
    currency: str,
    current_margin: float,
    year_one_growth: float,
    terminal_margin: float,
    terminal_growth: float,
    wacc: float,
    net_debt: float,
    shares_outstanding: float,
    share_price: float | None,
    base_year_one_growth: float,
    base_terminal_margin: float,
    base_terminal_growth: float,
    base_wacc: float,
) -> ScenarioCase | None:
    """Rebuild the forecast paths for one case, run the DCF, and package it."""
    horizon = len(inputs.growth_path)

    # Keep the perpetuity convergent even at the low-WACC / high-growth corner.
    terminal_growth = min(terminal_growth, wacc - _MIN_WACC_GROWTH_GAP)

    growth_path = decay_path(year_one_growth, terminal_growth, horizon, inputs.growth_decay)
    margin_path = fade(current_margin, terminal_margin, horizon)

    try:
        result = run_dcf(
            base_revenue=inputs.base_revenue,
            base_year=inputs.base_year,
            growth_path=growth_path,
            margin_path=margin_path,
            tax_rate=inputs.tax_rate,
            depreciation_pct=inputs.depreciation_pct,
            capex_pct=inputs.capex_pct,
            working_capital_pct=inputs.working_capital_pct,
            wacc=wacc,
            terminal_growth=terminal_growth,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            currency=currency,
        )
    except ValueError as exc:
        log.info("scenario %s could not be built: %s", key, exc)
        return None

    upside = (
        result.fair_value_per_share / share_price - 1
        if share_price and result.fair_value_per_share
        else None
    )

    drivers = [
        ScenarioDriver(
            key="year_one_growth",
            label="Year 1 revenue growth",
            value=year_one_growth,
            unit="%",
            base_value=base_year_one_growth,
        ),
        ScenarioDriver(
            key="terminal_margin",
            label="Terminal operating margin",
            value=terminal_margin,
            unit="%",
            base_value=base_terminal_margin,
        ),
        ScenarioDriver(
            key="terminal_growth",
            label="Terminal growth",
            value=terminal_growth,
            unit="%",
            base_value=base_terminal_growth,
        ),
        ScenarioDriver(
            key="wacc",
            label="Discount rate (WACC)",
            value=wacc,
            unit="%",
            base_value=base_wacc,
        ),
    ]

    return ScenarioCase(
        key=key,
        label=label,
        probability=probability,
        narrative=narrative,
        drivers=drivers,
        fair_value_per_share=result.fair_value_per_share,
        enterprise_value=result.enterprise_value,
        equity_value=result.equity_value,
        terminal_value_share=result.terminal_value_share,
        upside=upside,
    )


def build_scenarios(
    *,
    inputs: ForecastInputs,
    history: FinancialHistory,
    base_wacc: float,
    base_terminal_margin: float,
    net_debt: float,
    shares_outstanding: float,
    share_price: float | None,
    currency: str = "USD",
) -> ScenarioAnalysis:
    """Construct the bear / base / bull set for ``inputs``.

    The base case reuses the exact inputs of the headline DCF, so it reproduces
    that fair value to the cent. The bear and bull cases move growth, margin,
    terminal growth, and the discount rate together, by amounts sized from the
    company's own historical dispersion and bounded by config rails.
    """
    base_year_one_growth = inputs.growth_path[0]
    base_terminal_growth = inputs.terminal_growth
    current_margin = inputs.margin_path[0]

    growth_spread, growth_basis = _growth_spread(history)
    margin_spread, margin_basis, margin_band = _margin_spread(history)

    wacc_delta = config.SCENARIO_WACC_DELTA
    tg_delta = config.SCENARIO_TERMINAL_GROWTH_DELTA
    wacc_lo, wacc_hi = config.WACC_BOUNDS
    tg_lo, tg_hi = config.TERMINAL_GROWTH_BOUNDS

    # -- bear -------------------------------------------------------------
    bear_wacc = min(base_wacc + wacc_delta, wacc_hi)
    bear_tg = max(base_terminal_growth - tg_delta, tg_lo)
    bear_margin = base_terminal_margin - margin_spread
    if margin_band:
        bear_margin = max(bear_margin, margin_band[0])
    bear_margin = max(bear_margin, 0.0)
    # Near-term growth can dip in a downside, but not below the perpetual rate:
    # that would assert the business collapses to sub-GDP growth within a year.
    bear_growth = max(base_year_one_growth - growth_spread, bear_tg)

    # -- bull -------------------------------------------------------------
    bull_wacc = max(base_wacc - wacc_delta, wacc_lo)
    bull_tg = min(base_terminal_growth + tg_delta, tg_hi)
    bull_margin = base_terminal_margin + margin_spread
    if margin_band:
        # The bull case restores margins to their best observed level, no higher:
        # margin expansion beyond anything the company has ever posted is a claim
        # a DCF cannot support.
        bull_margin = min(bull_margin, margin_band[1])
    bull_growth = min(base_year_one_growth + growth_spread, _GROWTH_HARD_CEILING)

    spread_note = (
        f"Growth spread +/-{growth_spread * 100:.0f}pp ({growth_basis}); "
        f"margin spread +/-{margin_spread * 100:.0f}pp ({margin_basis}); "
        f"discount rate +/-{wacc_delta * 10000:.0f}bps; "
        f"terminal growth +/-{tg_delta * 10000:.0f}bps."
    )

    specs = [
        {
            "key": "bear",
            "label": "Bear case",
            "probability": config.SCENARIO_PROBABILITIES.get("bear", 0.25),
            "narrative": (
                f"Demand softens and competition bites: year-one growth slows to "
                f"{bear_growth * 100:.0f}%, the sustainable operating margin settles "
                f"near {bear_margin * 100:.0f}%, and the market prices the added risk "
                f"through a {bear_wacc * 100:.1f}% discount rate. {spread_note}"
            ),
            "year_one_growth": bear_growth,
            "terminal_margin": bear_margin,
            "terminal_growth": bear_tg,
            "wacc": bear_wacc,
        },
        {
            "key": "base",
            "label": "Base case",
            "probability": config.SCENARIO_PROBABILITIES.get("base", 0.50),
            "narrative": (
                f"The central case, identical to the headline DCF: year-one growth "
                f"of {base_year_one_growth * 100:.0f}% decaying toward "
                f"{base_terminal_growth * 100:.1f}%, a terminal margin of "
                f"{base_terminal_margin * 100:.0f}%, discounted at "
                f"{base_wacc * 100:.1f}%."
            ),
            "year_one_growth": base_year_one_growth,
            "terminal_margin": base_terminal_margin,
            "terminal_growth": base_terminal_growth,
            "wacc": base_wacc,
        },
        {
            "key": "bull",
            "label": "Bull case",
            "probability": config.SCENARIO_PROBABILITIES.get("bull", 0.25),
            "narrative": (
                f"Growth proves durable and operating leverage plays out: year-one "
                f"growth holds at {bull_growth * 100:.0f}%, margins recover toward "
                f"{bull_margin * 100:.0f}%, and an easing risk premium lowers the "
                f"discount rate to {bull_wacc * 100:.1f}%. {spread_note}"
            ),
            "year_one_growth": bull_growth,
            "terminal_margin": bull_margin,
            "terminal_growth": bull_tg,
            "wacc": bull_wacc,
        },
    ]

    analysis = ScenarioAnalysis(currency=currency, share_price=share_price)

    for spec in specs:
        case = _run_case(
            key=spec["key"],
            label=spec["label"],
            probability=spec["probability"],
            narrative=spec["narrative"],
            inputs=inputs,
            currency=currency,
            current_margin=current_margin,
            year_one_growth=spec["year_one_growth"],
            terminal_margin=spec["terminal_margin"],
            terminal_growth=spec["terminal_growth"],
            wacc=spec["wacc"],
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            share_price=share_price,
            base_year_one_growth=base_year_one_growth,
            base_terminal_margin=base_terminal_margin,
            base_terminal_growth=base_terminal_growth,
            base_wacc=base_wacc,
        )
        if case is not None:
            analysis.cases.append(case)

    # Probability-weighted expected value. Renormalise over the cases that
    # actually built, so a dropped corner does not silently deflate the mean.
    total_probability = sum(c.probability for c in analysis.cases)
    if total_probability > 0:
        analysis.expected_value = sum(
            c.probability * c.fair_value_per_share for c in analysis.cases
        ) / total_probability
        if share_price and analysis.expected_value:
            analysis.expected_upside = analysis.expected_value / share_price - 1

    return analysis
