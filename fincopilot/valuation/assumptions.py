"""
Deriving forecast assumptions from the company's own history.

This module is where the project's central rule is enforced: **the language
model proposes, the data disposes**.

Every assumption starts as a figure computed from the filings — trailing growth,
average margin, capex intensity. The model is then shown that history together
with management's own commentary retrieved from the filings, and may adjust a
small number of forward-looking inputs. Each adjustment is clamped to a range
derived from the company's actual results before it reaches the arithmetic, and
any clamping is recorded and reported.

That ordering matters. Asked to value a company growing 65%, a model will
happily project 65% forever and produce a fair value several times the market
cap. Bounding the proposal to what the history supports is what separates a
defensible model from a confident-sounding one, and the whole thing still works
with the model switched off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .. import config
from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company
from .dcf import decay_path, fade
from .models import (
    SOURCE_DEFAULT,
    SOURCE_HISTORICAL,
    SOURCE_MODEL,
    Assumption,
    AssumptionLedger,
)

log = logging.getLogger(__name__)

# Hard ceiling on year-one revenue growth regardless of history. Even a company
# that just grew 65% is not forecast above this without an explicit override.
_MAX_YEAR_ONE_GROWTH = 0.75
_MIN_YEAR_ONE_GROWTH = -0.30

# Fallback incremental working capital, as a share of incremental revenue.
_DEFAULT_WORKING_CAPITAL_PCT = 0.05

# Ceiling on incremental working capital intensity. Sustained absorption above
# this level would itself be the story of the company, not a forecast input.
_MAX_WORKING_CAPITAL_PCT = 0.25


@dataclass
class ForecastInputs:
    base_revenue: float
    base_year: int
    growth_path: list[float] = field(default_factory=list)
    margin_path: list[float] = field(default_factory=list)
    tax_rate: float = 0.21
    depreciation_pct: float = 0.03
    capex_pct: float = 0.04
    working_capital_pct: float = _DEFAULT_WORKING_CAPITAL_PCT
    terminal_growth: float = 0.025


def _ratio_to_revenue(history: FinancialHistory, attribute: str, last_n: int = 3) -> float | None:
    """Mean of an item as a share of revenue over recent years."""
    ratios = []
    for year in history.years[-last_n:]:
        value = getattr(year, attribute, None)
        if value is not None and year.revenue:
            ratios.append(abs(value) / year.revenue)
    return sum(ratios) / len(ratios) if ratios else None


def _clamp(value: float, bounds: tuple[float, float]) -> tuple[float, bool]:
    clamped = min(max(value, bounds[0]), bounds[1])
    return clamped, clamped != value


def _margin_floor_fraction(margins: list[float]) -> tuple[float, str]:
    """How much of today's margin the forecast must retain, from its durability.

    A blanket "the terminal margin may fall to 60% of today's" is wrong in both
    directions. A business that has held a steady margin has demonstrated pricing
    power the forecast should not assume away — its floor should sit near today's
    level. One whose margin swings has shown it cannot defend the level, so a
    deeper normalisation is justified. Measured over the recent regime (the same
    three-year window the margin mean uses), via the coefficient of variation, so
    the judgement comes from the company's own record rather than a fixed guess.
    """
    recent = margins[-3:]
    if len(recent) >= 2:
        import statistics

        mean_recent = statistics.fmean(recent)
        cv = statistics.pstdev(recent) / mean_recent if mean_recent > 0 else 1.0
    else:
        cv = 1.0

    if cv <= 0.15:
        return 0.80, "steady"
    if cv <= 0.35:
        return 0.68, "moderately steady"
    return 0.55, "volatile"


_REFINE_PROMPT = """You are setting the forward assumptions for a discounted cash flow valuation of {company}.

Historical results (from its own filings):
{history}

Management commentary and outlook retrieved from the filings:
{context}

Propose three forward assumptions. Ground each in the history and the commentary above; do not invent figures.

1. year_one_revenue_growth - revenue growth for the next fiscal year (decimal, e.g. 0.35)
2. terminal_operating_margin - the operating margin this business sustains at MATURITY, ~10 years out (decimal). Reason it from economics, not just recent history: competitive intensity, pricing power, product mix, scale limits, and likely normalisation. A company at a peak margin today may well settle below it. Justify the level you choose.
3. terminal_growth_rate - perpetual growth after the forecast period (decimal, between 0.01 and 0.04; must be below long-run GDP growth)

For each, give a one-sentence rationale citing something specific from the data or commentary.

Return JSON:
{{"year_one_revenue_growth": {{"value": 0.0, "rationale": "..."}},
  "terminal_operating_margin": {{"value": 0.0, "rationale": "..."}},
  "terminal_growth_rate": {{"value": 0.0, "rationale": "..."}}}}"""


def _history_table(history: FinancialHistory) -> str:
    lines = ["FY | revenue | growth | operating margin | FCF margin"]
    growth = dict(history.growth_rates("revenue"))
    for year in history.years:
        parts = [str(year.fiscal_year)]
        parts.append(f"{year.revenue / 1e9:.1f}B" if year.revenue else "-")
        parts.append(f"{growth[year.fiscal_year] * 100:.0f}%" if year.fiscal_year in growth else "-")
        parts.append(f"{year.operating_margin * 100:.1f}%" if year.operating_margin else "-")
        fcf = year.free_cash_flow
        parts.append(f"{fcf / year.revenue * 100:.1f}%" if fcf and year.revenue else "-")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _history_fingerprint(history: FinancialHistory) -> str:
    """A stable hash of the numeric series that drive the proposal.

    Changes only when the company's reported figures change (a new fiscal year,
    a restatement), which is exactly when the assumptions *should* be recomputed.
    """
    import hashlib

    parts = [
        f"{y.fiscal_year}:{y.revenue}:{y.operating_margin}:{y.free_cash_flow}"
        for y in history.years
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _propose(history: FinancialHistory, company: Company, context: str) -> dict:
    """Ask the model for forward assumptions, cached per company + data.

    The proposal is cached keyed on the ticker and a fingerprint of the reported
    numbers — deliberately NOT on the retrieved context, which varies run to run.
    Without this the same company produced a different fair value on every load
    (temperature 0 is not fully deterministic, and the context differs each
    time), which is unacceptable for a valuation. The cache guarantees a company
    reproduces exactly until its filings change.
    """
    import json

    cache_dir = config.CACHE_DIR / "assumptions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{company.slug}_{_history_fingerprint(history)}.json"

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt entry; recompute

    payload = complete_json(
        _REFINE_PROMPT.format(
            company=f"{company.name} ({company.ticker})",
            history=_history_table(history),
            context=(context or "No management commentary was retrieved.")[:4000],
        ),
        temperature=0.0,
        max_tokens=700,
    )
    result = payload if isinstance(payload, dict) else {}

    if result:
        try:
            cache_path.write_text(json.dumps(result), encoding="utf-8")
        except OSError:
            pass  # caching is an optimisation, never required

    return result


def _model_value(proposal: dict, key: str) -> tuple[float | None, str]:
    entry = proposal.get(key)
    if isinstance(entry, dict):
        try:
            return float(entry["value"]), str(entry.get("rationale", ""))
        except (KeyError, TypeError, ValueError):
            return None, ""
    try:
        return float(entry), ""
    except (TypeError, ValueError):
        return None, ""


def derive_inputs(
    history: FinancialHistory,
    company: Company,
    ledger: AssumptionLedger,
    *,
    tax_rate: float,
    qualitative_context: str = "",
    use_model: bool = True,
) -> ForecastInputs:
    """Build the full set of DCF inputs, recording each in ``ledger``."""
    latest = history.latest
    if latest is None or not latest.revenue:
        raise ValueError("no revenue history available to forecast from")

    horizon = config.DCF_FORECAST_YEARS
    growth_history = history.growth_rates("revenue")
    recent_growth = growth_history[-1][1] if growth_history else 0.05
    mean_growth = (
        sum(g for _y, g in growth_history[-3:]) / len(growth_history[-3:])
        if growth_history
        else 0.05
    )

    margins = [y.operating_margin for y in history.years if y.operating_margin is not None]
    mean_margin = sum(margins[-3:]) / len(margins[-3:]) if margins else 0.15

    proposal = _propose(history, company, qualitative_context) if use_model else {}

    # -- terminal growth --------------------------------------------------
    # Computed first because it forms the floor for the year-one growth bound.
    proposed_terminal, terminal_rationale = _model_value(proposal, "terminal_growth_rate")
    if proposed_terminal is None:
        terminal_growth, terminal_clamped = 0.025, False
        terminal_source = SOURCE_DEFAULT
        terminal_derivation = "Standard long-run rate of 2.5%"
        terminal_rationale = (
            "Perpetual growth must stay below long-run nominal GDP growth; no "
            "company outgrows the economy forever."
        )
        raw_terminal = None
    else:
        terminal_growth, terminal_clamped = _clamp(
            proposed_terminal, config.TERMINAL_GROWTH_BOUNDS
        )
        terminal_source = SOURCE_MODEL
        terminal_derivation = (
            f"Model estimate {proposed_terminal * 100:.1f}%, bounded to "
            f"[{config.TERMINAL_GROWTH_BOUNDS[0] * 100:.0f}%, "
            f"{config.TERMINAL_GROWTH_BOUNDS[1] * 100:.0f}%]"
        )
        raw_terminal = proposed_terminal if terminal_clamped else None

    ledger.add(
        Assumption(
            key="terminal_growth",
            label="Terminal growth rate",
            value=terminal_growth,
            unit="%",
            source=terminal_source,
            derivation=terminal_derivation,
            rationale=terminal_rationale,
            bounds=config.TERMINAL_GROWTH_BOUNDS,
            clamped=terminal_clamped,
            raw_value=raw_terminal,
        )
    )

    # -- year one growth --------------------------------------------------
    # Bounded on both sides.
    #
    # Ceiling: the company's own most recent growth. Forecasting an
    # acceleration requires evidence a DCF cannot supply.
    #
    # Floor: HALF the recent pace, not the terminal rate. An earlier version
    # floored year-one growth at the ~2.5% perpetual rate, which let the model
    # propose 20% against 65% reported growth — asserting a business growing
    # strongly nearly stops within twelve months, with no evidence, and cutting
    # fair value by three quarters. A company decelerating from 65% to 33% next
    # year is already a steep, believable slowdown; below that needs a specific
    # demand-cliff argument the model is not being given. The anchor is the more
    # conservative of last year's growth and the three-year mean, so one
    # anomalous spike cannot lift the floor.
    growth_ceiling = min(max(recent_growth, 0.05), _MAX_YEAR_ONE_GROWTH)
    if recent_growth > 0:
        growth_anchor = min(recent_growth, mean_growth) if mean_growth > 0 else recent_growth
        growth_floor = min(max(terminal_growth, growth_anchor * 0.5), growth_ceiling)
    else:
        growth_floor = _MIN_YEAR_ONE_GROWTH
    growth_bounds = (growth_floor, growth_ceiling)

    proposed_growth, growth_rationale = _model_value(proposal, "year_one_revenue_growth")
    if proposed_growth is None:
        year_one_growth, was_clamped = min(recent_growth, growth_ceiling), False
        growth_source = SOURCE_HISTORICAL
        growth_derivation = f"Most recent reported growth ({recent_growth * 100:.0f}%)"
        growth_rationale = (
            "No model estimate was available, so the latest observed growth rate "
            "is carried into year one."
        )
        raw_growth = None
    else:
        year_one_growth, was_clamped = _clamp(proposed_growth, growth_bounds)
        growth_source = SOURCE_MODEL
        growth_derivation = (
            f"Model estimate {proposed_growth * 100:.0f}%, bounded to "
            f"[{growth_bounds[0] * 100:.0f}%, {growth_bounds[1] * 100:.0f}%] "
            f"from reported growth of {recent_growth * 100:.0f}%"
        )
        raw_growth = proposed_growth if was_clamped else None

    ledger.add(
        Assumption(
            key="year_one_growth",
            label="Year 1 revenue growth",
            value=year_one_growth,
            unit="%",
            source=growth_source,
            derivation=growth_derivation,
            rationale=growth_rationale,
            bounds=growth_bounds,
            clamped=was_clamped,
            raw_value=raw_growth,
        )
    )

    # -- terminal margin --------------------------------------------------
    # The mature-state margin is reasoned economically, not forced into the
    # recent range. A company at a peak margin can plausibly normalise DOWN over
    # a decade as competition and mix shift, so the band deliberately allows the
    # terminal margin below the recent regime. Only two rails apply: a floor at
    # a fraction of today's margin (an unjustified collapse is still rejected)
    # and a ceiling at the company's own demonstrated peak (a margin the
    # business has never posted is a claim a DCF cannot carry).
    current_margin_now = margins[-1] if margins else mean_margin
    peak_margin = max(margins) if margins else 0.40
    floor_fraction, durability = _margin_floor_fraction(margins)
    margin_floor = max(
        config.TERMINAL_MARGIN_ABSOLUTE_FLOOR,
        current_margin_now * floor_fraction,
    )
    margin_bounds = (margin_floor, peak_margin)
    proposed_margin, margin_rationale = _model_value(proposal, "terminal_operating_margin")

    if proposed_margin is None:
        terminal_margin, margin_clamped = mean_margin, False
        margin_source = SOURCE_HISTORICAL
        margin_derivation = f"Three-year mean operating margin ({mean_margin * 100:.1f}%)"
        margin_rationale = "Recent average margin taken as the sustainable level."
        raw_margin = None
    else:
        terminal_margin, margin_clamped = _clamp(proposed_margin, margin_bounds)
        margin_source = SOURCE_MODEL
        if terminal_margin < current_margin_now:
            shape = f"normalised down from today's {current_margin_now * 100:.1f}%"
        else:
            shape = f"held near today's {current_margin_now * 100:.1f}%"
        margin_derivation = (
            f"Model estimate {proposed_margin * 100:.1f}%, {shape}; kept within a "
            f"maturity band of [{margin_bounds[0] * 100:.1f}%, {margin_bounds[1] * 100:.1f}%] "
            f"(floor {floor_fraction * 100:.0f}% of current, set by {durability} margins; "
            f"ceiling the demonstrated peak)"
        )
        raw_margin = proposed_margin if margin_clamped else None

    ledger.add(
        Assumption(
            key="terminal_margin",
            label="Terminal operating margin",
            value=terminal_margin,
            unit="%",
            source=margin_source,
            derivation=margin_derivation,
            rationale=margin_rationale,
            bounds=margin_bounds,
            clamped=margin_clamped,
            raw_value=raw_margin,
        )
    )

    # -- paths ------------------------------------------------------------
    growth_path = decay_path(
        year_one_growth, terminal_growth, horizon, config.DCF_GROWTH_DECAY
    )
    current_margin = latest.operating_margin or mean_margin
    margin_path = fade(current_margin, terminal_margin, horizon)

    ledger.add(
        Assumption(
            key="forecast_horizon",
            label="Explicit forecast period",
            value=float(horizon),
            unit="years",
            source=SOURCE_DEFAULT,
            derivation=f"{horizon}-year explicit forecast, then a perpetuity",
            rationale=(
                f"Revenue growth decays geometrically from {year_one_growth * 100:.0f}% "
                f"toward {terminal_growth * 100:.1f}%, closing "
                f"{config.DCF_GROWTH_DECAY * 100:.0f}% of the remaining gap each year, "
                f"while margin trends linearly from {current_margin * 100:.1f}% to "
                f"{terminal_margin * 100:.1f}%. Growth is not held flat, and it is not "
                f"faded in a straight line either: a linear fade over ten years "
                f"compounds far higher than any business sustains."
            ),
        )
    )

    # -- cash conversion --------------------------------------------------
    depreciation_pct = _ratio_to_revenue(history, "depreciation_amortisation") or 0.03
    capex_pct = _ratio_to_revenue(history, "capex") or 0.04

    ledger.add(
        Assumption(
            key="depreciation_pct",
            label="D&A as % of revenue",
            value=depreciation_pct,
            unit="%",
            source=SOURCE_HISTORICAL,
            derivation="Three-year mean of depreciation and amortisation over revenue",
            rationale="Non-cash charge added back to convert operating profit to cash flow.",
        )
    )
    ledger.add(
        Assumption(
            key="capex_pct",
            label="Capex as % of revenue",
            value=capex_pct,
            unit="%",
            source=SOURCE_HISTORICAL,
            derivation="Three-year mean of capital expenditure over revenue",
            rationale="Reinvestment required to sustain and grow the asset base.",
        )
    )

    # Incremental working capital per unit of incremental revenue, measured
    # from the two most recent years where both are available.
    working_capital_pct = _DEFAULT_WORKING_CAPITAL_PCT
    wc_derivation = f"Standard assumption ({_DEFAULT_WORKING_CAPITAL_PCT * 100:.0f}% of incremental revenue)"
    wc_source = SOURCE_DEFAULT

    # Years where revenue barely moved are excluded: the ratio divides by the
    # change in revenue, so a flat year makes the denominator approach zero and
    # the result explodes. NVIDIA FY2022->FY2023 (revenue $26.9B -> $27.0B)
    # computes to -13,307%, which is an artefact of the arithmetic and not a
    # working capital observation. Taking the median of material years is
    # robust to it; taking only the most recent year merely got lucky.
    observations: list[float] = []
    for previous, current in zip(history.years, history.years[1:]):
        if (
            current.working_capital is None
            or previous.working_capital is None
            or not current.revenue
            or not previous.revenue
        ):
            continue

        revenue_change = current.revenue - previous.revenue
        if abs(revenue_change) < 0.05 * previous.revenue:
            continue

        ratio = (current.working_capital - previous.working_capital) / revenue_change
        if 0.0 <= ratio <= 0.60:
            observations.append(ratio)

    if observations:
        observations.sort()
        median = observations[len(observations) // 2]
        working_capital_pct = min(median, _MAX_WORKING_CAPITAL_PCT)
        wc_source = SOURCE_HISTORICAL
        wc_derivation = (
            f"Median change in working capital over change in revenue across "
            f"{len(observations)} year(s) with material revenue growth"
            + (
                f", capped at {_MAX_WORKING_CAPITAL_PCT * 100:.0f}%"
                if median > _MAX_WORKING_CAPITAL_PCT
                else ""
            )
        )

    ledger.add(
        Assumption(
            key="working_capital_pct",
            label="Incremental working capital",
            value=working_capital_pct,
            unit="%",
            source=wc_source,
            derivation=wc_derivation,
            rationale=(
                "Growth absorbs cash into receivables and inventory before it is "
                "collected, so incremental revenue carries a working capital cost."
            ),
        )
    )

    return ForecastInputs(
        base_revenue=latest.revenue,
        base_year=latest.fiscal_year,
        growth_path=growth_path,
        margin_path=margin_path,
        tax_rate=tax_rate,
        depreciation_pct=depreciation_pct,
        capex_pct=capex_pct,
        working_capital_pct=working_capital_pct,
        terminal_growth=terminal_growth,
    )
