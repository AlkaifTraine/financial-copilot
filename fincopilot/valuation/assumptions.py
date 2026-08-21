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
import statistics
from dataclasses import dataclass, field

from .. import config
from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company
from .dcf import decay_path, fade
from .models import (
    SOURCE_ANALYST,
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
    growth_decay: float = 0.70


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


def _growth_decay(year_one_growth: float, steady: bool = False) -> float:
    """Geometric decay for the growth path, gentler for lower, sustainable growth.

    A single decay is wrong across the growth spectrum. Extreme growth cannot
    persist and must fade fast (a 60% starter faded slowly forecasts a company
    into a tenth of GDP); but a durable mid-teens compounder faded at that same
    rate reaches a mature CAGR within a few years, which no such business does —
    and that is precisely what made the engine too bearish on quality growers.
    So the decay scales with the starting rate: the faster the start, the faster
    it must fall. A company that has grown *steadily* (positive every year, low
    dispersion) has demonstrated persistence and fades a little slower still.
    """
    if year_one_growth >= 0.45:
        base = 0.68
    elif year_one_growth >= 0.30:
        base = 0.74
    elif year_one_growth >= 0.18:
        base = 0.80
    else:
        base = 0.86
    if steady:
        base = min(base + 0.05, 0.90)
    return base


def _growth_is_steady(growth_history: list[tuple[int, float]]) -> bool:
    """Whether recent revenue growth has been consistently positive and stable.

    A steady compounder earns slower decay; a company whose growth swings (a
    cyclical, or a hyper-grower spiking then falling) does not. Measured over the
    recent regime by the coefficient of variation, and gated on every recent year
    being positive so a bounce-off-a-decline does not read as steadiness.
    """
    recent = [g for _y, g in growth_history[-4:]]
    if len(recent) < 3 or any(g <= 0 for g in recent):
        return False
    mean = statistics.fmean(recent)
    return mean > 0 and statistics.pstdev(recent) / mean < 0.5


_REFINE_PROMPT = """You are setting the forward assumptions for a discounted cash flow valuation of {company}.

Historical results (from its own filings):
{history}

Management commentary and outlook retrieved from the filings:
{context}

Propose three forward assumptions. Ground each in the history and the commentary above; do not invent figures.

1. year_one_revenue_growth - revenue growth for the next fiscal year (decimal, e.g. 0.35)
2. terminal_operating_margin - the operating margin this business sustains at MATURITY, ~10 years out (decimal). Derive it ECONOMICALLY, not from recent history, weighing: normalized competitive economics, product/segment mix, pricing power, scale economies, competitive intensity and likely new entrants, industry structure, and long-run capital intensity — checked against, but NOT bound to, historical evidence. A company at a peak margin today may well settle below it; a scaling one may rise. If your estimate differs materially from the current/historical margin, that is allowed — say plainly WHY in the rationale. Do not anchor to the historical range.
3. terminal_growth_rate - perpetual growth after the forecast period (decimal, between 0.01 and 0.04; must be below long-run GDP growth)

For each, give a one-sentence rationale AND expose its PROVENANCE — the evidence chain that leads to your number. Each provenance field is a short phrase; use "n/a" only when the evidence genuinely does not exist (do not invent it):
- historical: what the company's own reported history shows on this metric
- guidance: current management guidance on it, if any (WITH its period). Never annualize a single quarter's guidance as if it were full-year guidance. If guidance is one quarter (e.g. a ~$78B Q1 number), compare it as a QUARTERLY figure (quarter-over-quarter or quarter YoY); if you must express a run-rate, label it a "run-rate" (4x the quarter) explicitly and never call it annual guidance.
- industry: relevant industry / end-market evidence
- competitive: competitive dynamics bearing on it
- management: specific management commentary you are relying on
Your value is the model output that follows from this chain.

For terminal_operating_margin ONLY, ALSO provide:
- margin_bridge: the quantitative walk from today's margin to your terminal figure, as an ordered list of {{"component": "...", "value": <number in PERCENTAGE POINTS>}}. The FIRST item is the starting level (the current or normalized operating margin, e.g. 60.0). Each later item is a SIGNED adjustment in percentage points — include only those supported by evidence, and size each: e.g. +scale economies, +software/ecosystem mix, then -competitive pressure, -pricing normalization, -R&D intensity, -mature-semiconductor economics. The starting level plus all adjustments MUST equal your terminal_operating_margin (as a %). This makes the terminal margin reproducible rather than a bare number.
- margin_confidence: exactly one of High / Medium / Low — how well-supported the terminal margin is (Low when it rests mostly on judgement, High when history/peers/guidance align).

Return JSON:
{{"year_one_revenue_growth": {{"value": 0.0, "rationale": "...", "provenance": {{"historical": "...", "guidance": "...", "industry": "...", "competitive": "...", "management": "..."}}}},
  "terminal_operating_margin": {{"value": 0.0, "rationale": "...", "provenance": {{"historical": "...", "guidance": "...", "industry": "...", "competitive": "...", "management": "..."}}, "margin_bridge": [{{"component": "Current operating margin", "value": 60.0}}, {{"component": "-Pricing normalization", "value": -8.0}}], "margin_confidence": "Medium"}},
  "terminal_growth_rate": {{"value": 0.0, "rationale": "...", "provenance": {{"historical": "...", "guidance": "...", "industry": "...", "competitive": "...", "management": "..."}}}}}}"""


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
        model=config.WRITER_MODEL,
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


_PROVENANCE_KEYS = ("historical", "guidance", "industry", "competitive", "management")


def _model_provenance(proposal: dict, key: str) -> dict:
    """The evidence chain the model gave for an assumption, cleaned of empties."""
    entry = proposal.get(key)
    if not isinstance(entry, dict):
        return {}
    raw = entry.get("provenance")
    if not isinstance(raw, dict):
        return {}
    chain = {}
    for field_name in _PROVENANCE_KEYS:
        value = str(raw.get(field_name) or "").strip()
        if value and value.lower() not in ("n/a", "none", "-"):
            chain[field_name] = value
    return chain


def _model_margin_bridge(proposal: dict) -> tuple[list, str]:
    """The terminal-margin bridge (start level + signed pp adjustments) and confidence."""
    entry = proposal.get("terminal_operating_margin")
    if not isinstance(entry, dict):
        return [], ""
    rows = entry.get("margin_bridge")
    bridge: list = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("component") is not None:
                try:
                    bridge.append({"component": str(row["component"]).strip(),
                                   "value": float(row.get("value"))})
                except (TypeError, ValueError):
                    continue
    confidence = str(entry.get("margin_confidence") or "").strip().title()
    if confidence not in ("High", "Medium", "Low"):
        confidence = ""
    return bridge, confidence


# Analyst-override keys -> (ledger key, hard sanity rails). Overrides win over the
# model AND the agent, and are bounded only by these generous physical rails — not the
# model's anti-hallucination bounds — because a human analyst is authoritative.
_ANALYST_OVERRIDES = {
    "year_one_revenue_growth": ("year_one_growth", (-0.90, 5.00)),
    "terminal_operating_margin": ("terminal_margin", (0.0, 0.95)),
    "terminal_growth_rate": ("terminal_growth", (0.0, 0.06)),
}


def _apply_overrides(
    overrides: dict | None,
    ledger: AssumptionLedger,
    year_one_growth: float,
    terminal_growth: float,
    terminal_margin: float,
) -> tuple[float, float, float, list[str]]:
    """Replace any analyst-pinned driver, re-tag its ledger entry, return the values.

    Called AFTER the three drivers are added to the ledger and BEFORE the forecast
    paths are built, so the paths (and the DCF) compute from the analyst's number.
    """
    values = {
        "year_one_growth": year_one_growth,
        "terminal_growth": terminal_growth,
        "terminal_margin": terminal_margin,
    }
    applied: list[str] = []
    for override_key, (ledger_key, (low, high)) in _ANALYST_OVERRIDES.items():
        raw = (overrides or {}).get(override_key)
        if raw is None:
            continue
        try:
            requested = float(raw)
        except (TypeError, ValueError):
            continue
        value = min(max(requested, low), high)
        values[ledger_key] = value
        applied.append(ledger_key)
        assumption = ledger.get(ledger_key)
        if assumption is not None:
            was_clamped = value != requested
            assumption.value = value
            assumption.source = SOURCE_ANALYST
            assumption.bounds = (low, high)
            assumption.clamped = was_clamped
            assumption.raw_value = requested if was_clamped else None
            assumption.provenance = {}
            assumption.bridge = []
            assumption.derivation = (
                "Analyst-set input" + (" (clamped to a hard sanity rail)" if was_clamped else "")
            )
            assumption.rationale = (
                "Set by the analyst; the model computes from this input, overriding its own estimate."
            )
    return values["year_one_growth"], values["terminal_growth"], values["terminal_margin"], applied


def derive_inputs(
    history: FinancialHistory,
    company: Company,
    ledger: AssumptionLedger,
    *,
    tax_rate: float,
    qualitative_context: str = "",
    use_model: bool = True,
    proposal_override: dict | None = None,
    overrides: dict | None = None,
) -> ForecastInputs:
    """Build the full set of DCF inputs, recording each in ``ledger``.

    ``proposal_override`` supplies a proposal directly (same shape as
    :func:`_propose` returns), bypassing the model call. The agentic critique
    pass uses it to feed back a revised assumption set, which is then clamped to
    the same data-derived bounds as any proposal — the revision cannot escape
    the guardrails.
    """
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

    # Normalised growth: the anchor for the forecast. For a mature company a single
    # soft (or strong) year is noise, so blend the latest with the three-year mean
    # — otherwise a one-off weak year (Coca-Cola printing 2% against a ~5% trend)
    # gets extrapolated forever and craters the value. For a faster grower the
    # recent deceleration is signal, not noise, so the latest number is kept. The
    # ~15% line is where year-to-year noise stops dominating the underlying trend.
    if 0 < recent_growth < 0.15 and mean_growth > recent_growth:
        normalized_growth = (recent_growth + mean_growth) / 2
    else:
        normalized_growth = recent_growth
    growth_steady = _growth_is_steady(growth_history)

    margins = [y.operating_margin for y in history.years if y.operating_margin is not None]
    mean_margin = sum(margins[-3:]) / len(margins[-3:]) if margins else 0.15

    margins = [y.operating_margin for y in history.years if y.operating_margin is not None]
    mean_margin = sum(margins[-3:]) / len(margins[-3:]) if margins else 0.15

    if proposal_override is not None:
        proposal = proposal_override
    elif use_model:
        proposal = _propose(history, company, qualitative_context)
    else:
        proposal = {}

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
            provenance=_model_provenance(proposal, "terminal_growth_rate"),
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
    growth_ceiling = min(max(normalized_growth, 0.05), _MAX_YEAR_ONE_GROWTH)
    if normalized_growth > 0:
        growth_floor = min(max(terminal_growth, normalized_growth * 0.5), growth_ceiling)
    else:
        growth_floor = _MIN_YEAR_ONE_GROWTH
    growth_bounds = (growth_floor, growth_ceiling)

    proposed_growth, growth_rationale = _model_value(proposal, "year_one_revenue_growth")
    if proposed_growth is None:
        year_one_growth, was_clamped = min(normalized_growth, growth_ceiling), False
        growth_source = SOURCE_HISTORICAL
        growth_derivation = f"Normalised recent growth ({normalized_growth * 100:.0f}%)"
        growth_rationale = (
            "No model estimate was available, so the normalised recent growth "
            "rate (latest blended with the three-year mean) is carried into year one."
        )
        raw_growth = None
    else:
        year_one_growth, was_clamped = _clamp(proposed_growth, growth_bounds)
        growth_source = SOURCE_MODEL
        growth_derivation = (
            f"Model estimate {proposed_growth * 100:.0f}%, bounded to "
            f"[{growth_bounds[0] * 100:.0f}%, {growth_bounds[1] * 100:.0f}%] "
            f"around normalised growth of {normalized_growth * 100:.0f}%"
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
            provenance=_model_provenance(proposal, "year_one_revenue_growth"),
        )
    )

    # -- terminal margin --------------------------------------------------
    # The mature-state margin is an ECONOMIC judgement — reasoned from competitive
    # structure, mix, pricing power, scale and long-run capital intensity — and is
    # NOT clamped back to the historical range. An earlier version floored it at a
    # fraction of the current margin, which forced an economically derived 40% up
    # to 48% on NVDA; that mechanical constraint is exactly what we set out to
    # remove. Only physical sanity rails apply now (a margin below zero or above
    # ~90% of revenue is not a business a DCF can carry). Where the estimate
    # departs materially from today's margin, the derivation EXPLAINS the
    # departure rather than overriding it.
    current_margin_now = margins[-1] if margins else mean_margin
    peak_margin = max(margins) if margins else 0.40
    margin_bounds = (
        config.TERMINAL_MARGIN_ABSOLUTE_FLOOR,
        config.TERMINAL_MARGIN_SANITY_CEILING,
    )
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
        deviation = terminal_margin - current_margin_now
        if abs(deviation) >= config.TERMINAL_MARGIN_MATERIAL_DEVIATION:
            direction = "below" if deviation < 0 else "above"
            above_peak = (
                " — above any level the company has posted"
                if terminal_margin > peak_margin + 1e-9 else ""
            )
            shape = (
                f"a mature margin {abs(deviation) * 100:.0f}pp {direction} today's "
                f"{current_margin_now * 100:.0f}%{above_peak}, an economic normalisation "
                f"the rationale explains"
            )
        else:
            shape = f"a mature margin near today's {current_margin_now * 100:.0f}%"
        margin_derivation = (
            f"Economically derived terminal margin of {terminal_margin * 100:.1f}% — {shape}. "
            f"Reasoned from competitive economics, mix, pricing power, scale and industry "
            f"structure; not clamped to the historical range (physical sanity rails only, "
            f"[{margin_bounds[0] * 100:.0f}%, {margin_bounds[1] * 100:.0f}%])."
        )
        raw_margin = proposed_margin if margin_clamped else None

    margin_bridge, margin_confidence = _model_margin_bridge(proposal)
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
            provenance=_model_provenance(proposal, "terminal_operating_margin"),
            bridge=margin_bridge,
            confidence=margin_confidence,
        )
    )

    # -- analyst overrides ------------------------------------------------
    # A human analyst can pin any of the three value drivers. Applied here — after the
    # model/agent set them, before the paths are built — so the forecast and the DCF
    # compute from the analyst's number, tagged as analyst-set.
    year_one_growth, terminal_growth, terminal_margin, _applied_overrides = _apply_overrides(
        overrides, ledger, year_one_growth, terminal_growth, terminal_margin
    )

    # -- paths ------------------------------------------------------------
    # The decay is adaptive: extreme growth cannot persist and fades fast, while a
    # durable mid-teens compounder should keep growing for years. A single decay
    # applied to both is what made the model systematically too bearish on quality
    # growers — a 15% grower faded at a hyper-growth rate reaches a mature CAGR
    # within a few years, which no such business does.
    growth_decay = _growth_decay(year_one_growth, steady=growth_steady)
    growth_path = decay_path(year_one_growth, terminal_growth, horizon, growth_decay)
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
                f"{growth_decay * 100:.0f}% of the remaining gap each year (the fade is "
                f"slower for lower, more sustainable starting growth), while margin trends "
                f"linearly from {current_margin * 100:.1f}% to {terminal_margin * 100:.1f}%. "
                f"Growth is not held flat, and it is not faded in a straight line either: a "
                f"linear fade over ten years compounds far higher than any business sustains."
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
        growth_decay=growth_decay,
    )
