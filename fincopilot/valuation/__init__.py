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
from .reverse import build_priced_in, implied_growth
from .agent import critique_assumptions
from .models import (
    SOURCE_ANALYST,
    Assumption,
    AssumptionLedger,
    BlendedValuation,
    CompsResult,
    DCFResult,
    PricedInComparison,
    PricedInRow,
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
    "PricedInComparison",
    "PricedInRow",
    "run_dcf",
]


def _dcf_from(inputs, wacc, net_debt, shares, currency):
    """Run the DCF from a :class:`ForecastInputs`. Shared by the probe and final builds."""
    return run_dcf(
        base_revenue=inputs.base_revenue,
        base_year=inputs.base_year,
        growth_path=inputs.growth_path,
        margin_path=inputs.margin_path,
        tax_rate=inputs.tax_rate,
        depreciation_pct=inputs.depreciation_pct,
        capex_pct=inputs.capex_pct,
        growth_capex_per_revenue=inputs.growth_capex_per_revenue,
        working_capital_pct=inputs.working_capital_pct,
        wacc=wacc,
        terminal_growth=inputs.terminal_growth,
        net_debt=net_debt,
        shares_outstanding=shares,
        currency=currency,
    )


def value_company(
    company: Company,
    history: FinancialHistory,
    *,
    qualitative_context: str = "",
    use_model: bool = True,
    overrides: dict | None = None,
) -> Valuation:
    """Produce a complete valuation for ``company``.

    Args:
        qualitative_context: management commentary retrieved from the filings,
            used to justify forward assumptions. The valuation runs without it.
        use_model: set False to derive every assumption from history alone —
            used by the tests, and as the fallback when the model is
            unavailable.
        overrides: analyst-pinned value drivers that override the model and the
            agent — any of ``year_one_revenue_growth``, ``terminal_operating_margin``,
            ``terminal_growth_rate`` (decimals) and ``wacc``. The model computes from
            these; each is recorded as analyst-set. This is the institutional pattern:
            the analyst owns the assumptions, the model enforces consistency.
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
    wacc_override = (overrides or {}).get("wacc")
    if wacc_override is not None:
        try:
            wacc = min(max(float(wacc_override), 0.03), 0.30)   # hard sanity rails
            wacc_assumption = ledger.get("wacc")
            if wacc_assumption is not None:
                wacc_assumption.value = wacc
                wacc_assumption.source = SOURCE_ANALYST
                wacc_assumption.derivation = "Analyst-set discount rate"
                wacc_assumption.rationale = (
                    "Set by the analyst; overrides the computed WACC."
                )
        except (TypeError, ValueError):
            pass
    tax_rate = ledger.value("tax_rate", 0.21)

    # -- share count ------------------------------------------------------
    # Diluted shares from the filings are preferred over the market data feed:
    # they are audited, and they include the dilution from unvested equity that
    # a raw outstanding-share count omits. Computed before the assumptions so the
    # critique probe below can value them.
    latest = history.latest
    shares = latest.diluted_shares or history.shares_outstanding or 0.0
    if shares <= 0:
        valuation.warnings.append("Share count unavailable; per-share value cannot be computed.")
        return valuation

    net_debt = latest.net_debt if latest.net_debt is not None else 0.0

    # -- forward assumptions, with an agentic critique pass ---------------
    # The model first proposes a full assumption set. A second "senior reviewer"
    # pass then sees the fair value those assumptions produce — and how it sits
    # against the market price and the analyst consensus — and may revise the
    # single least defensible lever, reasoning from the fundamentals and never to
    # match the price. Any revision is re-clamped to the same data-derived bounds,
    # so it stays inside the guardrails. The probe build is thrown away; only the
    # final assumption set is recorded in the ledger.
    proposal_override = None
    probe_ledger = None
    if use_model:
        try:
            probe_ledger = AssumptionLedger()
            probe_inputs = derive_inputs(
                history, company, probe_ledger, tax_rate=tax_rate,
                qualitative_context=qualitative_context, use_model=True, overrides=overrides,
            )
            probe_dcf = _dcf_from(probe_inputs, wacc, net_debt, shares, history.currency)
            proposal_override = critique_assumptions(
                company, history, probe_inputs, probe_dcf, wacc,
                qualitative_context=qualitative_context,
            )
        except ValueError as exc:
            log.info("assumption critique skipped: %s", exc)

    try:
        inputs = derive_inputs(
            history,
            company,
            ledger,
            tax_rate=tax_rate,
            qualitative_context=qualitative_context,
            use_model=use_model,
            proposal_override=proposal_override,
            overrides=overrides,
        )
    except ValueError as exc:
        valuation.warnings.append(str(exc))
        return valuation

    if proposal_override:
        valuation.warnings.append(
            "A second-pass calibration review revised one or more forward assumptions "
            "(see the assumption rationales below); the fair value reflects the revised set."
        )
        # The agent revised the VALUES; its override carries no evidence chain. Keep
        # the provenance the base proposal established (the historical/guidance/
        # industry/competitive/management evidence is about the company, not the
        # specific number), so model estimates still expose where they came from.
        if probe_ledger is not None:
            for key in ("year_one_growth", "terminal_margin", "terminal_growth"):
                final_a = ledger.get(key)
                base_a = probe_ledger.get(key)
                if final_a is not None and base_a is not None and not final_a.provenance:
                    final_a.provenance = base_a.provenance
            # The margin bridge is tied to the terminal margin; preserve it from the
            # probe when the override dropped it.
            margin = ledger.get("terminal_margin")
            base_margin = probe_ledger.get("terminal_margin")
            if margin is not None and base_margin is not None and not margin.bridge and base_margin.bridge:
                margin.bridge = base_margin.bridge
                margin.confidence = margin.confidence or base_margin.confidence

    # A bridge must sum to the terminal margin actually used. When the agent moved the
    # margin (or the model's arithmetic drifted), reconcile with a residual row so the
    # walk always lands on the reported figure.
    margin = ledger.get("terminal_margin")
    if margin is not None and margin.bridge:
        target_pp = margin.value * 100.0
        walked = sum(row.get("value", 0.0) for row in margin.bridge)
        if abs(walked - target_pp) > 0.5:
            margin.bridge = [*margin.bridge,
                             {"component": "Net other adjustments", "value": round(target_pp - walked, 1)}]

    # Record which drivers a human analyst pinned, so the report can flag the
    # valuation as analyst-adjusted (the model computed from the analyst's inputs).
    valuation.analyst_overrides = [
        item.key for item in ledger.items if item.source == SOURCE_ANALYST
    ]
    if valuation.analyst_overrides:
        valuation.warnings.append(
            "Analyst-adjusted valuation: the drivers "
            + ", ".join(valuation.analyst_overrides)
            + " were set by the analyst; the model computed the fair value from those inputs."
        )

    # -- the model --------------------------------------------------------
    try:
        valuation.dcf = _dcf_from(inputs, wacc, net_debt, shares, history.currency)
    except ValueError as exc:
        valuation.warnings.append(f"The discounted cash flow model could not be built: {exc}")
        return valuation

    # A non-positive equity value is a degenerate DCF, not a −100%+ SELL: at these
    # margins and growth the modelled cash flows do not cover the cost of capital
    # and net debt. Say so plainly and derive no rating from it (the upside
    # property already returns None), rather than print a nonsensical percentage.
    if valuation.dcf.fair_value_per_share <= 0:
        valuation.warnings.append(
            f"The discounted cash flow produces a non-positive equity value "
            f"({history.currency} {valuation.dcf.fair_value_per_share:,.2f} per share) under "
            f"these assumptions: the modelled operating margins and growth do not cover the "
            f"cost of capital and net debt over the forecast. The DCF is not meaningful for "
            f"this company in its current state, so no intrinsic rating is derived from it; "
            f"read the comparables and the scenario range instead."
        )

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

    # -- competition -> fair value sensitivity ----------------------------
    # Translate a competitive share loss into revenue and fair-value terms, so the
    # competition risk is a number a reader can weigh, not just a narrative.
    from .sensitivity import build_competition_sensitivity
    valuation.competition_sensitivity = build_competition_sensitivity(
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
        base_fair_value=valuation.dcf.fair_value_per_share,
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
        terminal_margin_value = ledger.value("terminal_margin", inputs.margin_path[-1])
        valuation.market_implied_growth = implied_growth(
            base_revenue=inputs.base_revenue,
            base_year=inputs.base_year,
            horizon=len(inputs.growth_path),
            terminal_margin=terminal_margin_value,
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
            growth_decay=inputs.growth_decay,
        )

        # The full comparison: every driver's market-implied level beside our
        # base case, each solved by inverting the same DCF one lever at a time.
        valuation.priced_in = build_priced_in(
            inputs=inputs,
            wacc=wacc,
            net_debt=net_debt,
            shares_outstanding=shares,
            terminal_margin=terminal_margin_value,
            base_dcf=valuation.dcf,
            share_price=history.share_price,
            currency=history.currency,
            implied_year_one_growth=valuation.market_implied_growth,
        )

        # Judge those requirements against what the company has actually
        # delivered. This carries the rating: a DCF cannot reach the prices
        # quality companies trade at, so rating off the gap returns SELL on
        # nearly everything, while "the price needs a margin the company has
        # never reported" is both checkable and specific to the business.
        from .expectations import assess as assess_expectations

        valuation.expectations = assess_expectations(valuation, history)

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
        comps_value=valuation.comps.implied_value_per_share if valuation.comps else None,
        scenario_value=valuation.scenarios.expected_value if valuation.scenarios else None,
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
                    f"{valuation.dcf.fair_value_per_share:,.2f}). The consensus is an external "
                    f"cross-check, not blended into our target — the gap is the contrarian call."
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

    # -- miscalibration flag ----------------------------------------------
    # Our view is intrinsic and may diverge from the market. But a value far from
    # BOTH the market price AND the independent analyst consensus, in the same
    # direction, means we are the outlier — the assumption agent has already had a
    # pass, so what remains is a genuine difference of view that the reader should
    # weigh as a specific, assumption-driven contrarian call rather than a fact.
    consensus_target = history.analyst_target_median or history.analyst_target_mean
    if valuation.fair_value and history.share_price and consensus_target:
        vs_price = valuation.fair_value / history.share_price - 1
        vs_consensus = valuation.fair_value / consensus_target - 1
        both_far = min(abs(vs_price), abs(vs_consensus)) > config.MISCALIBRATION_FLAG
        same_side = (vs_price < 0) == (vs_consensus < 0)
        if both_far and same_side:
            direction = "below" if vs_price < 0 else "above"
            valuation.warnings.append(
                f"Our intrinsic value ({history.currency} {valuation.fair_value:,.2f}) sits "
                f"{abs(vs_price) * 100:.0f}% {direction} the market price and "
                f"{abs(vs_consensus) * 100:.0f}% {direction} the analyst consensus "
                f"({history.currency} {consensus_target:,.2f}). On this name we are the outlier: "
                f"read it as a specific, assumption-driven contrarian view, and check the "
                f"assumptions and the 'what is priced in' table before relying on it."
            )

    for assumption in ledger.clamped:
        valuation.warnings.append(
            f"{assumption.label} was constrained to {assumption.display} "
            f"(estimate was {assumption.raw_display})."
            if assumption.raw_value is not None
            else f"{assumption.label} was constrained to {assumption.display}."
        )

    return valuation
