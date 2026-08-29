"""
Is this valuation economically possible, or is the model mis-specified?

A reverse DCF answers "what would the market have to believe?". It is a good
question, and it has a failure mode: when the forward model is wrong, the
solver does not fail — it returns whatever absurd input would close the gap,
and that number then gets published as though it described investor belief.

The case this module exists for: a Bikaji report priced the stock at INR 618.80
against a DCF fair value of INR 77.37, and reported that the market must expect
a **93.8% mature operating margin**. No snacks company earns 93.8%. The correct
reading was never "investors are irrational" — it was "our forward model does
not describe this business", and the causes were identifiable: a base year two
years stale, a discount rate near 14%, growth faded to roughly half a plausible
path, margins held flat while the company's own filings described a capacity
build-out at 46% utilisation, and growth capex treated as if it were permanent
maintenance capex.

So the tests below are deliberately **company-relative rather than absolute**.
"Above 40%" is a rule that needs re-tuning for every sector. "Six times the best
margin this company has ever reported" is a statement that means the same thing
for a snacks maker, a software company and a steel mill, and it survives being
pointed at a business nobody anticipated.

The output is findings, in the same shape the report's QA gate already
consumes, so an implausible valuation is withheld by the mechanism that already
withholds internally contradictory ones — rather than by a new one bolted on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# An implied margin far above a company's demonstrated best is REPORTED, not
# blocked. It used to be treated as proof the model was mis-specified, and that
# reading is wrong: a DCF on free cash flow cannot reach the prices quality
# companies trade at, so an extreme implied figure is usually the market's
# requirement being extreme rather than our arithmetic being broken. Saying "the
# price needs a margin this company has never approached" is the most useful
# sentence the tool produces — see valuation/expectations.py, which now carries
# the rating — and blocking on it suppressed exactly that finding.
#
# What is still blocked is the genuinely impossible: an implied margin above
# 100% of revenue is not a demanding expectation, it is an arithmetic
# impossibility, and no market prices one in.
# Two thresholds, because "demanding" and "broken" are different claims.
#
# Past 2.5x the company's best, the price is asking a lot — worth reporting, and
# the most useful sentence the tool writes. Past 5x, the solver is not describing
# a market expectation any more: it is describing a model that cannot reach the
# price on any input, which is what the original Bikaji failure looked like when
# it demanded a 93.8% margin from a business whose best was 14.2%.
_MARGIN_MULTIPLE_OF_BEST = 2.5      # demanding: reported
_MARGIN_MULTIPLE_BROKEN = 5.0       # broken: blocked
_MARGIN_BROKEN_FLOOR = 0.60         # for thin- or negative-margin histories

# A floor for companies whose history is loss-making or barely profitable,
# where a multiple of "best ever" is meaningless or negative.
_MARGIN_ABSOLUTE_CEILING = 0.45

# Perpetual growth above the economy's nominal growth means the company
# eventually becomes the economy. Country-specific because nominal GDP growth
# is: India runs far higher than the US in nominal terms.
_NOMINAL_GDP_GROWTH = {"IN": 0.105, "US": 0.045, "GB": 0.04, "JP": 0.025}
_DEFAULT_NOMINAL_GDP = 0.055

# A wide gap to the market is REPORTED, never blocked. Blocking on it asserted
# the market is right, which is the opposite of what this engine is for — and
# empirically the assertion is false: removing every conservative assumption
# still values Apple 55% below its price, because 44x free cash flow is simply
# not reachable by discounting cash flows at any defensible rate. A tool that
# refuses to publish whenever it disagrees with the market cannot do research.
#
# The gap is still surfaced, and the reverse DCF explains it in terms a reader
# can check: not "we are lower" but "here is what the price requires".
_GAP_WARN = 0.45


@dataclass
class Implausibility:
    """One reason to distrust the valuation rather than the market."""

    check: str
    severity: str          # "CRITICAL" blocks, "HIGH" blocks, "MEDIUM" warns
    message: str
    likely_causes: list[str]


def _best_historical_margin(history) -> float | None:
    margins = [
        value for _, value in (history.series("operating_margin") if history else [])
        if value is not None
    ]
    return max(margins) if margins else None


def _diagnose(valuation, history) -> list[str]:
    """The specific, checkable reasons a forward model lands too low.

    Returned to the caller rather than logged, because the point is to tell
    whoever reads the blocked report *where to look* — a bare "implausible" is
    not actionable.
    """
    causes: list[str] = []

    recency = getattr(history, "recency", None)
    if recency is not None and not getattr(recency, "is_current", True):
        causes.append(
            f"the base year is {getattr(recency, 'months_old', 0):.0f} months old "
            f"(FY{getattr(recency, 'latest_fiscal_year', '?')}), so growth, margin "
            f"and capital intensity all describe an older company"
        )

    dcf = getattr(valuation, "dcf", None)
    wacc = getattr(dcf, "wacc", None)
    if wacc and wacc > 0.13:
        causes.append(
            f"the discount rate is {wacc * 100:.1f}%, high enough to suppress "
            f"terminal value on its own"
        )

    # Flat margins alongside a rising asset base is the specific contradiction
    # that produced the Bikaji failure: the filings described capacity being
    # built ahead of revenue, and the model held margins constant anyway.
    if dcf is not None and getattr(dcf, "forecast", None):
        margins = [getattr(f, "operating_margin", None) for f in dcf.forecast]
        margins = [m for m in margins if m is not None]
        if margins and (max(margins) - min(margins)) < 0.005:
            causes.append(
                "the operating margin is held flat across the whole forecast, so "
                "no operating leverage is modelled even if the company is mid-build-out"
            )

    if history is not None:
        capex = [v for _, v in history.series("capex") if v is not None]
        da = [v for _, v in history.series("depreciation_amortisation") if v is not None]
        if capex and da and abs(capex[-1]) > 1.4 * abs(da[-1]):
            causes.append(
                "capital spending is well above depreciation, so treating all of it "
                "as recurring maintenance capex permanently understates free cash flow"
            )

    if not causes:
        causes.append(
            "no single input stands out — re-check the growth path, the margin "
            "path and the reinvestment rate against the company's own history"
        )
    return causes


def assess(valuation, history=None, *, country: str | None = None) -> list[Implausibility]:
    """Check a finished valuation for economic impossibility.

    Returns an empty list when nothing is wrong, so the caller can treat a
    clean valuation as the normal path.
    """
    findings: list[Implausibility] = []

    price = getattr(valuation, "share_price", None)
    fair_value = getattr(valuation, "fair_value", None) or getattr(
        valuation, "dcf_fair_value", None
    )

    # -- 1. an implied margin the business could not produce ---------------
    priced_in = getattr(valuation, "priced_in", None)
    implied_margin = None
    if priced_in is not None:
        for row in getattr(priced_in, "rows", []) or []:
            if getattr(row, "key", "") == "operating_margin":
                implied_margin = getattr(row, "implied_value", None)

    if implied_margin is not None:
        best = _best_historical_margin(history)
        ceiling = _MARGIN_ABSOLUTE_CEILING
        basis = f"an absolute ceiling of {ceiling * 100:.0f}%"
        if best and best > 0:
            ceiling = max(best * _MARGIN_MULTIPLE_OF_BEST, _MARGIN_ABSOLUTE_CEILING)
            basis = (
                f"{_MARGIN_MULTIPLE_OF_BEST:g}x this company's best reported "
                f"operating margin of {best * 100:.1f}%"
            )
        # The multiple governs whenever there is a positive margin to multiply.
        # Taking max() with the floor was wrong: for a 5%-margin business it
        # raised the bar from 25% to 60%, so a solver demanding a 60% margin
        # from a business that has never cleared 5% passed unremarked — the
        # thinner the margins, the more the check let through. The floor is for
        # histories where the multiple means nothing: no margin, or a negative one.
        broken_ceiling = (
            best * _MARGIN_MULTIPLE_BROKEN
            if best and best > 0
            else _MARGIN_BROKEN_FLOOR
        )
        if implied_margin > broken_ceiling:
            findings.append(Implausibility(
                check="valuation_plausibility",
                severity="CRITICAL",
                message=(
                    f"The reverse DCF implies a mature operating margin of "
                    f"{implied_margin * 100:.1f}%, more than "
                    f"{_MARGIN_MULTIPLE_BROKEN:g}x {basis}. At that distance the "
                    f"solver is no longer describing a market expectation but a "
                    f"model that cannot reach the price on any input — it is "
                    f"mis-specified, and must not be published as a belief the "
                    f"market holds."
                ),
                likely_causes=_diagnose(valuation, history),
            ))
        elif implied_margin > ceiling:
            findings.append(Implausibility(
                check="valuation_plausibility",
                severity="MEDIUM",
                message=(
                    f"The price requires a mature operating margin of "
                    f"{implied_margin * 100:.1f}%, above {basis}. Reported as a "
                    f"finding about how demanding the price is, not as an error: a "
                    f"discounted cash flow cannot reach the multiples quality "
                    f"companies trade at, so an extreme requirement usually means "
                    f"the price is extreme."
                ),
                likely_causes=[],
            ))

    # -- 2. perpetual growth above the economy -----------------------------
    implied_growth = None
    if priced_in is not None:
        for row in getattr(priced_in, "rows", []) or []:
            if getattr(row, "key", "") == "terminal_growth":
                implied_growth = getattr(row, "implied_value", None)

    gdp = _NOMINAL_GDP_GROWTH.get((country or "").upper(), _DEFAULT_NOMINAL_GDP)
    if implied_growth is not None and implied_growth > gdp:
        findings.append(Implausibility(
            check="valuation_plausibility",
            severity="MEDIUM",
            message=(
                f"The implied perpetual growth rate of {implied_growth * 100:.1f}% "
                f"exceeds nominal GDP growth of about {gdp * 100:.1f}%, which would "
                f"have the company eventually become the whole economy. Report it as "
                f"a bound, not as a rate investors expect."
            ),
            likely_causes=_diagnose(valuation, history),
        ))

    # -- 3. a gap too large to be a disagreement ---------------------------
    if price and fair_value and price > 0 and fair_value > 0:
        gap = abs(fair_value / price - 1)
        if gap > _GAP_WARN:
            findings.append(Implausibility(
                check="valuation_plausibility",
                severity="MEDIUM",
                message=(
                    f"Fair value differs from the market price by {gap * 100:.0f}%. "
                    f"That is a strong call and may well be right — state explicitly "
                    f"what the market is getting wrong, and check the inputs below."
                ),
                likely_causes=_diagnose(valuation, history),
            ))

    if findings:
        log.warning(
            "valuation plausibility: %d finding(s), most severe %s",
            len(findings), findings[0].severity,
        )
    return findings
