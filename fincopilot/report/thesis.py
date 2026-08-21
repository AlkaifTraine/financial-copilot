"""
The investment thesis: the analytical core of the report.

Everything else in the report is evidence; this is the argument. It answers the
questions a portfolio manager actually asks — what does the market believe, what
do we believe, where do those differ, and what would prove us wrong — and it
does so grounded in the model's own numbers (scenarios, the blended target, the
reverse-DCF), never in hardcoded prose.

The single most important idea here is the separation of **business quality**
from **stock attractiveness**. A great company can be an unattractive stock at
the wrong price; conflating the two is what makes naive research read as
cheerleading. The thesis states both explicitly, so a "SELL" on an exceptional
business is legible rather than contradictory.

The language model writes the argument, but it is handed the computed figures
and told to reason from them — the recommendation follows the valuation, not the
other way round. A deterministic fallback keeps the report intact when the model
is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company

log = logging.getLogger(__name__)


@dataclass
class InvestmentThesis:
    """The structured argument behind the recommendation."""

    rating: str                       # BUY / HOLD / SELL (from the model)
    business_quality: str = ""        # Exceptional / Strong / Average / Challenged
    business_quality_note: str = ""   # one line on why
    headline: str = ""                # the thesis in one sentence

    # The causal chain (P2.12): each link follows from the one before, so the
    # recommendation is the CONCLUSION of an argument, not just "DCF < price".
    what_market_prices_in: str = ""   # 1. market expectation
    what_we_expect: str = ""          # 2. our expectation
    fundamental_difference: str = ""  # 3. the fundamental reason they differ
    financial_impact: str = ""        # 4. what that does to the financials
    valuation_impact: str = ""        # 5. what that does to the valuation
    the_gap: str = ""                 # (legacy) short summary of the gap

    thesis_points: list[str] = field(default_factory=list)   # 3-5 reasons
    biggest_catalyst: str = ""
    biggest_risk: str = ""

    bull_triggers: list[str] = field(default_factory=list)   # would make us more positive
    bear_triggers: list[str] = field(default_factory=list)   # would make us more negative
    red_team: str = ""               # strongest argument against our own view

    generated: bool = False          # False if this is the deterministic fallback

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Grounding: turn the computed valuation into the facts the model reasons over
# ---------------------------------------------------------------------------

def _facts(company: Company, history: FinancialHistory, valuation) -> str:
    cur = valuation.currency
    lines: list[str] = [f"Company: {company.name} ({company.ticker}), {company.sector or 'n/a'}"]

    if valuation.share_price:
        lines.append(f"Current share price: {cur} {valuation.share_price:,.2f}")
    if valuation.dcf:
        lines.append(f"Our intrinsic DCF fair value: {cur} {valuation.dcf.fair_value_per_share:,.2f}")
    if valuation.blended and valuation.blended.blended_value:
        lines.append(
            f"Blended target (DCF + analyst consensus): {cur} "
            f"{valuation.blended.blended_value:,.2f}"
        )
        consensus = next(
            (e for e in valuation.blended.estimates if e.key == "analyst_consensus"), None
        )
        if consensus:
            lines.append(f"Wall Street consensus target: {cur} {consensus.value_per_share:,.2f}")
    if valuation.upside is not None:
        lines.append(f"Implied upside/downside to our target: {valuation.upside * 100:+.0f}%")
    lines.append(f"OUR recommendation (from our intrinsic analysis): {valuation.rating}")
    lines.append(
        "The Wall Street consensus target above is the MARKET's belief, not ours — it is "
        "what we are weighing our own view against. Where it diverges from our intrinsic "
        "value, treat that divergence as a central part of the thesis and a key risk, and "
        "reason toward OUR recommendation."
    )

    # Scenarios — the coherent range.
    sc = valuation.scenarios
    if sc and sc.cases:
        parts = [
            f"{c.label.split()[0]} {cur}{c.fair_value_per_share:,.0f} (p={c.probability:.0%})"
            for c in sc.cases
        ]
        lines.append("Scenario fair values: " + ", ".join(parts))
        if sc.expected_value:
            lines.append(f"Probability-weighted fair value: {cur} {sc.expected_value:,.2f}")

    # Reverse DCF — what the market is pricing in. Feed the FULL computed table, so
    # the prose describes what the price implies using ONLY these figures and never
    # invents a market-implied margin or growth number.
    pi = valuation.priced_in
    if pi and pi.rows:
        lines.append(
            "Reverse DCF — what today's price implies (these are the ONLY figures you may use "
            "for 'what the market prices in'; do NOT invent any other market-implied number):"
        )
        for row in pi.rows:
            if row.implied_value is not None and row.reachable:
                lines.append(
                    f"  - {row.label}: the price implies {row.implied_display} "
                    f"vs our base case {row.base_display}"
                )
            else:
                reason = (row.note or "").strip() or "not reachable within a plausible range"
                lines.append(
                    f"  - {row.label}: the price CANNOT be justified by this lever alone "
                    f"(our base is {row.base_display}; {reason})"
                )
    elif valuation.market_implied_growth is not None:
        lines.append(
            f"Reverse DCF: at today's price the market implies first-year revenue "
            f"growth of ~{valuation.market_implied_growth * 100:.0f}% (decaying)."
        )

    # Our forward assumptions.
    for key, label in (
        ("year_one_growth", "Our year-1 revenue growth"),
        ("terminal_margin", "Our terminal operating margin"),
        ("terminal_growth", "Our terminal growth"),
        ("wacc", "Our discount rate (WACC)"),
    ):
        a = valuation.assumptions.get(key)
        if a:
            lines.append(f"{label}: {a.display}")

    # History — the base rate the forecast decays from.
    growth = history.growth_rates("revenue")
    if growth:
        recent = ", ".join(f"FY{y}: {g * 100:+.0f}%" for y, g in growth[-4:])
        lines.append(f"Recent revenue growth — {recent}")
    margins = [(y.fiscal_year, y.operating_margin) for y in history.years if y.operating_margin]
    if margins:
        recent_m = ", ".join(f"FY{y}: {m * 100:.0f}%" for y, m in margins[-4:])
        lines.append(f"Recent operating margin — {recent_m}")

    return "\n".join(lines)


_SYSTEM = """You are a buy-side equity analyst writing the investment thesis for a research report.

Non-negotiable rules:
- Separate BUSINESS QUALITY (how good the company is) from STOCK ATTRACTIVENESS (whether to own it at this price). A great business can be a SELL if the price is too high. Never conflate the two.
- Reason from the numbers you are given. The recommendation follows the valuation — do not argue toward a predetermined answer.
- Be specific and quantitative. Tie every claim to a figure or a named fact.
- SINGLE SOURCE OF TRUTH: every number you write must come from the figures provided above (the DCF, the assumptions, the scenarios, the reverse-DCF). Do NOT invent, round differently, or estimate your own numbers. Quote the model's values exactly.
- MARKET-IMPLIED DISCIPLINE: when you say what the market "prices in" / "implies" / "expects", use ONLY the reverse-DCF figures listed above. If the table says a lever CANNOT justify the price alone, say exactly that — do NOT fabricate a percentage (e.g. never claim "the market assumes >60% margins" unless that exact figure is the computed implied value). The honest framing is usually: the price requires ~X% revenue CAGR versus our ~Y% base, so the valuation depends on a far more durable growth path than we assume.
- WHY THE MARKET IS WRONG must be SPECIFIC. For a BUY or SELL, pinpoint WHERE in the model the disagreement sits: name the exact assumption the price depends on (the implied revenue CAGR, the implied terminal margin, or the implied terminal growth) and why our number is better supported. Never write "the market is too optimistic/pessimistic" without saying which assumption carries the optimism. The bear_triggers/bull_triggers are the "what would prove us wrong" — make them check that exact assumption.
- Guidance discipline: every guidance figure must carry its period (a named quarter, a full fiscal year, a calendar year, or multi-year). Many companies (NVIDIA included) guide only ONE QUARTER ahead — never present single-quarter guidance as annual/full-year.
- Frame the thesis as: what the market is pricing in vs what we expect vs why the gap matters.
- No filler, no cheerleading, no "this underscores/highlights/demonstrates". Write like an analyst, not a summariser."""

_PROMPT = """Write the investment thesis for this company, grounded in the computed valuation below.

{facts}

Management commentary retrieved from the filings (context, evaluate it — do not treat claims as established fact):
{context}

Build the thesis as a CAUSAL CHAIN — each step must follow from the one before, so the recommendation is the conclusion of an argument, not "our DCF is below the price, therefore SELL". The chain is: market expectation -> our expectation -> the FUNDAMENTAL reason they differ -> what that does to the financials -> what that does to the valuation -> the recommendation.

Return JSON with exactly these keys:
{{
  "business_quality": "one of: Exceptional / Strong / Average / Challenged",
  "business_quality_note": "one sentence on why, citing a specific strength or weakness",
  "headline": "the thesis in ONE sentence, e.g. 'An exceptional franchise, but at {price} the market is paying for growth and margins we think normalise'",
  "what_market_prices_in": "step 1 - what the current price assumes, anchored to the reverse-DCF implied growth/margin",
  "what_we_expect": "step 2 - our base case, anchored to our specific growth/margin assumptions",
  "fundamental_difference": "step 3 - the FUNDAMENTAL business reason the two differ (competition, mix, capital intensity, TAM) — not just 'we are more conservative'",
  "financial_impact": "step 4 - what that fundamental difference does to revenue/margins/FCF over the forecast, with figures",
  "valuation_impact": "step 5 - what that does to fair value versus the price (the size of the gap and why)",
  "the_gap": "one-sentence summary of the gap",
  "thesis_points": ["3 to 5 short, specific reasons behind the recommendation"],
  "biggest_catalyst": "the single event most likely to move the stock, and its direction",
  "biggest_risk": "the single biggest risk to our view",
  "bull_triggers": ["2-3 FALSIFIABLE triggers, each tied to a SPECIFIC model assumption threshold above (e.g. 'revenue growth holds above our XX% year-1 assumption for two more quarters', 'operating margin stays above our XX% terminal assumption') — a reader must be able to check it against a number"],
  "bear_triggers": ["2-3 FALSIFIABLE triggers tied to specific model assumption thresholds, same discipline"],
  "red_team": "the strongest argument AGAINST our own recommendation, stated fairly"
}}

The recommendation is {rating}. Make the thesis consistent with it and with the numbers above. Every bull/bear trigger MUST reference a specific number from our assumptions so the thesis is falsifiable."""


def _fallback(valuation, history) -> InvestmentThesis:
    """A deterministic thesis when the model is unavailable, so the report holds."""
    cur = valuation.currency
    rating = valuation.rating
    t = InvestmentThesis(rating=rating, generated=False)

    price = f"{cur} {valuation.share_price:,.2f}" if valuation.share_price else "the current price"
    target = (
        f"{cur} {valuation.fair_value:,.2f}" if valuation.fair_value is not None else "our target"
    )
    verb = {"BUY": "below", "SELL": "above", "HOLD": "close to"}.get(rating, "near")
    t.headline = (
        f"Our valuation places fair value at {target}; the market at {price} sits {verb} that, "
        f"which is why we rate the stock {rating}."
    )
    if valuation.market_implied_growth is not None:
        t.what_market_prices_in = (
            f"At {price} the market implies roughly "
            f"{valuation.market_implied_growth * 100:.0f}% first-year revenue growth, decaying "
            f"toward the terminal rate."
        )
    g = history.growth_rates("revenue")
    if g:
        t.what_we_expect = (
            f"Our base case decays from recent growth of {g[-1][1] * 100:.0f}% toward a mature "
            f"rate, with margins normalising over the forecast."
        )
    t.the_gap = (
        "The gap between those two paths is what drives the recommendation: the more the price "
        "depends on sustained peak growth and margins, the less room there is for error."
    )
    return t


def generate_thesis(
    company: Company,
    history: FinancialHistory,
    valuation,
    *,
    qualitative_context: str = "",
    use_model: bool = True,
) -> InvestmentThesis:
    """Produce the investment thesis, grounded in the computed valuation."""
    if not use_model or valuation.dcf is None:
        return _fallback(valuation, history)

    rating = valuation.rating
    facts = _facts(company, history, valuation)
    price = f"{valuation.currency} {valuation.share_price:,.0f}" if valuation.share_price else "today's price"

    payload = complete_json(
        _PROMPT.format(facts=facts, context=(qualitative_context or "None retrieved.")[:4000],
                       rating=rating, price=price),
        system=_SYSTEM,
        temperature=0.2,
        max_tokens=1100,
    )

    if not isinstance(payload, dict):
        log.warning("thesis generation failed; using deterministic fallback")
        return _fallback(valuation, history)

    def _str(key: str) -> str:
        return str(payload.get(key) or "").strip()

    def _list(key: str) -> list[str]:
        return [str(x).strip() for x in (payload.get(key) or []) if str(x).strip()]

    return InvestmentThesis(
        rating=rating,
        business_quality=_str("business_quality") or "Average",
        business_quality_note=_str("business_quality_note"),
        headline=_str("headline"),
        what_market_prices_in=_str("what_market_prices_in"),
        what_we_expect=_str("what_we_expect"),
        fundamental_difference=_str("fundamental_difference"),
        financial_impact=_str("financial_impact"),
        valuation_impact=_str("valuation_impact"),
        the_gap=_str("the_gap"),
        thesis_points=_list("thesis_points"),
        biggest_catalyst=_str("biggest_catalyst"),
        biggest_risk=_str("biggest_risk"),
        bull_triggers=_list("bull_triggers"),
        bear_triggers=_list("bear_triggers"),
        red_team=_str("red_team"),
        generated=True,
    )
