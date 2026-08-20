"""
Quantified risk assessment.

The risk section used to be prose — a paragraph restating Item 1A. That reads as
box-ticking, and a reader cannot act on it: "supply chain risk" is true of every
company and tells you nothing about what to do. This replaces it with a table
that forces each material risk to carry the four things an investor actually
needs — how likely it is, what it hits, what it does to *this* valuation, and the
single indicator that would show it beginning to happen.

The valuation impact is where being grounded in the model pays off. A generic
report can only say a risk is "significant"; here the risk pass is handed the
scenario range and the reverse-DCF gaps, so it can anchor the impact to a number
the rest of the report already stands behind — the bear case, or the downside to
our intrinsic value — rather than inventing one.

Like the thesis, the model writes the assessment from figures it is handed, and
a deterministic fallback keeps the report intact when the model is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company

log = logging.getLogger(__name__)


@dataclass
class QuantifiedRisk:
    """One material risk, with the four things needed to act on it."""

    risk: str                    # short name
    description: str = ""        # one line on the mechanism
    probability: str = ""        # Low / Medium / High (optionally qualified)
    financial_impact: str = ""   # what it hits, sized where possible
    valuation_impact: str = ""   # effect on fair value, anchored to a number
    early_warning: str = ""      # the single indicator that it is starting

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskAssessment:
    """The material risks, most significant first."""

    risks: list[QuantifiedRisk] = field(default_factory=list)
    generated: bool = False      # False if this is the deterministic fallback

    def to_dict(self) -> dict:
        return {"risks": [r.to_dict() for r in self.risks], "generated": self.generated}


# ---------------------------------------------------------------------------
# Grounding: the numbers the risk pass anchors its impact estimates to
# ---------------------------------------------------------------------------

def _facts(history: FinancialHistory, valuation) -> str:
    cur = valuation.currency
    lines: list[str] = []

    if valuation.share_price:
        lines.append(f"Current share price: {cur} {valuation.share_price:,.2f}")
    if valuation.fair_value is not None:
        lines.append(f"Our intrinsic fair value: {cur} {valuation.fair_value:,.2f}")
    if valuation.upside is not None:
        lines.append(f"Upside/downside from price to our value: {valuation.upside * 100:+.0f}%")
    lines.append(f"Our rating: {valuation.rating}")

    # Scenarios give the downside a concrete, already-modelled number to cite.
    sc = valuation.scenarios
    if sc and sc.cases:
        for case in sc.cases:
            upside = f"{case.upside * 100:+.0f}% vs price" if case.upside is not None else "n/a"
            lines.append(
                f"{case.label} fair value {cur} {case.fair_value_per_share:,.2f} "
                f"({upside}, probability {case.probability:.0%})"
            )

    # Reverse-DCF gaps: the levers where the price already assumes more than we
    # do are exactly where a disappointment does the most damage.
    pi = valuation.priced_in
    if pi and pi.rows:
        for row in pi.rows:
            if row.implied_value is None:
                lines.append(
                    f"Priced in — {row.label}: our base {row.base_display}, and the price "
                    f"cannot be justified on this lever even at its ceiling ({row.note})"
                )
            else:
                lines.append(
                    f"Priced in — {row.label}: our base {row.base_display}, market-implied "
                    f"{row.implied_display} (gap {row.gap_display})"
                )

    # Our forward assumptions — the things that, if wrong, move the valuation.
    for key, label in (
        ("year_one_growth", "Our year-1 revenue growth"),
        ("terminal_margin", "Our terminal operating margin"),
        ("terminal_growth", "Our terminal growth"),
        ("wacc", "Our discount rate (WACC)"),
    ):
        assumption = valuation.assumptions.get(key)
        if assumption:
            lines.append(f"{label}: {assumption.display}")

    # Recent history — the base rate a risk would deflect the business away from.
    growth = history.growth_rates("revenue")
    if growth:
        recent = ", ".join(f"FY{y}: {g * 100:+.0f}%" for y, g in growth[-4:])
        lines.append(f"Recent revenue growth — {recent}")
    margins = [(y.fiscal_year, y.operating_margin) for y in history.years if y.operating_margin]
    if margins:
        recent_m = ", ".join(f"FY{y}: {m * 100:.0f}%" for y, m in margins[-4:])
        lines.append(f"Recent operating margin — {recent_m}")

    return "\n".join(lines)


_SYSTEM = """You are a buy-side equity analyst writing the risk section of a research report.

Rules:
- Each risk must be MATERIAL to the investment case and SPECIFIC to this company. Never list generic risks ("macroeconomic conditions", "competition") without saying concretely how they hit THIS business.
- Quantify. Size the financial impact against the segment or revenue it threatens. Anchor the valuation impact to a number already in the analysis — the bear-case fair value, the downside to our intrinsic value, or the reverse-DCF gap the price is leaning on.
- DIRECTION MUST BE CONSISTENT. A risk is a DOWNSIDE: reason it as risk -> operational impact -> financial metric -> forecast -> valuation, and every link must point the SAME way. A negative event LOWERS the affected metric and LOWERS fair value. Never write that a risk moves a metric UP or "toward" a HIGHER figure — e.g. do NOT say a risk reduces revenue growth "toward" the market-implied rate, which is ABOVE our base case. The downside target is always BELOW our base case, never above it.
- The early-warning indicator must be a single, observable metric a reader could actually watch each quarter — not "monitor the situation".
- Order by materiality, most damaging first. Prefer 4-6 risks over a long thin list.
- No filler, no hedging, no restating the risk in the impact field."""

_PROMPT = """Company: {company}

The computed valuation this report already stands behind:
{facts}

Risk factors and commentary retrieved from the filings (evaluate them; do not treat every claim as established):
{context}

Identify the material risks to the investment case. For each, give:
- risk: a short, specific name (under 8 words)
- description: one sentence on the mechanism — how it actually damages the business
- probability: exactly one of Low / Medium / High, optionally with a 2-3 word qualifier
- financial_impact: what it hits and roughly how big, tied to a segment or a figure
- valuation_impact: the effect on fair value, anchored to a number from the analysis above
- early_warning: the single observable metric that would show this starting to happen

Return JSON:
{{"risks": [
  {{"risk": "...", "description": "...", "probability": "...", "financial_impact": "...", "valuation_impact": "...", "early_warning": "..."}}
]}}"""


def _fallback(history: FinancialHistory, valuation) -> RiskAssessment:
    """A deterministic assessment from the numbers, so the report always holds."""
    cur = valuation.currency
    risks: list[QuantifiedRisk] = []

    # 1. Valuation risk, sized from the gap between price and our value.
    if valuation.upside is not None and valuation.upside < 0:
        bear = valuation.scenarios.bear if valuation.scenarios else None
        bear_txt = (
            f" our bear case sits at {cur} {bear.fair_value_per_share:,.2f}"
            if bear and bear.fair_value_per_share else ""
        )
        risks.append(QuantifiedRisk(
            risk="Price above intrinsic value",
            description=(
                "The market price exceeds our discounted-cash-flow value, so a re-rating "
                "toward intrinsic value is itself the primary risk to owning it here."
            ),
            probability="Medium",
            financial_impact="A de-rating affects the whole equity value, not one segment.",
            valuation_impact=(
                f"{valuation.upside * 100:+.0f}% to our intrinsic value;{bear_txt}."
                if bear_txt else f"{valuation.upside * 100:+.0f}% to our intrinsic value."
            ),
            early_warning="A quarter of revenue or margins below consensus.",
        ))

    # 2. Growth-normalisation risk, from the reverse-DCF growth gap.
    pi = valuation.priced_in
    growth_row = next((r for r in pi.rows if r.key == "revenue_cagr"), None) if pi else None
    if growth_row and growth_row.implied_value is not None:
        risks.append(QuantifiedRisk(
            risk="Growth normalises faster than priced",
            description=(
                "The price embeds a multi-year growth rate well above our base case; growth "
                "reverting sooner removes the support under the current multiple."
            ),
            probability="Medium",
            financial_impact="Directly reduces the revenue base every later year compounds from.",
            valuation_impact=(
                f"The price needs a {growth_row.implied_display} CAGR versus our "
                f"{growth_row.base_display}; closing that gap is the downside."
            ),
            early_warning="Year-on-year revenue growth decelerating below the implied path.",
        ))

    return RiskAssessment(risks=risks, generated=False)


def generate_risks(
    company: Company,
    history: FinancialHistory,
    valuation,
    *,
    qualitative_context: str = "",
    use_model: bool = True,
) -> RiskAssessment:
    """Produce the quantified risk assessment, grounded in the valuation."""
    if not use_model or valuation.dcf is None:
        return _fallback(history, valuation)

    payload = complete_json(
        _PROMPT.format(
            company=f"{company.name} ({company.ticker})",
            facts=_facts(history, valuation),
            context=(qualitative_context or "None retrieved.")[:4000],
        ),
        system=_SYSTEM,
        temperature=0.2,
        max_tokens=1300,
    )

    if not isinstance(payload, dict) or not isinstance(payload.get("risks"), list):
        log.warning("risk generation failed; using deterministic fallback")
        return _fallback(history, valuation)

    def _str(entry: dict, key: str) -> str:
        return str(entry.get(key) or "").strip()

    risks: list[QuantifiedRisk] = []
    for entry in payload["risks"]:
        if not isinstance(entry, dict):
            continue
        name = _str(entry, "risk")
        if not name:
            continue
        risks.append(QuantifiedRisk(
            risk=name,
            description=_str(entry, "description"),
            probability=_str(entry, "probability") or "Medium",
            financial_impact=_str(entry, "financial_impact"),
            valuation_impact=_str(entry, "valuation_impact"),
            early_warning=_str(entry, "early_warning"),
        ))

    if not risks:
        return _fallback(history, valuation)

    return RiskAssessment(risks=risks, generated=True)
