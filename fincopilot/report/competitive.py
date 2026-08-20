"""
Competitive moat and management-versus-us.

Two judgements a research report owes the reader, each with a discipline that
keeps it honest.

The moat is assessed in two separate dimensions, because they genuinely differ:
its STRENGTH today (how hard the advantage is to attack) and its DIRECTION (is it
widening or narrowing). A company can have a formidable current moat that is
quietly eroding; collapsing the two into one label hides exactly the thing an
investor needs. The strength is built up from the specific sources of advantage —
software ecosystem, hardware, switching costs, scale, network effects, customer
relationships, technology leadership — not asserted in the abstract, and tied to
the terminal margin the moat is meant to defend.

Management-versus-us has one rule: a "management says" row must quote an ACTUAL
sourced statement, never a forecast we put in management's mouth. Each row runs
statement -> source -> our interpretation -> our model -> the difference, so the
reader can see where we take management at their word and where we do not, and
why.

Grounded in the model's own numbers; a deterministic fallback keeps the report
intact when the model is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company

log = logging.getLogger(__name__)


@dataclass
class MoatDimension:
    """One structural source of advantage, assessed on its own."""

    dimension: str
    assessment: str
    strength: str      # Strong / Moderate / Weak / None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ManagementClaim:
    """One place a SOURCED management statement meets our modelled view."""

    topic: str
    management_statement: str    # an actual claim/guidance from the filings
    source: str                  # where it came from (filing, period)
    our_interpretation: str      # what we take it to mean
    our_model: str               # what we actually model
    difference: str              # aligned, or how/why we differ

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompetitiveAnalysis:
    """Moat (strength and direction, built up by dimension) and management-vs-us."""

    moat_strength: str = "Undetermined"     # Wide / Narrow / None (how strong TODAY)
    moat_direction: str = "Stable"          # Widening / Stable / Narrowing (trajectory)
    moat_summary: str = ""
    dimensions: list[MoatDimension] = field(default_factory=list)
    competitive_threats: list[str] = field(default_factory=list)
    management_vs_us: list[ManagementClaim] = field(default_factory=list)
    generated: bool = False

    # Kept so existing callers/renderers that reference a single label still work.
    @property
    def moat_rating(self) -> str:
        return self.moat_strength

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["dimensions"] = [d.to_dict() for d in self.dimensions]
        payload["management_vs_us"] = [m.to_dict() for m in self.management_vs_us]
        payload["moat_rating"] = self.moat_strength
        return payload


def _facts(valuation) -> str:
    cur = valuation.currency
    lines: list[str] = []
    if valuation.fair_value is not None and valuation.share_price:
        lines.append(
            f"Our fair value {cur} {valuation.fair_value:,.2f} vs price "
            f"{cur} {valuation.share_price:,.2f} (rating {valuation.rating})"
        )
    for key, label in (
        ("terminal_margin", "Our terminal operating margin (the margin the moat must defend)"),
        ("year_one_growth", "Our year-1 revenue growth"),
        ("terminal_growth", "Our terminal growth"),
    ):
        assumption = valuation.assumptions.get(key)
        if assumption:
            lines.append(f"{label}: {assumption.display}")
    pi = valuation.priced_in
    margin_row = next((r for r in pi.rows if r.key == "operating_margin"), None) if pi else None
    if margin_row and margin_row.implied_value is not None:
        lines.append(
            f"The price implies a mature operating margin of {margin_row.implied_display} "
            f"(ours: {margin_row.base_display}) — a moat question."
        )
    return "\n".join(lines)


_SYSTEM = """You are an equity analyst assessing competitive advantage and testing management's story.

Rules:
- Score the moat in TWO separate dimensions. STRENGTH = how hard the advantage is to attack today (Wide / Narrow / None). DIRECTION = its trajectory (Widening / Stable / Narrowing). They are independent: a very strong moat can be narrowing. Never merge them.
- Build the strength up from specific sources of advantage, each rated on its own; do not assert it in the abstract. Tie it to whether it defends our assumed terminal margin.
- MANAGEMENT-VERSUS-US IS STRICT: a "management says" entry must be an ACTUAL statement or guidance the company has made (quote or close paraphrase), with its source. NEVER invent a forecast and attribute it to management. If you cannot cite a real statement on a topic, omit that row. For each real statement give our interpretation, what we model, and the difference.
- No cheerleading. A strong company with a narrowing moat should read as exactly that."""

_PROMPT = """Company: {company}

Our valuation and the assumptions the moat must support:
{facts}

Filings context (competition, differentiation, and ACTUAL management statements/guidance):
{context}

Produce:
1. moat_strength: Wide / Narrow / None (how strong the advantage is TODAY)
2. moat_direction: Widening / Stable / Narrowing (its trajectory)
3. moat_summary: 1-2 sentences tying strength AND direction to whether they defend our assumed terminal margin
4. dimensions: assess each source of advantage that applies — software/ecosystem, hardware, switching costs, scale, network effects, customer relationships, technology leadership. Each: dimension, assessment (one line), strength (Strong / Moderate / Weak / None)
5. competitive_threats: 2-4 specific threats to the advantage
6. management_vs_us: 3-4 topics where an ACTUAL sourced management statement meets our view. Each: topic, management_statement (real quote/paraphrase — do NOT invent), source (filing/period), our_interpretation, our_model (what we model), difference. Omit any topic where you cannot cite a real statement.

Return JSON:
{{"moat_strength": "...", "moat_direction": "...", "moat_summary": "...",
  "dimensions": [{{"dimension": "...", "assessment": "...", "strength": "..."}}],
  "competitive_threats": ["..."],
  "management_vs_us": [{{"topic": "...", "management_statement": "...", "source": "...", "our_interpretation": "...", "our_model": "...", "difference": "..."}}]}}"""


def _fallback(valuation) -> CompetitiveAnalysis:
    """A deterministic assessment when the model is unavailable."""
    analysis = CompetitiveAnalysis(generated=False)
    margin = valuation.assumptions.get("terminal_margin")
    if margin:
        analysis.moat_summary = (
            f"Our valuation assumes a {margin.display} mature operating margin; that level only "
            f"holds if a durable advantage keeps competitors from eroding it, which is the moat "
            f"question this report turns on."
        )
    return analysis


def generate_competitive(
    company: Company,
    history: FinancialHistory,
    valuation,
    *,
    qualitative_context: str = "",
    use_model: bool = True,
) -> CompetitiveAnalysis:
    """Produce the moat assessment (strength + direction) and management-versus-us."""
    if not use_model or valuation.dcf is None:
        return _fallback(valuation)

    payload = complete_json(
        _PROMPT.format(
            company=f"{company.name} ({company.ticker})",
            facts=_facts(valuation),
            context=(qualitative_context or "None retrieved.")[:4000],
        ),
        system=_SYSTEM,
        temperature=0.2,
        max_tokens=1400,
    )

    if not isinstance(payload, dict):
        log.warning("competitive analysis failed; using deterministic fallback")
        return _fallback(valuation)

    def _str(source: dict, key: str) -> str:
        return str(source.get(key) or "").strip()

    def _list(key: str) -> list[str]:
        return [str(x).strip() for x in (payload.get(key) or []) if str(x).strip()]

    dimensions = [
        MoatDimension(
            dimension=_str(entry, "dimension"),
            assessment=_str(entry, "assessment"),
            strength=_str(entry, "strength") or "Moderate",
        )
        for entry in (payload.get("dimensions") or [])
        if isinstance(entry, dict) and _str(entry, "dimension")
    ]

    # Management rows are kept ONLY when a real statement is present — the whole
    # point of the discipline is that we never fabricate a management forecast.
    management = [
        ManagementClaim(
            topic=_str(entry, "topic"),
            management_statement=_str(entry, "management_statement"),
            source=_str(entry, "source"),
            our_interpretation=_str(entry, "our_interpretation"),
            our_model=_str(entry, "our_model"),
            difference=_str(entry, "difference"),
        )
        for entry in (payload.get("management_vs_us") or [])
        if isinstance(entry, dict) and _str(entry, "topic") and _str(entry, "management_statement")
    ]

    strength = _str(payload, "moat_strength") or "Undetermined"
    summary = _str(payload, "moat_summary")
    if strength == "Undetermined" and not summary and not dimensions and not management:
        return _fallback(valuation)

    return CompetitiveAnalysis(
        moat_strength=strength,
        moat_direction=_str(payload, "moat_direction") or "Stable",
        moat_summary=summary,
        dimensions=dimensions,
        competitive_threats=_list("competitive_threats"),
        management_vs_us=management,
        generated=True,
    )
