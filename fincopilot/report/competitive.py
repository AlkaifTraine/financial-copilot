"""
Competitive moat and management-versus-us.

Two related judgements a research report owes the reader. First, the moat: does a
durable advantage protect the returns the valuation assumes? This is not a
decoration — a terminal operating margin only holds if something stops
competitors competing it away, so the moat rating and the terminal-margin
assumption have to tell the same story. Second, management-versus-us: where our
modelled view agrees with management's guidance and, more usefully, where it does
not — because the places we are more cautious than the company are exactly where
the thesis lives.

Grounded, like the thesis and risks, in the model's own numbers so the moat is
argued against the margin it is meant to defend, not asserted in the abstract. A
deterministic fallback keeps the report intact when the model is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company

log = logging.getLogger(__name__)


@dataclass
class ManagementClaim:
    """One place management's stated view meets our modelled view."""

    topic: str
    management_says: str
    our_view: str
    assessment: str              # Aligned / More cautious / More optimistic

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompetitiveAnalysis:
    """The moat assessment and the management-versus-us comparison."""

    moat_rating: str = "Undetermined"     # Wide / Narrow / None / Undetermined
    moat_summary: str = ""
    moat_sources: list[str] = field(default_factory=list)      # durable advantages
    competitive_threats: list[str] = field(default_factory=list)
    management_vs_us: list[ManagementClaim] = field(default_factory=list)
    generated: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["management_vs_us"] = [m.to_dict() for m in self.management_vs_us]
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
- A moat is a DURABLE, structural reason returns persist (switching costs, network effects, scale, IP, brand). Rate it Wide / Narrow / None and justify from evidence, not reputation.
- Tie the moat to the numbers: a high terminal margin is only defensible if a moat protects it. Say whether the assumed margin and the moat are consistent.
- For management-versus-us, be concrete about WHERE we differ and WHY. The interesting rows are the ones where we are more cautious than the company.
- No cheerleading. A strong company with a narrowing moat should read as exactly that."""

_PROMPT = """Company: {company}

Our valuation and the assumptions the moat must support:
{facts}

Filings context (competition, differentiation, management guidance and claims):
{context}

Produce:
1. moat_rating: Wide / Narrow / None
2. moat_summary: 1-2 sentences, tying the moat to whether it defends our assumed terminal margin
3. moat_sources: 2-4 specific, durable sources of advantage (or note their absence)
4. competitive_threats: 2-4 specific threats to that advantage
5. management_vs_us: 3-4 topics where management's stated view meets ours. Each: topic, management_says (their claim/guidance), our_view (what we model), assessment (Aligned / More cautious / More optimistic)

Return JSON:
{{"moat_rating": "...", "moat_summary": "...", "moat_sources": ["..."], "competitive_threats": ["..."],
  "management_vs_us": [{{"topic": "...", "management_says": "...", "our_view": "...", "assessment": "..."}}]}}"""


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
    """Produce the moat assessment and management-versus-us comparison."""
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
        max_tokens=1200,
    )

    if not isinstance(payload, dict):
        log.warning("competitive analysis failed; using deterministic fallback")
        return _fallback(valuation)

    def _str(source: dict, key: str) -> str:
        return str(source.get(key) or "").strip()

    def _list(key: str) -> list[str]:
        return [str(x).strip() for x in (payload.get(key) or []) if str(x).strip()]

    management = [
        ManagementClaim(
            topic=_str(entry, "topic"),
            management_says=_str(entry, "management_says"),
            our_view=_str(entry, "our_view"),
            assessment=_str(entry, "assessment") or "Aligned",
        )
        for entry in (payload.get("management_vs_us") or [])
        if isinstance(entry, dict) and _str(entry, "topic")
    ]

    rating = _str(payload, "moat_rating") or "Undetermined"
    summary = _str(payload, "moat_summary")
    if rating == "Undetermined" and not summary and not management:
        return _fallback(valuation)

    return CompetitiveAnalysis(
        moat_rating=rating,
        moat_summary=summary,
        moat_sources=_list("moat_sources"),
        competitive_threats=_list("competitive_threats"),
        management_vs_us=management,
        generated=True,
    )
