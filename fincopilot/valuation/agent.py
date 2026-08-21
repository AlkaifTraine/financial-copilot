"""
The assumption agent: propose, then critique against the fundamentals.

The single-shot proposal (``assumptions._propose``) sets each forward assumption
in isolation. That is where systematic conservatism creeps in — three separately
"reasonable" choices (growth a little low, margin normalised a little hard, a
discount rate a little high) compound into a fair value several times below the
market, and nothing in a one-pass design ever looks at the *combined* result and
asks whether it hangs together.

This adds that second look. The agent is shown the assumptions it proposed AND
the fair value they produce — together with how that value sits against the
market price and the analyst consensus — and asked one question: reasoning only
from the fundamentals, is any single assumption indefensibly conservative or
aggressive? The market and consensus are used strictly as a *miscalibration
signal* (a value far below BOTH is a reason to re-examine which lever is most
extreme), never as a target to match. The recommendation stays intrinsic.

Any revision it returns is re-clamped to the same data-derived bounds as any
proposal, so the agent reasons *within* the guardrails — it cannot talk the model
into a number the history will not support. Cached per company + data fingerprint,
so a valuation still reproduces exactly.
"""

from __future__ import annotations

import json
import logging

from .. import config
from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company
from .assumptions import _history_fingerprint, _history_table
from .reverse import _cagr

log = logging.getLogger(__name__)


def _consensus(history: FinancialHistory) -> tuple[float | None, int | None]:
    target = history.analyst_target_median or history.analyst_target_mean
    return target, history.analyst_opinion_count


def _facts(history: FinancialHistory, inputs, dcf, wacc: float) -> str:
    cur = history.currency
    price = history.share_price
    fv = dcf.fair_value_per_share
    cagr = _cagr(inputs.growth_path)
    current_margin = inputs.margin_path[0]
    terminal_margin = inputs.margin_path[-1]

    lines = [
        _history_table(history),
        "",
        "Assumptions currently proposed, and the value they produce:",
        f"- Year-1 revenue growth {inputs.growth_path[0] * 100:.0f}%, "
        f"decaying to a {len(inputs.growth_path)}-year CAGR of {cagr * 100:.0f}%",
        f"- Operating margin fading from {current_margin * 100:.0f}% today to a terminal "
        f"{terminal_margin * 100:.0f}%",
        f"- Terminal growth {inputs.terminal_growth * 100:.1f}%",
        f"- Discount rate (WACC) {wacc * 100:.1f}%",
        f"- Resulting intrinsic fair value: {cur} {fv:,.2f}",
        f"- {dcf.terminal_value_share * 100:.0f}% of value is terminal",
    ]
    if price:
        lines.append(f"- Current market price: {cur} {price:,.2f} ({fv / price - 1:+.0%} vs our value)")
    target, count = _consensus(history)
    if target:
        cov = f" across {count} analysts" if count else ""
        lines.append(
            f"- Analyst consensus target: {cur} {target:,.2f}{cov} "
            f"({fv / target - 1:+.0%} vs our value)"
        )
    return "\n".join(lines)


_SYSTEM = """You are the senior reviewer on a DCF valuation. A junior analyst has set the forward assumptions; you check whether the fair value they produce is defensible.

Your job is calibration, not cheerleading:
- Reason ONLY from the fundamentals — the revenue and margin history, demonstrated durability, reinvestment, competitive dynamics. NEVER change an assumption merely to move the fair value toward the market price. The recommendation is intrinsic and is allowed to disagree with the market.
- BUT: if the fair value sits far below BOTH the market price AND the independent analyst consensus, that is a signal that one assumption is probably too conservative — a lone DCF being 70% below the entire street usually means the model, not the whole market, is wrong. Identify WHICH single assumption is least defensible and correct that one, from the fundamentals.
- Symmetrically, a value far ABOVE both is a signal an assumption is too aggressive.
- Change as little as possible. Revise only assumptions you can justify from the data; leave defensible ones alone.
- A company decelerating from very high growth should still be modelled as a strong grower for years, not snapped to a mature rate. A business with a steady high margin and a real moat should keep most of it."""

_PROMPT = """Company: {company}

{facts}

Management commentary from the filings (context, weigh it):
{context}

Review the assumptions. Reasoning from the fundamentals, is any single one indefensibly conservative or aggressive? Consider the divergence from the market and consensus only as a signal of possible miscalibration, never as a target.

All three values are DECIMALS, not percentages: 0.35 means 35%, 0.48 means 48%. Never write 35 or 48.

Return JSON:
{{"assessment": "one sentence: are these jointly defensible, and if not which lever is the problem",
  "revise": true or false,
  "year_one_revenue_growth": {{"value": 0.35, "rationale": "grounded in the history/segments"}},
  "terminal_operating_margin": {{"value": 0.48, "rationale": "grounded in durability/competition"}},
  "terminal_growth_rate": {{"value": 0.025, "rationale": "below long-run GDP"}}}}

If the assumptions are already defensible, set "revise" to false and repeat the current values."""


def _cache_path(company: Company, history: FinancialHistory):
    cache_dir = config.CACHE_DIR / "critique"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{company.slug}_{_history_fingerprint(history)}.json"


def critique_assumptions(
    company: Company,
    history: FinancialHistory,
    inputs,
    dcf,
    wacc: float,
    *,
    qualitative_context: str = "",
) -> dict | None:
    """Return a revised proposal (same shape as ``_propose``) or None to keep.

    Cached per company + data fingerprint so the valuation reproduces exactly.
    """
    cache_path = _cache_path(company, history)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached or None
        except (json.JSONDecodeError, OSError):
            pass

    payload = complete_json(
        _PROMPT.format(
            company=f"{company.name} ({company.ticker})",
            facts=_facts(history, inputs, dcf, wacc),
            context=(qualitative_context or "None retrieved.")[:3000],
        ),
        system=_SYSTEM,
        model=config.WRITER_MODEL,
        temperature=0.0,
        max_tokens=700,
    )

    revision = _parse(payload)

    # Cache the decision either way ({} means "no revision"), so the second pass
    # is never re-run for the same fingerprint.
    try:
        cache_path.write_text(json.dumps(revision or {}), encoding="utf-8")
    except OSError:
        pass

    return revision


def _normalise(value) -> float | None:
    """Coerce to a decimal rate, defending against a percentage slipping through.

    The reviewer is asked for decimals, but a model shown "growth 65%" in the
    facts sometimes answers 60 instead of 0.60. Any assumption here (growth,
    margin, terminal growth) is well under 1.5 as a decimal, so a magnitude above
    that is unambiguously a percentage and is rescaled — the alternative is a
    silent 100x error that the clamp only partly masks by pinning to a bound.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) > 1.5:
        number /= 100.0
    # Quantise to the nearest 0.5 percentage point. The reviewer is not precise to
    # the basis point, and rounding means run-to-run wording noise in the model
    # ("14.8%" one run, "15.1%" the next) collapses to the same input, so the
    # cached result and the live result agree. Reproducibility proper comes from
    # the per-fingerprint cache; this keeps the first run stable enough to trust.
    return round(number / 0.005) * 0.005


def _parse(payload) -> dict | None:
    if not isinstance(payload, dict) or not payload.get("revise"):
        return None
    keys = ("year_one_revenue_growth", "terminal_operating_margin", "terminal_growth_rate")
    revision: dict = {}
    for key in keys:
        entry = payload.get(key)
        if isinstance(entry, dict) and "value" in entry:
            value = _normalise(entry["value"])
            if value is not None:
                revision[key] = {"value": value, "rationale": str(entry.get("rationale", ""))}
    # A revision must carry at least one usable assumption to be worth a rebuild.
    return revision or None
