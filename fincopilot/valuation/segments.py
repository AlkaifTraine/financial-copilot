"""
Segment-level revenue forecasting, as a bottom-up cross-check on the total.

A single top-down revenue growth number hides the thing that actually decides a
forecast: which part of the business is growing. NVIDIA's ~8% base-case CAGR is a
blend of a Data Center business compounding fast and a Graphics business that is
nearly flat — and "8% on the blend" is far less defensible than "40% on the half
that is two-thirds of revenue, fading to nothing on the rest, which nets to X".

This module forecasts each reportable segment on its own trajectory and sums
them, then compares that bottom-up total CAGR to the top-down number the DCF
actually uses. The point is not to replace the headline valuation — that stays
the deterministic top-down DCF — but to *stress-test its central assumption*: if
the bottom-up and top-down growth disagree sharply, the report says so.

The segment history is not in the structured filing data (SEC company-facts does
not break revenue out by segment), so it is extracted from the segment footnote
by the model and then cached per company + data fingerprint — the same
"model proposes, cached for reproducibility" contract as the DCF assumptions. The
arithmetic that turns those figures into a forecast is pure and deterministic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

from .. import config
from ..fundamentals import FinancialHistory
from ..llm import complete_json
from ..resolve import Company
from .assumptions import _history_fingerprint
from .dcf import decay_path
from .reverse import _cagr

log = logging.getLogger(__name__)

# Segment growth is bounded exactly like the top-down year-one growth: a segment
# that just grew 200% is not forecast to keep doing so, and one in decline is not
# forecast into oblivion within a year.
_SEG_MAX_GROWTH = 0.75
_SEG_MIN_GROWTH = -0.30
_SEG_TERMINAL_GROWTH = 0.025

# Above this reconciliation gap between the summed segments and reported total,
# the extraction is treated as indicative only (eliminations, an "all other"
# bucket, or a mis-extraction), not a clean bottom-up build.
_RECONCILE_TOLERANCE = 0.15

# Above this gap the extraction has simply failed — the segments cannot be a
# cross-check on a total they are nowhere near — so the exhibit is suppressed
# rather than printing figures that would embarrass the report. A bad extraction
# is silently dropped; a good one is shown; a partial one is shown as indicative.
_SUPPRESS_TOLERANCE = 0.35


@dataclass
class SegmentLine:
    """One reportable segment: its recent history and its forecast."""

    name: str
    history: list[tuple[int, float]]        # (fiscal_year, revenue), oldest first
    latest_revenue: float
    year_one_growth: float
    implied_cagr: float
    terminal_revenue: float                 # revenue in the final forecast year

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentAnalysis:
    """Bottom-up segment forecast and its reconciliation to the top-down total."""

    segments: list[SegmentLine] = field(default_factory=list)
    currency: str = "USD"
    base_year: int = 0
    horizon: int = 0

    latest_segment_sum: float = 0.0
    latest_total_revenue: float = 0.0
    reconciliation_gap: float = 0.0         # segment_sum / total - 1

    bottom_up_terminal_revenue: float = 0.0
    bottom_up_cagr: float = 0.0
    top_down_cagr: float | None = None

    generated: bool = False
    note: str = ""

    @property
    def reconciles(self) -> bool:
        return abs(self.reconciliation_gap) <= _RECONCILE_TOLERANCE

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["segments"] = [s.to_dict() for s in self.segments]
        payload["reconciles"] = self.reconciles
        return payload


_EXTRACT_PROMPT = """From the segment disclosures below for {company}, extract FULL-FISCAL-YEAR reportable operating segment revenue.

Reported TOTAL revenue, for reference — the segments must sum to approximately these figures:
{totals}

Segment disclosures (note: some tables are quarterly or year-to-date, NOT full year — only use full fiscal-year figures, and if only interim figures are shown, do not guess):
{context}

Rules:
- ONLY segments the company actually reports (business/product segments), never geographic splits.
- Values in MILLIONS of the reporting currency, matching the "$ in millions" the filings use (e.g. 116193 for $116.2 billion).
- Use the SAME fiscal years as the totals above. The segment revenues for a year must sum to roughly that year's total.

Return JSON:
{{"segments": [
  {{"name": "segment name", "revenue_by_year": {{"{recent_year}": 0.0}}}}
]}}

If the text does not disclose full-year segment revenue, return {{"segments": []}}."""


def _totals_anchor(history: FinancialHistory) -> str:
    lines = []
    for year in history.years[-3:]:
        if year.revenue:
            lines.append(f"FY{year.fiscal_year}: {year.revenue / 1e6:,.0f} million")
    return "\n".join(lines) or "not available"


def _extract_segments(company: Company, history: FinancialHistory, context: str) -> list[dict]:
    """Extract segment revenue from the filing text, cached per company + data.

    Cached on the ticker and a fingerprint of the reported numbers — not on the
    retrieved context, which varies run to run — so a company's segment exhibit
    reproduces exactly until its filings change, matching the DCF's contract.
    """
    cache_dir = config.CACHE_DIR / "segments"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{company.slug}_{_history_fingerprint(history)}.json"

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8")).get("segments", [])
        except (json.JSONDecodeError, OSError):
            pass

    recent_year = history.latest.fiscal_year if history.latest else ""
    payload = complete_json(
        _EXTRACT_PROMPT.format(
            company=f"{company.name} ({company.ticker})",
            totals=_totals_anchor(history),
            recent_year=recent_year,
            context=(context or "No segment disclosure was retrieved.")[:5000],
        ),
        temperature=0.0,
        max_tokens=600,
    )
    result = payload if isinstance(payload, dict) else {"segments": []}
    if not isinstance(result.get("segments"), list):
        result = {"segments": []}

    try:
        cache_path.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass

    return result["segments"]


def _clean_history(raw: dict) -> list[tuple[int, float]]:
    """Parse a segment's ``revenue_by_year`` (millions) into (year, base-units)."""
    out: list[tuple[int, float]] = []
    for year, value in (raw or {}).items():
        try:
            fiscal_year = int(str(year)[:4])
            revenue = float(value) * 1e6
        except (TypeError, ValueError):
            continue
        if revenue > 0:
            out.append((fiscal_year, revenue))
    out.sort()
    return out


def forecast_segments(
    extracted: list[dict],
    *,
    history: FinancialHistory,
    top_down_growth: list[float],
    generated: bool,
) -> SegmentAnalysis | None:
    """Turn extracted segment history into a bottom-up forecast (pure/deterministic)."""
    if not extracted or not top_down_growth or history.latest is None:
        return None

    horizon = len(top_down_growth)
    base_year = history.latest.fiscal_year
    latest_total = history.latest.revenue or 0.0

    segments: list[SegmentLine] = []
    segment_sum_latest = 0.0
    totals = [0.0] * horizon

    for entry in extracted:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        hist = _clean_history(entry.get("revenue_by_year", {}))
        if not name or not hist:
            continue

        latest_revenue = hist[-1][1]
        segment_sum_latest += latest_revenue

        # Year-one growth from the segment's own most recent step, else fall back
        # to the top-down rate when only one year could be extracted.
        if len(hist) >= 2 and hist[-2][1] > 0:
            year_one = hist[-1][1] / hist[-2][1] - 1
        else:
            year_one = top_down_growth[0]
        year_one = max(min(year_one, _SEG_MAX_GROWTH), _SEG_MIN_GROWTH)

        path = decay_path(year_one, _SEG_TERMINAL_GROWTH, horizon, config.DCF_GROWTH_DECAY)
        revenue = latest_revenue
        for offset, growth in enumerate(path):
            revenue *= 1 + growth
            totals[offset] += revenue

        segments.append(SegmentLine(
            name=name,
            history=hist,
            latest_revenue=latest_revenue,
            year_one_growth=year_one,
            implied_cagr=_cagr(path),
            terminal_revenue=revenue,
        ))

    if not segments:
        return None

    reconciliation_gap = (segment_sum_latest / latest_total - 1) if latest_total else 0.0

    # A sum that is nowhere near the reported total is a failed extraction, not a
    # cross-check. Drop it rather than render misleading figures.
    if latest_total and abs(reconciliation_gap) > _SUPPRESS_TOLERANCE:
        log.info(
            "segment sum off reported total by %.0f%%; suppressing exhibit",
            reconciliation_gap * 100,
        )
        return None

    bottom_up_cagr = (
        (totals[-1] / segment_sum_latest) ** (1 / horizon) - 1
        if segment_sum_latest > 0 else 0.0
    )
    top_down_cagr = _cagr(top_down_growth)

    analysis = SegmentAnalysis(
        segments=segments,
        currency=history.currency,
        base_year=base_year,
        horizon=horizon,
        latest_segment_sum=segment_sum_latest,
        latest_total_revenue=latest_total,
        reconciliation_gap=reconciliation_gap,
        bottom_up_terminal_revenue=totals[-1],
        bottom_up_cagr=bottom_up_cagr,
        top_down_cagr=top_down_cagr,
        generated=generated,
    )
    analysis.note = _reconciliation_note(analysis)
    return analysis


def _reconciliation_note(analysis: SegmentAnalysis) -> str:
    cur = analysis.currency
    parts: list[str] = []

    if analysis.reconciles:
        parts.append(
            f"The reported segments sum to {cur} {analysis.latest_segment_sum / 1e9:,.1f}bn, "
            f"within {abs(analysis.reconciliation_gap) * 100:.0f}% of total revenue — a clean "
            f"bottom-up build."
        )
    else:
        parts.append(
            f"The reported segments sum to {cur} {analysis.latest_segment_sum / 1e9:,.1f}bn "
            f"versus {cur} {analysis.latest_total_revenue / 1e9:,.1f}bn of total revenue "
            f"({analysis.reconciliation_gap * 100:+.0f}%); read the split as indicative, since "
            f"eliminations or an unreported bucket do not reconcile exactly."
        )

    if analysis.top_down_cagr is not None:
        diff = analysis.bottom_up_cagr - analysis.top_down_cagr
        if abs(diff) <= 0.02:
            parts.append(
                f"Summed bottom-up, the segments imply a "
                f"{analysis.bottom_up_cagr * 100:.0f}% total CAGR — in line with the "
                f"{analysis.top_down_cagr * 100:.0f}% our top-down DCF assumes, which supports "
                f"the headline forecast."
            )
        else:
            direction = "above" if diff > 0 else "below"
            parts.append(
                f"Summed bottom-up, the segments imply a "
                f"{analysis.bottom_up_cagr * 100:.0f}% total CAGR — {abs(diff) * 100:.0f}pp "
                f"{direction} the {analysis.top_down_cagr * 100:.0f}% our top-down DCF assumes. "
                f"That gap is where the growth debate on this name actually sits."
            )

    return " ".join(parts)


def build_segments(
    company: Company,
    history: FinancialHistory,
    valuation,
    *,
    qualitative_context: str = "",
    use_model: bool = True,
) -> SegmentAnalysis | None:
    """Extract segment revenue and forecast each, as a cross-check on the total."""
    if valuation.dcf is None or not valuation.dcf.forecast:
        return None

    extracted = _extract_segments(company, history, qualitative_context) if use_model else []
    top_down_growth = [f.revenue_growth for f in valuation.dcf.forecast]
    return forecast_segments(
        extracted, history=history, top_down_growth=top_down_growth, generated=use_model
    )
