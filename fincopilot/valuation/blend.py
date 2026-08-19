"""
Valuation triangulation: reconciling several independent valuations into one.

A discounted cash flow is a single, intrinsic opinion. On a highly-rated
company it routinely lands far below what the whole market of analysts will pay,
and reported on its own that reads as "the market is wrong" — a strong claim for
one model to make. The honest move is to triangulate: place our DCF next to the
Wall Street consensus price target and reconcile the two into a single figure,
weighting each by the confidence it earns.

Two design rules keep the blend meaningful rather than a mush:

* **Weighted, not naive.** The consensus is weighted by how many analysts stand
  behind it, so a target followed by sixty analysts counts for more than one
  followed by three — but the weight is capped, so the consensus can pull the
  blend well above our DCF without erasing it (a *balanced* triangulation).

* **Robust to outliers.** With three or more estimates, any value far outside the
  pack is dropped and reported as excluded, so one stale or mistaken number
  cannot swing the answer. The blend of what remains is a weighted average.

Every estimate, its weight, and any exclusion is recorded on the result, so the
number is auditable — the reader can see exactly which opinions produced it.

The blending itself is a pure function: estimates in, :class:`BlendedValuation`
out, no network and no global state, so it is unit-testable and reproducible.
"""

from __future__ import annotations

import logging
import statistics

from .. import config
from ..fundamentals import FinancialHistory
from .models import BlendedValuation, ValuationEstimate

log = logging.getLogger(__name__)


def _analyst_weight_multiple(count: int | None) -> float:
    """Confidence multiple for the consensus, scaled by analyst coverage.

    Reaches one full unit at the reference count and is capped, so a mega-cap
    with a hundred analysts cannot drown out every other method. With no count
    reported the target still counts, at a single unit of weight.
    """
    if not count or count <= 0:
        return 1.0
    multiple = count / config.BLEND_ANALYST_REFERENCE_COUNT
    return max(1.0, min(multiple, config.BLEND_ANALYST_MAX_WEIGHT_MULTIPLE))


def build_analyst_estimate(
    history: FinancialHistory, ticker: str, currency: str
) -> ValuationEstimate | None:
    """A per-share estimate from the Wall Street consensus price target.

    Uses the median target as the point value — robust to a single extreme
    analyst — and records the mean and the full high/low range as detail.
    Returns ``None`` when no usable target is available.
    """
    median = history.analyst_target_median
    mean = history.analyst_target_mean
    value = median or mean
    if not value or value <= 0:
        return None

    count = history.analyst_opinion_count
    weight = config.BLEND_SOURCE_WEIGHTS["analyst"] * _analyst_weight_multiple(count)

    low, high = history.analyst_target_low, history.analyst_target_high
    detail_parts = []
    if count:
        detail_parts.append(f"{count} analyst{'s' if count != 1 else ''}")
    if median:
        detail_parts.append(f"median {currency} {median:,.2f}")
    if mean and mean != median:
        detail_parts.append(f"mean {currency} {mean:,.2f}")
    if low and high:
        detail_parts.append(f"range {currency} {low:,.0f}–{high:,.0f}")
    detail = "; ".join(detail_parts)

    return ValuationEstimate(
        key="analyst_consensus",
        label="Analyst consensus",
        source_type="analyst",
        value_per_share=value,
        weight=weight,
        currency=currency,
        source_name="Wall Street consensus (via yfinance)",
        # A page the reader can open to check the targets and coverage.
        url=f"https://finance.yahoo.com/quote/{ticker}/analysis",
        detail=detail,
    )


def blend_estimates(
    estimates: list[ValuationEstimate],
    *,
    share_price: float | None,
    currency: str,
) -> BlendedValuation:
    """Reconcile ``estimates`` into one figure via a robust weighted average.

    Estimates are mutated in place to record whether they were included and, if
    not, why — so the returned object carries a complete audit trail.
    """
    result = BlendedValuation(
        estimates=estimates,
        currency=currency,
        share_price=share_price,
    )

    usable = [e for e in estimates if e.value_per_share and e.value_per_share > 0 and e.weight > 0]
    if not usable:
        return result

    # -- outlier rejection (only meaningful with three or more) -----------
    included = usable
    if len(usable) >= 3:
        median = statistics.median(e.value_per_share for e in usable)
        low_cut = median * config.BLEND_OUTLIER_LOW_MULTIPLE
        high_cut = median * config.BLEND_OUTLIER_HIGH_MULTIPLE

        kept: list[ValuationEstimate] = []
        for estimate in usable:
            if low_cut <= estimate.value_per_share <= high_cut:
                kept.append(estimate)
            else:
                estimate.included = False
                estimate.exclude_reason = (
                    f"{estimate.currency} {estimate.value_per_share:,.2f} is far from the "
                    f"{estimate.currency} {median:,.2f} median of sources; excluded as an outlier"
                )

        # Never reject so aggressively that fewer than two survive — a blend of
        # one is just that source, not a triangulation.
        if len(kept) >= 2:
            included = kept
        else:
            for estimate in usable:
                estimate.included = True
                estimate.exclude_reason = ""
            included = usable

    # -- weighted average of what survived --------------------------------
    total_weight = sum(e.weight for e in included)
    if total_weight <= 0:
        return result

    result.blended_value = sum(e.weight * e.value_per_share for e in included) / total_weight
    values = [e.value_per_share for e in included]
    result.low = min(values)
    result.high = max(values)

    labels = ", ".join(f"{e.label} ({e.weight:.1f})" for e in included)
    result.method = (
        f"Weighted average of {len(included)} source(s) by confidence: {labels}"
        + (
            f". {len(result.excluded)} excluded as outlier(s)."
            if result.excluded
            else "."
        )
    )
    return result


def build_blend(
    *,
    dcf_value: float | None,
    history: FinancialHistory,
    ticker: str,
    share_price: float | None,
    currency: str,
    comps_value: float | None = None,
) -> BlendedValuation:
    """Assemble every available valuation and blend them.

    Always includes the DCF (our intrinsic view) when present. Adds the analyst
    consensus when the stock has coverage, and a comps-implied value when one is
    supplied. Returns an empty blend if there is nothing to combine.
    """
    estimates: list[ValuationEstimate] = []

    if dcf_value and dcf_value > 0:
        estimates.append(
            ValuationEstimate(
                key="dcf",
                label="Intrinsic DCF",
                source_type="model",
                value_per_share=dcf_value,
                weight=config.BLEND_SOURCE_WEIGHTS["model"],
                currency=currency,
                source_name="Financial Copilot DCF",
                detail="Discounted free cash flow on audited figures; this model's own estimate",
            )
        )

    analyst = build_analyst_estimate(history, ticker, currency)
    if analyst is not None:
        estimates.append(analyst)

    if comps_value and comps_value > 0:
        estimates.append(
            ValuationEstimate(
                key="comps",
                label="Comparable companies",
                source_type="comps",
                value_per_share=comps_value,
                weight=config.BLEND_SOURCE_WEIGHTS["comps"],
                currency=currency,
                source_name="Peer median multiple",
                detail="Peer median P/E applied to trailing EPS",
            )
        )

    return blend_estimates(estimates, share_price=share_price, currency=currency)
