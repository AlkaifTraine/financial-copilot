"""
How old is the audited data, and is it too old to value the company on?

This exists because of a real failure. A Bikaji report generated on 2026-08-28
led with "REVENUE INR 23.3bn — FY2024" beside a report date of today. The
figures were audited and correctly sourced, and they were **two years and five
months old**: the company had since reported FY2025 and FY2026, and FY2026
revenue was 29% higher. Every number downstream — the DCF base year, the growth
series, the margin path, the thesis — inherited that staleness, and nothing in
the system noticed or said so.

The audited-provenance rule ("statement numbers come only from concept-tagged
XBRL") is what caused it: the NSE results-XBRL endpoint retains roughly three
years, so it served FY2023-FY2024 and stopped, while the annual reports the
same pipeline had already downloaded ran to FY2026. Provenance was preserved
and recency was silently lost.

So recency is treated here as a first-class property of a
:class:`~fincopilot.fundamentals.models.FinancialHistory`, on the same footing
as its source. A valuation built on data that is two reporting cycles behind is
not a conservative valuation — it is a wrong one, and the honest response is to
say so rather than to publish it with a fresh-looking date on the cover.

The thresholds below are keyed to the reporting calendar rather than to round
numbers. A fiscal year that ended eleven months ago is normal; annual results
take a quarter or two to be filed and tagged. A fiscal year that ended two
years ago means at least one full annual report exists that this system has not
read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

log = logging.getLogger(__name__)

CURRENT = "current"
AGING = "aging"
STALE = "stale"
UNUSABLE = "unusable"

# Months since the latest fiscal year END. Filing and tagging lag is real, so
# the "current" band is generous: a March year-end is not fully tagged until
# the following autumn in many markets.
_AGING_AFTER = 15      # a newer year has probably been reported by now
_STALE_AFTER = 21      # a newer year has certainly been reported
_UNUSABLE_AFTER = 33   # two or more reporting cycles missed

_LABEL = {
    CURRENT: "Current",
    AGING: "One cycle behind",
    STALE: "Stale",
    UNUSABLE: "Too old to use",
}


@dataclass(frozen=True)
class Recency:
    """The age of a financial history, and what to do about it."""

    status: str
    months_old: float
    latest_fiscal_year: int | None
    latest_period_end: str | None
    as_of: str

    @property
    def label(self) -> str:
        return _LABEL.get(self.status, self.status)

    @property
    def is_current(self) -> bool:
        return self.status == CURRENT

    @property
    def blocks_valuation(self) -> bool:
        """Whether a valuation built on this data would mislead more than help.

        Deliberately the same consequence as having no audited source at all.
        A DCF anchored to a base year two reporting cycles old is not
        "conservative" — its growth rates, margins and capital intensity all
        describe a company that no longer exists in that form.
        """
        return self.status in (STALE, UNUSABLE)

    @property
    def summary(self) -> str:
        """One line a reader can act on."""
        if self.latest_fiscal_year is None:
            return "No audited fiscal year could be dated."
        age = f"{self.months_old:.0f} months"
        if self.status == CURRENT:
            return f"Latest audited year FY{self.latest_fiscal_year}, {age} old."
        if self.status == AGING:
            return (
                f"Latest audited year is FY{self.latest_fiscal_year}, {age} old — "
                f"a more recent year has probably been reported and is not "
                f"reflected here."
            )
        return (
            f"Latest audited year is FY{self.latest_fiscal_year}, {age} old. At "
            f"least one full year has been reported since and is missing, so "
            f"every figure below describes a materially older company."
        )

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "label": self.label,
            "months_old": round(self.months_old, 1),
            "latest_fiscal_year": self.latest_fiscal_year,
            "latest_period_end": self.latest_period_end,
            "as_of": self.as_of,
            "blocks_valuation": self.blocks_valuation,
            "summary": self.summary,
        }


def _parse(value) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def assess(history, *, as_of: str | date | None = None) -> Recency:
    """Date the newest audited year in ``history`` against today.

    Anchored on ``period_end`` rather than ``fiscal_year``, because the label a
    company puts on a year is not a date: an Indian "FY2026" ends in March 2026
    while a US "FY2026" may end in December 2026 or in June. Only the period
    end supports arithmetic against a calendar.
    """
    reference = _parse(as_of) or date.today()

    periods = []
    for entry in getattr(history, "years", []) or []:
        ended = _parse(getattr(entry, "period_end", None))
        if ended is not None:
            periods.append((ended, entry.fiscal_year))

    if not periods:
        # Undatable is treated as unusable rather than assumed fine: a history
        # whose periods cannot be read cannot be checked for staleness either.
        log.warning("no datable period_end on any fiscal year; treating as unusable")
        return Recency(
            status=UNUSABLE, months_old=float("inf"), latest_fiscal_year=None,
            latest_period_end=None, as_of=reference.isoformat(),
        )

    latest_end, latest_fy = max(periods)
    months = (reference - latest_end).days / 30.44

    if months < 0:
        # A period ending in the future is a data error, not freshness.
        log.warning("latest period_end %s is after %s", latest_end, reference)
        months = 0.0

    if months <= _AGING_AFTER:
        status = CURRENT
    elif months <= _STALE_AFTER:
        status = AGING
    elif months <= _UNUSABLE_AFTER:
        status = STALE
    else:
        status = UNUSABLE

    if status != CURRENT:
        log.warning(
            "financial data for %s is %s: latest FY%s ended %s, %.0f months ago",
            getattr(history, "ticker", "?"), status, latest_fy, latest_end, months,
        )

    return Recency(
        status=status, months_old=months, latest_fiscal_year=latest_fy,
        latest_period_end=latest_end.isoformat(), as_of=reference.isoformat(),
    )


def freshest(*histories):
    """Pick the history with the newest audited period end.

    Sources disagree about how far back they retain: the NSE results-XBRL
    endpoint holds roughly three years, exchange annual-report archives hold
    far more, and a company's own investor site is arbitrary. When more than
    one source yields a history, recency decides — an older but "nicer" source
    is not worth two years of drift.
    """
    candidates = [h for h in histories if h is not None and getattr(h, "years", None)]
    if not candidates:
        return None
    return min(candidates, key=lambda h: assess(h).months_old)
