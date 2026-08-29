"""
Judging what the market's price requires against what the company has done.

A discounted cash flow on free cash flow cannot reach the prices quality
companies trade at. The market pays around 44x free cash flow for Apple, which
a DCF reaches only if the spread between the discount rate and terminal growth
is about 2.25%; a defensible spread for a business like that is 5-7%, which
pays 14-20x. No honest pair of inputs closes that. The same is true of most
large, well-regarded companies in a richly-valued market.

That has an awkward consequence for a research tool: the fair value is the
*weakest* thing it produces. It will report almost everything as overvalued,
which is methodologically unsurprising and commercially useless — nobody needs
a tool to tell them the whole market is expensive.

The reverse DCF is the strong output and it was already being computed. Rather
than asserting a price the model cannot defend, it solves for what today's
price requires of each value driver, and that claim survives the DCF's
inability to span the price. This module does the part that was missing:
comparing each of those requirements against what the company has actually
delivered, so the output is not "our number is lower than the market's" but

    "at this price the market needs a 26% operating margin; the best this
     company has ever reported is 14.2%, and it has never held that for
     two consecutive years."

That is checkable, it is specific to the company, and — unlike a fair value —
it does not quietly depend on the analyst's discount rate being right.

The rating follows from it. Expectations the company has beaten before are
undemanding; expectations beyond anything in its record are not, and the gap
between the two is the investment view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Verdicts, ordered from cheapest to most expensive.
UNDEMANDING = "undemanding"        # the price asks less than the company delivers
ACHIEVABLE = "achievable"          # in line with what it has done
DEMANDING = "demanding"            # needs its best-ever performance, sustained
UNPRECEDENTED = "unprecedented"    # beyond anything in its record
UNREACHABLE = "unreachable"        # no level of this driver reaches the price

# Ordered cheapest to most expensive. UNREACHABLE is deliberately absent: it is
# not a point on this scale.
_ORDER = [UNDEMANDING, ACHIEVABLE, DEMANDING, UNPRECEDENTED]

# "Unreachable" means no level of that driver alone reaches the price. That
# sounds like the most bearish verdict possible and is not one at all — it is
# the DCF failing to span the price on a single lever, which happens routinely
# because a DCF cannot reach the multiples quality companies trade at. Reading
# it as maximally expensive let one uninformative lever set the rating: Apple
# was rated on "no operating margin justifies this price" while the lever that
# did carry information said the price needs a 21% revenue CAGR against a
# best-ever 33.3%, a far more interesting and far less damning fact.
#
# So an unreachable driver is reported and excluded from the verdict. If EVERY
# driver is unreachable there is genuinely nothing to say, and the rating falls
# back to the DCF gap.

# How close to the demonstrated level still counts as "in line with it".
_ACHIEVABLE_TOLERANCE = 0.15


@dataclass
class DriverExpectation:
    """What the price requires of one driver, against what the company has done."""

    key: str
    label: str
    unit: str
    implied: float | None
    achieved_recent: float | None
    achieved_best: float | None
    verdict: str
    message: str

    def _fmt(self, value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value * 100:.1f}%" if self.unit == "%" else f"{value:,.2f}"

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "unit": self.unit,
            "implied": self.implied, "achieved_recent": self.achieved_recent,
            "achieved_best": self.achieved_best, "verdict": self.verdict,
            "message": self.message,
            "implied_display": self._fmt(self.implied),
            "recent_display": self._fmt(self.achieved_recent),
            "best_display": self._fmt(self.achieved_best),
        }


@dataclass
class Expectations:
    """The market's requirements, judged against the company's own record."""

    drivers: list[DriverExpectation] = field(default_factory=list)
    verdict: str = ACHIEVABLE
    rating: str = "HOLD"
    summary: str = ""

    @property
    def binding(self) -> DriverExpectation | None:
        """The single most demanding requirement — where the thesis lives."""
        if not self.drivers:
            return None
        return max(self.drivers, key=lambda d: _ORDER.index(d.verdict))

    def to_dict(self) -> dict:
        return {
            "drivers": [d.to_dict() for d in self.drivers],
            "verdict": self.verdict,
            "rating": self.rating,
            "summary": self.summary,
        }


def _classify(implied: float | None, recent: float | None, best: float | None,
              reachable: bool) -> str:
    """Where a required level sits against what the company has demonstrated."""
    if not reachable or implied is None:
        return UNREACHABLE
    if recent is None and best is None:
        return ACHIEVABLE          # nothing to judge against; do not invent a view

    ceiling = best if best is not None else recent
    floor = recent if recent is not None else best

    if implied <= floor * (1 + _ACHIEVABLE_TOLERANCE):
        # At or below what it currently does. Below it by a clear margin is the
        # interesting case: the price is asking for less than the run-rate.
        return UNDEMANDING if implied < floor * (1 - _ACHIEVABLE_TOLERANCE) else ACHIEVABLE
    if implied <= ceiling:
        return DEMANDING           # needs its best year to become the norm
    return UNPRECEDENTED


_MESSAGES = {
    UNDEMANDING: (
        "the price asks for less than the company currently delivers, so the "
        "bar is below the run rate"
    ),
    ACHIEVABLE: "the price asks for roughly what the company already does",
    DEMANDING: (
        "the price requires its best-ever level to become the normal level and "
        "hold there"
    ),
    UNPRECEDENTED: (
        "the price requires a level the company has never reached in its "
        "reported history"
    ),
    UNREACHABLE: (
        "no level of this driver alone reaches the price, so the price cannot "
        "be justified on this lever at all"
    ),
}


def _series(history, attribute: str) -> tuple[float | None, float | None]:
    """(most recent, best) for a per-year attribute."""
    values = [
        getattr(year, attribute) for year in getattr(history, "years", [])
        if getattr(year, attribute, None) is not None
    ]
    if not values:
        return None, None
    return values[-1], max(values)


def assess(valuation, history) -> Expectations:
    """Judge the priced-in requirements against the company's record."""
    comparison = getattr(valuation, "priced_in", None)
    rows = getattr(comparison, "rows", None) or []
    if not rows:
        return Expectations(summary="")

    growth_rates = []
    try:
        growth_rates = [g for _fy, g in history.growth_rates("revenue")]
    except Exception:
        pass
    recent_growth = growth_rates[-1] if growth_rates else None
    best_growth = max(growth_rates) if growth_rates else None

    recent_margin, best_margin = _series(history, "operating_margin")

    benchmarks = {
        "revenue_cagr": (recent_growth, best_growth),
        "operating_margin": (recent_margin, best_margin),
        "fcf_margin": (None, None),          # no clean historical analogue
        "terminal_growth": (None, None),     # a perpetual rate has no realised counterpart
    }

    drivers: list[DriverExpectation] = []
    for row in rows:
        recent, best = benchmarks.get(row.key, (None, None))
        verdict = _classify(row.implied_value, recent, best, row.reachable)
        drivers.append(
            DriverExpectation(
                key=row.key, label=row.label, unit=row.unit,
                implied=row.implied_value,
                achieved_recent=recent, achieved_best=best,
                verdict=verdict, message=_MESSAGES[verdict],
            )
        )

    # Only drivers with something to compare against can carry the verdict. A
    # perpetual growth rate has no realised counterpart, so judging the market
    # by it would be judging it against nothing.
    judged = [
        d for d in drivers
        if (d.achieved_recent is not None or d.achieved_best is not None)
        and d.verdict != UNREACHABLE
    ]
    if not judged:
        return Expectations(drivers=drivers, summary="")

    # The binding constraint sets the view: a price is only as justified as its
    # most demanding requirement, because every driver has to hold at once.
    worst = max(judged, key=lambda d: _ORDER.index(d.verdict))
    verdict = worst.verdict

    # "Demanding" is a HOLD, not a SELL. It means the price needs the company's
    # best year to become its normal year — a real bar, and one plenty of good
    # businesses clear. Only a requirement with no precedent in the company's
    # own record is a SELL.
    rating = {
        UNDEMANDING: "BUY",
        ACHIEVABLE: "HOLD",
        DEMANDING: "HOLD",
        UNPRECEDENTED: "SELL",
    }[verdict]

    summary = _summarise(worst, verdict)
    return Expectations(drivers=drivers, verdict=verdict, rating=rating, summary=summary)


def _summarise(worst: DriverExpectation, verdict: str) -> str:
    implied = worst._fmt(worst.implied)
    best = worst._fmt(worst.achieved_best)
    recent = worst._fmt(worst.achieved_recent)

    if verdict == UNREACHABLE:
        return (
            f"Today's price cannot be justified through {worst.label.lower()} at "
            f"any level, so it rests entirely on the other drivers holding."
        )
    if verdict == UNPRECEDENTED:
        return (
            f"Today's price requires {worst.label.lower()} of {implied}. The best "
            f"this company has reported is {best}, and it is currently at {recent} "
            f"— the price asks for something without precedent in its own record."
        )
    if verdict == DEMANDING:
        return (
            f"Today's price requires {worst.label.lower()} of {implied} against a "
            f"best-ever {best} and a current {recent}. The price is achievable only "
            f"if the company's best year becomes its normal year."
        )
    if verdict == UNDEMANDING:
        return (
            f"Today's price only requires {worst.label.lower()} of {implied}, below "
            f"the {recent} the company currently delivers. The market is asking for "
            f"less than the business is doing."
        )
    return (
        f"Today's price requires {worst.label.lower()} of {implied}, close to the "
        f"{recent} the company already delivers. Expectations are in line with the "
        f"record."
    )
