"""
Valuation data structures, built around an explicit assumption ledger.

A discounted cash flow model is only as credible as its inputs, and the inputs
are where every DCF actually gets decided. So an assumption here is never a
bare number: it records where the value came from, how it was computed, why it
is reasonable, and whether it had to be clamped to stay defensible.

That ledger is what the report prints. A reader who disagrees with the output
can see exactly which input to argue with — which is the difference between a
valuation and a number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Where an assumption's value came from. Ordered by how much it should be
# trusted, and shown in the report so measured inputs are distinguishable from
# fallbacks.
SOURCE_HISTORICAL = "historical"   # computed from the company's own filings
SOURCE_MARKET = "market"           # live market data (price, beta, yields)
SOURCE_MODEL = "model"             # proposed by the language model, then bounded
SOURCE_DEFAULT = "default"         # configured fallback; nothing better available

SOURCE_LABELS = {
    SOURCE_HISTORICAL: "Derived from filings",
    SOURCE_MARKET: "Live market data",
    SOURCE_MODEL: "Model estimate (bounded)",
    SOURCE_DEFAULT: "Standard default",
}


@dataclass
class Assumption:
    """One input to the valuation, with its full justification."""

    key: str
    label: str
    value: float
    unit: str                       # "%", "x", "years", "currency"
    source: str
    derivation: str                 # how the number was arrived at
    rationale: str = ""             # why it is a reasonable choice
    bounds: tuple[float, float] | None = None
    clamped: bool = False           # True if a proposal was pulled into bounds
    raw_value: float | None = None  # pre-clamp value, when clamped
    # Evidence chain for a model estimate: historical -> guidance -> industry ->
    # competitive -> management -> model output. Empty for mechanical assumptions.
    provenance: dict = field(default_factory=dict)

    def format(self, value: float | None) -> str:
        """Render a value in this assumption's own unit.

        Shared by ``display`` and the clamp warnings so a beta is never
        reported as a percentage — an earlier version printed a clamped beta of
        0.241 as "24.1%", which reads as a completely different quantity.
        """
        if value is None:
            return "-"
        if self.unit == "%":
            return f"{value * 100:.1f}%"
        if self.unit == "x":
            return f"{value:.2f}x"
        if self.unit == "years":
            return f"{value:.0f}"
        return f"{value:,.2f}"

    @property
    def display(self) -> str:
        return self.format(self.value)

    @property
    def raw_display(self) -> str:
        return self.format(self.raw_value)

    def to_dict(self) -> dict:
        return {**asdict(self), "display": self.display,
                "raw_display": self.raw_display,
                "source_label": SOURCE_LABELS.get(self.source, self.source)}


@dataclass
class AssumptionLedger:
    """Every assumption behind a valuation, in presentation order."""

    items: list[Assumption] = field(default_factory=list)

    def add(self, assumption: Assumption) -> Assumption:
        self.items.append(assumption)
        return assumption

    def get(self, key: str) -> Assumption | None:
        return next((a for a in self.items if a.key == key), None)

    def value(self, key: str, default: float | None = None) -> float | None:
        found = self.get(key)
        return found.value if found else default

    @property
    def clamped(self) -> list[Assumption]:
        """Assumptions whose proposed value had to be constrained.

        Surfaced deliberately: a clamped assumption is a signal that the model
        wanted something the data would not support.
        """
        return [a for a in self.items if a.clamped]

    def by_source(self, source: str) -> list[Assumption]:
        return [a for a in self.items if a.source == source]

    def to_dict(self) -> list[dict]:
        return [a.to_dict() for a in self.items]


@dataclass
class ForecastYear:
    """One projected year of the explicit forecast period."""

    year: int
    revenue: float
    revenue_growth: float
    ebit: float
    operating_margin: float
    nopat: float
    depreciation: float
    capex: float
    change_in_working_capital: float
    free_cash_flow: float
    discount_factor: float
    present_value: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DCFResult:
    """Output of the discounted cash flow model."""

    forecast: list[ForecastYear] = field(default_factory=list)

    wacc: float = 0.0
    terminal_growth: float = 0.0

    pv_forecast: float = 0.0
    terminal_value: float = 0.0
    pv_terminal: float = 0.0
    enterprise_value: float = 0.0
    net_debt: float = 0.0
    equity_value: float = 0.0
    shares_outstanding: float = 0.0
    fair_value_per_share: float = 0.0

    currency: str = "USD"

    @property
    def terminal_value_share(self) -> float:
        """Fraction of enterprise value coming from the terminal value.

        A standard diagnostic. Above roughly 75% the valuation is driven by the
        perpetuity assumption rather than by the forecast, and should be read
        with that in mind — so the report states it rather than hiding it.
        """
        if self.enterprise_value <= 0:
            return 0.0
        return self.pv_terminal / self.enterprise_value

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["forecast"] = [f.to_dict() for f in self.forecast]
        payload["terminal_value_share"] = self.terminal_value_share
        return payload


@dataclass
class SensitivityGrid:
    """Fair value per share across WACC and terminal growth."""

    wacc_values: list[float] = field(default_factory=list)
    growth_values: list[float] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)   # [wacc][growth]
    base_wacc: float = 0.0
    base_growth: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PeerMultiple:
    ticker: str
    name: str
    market_cap: float | None = None
    pe: float | None = None
    ev_to_ebitda: float | None = None
    ev_to_sales: float | None = None
    revenue_growth: float | None = None
    tier: str = "direct"          # "direct" or "adjacent" (structurally different)
    rationale: str = ""           # why this peer is included
    in_median: bool = True        # whether it feeds the peer median


@dataclass
class CompsResult:
    peers: list[PeerMultiple] = field(default_factory=list)
    median_pe: float | None = None
    median_ev_sales: float | None = None
    implied_value_per_share: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["peers"] = [asdict(p) for p in self.peers]
        return payload


@dataclass
class ScenarioDriver:
    """One value driver as it is set in a single scenario.

    Carried so the report can show *why* a case differs from the base — not
    just that the number moved, but which lever moved it and by how much.
    """

    key: str
    label: str
    value: float
    unit: str                       # "%", "x"
    base_value: float | None = None

    def _fmt(self, value: float | None) -> str:
        if value is None:
            return "-"
        if self.unit == "%":
            return f"{value * 100:.1f}%"
        if self.unit == "x":
            return f"{value:.2f}x"
        return f"{value:,.2f}"

    @property
    def display(self) -> str:
        return self._fmt(self.value)

    @property
    def delta_display(self) -> str:
        """Signed move from the base case, in percentage points where relevant."""
        if self.base_value is None:
            return ""
        delta = self.value - self.base_value
        if self.unit == "%":
            return f"{delta * 100:+.1f}pp"
        if self.unit == "x":
            return f"{delta:+.2f}x"
        return f"{delta:+,.2f}"

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "display": self.display,
            "delta_display": self.delta_display,
        }


@dataclass
class ScenarioCase:
    """One coherent state of the world and the value it implies.

    A scenario is not a grid cell: the drivers move *together* in a way that
    tells a single story (demand softens, so growth and margins fall and the
    market demands a higher return, all at once). Each case therefore carries
    both its driver settings and its resulting per-share value.
    """

    key: str                        # "bear" | "base" | "bull"
    label: str
    probability: float
    narrative: str

    drivers: list[ScenarioDriver] = field(default_factory=list)

    fair_value_per_share: float = 0.0
    enterprise_value: float = 0.0
    equity_value: float = 0.0
    terminal_value_share: float = 0.0

    # Fractional return from the current price to this case's fair value.
    upside: float | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["drivers"] = [d.to_dict() for d in self.drivers]
        return payload


@dataclass
class ScenarioAnalysis:
    """The bear / base / bull set, plus what they say jointly."""

    cases: list[ScenarioCase] = field(default_factory=list)
    currency: str = "USD"
    share_price: float | None = None

    # Probability-weighted fair value across the three cases. Reported as an
    # expectation, never as a point forecast: the spread beside it is the
    # honest part of the answer.
    expected_value: float = 0.0
    expected_upside: float | None = None

    def case(self, key: str) -> ScenarioCase | None:
        return next((c for c in self.cases if c.key == key), None)

    @property
    def bear(self) -> ScenarioCase | None:
        return self.case("bear")

    @property
    def base(self) -> ScenarioCase | None:
        return self.case("base")

    @property
    def bull(self) -> ScenarioCase | None:
        return self.case("bull")

    @property
    def value_range(self) -> tuple[float, float] | None:
        """(low, high) per-share fair value across the cases."""
        values = [c.fair_value_per_share for c in self.cases if c.fair_value_per_share]
        if not values:
            return None
        return (min(values), max(values))

    @property
    def dispersion(self) -> float | None:
        """Bull-minus-bear spread as a multiple of the base value.

        A compact read on how much the valuation depends on which world you
        believe you are in. A dispersion of 1.5 means the bull case is worth
        150% of the base *more* than the bear case — a wide, assumption-driven
        range that the reader should weigh accordingly.
        """
        base = self.base
        rng = self.value_range
        if not base or not base.fair_value_per_share or not rng:
            return None
        return (rng[1] - rng[0]) / base.fair_value_per_share

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["cases"] = [c.to_dict() for c in self.cases]
        payload["value_range"] = self.value_range
        payload["dispersion"] = self.dispersion
        return payload


@dataclass
class ValuationEstimate:
    """One independent per-share valuation, from one method or source.

    The blend is only as honest as its inputs are visible, so every estimate
    carries where it came from, what it is worth, how much weight it was given,
    and — when it was thrown out as an outlier — why.
    """

    key: str                        # "dcf" | "analyst_consensus" | ...
    label: str
    source_type: str                # "model" | "analyst" | "comps" | "web"
    value_per_share: float
    weight: float
    currency: str = "USD"

    source_name: str = ""           # e.g. "Wall Street consensus (yfinance)"
    url: str | None = None          # a page a reader can open to check it
    as_of: str | None = None
    detail: str = ""                # e.g. "Median of 58 analysts; USD 150–260"

    included: bool = True           # False when rejected from the blend
    exclude_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlendedValuation:
    """Several independent valuations reconciled into one figure."""

    estimates: list[ValuationEstimate] = field(default_factory=list)
    currency: str = "USD"
    share_price: float | None = None

    blended_value: float = 0.0
    method: str = ""                # human description of how it was combined

    # Range across the estimates that were actually blended.
    low: float | None = None
    high: float | None = None

    @property
    def included(self) -> list[ValuationEstimate]:
        return [e for e in self.estimates if e.included]

    @property
    def excluded(self) -> list[ValuationEstimate]:
        return [e for e in self.estimates if not e.included]

    @property
    def upside(self) -> float | None:
        """Fractional upside from the current price to the blended value."""
        if not self.share_price or not self.blended_value:
            return None
        return self.blended_value / self.share_price - 1

    @property
    def rating(self) -> str:
        """Rating derived from the blended value, on the same thresholds as the DCF."""
        from .. import config

        upside = self.upside
        if upside is None:
            return "NOT RATED"
        if upside >= config.RATING_THRESHOLDS["BUY"]:
            return "BUY"
        if upside >= config.RATING_THRESHOLDS["HOLD"]:
            return "HOLD"
        return "SELL"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["estimates"] = [e.to_dict() for e in self.estimates]
        payload["upside"] = self.upside
        payload["rating"] = self.rating
        return payload


@dataclass
class PricedInRow:
    """One driver, compared between our base case and what the price implies.

    A single reverse-DCF answers "what growth is priced in?"; this generalises
    it. For each value driver we solve — holding every *other* driver at our base
    case — for the level of that one driver that makes the DCF equal today's
    price. The result reads as "if the market is right about everything else, it
    must believe X about this", which is a checkable claim rather than a bet.
    """

    key: str
    label: str
    unit: str                       # "%"
    base_value: float
    implied_value: float | None     # None when the price is unreachable on this lever alone
    reachable: bool = True
    note: str = ""                  # economic caveat, e.g. an implied margin above 100%

    def _fmt(self, value: float | None) -> str:
        if value is None:
            return "—"
        if self.unit == "%":
            return f"{value * 100:.1f}%"
        return f"{value:,.2f}"

    @property
    def base_display(self) -> str:
        return self._fmt(self.base_value)

    @property
    def implied_display(self) -> str:
        return self._fmt(self.implied_value)

    @property
    def gap_display(self) -> str:
        """Signed distance from our base case to the market-implied level."""
        if self.implied_value is None:
            return "—"
        delta = self.implied_value - self.base_value
        if self.unit == "%":
            return f"{delta * 100:+.1f}pp"
        return f"{delta:+,.2f}"

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "base_display": self.base_display,
            "implied_display": self.implied_display,
            "gap_display": self.gap_display,
        }


@dataclass
class PricedInComparison:
    """The full "what is priced in" table: our base case vs the market's price.

    Each row isolates one driver. Together they answer the question a reader of a
    DCF actually has when the fair value sits below the price — not "is the model
    right?" but "what would have to be true for the price to be right?".
    """

    rows: list[PricedInRow] = field(default_factory=list)
    currency: str = "USD"
    share_price: float | None = None
    dcf_fair_value: float | None = None
    horizon: int = 0
    summary: str = ""     # which assumption the price leans on most

    def to_dict(self) -> dict:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "currency": self.currency,
            "share_price": self.share_price,
            "dcf_fair_value": self.dcf_fair_value,
            "horizon": self.horizon,
            "summary": self.summary,
        }


@dataclass
class Valuation:
    """The complete valuation: DCF, comps, sensitivity, scenarios, and ledger."""

    ticker: str
    company_name: str
    currency: str = "USD"

    dcf: DCFResult | None = None
    sensitivity: SensitivityGrid | None = None
    scenarios: ScenarioAnalysis | None = None
    comps: CompsResult | None = None
    blended: BlendedValuation | None = None
    assumptions: AssumptionLedger = field(default_factory=AssumptionLedger)

    share_price: float | None = None

    # Year-one revenue growth the current price implies, holding every other
    # assumption fixed. Reframes a large DCF discount as a testable statement
    # about market expectations rather than a prediction that the market is wrong.
    market_implied_growth: float | None = None

    # The full reverse-DCF comparison: for each driver, what the price implies
    # vs our base case, shown side by side. Generalises market_implied_growth.
    priced_in: "PricedInComparison | None" = None

    warnings: list[str] = field(default_factory=list)

    @property
    def fair_value(self) -> float | None:
        return self.dcf.fair_value_per_share if self.dcf else None

    @property
    def upside(self) -> float | None:
        """Fractional upside from the current price to the DCF fair value.

        A non-positive fair value is not a -100%-plus SELL; it means the model is
        degenerate here (margins and growth do not cover the cost of capital and
        net debt), so there is no meaningful upside to quote and no rating to
        derive. Surfaced as a warning instead, in ``value_company``.
        """
        if not self.share_price or not self.fair_value or self.fair_value <= 0:
            return None
        return self.fair_value / self.share_price - 1

    # -- headline (blended if available, else DCF) ------------------------
    # The figure the product leads with. A DCF alone reads as "the market is
    # wrong"; the blend reconciles it with the analyst consensus into a single
    # number a reader can act on. When no blend could be built (no analyst
    # coverage), the headline falls back to the DCF so nothing is ever blank.

    @property
    def headline_value(self) -> float | None:
        if self.blended and self.blended.blended_value:
            return self.blended.blended_value
        return self.fair_value

    @property
    def headline_upside(self) -> float | None:
        if not self.share_price or not self.headline_value or self.headline_value <= 0:
            return None
        return self.headline_value / self.share_price - 1

    @property
    def headline_rating(self) -> str:
        if self.blended and self.blended.blended_value:
            return self.blended.rating
        return self.rating

    @property
    def rating(self) -> str:
        """Rating derived from computed upside.

        Replaces the hard-coded "BUY" banner in the previous report generator,
        which printed the same recommendation for every company regardless of
        what the numbers said.
        """
        from .. import config

        upside = self.upside
        if upside is None:
            return "NOT RATED"
        if upside >= config.RATING_THRESHOLDS["BUY"]:
            return "BUY"
        if upside >= config.RATING_THRESHOLDS["HOLD"]:
            return "HOLD"
        return "SELL"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "currency": self.currency,
            "dcf": self.dcf.to_dict() if self.dcf else None,
            "sensitivity": self.sensitivity.to_dict() if self.sensitivity else None,
            "scenarios": self.scenarios.to_dict() if self.scenarios else None,
            "comps": self.comps.to_dict() if self.comps else None,
            "blended": self.blended.to_dict() if self.blended else None,
            "assumptions": self.assumptions.to_dict(),
            "priced_in": self.priced_in.to_dict() if self.priced_in else None,
            "share_price": self.share_price,
            "fair_value": self.fair_value,
            "upside": self.upside,
            "rating": self.rating,
            "headline_value": self.headline_value,
            "headline_upside": self.headline_upside,
            "headline_rating": self.headline_rating,
            "warnings": self.warnings,
        }
