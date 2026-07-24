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
class Valuation:
    """The complete valuation: DCF, comps, sensitivity, and the ledger."""

    ticker: str
    company_name: str
    currency: str = "USD"

    dcf: DCFResult | None = None
    sensitivity: SensitivityGrid | None = None
    comps: CompsResult | None = None
    assumptions: AssumptionLedger = field(default_factory=AssumptionLedger)

    share_price: float | None = None

    # Year-one revenue growth the current price implies, holding every other
    # assumption fixed. Reframes a large DCF discount as a testable statement
    # about market expectations rather than a prediction that the market is wrong.
    market_implied_growth: float | None = None

    warnings: list[str] = field(default_factory=list)

    @property
    def fair_value(self) -> float | None:
        return self.dcf.fair_value_per_share if self.dcf else None

    @property
    def upside(self) -> float | None:
        """Fractional upside from the current price to fair value."""
        if not self.share_price or not self.fair_value:
            return None
        return self.fair_value / self.share_price - 1

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
            "comps": self.comps.to_dict() if self.comps else None,
            "assumptions": self.assumptions.to_dict(),
            "share_price": self.share_price,
            "fair_value": self.fair_value,
            "upside": self.upside,
            "rating": self.rating,
            "warnings": self.warnings,
        }
