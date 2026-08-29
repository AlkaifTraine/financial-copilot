"""
Normalised financial statements.

One shape regardless of where the numbers came from — SEC XBRL for US filers,
structured statement data for everyone else — so the valuation engine never
needs to know its source. Every field records enough provenance for the report
to state, per line item, where the figure came from.

All monetary values are in the reporting currency's base units (dollars, not
millions). Formatting happens at the presentation layer; doing it here is how
scale errors get baked in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Where a set of statements came from. Every entry here is a regulatory filing
# by the company itself, concept-tagged and audited — that is the entry
# requirement, not a nice-to-have. A market-data vendor is deliberately absent:
# its figures cannot be traced to a filing, so they cannot support a valuation.
SOURCE_LABELS: dict[str, str] = {
    "sec_xbrl": (
        "SEC XBRL company facts — the company's own audited, tagged filing data"
    ),
    "nse_indas_xbrl": (
        "audited consolidated Ind-AS XBRL filed with the NSE — the company's "
        "own tagged filing data"
    ),
}


@dataclass
class FiscalYear:
    """One fiscal year of audited figures."""

    fiscal_year: int
    period_end: str                       # ISO date, the authoritative anchor

    # Income statement
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    pretax_income: float | None = None
    tax_expense: float | None = None
    net_income: float | None = None
    diluted_eps: float | None = None
    diluted_shares: float | None = None

    # Cash flow
    operating_cash_flow: float | None = None
    capex: float | None = None
    depreciation_amortisation: float | None = None
    stock_compensation: float | None = None

    # Balance sheet
    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
    total_debt: float | None = None
    total_assets: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    shareholders_equity: float | None = None

    # -- derived ----------------------------------------------------------

    @property
    def gross_margin(self) -> float | None:
        if self.revenue and self.gross_profit is not None:
            return self.gross_profit / self.revenue
        if self.revenue and self.cost_of_revenue is not None:
            return (self.revenue - self.cost_of_revenue) / self.revenue
        return None

    @property
    def operating_margin(self) -> float | None:
        if self.revenue and self.operating_income is not None:
            return self.operating_income / self.revenue
        return None

    @property
    def net_margin(self) -> float | None:
        if self.revenue and self.net_income is not None:
            return self.net_income / self.revenue
        return None

    @property
    def effective_tax_rate(self) -> float | None:
        """Tax actually paid as a share of pre-tax profit.

        Preferred over a statutory rate in the DCF: it reflects the company's
        real mix of jurisdictions, credits and incentives.
        """
        if self.pretax_income and self.tax_expense is not None and self.pretax_income > 0:
            rate = self.tax_expense / self.pretax_income
            # Guard against one-off charges producing a nonsensical rate.
            return rate if 0.0 <= rate <= 0.60 else None
        return None

    @property
    def free_cash_flow(self) -> float | None:
        if self.operating_cash_flow is not None and self.capex is not None:
            return self.operating_cash_flow - abs(self.capex)
        return None

    @property
    def net_debt(self) -> float | None:
        """Debt net of cash and liquid investments."""
        if self.total_debt is None:
            return None
        liquid = (self.cash_and_equivalents or 0.0) + (self.short_term_investments or 0.0)
        return self.total_debt - liquid

    @property
    def working_capital(self) -> float | None:
        """Operating working capital: receivables and inventory net of payables.

        Cash and short-term investments are removed deliberately. Including
        them measures the treasury, not the operating cycle — a profitable
        company that parks its earnings in term deposits shows "working
        capital" rising in step with revenue, and a DCF then charges that
        build-up against free cash flow every forecast year as though the cash
        had been consumed by growth.

        The effect is large and one-directional: for Bikaji it put incremental
        working capital at 25% of incremental revenue — the model's ceiling —
        against a cash conversion cycle of roughly 23 days, which is nearer 6%.
        The better the company is at converting profit to cash, the more this
        definition penalises it.
        """
        if self.current_assets is None or self.current_liabilities is None:
            return None
        operating_assets = self.current_assets
        for treasury in (self.cash_and_equivalents, self.short_term_investments):
            if treasury is not None:
                operating_assets -= treasury
        return operating_assets - self.current_liabilities

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinancialHistory:
    """A company's fiscal years, oldest first, plus current market data."""

    ticker: str
    company_name: str
    currency: str = "USD"
    source: str = ""                      # a key of SOURCE_LABELS
    years: list[FiscalYear] = field(default_factory=list)

    # Live market data — the valuation's link to today's price.
    share_price: float | None = None
    shares_outstanding: float | None = None
    market_cap: float | None = None
    beta: float | None = None

    # Wall Street consensus price targets, aggregated across the analysts who
    # cover the stock. An external, market-based valuation the blend triangulates
    # our intrinsic DCF against. All in the quote currency.
    analyst_target_mean: float | None = None
    analyst_target_median: float | None = None
    analyst_target_high: float | None = None
    analyst_target_low: float | None = None
    analyst_opinion_count: int | None = None

    notes: list[str] = field(default_factory=list)

    # How old the newest audited year is, attached by ``load_financials`` (see
    # fundamentals/recency.py). Optional so a hand-built history in a test or a
    # script still constructs; absent, recency simply is not enforced.
    recency: object | None = field(default=None, repr=False)

    # -- access -----------------------------------------------------------

    @property
    def source_label(self) -> str:
        """Human-readable provenance, for the UI and the report."""
        return SOURCE_LABELS.get(self.source, self.source or "an unknown source")

    @property
    def is_audited_filing(self) -> bool:
        """Whether these figures come from the company's own audited filing.

        Always true for a history the loader will return — it declines to build
        one otherwise. Kept explicit so the report states provenance from the
        data rather than from an assumption about how it was loaded.
        """
        return self.source in SOURCE_LABELS

    @property
    def latest(self) -> FiscalYear | None:
        return self.years[-1] if self.years else None

    @property
    def fiscal_years(self) -> list[int]:
        return [y.fiscal_year for y in self.years]

    def year(self, fiscal_year: int) -> FiscalYear | None:
        return next((y for y in self.years if y.fiscal_year == fiscal_year), None)

    def series(self, attribute: str) -> list[tuple[int, float]]:
        """(fiscal_year, value) pairs where the value is present."""
        out = []
        for entry in self.years:
            value = getattr(entry, attribute, None)
            if value is not None:
                out.append((entry.fiscal_year, value))
        return out

    # -- derived ----------------------------------------------------------

    def cagr(self, attribute: str = "revenue") -> float | None:
        """Compound annual growth rate across the available history."""
        points = self.series(attribute)
        if len(points) < 2:
            return None

        (_first_year, first), (_last_year, last) = points[0], points[-1]
        periods = len(points) - 1
        if first <= 0 or last <= 0:
            return None
        return (last / first) ** (1 / periods) - 1

    def growth_rates(self, attribute: str = "revenue") -> list[tuple[int, float]]:
        """Year-over-year growth for each consecutive pair."""
        points = self.series(attribute)
        return [
            (points[i][0], points[i][1] / points[i - 1][1] - 1)
            for i in range(1, len(points))
            if points[i - 1][1]
        ]

    def mean(self, attribute: str, last_n: int | None = None) -> float | None:
        values = [value for _year, value in self.series(attribute)]
        if last_n:
            values = values[-last_n:]
        return sum(values) / len(values) if values else None

    @property
    def is_sufficient_for_dcf(self) -> bool:
        """Whether there is enough *and* recent enough history to forecast on.

        Two years of revenue and one usable free cash flow figure is the
        minimum; below that a DCF is arithmetic dressed up as analysis.

        Staleness disqualifies on the same terms. A base year two reporting
        cycles old does not make a valuation conservative, it makes it a
        valuation of a company that no longer exists in that form — its growth
        rate, margin and capital intensity have all since been superseded by
        results the company has already published. This is checked here rather
        than at the call sites because every caller already asks this question
        before valuing, so there is no path that can skip it.
        """
        revenue_points = len(self.series("revenue"))
        fcf_points = len(self.series("free_cash_flow"))
        if revenue_points < 2 or fcf_points < 1:
            return False
        if self.recency is not None and getattr(self.recency, "blocks_valuation", False):
            return False
        return True

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["years"] = [y.to_dict() for y in self.years]
        return payload
