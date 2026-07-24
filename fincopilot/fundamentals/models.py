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
        if self.current_assets is not None and self.current_liabilities is not None:
            return self.current_assets - self.current_liabilities
        return None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinancialHistory:
    """A company's fiscal years, oldest first, plus current market data."""

    ticker: str
    company_name: str
    currency: str = "USD"
    source: str = ""                      # "sec_xbrl" or "yfinance"
    years: list[FiscalYear] = field(default_factory=list)

    # Live market data — the valuation's link to today's price.
    share_price: float | None = None
    shares_outstanding: float | None = None
    market_cap: float | None = None
    beta: float | None = None

    notes: list[str] = field(default_factory=list)

    # -- access -----------------------------------------------------------

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
        """Whether there is enough history to build a defensible forecast.

        Two years of revenue and one usable free cash flow figure is the
        minimum; below that a DCF is arithmetic dressed up as analysis.
        """
        revenue_points = len(self.series("revenue"))
        fcf_points = len(self.series("free_cash_flow"))
        return revenue_points >= 2 and fcf_points >= 1

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["years"] = [y.to_dict() for y in self.years]
        return payload
