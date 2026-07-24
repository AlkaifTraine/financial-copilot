"""
Market data, and statements for companies outside SEC jurisdiction.

Two jobs:

* **Market data for everyone.** Share price, share count and beta are what
  connect a computed enterprise value to a per-share fair value and to today's
  quote. They are not in any filing — a filing is a snapshot of the past.

* **Statements for non-SEC filers.** Indian issuers such as TCS and Reliance
  file with Indian regulators, not the SEC, so `companyfacts` has nothing for
  them. yfinance exposes their statements in structured form, which is still
  far better than reading numbers out of a PDF, though it lacks XBRL's
  concept-level provenance. The difference is recorded in ``source`` and
  surfaced in the report.
"""

from __future__ import annotations

import logging
import math

from ..resolve import Company
from .models import FinancialHistory, FiscalYear

log = logging.getLogger(__name__)


# yfinance normalises statement rows to human-readable labels. Several are
# tried per field because coverage varies by exchange and company.
_INCOME_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "EBIT", "Total Operating Income As Reported"],
    "pretax_income": ["Pretax Income"],
    "tax_expense": ["Tax Provision"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "diluted_eps": ["Diluted EPS"],
    "diluted_shares": ["Diluted Average Shares"],
}

_BALANCE_ROWS = {
    "cash_and_equivalents": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "short_term_investments": ["Other Short Term Investments"],
    "total_debt": ["Total Debt"],
    "total_assets": ["Total Assets"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "shareholders_equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
}

_CASHFLOW_ROWS = {
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
    "depreciation_amortisation": ["Depreciation And Amortization", "Reconciled Depreciation"],
    "stock_compensation": ["Stock Based Compensation"],
}


def _clean(value) -> float | None:
    """Coerce a pandas cell to a usable float, or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _read_rows(frame, mapping: dict[str, list[str]]) -> dict[int, dict[str, float]]:
    """Extract ``{fiscal_year: {field: value}}`` from a yfinance statement."""
    if frame is None or getattr(frame, "empty", True):
        return {}

    out: dict[int, dict[str, float]] = {}
    labels = {str(label).strip(): label for label in frame.index}

    for column in frame.columns:
        try:
            fiscal_year = column.year
        except AttributeError:
            continue

        values: dict[str, float] = {}
        for field_name, candidates in mapping.items():
            for candidate in candidates:
                label = labels.get(candidate)
                if label is None:
                    continue
                value = _clean(frame.at[label, column])
                if value is not None:
                    values[field_name] = value
                    break

        if values:
            out[fiscal_year] = values

    return out


def fetch_market_data(company: Company) -> dict:
    """Current price, share count, market cap and beta."""
    import yfinance as yf

    try:
        info = yf.Ticker(company.ticker).info or {}
    except Exception as exc:
        log.warning("market data unavailable for %s: %s", company.ticker, exc)
        return {}

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    return {
        "share_price": _clean(price),
        "shares_outstanding": _clean(info.get("sharesOutstanding")),
        "market_cap": _clean(info.get("marketCap")),
        "beta": _clean(info.get("beta")),
        "currency": info.get("currency") or company.currency,
    }


def fetch_statements(company: Company, *, max_years: int = 6) -> FinancialHistory | None:
    """Build a history from yfinance statements, for non-SEC filers."""
    import yfinance as yf

    try:
        ticker = yf.Ticker(company.ticker)
        income = _read_rows(ticker.income_stmt, _INCOME_ROWS)
        balance = _read_rows(ticker.balance_sheet, _BALANCE_ROWS)
        cashflow = _read_rows(ticker.cashflow, _CASHFLOW_ROWS)
    except Exception as exc:
        log.warning("could not load statements for %s: %s", company.ticker, exc)
        return None

    years = sorted(set(income) | set(balance) | set(cashflow))
    if not years:
        return None

    history = FinancialHistory(
        ticker=company.ticker,
        company_name=company.name,
        currency=company.currency,
        source="yfinance",
    )

    for fiscal_year in years[-max_years:]:
        entry = FiscalYear(fiscal_year=fiscal_year, period_end=f"{fiscal_year}-12-31")
        for source in (income, balance, cashflow):
            for field_name, value in source.get(fiscal_year, {}).items():
                setattr(entry, field_name, value)
        history.years.append(entry)

    history.notes.append(
        f"Financial statement data from yfinance for {company.ticker}. "
        f"{company.name} does not file with the SEC, so audited XBRL data is "
        f"not available; figures should be checked against the company's own filings."
    )
    return history


def attach_market_data(history: FinancialHistory, company: Company) -> None:
    """Populate live market fields on an existing history, in place."""
    data = fetch_market_data(company)
    if not data:
        history.notes.append("Live market data was unavailable.")
        return

    history.share_price = data.get("share_price")
    history.shares_outstanding = data.get("shares_outstanding")
    history.market_cap = data.get("market_cap")
    history.beta = data.get("beta")

    # A history sourced from XBRL is always in USD; the quote may not be.
    if data.get("currency") and history.source != "sec_xbrl":
        history.currency = data["currency"]
