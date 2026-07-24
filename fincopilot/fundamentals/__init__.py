"""Audited financial statements and live market data."""

from __future__ import annotations

import logging

from ..resolve import Company
from .models import FinancialHistory, FiscalYear

log = logging.getLogger(__name__)

__all__ = ["FinancialHistory", "FiscalYear", "load_financials"]


def load_financials(company: Company, *, max_years: int = 6) -> FinancialHistory | None:
    """Load the best available financial history for ``company``.

    SEC XBRL is preferred wherever it exists: it is the company's own tagged,
    audited data, with a concept and an accession number behind every figure.
    yfinance is the fallback for issuers outside SEC jurisdiction.
    """
    from . import market, xbrl

    history = xbrl.fetch(company, max_years=max_years)

    if history is None:
        log.info("falling back to yfinance statements for %s", company.ticker)
        history = market.fetch_statements(company, max_years=max_years)

    if history is None:
        return None

    market.attach_market_data(history, company)
    return history
