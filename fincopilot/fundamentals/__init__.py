"""Audited financial statements and live market data."""

from __future__ import annotations

import logging

from ..resolve import Company
from .models import FinancialHistory, FiscalYear
from .recency import Recency, assess as assess_recency

log = logging.getLogger(__name__)

__all__ = [
    "FinancialHistory", "FiscalYear", "Recency",
    "load_financials", "assess_recency",
]


def load_financials(company: Company, *, max_years: int = 6) -> FinancialHistory | None:
    """Load the audited financial history for ``company``.

    Statement figures come only from the company's own regulatory filings:

    * **SEC XBRL** (`companyfacts`) for SEC filers, and
    * **Ind-AS XBRL** filed with the NSE for Indian issuers.

    Both are concept-tagged, audited, and traceable to a specific filing.
    There is deliberately **no fallback to a market-data vendor for statement
    figures**: a vendor's numbers are unattributable, silently restated, and
    cannot be tied to a filing, which is precisely the provenance a valuation
    depends on. When no audited source is available this returns ``None`` and
    the caller must decline to value the company rather than proceed on
    figures it cannot stand behind.

    Live market data — share price, share count, beta, analyst targets — is a
    separate concern and does come from yfinance: it is a quote, not a
    reported figure, and no filing contains it.
    """
    from . import indas, market, xbrl

    history = xbrl.fetch(company, max_years=max_years)

    if history is None:
        history = indas.fetch(company, max_years=max_years)

    if history is None:
        log.warning(
            "no audited filing-derived financials for %s; declining to value it",
            company.ticker,
        )
        return None

    # Recency is attached here, at the single point where a history is built,
    # so no caller can obtain one without also being told how old it is. The
    # failure this prevents: a report dated today leading with FY2024 figures
    # because the XBRL endpoint's retention window ended there, while the
    # annual reports in the same index ran two years further.
    history.recency = assess_recency(history)
    if not history.recency.is_current:
        history.notes.append(history.recency.summary)

    market.attach_market_data(history, company)
    return history
