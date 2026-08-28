"""
Live market data: price, share count, beta, and analyst targets.

One job, deliberately. Share price and share count are what connect a computed
enterprise value to a per-share fair value and to today's quote, and no filing
contains them — a filing is a snapshot of the past, and a quote is now. That is
the whole reason a market-data vendor appears in this project at all.

This module used to have a second job: building *statements* for issuers
outside SEC jurisdiction from yfinance's structured financials. That job has
been removed. Vendor statement data is unattributable — it cannot be traced to
a filing, it is silently restated, and its line-item definitions do not match
the company's own presentation — so it cannot support a valuation that claims
every number is verifiable. Statement figures now come only from the company's
own audited XBRL: SEC ``companyfacts`` for SEC filers
(:mod:`fincopilot.fundamentals.xbrl`), Ind-AS XBRL filed with the exchange for
Indian issuers (:mod:`fincopilot.fundamentals.indas`). Where neither exists,
:func:`fincopilot.fundamentals.load_financials` returns nothing and the company
is not valued.
"""

from __future__ import annotations

import logging
import math

from ..resolve import Company
from .models import FinancialHistory

log = logging.getLogger(__name__)


def _clean(value) -> float | None:
    """Coerce a pandas cell to a usable float, or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


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

    opinion_count = info.get("numberOfAnalystOpinions")
    try:
        opinion_count = int(opinion_count) if opinion_count is not None else None
    except (TypeError, ValueError):
        opinion_count = None

    return {
        "share_price": _clean(price),
        "shares_outstanding": _clean(info.get("sharesOutstanding")),
        "market_cap": _clean(info.get("marketCap")),
        "beta": _clean(info.get("beta")),
        "currency": info.get("currency") or company.currency,
        # Analyst price targets. Pulled from the same .info payload so no extra
        # network round-trip is spent — they ride along with price and beta.
        "analyst_target_mean": _clean(info.get("targetMeanPrice")),
        "analyst_target_median": _clean(info.get("targetMedianPrice")),
        "analyst_target_high": _clean(info.get("targetHighPrice")),
        "analyst_target_low": _clean(info.get("targetLowPrice")),
        "analyst_opinion_count": opinion_count,
    }



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

    history.analyst_target_mean = data.get("analyst_target_mean")
    history.analyst_target_median = data.get("analyst_target_median")
    history.analyst_target_high = data.get("analyst_target_high")
    history.analyst_target_low = data.get("analyst_target_low")
    history.analyst_opinion_count = data.get("analyst_opinion_count")

    # The filing states its own reporting currency (USD for SEC XBRL, INR for
    # Ind-AS), and that is authoritative for the statements. A quote currency
    # from the vendor must never overwrite it: a cross-listing quoted in USD
    # would silently relabel a set of rupee statements as dollars.
    quote_currency = data.get("currency")
    if quote_currency and not history.is_audited_filing:
        history.currency = quote_currency
    elif quote_currency and quote_currency != history.currency:
        history.notes.append(
            f"Quote is in {quote_currency}; statements are reported in "
            f"{history.currency}."
        )
