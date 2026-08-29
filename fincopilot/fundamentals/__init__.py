"""Audited financial statements and live market data."""

from __future__ import annotations

import logging

from ..resolve import Company
from .models import FinancialHistory, FiscalYear
from .recency import Recency, assess as assess_recency, combine

log = logging.getLogger(__name__)

__all__ = [
    "FinancialHistory", "FiscalYear", "Recency",
    "load_financials", "assess_recency",
]


def _from_results_pdf(company: Company, ingest) -> FinancialHistory | None:
    """Read the newest audited annual results PDF that was already downloaded.

    Only annual filings are considered — a quarterly results PDF carries the
    same layout and would yield a quarter dressed as a year. Documents are
    tried newest first and the first that extracts cleanly wins, since a filing
    also restates its prior year and so covers two.
    """
    from . import results_pdf

    candidates = [
        document for document in getattr(ingest, "accepted", [])
        if document.local_path
        and (document.fiscal_period or "").upper() in ("FY", "", "None")
        and document.doc_type in ("earnings_release", "annual_report")
    ]
    candidates.sort(key=lambda d: d.fiscal_year or 0, reverse=True)

    # Every filing of this kind carries the current year AND its restated prior
    # year, so successive filings overlap by one. Merging across them is what
    # lets the document window be short without the financial history becoming
    # short with it: two annual filings give three fiscal years, three give
    # four. A single filing would give only two, which is barely enough to
    # compute one growth rate.
    merged: dict[int, object] = {}
    used: list[str] = []

    for document in candidates[:4]:
        history = results_pdf.extract(
            document.local_path,
            doc_label=document.label,
            ticker=company.ticker,
            company_name=company.name,
            currency=company.currency,
        )
        if history is None or not history.years:
            continue
        used.append(document.label)
        for year in history.years:
            # Filings are walked newest first, and a year already taken from a
            # newer filing is kept: that copy is the restated one, and the
            # restatement is the figure the company now stands behind.
            merged.setdefault(year.fiscal_year, year)

    if not merged:
        return None

    combined = FinancialHistory(
        ticker=company.ticker, company_name=company.name,
        currency=company.currency, source=results_pdf.SOURCE,
    )
    combined.years = [merged[y] for y in sorted(merged)]
    combined.notes.append(
        f"Audited consolidated figures extracted from {len(used)} results "
        f"filing(s): {', '.join(used)}. Where filings overlap, the restated "
        f"figure from the more recent one is used."
    )
    log.info(
        "read audited figures for %s from %d filing(s): FY%s",
        company.ticker, len(used), [y.fiscal_year for y in combined.years],
    )
    return combined


def load_financials(
    company: Company, *, max_years: int = 6, ingest=None
) -> FinancialHistory | None:
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
    from .recency import freshest

    history = xbrl.fetch(company, max_years=max_years)

    if history is None:
        history = indas.fetch(company, max_years=max_years)

        # The Ind-AS results-XBRL endpoint retains roughly three years, so for
        # many Indian issuers it stops one or two years short of what has
        # actually been filed. The audited results PDFs go further, and their
        # SEBI-mandated layout is regular enough to read reliably — see
        # results_pdf.py for the safeguards. Whichever source carries the newer
        # audited year wins: an older source is not preferable for being
        # tidier when the difference is a year of reported results.
        if ingest is not None:
            from_pdf = _from_results_pdf(company, ingest)
            # Combined rather than chosen between: the XBRL endpoint and the
            # results filings cover different, largely non-overlapping spans,
            # and taking only one leaves the DCF with two fiscal years and a
            # single growth rate. Shared years are cross-checked.
            history = combine(history, from_pdf) or history

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
