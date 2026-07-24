"""
SEC EDGAR as the primary document source for US filers.

Why EDGAR is preferred over web search wherever it is available:

  * The filing is the regulator's own copy — no mirror, no paywall, no risk of
    an altered or truncated document.
  * URLs are permanent and auditable, which is exactly the provenance a
    finance user needs before trusting a generated number.
  * Filings are served as HTML. HTML tables survive extraction with their row
    and column structure intact, whereas the same table in a PDF has to be
    reconstructed from text coordinates and frequently is not recoverable.
  * The same accession numbers key into the XBRL `companyfacts` API, which is
    where the valuation's audited inputs come from.

EDGAR is used for filings; glossy investor-relations PDFs and slide decks come
from the web-search path instead, since they are not filed with the SEC.
"""

from __future__ import annotations

import logging
import re

from .. import config
from ..http_client import get_json_cached, sec_get
from ..resolve import Company
from .models import ANNUAL, EARNINGS, ORIGIN_EDGAR, QUARTERLY, SourceDocument

log = logging.getLogger(__name__)


# Form type -> our document taxonomy.
#   10-K / 20-F / 40-F  annual reports (domestic, foreign private issuer, Canada)
#   10-Q                quarterly reports
#   8-K / 6-K           current reports; only earnings-bearing ones are kept
_FORM_MAP = {
    "10-K": ANNUAL,
    "10-K/A": ANNUAL,
    "20-F": ANNUAL,
    "40-F": ANNUAL,
    "10-Q": QUARTERLY,
    "10-Q/A": QUARTERLY,
    "8-K": EARNINGS,
    "6-K": EARNINGS,
}

# 8-K item 2.02 is "Results of Operations and Financial Condition" — the item
# under which quarterly earnings are released. Without this filter an 8-K sweep
# returns director appointments and bylaw amendments, which are noise here.
_EARNINGS_ITEM = "2.02"


def _submissions_url(cik: str) -> str:
    return f"{config.SEC_DATA_URL}/submissions/CIK{cik}.json"


def _archive_url(cik: str, accession: str, document: str) -> str:
    """Build the canonical URL for a filing's primary document."""
    return (
        f"{config.SEC_BASE_URL}/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{document}"
    )


def _earnings_exhibit_url(cik: str, accession: str) -> str | None:
    """Locate the press release exhibit inside an 8-K/6-K filing.

    An 8-K's ``primaryDocument`` is only the cover page: a few hundred words
    stating that a press release is attached. The release itself — with the
    revenue, margin and guidance figures — is filed as exhibit EX-99.1.

    Fetching the cover page instead of the exhibit is why every NVIDIA earnings
    8-K was discarded by the relevance gate as "not a financial document": the
    gate was right, the document simply had no content in it.

    Exhibits are identified by EDGAR's declared exhibit *type*, not by
    filename. Filenames are chosen by the filer and follow no convention:
    NVIDIA names its Q1 FY27 release ``q1fy27pr.htm``, while other issuers use
    ``ex-99_1.htm``. Matching on the filename works for one company and
    silently fails for the next; the ``Type`` column on the filing index is
    authoritative for all of them.
    """
    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    index_url = (
        f"{config.SEC_BASE_URL}/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{accession}-index.html"
    )
    response = sec_get(index_url)
    if response is None or response.status_code != 200:
        return None

    try:
        soup = BeautifulSoup(response.text, "lxml")
    except Exception:
        return None

    candidates: list[tuple[int, str]] = []
    for row in soup.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in row.find_all("td")]
        if not cells:
            continue

        match = next(
            (re.fullmatch(r"EX-99(?:\.(\d+))?", cell, re.I) for cell in cells
             if re.fullmatch(r"EX-99(?:\.(\d+))?", cell, re.I)),
            None,
        )
        if not match:
            continue

        link = row.find("a", href=True)
        if not link or not link["href"].lower().endswith((".htm", ".html")):
            continue

        # EX-99.1 is conventionally the earnings release itself; higher
        # numbers are supporting schedules and CFO commentary.
        candidates.append((int(match.group(1) or 1), urljoin(config.SEC_BASE_URL, link["href"])))

    if not candidates:
        return None

    candidates.sort()
    return candidates[0][1]


def _fiscal_period(
    form: str,
    report_date: str | None,
    fiscal_year_end: str | None,
) -> tuple[int | None, str | None]:
    """Derive fiscal year and quarter from a filing's report date.

    Fiscal calendars are not calendar calendars. NVIDIA's fiscal year ends in
    late January, so its quarter ending 26 April 2026 is **Q1 FY2027**, not
    "Q2 2026" as naive calendar arithmetic gives. Mislabelling that propagates
    into retrieval metadata filters and into the report, where it would present
    a quarter's results under the wrong year.

    Args:
        fiscal_year_end: EDGAR's ``fiscalYearEnd`` field, "MMDD" (e.g. "0125").
    """
    if not report_date or len(report_date) < 7:
        return None, None

    year = int(report_date[:4])
    month = int(report_date[5:7])

    try:
        fye_month = int((fiscal_year_end or "1231")[:2])
    except ValueError:
        fye_month = 12
    if not 1 <= fye_month <= 12:
        fye_month = 12

    # An 8-K's reportDate is the announcement date, which falls a few weeks
    # after the quarter it discusses actually closed. Step back so the release
    # is attributed to the period it reports on.
    if form in ("8-K", "6-K"):
        month -= 1
        if month == 0:
            month, year = 12, year - 1

    # Fiscal years are labelled by the calendar year in which they end, so any
    # month past the fiscal year end already belongs to the next fiscal year.
    fiscal_year = year + 1 if month > fye_month else year

    if form.startswith(("10-K", "20-F", "40-F")):
        return fiscal_year, "FY"

    months_into_year = (month - fye_month - 1) % 12
    return fiscal_year, f"Q{months_into_year // 3 + 1}"


def discover(company: Company, *, max_per_type: int | None = None) -> list[SourceDocument]:
    """Return recent EDGAR filings for ``company``, newest first.

    Returns an empty list — never raises — when EDGAR is unconfigured, the
    company is not an SEC filer, or the API is unreachable. The caller then
    falls back to web search.
    """
    if not company.is_sec_filer:
        log.info("%s is not an SEC filer; skipping EDGAR", company.ticker)
        return []

    if not config.is_sec_configured():
        log.warning(
            "SEC_USER_AGENT is not configured with a contact email, so EDGAR "
            "would return 403. Skipping EDGAR and relying on web search."
        )
        return []

    payload = get_json_cached(_submissions_url(company.cik), sec=True, ttl_seconds=21_600)
    if not payload:
        log.warning("EDGAR submissions unavailable for CIK %s", company.cik)
        return []

    fiscal_year_end = payload.get("fiscalYearEnd")  # "MMDD"
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    if not forms:
        return []

    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    items = recent.get("items", [])

    def at(seq: list, i: int):
        return seq[i] if i < len(seq) else None

    limits = dict(config.DOC_TYPE_LIMITS)
    if max_per_type is not None:
        limits = {k: max_per_type for k in limits}

    counts: dict[str, int] = {}
    documents: list[SourceDocument] = []

    # `recent` is ordered newest-first, so a simple forward scan with per-type
    # counters yields the most recent N of each kind.
    for i, form in enumerate(forms):
        doc_type = _FORM_MAP.get(form)
        if not doc_type:
            continue

        # Keep only earnings-bearing current reports.
        if form in ("8-K", "6-K") and _EARNINGS_ITEM not in (at(items, i) or ""):
            continue

        if counts.get(doc_type, 0) >= limits.get(doc_type, 0):
            continue

        primary = at(primary_docs, i)
        accession = at(accessions, i)
        if not primary or not accession:
            continue

        # For current reports, swap the cover page for the actual press release.
        if form in ("8-K", "6-K"):
            exhibit = _earnings_exhibit_url(company.cik, accession)
            if not exhibit:
                continue  # no release attached; nothing worth indexing
            document_url = exhibit
        else:
            document_url = _archive_url(company.cik, accession, primary)

        fiscal_year, fiscal_period = _fiscal_period(
            form, at(report_dates, i), fiscal_year_end
        )

        if fiscal_year and fiscal_year < config.current_year() - config.MAX_DOCUMENT_AGE_YEARS:
            continue

        documents.append(
            SourceDocument(
                doc_type=doc_type,
                title=f"{company.name} {form}",
                url=document_url,
                origin=ORIGIN_EDGAR,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                filed_date=at(filing_dates, i),
                form_type=form,
                accession=accession,
                content_type=(
                    "html" if document_url.lower().endswith((".htm", ".html")) else "pdf"
                ),
            )
        )
        counts[doc_type] = counts.get(doc_type, 0) + 1

    log.info("EDGAR returned %d filings for %s", len(documents), company.ticker)
    return documents
