"""
Official filings for Indian issuers, straight from the exchange.

For an SEC filer, EDGAR is the authoritative source and web search is a
convenience. For an Indian issuer there was, until this module, no
authoritative source at all: discovery ran entirely through DuckDuckGo, so the
document set depended on what a search engine happened to surface that day.
That is acceptable for finding a glossy investor deck and unacceptable as the
provenance of a valuation.

The NSE publishes two endpoints that solve this:

``/api/annual-reports``
    Every annual report the company has filed, as a PDF on ``nsearchives``.

``/api/corporates-financial-results``
    Every quarterly and annual results filing — and, crucially, a link to the
    **Ind-AS XBRL instance** for each one. That is structured, concept-tagged,
    audited data, the Indian equivalent of SEC ``companyfacts``, and it is what
    :mod:`fincopilot.fundamentals.indas` reads the numbers out of.

Each results record also carries two flags that remove the two most dangerous
ambiguities in Indian reporting:

``consolidated``  "Consolidated" vs "Non-Consolidated" — a standalone statement
                  excludes subsidiaries and can understate a group materially.
``audited``       "Audited" vs "Un-Audited" — only audited figures should reach
                  a valuation.

Both are filtered on here rather than inferred from the document later.

Access note: ``www.nseindia.com`` returns 403 to a bare client on the HTML
homepage, but serves the JSON APIs to any request carrying ordinary browser
headers and a Referer. No cookie priming or key is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from ..http_client import get_json_cached

log = logging.getLogger(__name__)

_API = "https://www.nseindia.com/api"

# NSE serves the JSON APIs to a normal browser-shaped request. The Referer is
# the part it actually checks; without it the endpoints 403.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Results filings change only when a company reports; a day is plenty.
_TTL_SECONDS = 21_600

CONSOLIDATED = "Consolidated"
AUDITED = "Audited"


@dataclass(frozen=True)
class ResultsFiling:
    """One results filing announced to the exchange."""

    symbol: str
    period_end: str                # ISO date, parsed from NSE's "31-Mar-2024"
    relating_to: str               # "Annual", "Fourth Quarter", ...
    consolidated: bool
    audited: bool
    xbrl_url: str | None
    filing_date: str | None = None

    @property
    def is_annual(self) -> bool:
        """Whether this filing carries a full audited year.

        Both the explicit "Annual" filing and the fourth-quarter filing report
        year-ended columns; either is a valid source for a fiscal year. The
        period covered is confirmed from the XBRL itself, not from this flag.
        """
        return self.relating_to in ("Annual", "Fourth Quarter")


def _parse_date(value: str | None) -> str | None:
    """NSE's "31-Mar-2024" -> "2024-03-31". Returns None if unparseable."""
    if not value:
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip()[:11], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _clean_url(value: str | None) -> str | None:
    """A usable http(s) URL, or None for NSE's empty placeholders."""
    text = (value or "").strip()
    if not text.lower().startswith("http"):
        return None
    return text


def _get(path: str) -> list | dict | None:
    payload = get_json_cached(
        f"{_API}/{path}", ttl_seconds=_TTL_SECONDS, headers=_HEADERS
    )
    if payload is None:
        log.warning("NSE endpoint unavailable: %s", path)
    return payload


def results_filings(symbol: str) -> list[ResultsFiling]:
    """Every consolidated, audited results filing NSE lists for ``symbol``.

    Both the Annual and Quarterly feeds are queried and merged: NSE files the
    audited year under either heading depending on the company, and the two
    feeds overlap rather than partition. De-duplicated on (period_end,
    relating_to), preferring a record that actually carries an XBRL link.
    """
    merged: dict[tuple[str, str], ResultsFiling] = {}

    for period in ("Annual", "Quarterly"):
        payload = _get(
            f"corporates-financial-results?index=equities"
            f"&symbol={symbol}&period={period}"
        )
        if not isinstance(payload, list):
            continue

        for record in payload:
            period_end = _parse_date(record.get("toDate"))
            if not period_end:
                continue

            filing = ResultsFiling(
                symbol=symbol,
                period_end=period_end,
                relating_to=(record.get("relatingTo") or "").strip(),
                consolidated=record.get("consolidated") == CONSOLIDATED,
                audited=record.get("audited") == AUDITED,
                # NSE writes a bare "-" where no XBRL was filed; that would
                # otherwise become a valid-looking ".../xbrl/-" URL.
                xbrl_url=_clean_url(record.get("xbrl")),
                filing_date=record.get("filingDate"),
            )

            key = (filing.period_end, filing.relating_to)
            existing = merged.get(key)
            # Prefer consolidated over standalone, then a record with XBRL.
            if existing is None:
                merged[key] = filing
            elif not existing.consolidated and filing.consolidated:
                merged[key] = filing
            elif existing.xbrl_url is None and filing.xbrl_url:
                merged[key] = filing

    filings = sorted(merged.values(), key=lambda f: f.period_end, reverse=True)
    log.info("NSE lists %d results filings for %s", len(filings), symbol)
    return filings


def annual_xbrl_filings(symbol: str) -> list[ResultsFiling]:
    """Audited annual filings that have an XBRL instance, best first.

    Unaudited and PDF-only filings are dropped here rather than downstream: a
    valuation must not be built on an unaudited statement, and there is no way
    to tell from the numbers alone.

    Consolidated is *preferred* but standalone is not excluded. A group's
    standalone statement omits its subsidiaries and would understate it — but
    a company with no material subsidiaries files standalone only, and that
    statement is complete and correct for it. Nestle India is a real example:
    every one of its annual filings is standalone, so a consolidated-only rule
    refuses a company it should value. Ordering puts consolidated first for
    each period so the caller takes it whenever both exist.
    """
    filings = [
        f
        for f in results_filings(symbol)
        if f.is_annual and f.audited and f.xbrl_url
    ]
    filings.sort(key=lambda f: (f.period_end, f.consolidated), reverse=True)
    return filings


def annual_report_pdfs(symbol: str, *, limit: int = 6) -> list[dict]:
    """Annual-report PDFs NSE holds for ``symbol``, newest first.

    Returned as plain dicts ({url, from_year, to_year}) for the ingest
    pipeline to turn into :class:`SourceDocument` candidates. These feed
    retrieval and the narrative sections; the *numbers* come from XBRL.
    """
    payload = _get(f"annual-reports?index=equities&symbol={symbol}")
    records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []

    out: list[dict] = []
    for record in records:
        url = (record.get("fileName") or "").strip()
        if not url.lower().endswith(".pdf"):
            continue
        out.append(
            {
                "url": url,
                "from_year": record.get("fromYr"),
                "to_year": record.get("toYr"),
            }
        )

    out.sort(key=lambda r: str(r.get("to_year") or ""), reverse=True)
    return out[:limit]
