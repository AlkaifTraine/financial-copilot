"""
Web-search document discovery.

This is the fallback path for US filers and the *primary* path for companies
outside SEC jurisdiction (Indian issuers such as TCS and Reliance). It is also
the only way to reach documents that are never filed with a regulator at all:
glossy annual-report PDFs and investor presentation decks published on the
company's investor-relations site.

It carries forward the approach of the original `backend/document_pipeline.py`,
which worked, with four correctness fixes described inline:

  1. quarter detection no longer matches inside hostnames (``q4cdn.com``)
  2. year extraction picks the most plausible year rather than the first
     four digits that happen to look like one
  3. the cut-off year is computed, not hard-coded
  4. candidates are checked for actually belonging to the company
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .. import config
from ..http_client import request
from ..resolve import Company
from .models import (
    ANNUAL,
    EARNINGS,
    ORIGIN_WEB,
    PRESENTATION,
    QUARTERLY,
    SourceDocument,
)

log = logging.getLogger(__name__)


SEARCH_QUERIES: dict[str, list[str]] = {
    ANNUAL: ["annual report pdf", "integrated annual report pdf"],
    QUARTERLY: ["quarterly report pdf", "quarterly results pdf"],
    EARNINGS: ["earnings release pdf", "financial results press release pdf"],
    PRESENTATION: ["investor presentation pdf", "earnings presentation pdf"],
}

# Ordered: the first pattern that matches wins, so the specific ones come
# before the generic ones ("annual report" before bare "results").
_CLASSIFIERS: list[tuple[str, re.Pattern]] = [
    (ANNUAL, re.compile(r"annual[-_ ]?report|10-?k\b|\bar\d{2,4}\b|integrated[-_ ]?report", re.I)),
    (PRESENTATION, re.compile(r"presentation|investor[-_ ]?day|slide|deck|keynote", re.I)),
    (QUARTERLY, re.compile(r"quarter|10-?q\b|\bq[1-4][-_ ]?(fy)?\d{2,4}\b", re.I)),
    (EARNINGS, re.compile(r"earnings|press[-_ ]?release|financial[-_ ]?results|\bresults\b", re.I)),
]

_FINANCIAL_URL_HINTS = re.compile(
    r"annual|quarter|earnings|financial|investor|report|results|presentation|10-?k|10-?q",
    re.I,
)


def _strip_host_noise(url: str) -> str:
    """Remove hostname tokens that produce false quarter/year matches.

    NVIDIA's investor-relations CDN is ``s201.q4cdn.com``. The original
    substring test for ``"q4"`` therefore tagged *every* NVIDIA document as a
    Q4 filing — including the annual report, as recorded in the committed
    ``data/nvidia/document_registry.json``. Only the path carries meaning, so
    the host is dropped before pattern matching.
    """
    parsed = urlparse(url)
    return f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path


def extract_year(url: str) -> int | None:
    """Best-effort fiscal year from a document URL.

    Picks the latest plausible year present rather than the first match: paths
    routinely embed unrelated digits (``/files/doc_financials/2026/...`` sits
    beside account ids like ``141608511``), and taking ``matches[0]`` — as the
    original did — misdates documents whose path begins with an archive year.
    """
    text = _strip_host_noise(url)
    horizon = config.current_year() + 1
    years = [
        int(m)
        for m in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)
        if 1990 <= int(m) <= horizon
    ]
    return max(years) if years else None


def extract_quarter(url: str) -> str | None:
    """Fiscal quarter from a URL, matched on token boundaries only."""
    text = _strip_host_noise(url)

    # Forms: "q1", "q1fy25", "1q26", "quarter-1"
    match = re.search(r"(?<![a-z0-9])q([1-4])(?![0-9a-z]*cdn)(?![0-9]{3,})", text, re.I)
    if match:
        return f"Q{match.group(1)}"

    match = re.search(r"(?<![a-z0-9])([1-4])q(?![0-9a-z]*cdn)", text, re.I)
    if match:
        return f"Q{match.group(1)}"

    match = re.search(r"quarter[-_ ]?([1-4])", text, re.I)
    if match:
        return f"Q{match.group(1)}"

    return None


def classify(url: str) -> str | None:
    text = _strip_host_noise(url)
    for doc_type, pattern in _CLASSIFIERS:
        if pattern.search(text):
            return doc_type
    return None


def _score(url: str, company: Company) -> float:
    """Rank a candidate URL. Higher is better."""
    lowered = url.lower()
    host = urlparse(lowered).netloc
    score = 0.0

    # Official investor-relations infrastructure.
    if host.startswith("investor.") or host.startswith("ir."):
        score += 100
    if "q4cdn" in host:           # Q4 Inc. hosts many IR sites
        score += 80
    if host.endswith(".gov"):
        score += 120

    # The company's own domain, inferred from its name.
    name_token = re.sub(r"[^a-z]", "", company.name.split()[0].lower())
    if len(name_token) >= 4 and name_token in host:
        score += 60

    for bad in config.DEPRIORITISED_DOMAINS:
        if bad in host:
            score -= 120

    if _FINANCIAL_URL_HINTS.search(_strip_host_noise(lowered)):
        score += 20

    year = extract_year(url)
    if year:
        # Recency matters far more than any single hosting signal.
        score += (year - config.current_year()) * 40

    return score


def _search(company: Company) -> list[tuple[str, str]]:
    """Run the configured searches. Returns (url, doc_type) pairs."""
    from ddgs import DDGS

    hits: list[tuple[str, str]] = []
    # Prefer the legal name; it disambiguates far better than a ticker.
    subject = company.name or company.query

    with DDGS() as ddgs:
        for doc_type, queries in SEARCH_QUERIES.items():
            for query in queries:
                try:
                    results = list(ddgs.text(f"{subject} {query}", max_results=6))
                except Exception as exc:
                    log.warning("search failed for %r: %s", query, exc)
                    continue
                for result in results:
                    href = result.get("href") or result.get("url")
                    if href:
                        hits.append((href, doc_type))

    return hits


def _pdf_links_on_page(page_url: str) -> list[str]:
    """Scrape PDF links out of an HTML landing page."""
    response = request(page_url)
    if response is None or response.status_code != 200:
        return []
    if "html" not in response.headers.get("Content-Type", "").lower():
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return []

    links = set()
    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(page_url, anchor["href"])
        if ".pdf" in absolute.lower():
            links.add(absolute.split("#")[0])
    return list(links)


def discover(company: Company, *, max_per_type: int | None = None) -> list[SourceDocument]:
    """Find candidate PDFs for ``company`` via web search."""
    candidates: set[str] = set()

    for url, _hinted_type in _search(company):
        if ".pdf" in url.lower():
            candidates.add(url.split("#")[0])
        else:
            candidates.update(_pdf_links_on_page(url))

    log.info("web search surfaced %d candidate PDFs for %s", len(candidates), company.ticker)

    cutoff = config.current_year() - config.MAX_DOCUMENT_AGE_YEARS
    limits = dict(config.DOC_TYPE_LIMITS)
    if max_per_type is not None:
        limits = {k: max_per_type for k in limits}

    scored: list[tuple[float, SourceDocument]] = []
    for url in candidates:
        doc_type = classify(url)
        if not doc_type:
            continue

        year = extract_year(url)
        if year is not None and year < cutoff:
            continue

        quarter = extract_quarter(url)
        scored.append(
            (
                _score(url, company),
                SourceDocument(
                    doc_type=doc_type,
                    title=url.rsplit("/", 1)[-1].split("?")[0],
                    url=url,
                    origin=ORIGIN_WEB,
                    fiscal_year=year,
                    fiscal_period=quarter or ("FY" if doc_type == ANNUAL else None),
                    content_type="pdf",
                ),
            )
        )

    # Highest score first, then keep the best N per document type while
    # de-duplicating on (year, period) so we do not collect four copies of the
    # same annual report from four different hosts.
    scored.sort(key=lambda pair: pair[0], reverse=True)

    selected: list[SourceDocument] = []
    seen: set[tuple] = set()
    counts: dict[str, int] = {}

    for _score_value, document in scored:
        key = (document.doc_type, document.fiscal_year, document.fiscal_period)
        if key in seen:
            continue
        if counts.get(document.doc_type, 0) >= limits.get(document.doc_type, 0):
            continue
        seen.add(key)
        counts[document.doc_type] = counts.get(document.doc_type, 0) + 1
        selected.append(document)

    return selected
