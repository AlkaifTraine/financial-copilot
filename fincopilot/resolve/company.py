"""
Resolve a free-text company name to a precise, tradeable identity.

Everything downstream keys off the result of this module: which data source can
be used (SEC EDGAR needs a CIK), which currency the statements are in, which
tax rate and equity risk premium the valuation applies, and which folder the
documents cache into. Getting it wrong silently poisons the entire run.

Why scoring rather than "take the first match":

    yahooquery's search for "TCS" returns, in order:
        0221.KL   TCS Group Holdings Berhad      (Malaysia)
        002100.SZ TECON BIOLOGY Co.LTD           (Shenzhen)
        TCS.NS    Tata Consultancy Services Ltd  (India)  <- the intended one

    Naively taking the first equity match indexes a Malaysian company under the
    name "TCS". The candidates are therefore scored on exchange quality, exact
    ticker match, and name similarity.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import asdict, dataclass

from .. import config
from ..http_client import get_json_cached

log = logging.getLogger(__name__)

SEC_TICKER_MAP_URL = f"{config.SEC_BASE_URL}/files/company_tickers.json"


# ---------------------------------------------------------------------------
# Exchange reference data
# ---------------------------------------------------------------------------
# priority breaks ties between multiple listings of the same company: a primary
# domestic listing beats a foreign cross-listing of the same issuer.
#                    country, currency, priority
_EXCHANGE_META: dict[str, tuple[str, str, int]] = {
    # United States
    "NMS": ("US", "USD", 100),   # Nasdaq Global Select
    "NYQ": ("US", "USD", 100),   # NYSE
    "NGM": ("US", "USD", 95),    # Nasdaq Global
    "NCM": ("US", "USD", 90),    # Nasdaq Capital
    "ASE": ("US", "USD", 80),    # NYSE American
    "PCX": ("US", "USD", 70),    # NYSE Arca
    "BTS": ("US", "USD", 60),    # BATS
    "PNK": ("US", "USD", 20),    # OTC Pink — thin, unreliable
    # India
    "NSI": ("IN", "INR", 100),   # NSE
    "BSE": ("IN", "INR", 90),    # BSE
    # Other majors, cross-listings mostly
    "LSE": ("GB", "GBP", 55),
    "TOR": ("CA", "CAD", 50),
    "GER": ("DE", "EUR", 30),
    "FRA": ("DE", "EUR", 25),
    "STU": ("DE", "EUR", 20),
    "HKG": ("HK", "HKD", 50),
    "TYO": ("JP", "JPY", 50),
    "KLS": ("MY", "MYR", 25),
    "SHZ": ("CN", "CNY", 25),
    "SHH": ("CN", "CNY", 25),
}

_DEFAULT_EXCHANGE_META = ("XX", "USD", 10)


@dataclass(frozen=True)
class Company:
    """A resolved, unambiguous company identity."""

    query: str            # what the user typed
    name: str             # official long name
    ticker: str           # full symbol, e.g. "NVDA" or "TCS.NS"
    exchange: str         # exchange code, e.g. "NMS"
    country: str          # ISO-2, drives tax rate and risk premium
    currency: str         # reporting currency of market data
    cik: str | None       # 10-digit zero-padded SEC CIK; None for non-filers
    sector: str | None
    industry: str | None

    @property
    def is_sec_filer(self) -> bool:
        """Whether SEC EDGAR is available as the primary document source."""
        return self.cik is not None

    @property
    def slug(self) -> str:
        """Filesystem-safe key for caches.

        Derived from the ticker, not the user's query, so that "NVIDIA",
        "nvidia" and "NVDA" all resolve to a single shared cache entry.
        """
        return re.sub(r"[^a-z0-9]+", "_", self.ticker.lower()).strip("_")

    @property
    def base_ticker(self) -> str:
        """Symbol without its exchange suffix: 'TCS.NS' -> 'TCS'."""
        return self.ticker.split(".")[0]

    def to_dict(self) -> dict:
        return {**asdict(self), "slug": self.slug, "is_sec_filer": self.is_sec_filer}


# ---------------------------------------------------------------------------
# SEC ticker -> CIK map
# ---------------------------------------------------------------------------

def _load_sec_ticker_map() -> dict[str, str]:
    """Map of uppercase ticker -> zero-padded 10-digit CIK.

    The SEC publishes ~10,400 entries covering every ticker that files with it,
    including foreign private issuers filing 20-F (Infosys, for example).
    Cached on disk for 24 hours.
    """
    payload = get_json_cached(SEC_TICKER_MAP_URL, sec=True)
    if not payload:
        log.warning("could not load the SEC ticker map; CIK lookup unavailable")
        return {}

    return {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in payload.values()
        if entry.get("ticker")
    }


def _lookup_cik(ticker: str, country: str) -> str | None:
    """Find the SEC CIK for a ticker, but only for US-listed securities.

    The country guard is essential and not merely an optimisation. Ticker
    symbols are only unique within an exchange: "TCS" is Tata Consultancy
    Services on the NSE, and was The Container Store Group on the NYSE. Looking
    up a bare Indian ticker in the SEC map would confidently return the CIK of
    an unrelated American retailer.
    """
    if country != "US":
        return None
    return _load_sec_ticker_map().get(ticker.split(".")[0].upper())


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _name_similarity(query: str, name: str) -> float:
    """Similarity in [0, 1] between the user's query and a candidate's name."""
    query_l = query.lower().strip()
    name_l = (name or "").lower().strip()
    if not name_l:
        return 0.0

    ratio = difflib.SequenceMatcher(None, query_l, name_l).ratio()

    # A short query that is a clean prefix of the name ("apple" -> "Apple Inc.")
    # scores poorly on raw ratio because of the length mismatch, so reward
    # whole-word containment explicitly.
    query_tokens = set(re.findall(r"[a-z0-9]+", query_l))
    name_tokens = set(re.findall(r"[a-z0-9]+", name_l))
    if query_tokens and query_tokens <= name_tokens:
        ratio = max(ratio, 0.85)

    return ratio


def _score_candidate(query: str, quote: dict) -> float:
    symbol = (quote.get("symbol") or "").upper()
    base = symbol.split(".")[0]
    exchange = (quote.get("exchange") or "").upper()
    name = quote.get("longname") or quote.get("shortname") or ""

    _country, _currency, exchange_priority = _EXCHANGE_META.get(
        exchange, _DEFAULT_EXCHANGE_META
    )

    score = float(exchange_priority)

    # The user typing an exact ticker is the strongest possible signal.
    if base == query.upper().strip():
        score += 200.0

    score += 150.0 * _name_similarity(query, name)

    return score


def _quote_to_company(query: str, quote: dict) -> Company:
    symbol = (quote.get("symbol") or "").upper()
    exchange = (quote.get("exchange") or "").upper()
    country, currency, _priority = _EXCHANGE_META.get(exchange, _DEFAULT_EXCHANGE_META)

    return Company(
        query=query,
        name=quote.get("longname") or quote.get("shortname") or symbol,
        ticker=symbol,
        exchange=exchange,
        country=country,
        currency=currency,
        cik=_lookup_cik(symbol, country),
        sector=quote.get("sectorDisp") or quote.get("sector"),
        industry=quote.get("industryDisp") or quote.get("industry"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_company(query: str, *, max_candidates: int = 10) -> Company:
    """Resolve free text to a :class:`Company`.

    Raises:
        LookupError: if no listed equity can be matched to the query.
    """
    query = (query or "").strip()
    if not query:
        raise LookupError("Please enter a company name or ticker.")

    from yahooquery import search  # imported lazily; it is slow to load

    try:
        results = search(query)
    except Exception as exc:  # network flake, upstream schema change, ...
        raise LookupError(f"Company lookup failed for {query!r}: {exc}") from exc

    quotes = [
        q
        for q in (results or {}).get("quotes", [])[:max_candidates]
        if q.get("quoteType") == "EQUITY" and q.get("symbol")
    ]
    if not quotes:
        raise LookupError(
            f"No listed company found for {query!r}. "
            f"Try the full legal name or the ticker symbol."
        )

    ranked = sorted(quotes, key=lambda q: _score_candidate(query, q), reverse=True)

    if log.isEnabledFor(logging.DEBUG):
        for quote in ranked:
            log.debug(
                "  candidate %-12s %-5s score=%6.1f  %s",
                quote.get("symbol"),
                quote.get("exchange"),
                _score_candidate(query, quote),
                quote.get("longname"),
            )

    return _quote_to_company(query, ranked[0])
