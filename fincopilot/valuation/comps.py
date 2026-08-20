"""
Comparable company analysis.

A DCF says what the business is worth on its own assumptions; comps say what
the market currently pays for similar businesses. They answer different
questions, and disagreement between them is informative rather than a problem.

Peer selection is the weak point of any comps analysis, and there is no free
API that returns a defensible peer set. Rather than invent one, this module
uses a small curated map of well-known peer groups and **declines to produce
comps at all** when the company is not covered. An arbitrary peer set produces
an arbitrary valuation with a misleading air of objectivity.
"""

from __future__ import annotations

import logging

from .models import CompsResult, PeerMultiple

log = logging.getLogger(__name__)

# Curated peer groups. Deliberately narrow: every entry is a set a practitioner
# would actually accept as comparable. Each peer is tagged DIRECT (same business
# model — a like-for-like multiple comparison) or ADJACENT (strategically relevant
# but STRUCTURALLY different, e.g. a foundry versus a fabless designer), with the
# reason it is included. Only DIRECT peers feed the median; adjacent names are
# shown for context but would distort a like-for-like multiple.
#
# Entries are either a bare ticker (DIRECT, generic sector-peer rationale) or a
# (ticker, tier, rationale) tuple where the distinction matters.
_PEER_GROUPS: dict[str, list] = {
    "NVDA": [
        ("AMD", "direct", "Fabless designer competing directly in GPUs and AI accelerators"),
        ("AVGO", "direct", "Fabless designer of AI/networking silicon; direct AI-infrastructure competitor"),
        ("QCOM", "direct", "Fabless designer of advanced SoCs; same fabless, high-margin economics"),
        ("INTC", "direct", "Designs and sells competing datacenter CPUs and accelerators"),
        ("TSM", "adjacent", "Foundry that MANUFACTURES chips rather than designing them — "
                            "strategically vital to NVIDIA but a structurally different, far more "
                            "capital-intensive and lower-margin business; shown for context, "
                            "excluded from the median"),
    ],
    "AMD": ["NVDA", "AVGO", "QCOM", ("INTC", "direct", "Competing x86 CPU and datacenter vendor")],
    "AVGO": ["NVDA", "QCOM", "TXN", "AMD"],
    "INTC": ["AMD", "NVDA", "TXN", "MU"],
    "MSFT": ["GOOGL", "AMZN", "ORCL", "CRM", "AAPL"],
    "GOOGL": ["MSFT", "META", "AMZN"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "AAPL": ["MSFT", "GOOGL", "SONY", "DELL"],
    "AMZN": ["MSFT", "GOOGL", "WMT"],
    "ORCL": ["MSFT", "SAP", "CRM", "IBM"],
    "CRM": ["MSFT", "ORCL", "NOW", "WDAY"],
    "TCS.NS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "INFY.NS": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS"],
    "INFY": ["WIT", "ACN", "CTSH"],
    "WIPRO.NS": ["TCS.NS", "INFY.NS", "HCLTECH.NS"],
    "RELIANCE.NS": ["ONGC.NS", "BPCL.NS", "IOC.NS"],
}


def _normalize_peer(entry) -> tuple[str, str, str]:
    """(ticker, tier, rationale) from a bare ticker or an annotated tuple."""
    if isinstance(entry, tuple):
        ticker, tier, rationale = entry
        return ticker, tier, rationale
    return entry, "direct", "Sector peer with a comparable business model"


def _fetch_multiple(ticker: str) -> PeerMultiple | None:
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        log.info("peer data unavailable for %s: %s", ticker, exc)
        return None

    if not info.get("marketCap"):
        return None

    return PeerMultiple(
        ticker=ticker,
        name=info.get("longName") or ticker,
        market_cap=info.get("marketCap"),
        pe=info.get("trailingPE"),
        ev_to_ebitda=info.get("enterpriseToEbitda"),
        ev_to_sales=info.get("enterpriseToRevenue"),
        revenue_growth=info.get("revenueGrowth"),
    )


def _median(values: list[float]) -> float | None:
    clean = sorted(v for v in values if v is not None and v > 0)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2


def build_comps(
    ticker: str,
    *,
    net_income: float | None = None,
    shares_outstanding: float | None = None,
) -> CompsResult:
    """Peer multiples and the value they imply for ``ticker``."""
    result = CompsResult()

    peers = _PEER_GROUPS.get(ticker.upper())
    if not peers:
        result.notes.append(
            f"No curated peer group is defined for {ticker}, so comparable "
            f"company analysis was skipped. Selecting peers automatically "
            f"would produce an arbitrary result presented as an objective one."
        )
        return result

    for entry in peers:
        peer_ticker, tier, rationale = _normalize_peer(entry)
        multiple = _fetch_multiple(peer_ticker)
        if multiple:
            multiple.tier = tier
            multiple.rationale = rationale
            multiple.in_median = tier == "direct"
            result.peers.append(multiple)

    if not result.peers:
        result.notes.append("Peer market data could not be retrieved.")
        return result

    # Only DIRECT peers set the multiple. A structurally different peer (a foundry
    # against a fabless designer) would distort a like-for-like median, so it is
    # shown for context but kept out of the number.
    median_peers = [p for p in result.peers if p.in_median] or result.peers
    result.median_pe = _median([p.pe for p in median_peers])
    result.median_ev_sales = _median([p.ev_to_sales for p in median_peers])

    direct_names = ", ".join(p.ticker for p in median_peers)
    adjacent = [p for p in result.peers if not p.in_median]
    selection = f"Peer median is taken over the DIRECT peers ({direct_names})."
    if adjacent:
        selection += (
            " Shown but excluded as structurally different: "
            + ", ".join(f"{p.ticker} ({p.rationale.split(';')[0].split('—')[0].strip()})" for p in adjacent)
            + "."
        )
    result.notes.append(selection)

    if result.median_pe and net_income and shares_outstanding:
        eps = net_income / shares_outstanding
        result.implied_value_per_share = result.median_pe * eps
        result.notes.append(
            f"Implied value applies the direct-peer median P/E of {result.median_pe:.1f}x "
            f"to trailing EPS of {eps:.2f}."
        )

    return result
