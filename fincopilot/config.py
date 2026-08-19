"""
Central configuration.

Every tunable knob in the system lives here so that behaviour can be reasoned
about (and defended in an interview) from a single file. Nothing else in the
codebase should hard-code a model name, a threshold, or a path.

Secrets are read from, in order of precedence:
  1. real environment variables
  2. a local `.env` file        (local development)
  3. Streamlit's secrets store  (deployed on Streamlit Community Cloud)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Secret access
# ---------------------------------------------------------------------------

def get_secret(name: str, default: str | None = None) -> str | None:
    """Fetch a secret from the environment, falling back to Streamlit secrets.

    The Streamlit import is deliberately lazy and guarded: this module must stay
    importable from plain CLI scripts (indexing jobs, the eval harness, tests)
    that never start a Streamlit runtime.
    """
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        return st.secrets[name]  # type: ignore[index]
    except Exception:
        return default


def require_secret(name: str) -> str:
    value = get_secret(name)
    if not value:
        raise RuntimeError(
            f"Missing required secret {name!r}. "
            f"Set it in your .env file (local) or in Streamlit secrets (deployed)."
        )
    return value


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"          # downloaded source PDFs, per company
INDEX_DIR = ROOT_DIR / "vector_db"    # FAISS + BM25 indexes, per company
REPORTS_DIR = ROOT_DIR / "reports"    # generated equity research reports
CHARTS_DIR = ROOT_DIR / "charts"      # chart images embedded into reports
CACHE_DIR = ROOT_DIR / ".cache"       # HTTP + embedding caches

for _d in (DATA_DIR, INDEX_DIR, REPORTS_DIR, CHARTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Cheap, fast model used for the high-volume calls: query rewriting, reranking,
# metric extraction. These run many times per user action.
FAST_MODEL = "gpt-4.1-mini"

# Model used for prose a human will actually read end-to-end (report sections,
# assumption rationales). Same model today; split out so it can be upgraded
# independently without touching call sites.
WRITER_MODEL = "gpt-4.1-mini"

# Deterministic by default. Report prose gets a small amount of temperature so
# it does not read like a form letter; anything numeric runs at 0.
TEMPERATURE_FACTUAL = 0.0
TEMPERATURE_PROSE = 0.2

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 128          # OpenAI accepts up to 2048 inputs per call
EMBEDDING_MAX_TOKENS = 8000         # model limit is 8191; leave headroom


# ---------------------------------------------------------------------------
# Document discovery + ingestion
# ---------------------------------------------------------------------------

# The SEC requires a descriptive User-Agent with real contact details on every
# request, and throttles above ~10 requests/second. Violating either gets your
# IP blocked, so both are enforced in ingest/edgar.py.
SEC_USER_AGENT = get_secret(
    "SEC_USER_AGENT",
    "Financial Copilot research tool (contact: set SEC_USER_AGENT in .env)",
)
SEC_MAX_REQUESTS_PER_SECOND = 8.0
SEC_BASE_URL = "https://www.sec.gov"
SEC_DATA_URL = "https://data.sec.gov"


def is_sec_configured() -> bool:
    """Whether SEC EDGAR can be used.

    A User-Agent without real contact details is rejected with HTTP 403 (the
    SEC serves its "Request Rate Threshold Exceeded" page even on the very
    first request). Detecting that here lets the pipeline skip EDGAR and fall
    back to web search with a clear explanation, instead of burning retries on
    requests that can never succeed.
    """
    agent = SEC_USER_AGENT or ""
    return "@" in agent and "REPLACE_WITH_YOUR_EMAIL" not in agent

HTTP_TIMEOUT_SECONDS = 30
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

# How many documents of each type to keep, newest first.
DOC_TYPE_LIMITS = {
    "annual_report": 3,
    "quarterly_report": 4,
    "earnings_release": 4,
    "investor_presentation": 2,
}

# Documents older than this many years are dropped: an equity view is driven by
# recent performance, and stale filings dilute retrieval.
MAX_DOCUMENT_AGE_YEARS = 4


def current_year() -> int:
    """Today's year.

    Computed rather than hard-coded: the previous implementation pinned
    ``CURRENT_YEAR = 2026`` in the discovery module, which would silently start
    discarding every current filing the moment the calendar rolled over.
    """
    from datetime import date

    return date.today().year

# Aggregator / mirror / paywall domains. These host stale or altered copies of
# filings, so they rank below any primary investor-relations source.
DEPRIORITISED_DOMAINS = (
    "annualreports.com",
    "companiesmarketcap.com",
    "scribd.com",
    "studylib.net",
    "coursehero.com",
    "slideshare.net",
    "researchgate.net",
)

# A downloaded PDF must clear all of these to enter the index. Rejections are
# recorded with a reason and surfaced in the UI (see ingest/validate.py).
#
# The thresholds are per document type because the types genuinely differ. A
# uniform 2-page / 3-keyword gate rejected TCS's earnings releases, which are
# single-page press releases, and investor decks, which are legitimately
# text-sparse. Both are valid sources.
MIN_PDF_PAGES = 2
MIN_PDF_CHARS = 1500                # guards against scanned/image-only PDFs
MIN_FINANCIAL_KEYWORD_HITS = 3      # guards against off-topic documents

MIN_PDF_PAGES_BY_TYPE = {
    "earnings_release": 1,
}
MIN_PDF_CHARS_BY_TYPE = {
    "earnings_release": 800,
    "investor_presentation": 600,   # slides carry little text per page
}
MIN_FINANCIAL_KEYWORD_HITS_BY_TYPE = {
    "investor_presentation": 2,
}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Sized in tokens rather than characters so budgeting against the context window
# is exact. ~600 tokens keeps a chunk topically tight while still holding a full
# paragraph of MD&A commentary.
CHUNK_TARGET_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 100
CHUNK_MIN_TOKENS = 50               # drop fragments (page numbers, headers)

# Financial tables are converted to markdown and kept whole up to this size.
# Splitting a table mid-row destroys the row/column relationship that makes it
# answerable at all, so this limit is generous on purpose.
TABLE_MAX_TOKENS = 1200


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DENSE_TOP_K = 25        # candidates from FAISS  (semantic similarity)
SPARSE_TOP_K = 25       # candidates from BM25   (exact terms, tickers, numbers)
RRF_K = 60              # Reciprocal Rank Fusion damping constant (standard)
RERANK_CANDIDATES = 20  # fused candidates handed to the reranker
FINAL_TOP_K = 8         # passages actually sent to the answering model

# Number of paraphrases generated per user question. Multi-query expansion
# covers vocabulary mismatch: a user asks "how profitable", the filing says
# "gross margin". Costs one extra fast-model call per question.
QUERY_EXPANSIONS = 3


# ---------------------------------------------------------------------------
# Valuation defaults
# ---------------------------------------------------------------------------
# These are *fallbacks*, used only when a value cannot be derived from live
# data. Every assumption that reaches the report records which path it took, so
# a reader can always tell a measured input from a default.

DEFAULT_EQUITY_RISK_PREMIUM = 0.055     # US mature-market ERP (Damodaran)
DEFAULT_INDIA_RISK_PREMIUM = 0.078      # ERP + India country risk premium
DEFAULT_RISK_FREE_RATE = 0.042          # overridden by live 10Y treasury yield

# The risk-free rate must match the currency the cash flows are in. Applying a
# US Treasury yield to a rupee-denominated valuation understates the discount
# rate by roughly 250bps and materially overstates the resulting value.
DEFAULT_RISK_FREE_RATE_BY_COUNTRY = {
    "US": 0.042,        # 10-year US Treasury
    "IN": 0.069,        # 10-year Indian government security
    "GB": 0.043,        # 10-year gilt
    "DE": 0.026,        # 10-year bund
    "JP": 0.016,        # 10-year JGB
}

DEFAULT_TAX_RATE = {
    "US": 0.21,
    "IN": 0.2517,                       # 22% concessional + surcharge + cess
}

# A 10-year explicit forecast, not 5.
#
# With a 5-year horizon, 61-73% of enterprise value fell into the terminal
# value — the model was mostly asserting a perpetuity rather than forecasting a
# business, and it triggered the project's own >80% concentration warning on
# some names. A Gordon terminal at 2.5% implies roughly a 12x exit multiple on
# free cash flow, which systematically undervalues companies that compound for
# longer than five years. Ten years is the standard horizon for a business
# still growing well above GDP, and it moves value out of the perpetuity and
# into cash flows the model actually projects.
DCF_FORECAST_YEARS = 10

# Geometric decay factor for the revenue growth path: each year closes this
# share of the remaining gap to the terminal rate.
#
# Originally 0.55, which took a 65% starter to only ~4x cumulative growth over
# ten years — the fade was so fast that genuine multi-year compounders (the
# AI-cycle semiconductor names above all) were systematically undervalued: NVDA
# collapsed to single-digit growth by year five while every analyst on the
# street modelled far more. 0.70 lets growth persist longer — ~7x cumulative
# over ten years, with an 18% rate still in year five — while staying far below
# the 16x a linear fade compounds to, which is the absurdity this decay exists
# to prevent. It is the single largest lever on a growth name's fair value, so
# it is set conservatively but not punitively.
DCF_GROWTH_DECAY = 0.70
TERMINAL_GROWTH_BOUNDS = (0.01, 0.04)   # must stay below long-run GDP growth
WACC_BOUNDS = (0.05, 0.20)              # sanity rails on the discount rate

# Terminal operating margin is an economic judgement about the mature state of
# the business, not a mechanical clamp to the recent range. A decade out,
# competition, product mix and the limits of scale can pull even a dominant
# company's peak margin down — so the model is allowed to normalise the margin
# BELOW the recent range when it can justify it. Two rails only: a floor (a
# fraction of today's margin) stops an unjustified collapse, and a ceiling (the
# company's own demonstrated peak) stops a margin no evidence supports.
TERMINAL_MARGIN_FLOOR_FRACTION = 0.60   # floor = 60% of the current operating margin
TERMINAL_MARGIN_ABSOLUTE_FLOOR = 0.05

# Blume beta adjustment: beta_used = w * beta_raw + (1 - w) * 1.0.
#
# A raw historical beta is a noisy, backward-looking estimate that reliably
# overstates how extreme a stock's forward beta will be — betas regress toward
# the market over time. Marshall Blume's adjustment, the Bloomberg/practitioner
# default, corrects for that mean reversion. It matters most exactly where it
# should: NVIDIA's raw 5-year beta of ~2.2 implied a 16.8% discount rate that no
# analyst uses, and adjusting it to ~1.8 removes roughly 220bps of cost of
# equity. Applied to every name, so low-beta stocks are nudged up toward 1.0 and
# high-beta stocks pulled down — the standard, symmetric treatment.
BETA_BLUME_WEIGHT = 0.67

# Long-horizon (DCF) beta reversion, applied on top of Blume for the discount
# rate only: beta_dcf = w * beta_blume + (1 - w) * 1.0.
#
# A 10-year DCF discounts a cash-flow stream whose value is 60-80% terminal —
# money that accrues in the company's mature phase, a decade out, where its beta
# will be far closer to the market than a 5-year trailing estimate. Discounting
# those distant, mature cash flows at a spot high-growth beta systematically
# over-penalises them. The horizon reversion produces the *average* beta over the
# life of the forecast rather than today's spot beta. Combined with Blume this
# places roughly equal weight on the measured beta and the market (0.67 * 0.75 ≈
# 0.5) — NVIDIA's raw 2.2 becomes a ~1.6 horizon beta, ~11.5-12% WACC, in the
# range practitioners actually use for the name rather than 14.5%.
BETA_HORIZON_WEIGHT = 0.75

# Build-up size premium added to the cost of equity, by market capitalisation.
# The Duff & Phelps / Ibbotson size effect: smaller companies carry a higher
# required return (illiquidity, fragility, less coverage), the largest a small
# discount (deep liquidity, quality, lower distress risk). Standard in the
# build-up method — Ke = Rf + beta*ERP + size premium — and symmetric, so it is
# not a mega-cap thumb on the scale: a micro-cap is penalised exactly as a
# mega-cap is credited. Bounds are (min_market_cap_inclusive, premium).
SIZE_PREMIUM_TIERS = (
    (200e9, -0.015),    # >= $200bn  : mega-cap, -1.5%
    (10e9, -0.005),     # $10-200bn  : large-cap, -0.5%
    (2e9, 0.010),       # $2-10bn    : mid-cap, +1.0%
    (0.0, 0.025),       # < $2bn     : small-cap, +2.5%
)

# Sensitivity grid: WACC on one axis, terminal growth on the other.
SENSITIVITY_WACC_STEPS = 5
# +/-100bps per step, spanning +/-200bps overall. A narrower band understated
# how much the answer rides on the discount rate: for a high-beta name the
# CAPM rate is itself uncertain by more than 50bps, so a grid that only moved
# +/-100bps in total implied a false precision.
SENSITIVITY_WACC_DELTA = 0.010
SENSITIVITY_GROWTH_STEPS = 5
SENSITIVITY_GROWTH_DELTA = 0.0025       # +/- 25bps per step

# Upside thresholds that map a computed fair value to a rating. Replaces the
# previously hard-coded "BUY" banner.
RATING_THRESHOLDS = {
    "BUY": 0.15,        # >= +15% upside to fair value
    "HOLD": -0.10,      # between -10% and +15%
    # below -10% -> SELL
}


# ---------------------------------------------------------------------------
# Scenario analysis (bear / base / bull)
# ---------------------------------------------------------------------------
# The sensitivity grid above flexes two inputs one at a time. A scenario is a
# different, complementary idea: a coherent *state of the world* in which the
# value drivers move together — a downside where growth disappoints AND margins
# compress AND the market demands a higher risk premium, all at once. That
# correlation is what makes a bear/bull range meaningful rather than a
# mechanical grid corner.
#
# The magnitude of the moves is not a fixed +/-X%. Where the history allows it,
# the growth and margin spreads are sized from the company's *own* historical
# dispersion (see valuation/scenarios.py): a business whose growth has swung
# wildly gets a wide range; a steady compounder gets a narrow one. These bounds
# only floor and cap that data-derived spread so a single anomalous year cannot
# produce an absurd band, and provide a fallback when history is too short to
# measure dispersion at all.

# Prior probabilities placed on each scenario, used only to compute a
# probability-weighted expected value. Deliberately conservative and symmetric:
# the base case carries the weight, and the tails are treated as equally likely
# so the expected value does not smuggle in a directional view. They must sum
# to 1.0.
SCENARIO_PROBABILITIES = {"bear": 0.25, "base": 0.50, "bull": 0.25}

# Discount-rate move between scenarios. In a downside the market demands a
# higher risk premium; in an upside it accepts a lower one. +/-150bps is a
# little wider than one sensitivity step (100bps) because a scenario moves the
# whole risk picture, not just the rate in isolation. Clamped to WACC_BOUNDS.
SCENARIO_WACC_DELTA = 0.015

# Perpetual-growth move between scenarios, in the same spirit. Kept small
# because the terminal rate is already tightly bounded near long-run GDP; a
# scenario should not imply the economy itself grows a full point faster
# forever. Clamped to TERMINAL_GROWTH_BOUNDS.
SCENARIO_TERMINAL_GROWTH_DELTA = 0.005

# Floors and caps on the data-derived spreads. The growth spread is one
# standard deviation of the company's historical revenue growth, bounded here;
# the margin spread is one standard deviation of its historical operating
# margin, bounded here. Floors guarantee the three cases always differ
# materially; caps stop a single volatile year from opening an indefensible gap.
SCENARIO_MIN_GROWTH_SPREAD = 0.05      # at least +/-5pp on year-one growth
SCENARIO_MAX_GROWTH_SPREAD = 0.20      # at most +/-20pp
SCENARIO_MIN_MARGIN_SPREAD = 0.02      # at least +/-2pp on the terminal margin
SCENARIO_MAX_MARGIN_SPREAD = 0.12      # at most +/-12pp


# ---------------------------------------------------------------------------
# Valuation triangulation (blend)
# ---------------------------------------------------------------------------
# A DCF is one opinion, and an intrinsic one: it says what the business is worth
# on its own cash flows, which on a highly-rated name frequently sits well below
# what the whole market of analysts will pay. Rather than assert the market is
# wrong, the blend triangulates — it places our DCF alongside the Wall Street
# consensus price target and reconciles them into one figure, weighting each
# source by how much confidence it deserves.
#
# The point is not to reverse-engineer the share price. It is to stop a single
# conservative (or single optimistic) method being reported as the answer when
# several independent methods disagree. Every source, its value and its weight
# are shown, so the blend is auditable rather than a black box.

# Base weight for each source *type*. The model (our DCF) is one independent
# voice; the analyst-consensus weight is scaled separately by how many analysts
# stand behind it (below), because a target followed by sixty analysts deserves
# more confidence than one followed by three. Comps and web sources are wired
# for later use and weighted lower, reflecting their looser link to value.
BLEND_SOURCE_WEIGHTS = {
    "model": 1.0,       # our DCF
    "analyst": 1.0,     # per-unit weight, then scaled by analyst count
    "comps": 0.7,
    "web": 0.4,
}

# Analyst-consensus confidence scales with coverage: full unit weight is reached
# at this many analysts, and the multiplier is capped so a single mega-cap with
# a hundred analysts cannot drown out every other method entirely. A "balanced
# triangulation": the consensus can pull the blend well above our DCF, but the
# DCF still counts.
BLEND_ANALYST_REFERENCE_COUNT = 12      # analysts for one full weight unit
BLEND_ANALYST_MAX_WEIGHT_MULTIPLE = 3.0  # cap on the consensus weight multiple

# Robust outlier rejection. With three or more estimates, any value outside this
# band around the (unweighted) median is dropped from the blend and reported as
# excluded, so one stale target or bad extraction cannot swing the result. With
# only two estimates there is nothing to reject against, so both are kept.
BLEND_OUTLIER_LOW_MULTIPLE = 0.40       # drop below 0.40x the median estimate
BLEND_OUTLIER_HIGH_MULTIPLE = 2.50      # drop above 2.50x the median estimate

# When our intrinsic DCF and the analyst consensus disagree by more than this,
# the report says so explicitly: a wide gap is a signal to read the blend as a
# reconciliation of two genuinely different views, not a precise number.
BLEND_DIVERGENCE_FLAG = 0.35            # +/-35% between DCF and consensus

# When our intrinsic value sits more than this away from BOTH the market price
# AND the analyst consensus, in the same direction, we are the outlier. The
# report flags it: an intrinsic view is allowed to diverge from the market, but a
# lone value far from the whole street should be read as a specific, contrarian,
# assumption-driven call — the assumption agent has already had a pass at it, so
# what remains is a genuine difference of view, surfaced rather than buried.
MISCALIBRATION_FLAG = 0.35             # +/-35% from BOTH price and consensus


# ---------------------------------------------------------------------------
# Branding (report + UI)
# ---------------------------------------------------------------------------

BRAND_NAVY = "#0F2942"
BRAND_ACCENT = "#1F6FEB"
BRAND_POSITIVE = "#137333"
BRAND_NEGATIVE = "#B3261E"
BRAND_MUTED = "#5B6B7C"
BRAND_SURFACE = "#F5F7FA"
