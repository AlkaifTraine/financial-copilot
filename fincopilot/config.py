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
OVERRIDES_DIR = ROOT_DIR / "overrides"  # analyst assumption overrides, per company

for _d in (DATA_DIR, INDEX_DIR, REPORTS_DIR, CHARTS_DIR, CACHE_DIR, OVERRIDES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Cheap, fast model for MECHANICAL and structured-extraction calls: query
# rewriting, reranking, catalyst/segment extraction. These run many times and are
# constrained enough that a stronger model adds little — and several have
# deterministic post-processing (date anchoring, reconciliation) on top.
FAST_MODEL = "gpt-4.1-mini"

# Stronger model for the ANALYSIS and reader-facing PROSE, where reasoning quality
# actually changes the output: the assumption drivers (which set the whole DCF), the
# agent calibration, the thesis argument, the quantified risks, the competitive/
# moat analysis, and the narrative sections a human reads end-to-end. Wired per call
# site via model=config.WRITER_MODEL, so the blend is explicit and tunable: move any
# call between FAST_MODEL and WRITER_MODEL by editing that one call.
WRITER_MODEL = "gpt-4.1"

# Deterministic by default. Report prose gets a small amount of temperature so
# it does not read like a form letter; anything numeric runs at 0.
TEMPERATURE_FACTUAL = 0.0
TEMPERATURE_PROSE = 0.2

# ---------------------------------------------------------------------------
# Routing: providers, load balancing, fallbacks
# ---------------------------------------------------------------------------
# Calls go through a LiteLLM Router rather than a single provider SDK, so a
# provider outage degrades the service instead of stopping it. Two router
# groups mirror the FAST/WRITER blend above; each has an OpenAI primary and a
# Gemini fallback of comparable capability.
#
# Load balancing happens *within* a group: add more deployments (extra keys,
# regions, or an Azure mirror) under the same group name and the router spreads
# traffic across them, respecting each one's rpm/tpm ceiling. Fallback happens
# *between* groups: only when every deployment in the primary group fails or is
# in cooldown does the router cross to another provider. The two are kept
# separate deliberately — silently load-balancing a report's prose across two
# vendors would make output quality vary run to run for no stated reason.
FAST_GROUP = "fincopilot-fast"
WRITER_GROUP = "fincopilot-writer"
FAST_FALLBACK_GROUP = "fincopilot-fast-fallback"
WRITER_FALLBACK_GROUP = "fincopilot-writer-fallback"

# Gemini stand-ins, matched to the tier they cover. The key is read at router
# build time from GEMINI_API_KEY; absent one, the fallback deployments are
# simply not registered and the router runs OpenAI-only.
FALLBACK_FAST_MODEL = "gemini/gemini-2.5-flash"
FALLBACK_WRITER_MODEL = "gemini/gemini-2.5-pro"

# Per-deployment ceilings the router load-balances against. Set below the
# account's real limits so the router shifts traffic before the provider starts
# returning 429s.
OPENAI_RPM = int(get_secret("OPENAI_RPM", "400") or 400)
GEMINI_RPM = int(get_secret("GEMINI_RPM", "150") or 150)

ROUTER_TIMEOUT_SECONDS = 120
ROUTER_NUM_RETRIES = 2          # per deployment, before the router moves on
ROUTER_ALLOWED_FAILS = 3        # failures before a deployment is cooled down
ROUTER_COOLDOWN_SECONDS = 60

# "simple-shuffle" is weighted-random across healthy deployments and needs no
# shared state. The usage-based strategies need Redis to coordinate across
# processes; this app runs as a single Streamlit process, so the extra
# dependency would buy nothing.
ROUTER_STRATEGY = get_secret("ROUTER_STRATEGY", "simple-shuffle") or "simple-shuffle"

# A hard ceiling on what one report/session may spend, enforced in-process by
# the guardrail layer. A runaway retry loop is a billing incident, not just a
# bug, so the limit is on money rather than only on call count.
MAX_USD_PER_PROCESS = float(get_secret("MAX_USD_PER_PROCESS", "25") or 25)

# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------
# Two ways in, because the two have completely different cost exposure:
#
#   own key      the visitor pastes their own OpenAI key. They pay, so they are
#                not metered against the owner's budget or rate limits.
#   access code  a shared secret the owner hands out. These visitors spend the
#                OWNER'S credits, so every limit applies to them.
#
# The code is read from the environment and has NO default. It must never be
# committed: this repository is public, so a literal code in source is
# harvestable by anyone reading GitHub, and it authorises spending on the
# owner's account. Unset, the access-code route is simply unavailable and the
# app is bring-your-own-key only — which is the safe way to fail.
ACCESS_CODE = get_secret("ACCESS_CODE", "") or ""

# ON by default. This protects the owner's API key on any deployment where
# nobody remembered to configure anything, which is exactly the deployment most
# at risk. The safe default for "a public URL in front of a paid API" is not
# "let everyone spend the owner's money".
#
# With the gate on and no ACCESS_CODE set, the app is bring-your-own-key only:
# visitors must supply their own key, so the owner's credits cannot be spent by
# a stranger at all. Setting ACCESS_CODE is what re-opens the owner's key to
# people holding the code.
#
# Local development opts out with REQUIRE_ACCESS=0 in .env.
REQUIRE_ACCESS = (get_secret("REQUIRE_ACCESS", "1") or "1") not in (
    "0", "", "false", "False",
)


# ---------------------------------------------------------------------------
# Usage analytics
# ---------------------------------------------------------------------------
# Which companies get loaded, what gets asked, what fails, and what it costs.
# Without this there is no way to answer "is anyone using this, and what breaks
# for them" other than guessing.
ANALYTICS_ENABLED = (get_secret("ANALYTICS_ENABLED", "1") or "1") not in (
    "0", "", "false", "False",
)
ANALYTICS_DB_PATH = get_secret("ANALYTICS_DB_PATH") or str(DATA_DIR / "usage.db")

# Whether to keep the text of questions. On, the log shows what people actually
# asked, which is where the product lessons are; off, only the length and the
# classifier's category are kept. Question text is scrubbed of secrets and PII
# before storage either way, and API keys are never stored under any setting.
ANALYTICS_STORE_QUESTION_TEXT = (
    get_secret("ANALYTICS_STORE_QUESTION_TEXT", "1") or "1"
) not in ("0", "", "false", "False")


# Every chat question is first classified by FAST_MODEL to judge what it is for
# (see guardrails.classify_query). Set to "0" to disable — the deterministic
# scans and the grounding requirement still apply.
QUERY_CLASSIFIER_ENABLED = (
    get_secret("QUERY_CLASSIFIER_ENABLED", "1") or "1"
) not in ("0", "", "false", "False")

# How sure the classifier must be before a question is actually refused. A
# false negative is cheap — the question proceeds to a grounded, cited answer.
# A false positive tells a real analyst no, which is expensive, so uncertain
# verdicts are not acted on.
QUERY_BLOCK_CONFIDENCE = float(get_secret("QUERY_BLOCK_CONFIDENCE", "0.7") or 0.7)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Deliberately NOT routed and NOT given a cross-provider fallback. An index is
# built from one embedding model's vector space; serving a query from a
# different model — even at the same dimension — silently returns nonsense
# neighbours, and at a different dimension it fails outright. A provider
# fallback here would corrupt retrieval rather than protect it.
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
    "annual_report": 2,
    "quarterly_report": 3,
    "earnings_release": 3,
    "investor_presentation": 2,
}

# Documents older than this many years are dropped. Retrieval quality is the
# reason, not storage: a four-year window put five annual reports into the same
# index, and a question about revenue or strategy would retrieve the FY2022
# discussion alongside the FY2026 one with nothing in the passage to say which
# was current. Narrowing the window removes that whole class of confidently
# outdated answer, and cuts indexing time and embedding cost with it.
#
# At 2, the window keeps the current year and the two before it — for Bikaji,
# FY2024 through FY2026, dropping the FY2022 and FY2023 annual reports.
#
# This does NOT shorten the financial history: statements come from audited
# XBRL or from results filings, and each filing carries a restated prior year,
# so those are merged across filings (see fundamentals/_from_results_pdf).
MAX_DOCUMENT_AGE_YEARS = int(get_secret("MAX_DOCUMENT_AGE_YEARS", "2") or 2)


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
# Terminal growth is a NOMINAL rate and must match the currency the cash flows
# are in — the same rule this file already applies to the risk-free rate, and
# for the same reason. A rupee DCF discounts at an Indian nominal rate near 13%
# because that rate carries Indian inflation; growing the same cash flows at a
# US-style 2.5% then has the company shrinking a few points a year in real terms
# forever, which is not a conservative assumption but an incoherent one.
#
# The error is invisible and one-directional. It never looks wrong — 2.5% reads
# as prudence — while it widens the (WACC - g) denominator that sets terminal
# value, and terminal value is most of a DCF. For India it was pricing a growing
# consumer business as a melting ice cube.
#
# Bounds are floored near expected inflation (below it means real decline
# forever) and capped below long-run nominal GDP growth (above it means the
# company eventually becomes the whole economy).
#            (floor, cap, default)
TERMINAL_GROWTH_BY_COUNTRY = {
    "US": (0.01, 0.04, 0.025),
    "IN": (0.040, 0.065, 0.050),    # floor at RBI's 4% inflation target;
                                    # nominal GDP ~10-11% is the ceiling
    "GB": (0.01, 0.04, 0.025),
    "DE": (0.01, 0.035, 0.020),
    "JP": (0.00, 0.025, 0.010),
    "CN": (0.02, 0.055, 0.035),
}
_DEFAULT_TERMINAL_GROWTH = (0.01, 0.04, 0.025)


def terminal_growth_for(country: str | None) -> tuple[float, float, float]:
    """(floor, cap, default) terminal growth for a country's nominal currency."""
    return TERMINAL_GROWTH_BY_COUNTRY.get(
        (country or "").upper(), _DEFAULT_TERMINAL_GROWTH
    )


# Retained for callers without a country to hand; prefer terminal_growth_for().
TERMINAL_GROWTH_BOUNDS = (0.01, 0.04)
WACC_BOUNDS = (0.05, 0.20)              # sanity rails on the discount rate

# Terminal operating margin is an economic judgement about the mature state of
# the business, not a mechanical clamp to the recent range. A decade out,
# competition, product mix and the limits of scale can pull even a dominant
# company's peak margin down — so the model is allowed to normalise the margin
# BELOW the recent range when it can justify it. Two rails only: a floor (a
# fraction of today's margin) stops an unjustified collapse, and a ceiling (the
# company's own demonstrated peak) stops a margin no evidence supports.
TERMINAL_MARGIN_FLOOR_FRACTION = 0.60   # (deprecated) old mechanical floor
TERMINAL_MARGIN_ABSOLUTE_FLOOR = 0.05
# The terminal operating margin is an ECONOMIC judgement and is no longer clamped
# to the historical range — a mechanical bound was forcing an economically derived
# 40% back up to 48% on NVDA, exactly the methodological error we set out to
# remove. Only physical sanity rails apply now: an operating margin cannot be
# negative, and one above ~90% of revenue is not a business a DCF can carry. A
# material departure from the current margin is EXPLAINED (competitive
# normalisation, mix, scale), not overridden.
TERMINAL_MARGIN_SANITY_CEILING = 0.90
# How far the terminal margin must sit from today's before the report explains
# the departure rather than treating it as "near current".
TERMINAL_MARGIN_MATERIAL_DEVIATION = 0.10   # 10 percentage points

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

# Size premium added to the cost of equity, by market capitalisation. The
# empirical size effect (Duff & Phelps / CRSP deciles) is a SMALL-CAP phenomenon:
# smaller companies carry a higher required return (illiquidity, fragility, less
# coverage), and the effect decays to ~0 for the largest decile. It is NOT
# symmetric — the data do not support a NEGATIVE premium (a "credit") for large or
# mega caps, so an earlier -1.5% mega-cap discount was a house thumb on the scale
# dressed as a standard default. Large and mega caps therefore get NO size
# adjustment: their cost of equity is pure CAPM (Ke = Rf + beta*ERP), which is the
# transparent, defensible construction. Only small and mid caps carry the (positive,
# empirically grounded) premium. Bounds are (min_market_cap_inclusive, premium).
# Values are anchors, not steps: `wacc._interpolated_size_premium` interpolates
# between them on log market cap. Read as steps they put a cliff at each
# boundary — a company at $1.9bn taking 2.5% while one at $2.1bn takes 1.0%,
# two businesses of indistinguishable size handed discount rates 1.5 points
# apart, which is a double-digit swing in fair value decided by which side of a
# round number they closed at.
#
# The $250m anchor exists because the band below $2bn was previously flat at the
# full small-cap premium, which charged a $1.9bn mid-cap the same risk as a
# $50m micro-cap. The size effect is steep only among genuinely small
# companies; by CRSP/Duff & Phelps deciles a ~$2bn company sits near the middle,
# not at the small-cap extreme.
SIZE_PREMIUM_TIERS = (
    (200e9, 0.0),       # >= $200bn  : mega-cap, none (pure CAPM)
    (10e9, 0.0),        # $10-200bn  : large-cap, none (pure CAPM)
    (2e9, 0.010),       # $2bn       : mid-cap anchor, +1.0%
    (250e6, 0.025),     # $250m      : small-cap anchor, +2.5%
    (0.0, 0.025),       # < $250m    : micro-cap, +2.5% (no lower anchor)
)

# The size tiers above are in absolute USD, but market cap arrives in the
# company's own currency (yfinance reports it in the local unit). Without
# converting, a mid-cap Indian company's rupee market cap clears the USD mega-cap
# threshold and is wrongly handed the mega-cap discount. Only a rough magnitude is
# needed to pick a tier, so an approximate static rate is used rather than a
# fragile extra FX fetch — being 20% off never crosses a 5x tier boundary. A
# currency absent from this map skips the size premium entirely (safer than
# assuming parity with the dollar). Rates are USD per one unit of the currency.
APPROX_USD_FX = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "INR": 0.012,
    "CNY": 0.14, "HKD": 0.128, "CAD": 0.73, "AUD": 0.66, "CHF": 1.12,
    "KRW": 0.00072, "TWD": 0.031, "BRL": 0.18, "SGD": 0.74,
}

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

# Base weight for each source *type*. Only the DCF carries weight in the target:
# the target price IS the intrinsic DCF (base case). The scenario expected value,
# the comps value and the analyst consensus are shown as independent CROSS-CHECKS
# and are forced to weight 0 in build_blend — the scenario set is built from the
# same DCF (its base case is the DCF), so blending it back in would double-count
# the one framework, and a peer multiple is a different valuation basis. The
# non-model weights below are retained only for reference/experiments; build_blend
# does not read them for scenario/comps.
BLEND_SOURCE_WEIGHTS = {
    "model": 1.0,       # our DCF — the intrinsic target
    "scenario": 0.0,    # cross-check only (built FROM the DCF; blending double-counts)
    "analyst": 1.0,     # per-unit weight, then scaled by analyst count (external ref)
    "comps": 0.0,       # market-based cross-check only, not blended into intrinsic value
    "web": 0.4,
}

# The analyst consensus is an EXTERNAL market reference, NOT an intrinsic input.
# Our target price is derived from OUR OWN methods (DCF, comps, the
# probability-weighted scenario value); the consensus is shown alongside for
# comparison but is not blended into the number. An earlier design gave the
# consensus up to 3x the DCF weight and folded it in, which quietly turned the
# "intrinsic" target into a mostly-market number — the opposite of the stated
# philosophy. Flip to True only with a principled methodology for weighting it.
BLEND_INCLUDE_CONSENSUS = False

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
