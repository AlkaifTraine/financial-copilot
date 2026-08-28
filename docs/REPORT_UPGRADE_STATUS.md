# Report-quality upgrade — status & handoff

This tracks the "AI summary → institutional-quality equity research" upgrade.
Read this first in a new session; do **not** re-explore or rebuild the platform.

**Current as of 2026-08-28.** 368 tests pass; `main` was clean at `89a1e5d` before
the audited-provenance work below. If you change code, update this file in the same commit —
a stale handoff here is what makes a fresh session hallucinate the project's state.

## Verifying the deployment (read before claiming the site is down)

The live demo is `https://fincopilot.duckdns.org`. **From the IIT KGP network it is
unreachable — `duckdns.org` is blocked by a FortiGuard web filter** (port 443 resets,
port 80 returns a "Web Filter Violation" page). That is a network block, *not* an
outage. Never diagnose the deployment with `curl` from this machine.

Because CI hard-resets the instance to `origin/main` on every push, **whatever is on
`origin/main` is what is live.** Check deployed state with `git rev-parse origin/main`
plus the last green Actions run. `gh` is not installed, so the Actions tab must be
checked in a browser; the site itself opens fine off-network (e.g. mobile data).

## Guardrails (do not break these)

- **Deterministic valuation.** All valuation math is NumPy. The LLM only
  *proposes* forward assumptions (growth, terminal margin, terminal growth),
  which are then clamped to data-derived bounds. Never let the model do
  arithmetic. Assumptions are cached per company + data fingerprint
  (`fincopilot/valuation/assumptions.py::_propose`) so a valuation reproduces
  exactly — keep it that way.
- **The recommendation is OUR intrinsic view** (DCF/scenarios). The analyst
  consensus is shown as "what the market believes" — the counter-view the
  thesis argues against — never folded into our own rating. (User decision.)
- **Auto-deploy is live.** Push to `main` → GitHub Actions runs `pytest tests/`
  → deploys to fincopilot.duckdns.org. **Never push broken code.** Before any
  commit: `venv/Scripts/python.exe -m pytest tests/ -q` must pass (368 tests), and a
  report must render (see Verify below).
- **A blocking QA finding withholds the report.** Do not "fix" a blocked report by
  demoting its check to advisory — that was the v7 failure the v8 gate exists to
  prevent. Fix the generator, or let the correction loop fix it.
- **Never blend the cross-checks into the target.** The 12-month price target IS the
  intrinsic DCF base case. Scenario expected value, comps and analyst consensus are
  shown alongside as cross-checks with weight forced to 0 in `build_blend`; blending
  the scenario EV double-counts the DCF it is derived from, and
  `check_valuation_integrity` guards this at the gate.
- **Don't clear `.cache/assumptions` casually.** The assumption proposal is cached on
  ticker + a fingerprint of the reported numbers (not the retrieved context), which
  is what makes a company's DCF reproduce exactly until its filings change. Clearing
  it is what makes the valuation wander between runs.
- **Statement numbers come only from audited, concept-tagged XBRL.** SEC
  `companyfacts` or Ind-AS XBRL filed with the NSE — never PDF text, never a
  market-data vendor. yfinance is for live market data only (price, shares,
  beta, analyst targets). If neither XBRL source has the company,
  `load_financials` returns `None` and it is **not valued**. Do not reintroduce
  a vendor statement fallback to "improve coverage": that trades the project's
  central claim for breadth. (User decision, 2026-08-28.)
- **All model calls go through `llm.complete` / `llm.complete_json`.** That is
  the only chokepoint where guardrails, spend metering and provider fallback
  are applied; a new call site that imports a provider SDK directly bypasses
  all three silently. Never put OpenAI and Gemini deployments in the same
  router group — load balancing is *within* a provider, fallback is *between*
  groups, and mixing them splits a report's prose across vendors run to run.
- **Embeddings are not routed and must never be given a provider fallback.**
  An index lives in one embedding model's vector space; serving a query from
  another returns nonsense neighbours at the same dimension and fails at a
  different one.
- **A QA-blocked report is never served from the store.** The cache must not
  become the route by which an integrity block gets reversed.
- **Verify by looking.** Generate an NVDA report, rasterize page 1 with PyMuPDF,
  and actually view it — several bugs were only visible in the render.
- `gh` is not installed; commit locally, `git push` uses cached credentials.
- Committed portfolio artifact: `reports/SAMPLE_NVDA_report.{pdf,html}`.

## Done (shipped, verified)

### The trust layer (v8, Aug 21-22 — newest first)

- **Reliability scorecard** (`report/reliability.py::compute_reliability`, on
  `report.reliability`). Deterministic, no LLM, computed after QA. Headline is
  numeric traceability: every figure in the prose is matched against a registry of
  the model's own numbers (`_walk` over canonical / assumptions / scenarios /
  priced_in / financial / forecast / segment / comps / competition) **and** against
  the text of the cited source snippets. A figure matching neither is "unverified" —
  a verify prompt, not proof of error. Plus citation coverage, QA status, source
  freshness and valuation confidence → a 0-100 score and an A-D grade. Rendered on
  the site (`app.py::_render_reliability`) and in the PDF. Unverified figures are
  **listed individually** with location + surrounding context, and the HTML report
  makes each one a clickable jump-to-figure. NVDA ≈ 85/A, ≈24% unverified.
- **Hard QA gate + self-correction loop (v8).** This reverses the earlier v5-v7 rule
  that prose heuristics must never block. Now `BLOCKING_SEVERITIES` = {CRITICAL,
  HIGH} — see `CHECK_SEVERITY` in `report/qa.py`. CRITICAL is reserved for
  deterministic contradictions (probability sums, scenario order, rating sign,
  figure-vs-canonical, valuation double-count, segment non-reconciliation,
  annualization arithmetic); HIGH covers integrity failures a report must not ship
  (citation grounding, unsupported market-implied claims, directional impossibility,
  quarterly-label-on-annual). False positives are mitigated by **regenerating, not by
  demoting to advisory**: `report/correction.py::run_correction_loop` maps each
  blocking check to the component that owns it (`CHECK_TO_COMPONENT`; deterministic
  checks → None = unfixable), regenerates that component alone with the QA feedback
  appended, and re-runs the audit, up to `MAX_QA_RETRIES = 3`. Unresolved → 
  `report.blocked = True` and the app withholds the document. The $78B-annualization
  case is the acceptance test.
- **Model blend.** `config.WRITER_MODEL = "gpt-4.1"` for analysis/prose,
  `config.FAST_MODEL = "gpt-4.1-mini"` for mechanical calls; every call site names
  one explicitly, so moving a call between tiers is a one-line edit.
  `llm.get_usage()` tracks calls and tokens per process (≈$0.15/report).
- **Institutional apparatus.** `builder._disclosures` (computed last, on
  `report.disclosures`) closes both outputs with a deterministic "Methodology,
  Ratings & Disclosures" block: the 12-month price target (= the intrinsic DCF),
  rating definitions from `config.RATING_THRESHOLDS`, valuation methodology
  ("cross-checked, never blended"), key risks to the target, an honest AI-analyst
  certification, and data/distribution disclosures.
- **Analyst overrides** (`valuation/overrides.py::load_overrides`, auto-loaded in
  `app.py`). `overrides/{slug}.json` = any of `year_one_revenue_growth`,
  `terminal_operating_margin`, `terminal_growth_rate`, `wacc` as decimals. Applied in
  `derive_inputs._apply_overrides` **after** the model proposal and the critique
  agent but before the paths build, bounded only by hard sanity rails, tagged
  `source=analyst`. The override wins over the LLM; the QA gate still applies.

### Earlier work

- **Valuation-accuracy calibration (4 phases) + robustness harness.** Direction:
  strict intrinsic value + honest flags (user decision) — never calibrate to
  price. Phase 3 added adaptive growth decay (a single 0.70 faded durable
  compounders to a mature CAGR in a few years) and degenerate-DCF handling (a
  non-positive equity value → NOT RATED, not "SELL −151%"). Phase 4 added
  normalized growth anchoring (a mature company's single soft year no longer
  extrapolates forever — KO/AAPL y1 2%→5-6%), durable-compounder persistence
  (`_growth_is_steady` → gentler decay; MSFT CAGR 9%→11%), and agent-output
  quantisation for run-to-run stability. `scripts/robustness_check.py` runs the
  engine over 8 diverse profiles and asserts the hard invariants — **all pass**.
  Known-and-accepted: premium mega-caps still price 60-70% below market and flag
  as outliers (strict-intrinsic working as chosen; the market pays multiples a
  disciplined DCF won't credit). Phases 1-2 below.
- **Valuation-accuracy calibration (phases 1-2).** Diagnosis: NVDA and AAPL both
  priced ~80% below market from THREE compounding conservative levers, not one
  (decomposition — WACC 14.7→10 = +60%, margin 36→60 = +38%, growth 20→45% = +71%;
  all three at still-conservative levels ≈ market). All fixes deterministic-math,
  fundamentals-only, never calibrated to price (user decision).
  - *Phase 1 (rule fixes):* long-horizon beta (`wacc.py`, terminal-value-dominated
    horizon reverts beta toward market), symmetric build-up **size premium**
    (`wacc.py`, mega-cap −1.5% … small-cap +2.5%; NVDA WACC 14.5→12.0%),
    durability-based **terminal-margin floor** (`assumptions.py`, steady 0.80 /
    volatile 0.55), and a **year-1 growth floor** at half the recent pace, not the
    terminal rate. NVDA −81→−61%.
  - *Phase 2 (assumption agent):* `valuation/agent.py::critique_assumptions` — a
    second "senior reviewer" pass sees the assumptions AND the fair value they
    produce vs price and consensus, and revises the least-defensible lever from
    fundamentals; re-clamped to the same bounds, cached per fingerprint. Wired in
    `value_company` as probe-build → critique → final-build. NVDA −61→−29%
    (agent lifts growth to its own 22% segment bottom-up); a `_normalise` guard
    catches percent-vs-decimal slips. Deterministic **miscalibration flag**: when
    our value is >35% from BOTH price and consensus (e.g. AAPL −66%/−68%), the
    report says we are the outlier. Tests: `test_calibration.py`, `test_agent.py`.
- Deterministic backbone: bull/base/bear **scenarios** + probability-weighted
  value (`valuation/scenarios.py`); DCF↔analyst↔comps **blend** with outlier
  rejection (`valuation/blend.py`); reverse-DCF market-implied growth.
- **Investment thesis engine** (`report/thesis.py`): market-vs-us, business
  quality vs stock attractiveness, why-{rating}, catalyst/risk, what-would-
  change-our-view (bull/bear triggers), red-team. Rendered first on HTML + PDF.
- **Terminal-margin methodology** (spec #5/#6): economic maturity band, not a
  clamp to the recent range.
- **Reproducibility** (spec #22): valuation identical run-to-run.
- **#4 "What is priced in" table** — reverse-DCF generalised from year-1 growth
  to four drivers (revenue CAGR, mature operating margin, mature FCF margin,
  terminal growth). Each solved by inverting the same `run_dcf` one lever at a
  time, holding the others at base, so both columns are strictly comparable.
  Levers bounded to the economically possible (margin ≤100%, capex ≥0, terminal
  growth < WACC); an unreachable lever renders an em-dash + caveat rather than an
  absurd figure. `valuation/reverse.py::build_priced_in`, models
  `PricedInRow`/`PricedInComparison`, rendered HTML + PDF. Tests in
  `tests/test_reverse.py` (round-trip / direction / unreachable). NVDA: growth
  32.6% and terminal growth 13.1% priced in; margin and FCF unreachable at any
  ceiling — a sharp SELL exhibit.
- **#17 Quantified risks** — the prose risk section is replaced by a table:
  each material risk carries probability / financial impact / valuation impact /
  early-warning indicator, ranked by materiality. LLM pass (`report/risks.py`,
  models `QuantifiedRisk`/`RiskAssessment`) grounded in a targeted risk-factor
  retrieval AND the model's own numbers — the scenario range and reverse-DCF
  gaps give the valuation-impact column real figures to anchor to (NVDA risks
  cite the $21.58 bear and $42.26 base). Deterministic fallback from the
  priced-in/scenario numbers. `risks` SectionSpec dropped from
  `report/sections.py`. Rendered HTML + PDF; tests in `tests/test_risks.py`.
- **#3 / #24 Narrative → interpretation** — `report/sections.py` prompts rewritten
  to demand fact → interpretation → sustainability, closing each section on an
  explicit investment implication (`Section.implication`, "What it means."). Each
  section is handed a one-line map of the others and told to stay in its lane
  (#24); banned-phrase list kills press-release filler.
- **#16 Segment forecasting** — `valuation/segments.py`: segment revenue extracted
  from the segment footnote by the model, ANCHORED to reported per-year total
  revenue and required to reconcile, cached per company+fingerprint; deterministic
  clamped forecast summed bottom-up as a cross-check on the top-down CAGR. Two
  guardrails: >15% gap → "indicative", >35% gap → suppress the exhibit. NVDA
  reconciles cleanly and shows the real debate: 22% bottom-up vs 8% top-down.
  Tests in `tests/test_segments.py`.
- **#19/#20 Catalysts + monitoring** — `report/monitoring.py`: dated catalysts
  (event/timing/direction/metric) and a monitoring dashboard (metric/current/
  trend/expected/why), grounded in the reverse-DCF gaps. Fallback + tests.
- **#11/#13/#14 Competitive moat + management-vs-us** — `report/competitive.py`:
  moat rating argued against the terminal margin it must defend (NVDA "Narrow"),
  plus a management-says-vs-our-view table surfacing where we diverge. Fallback +
  tests (`tests/test_forward_competitive.py`).
- **#22/#23 QA passes** — `report/qa.py`: deterministic citation-grounding audit
  (dangling markers, uncited multi-para sections) and consistency checks
  (scenario ordering, probability sum, blended-in-range, rating/upside sign),
  run last in `build_report`, findings surfaced in limitations. NVDA: 0 findings.
  Tests in `tests/test_qa.py`.

## Next batch (priority order)

The full spec list (#3–#24) and a two-phase valuation-accuracy calibration are
shipped. Candidate follow-ups if the work continues:

1. **Pass the segment bottom-up CAGR directly into the assumption agent.** The
   agent currently re-derives growth from the history table and lands near the
   segment number by reasoning; handing `segments.forecast_segments`'s CAGR to
   `agent.critique_assumptions` as an explicit reference would make the link
   deterministic and let it cite "your CAGR is below your own segment build".
   Needs the segment build to run before `value_company` (compute once, share the
   cached extraction) — the only mild plumbing change.
2. **A second agent iteration + convergence guard.** Today the critique runs once.
   A bounded loop (revise → re-evaluate → stop when the change is <2%) would catch
   cases where one revision exposes a second stale lever. Keep it capped at 2.
3. ~~**Feed `priced_in` into the thesis LLM.**~~ **DONE** — `report/thesis.py` reads
   `valuation.priced_in` directly (and still falls back to `market_implied_growth`
   when the reverse DCF is unavailable).
4. **Broaden the regression.** Verified on NVDA (SELL, −29%) and AAPL (SELL, flagged
   outlier). `scripts/robustness_check.py` covers 8 profiles but **all are US
   tickers** (NVDA/MSFT/AAPL/KO/PFE/INTC/GOOGL). Adding a non-SEC filer would
   exercise the rupee/ERP + yfinance path and the fallbacks in CI rather than by
   hand. The path itself is confirmed working — see below.
5. **Refresh the committed sample report.** `reports/SAMPLE_NVDA_report.{pdf,html}`
   dates from commit `15d8e07` (2026-08-19) and is now 32 commits stale: it predates
   the reliability scorecard, the disclosures block and the price target, so the
   README's portfolio artifact undersells the current build.

## Audited-provenance rewrite (Indian issuers) — shipped 2026-08-28

**The rule now: statement figures come only from the company's own audited,
concept-tagged XBRL. There is no vendor fallback.** yfinance keeps exactly one
job — live market data (price, share count, beta, analyst targets), which no
filing contains. `market.fetch_statements` and its row-label maps are deleted,
not deprecated.

`load_financials` is therefore: SEC XBRL -> Ind-AS XBRL -> **None**. A `None`
means the company is not valued; `app.py` shows the `financials_unavailable`
warning and leaves retrieval and chat working. This was a deliberate user
decision (strict over graceful): an unverifiable valuation is worse than none.

### New modules

| Module | Job |
|---|---|
| `fincopilot/ingest/nse.py` | NSE corporate-filings + annual-reports APIs. Returns audited annual filings with an Ind-AS XBRL link, consolidated ranked first. |
| `fincopilot/fundamentals/indas.py` | Ind-AS XBRL -> `FinancialHistory`. Deterministic, no LLM. |

`FinancialHistory.source_label` / `.is_audited_filing` (from `SOURCE_LABELS` in
`fundamentals/models.py`) replace the `source == "sec_xbrl"` string checks that
were in `app.py` and `report/builder.py`.

### The four defects this had to solve (all real, all tested)

1. **Quarter vs year.** An annual filing tags Q4 and FY side by side, and
   `xbrli:period` carries *the quarter's* dates on both contexts. Trusting them
   understates revenue ~4x, plausibly. The instance separately tags
   `DateOfStartOfReportingPeriod`/`DateOfEndOfReportingPeriod` per context and
   those are correct — period selection uses only those.
2. **Malformed instances.** Every pre-2023 filing tested references `OneD` /
   `FourD` / `OneI` contexts it never defines. Dropping them silently discards
   older years (TCS: 6 years -> 2). They are reconstructed as stubs and dated
   from the reporting-period facts.
3. **`ProfitOrLossAttributableToOwnersOfParent` tagged `0.00`** by filers with
   no NCI — real in Bikaji FY2024, where the actual 2,634,626,000 is only in
   `ProfitLossForPeriod`. A zero here is read as "not tagged".
4. **Standalone vs consolidated.** Consolidated is preferred, not required:
   Nestle India files *only* standalone, and a consolidated-only rule refuses a
   company it should value. Where standalone is used the history says so in its
   notes.

Ind-AS has no operating-income concept, so EBIT is reconstructed as
`PBT before exceptionals + finance costs - other income`, and left `None` rather
than guessed when a component is missing. Every year is then checked against
accounting identities (`indas._validate`: tax bridge reconciles, margins
economically possible, cash non-negative) and a failing year is **rejected and
recorded**, never repaired.

### Discovery

`ingest/pipeline.nse_documents` adds the NSE annual-report archive as a document
source for `country == "IN"`, with `ORIGIN_NSE` ranked above `ORIGIN_WEB` and
below `ORIGIN_EDGAR` in `ORIGIN_TRUST`. DDG remains the fallback. The NSE
archive returns the *complete* set of annual reports rather than whatever ranks
that day, and reaches FY2026 even where the results-XBRL API stops earlier.

Bikaji ingest, verified: **10 documents accepted**, 3 annual reports from NSE
(382 / 354 / 170 pages) plus earnings-call transcripts, an investor presentation
and results PDFs. Restricting the *valuation* to XBRL did not reduce the
document set — it grew it.

### Chat gets the audited figures too (`chat/qa.py`)

`ask(..., financials=...)` now takes the `FinancialHistory` and injects
`_audited_block(...)` into the prompt. Without it, chat answered a fundamentals
question by reading a number out of a retrieved PDF table — an OCR'd scan for
most Indian filings — while the DCF used the XBRL figure, so the two could
disagree about something as basic as revenue.

The block is authoritative **only for the years it covers**. The annual reports
routinely extend past the XBRL (Bikaji: reports to FY2026, XBRL to FY2024), and
for those periods the prompt directs the model back to the numbered sources and
tells it to name the document and period. So the documents are still doing real
fundamentals work, not just narrative.

Two details that matter:
- Figures are handed over **pre-formatted in crore/bn AND as the exact value**
  (`_format_money`). Asking the model to turn 29,347,432,000 into "2,935 crore"
  is arithmetic, and the model does not do arithmetic — Python does it first.
- The system prompt **exempts audited figures from `[n]` markers**. They have no
  numbered passage behind them, so requiring a citation would produce a dangling
  marker that `_resolve_citations` then strips.

`app.py` passes `financials=st.session_state["history"]`; chat still works when
it is `None`.

### Verified (network, no LLM calls)

| Company | Result |
|---|---|
| NVIDIA | `sec_xbrl`, 6y — SEC path untouched |
| TCS | `nse_indas_xbrl`, 6y, FY2024 revenue Rs2,40,893cr / PAT Rs45,908cr, debt-free |
| HUL | `nse_indas_xbrl`, 6y consolidated, FY2024 revenue Rs61,896cr / PAT Rs10,277cr |
| ITC | `nse_indas_xbrl`, 6y, margins 29-34% |
| Nestle India | `nse_indas_xbrl`, standalone basis, noted in the history |
| Bikaji | `nse_indas_xbrl`, 2y (listed Nov 2022 — genuinely all there is) |

Cross-check that validated the mapping: Bikaji FY2024 OCF `2,446,826,000` and
capex `-1,283,016,000` from XBRL match the old yfinance figures exactly.

**Known limits.** The NSE results API returned nothing after 2024-12-31 for any
symbol tested, so XBRL-derived years currently stop at FY2024 even though the
annual-report PDFs reach FY2026 — worth re-checking against live NSE. Older
filings (pre-2022) tag only the P&L, so those years have no cash-flow or
balance-sheet data; the latest year, which is what net debt comes from, is
complete. Extraction is XBRL-only: a company whose results exist solely as a
scanned PDF is not valued.
## Production hardening — shipped 2026-08-28 (user request)

Three things, aimed at "people can actually use this" rather than "this can be
demonstrated".

### 1. LiteLLM router (`fincopilot/llm.py`, rewritten)

`complete()` / `complete_json()` keep their signatures; everything under them
changed. Router groups mirror the FAST/WRITER blend:

| Group | Primary | Fallback group | Fallback model |
|---|---|---|---|
| `fincopilot-fast` | `openai/gpt-4.1-mini` | `fincopilot-fast-fallback` | `gemini/gemini-2.5-flash` |
| `fincopilot-writer` | `openai/gpt-4.1` | `fincopilot-writer-fallback` | `gemini/gemini-2.5-pro` |

**Load balancing is within a group** (add deployments under the same
`model_name`, each with its own `rpm`); **fallback is between groups**. They are
kept apart on purpose and a test asserts no group ever mixes providers —
splitting a report's prose across two vendors would make output vary run to run.
Tiers are matched so a fallback costs availability, not analysis.

`GEMINI_API_KEY` is read at router-build time. Absent, the router is built
OpenAI-only and logs that it has no fallback — it does not fail. **The user will
add the key later; cross-provider failover is therefore wired and unit-tested
but has not been exercised live.**

Retries/cooldowns are the router's (`ROUTER_NUM_RETRIES`, `ROUTER_ALLOWED_FAILS`,
`ROUTER_COOLDOWN_SECONDS`); `complete()` deliberately does **not** loop on top —
a second loop would multiply attempts and defeat the cooldown protecting a
struggling provider. `retries=` is still accepted for signature compatibility
and ignored. Real cost comes from `litellm.completion_cost` on the response, so
it reflects the model that actually served the call after a fallback.

LiteLLM logs the selected deployment dict at INFO on every call; that is
suppressed at router build, or a report buries every other log line.

### 2. Guardrails (`fincopilot/guardrails.py`, new)

**Layer 0 — LLM intent gate (`classify_query`), added on user feedback.** The
user's objection to a purely deterministic filter was correct: people rephrase.
Measured on four hand-written paraphrased attacks, the regex layer caught 1 of 4
("act as"); the classifier caught 4 of 4 at confidence 0.95, while allowing
genuine questions including blunt bearish ones ("Is this company overvalued and
heading for trouble?" -> research, 0.90).

Runs on FAST_MODEL before retrieval in `chat/qa.py::ask`. Categories: research /
advice / off_topic / injection / exfiltration; the last three block.
`advice` is ALLOWED — refusing "should I buy this" outright would be unhelpful
for a research tool; the answering prompt is what keeps it from becoming a
personal recommendation.

Three deliberate properties:
- **Fails open.** Classifier down, unparseable, or unknown category -> the
  question proceeds with `checked=False`, logged. A filter that takes chat down
  when the model hiccups is worse than the attack it prevents, and the layers
  underneath still apply.
- **Low-confidence blocks are not honoured** (`QUERY_BLOCK_CONFIDENCE`, 0.7). A
  false negative is cheap (a grounded, cited answer); a false positive tells a
  real analyst no.
- **Ordered first**, which is both the safe order and the cheap one: a refused
  question skips retrieval, reranking and answering, which cost far more than
  the check. A test asserts retrieval never runs for a blocked question.

Disable with `QUERY_CLASSIFIER_ENABLED=0`. Do NOT delete the deterministic
scans in favour of it — they cover secrets/PII where the target has an exact
shape and no judgment is needed, and the two layers fail differently.


Deterministic, in-process, no second model call. Applied at two boundaries:
`retrieve/pipeline.py::context_block` (untrusted document text) and
`llm.complete` (outbound prompt + inbound response).

- **Indirect prompt injection** — patterns require an imperative aimed at an
  assistant, so ordinary filing prose survives. The span is redacted, the
  document is **not** dropped: dropping it would let an attacker delete a
  company's filings from the index by poisoning one sentence.
- **Secrets** (OpenAI/Google/AWS keys, bearer tokens, PEM blocks) scanned both
  directions. **PII** (PAN, Aadhaar, SSN, card numbers) stripped from prompts
  and responses — Indian annual-report signature blocks are full of them.
- **Advice-like phrasing** is recorded in `findings`, never rewritten. The
  report must stay free to conclude SELL. There is a test asserting a bearish
  paragraph is left completely untouched.
- **`MAX_USD_PER_PROCESS`** enforced *before* each call, so it caps what can be
  spent rather than reporting what already was.

### Deferred: verifying Gemini failover (user will set the key later)

Wired and unit-tested; **never exercised live** as of 2026-08-28. When the key
is added, this is the check — it does not need a real outage, just a broken
primary:

```bash
# 1. Key present -> 4 groups and 2 fallback routes (currently 2 and 0)
python -c "from fincopilot import llm; print(sorted({d['model_name'] for d in llm._model_list()})); print(llm._fallback_map())"

# 2. Force the primary to fail and confirm Gemini actually serves the call.
#    A deliberately invalid OpenAI key exhausts both OpenAI deployments, so the
#    router crosses to the fallback group.
OPENAI_API_KEY=sk-invalid python -c "
from fincopilot import llm
print(llm.complete('Reply with exactly: FALLBACK OK', max_tokens=10))
print(llm.get_usage())"
```

Expect a real answer plus `fallbacks: 1` in the usage dict. `fallbacks: 0` with
a `None` answer means the fallback group is not registered — check the key name
is exactly `GEMINI_API_KEY`. Watch for a WARNING line reading
`call for fincopilot-fast fell back to gemini/...`, which is the positive signal.

Also worth one manual check that Gemini's JSON mode is strict enough for
`complete_json` on the structured extractors (segments, catalysts): the parser
already tolerates fences and preamble, which is why that tolerance was kept
rather than relying on `response_format`.

### 3. Report store (`fincopilot/report/store.py`, new)

SQLite at `data/reports.db` (override with `REPORT_DB_PATH`). Stores the
**ReportModel**, not rendered HTML/PDF — rendering is deterministic and free, so
a hit costs nothing and templates stay free to change without invalidating.

Fingerprint = filing content hashes + the reported figures + analyst overrides +
`REPORT_LOGIC_VERSION`. Hashing the figures as well as the documents is what
makes a **restatement invalidate with no new document**. Bump
`REPORT_LOGIC_VERSION` when a generator change should stop older reports being
served; a template change needs no bump.

`get()` refuses to serve a report whose `blocked` flag is set. `app.py` offers a
"Force regeneration" checkbox, and a cache hit consumes no rate-limit allowance
because nothing was generated. Sidebar "Service status" shows session spend,
reuse count and whether fallback is configured.

### Also fixed: silently dropped citations

The probe below surfaced a real pre-existing bug. The context block labels
passages `[SOURCE 3]`, and the model echoes that label back — often grouped, as
`[SOURCE 1, SOURCE 2]` — instead of the bare `[3]` the prompt asks for. Neither
form matched `_resolve_citations`' `\[(\d+)\]`, so **every citation on those
answers was dropped** and a correctly-grounded answer was shown with no sources.
`chat/qa.py::_normalise_markers` rewrites the label forms before resolution.

### Verified live (Bikaji, full pipeline)

Index: 5,483 chunks from 10 documents, FY2023-FY2026.

| Question | Answer |
|---|---|
| revenue FY2024 (XBRL-covered) | "INR 2,329.34 crore (23,293,366,000) as per the audited figures" |
| revenue FY2026 (beyond XBRL) | "INR 2,99,386.34 lakh according to the FY2026 Annual Report", citing Note 27 p314 |
| main risks | market/liquidity/credit risk, cited to FY2026 AR p251 and FY2025 AR p240 |

Router smoke-tested live: real calls served, cost metered ($0.000006/call on the
fast tier), fallback counter reads 0 when the primary serves.

## Access control + analytics (2026-08-28)

`fincopilot/access.py` — two entry modes. `own_key`: visitor pastes an OpenAI
key, billed to them, **not** rate limited here and **not** routed (the router's
fallback is the OWNER's Gemini key; failing a guest over to it would spend the
owner's money). `access_code`: shared secret, runs on owner credits, every limit
applies. `Grant.uses_owner_credits` is the single predicate everything keys off.

**`ACCESS_CODE` has no default and must never be committed** — the repo is
public and the code authorises spending. A test asserts the literal is absent
from source. Unset -> the code route does not exist (fails closed).
Set on the instance only: `REQUIRE_ACCESS=1` and `ACCESS_CODE=<the code>`.
The live value is deliberately NOT written down here — this file is public.

Visitor keys are held in `llm._credentials`, a **threading.local** — Streamlit
runs each session in its own thread, so a module global would leak one
visitor's key into another's concurrent request. Verified isolated under
concurrent threads.

`fincopilot/analytics.py` — SQLite at `data/usage.db`. Events: session_start,
company_load, question, report, error. **A question with zero citations is
recorded as a failure** (retrieval missed, or the filings genuinely lack it) —
that view is the point of the whole module. Never stores credentials (no column
exists for one); question text is scrubbed via `guardrails.scan_outbound` and
can be dropped entirely with `ANALYTICS_STORE_QUESTION_TEXT=0`. All writes are
wrapped: analytics must never become the failure they were meant to observe.
Owner dashboard is in the sidebar, gated on `uses_owner_credits`.

Smoke-tested in the browser: gate blocks, wrong code rejected and logged as a
failure, the correct code admits, usage panel appears, no exceptions.

## Key files

| Area | File |
|---|---|
| Valuation orchestration | `fincopilot/valuation/__init__.py` (`value_company`) |
| Assumptions (cached) | `fincopilot/valuation/assumptions.py` |
| Scenarios / blend / reverse | `fincopilot/valuation/{scenarios,blend,reverse}.py` |
| Segment forecast (cross-check) | `fincopilot/valuation/segments.py` |
| Thesis engine | `fincopilot/report/thesis.py` |
| Risks / competitive / forward | `fincopilot/report/{risks,competitive,monitoring}.py` |
| LLM routing + guardrails + cost | `fincopilot/llm.py`, `fincopilot/guardrails.py` |
| Report store (generate once per filing) | `fincopilot/report/store.py` |
| Report QA + severity gate | `fincopilot/report/qa.py` (`CHECK_SEVERITY`, `BLOCKING_SEVERITIES`) |
| Self-correction loop | `fincopilot/report/correction.py` (`CHECK_TO_COMPONENT`, `MAX_QA_RETRIES`) |
| Reliability scorecard | `fincopilot/report/reliability.py` |
| Analyst overrides | `fincopilot/valuation/overrides.py` + `overrides/{slug}.json` |
| Assumption agent (critique/revise) | `fincopilot/valuation/agent.py` |
| Robustness harness (run before core changes) | `scripts/robustness_check.py` |
| Report assembly | `fincopilot/report/builder.py` + `models.py` |
| Narrative sections | `fincopilot/report/sections.py` |
| Renderers | `fincopilot/report/template.html` (HTML) · `pdf.py` (PDF) |

## Verify recipe

```python
# venv/Scripts/python.exe - <<'PY'  (from D:\financial-copilot)
from fincopilot.resolve import resolve_company
from fincopilot.fundamentals import load_financials
from fincopilot.index import build_index
from fincopilot.retrieve import retrieve
from fincopilot.valuation import value_company
from fincopilot.report import build_report, render_pdf
c=resolve_company('NVIDIA'); h=load_financials(c); idx,ing=build_index(c)
ctx=retrieve('margins outlook growth risks competition', idx, top_k=8).context_block
v=value_company(c,h,qualitative_context=ctx)
r=build_report(c,h,v,ing,idx); render_pdf(r,'reports/SAMPLE_NVDA_report.pdf')
import fitz; fitz.open('reports/SAMPLE_NVDA_report.pdf').load_page(0).get_pixmap(dpi=100).save('.cache/p1.png')
# then Read .cache/p1.png to eyeball it
PY
```
