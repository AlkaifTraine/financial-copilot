# Report-quality upgrade — status & handoff

This tracks the "AI summary → institutional-quality equity research" upgrade.
Read this first in a new session; do **not** re-explore or rebuild the platform.

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
  commit: `venv/Scripts/python.exe -m pytest tests/ -q` must pass, and a report
  must render (see Verify below).
- **Verify by looking.** Generate an NVDA report, rasterize page 1 with PyMuPDF,
  and actually view it — several bugs were only visible in the render.
- `gh` is not installed; commit locally, `git push` uses cached credentials.
- Committed portfolio artifact: `reports/SAMPLE_NVDA_report.{pdf,html}`.

## Done (shipped, verified)

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

## Next batch (priority order)

1. **#3 Narrative → interpretation** — rewrite `report/sections.py` prompts so
   each section follows fact → interpretation → sustainability → investment
   implication, not description. Reduce repetition across sections (#24).
2. **#16 Segment forecasting** — where segment revenue exists (SEC XBRL has it
   for NVDA: Compute & Networking, Graphics), forecast segments and sum to
   total. New `valuation/segments.py`.
3. **#19/#20 Catalysts + monitoring dashboard** — upcoming catalysts (event /
   timing / direction / metric) and a "key things to watch" table
   (current / trend / expected / why it matters). Can extend `report/thesis.py`.
4. **#11 Management-says vs our-view**, **#13/#14 competitive & moat analysis**,
   **#23 stricter citation QA**, **#22 full consistency-check pass**.

## Key files

| Area | File |
|---|---|
| Valuation orchestration | `fincopilot/valuation/__init__.py` (`value_company`) |
| Assumptions (cached) | `fincopilot/valuation/assumptions.py` |
| Scenarios / blend / reverse | `fincopilot/valuation/{scenarios,blend,reverse}.py` |
| Thesis engine | `fincopilot/report/thesis.py` |
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
