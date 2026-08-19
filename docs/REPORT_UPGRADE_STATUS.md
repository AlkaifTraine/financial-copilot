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

## Next batch (priority order)

1. **#4 "What is priced in" table** — a proper reverse-DCF that solves for the
   market-implied revenue CAGR, operating margin, FCF margin and terminal
   growth, shown side-by-side against our base case in one table. Extend
   `valuation/reverse.py` (currently solves only for year-1 growth). Render as a
   two-column comparison in the report.
2. **#17 Quantified risks** — each material risk with probability / financial
   impact / valuation impact / early-warning indicator. LLM pass grounded in the
   filings + the numbers; render as a table. Replace the descriptive risk prose.
3. **#3 Narrative → interpretation** — rewrite `report/sections.py` prompts so
   each section follows fact → interpretation → sustainability → investment
   implication, not description. Reduce repetition across sections (#24).
4. **#16 Segment forecasting** — where segment revenue exists (SEC XBRL has it
   for NVDA: Compute & Networking, Graphics), forecast segments and sum to
   total. New `valuation/segments.py`.
5. **#19/#20 Catalysts + monitoring dashboard** — upcoming catalysts (event /
   timing / direction / metric) and a "key things to watch" table
   (current / trend / expected / why it matters). Can extend `report/thesis.py`.
6. **#11 Management-says vs our-view**, **#13/#14 competitive & moat analysis**,
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
