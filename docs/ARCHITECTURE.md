# Financial Copilot — Architecture

This document is the map of the system. It is kept in sync with the code as
each phase lands, and it is written to be read top-to-bottom by someone who has
never seen the repo.

---

## 1. What the product does

Given a company name, the system:

1. **Resolves** the name to a ticker, exchange, and (for US issuers) an SEC CIK.
2. **Fetches** that company's primary source documents — annual reports,
   quarterly reports, earnings releases, investor presentations — and lets the
   user download the exact PDFs it read.
3. **Answers questions** about those filings with citations that point to a
   specific document and page.
4. **Values the company** with a discounted cash flow model built from audited
   financials, publishing every assumption and where it came from.
5. **Generates a one-click equity research report** combining the above.

---

## 2. Design principles

These three rules explain most of the structural decisions below.

### 2.1 The language model never does arithmetic

Every number in the valuation is computed in NumPy from structured inputs. The
model's role is confined to *proposing and justifying assumptions* (revenue
growth, margin trajectory), and each proposal is clamped to validated bounds
before it reaches the math.

*Why:* LLMs produce plausible-looking arithmetic that is wrong in ways that are
hard to spot. A valuation whose numbers cannot be reproduced is worthless.

### 2.2 Numbers come from structured sources; prose comes from RAG

Revenue, operating income, share count, debt and cash are pulled from SEC XBRL
`companyfacts` (the same tagged data the SEC itself indexes) or, for non-US
issuers, from structured statement APIs. RAG over PDFs is used for the things
it is genuinely good at: management commentary, risk factors, strategy,
competitive positioning.

*Why:* asking a model to read "$60,922" out of a mangled PDF table is the single
largest hallucination source in financial RAG. The old pipeline did exactly
this, in `utils/financial_data_extractor.py`.

### 2.3 Every claim is verifiable

Chunks carry `(document, page, section)` metadata from ingestion through to the
answer. The UI renders the supporting snippet next to each claim and links to
the downloadable source PDF.

*Why:* trust is the product. A finance user who cannot check a number will not
use the tool twice.

---

## 3. Pipeline

```
company name
     |
     v
[resolve]      name -> ticker / CIK / exchange / country
     |
     +---------------------------+
     |                           |
     v                           v
[ingest]                    [fundamentals]
 EDGAR primary               SEC XBRL companyfacts (US)
 web search fallback         yfinance statements (non-US)
 relevance validation        market data: price, shares, beta
 content-hash dedupe              |
     |                           |
     v                           |
[parse]                          |
 PyMuPDF text + tables           |
 page numbers preserved          |
 section detection               |
     |                           |
     v                           |
[chunk]                          |
 structure-aware splitting       |
 contextual headers              |
 rich metadata                   |
     |                           |
     v                           |
[index]                          |
 FAISS dense + BM25 sparse       |
     |                           |
     v                           v
[retrieve]  <-------------> [valuation]
 query expansion             DCF / WACC / comps
 hybrid + RRF fusion         sensitivity grid
 LLM reranking               assumption ledger
 parent expansion                 |
     |                            |
     +--------------+-------------+
                    |
                    v
              [chat]   [report]
           citations   typed model -> HTML + PDF
```

---

## 4. Retrieval design

The retrieval stack is the technical core. Each stage exists to fix a specific
failure of the stage before it.

| Stage | Technique | Failure it addresses |
|---|---|---|
| Chunking | Structure-aware, table-preserving | Fixed-size splitting cuts through income statements, destroying row/column meaning |
| Chunk context | Deterministic header (`company / doc / FY / section`) prepended before embedding | A bare chunk saying "increased 12%" is unretrievable — nothing says *what* increased, or when |
| Query rewriting | Multi-query expansion + finance vocabulary normalisation | User says "how profitable"; the filing says "gross margin" |
| Sparse search | BM25 | Dense embeddings are weak on exact tokens: tickers, "Item 1A", "$60,922" |
| Dense search | `text-embedding-3-small` + FAISS | BM25 misses paraphrase and concept-level matches |
| Fusion | Reciprocal Rank Fusion | Dense and sparse scores are not comparable; RRF combines by rank, needing no calibration |
| Reranking | LLM cross-encoder scoring | Bi-encoder retrieval scores similarity, not answer-relevance |
| Expansion | Parent-document retrieval | Small chunks retrieve precisely but read out of context; return the surrounding section |

Ablation results for each stage are produced by the evaluation harness
(`fincopilot/eval/`) and published in the README.

---

## 5. Repository layout

```
app.py                  Streamlit entry point (thin: UI only)
fincopilot/
  config.py             every tunable, single source of truth
  resolve/              company identity resolution
  ingest/               document discovery, download, validation, manifest
  parse/                PDF -> page-aware text and tables
  chunk/                structure-aware chunking
  index/                embeddings + FAISS + BM25
  retrieve/             query rewriting, hybrid search, fusion, reranking
  chat/                 grounded Q&A
  fundamentals/         XBRL + market data -> normalised statements
  valuation/            DCF, WACC, comps, sensitivity, assumption ledger
  report/               typed report model + HTML and PDF renderers
  eval/                 retrieval and groundedness evaluation
tests/                  unit tests (valuation math is fully covered)
docs/                   this file and design notes
```

### Legacy modules

The original `tools/`, `utils/`, `backend/` packages and `main.py` are the
pre-rewrite implementation. They are retired progressively as each phase lands,
and removed in Phase 9. `git log` preserves them.

---

## 6. Cost and performance

Per company, first load (cold):

| Item | Cost |
|---|---|
| Document download | free |
| Embedding a ~200-page 10-K (~1,500 chunks) | ~$0.02 |
| Report generation (sectioned, ~10 calls) | ~$0.03 |
| Chat question (expansion + rerank + answer) | ~$0.002 |

Indexes are cached per company on disk, so repeat loads cost nothing.
