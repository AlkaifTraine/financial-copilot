"""
Financial Copilot — RAG-based equity research assistant.

Package layout (each sub-package is one stage of the pipeline):

    resolve/       company name -> ticker, CIK, exchange, country
    ingest/        find + download + validate primary source documents
    parse/         PDF -> page-aware text and tables
    chunk/         structure-aware chunking with retrieval metadata
    index/         embeddings, FAISS (dense) + BM25 (sparse) stores
    retrieve/      query rewriting -> hybrid search -> RRF -> rerank
    chat/          grounded Q&A with verifiable citations
    fundamentals/  audited financials from SEC XBRL / yfinance
    valuation/     deterministic DCF, WACC, comps, sensitivity
    report/        typed report model -> HTML + PDF renderers
    eval/          retrieval + groundedness evaluation harness
"""

__version__ = "0.2.0"
