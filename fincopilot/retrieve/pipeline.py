"""
The retrieval pipeline.

    question
       |
       v
  [plan]        expand into paraphrases; infer fiscal-year / doc-type filters
       |
       v
  [search]      every variant searched against BOTH dense and sparse indexes
       |
       v
  [fuse]        Reciprocal Rank Fusion over all result lists
       |
       v
  [filter]      apply inferred metadata constraints (softly)
       |
       v
  [rerank]      LLM scores each candidate against the question
       |
       v
  [expand]      attach neighbouring chunks for readable context
       |
       v
  passages + citations

Each stage can be switched off individually, which is what lets the evaluation
harness produce an ablation table (naive -> hybrid -> +rerank).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .. import config
from ..chunk.models import Chunk
from ..index.store import HybridIndex
from .fusion import reciprocal_rank_fusion
from .query import QueryPlan, plan_query
from .rerank import rerank as rerank_candidates

log = logging.getLogger(__name__)

# Multiplier applied to table chunks when the question asks for a figure.
_TABLE_BOOST = 1.25

# Applied when the question names no fiscal period, so an undated question
# lands on the current year rather than an arbitrary one.
_RECENCY_BOOST = 1.35
_PRIOR_YEAR_BOOST = 1.10

# If metadata filtering leaves fewer than this many candidates, it is dropped.
# An inferred filter is a heuristic; it should improve precision, never leave
# the question unanswerable.
_MIN_FILTERED_CANDIDATES = 5


@dataclass
class RetrievedPassage:
    chunk: Chunk
    score: float
    rank: int
    context: str = ""      # neighbouring text, for display

    @property
    def citation(self) -> str:
        return self.chunk.citation


@dataclass
class RetrievalResult:
    question: str
    passages: list[RetrievedPassage] = field(default_factory=list)
    plan: QueryPlan | None = None
    diagnostics: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.passages)

    @property
    def context_block(self) -> str:
        """Passages formatted for an answering prompt, with source labels."""
        parts = []
        for position, passage in enumerate(self.passages, start=1):
            parts.append(
                f"[SOURCE {position}] {passage.chunk.doc_title}"
                f"{' — ' + passage.chunk.section if passage.chunk.section else ''}"
                f" (page {passage.chunk.page})\n{passage.chunk.body}"
            )
        return "\n\n".join(parts)


def _neighbour_context(index: HybridIndex, chunk: Chunk, window: int = 1) -> str:
    """Text of the chunks immediately surrounding ``chunk`` in its document.

    Small chunks retrieve precisely but can read as fragments. Returning the
    neighbours restores enough surrounding narrative for a human to judge the
    evidence — the parent-document idea, without a second index.
    """
    neighbours = [
        other.body
        for other in index.chunks
        if other.doc_id == chunk.doc_id
        and abs(other.chunk_index - chunk.chunk_index) <= window
        and other.chunk_index != chunk.chunk_index
    ]
    return "\n\n".join(neighbours)


def retrieve(
    question: str,
    index: HybridIndex,
    *,
    top_k: int | None = None,
    use_expansion: bool = True,
    use_sparse: bool = True,
    use_dense: bool = True,
    use_rerank: bool = True,
    use_filters: bool = True,
    attach_context: bool = False,
) -> RetrievalResult:
    """Run the full retrieval pipeline for ``question``."""
    from ..index.embed import Embedder

    top_k = top_k or config.FINAL_TOP_K
    result = RetrievalResult(question=question)

    if not index.chunks:
        return result

    # -- plan -------------------------------------------------------------
    plan = plan_query(
        question,
        available_years=index.fiscal_years,
        expansions=None if use_expansion else 0,
    )
    result.plan = plan

    # -- search -----------------------------------------------------------
    rankings: list[list[int]] = []
    embedder = Embedder() if use_dense else None

    for query in plan.all_queries:
        if use_dense and embedder is not None:
            vector = embedder.embed_query(query)
            rankings.append([i for i, _s in index.dense(vector, config.DENSE_TOP_K)])
        if use_sparse:
            rankings.append([i for i, _s in index.sparse(query, config.SPARSE_TOP_K)])

    if not rankings:
        return result

    # -- fuse -------------------------------------------------------------
    boosts: dict[int, float] | None = None
    if plan.prefer_tables:
        boosts = {
            i: _TABLE_BOOST for i, chunk in enumerate(index.chunks) if chunk.kind == "table"
        }

    # When the question names no period, prefer the most recent filings. Three
    # years of near-identical 10-K language otherwise compete on equal terms,
    # and "what drove the change in gross margin?" is nearly always a question
    # about the latest year. Applied as a soft boost rather than a filter, so
    # historical comparisons remain reachable.
    if not plan.fiscal_years and index.fiscal_years:
        latest = max(index.fiscal_years)
        boosts = boosts or {}
        for position, chunk in enumerate(index.chunks):
            if chunk.fiscal_year == latest:
                boosts[position] = boosts.get(position, 1.0) * _RECENCY_BOOST
            elif chunk.fiscal_year == latest - 1:
                boosts[position] = boosts.get(position, 1.0) * _PRIOR_YEAR_BOOST

    fused = reciprocal_rank_fusion(rankings, boosts=boosts)

    # -- filter -----------------------------------------------------------
    filtered = fused
    if use_filters and (plan.fiscal_years or plan.doc_types):
        allowed = index.matching(
            doc_types=plan.doc_types or None,
            fiscal_years=plan.fiscal_years or None,
        )
        candidate = [pair for pair in fused if pair[0] in allowed]
        if len(candidate) >= _MIN_FILTERED_CANDIDATES:
            filtered = candidate
        else:
            log.info(
                "metadata filter matched only %d chunks; searching unfiltered",
                len(candidate),
            )

    shortlist = filtered[: config.RERANK_CANDIDATES]

    # -- rerank -----------------------------------------------------------
    if use_rerank and shortlist:
        scored = rerank_candidates(
            question,
            [(i, index.chunks[i]) for i, _s in shortlist],
            top_k=top_k,
        )
    else:
        scored = [(i, s) for i, s in shortlist[:top_k]]

    # -- assemble ---------------------------------------------------------
    for rank, (chunk_index, score) in enumerate(scored, start=1):
        chunk = index.chunks[chunk_index]
        result.passages.append(
            RetrievedPassage(
                chunk=chunk,
                score=score,
                rank=rank,
                context=_neighbour_context(index, chunk) if attach_context else "",
            )
        )

    result.diagnostics = {
        "queries": plan.all_queries,
        "rankings": len(rankings),
        "fused_candidates": len(fused),
        "after_filter": len(filtered),
        "reranked": use_rerank,
        "fiscal_years": sorted(plan.fiscal_years),
        "doc_types": sorted(plan.doc_types),
        "prefer_tables": plan.prefer_tables,
    }
    return result
