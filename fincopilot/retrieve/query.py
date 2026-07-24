"""
Query understanding: expansion and metadata filter inference.

A user's question and the language of a filing rarely share vocabulary. Someone
asks "is the business getting more profitable?"; the 10-K says "gross margin
was 71.1% compared with 75.0%". Embedding the question verbatim and hoping for
the best is where naive RAG loses most of its recall.

Two transformations run before search:

* **Multi-query expansion** — the question is rewritten into several
  paraphrases using the vocabulary a filing would actually use. Each variant is
  searched, and the results are fused. One extra fast-model call.

* **Filter inference** — an explicit fiscal year or document type in the
  question becomes a hard metadata filter. Asking about FY2024 should not have
  to out-rank three years of nearly identical filings; it should only search
  FY2024. This is cheap, deterministic, and needs no model call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .. import config
from ..ingest.models import ANNUAL, EARNINGS, PRESENTATION, QUARTERLY
from ..llm import complete_json

log = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    original: str
    variants: list[str] = field(default_factory=list)
    fiscal_years: set[int] = field(default_factory=set)
    doc_types: set[str] = field(default_factory=set)
    prefer_tables: bool = False

    @property
    def all_queries(self) -> list[str]:
        seen, ordered = set(), []
        for query in [self.original, *self.variants]:
            key = query.strip().lower()
            if key and key not in seen:
                seen.add(key)
                ordered.append(query.strip())
        return ordered


# Words implying the answer is a figure, which means table chunks should be
# favoured over narrative prose.
_NUMERIC_INTENT = re.compile(
    r"\b(revenue|margin|eps|earnings per share|income|profit|cash flow|capex|"
    r"debt|assets|liabilities|growth|how much|how many|total|percent|ratio|"
    r"opex|expenses|dividend|buyback|segment)\b",
    re.I,
)

_DOC_TYPE_HINTS = (
    (re.compile(r"\b(annual report|10-?k|full year|fiscal year)\b", re.I), ANNUAL),
    (re.compile(r"\b(quarter|10-?q|q[1-4])\b", re.I), QUARTERLY),
    (re.compile(r"\b(earnings (release|call)|press release)\b", re.I), EARNINGS),
    (re.compile(r"\b(presentation|slide|deck)\b", re.I), PRESENTATION),
)

_EXPANSION_PROMPT = """You rewrite questions about a company's financial filings so they match the wording used in SEC filings and annual reports.

Given the user's question, produce {n} alternative phrasings. Each should:
- use terminology a filing would actually use (e.g. "gross margin" not "profitability")
- vary which aspect is emphasised, so together they cover the question broadly
- stay a self-contained question or search phrase

Return JSON: {{"queries": ["...", "..."]}}

Question: {question}"""


def _infer_years(question: str, available: list[int]) -> set[int]:
    """Extract fiscal years mentioned in the question."""
    years: set[int] = set()

    for match in re.findall(r"\b(?:fy\s*)?(19\d{2}|20\d{2})\b", question, re.I):
        year = int(match)
        if year in available:
            years.add(year)

    # Relative references resolve against what is actually indexed.
    if available and re.search(r"\b(latest|most recent|current|this year|last year)\b", question, re.I):
        years.add(max(available))
        if re.search(r"\blast year\b", question, re.I) and len(available) > 1:
            years.add(sorted(available)[-2])

    return years


def _infer_doc_types(question: str) -> set[str]:
    return {doc_type for pattern, doc_type in _DOC_TYPE_HINTS if pattern.search(question)}


def plan_query(
    question: str,
    *,
    available_years: list[int] | None = None,
    expansions: int | None = None,
) -> QueryPlan:
    """Build a retrieval plan for ``question``."""
    plan = QueryPlan(original=question)
    plan.fiscal_years = _infer_years(question, available_years or [])
    plan.doc_types = _infer_doc_types(question)
    plan.prefer_tables = bool(_NUMERIC_INTENT.search(question))

    count = config.QUERY_EXPANSIONS if expansions is None else expansions
    if count <= 0:
        return plan

    payload = complete_json(
        _EXPANSION_PROMPT.format(n=count, question=question),
        temperature=0.3,
        max_tokens=300,
    )

    if isinstance(payload, dict):
        variants = payload.get("queries") or []
        plan.variants = [v for v in variants if isinstance(v, str)][:count]

    if not plan.variants:
        # Expansion is an enhancement, never a dependency: if the model call
        # fails the original question still gets searched.
        log.info("query expansion unavailable; using the original question only")

    return plan
