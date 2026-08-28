"""
Grounded question answering.

The answering prompt is deliberately strict about two things:

* **Citations must be inline and specific.** Every factual sentence carries a
  `[n]` marker pointing at the source it came from. Those markers are resolved
  back to a document and page after generation, so a reader can open the exact
  filing page. A citation that cannot be resolved is worse than none, so
  unresolvable markers are stripped rather than shown.

* **Refusal is an acceptable answer.** If the retrieved passages do not contain
  the answer, saying so is correct behaviour. Financial users are far better
  served by "the filings do not disclose this" than by a confident guess.

The previous implementation asked the model to "cite source numbers like
[SOURCE 1]", but those numbers referred to nothing outside the prompt — the
user saw a citation that pointed nowhere and could not be checked.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .. import config
from ..guardrails import classify_query
from ..index.store import HybridIndex
from ..llm import complete
from ..retrieve import RetrievalResult, retrieve

log = logging.getLogger(__name__)

_SYSTEM = """You are an equity research analyst answering questions from a company's own regulatory filings.

Rules:
- Use ONLY the audited figures and the numbered sources provided. Never use outside knowledge.
- Cite inline with [n] after each factual claim drawn from a numbered source, where n is the source number.
- A figure taken from the AUDITED FIGURES block is already attributed: state the fiscal year with it and do NOT attach an [n] marker to it.
- Quote figures exactly as given, with their units and period. Never recompute, rescale or convert a figure yourself.
- If neither the audited figures nor the sources answer the question, say so plainly and state what is missing. Do not speculate.
- If sources disagree or cover different periods, say which period each figure belongs to.
- When the question does not name a fiscal period, lead with the MOST RECENT fiscal year available and use earlier years only as comparison. Do not open with an old year.
- Be concise and analytical. No filler, no restating the question."""

_PROMPT = """{history}Most recent fiscal year in the sources: {latest_fy}
{audited}
Sources:

{context}

Question: {question}

Answer the question, citing [n] inline for anything drawn from the numbered sources. If the question has no explicit period, lead with the most recent period available."""

_MAX_HISTORY_TURNS = 6


@dataclass
class Citation:
    number: int
    doc_title: str
    section: str | None
    page: int
    source_url: str
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    retrieval: RetrievalResult | None = None

    @property
    def grounded(self) -> bool:
        return bool(self.citations)


def _format_history(history: list[dict] | None) -> str:
    """Render recent turns so follow-up questions resolve pronouns correctly."""
    if not history:
        return ""

    lines = []
    for message in history[-_MAX_HISTORY_TURNS:]:
        role = "User" if message.get("role") == "user" else "Analyst"
        content = (message.get("content") or "").strip()
        if content:
            # Prior answers are truncated: they are context for interpreting the
            # next question, not evidence, and the sources carry the facts.
            lines.append(f"{role}: {content[:400]}")

    return "Conversation so far:\n" + "\n".join(lines) + "\n\n" if lines else ""


def _contextualise(question: str, history: list[dict] | None) -> str:
    """Rewrite a follow-up into a standalone question for retrieval.

    "What about last year?" carries no retrievable content on its own. Searching
    it verbatim returns noise, so it is resolved against the conversation first.
    """
    if not history:
        return question

    recent = [m for m in history[-4:] if (m.get("content") or "").strip()]
    if not recent:
        return question

    transcript = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Analyst'}: {(m.get('content') or '')[:300]}"
        for m in recent
    )

    rewritten = complete(
        "Rewrite the follow-up question as a standalone question that makes sense "
        "without the conversation. Keep it short. Return only the question.\n\n"
        f"{transcript}\n\nFollow-up: {question}\n\nStandalone question:",
        temperature=0.0,
        max_tokens=100,
    )
    return (rewritten or question).strip()


# Line items worth putting in front of the model, in statement order.
_AUDITED_FIELDS: list[tuple[str, str]] = [
    ("revenue", "revenue"),
    ("gross_profit", "gross profit"),
    ("operating_income", "operating income (EBIT)"),
    ("pretax_income", "profit before tax"),
    ("net_income", "net income"),
    ("operating_cash_flow", "operating cash flow"),
    ("capex", "capital expenditure"),
    ("depreciation_amortisation", "depreciation & amortisation"),
    ("total_debt", "total debt"),
    ("cash_and_equivalents", "cash & equivalents"),
    ("shareholders_equity", "shareholders' equity"),
    ("total_assets", "total assets"),
]


def _format_money(value: float, currency: str) -> str:
    """A readable figure *and* its exact value, so the model never rescales.

    Both representations are given deliberately. Asking the model to convert
    29,347,432,000 into "2,935 crore" is arithmetic, and the model does not do
    arithmetic here — so the conversion is done in Python and handed over
    pre-computed. Indian filings are discussed in crore, everything else in
    billions/millions, which is what a reader of each expects.
    """
    exact = f"{value:,.0f}"
    magnitude = abs(value)

    if currency == "INR":
        if magnitude >= 1e7:
            return f"INR {value / 1e7:,.2f} crore ({exact})"
        if magnitude >= 1e5:
            return f"INR {value / 1e5:,.2f} lakh ({exact})"
        return f"INR {exact}"

    for threshold, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m")):
        if magnitude >= threshold:
            return f"{currency} {value / threshold:,.2f}{suffix} ({exact})"
    return f"{currency} {exact}"


def _audited_block(financials) -> str:
    """The audited statement figures, as an authoritative prompt block.

    Chat and the valuation must not disagree about what revenue was. The
    valuation is built on concept-tagged XBRL; without this block chat would
    answer the same question by reading a number out of a retrieved PDF table —
    which, for an Indian filing, is frequently a scanned page whose OCR mangles
    both labels and digits, and which may be a standalone, restated or quarterly
    variant of the metric asked about.

    The block does not replace the documents. It covers only the fiscal years
    the XBRL covers; the annual reports routinely extend further (Bikaji's
    FY2025 and FY2026 reports are indexed while its XBRL stops at FY2024), and
    for those periods the model is told to fall back to the sources and say so.
    """
    if financials is None or not getattr(financials, "years", None):
        return ""

    years = financials.years
    covered = ", ".join(f"FY{y.fiscal_year}" for y in years)

    lines = [
        "",
        f"AUDITED FIGURES — from {financials.source_label}.",
        "These are the company's own tagged, audited numbers and are AUTHORITATIVE "
        f"for the fiscal years shown ({covered}). Use these exact values for those "
        "periods and do not recompute them from the passages below, which may "
        "contain quarterly, restated, standalone or non-GAAP variants of the same "
        "line item. For any period NOT listed here, use the numbered sources and "
        "say which document and period the figure came from.",
        "",
    ]

    for entry in years:
        parts = []
        for attribute, label in _AUDITED_FIELDS:
            value = getattr(entry, attribute, None)
            if value is not None:
                parts.append(f"{label} {_format_money(value, financials.currency)}")
        if parts:
            lines.append(
                f"FY{entry.fiscal_year} (year ended {entry.period_end}): "
                + "; ".join(parts)
            )

    return "\n".join(lines) + "\n"


# The context block labels passages "[SOURCE 3]", and models routinely echo that
# label back instead of the bare "[3]" the prompt asks for — often grouping them,
# as "[SOURCE 1, SOURCE 2]". Left alone, none of those match the [n] pattern, so
# every citation on the answer is dropped and a correctly-grounded answer is
# presented with no sources at all. Observed on real answers, and the failure is
# silent, which is the worst kind here. Normalise before resolving.
_SOURCE_LABEL = re.compile(r"\[\s*SOURCE\s+([\d\s,and]+?)\s*\]", re.I)


def _normalise_markers(text: str) -> str:
    """Rewrite "[SOURCE 1, SOURCE 2]" and "[SOURCE 3]" into "[1][2]" / "[3]"."""
    def expand(match: re.Match) -> str:
        numbers = re.findall(r"\d+", match.group(1))
        return "".join(f"[{n}]" for n in numbers)

    # Handle grouped forms first: "[SOURCE 1, SOURCE 2]" collapses to one match
    # whose body still contains every number.
    text = re.sub(
        r"\[\s*SOURCE\s+\d+(?:\s*,\s*(?:and\s+)?SOURCE\s+\d+)*\s*\]",
        lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(0))),
        text,
        flags=re.I,
    )
    return _SOURCE_LABEL.sub(expand, text)


def _resolve_citations(text: str, result: RetrievalResult) -> tuple[str, list[Citation]]:
    """Map inline [n] markers to real sources, dropping any that dangle."""
    text = _normalise_markers(text)
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})

    citations: list[Citation] = []
    valid: set[int] = set()

    for number in used:
        if not 1 <= number <= len(result.passages):
            continue
        chunk = result.passages[number - 1].chunk
        valid.add(number)
        citations.append(
            Citation(
                number=number,
                doc_title=chunk.doc_title,
                section=chunk.section,
                page=chunk.page,
                source_url=chunk.source_url,
                snippet=chunk.body[:400],
            )
        )

    # A marker pointing outside the source list would send the reader to a
    # citation that does not exist; remove it rather than display it.
    def strip_invalid(match: re.Match) -> str:
        return match.group(0) if int(match.group(1)) in valid else ""

    return re.sub(r"\[(\d+)\]", strip_invalid, text), citations


def ask(
    question: str,
    index: HybridIndex,
    *,
    history: list[dict] | None = None,
    financials=None,
    top_k: int | None = None,
) -> Answer:
    """Answer ``question`` from ``index`` with verifiable citations.

    ``history`` is the conversation so far. ``financials`` is the company's
    :class:`~fincopilot.fundamentals.models.FinancialHistory` when one is
    loaded: its audited figures are handed to the model as authoritative for
    the years they cover, so a fundamentals question is answered from the same
    numbers the valuation uses rather than from a number read out of a
    retrieved table. Chat still works without it.

    The question is classified before anything expensive happens. Placing the
    check first is both the safe order and the cheap one: a refused question
    never reaches retrieval, reranking or the answering model, which together
    cost far more than the classification.
    """
    if config.QUERY_CLASSIFIER_ENABLED:
        verdict = classify_query(question)
        if not verdict.allowed:
            return Answer(text=verdict.refusal, retrieval=RetrievalResult(question))

    search_query = _contextualise(question, history)

    result = retrieve(search_query, index, top_k=top_k or config.FINAL_TOP_K)

    if not result:
        return Answer(
            text=(
                "I could not find anything in the indexed filings that addresses "
                "that question."
            ),
            retrieval=result,
        )

    raw = complete(
        _PROMPT.format(
            history=_format_history(history),
            audited=_audited_block(financials),
            context=result.context_block,
            question=question,
            latest_fy=(f"FY{max(index.fiscal_years)}" if index.fiscal_years else "the latest year"),
        ),
        system=_SYSTEM,
        temperature=config.TEMPERATURE_FACTUAL,
        max_tokens=900,
    )

    if not raw:
        return Answer(
            text="The answering model is unavailable right now. Please try again.",
            retrieval=result,
        )

    text, citations = _resolve_citations(raw.strip(), result)
    return Answer(text=text, citations=citations, retrieval=result)
