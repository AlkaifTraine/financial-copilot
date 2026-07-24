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
from ..index.store import HybridIndex
from ..llm import complete
from ..retrieve import RetrievalResult, retrieve

log = logging.getLogger(__name__)

_SYSTEM = """You are an equity research analyst answering questions from a company's own regulatory filings.

Rules:
- Use ONLY the numbered sources provided. Never use outside knowledge.
- Cite inline with [n] after each factual claim, where n is the source number.
- Quote figures exactly as they appear, with their units and period.
- If the sources do not answer the question, say so plainly and state what is missing. Do not speculate.
- If sources disagree or cover different periods, say which period each figure belongs to.
- Be concise and analytical. No filler, no restating the question."""

_PROMPT = """{history}Sources:

{context}

Question: {question}

Answer using only the sources above, citing [n] inline."""

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


def _resolve_citations(text: str, result: RetrievalResult) -> tuple[str, list[Citation]]:
    """Map inline [n] markers to real sources, dropping any that dangle."""
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
    top_k: int | None = None,
) -> Answer:
    """Answer ``question`` from ``index`` with verifiable citations."""
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
            context=result.context_block,
            question=question,
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
