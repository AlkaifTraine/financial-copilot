"""
Cross-encoder style reranking.

Embedding search scores *similarity*, which is not the same as *answering the
question*. A passage restating the question's own vocabulary can sit above the
passage containing the actual figure, because the first looks more like the
query while the second merely contains the answer.

A reranker reads the question and each candidate **together** and scores how
well the passage answers it. That joint reading is what a bi-encoder cannot do,
since it must encode the query and the document independently.

Implemented against the fast chat model rather than a dedicated cross-encoder
(such as a BGE reranker) on purpose: those pull in torch and a few hundred
megabytes of weights, which does not fit the deployment target's memory
ceiling. One batched call scores every candidate at once, so the latency cost
is a single round trip.
"""

from __future__ import annotations

import logging

from .. import config
from ..chunk.models import Chunk
from ..llm import complete_json

log = logging.getLogger(__name__)

# Candidates are truncated before scoring: relevance is judged well from the
# opening of a passage, and full text for 20 candidates would be a large prompt
# for little gain.
_SNIPPET_CHARS = 700

_PROMPT = """You are ranking passages from company financial filings by how well each one answers a question.

Question: {question}

Score every passage from 0 to 10:
  10 = directly and completely answers the question
   7 = contains most of the answer, or the specific figures asked for
   4 = related background, but does not answer it
   0 = irrelevant

Judge only whether the passage answers THIS question. Do not reward passages for merely repeating the question's wording.

Passages:
{passages}

Return JSON: {{"scores": [{{"id": <passage id>, "score": <0-10>}}, ...]}}
Include every passage id exactly once."""


def rerank(
    question: str,
    candidates: list[tuple[int, Chunk]],
    *,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Score ``candidates`` against ``question``.

    Args:
        candidates: ``(chunk_index, chunk)`` pairs to score.

    Returns:
        ``(chunk_index, score)`` sorted best first. Falls back to the incoming
        order if the model call fails, so retrieval degrades to fusion-only
        rather than breaking.
    """
    if not candidates:
        return []

    top_k = top_k or config.FINAL_TOP_K

    passages = "\n\n".join(
        f"[{position}] ({chunk.kind}, {chunk.doc_title}"
        f"{', ' + chunk.section if chunk.section else ''})\n"
        f"{chunk.body[:_SNIPPET_CHARS]}"
        for position, (_index, chunk) in enumerate(candidates)
    )

    payload = complete_json(
        _PROMPT.format(question=question, passages=passages),
        temperature=0.0,
        max_tokens=700,
    )

    scores: dict[int, float] = {}
    if isinstance(payload, dict):
        for entry in payload.get("scores", []) or []:
            try:
                position = int(entry["id"])
                if 0 <= position < len(candidates):
                    scores[position] = float(entry["score"])
            except (KeyError, TypeError, ValueError):
                continue

    if not scores:
        log.info("reranking unavailable; keeping fusion order")
        return [(index, 0.0) for index, _chunk in candidates[:top_k]]

    # Candidates the model omitted keep a neutral-low score rather than being
    # dropped, so an incomplete response cannot silently discard evidence.
    ranked = sorted(
        ((candidates[position][0], scores.get(position, 2.0)) for position in range(len(candidates))),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_k]
