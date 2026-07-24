"""
Reciprocal Rank Fusion.

The problem RRF solves: FAISS returns cosine similarities (roughly 0.2-0.9)
and BM25 returns unbounded term-frequency scores (0 to 30+, depending on corpus
statistics). The two are not comparable, and normalising them requires
calibration that shifts with every corpus.

RRF sidesteps this by discarding the scores and using only the *ranks*:

    score(d) = sum over retrievers of  1 / (k + rank(d))

A document ranked highly by several retrievers beats one ranked highly by only
one. The constant k (60 by convention) damps the influence of the very top
ranks so a single retriever cannot dominate the fused ordering.

Because it consumes ranks alone, the same function also fuses the result lists
of the several query paraphrases without any extra machinery.
"""

from __future__ import annotations

from collections import defaultdict

from .. import config


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int | None = None,
    weights: list[float] | None = None,
    boosts: dict[int, float] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranked candidate lists into one ordering.

    Args:
        rankings: each entry is a list of chunk indices, best first.
        k: RRF damping constant.
        weights: optional per-ranking weight, same length as ``rankings``.
        boosts: optional multiplier per chunk index, applied after fusion —
            used to favour table chunks on questions that want a figure.

    Returns:
        ``(chunk_index, score)`` pairs sorted by descending score.
    """
    k = config.RRF_K if k is None else k
    weights = weights or [1.0] * len(rankings)

    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, index in enumerate(ranking):
            scores[index] += weight / (k + rank + 1)

    if boosts:
        for index in list(scores):
            scores[index] *= boosts.get(index, 1.0)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
