"""
Retrieval ablation harness.

Runs the golden set through progressively richer retrieval configurations and
reports what each stage buys. The point is to be able to say *how much* hybrid
search and reranking help on this corpus, rather than asserting that they do.

Metrics, both computed without human labels:

* **Hit rate @k** - the share of questions whose expected figure appears
  anywhere in the retrieved context. This is recall of the answer-bearing
  passage, and it is the metric that determines whether the answering model can
  possibly be right.
* **MRR** - mean reciprocal rank of the first passage containing the expected
  figure. Rewards putting the right passage first, which matters because the
  answering model attends most to the earliest sources.

Run it with:  python -m fincopilot.eval.run NVIDIA
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field

from .. import config
from ..index import build_index
from ..resolve import resolve_company
from ..retrieve import retrieve
from .golden import GoldenQuestion, for_ticker

log = logging.getLogger(__name__)


@dataclass
class Configuration:
    """One point in the ablation."""

    name: str
    description: str
    dense: bool = True
    sparse: bool = True
    rerank: bool = True
    expansion: bool = True
    filters: bool = True


# Ordered from the previous implementation's behaviour to the current one, so
# each row shows the marginal contribution of one idea.
ABLATIONS = [
    Configuration(
        name="dense only",
        description="Plain vector similarity, no fusion or reranking (the previous system)",
        sparse=False, rerank=False, expansion=False, filters=False,
    ),
    Configuration(
        name="+ sparse (hybrid RRF)",
        description="BM25 fused with dense retrieval via Reciprocal Rank Fusion",
        rerank=False, expansion=False, filters=False,
    ),
    Configuration(
        name="+ query expansion",
        description="Multi-query paraphrasing before fusion",
        rerank=False, filters=False,
    ),
    Configuration(
        name="+ metadata filters",
        description="Fiscal year and document type inferred from the question",
        rerank=False,
    ),
    Configuration(
        name="+ reranking (full)",
        description="LLM cross-encoder reranking of fused candidates",
    ),
]


@dataclass
class QuestionResult:
    question: str
    hit: bool
    rank: int | None            # 1-indexed rank of the first matching passage
    top_citation: str = ""


@dataclass
class ConfigurationResult:
    name: str
    description: str
    hit_rate: float = 0.0
    mrr: float = 0.0
    seconds_per_query: float = 0.0
    questions: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _evaluate_question(
    question: GoldenQuestion,
    index,
    configuration: Configuration,
    top_k: int,
) -> QuestionResult:
    result = retrieve(
        question.question,
        index,
        top_k=top_k,
        use_dense=configuration.dense,
        use_sparse=configuration.sparse,
        use_rerank=configuration.rerank,
        use_expansion=configuration.expansion,
        use_filters=configuration.filters,
    )

    rank = None
    for position, passage in enumerate(result.passages, start=1):
        if question.matches(passage.chunk.body):
            rank = position
            break

    return QuestionResult(
        question=question.question,
        hit=rank is not None,
        rank=rank,
        top_citation=result.passages[0].citation if result.passages else "",
    )


def evaluate(
    index,
    questions: list[GoldenQuestion],
    configuration: Configuration,
    *,
    top_k: int | None = None,
) -> ConfigurationResult:
    top_k = top_k or config.FINAL_TOP_K
    outcome = ConfigurationResult(
        name=configuration.name, description=configuration.description
    )

    started = time.time()
    for question in questions:
        outcome.questions.append(
            _evaluate_question(question, index, configuration, top_k)
        )
    elapsed = time.time() - started

    hits = [q for q in outcome.questions if q.hit]
    outcome.hit_rate = len(hits) / len(questions) if questions else 0.0
    outcome.mrr = (
        sum(1.0 / q.rank for q in hits) / len(questions) if questions else 0.0
    )
    outcome.seconds_per_query = elapsed / len(questions) if questions else 0.0
    return outcome


def format_table(results: list[ConfigurationResult], top_k: int) -> str:
    """Markdown table, ready to paste into the README."""
    lines = [
        f"| Retrieval configuration | Hit rate @{top_k} | MRR | s/query |",
        "|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.hit_rate * 100:.0f}% | "
            f"{result.mrr:.3f} | {result.seconds_per_query:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval ablation evaluation")
    parser.add_argument("company", help="Company name or ticker")
    parser.add_argument("--top-k", type=int, default=config.FINAL_TOP_K)
    parser.add_argument("--json", help="Write full results to this path")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    company = resolve_company(arguments.company)
    questions = for_ticker(company.base_ticker)
    if not questions:
        raise SystemExit(
            f"No golden set defined for {company.ticker}. "
            f"Available: {', '.join(sorted(k for k in ('NVDA',)))}"
        )

    print(f"Company : {company.name} ({company.ticker})")
    print(f"Questions: {len(questions)}   top_k={arguments.top_k}\n")

    index, _ingest = build_index(company)
    if index is None:
        raise SystemExit("Could not build an index for this company.")
    print(f"Index   : {len(index.chunks):,} passages\n")

    results = []
    for configuration in ABLATIONS:
        print(f"  running: {configuration.name} ...", flush=True)
        results.append(evaluate(index, questions, configuration, top_k=arguments.top_k))

    print()
    print(format_table(results, arguments.top_k))
    print()

    # Per-question detail for the strongest configuration, so a failure can be
    # diagnosed rather than just counted.
    best = results[-1]
    print(f"Per-question detail ({best.name}):")
    for question in best.questions:
        marker = f"rank {question.rank}" if question.hit else "MISS"
        print(f"  [{marker:>7}] {question.question}")

    if arguments.json:
        payload = {
            "company": company.to_dict(),
            "top_k": arguments.top_k,
            "chunks": len(index.chunks),
            "results": [r.to_dict() for r in results],
        }
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nFull results written to {arguments.json}")


if __name__ == "__main__":
    main()
