"""
Hybrid index: dense vectors (FAISS) plus sparse lexical search (BM25).

Neither retriever alone is adequate on financial filings, and their failures
are close to complementary:

* **Dense** embeddings capture meaning, so "how profitable is the business"
  finds a passage about gross margin. They are poor at exact tokens — a ticker,
  "Item 1A", or the figure "215,938" — because those carry little semantic
  weight and get smoothed away.

* **BM25** matches exact terms and rare tokens, which is precisely what
  financial questions hinge on. It fails completely on paraphrase.

Both are kept, queried in parallel, and combined with Reciprocal Rank Fusion.

FAISS is used directly rather than through LangChain's wrapper: the index is
small enough that a flat inner-product index is exact and instant, and owning
the persistence format avoids breakage when the wrapper's serialisation
changes. It also removes the `allow_dangerous_deserialization=True` pickle load
the previous implementation relied on.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..chunk.models import Chunk
from .embed import Embedder, normalize

log = logging.getLogger(__name__)

# Keeps figures intact as single tokens: "215,938", "71.1%", "$4.90", "10-k".
_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9.,%$/-]*")


def tokenize(text: str) -> list[str]:
    """Lexical tokenisation tuned for filings."""
    return [token.strip(".,-/") for token in _TOKEN_PATTERN.findall(text.lower()) if token]


@dataclass
class HybridIndex:
    """Dense + sparse index over one company's chunks."""

    chunks: list[Chunk] = field(default_factory=list)
    model: str = ""
    company_name: str = ""

    _faiss: object | None = None
    _bm25: object | None = None

    # -- construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        company_name: str,
        *,
        embedder: Embedder | None = None,
        progress=None,
    ) -> "HybridIndex":
        import faiss

        if not chunks:
            raise ValueError("cannot build an index with no chunks")

        embedder = embedder or Embedder()
        vectors = normalize(embedder.embed_documents([c.text for c in chunks], progress=progress))

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        instance = cls(chunks=chunks, model=embedder.model, company_name=company_name)
        instance._faiss = index
        instance._build_bm25()
        log.info("built index: %d chunks, dim=%d", len(chunks), vectors.shape[1])
        return instance

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        # Indexed on the body rather than the embedded text: the context header
        # repeats the company and document name in every chunk, and feeding
        # that to BM25 would flatten the term statistics it ranks by.
        self._bm25 = BM25Okapi([tokenize(chunk.body) for chunk in self.chunks])

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss, str(path / "index.faiss"))

        with (path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                payload = {**chunk.to_metadata(), "text": chunk.text}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        (path / "meta.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "company_name": self.company_name,
                    "chunk_count": len(self.chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HybridIndex | None":
        import faiss

        path = Path(path)
        index_file = path / "index.faiss"
        chunks_file = path / "chunks.jsonl"
        meta_file = path / "meta.json"

        if not (index_file.exists() and chunks_file.exists() and meta_file.exists()):
            return None

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            chunks = [
                Chunk(**json.loads(line))
                for line in chunks_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            instance = cls(
                chunks=chunks,
                model=meta.get("model", ""),
                company_name=meta.get("company_name", ""),
            )
            instance._faiss = faiss.read_index(str(index_file))
            instance._build_bm25()
        except Exception as exc:
            log.warning("could not load index at %s: %s", path, exc)
            return None

        # A stale index built with a different embedding model would return
        # meaningless neighbours, so treat it as absent and force a rebuild.
        from .. import config

        if instance.model != config.EMBEDDING_MODEL:
            log.warning(
                "index was built with %s but %s is configured; rebuilding",
                instance.model,
                config.EMBEDDING_MODEL,
            )
            return None

        return instance

    # -- search -----------------------------------------------------------

    def dense(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        vector = normalize(query_vector.reshape(1, -1).astype(np.float32))
        scores, indices = self._faiss.search(vector, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        if not len(scores):
            return []
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    # -- metadata filtering -----------------------------------------------

    def matching(
        self,
        *,
        doc_types: set[str] | None = None,
        fiscal_years: set[int] | None = None,
        sections: set[str] | None = None,
        kinds: set[str] | None = None,
    ) -> set[int]:
        """Indices of chunks satisfying every supplied constraint.

        Lets a question about FY2026 search only FY2026 documents instead of
        competing against three years of near-identical filings.
        """
        selected = set()
        for index, chunk in enumerate(self.chunks):
            if doc_types and chunk.doc_type not in doc_types:
                continue
            if fiscal_years and chunk.fiscal_year not in fiscal_years:
                continue
            if kinds and chunk.kind not in kinds:
                continue
            if sections and not any(
                section.lower() in (chunk.section or "").lower() for section in sections
            ):
                continue
            selected.add(index)
        return selected

    @property
    def fiscal_years(self) -> list[int]:
        return sorted({c.fiscal_year for c in self.chunks if c.fiscal_year})
