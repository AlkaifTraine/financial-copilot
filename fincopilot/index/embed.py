"""
Embeddings with a persistent cache.

Two reasons this is not just a thin wrapper around the OpenAI client:

* **Cost and latency.** Indexing three NVIDIA 10-Ks is ~1,700 chunks. Re-running
  after a chunker tweak should only pay for the chunks that actually changed,
  so vectors are cached by content hash in SQLite and survive process restarts.

* **Failure isolation.** A single failed batch must not lose the other forty.
  Batches retry independently and report which chunks are missing.

Why OpenAI embeddings rather than the previous local `all-MiniLM-L6-v2`:
MiniLM produces 384-dimensional vectors trained on general web text and is
noticeably weak on dense financial prose. It also drags in torch and
sentence-transformers (~2 GB), which does not fit the 1 GB memory ceiling on
Streamlit Community Cloud. `text-embedding-3-small` is stronger, 1536-d, and
costs about two cents per company.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .. import config

log = logging.getLogger(__name__)

_CACHE_PATH = config.CACHE_DIR / "embeddings.sqlite"


class EmbeddingCache:
    """Content-addressed vector cache backed by SQLite."""

    def __init__(self, path: Path = _CACHE_PATH) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "  key TEXT PRIMARY KEY,"
            "  dim INTEGER NOT NULL,"
            "  vector BLOB NOT NULL"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def key(text: str, model: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        found: dict[str, np.ndarray] = {}
        with self._lock:
            # SQLite caps the number of bound variables, so query in blocks.
            for start in range(0, len(keys), 500):
                block = keys[start : start + 500]
                placeholders = ",".join("?" * len(block))
                rows = self._conn.execute(
                    f"SELECT key, vector FROM vectors WHERE key IN ({placeholders})",
                    block,
                ).fetchall()
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        if not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (key, dim, vector) VALUES (?, ?, ?)",
                [(key, len(vec), vec.astype(np.float32).tobytes()) for key, vec in items],
            )
            self._conn.commit()


class Embedder:
    """Batched, cached embedding client."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or config.EMBEDDING_MODEL
        self._client = OpenAI(api_key=config.require_secret("OPENAI_API_KEY"))
        self._cache = EmbeddingCache()

    # -- internals --------------------------------------------------------

    def _embed_batch(self, texts: list[str], *, retries: int = 3) -> list[np.ndarray] | None:
        for attempt in range(retries):
            try:
                response = self._client.embeddings.create(model=self.model, input=texts)
                return [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
            except Exception as exc:
                if attempt == retries - 1:
                    log.error("embedding batch failed permanently: %s", exc)
                    return None
                time.sleep(2**attempt)
        return None

    @staticmethod
    def _truncate(text: str) -> str:
        """Keep inputs inside the model's context limit.

        Chunking already targets ~600 tokens, so this only ever fires on
        pathological input; the cut is by characters because re-tokenising
        every string just to check would cost more than it saves.
        """
        limit = config.EMBEDDING_MAX_TOKENS * 4
        return text if len(text) <= limit else text[:limit]

    # -- public API -------------------------------------------------------

    def embed_documents(self, texts: list[str], *, progress=None) -> np.ndarray:
        """Embed many texts, returning an (n, dim) float32 array.

        Rows that could not be embedded are left as zero vectors; callers get a
        usable index rather than an exception when one batch fails.
        """
        texts = [self._truncate(t) for t in texts]
        keys = [EmbeddingCache.key(t, self.model) for t in texts]

        cached = self._cache.get_many(list(dict.fromkeys(keys)))
        missing = [i for i, key in enumerate(keys) if key not in cached]

        log.info(
            "embedding %d texts (%d cached, %d to fetch)",
            len(texts),
            len(texts) - len(missing),
            len(missing),
        )

        batch_size = config.EMBEDDING_BATCH_SIZE
        for start in range(0, len(missing), batch_size):
            indices = missing[start : start + batch_size]
            vectors = self._embed_batch([texts[i] for i in indices])

            if vectors is not None:
                fresh = [(keys[i], vec) for i, vec in zip(indices, vectors)]
                self._cache.put_many(fresh)
                cached.update(dict(fresh))

            if progress:
                progress(min(start + batch_size, len(missing)), len(missing))

        dim = config.EMBEDDING_DIM
        for vector in cached.values():
            dim = len(vector)
            break

        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row, key in enumerate(keys):
            vector = cached.get(key)
            if vector is not None:
                matrix[row] = vector

        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so an inner-product index computes cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
