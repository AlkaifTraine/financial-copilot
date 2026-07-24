"""
Index construction: company -> searchable hybrid index.

Chains the phases that precede retrieval — ingest, parse, chunk, embed — and
caches the result per company so a repeat load costs nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config
from ..chunk import chunk_document
from ..ingest import IngestResult, ingest
from ..parse import parse_document
from ..resolve import Company
from .store import HybridIndex

log = logging.getLogger(__name__)


def index_dir(company: Company) -> Path:
    return config.INDEX_DIR / company.slug


def build_index(
    company: Company,
    *,
    refresh: bool = False,
    progress=None,
) -> tuple[HybridIndex | None, IngestResult]:
    """Return a ready-to-query index for ``company``, building it if needed.

    Returns the index alongside the ingestion result so the caller can show
    which documents were used and which were rejected.
    """

    def report(stage: str, detail: str = "") -> None:
        log.info("[%s] %s", stage, detail)
        if progress:
            progress(stage, detail)

    result = ingest(company, refresh=refresh, progress=progress)
    if not result.ok:
        return None, result

    target = index_dir(company)

    if not refresh:
        cached = HybridIndex.load(target)
        if cached:
            report("index", f"loaded cached index ({len(cached.chunks)} chunks)")
            return cached, result

    # -- parse ------------------------------------------------------------
    chunks = []
    for position, document in enumerate(result.accepted, start=1):
        report("parse", f"[{position}/{len(result.accepted)}] {document.label}")

        parsed = parse_document(document)
        if parsed is None:
            document.notes.append("could not be parsed; excluded from the index")
            continue

        chunks.extend(chunk_document(parsed, company.name))

    if not chunks:
        result.notes.append(
            "Documents were downloaded but none could be parsed into searchable text."
        )
        return None, result

    # -- embed ------------------------------------------------------------
    def embed_progress(done: int, total: int) -> None:
        report("embed", f"{done}/{total} chunks embedded")

    report("embed", f"embedding {len(chunks)} chunks")
    index = HybridIndex.build(chunks, company.name, progress=embed_progress)
    index.save(target)

    report("index", f"index ready ({len(chunks)} chunks from {len(result.accepted)} documents)")
    return index, result
