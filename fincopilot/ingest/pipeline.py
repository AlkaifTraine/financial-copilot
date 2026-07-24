"""
Ingestion orchestrator: company -> validated set of source documents.

Source selection strategy
-------------------------
EDGAR and web search are complementary rather than redundant, so both run when
both are available:

  * EDGAR supplies the *filings* — 10-K, 10-Q, earnings 8-K — as HTML, with
    permanent regulator-hosted URLs. Best provenance, best table fidelity.
  * Web search supplies documents that are never filed with a regulator: the
    designed annual-report PDF and investor presentation decks. It is also the
    only source for companies outside SEC jurisdiction.

When both return a document covering the same period and type, the EDGAR copy
wins on provenance and the web copy is dropped as a duplicate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..resolve import Company
from . import edgar, registry, websearch
from .models import ORIGIN_EDGAR, ORIGIN_TRUST, SourceDocument

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    company: Company
    accepted: list[SourceDocument] = field(default_factory=list)
    rejected: list[SourceDocument] = field(default_factory=list)
    manifest_path: Path | None = None
    from_cache: bool = False
    sources_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.accepted)

    def by_type(self, doc_type: str) -> list[SourceDocument]:
        return [d for d in self.accepted if d.doc_type == doc_type]


def _merge(*groups: list[SourceDocument]) -> list[SourceDocument]:
    """Combine candidate lists, preferring the most trustworthy provenance.

    Two documents are considered the same when they cover the same type and
    period. The EDGAR 10-K and the IR-site annual report PDF for FY2026 are the
    same disclosure in two wrappers, so only the better-sourced one is fetched.
    """
    best: dict[tuple, SourceDocument] = {}
    ordering: list[tuple] = []

    for group in groups:
        for document in group:
            key = (document.doc_type, document.fiscal_year, document.fiscal_period)
            incumbent = best.get(key)
            if incumbent is None:
                best[key] = document
                ordering.append(key)
            elif ORIGIN_TRUST.get(document.origin, 0) > ORIGIN_TRUST.get(incumbent.origin, 0):
                best[key] = document

    merged = [best[key] for key in ordering]

    # Drop undated web documents once any dated document of the same type
    # exists. A document whose fiscal year cannot be determined is unbounded in
    # age, and one such file — a historical NVIDIA annual report picked up by
    # web search — put figures from fiscal 2011 into the index alongside the
    # FY2026 filings. Asked what drove the change in gross margin, the system
    # answered with 2011 versus 2010. Undated content cannot be filtered by
    # year, so it cannot be kept honest.
    dated_types = {
        d.doc_type for d in merged if d.fiscal_year is not None
    }
    kept = [
        d for d in merged
        if d.fiscal_year is not None or d.doc_type not in dated_types
    ]

    dropped = len(merged) - len(kept)
    if dropped:
        log.info("dropped %d undated document(s) superseded by dated filings", dropped)

    return kept


def ingest(
    company: Company,
    *,
    refresh: bool = False,
    progress=None,
) -> IngestResult:
    """Assemble the validated document set for ``company``.

    Args:
        refresh: re-run discovery even when a manifest already exists.
        progress: optional ``callable(stage: str, detail: str)`` for UI updates.
    """

    def report(stage: str, detail: str = "") -> None:
        log.info("[%s] %s", stage, detail)
        if progress:
            progress(stage, detail)

    # -- cache ------------------------------------------------------------
    if not refresh:
        cached = registry.read_manifest(company)
        if cached and cached["accepted"]:
            report("cache", f"{len(cached['accepted'])} documents already downloaded")
            return IngestResult(
                company=company,
                accepted=cached["accepted"],
                rejected=cached["rejected"],
                manifest_path=registry.manifest_path(company),
                from_cache=True,
                sources_used=sorted({d.origin for d in cached["accepted"]}),
            )

    result = IngestResult(company=company)

    # -- discovery --------------------------------------------------------
    edgar_docs: list[SourceDocument] = []
    if company.is_sec_filer:
        if config.is_sec_configured():
            report("discover", "querying SEC EDGAR")
            edgar_docs = edgar.discover(company)
            if edgar_docs:
                result.sources_used.append(ORIGIN_EDGAR)
            report("discover", f"EDGAR returned {len(edgar_docs)} filings")
        else:
            result.notes.append(
                "SEC EDGAR was skipped because SEC_USER_AGENT has no contact "
                "email. Set it in .env to enable official filings and audited "
                "XBRL financials."
            )
            report("discover", "EDGAR skipped (SEC_USER_AGENT not configured)")

    report("discover", "searching investor-relations sites")
    web_docs = websearch.discover(company)
    report("discover", f"web search returned {len(web_docs)} candidates")

    candidates = _merge(edgar_docs, web_docs)

    if not candidates:
        result.notes.append(
            f"No documents could be found for {company.name} ({company.ticker})."
        )
        return result

    # -- download + validate ----------------------------------------------
    def download_progress(index: int, total: int, label: str) -> None:
        report("download", f"[{index}/{total}] {label}")

    accepted, rejected = registry.acquire(
        candidates, company, progress=download_progress
    )

    result.accepted = accepted
    result.rejected = rejected
    result.sources_used = sorted({d.origin for d in accepted})
    result.manifest_path = registry.write_manifest(company, accepted, rejected)

    report(
        "done",
        f"{len(accepted)} documents accepted, {len(rejected)} rejected",
    )
    return result
