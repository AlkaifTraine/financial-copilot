"""
The document record shared by every ingestion source.

Both the EDGAR path and the web-search path emit :class:`SourceDocument`, so
everything downstream — download, validation, parsing, the UI's source panel,
and report citations — works against one shape regardless of provenance.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

ANNUAL = "annual_report"
QUARTERLY = "quarterly_report"
EARNINGS = "earnings_release"
PRESENTATION = "investor_presentation"

DOC_TYPES = (ANNUAL, QUARTERLY, EARNINGS, PRESENTATION)

DOC_TYPE_LABELS = {
    ANNUAL: "Annual Report",
    QUARTERLY: "Quarterly Report",
    EARNINGS: "Earnings Release",
    PRESENTATION: "Investor Presentation",
}

# Provenance. Ranked: a filing straight from the regulator outranks a PDF found
# by a search engine, and this ordering is used to break ties during merging.
ORIGIN_EDGAR = "sec_edgar"
ORIGIN_NSE = "nse_exchange"
ORIGIN_WEB = "web_search"

# Higher wins when the same disclosure turns up from two places. A document
# served by the regulator or the exchange is the authoritative copy; a search
# engine's copy of it is a convenience mirror.
ORIGIN_TRUST = {ORIGIN_EDGAR: 3, ORIGIN_NSE: 2, ORIGIN_WEB: 1}


@dataclass
class SourceDocument:
    """One primary-source document belonging to a company."""

    doc_type: str
    title: str
    url: str                          # canonical, user-visible source URL
    origin: str

    fiscal_year: int | None = None
    fiscal_period: str | None = None  # "FY" or "Q1".."Q4"
    filed_date: str | None = None     # ISO date
    form_type: str | None = None      # "10-K", "10-Q", "8-K", "20-F", ...
    accession: str | None = None      # EDGAR accession number

    content_type: str = "pdf"         # "pdf" or "html"

    # Populated once the document has been fetched and checked.
    local_path: str | None = None
    sha256: str | None = None
    page_count: int | None = None
    char_count: int | None = None
    size_bytes: int | None = None

    rejected: bool = False
    rejection_reason: str | None = None

    notes: list[str] = field(default_factory=list)

    # -- presentation ------------------------------------------------------

    @property
    def label(self) -> str:
        """Human-readable name, e.g. 'FY2026 Annual Report (10-K)'."""
        period = self.fiscal_period or "FY"
        year = self.fiscal_year or "?"
        prefix = f"{period}{year}" if period.startswith("Q") else f"FY{year}"
        base = f"{prefix} {DOC_TYPE_LABELS.get(self.doc_type, self.doc_type)}"
        return f"{base} ({self.form_type})" if self.form_type else base

    @property
    def filename(self) -> str:
        """Deterministic, collision-free local filename.

        The old pipeline named files ``{year}_annual_report.pdf``, which caused
        two distinct 2026 NVIDIA filings to be written to the same path — the
        second silently overwrote the first. Including a short content-address
        makes distinct documents distinct on disk.
        """
        period = self.fiscal_period or "FY"
        stem = f"{self.fiscal_year or 'unknown'}_{period}_{self.doc_type}"
        if self.sha256:
            stem += f"_{self.sha256[:8]}"
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
        return f"{stem}.{'htm' if self.content_type == 'html' else 'pdf'}"

    @property
    def sort_key(self) -> tuple:
        """Newest first, then by provenance trust."""
        quarter = 0
        if self.fiscal_period and self.fiscal_period.startswith("Q"):
            quarter = int(self.fiscal_period[1])
        return (
            self.fiscal_year or 0,
            quarter,
            ORIGIN_TRUST.get(self.origin, 0),
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.label}

    @classmethod
    def from_dict(cls, payload: dict) -> "SourceDocument":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})
