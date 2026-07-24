"""The chunk: the unit that gets embedded, retrieved, and cited."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    """One retrievable passage.

    Two text fields, deliberately:

      * ``text`` is what gets embedded and indexed — the passage prefixed with
        a context header naming the company, document, section and page.
      * ``body`` is the original passage, shown to the user as evidence.

    Embedding the header and displaying the body keeps retrieval accurate
    without cluttering the citation a human reads.
    """

    body: str
    text: str

    doc_id: str
    doc_title: str
    doc_type: str
    source_url: str
    origin: str

    page: int
    section: str | None
    kind: str                       # "text" or "table"
    chunk_index: int
    token_count: int

    fiscal_year: int | None = None
    fiscal_period: str | None = None

    @property
    def citation(self) -> str:
        """Short human-readable provenance string."""
        parts = [self.doc_title]
        if self.section:
            parts.append(self.section)
        if self.page:
            parts.append(f"p.{self.page}")
        return " - ".join(parts)

    def to_metadata(self) -> dict:
        """Metadata persisted alongside the vector."""
        payload = asdict(self)
        payload.pop("text")  # the embedded form is not needed at query time
        return payload
