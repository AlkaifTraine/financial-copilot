"""
Parsed document representation.

A parsed document is an ordered list of *blocks*, each carrying the page it
came from and the section heading it sits under. That metadata is the whole
point: it survives into the chunker, into the vector store, and finally into a
citation the user can click and verify.

Blocks are typed because tables and prose need different downstream handling —
a table must never be split across chunks, while a long paragraph must be.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

BLOCK_TEXT = "text"
BLOCK_TABLE = "table"
BLOCK_HEADING = "heading"


@dataclass
class Block:
    """One contiguous piece of a document."""

    kind: str
    text: str
    page: int                      # 1-indexed, matches what a reader sees
    section: str | None = None     # nearest preceding heading

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class ParsedDocument:
    """A source document after extraction, before chunking."""

    doc_id: str                    # stable short id, derived from content hash
    title: str                     # e.g. "FY2026 Annual Report (10-K)"
    doc_type: str
    source_url: str
    origin: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    local_path: str | None = None
    page_count: int = 0
    blocks: list[Block] = field(default_factory=list)

    @property
    def tables(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == BLOCK_TABLE]

    @property
    def prose(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == BLOCK_TEXT]

    @property
    def sections(self) -> list[str]:
        seen: list[str] = []
        for block in self.blocks:
            if block.section and block.section not in seen:
                seen.append(block.section)
        return seen

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    def to_dict(self) -> dict:
        return asdict(self)
