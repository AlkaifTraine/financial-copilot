"""
Structure-aware chunking with contextual headers.

The previous implementation was a fixed-size split::

    RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    metadata = {"source": file_name}

Three things go wrong with that on financial filings:

1. **Tables get cut in half.** A 1,500-character window lands mid-table, so one
   chunk holds the row labels and the next holds the numbers. Neither can
   answer a question, and the model will happily pair a label from one with a
   figure from the other.

2. **Chunks lose their referent.** A passage reading "increased 12% year over
   year, driven by higher utilisation" is unretrievable — nothing in it says
   *what* increased, for *which company*, in *which year*. Embedding it as-is
   places it nowhere useful in vector space.

3. **Metadata is a filename.** With only `source`, a citation cannot point at a
   page, retrieval cannot be filtered to a fiscal year, and the user cannot
   verify anything.

This module fixes all three: tables are emitted whole, every chunk carries a
deterministic context header naming company / document / section / page, and
the full provenance travels with the chunk.

On the context header: prepending situating context before embedding is the
core idea behind contextual retrieval. The published version of the technique
generates that context with an LLM per chunk, which for a 200-page 10-K means
thousands of extra model calls. The structural facts — which company, which
filing, which section, which page — are already known exactly from parsing, so
they are written deterministically here: no cost, no latency, no hallucination
risk, and it captures most of the benefit.
"""

from __future__ import annotations

import logging

from .. import config
from ..parse.models import BLOCK_TABLE, BLOCK_TEXT, Block, ParsedDocument
from .models import Chunk

log = logging.getLogger(__name__)

_ENCODER = None


def _encoder():
    """Lazily build the tokenizer shared by every chunking call."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        # cl100k_base is the tokenizer used by the text-embedding-3 family, so
        # counting with it makes the budget exact rather than approximate.
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def _context_header(
    document: ParsedDocument,
    company_name: str,
    section: str | None,
    page: int,
) -> str:
    """Situating prefix embedded with every chunk."""
    parts = [company_name, document.title]
    if section:
        parts.append(section)
    if page:
        parts.append(f"page {page}")
    return "[" + " | ".join(parts) + "]"


def _split_table(markdown: str, budget: int) -> list[str]:
    """Split an oversized table by rows, repeating the header in each part.

    Without the repeated header the second half of a split income statement is
    a grid of unlabelled numbers.
    """
    lines = markdown.split("\n")
    if len(lines) < 3:
        return [markdown]

    header, separator, *rows = lines
    preamble = f"{header}\n{separator}"
    preamble_tokens = count_tokens(preamble)

    parts: list[str] = []
    current: list[str] = []
    current_tokens = preamble_tokens

    for row in rows:
        row_tokens = count_tokens(row)
        if current and current_tokens + row_tokens > budget:
            parts.append("\n".join([preamble, *current]))
            current = []
            current_tokens = preamble_tokens
        current.append(row)
        current_tokens += row_tokens

    if current:
        parts.append("\n".join([preamble, *current]))

    # A single row can exceed the budget on its own (a wide table whose cells
    # hold whole sentences), leaving a part still far over budget. Fall back to
    # a hard character split so no chunk can blow past the embedding model's
    # input limit.
    bounded: list[str] = []
    for part in parts or [markdown]:
        if count_tokens(part) <= budget * 2:
            bounded.append(part)
            continue
        # ~4 characters per token is a safe conservative ratio for English.
        span = budget * 4
        bounded.extend(part[i : i + span] for i in range(0, len(part), span))

    return bounded


def _split_oversized(block: Block, budget: int) -> list[Block]:
    """Break a single over-long block into budget-sized pieces.

    Chunk boundaries are normally decided by accumulating blocks, which assumes
    each block is smaller than the budget. That assumption does not hold: an
    EDGAR filing sometimes places a whole subsection inside one element, giving
    a single 7,700-token block that would otherwise be emitted intact and blow
    past the embedding input limit.

    Splitting prefers sentence boundaries, falling back to a hard character cut
    only for text with no sentence structure at all.
    """
    import re

    if count_tokens(block.text) <= budget:
        return [block]

    pieces: list[Block] = []
    current: list[str] = []
    current_tokens = 0

    def emit() -> None:
        if current:
            pieces.append(Block(block.kind, " ".join(current), block.page, block.section))

    for sentence in re.split(r"(?<=[.!?])\s+", block.text):
        if not sentence:
            continue

        sentence_tokens = count_tokens(sentence)

        # A single sentence over budget (tables of contents, exhibit indexes)
        # has no internal boundary to split on, so cut it by characters.
        if sentence_tokens > budget:
            emit()
            current, current_tokens = [], 0
            span = budget * 4  # ~4 characters per token
            for start in range(0, len(sentence), span):
                pieces.append(
                    Block(block.kind, sentence[start : start + span], block.page, block.section)
                )
            continue

        if current and current_tokens + sentence_tokens > budget:
            emit()
            current, current_tokens = [], 0

        current.append(sentence)
        current_tokens += sentence_tokens

    emit()
    return pieces or [block]


def _flush(
    buffer: list[Block],
    document: ParsedDocument,
    company_name: str,
    index: int,
) -> Chunk | None:
    """Turn accumulated prose blocks into a chunk."""
    if not buffer:
        return None

    body = " ".join(block.text for block in buffer).strip()
    if not body:
        return None

    tokens = count_tokens(body)
    if tokens < config.CHUNK_MIN_TOKENS:
        # Page furniture: running headers, page numbers, stray captions.
        return None

    page = buffer[0].page
    section = buffer[0].section
    header = _context_header(document, company_name, section, page)

    return Chunk(
        body=body,
        text=f"{header}\n{body}",
        doc_id=document.doc_id,
        doc_title=document.title,
        doc_type=document.doc_type,
        source_url=document.source_url,
        origin=document.origin,
        page=page,
        section=section,
        kind=BLOCK_TEXT,
        chunk_index=index,
        token_count=tokens,
        fiscal_year=document.fiscal_year,
        fiscal_period=document.fiscal_period,
    )


def chunk_document(document: ParsedDocument, company_name: str) -> list[Chunk]:
    """Convert a parsed document into retrievable chunks."""
    chunks: list[Chunk] = []
    buffer: list[Block] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        chunk = _flush(buffer, document, company_name, len(chunks))
        if chunk:
            chunks.append(chunk)

        # Carry the tail of this chunk into the next so a statement split
        # across the boundary is still retrievable from either side.
        overlap: list[Block] = []
        overlap_tokens = 0
        for block in reversed(buffer):
            block_tokens = count_tokens(block.text)
            if overlap_tokens + block_tokens > config.CHUNK_OVERLAP_TOKENS:
                break
            overlap.insert(0, block)
            overlap_tokens += block_tokens

        buffer = overlap
        buffer_tokens = overlap_tokens

    # Oversized prose blocks are broken up first, so the accumulation loop can
    # rely on every block fitting inside the budget.
    blocks: list[Block] = []
    for block in document.blocks:
        if block.kind == BLOCK_TABLE:
            blocks.append(block)
        else:
            blocks.extend(_split_oversized(block, config.CHUNK_TARGET_TOKENS))

    for block in blocks:
        # -- tables: emitted whole, never merged with surrounding prose ----
        if block.kind == BLOCK_TABLE:
            flush()
            buffer, buffer_tokens = [], 0

            header = _context_header(document, company_name, block.section, block.page)
            for part in _split_table(block.text, config.TABLE_MAX_TOKENS):
                chunks.append(
                    Chunk(
                        body=part,
                        text=f"{header}\n{part}",
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        doc_type=document.doc_type,
                        source_url=document.source_url,
                        origin=document.origin,
                        page=block.page,
                        section=block.section,
                        kind=BLOCK_TABLE,
                        chunk_index=len(chunks),
                        token_count=count_tokens(part),
                        fiscal_year=document.fiscal_year,
                        fiscal_period=document.fiscal_period,
                    )
                )
            continue

        # -- section boundary: start a new chunk ---------------------------
        # A chunk spanning two sections dilutes both, and its section metadata
        # would be wrong for half its content.
        if buffer and block.section != buffer[-1].section:
            flush()
            buffer, buffer_tokens = [], 0

        block_tokens = count_tokens(block.text)

        if buffer and buffer_tokens + block_tokens > config.CHUNK_TARGET_TOKENS:
            flush()

        buffer.append(block)
        buffer_tokens += block_tokens

    flush()

    log.info(
        "chunked %s -> %d chunks (%d tables)",
        document.title,
        len(chunks),
        sum(1 for c in chunks if c.kind == BLOCK_TABLE),
    )
    return chunks
