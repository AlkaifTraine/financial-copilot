"""Document parsing: source file -> page- and section-tagged blocks."""

from __future__ import annotations

import logging

from ..ingest.models import SourceDocument
from .models import BLOCK_HEADING, BLOCK_TABLE, BLOCK_TEXT, Block, ParsedDocument

log = logging.getLogger(__name__)

__all__ = [
    "Block",
    "ParsedDocument",
    "BLOCK_TEXT",
    "BLOCK_TABLE",
    "BLOCK_HEADING",
    "parse_document",
]


def parse_document(source: SourceDocument) -> ParsedDocument | None:
    """Parse a downloaded :class:`SourceDocument`.

    Returns ``None`` when the document cannot be parsed; one unreadable file
    must not abort indexing of an entire company.
    """
    if not source.local_path:
        return None

    from .html import parse_html
    from .pdf import parse_pdf

    parser = parse_html if source.content_type == "html" else parse_pdf

    kwargs = dict(
        doc_id=(source.sha256 or "unknown")[:12],
        title=source.label,
        doc_type=source.doc_type,
        source_url=source.url,
        origin=source.origin,
        fiscal_year=source.fiscal_year,
        fiscal_period=source.fiscal_period,
        form_type=source.form_type,
    )

    try:
        return parser(source.local_path, **kwargs)
    except Exception as exc:
        log.warning("failed to parse %s: %s", source.label, exc, exc_info=True)
        return None
