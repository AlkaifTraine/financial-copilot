"""
Post-download quality gate.

Search engines return plenty of documents that look right and are not: a
different company with a similar name, a broker's marketing deck, a scanned
image with no extractable text, an HTML error page served with a ``.pdf``
suffix. Anything that reaches the index becomes something the model can cite,
so rejection happens here, before indexing, and every rejection records a
reason that is shown in the UI.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..resolve import Company

log = logging.getLogger(__name__)


# Vocabulary that any genuine financial disclosure contains. Deliberately
# generic so it works for both US GAAP and Indian Ind-AS filings.
_FINANCIAL_TERMS = (
    "revenue",
    "net income",
    "operating income",
    "total assets",
    "cash flow",
    "balance sheet",
    "shareholders",
    "earnings per share",
    "gross margin",
    "liabilities",
    "fiscal year",
    "profit",
    "consolidated",
    "auditor",
)

# How much of a document to read for validation. Enough to judge relevance
# without paying to extract a 200-page filing twice.
_SAMPLE_PAGES = 16
_SAMPLE_CHARS = 60_000


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None = None
    page_count: int | None = None
    char_count: int | None = None


def _sample_pdf(path: Path) -> tuple[str, int]:
    """Return (sample text, total page count) for a PDF.

    Pages are sampled at an even stride across the whole document rather than
    taken from the front. Designed annual reports open with ten or more pages
    of full-bleed photography and pull quotes carrying almost no extractable
    text — sampling only the front of NVIDIA's FY2026 annual report matched a
    single financial term and rejected the best document available for the
    company. Striding guarantees the sample reaches the statements at the back.
    """
    import fitz  # PyMuPDF

    with fitz.open(path) as document:
        total_pages = document.page_count

        if total_pages <= _SAMPLE_PAGES:
            indices = range(total_pages)
        else:
            stride = total_pages / _SAMPLE_PAGES
            indices = [int(i * stride) for i in range(_SAMPLE_PAGES)]

        parts = [document.load_page(index).get_text() for index in indices]
        return "\n".join(parts), total_pages


def _sample_html(path: Path) -> tuple[str, int]:
    """Return (sample text, page count placeholder) for an HTML filing."""
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:_SAMPLE_CHARS], 0


def _company_tokens(company: Company) -> set[str]:
    """Distinctive lowercase tokens identifying the company.

    Corporate suffixes are stripped: matching on "limited" or "inc" would make
    the relevance check pass for essentially any filing.
    """
    stopwords = {
        "the", "inc", "inc.", "corp", "corp.", "corporation", "company", "co",
        "ltd", "ltd.", "limited", "plc", "holdings", "group", "sa", "nv", "ag",
        "and", "of",
    }
    tokens = {
        token
        for token in re.findall(r"[a-z]+", (company.name or "").lower())
        if token not in stopwords and len(token) >= 3
    }
    tokens.add(company.base_ticker.lower())
    return tokens


def validate(
    path: Path,
    company: Company,
    content_type: str,
    doc_type: str = "",
) -> ValidationResult:
    """Decide whether a downloaded document belongs in the index."""
    if not path.exists() or path.stat().st_size == 0:
        return ValidationResult(False, "file is empty")

    min_pages = config.MIN_PDF_PAGES_BY_TYPE.get(doc_type, config.MIN_PDF_PAGES)
    min_chars = config.MIN_PDF_CHARS_BY_TYPE.get(doc_type, config.MIN_PDF_CHARS)
    min_keywords = config.MIN_FINANCIAL_KEYWORD_HITS_BY_TYPE.get(
        doc_type, config.MIN_FINANCIAL_KEYWORD_HITS
    )

    try:
        if content_type == "html":
            text, page_count = _sample_html(path)
        else:
            text, page_count = _sample_pdf(path)
    except Exception as exc:
        return ValidationResult(False, f"unreadable ({type(exc).__name__})")

    char_count = len(text.strip())

    if content_type != "html" and page_count < min_pages:
        return ValidationResult(
            False, f"only {page_count} page(s)", page_count, char_count
        )

    # A near-empty text layer means a scanned or image-only document. Without
    # OCR there is nothing here to retrieve, so it is dropped rather than
    # silently contributing empty chunks.
    if char_count < min_chars:
        return ValidationResult(
            False,
            f"no extractable text layer ({char_count} chars) — likely a scanned document",
            page_count,
            char_count,
        )

    lowered = text.lower()

    keyword_hits = sum(1 for term in _FINANCIAL_TERMS if term in lowered)
    if keyword_hits < min_keywords:
        return ValidationResult(
            False,
            f"not a financial document (matched {keyword_hits} financial terms)",
            page_count,
            char_count,
        )

    tokens = _company_tokens(company)
    if tokens and not any(token in lowered for token in tokens):
        return ValidationResult(
            False,
            f"does not mention {company.name} — probably a different issuer",
            page_count,
            char_count,
        )

    return ValidationResult(True, None, page_count, char_count)
