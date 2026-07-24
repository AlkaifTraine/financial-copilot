"""
Download, content-addressed de-duplication, and the on-disk manifest.

The manifest is what the UI's "Source Documents" panel renders: the exact set
of documents the answers are grounded in, each with its original URL and a
local copy the user can download. Being able to open the source is the whole
trust story, so the manifest is written for humans as much as for the loader.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .. import config
from ..http_client import request, sec_get
from ..resolve import Company
from .models import ORIGIN_EDGAR, SourceDocument
from .validate import validate

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
_DOWNLOAD_CHUNK = 64 * 1024
_MAX_DOWNLOAD_BYTES = 120 * 1024 * 1024  # a 10-K is ~5-20MB; 120MB is a runaway


def company_dir(company: Company) -> Path:
    path = config.DATA_DIR / company.slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path(company: Company) -> Path:
    return company_dir(company) / MANIFEST_NAME


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _fetch(document: SourceDocument, destination: Path) -> tuple[bool, str | None]:
    """Stream a document to ``destination``. Returns (ok, error)."""
    # Route by host, not by origin: web search regularly surfaces sec.gov
    # Archives URLs, and fetching those without the SEC's mandated User-Agent
    # and rate limit earns a 403.
    is_sec = document.origin == ORIGIN_EDGAR or "sec.gov" in document.url.lower()
    getter = sec_get if is_sec else request
    response = getter(document.url, stream=True)

    if response is None:
        return False, "download failed (network error)"
    if response.status_code != 200:
        return False, f"download failed (HTTP {response.status_code})"

    digest = hashlib.sha256()
    written = 0

    try:
        with destination.open("wb") as handle:
            for chunk in response.iter_content(_DOWNLOAD_CHUNK):
                if not chunk:
                    continue
                written += len(chunk)
                if written > _MAX_DOWNLOAD_BYTES:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    return False, "file exceeds the size limit"
                digest.update(chunk)
                handle.write(chunk)
    except OSError as exc:
        destination.unlink(missing_ok=True)
        return False, f"write failed ({exc})"

    document.sha256 = digest.hexdigest()
    document.size_bytes = written

    # A PDF must start with the %PDF magic bytes. Search results frequently
    # point at an HTML error or consent page served under a .pdf URL, and
    # feeding one of those to the parser produces confident nonsense.
    if document.content_type == "pdf":
        with destination.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                destination.unlink(missing_ok=True)
                return False, "not a PDF (server returned an HTML page)"

    return True, None


def acquire(
    documents: list[SourceDocument],
    company: Company,
    *,
    progress=None,
) -> tuple[list[SourceDocument], list[SourceDocument]]:
    """Download, de-duplicate and validate. Returns (accepted, rejected)."""
    target_dir = company_dir(company)
    accepted: list[SourceDocument] = []
    rejected: list[SourceDocument] = []
    seen_hashes: dict[str, SourceDocument] = {}

    for index, document in enumerate(documents, start=1):
        if progress:
            progress(index, len(documents), document.label)

        # Download to a temporary name first: the final filename embeds the
        # content hash, which is unknown until the bytes have arrived.
        staging = target_dir / f".staging_{index}.tmp"
        ok, error = _fetch(document, staging)

        if not ok:
            document.rejected = True
            document.rejection_reason = error
            rejected.append(document)
            continue

        # Content-addressed de-duplication. The same annual report is commonly
        # reachable through several URLs; the previous implementation wrote
        # them all to one filename, so later copies clobbered earlier ones and
        # the manifest disagreed with what was actually on disk.
        if document.sha256 in seen_hashes:
            staging.unlink(missing_ok=True)
            original = seen_hashes[document.sha256]
            document.rejected = True
            document.rejection_reason = f"duplicate of {original.label}"
            rejected.append(document)
            continue

        final_path = target_dir / document.filename
        staging.replace(final_path)
        document.local_path = str(final_path)

        result = validate(final_path, company, document.content_type, document.doc_type)
        document.page_count = result.page_count
        document.char_count = result.char_count

        if not result.ok:
            final_path.unlink(missing_ok=True)
            document.local_path = None
            document.rejected = True
            document.rejection_reason = result.reason
            rejected.append(document)
            continue

        seen_hashes[document.sha256] = document
        accepted.append(document)

    accepted.sort(key=lambda d: d.sort_key, reverse=True)
    return accepted, rejected


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    company: Company,
    accepted: list[SourceDocument],
    rejected: list[SourceDocument],
) -> Path:
    from datetime import datetime, timezone

    payload = {
        "company": company.to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accepted": [d.to_dict() for d in accepted],
        "rejected": [d.to_dict() for d in rejected],
    }
    path = manifest_path(company)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_manifest(company: Company) -> dict | None:
    path = manifest_path(company)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    payload["accepted"] = [SourceDocument.from_dict(d) for d in payload.get("accepted", [])]
    payload["rejected"] = [SourceDocument.from_dict(d) for d in payload.get("rejected", [])]

    # A manifest whose files have been deleted is worse than no manifest: the
    # UI would offer downloads that 404 and the loader would find nothing.
    payload["accepted"] = [
        d for d in payload["accepted"] if d.local_path and Path(d.local_path).exists()
    ]
    return payload
