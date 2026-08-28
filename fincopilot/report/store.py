"""
Persistent report store: generate once per filing, serve until the facts change.

A full report costs roughly $0.15 and a couple of minutes of model time. The
inputs it is built from — the company's filings and its reported numbers —
change a handful of times a year. Regenerating on every click therefore spends
real money to produce, at best, the same document, and at worst a *slightly
different* one, which is worse: two readers comparing notes on the same company
on the same day should not see two different reports.

So a finished report is stored and reused until the company actually publishes
something new. The cache key is a fingerprint of everything that can legitimately
change the output:

``documents``   the content hashes of the indexed filings. A new annual report,
                a new results release, a re-filed document — all change this.
``financials``  the reported figures themselves, so a restatement of a prior year
                invalidates the report even with no new document.
``overrides``   the analyst's pinned value drivers, because a human changing an
                assumption must produce a new report immediately.
``version``     :data:`REPORT_LOGIC_VERSION`, bumped whenever the generator
                changes in a way that should not serve pre-change documents.

What is stored is the :class:`~fincopilot.report.models.ReportModel`, not the
rendered HTML or PDF. Rendering is deterministic and free; re-rendering from the
stored model on each request costs nothing, keeps the templates free to improve
without invalidating anything, and avoids stashing large binaries in the
database.

SQLite is the right size of tool here. The deployment is a single Streamlit
process on one instance, the data is small and structured, and it needs to
survive a process restart and a `git reset --hard` deploy — which rules out
in-memory state and the gitignored `.cache/`. The file lives under ``data/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from .models import Evidence, KPI, ReportModel, Section

log = logging.getLogger(__name__)

# Bump when a generator change should stop older stored reports being served.
# A template or styling change does NOT need a bump — rendering happens fresh
# from the stored model every time.
REPORT_LOGIC_VERSION = "v8.1"

_DB_PATH = Path(config.get_secret("REPORT_DB_PATH") or (config.DATA_DIR / "reports.db"))
_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    fingerprint   TEXT PRIMARY KEY,
    ticker        TEXT NOT NULL,
    company_name  TEXT NOT NULL,
    as_of         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_served   TEXT,
    serve_count   INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL NOT NULL DEFAULT 0,
    blocked       INTEGER NOT NULL DEFAULT 0,
    payload       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS reports_ticker ON reports (ticker, created_at DESC);
"""


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.executescript(_SCHEMA)
    return connection


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def fingerprint(company, history, ingest=None, overrides=None) -> str:
    """A stable key for "this company, as currently reported".

    Deliberately built from *inputs*, never from the generated report: a key
    derived from the output could not detect that the inputs had changed.
    """
    parts: list[str] = [
        REPORT_LOGIC_VERSION,
        getattr(company, "ticker", ""),
    ]

    # The document set. Content hashes, so a re-upload of the same PDF under a
    # new URL does not invalidate, and an amended filing does.
    if ingest is not None:
        digests = sorted(
            (getattr(d, "sha256", "") or getattr(d, "url", ""))
            for d in getattr(ingest, "accepted", [])
        )
        parts.append("docs:" + ",".join(digests))

    # The reported numbers, so a restatement invalidates even with no new file.
    if history is not None:
        parts.append("src:" + getattr(history, "source", ""))
        for year in getattr(history, "years", []):
            parts.append(
                f"{year.fiscal_year}:{year.revenue}:{year.operating_income}:"
                f"{year.net_income}:{year.operating_cash_flow}:{year.capex}"
            )

    # An analyst changing a driver must produce a new report at once.
    if overrides:
        parts.append("ovr:" + json.dumps(overrides, sort_keys=True, default=str))

    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# (De)serialisation
# ---------------------------------------------------------------------------

_NESTED = {"kpis": KPI, "sections": Section}


def _to_payload(report: ReportModel) -> str:
    return json.dumps(asdict(report), default=str)


def _rebuild(cls, value):
    """Rebuild a dataclass from a dict, ignoring fields it no longer has.

    Tolerating unknown keys matters: a stored report written by an older build
    must not crash a newer one. Anything it cannot map is dropped, and the
    version in the fingerprint is what stops genuinely incompatible payloads
    being served at all.
    """
    if not is_dataclass(cls) or not isinstance(value, dict):
        return value
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in value.items() if k in known})


def _from_payload(payload: str) -> ReportModel | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        log.warning("stored report payload was not valid JSON")
        return None

    known = {f.name for f in fields(ReportModel)}
    data = {k: v for k, v in data.items() if k in known}

    for key, cls in _NESTED.items():
        if isinstance(data.get(key), list):
            data[key] = [_rebuild(cls, item) for item in data[key]]

    for section in data.get("sections", []):
        if isinstance(getattr(section, "evidence", None), list):
            section.evidence = [_rebuild(Evidence, e) for e in section.evidence]

    try:
        return ReportModel(**data)
    except TypeError as exc:
        log.warning("stored report could not be rebuilt: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get(key: str) -> ReportModel | None:
    """The stored report for ``key``, or None.

    A stored report that was *blocked* by the QA gate is never served: it was
    withheld for an integrity failure, and a cache must not be the way that
    decision gets reversed. Regeneration gets the correction loop another go.
    """
    try:
        with _lock, _connect() as connection:
            row = connection.execute(
                "SELECT payload, blocked FROM reports WHERE fingerprint = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if row["blocked"]:
                log.info("stored report %s was blocked by QA; regenerating", key[:12])
                return None
            connection.execute(
                "UPDATE reports SET serve_count = serve_count + 1, last_served = ? "
                "WHERE fingerprint = ?",
                (datetime.now(timezone.utc).isoformat(), key),
            )
    except sqlite3.Error as exc:
        # The store is an optimisation. If it is unavailable the app must still
        # work, just more expensively.
        log.warning("report store unavailable on read: %s", exc)
        return None

    report = _from_payload(row["payload"])
    if report is not None:
        log.info("served report %s from the store", key[:12])
    return report


def put(key: str, report: ReportModel, *, cost_usd: float = 0.0) -> None:
    """Store ``report`` under ``key``, replacing any previous entry."""
    try:
        with _lock, _connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO reports "
                "(fingerprint, ticker, company_name, as_of, created_at, "
                " last_served, serve_count, cost_usd, blocked, payload) "
                "VALUES (?, ?, ?, ?, ?, NULL, 0, ?, ?, ?)",
                (
                    key,
                    report.ticker,
                    report.company_name,
                    report.as_of,
                    datetime.now(timezone.utc).isoformat(),
                    float(cost_usd or 0.0),
                    int(bool(report.blocked)),
                    _to_payload(report),
                ),
            )
        log.info("stored report %s (%s, $%.3f)", key[:12], report.ticker, cost_usd)
    except sqlite3.Error as exc:
        log.warning("report store unavailable on write: %s", exc)


def invalidate(ticker: str) -> int:
    """Drop every stored report for ``ticker``. Returns rows removed."""
    try:
        with _lock, _connect() as connection:
            cursor = connection.execute(
                "DELETE FROM reports WHERE ticker = ?", (ticker,)
            )
            return cursor.rowcount
    except sqlite3.Error as exc:
        log.warning("report store unavailable on invalidate: %s", exc)
        return 0


def stats() -> dict:
    """Store-wide counters, for the UI and for judging whether it is paying off."""
    try:
        with _lock, _connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS reports, "
                "       COALESCE(SUM(serve_count), 0) AS serves, "
                "       COALESCE(SUM(cost_usd), 0) AS spent "
                "FROM reports"
            ).fetchone()
    except sqlite3.Error:
        return {"reports": 0, "serves": 0, "spent_usd": 0.0, "saved_usd": 0.0}

    reports = row["reports"] or 0
    serves = row["serves"] or 0
    spent = row["spent"] or 0.0
    # Every serve after the first is a generation that did not happen.
    average = (spent / reports) if reports else 0.0
    return {
        "reports": reports,
        "serves": serves,
        "spent_usd": round(spent, 3),
        "saved_usd": round(serves * average, 3),
    }
