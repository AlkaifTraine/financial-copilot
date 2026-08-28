"""
Usage analytics: what people actually do with this, and what breaks.

Without this there is no way to answer "is anyone using it, and what fails for
them" except by guessing — and a claim about usage that cannot be quantified is
worth less than no claim at all. Every event carries whether it succeeded,
how long it took, and what it cost, because the interesting questions are
mostly about failure: which companies come back with no audited XBRL, which
questions the retrieval cannot answer, where people give up.

Design decisions worth stating:

* **Never store credentials.** A visitor-supplied API key is never written
  here, in any field, under any setting. The event payload is built from a
  fixed set of columns, so there is no path by which a key reaches the file
  even if one were passed in by mistake.

* **Question text is scrubbed, and optional.** Questions are the most useful
  thing in the log — they are where the product lessons are — but they are also
  the only free text a visitor writes. They pass through
  :func:`fincopilot.guardrails.scan_outbound` before storage, so secrets and
  personal identifiers are stripped, and
  ``ANALYTICS_STORE_QUESTION_TEXT=0`` reduces them to a length and a category.

* **Sessions are opaque and random.** A session id groups one visitor's events
  so a journey can be followed; it is not derived from anything about them and
  identifies nobody.

* **Failure here is never allowed to surface.** Analytics are diagnostics, not
  the product. Every write is wrapped: a locked or missing database degrades to
  a log line, never to an error in front of a user.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

_lock = threading.Lock()

# Event names, kept as constants so a typo cannot silently create a new stream.
SESSION_START = "session_start"
COMPANY_LOAD = "company_load"
QUESTION = "question"
REPORT = "report"
ERROR = "error"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT    NOT NULL,
    session_id   TEXT    NOT NULL,
    event        TEXT    NOT NULL,
    access_mode  TEXT,
    ticker       TEXT,
    company      TEXT,
    ok           INTEGER,
    duration_ms  INTEGER,
    cost_usd     REAL,
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS events_ts      ON events (ts DESC);
CREATE INDEX IF NOT EXISTS events_session ON events (session_id, ts);
CREATE INDEX IF NOT EXISTS events_kind    ON events (event, ts DESC);
"""


def _db_path() -> Path:
    return Path(config.ANALYTICS_DB_PATH)


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.executescript(_SCHEMA)
    return connection


def record(
    event: str,
    *,
    session_id: str,
    access_mode: str | None = None,
    ticker: str | None = None,
    company: str | None = None,
    ok: bool = True,
    duration_ms: int | None = None,
    cost_usd: float | None = None,
    detail: dict | None = None,
) -> None:
    """Write one event. Never raises."""
    if not config.ANALYTICS_ENABLED:
        return

    try:
        payload = json.dumps(detail, default=str)[:4000] if detail else None
        with _lock, _connect() as connection:
            connection.execute(
                "INSERT INTO events (ts, session_id, event, access_mode, ticker, "
                " company, ok, duration_ms, cost_usd, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    session_id, event, access_mode, ticker, company,
                    int(bool(ok)),
                    int(duration_ms) if duration_ms is not None else None,
                    float(cost_usd) if cost_usd is not None else None,
                    payload,
                ),
            )
    except Exception as exc:
        # Diagnostics must never become the failure they were meant to observe.
        log.warning("analytics write failed (%s): %s", event, exc)


def record_question(
    question: str,
    *,
    session_id: str,
    access_mode: str | None = None,
    ticker: str | None = None,
    ok: bool = True,
    duration_ms: int | None = None,
    cost_usd: float | None = None,
    category: str | None = None,
    citations: int | None = None,
    refused: bool = False,
) -> None:
    """Record a question, scrubbing it first.

    ``category`` comes from the guardrail classifier, so the log shows the mix
    of genuine research questions against off-topic and hostile ones without
    needing the text to work that out.
    """
    from .guardrails import scan_outbound

    text = (question or "").strip()
    detail: dict = {
        "length": len(text),
        "category": category,
        "citations": citations,
        "refused": refused,
    }
    if config.ANALYTICS_STORE_QUESTION_TEXT and text:
        detail["question"] = scan_outbound(text[:600]).text

    record(
        QUESTION,
        session_id=session_id, access_mode=access_mode, ticker=ticker,
        ok=ok, duration_ms=duration_ms, cost_usd=cost_usd, detail=detail,
    )


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------

def summary(days: int = 30) -> dict:
    """Headline counters for the owner's dashboard."""
    empty = {
        "sessions": 0, "companies_loaded": 0, "questions": 0, "reports": 0,
        "failures": 0, "cost_usd": 0.0, "top_companies": [], "days": days,
    }
    try:
        with _lock, _connect() as connection:
            since = f"-{int(days)} days"
            row = connection.execute(
                "SELECT COUNT(DISTINCT session_id) AS sessions, "
                "  SUM(event = ?) AS companies, SUM(event = ?) AS questions, "
                "  SUM(event = ?) AS reports, SUM(ok = 0) AS failures, "
                "  COALESCE(SUM(cost_usd), 0) AS cost "
                "FROM events WHERE ts >= datetime('now', ?)",
                (COMPANY_LOAD, QUESTION, REPORT, since),
            ).fetchone()

            top = connection.execute(
                "SELECT ticker, COUNT(*) AS n FROM events "
                "WHERE event = ? AND ticker IS NOT NULL "
                "  AND ts >= datetime('now', ?) "
                "GROUP BY ticker ORDER BY n DESC LIMIT 10",
                (COMPANY_LOAD, since),
            ).fetchall()
    except Exception as exc:
        log.warning("analytics summary unavailable: %s", exc)
        return empty

    return {
        "sessions": row["sessions"] or 0,
        "companies_loaded": row["companies"] or 0,
        "questions": row["questions"] or 0,
        "reports": row["reports"] or 0,
        "failures": row["failures"] or 0,
        "cost_usd": round(row["cost"] or 0.0, 3),
        "top_companies": [(r["ticker"], r["n"]) for r in top],
        "days": days,
    }


def recent(limit: int = 100) -> list[dict]:
    """The latest events, newest first."""
    try:
        with _lock, _connect() as connection:
            rows = connection.execute(
                "SELECT ts, session_id, event, access_mode, ticker, company, "
                "       ok, duration_ms, cost_usd, detail "
                "FROM events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    except Exception as exc:
        log.warning("analytics read failed: %s", exc)
        return []
    return [dict(r) for r in rows]


def failures(limit: int = 50) -> list[dict]:
    """Only what went wrong — the most useful view there is."""
    try:
        with _lock, _connect() as connection:
            rows = connection.execute(
                "SELECT ts, session_id, event, ticker, company, detail "
                "FROM events WHERE ok = 0 ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    except Exception as exc:
        log.warning("analytics read failed: %s", exc)
        return []
    return [dict(r) for r in rows]
