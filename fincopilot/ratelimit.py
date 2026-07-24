"""
Rate limiting for the public demo.

A demo link on a CV is open to the internet, and every expensive action -
loading a company, asking a question, generating a report - spends OpenAI
credits. Without a limit, one looping visitor (or one crawler) can run up a real
bill on the owner's card. This module bounds that exposure.

Two layers:

* **Per session** - a single visitor gets a modest hourly allowance of each
  action, tracked in Streamlit session state.
* **Global** - a process-wide daily ceiling across all visitors, so even a flood
  of distinct sessions cannot exceed a known daily spend.

Deliberately in-process: the demo runs as one Streamlit process on one instance,
so a module-level counter is sufficient and needs no external store. It resets if
the process restarts, which is acceptable - a restart is not an attack vector.

These limits are a safety net, not a paywall. They sit well above what any
genuine reviewer clicking through the app will reach.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

# Whether limiting is active at all. Off by default so local development and the
# test suite are never throttled; the deployment sets DEMO_MODE=1.
from . import config


def _enabled() -> bool:
    return (config.get_secret("DEMO_MODE", "0") or "0") not in ("0", "", "false", "False")


# action -> (per-session per hour, global per day)
_LIMITS: dict[str, tuple[int, int]] = {
    "load": (6, 120),        # company loads: the most expensive action
    "chat": (30, 800),       # questions
    "report": (4, 80),       # full report generation
}

_HOUR = 3600
_DAY = 86_400


@dataclass
class _GlobalCounter:
    day_start: float = field(default_factory=time.time)
    counts: dict[str, int] = field(default_factory=dict)


_global = _GlobalCounter()
_lock = threading.Lock()


class RateLimitExceeded(Exception):
    """Raised when an action is refused. The message is shown to the user."""


def _check_global(action: str) -> None:
    limit = _LIMITS[action][1]
    with _lock:
        if time.time() - _global.day_start > _DAY:
            _global.day_start = time.time()
            _global.counts.clear()
        used = _global.counts.get(action, 0)
        if used >= limit:
            raise RateLimitExceeded(
                "The shared daily demo limit for this action has been reached, so "
                "further requests are paused to protect the running costs of this "
                "public demo. It resets within 24 hours. To run without limits, "
                "clone the repo and add your own OpenAI key."
            )
        _global.counts[action] = used + 1


def _check_session(action: str) -> None:
    import streamlit as st

    limit = _LIMITS[action][0]
    key = f"_rl_{action}"
    now = time.time()

    events = [t for t in st.session_state.get(key, []) if now - t < _HOUR]
    if len(events) >= limit:
        wait_min = int((_HOUR - (now - events[0])) / 60) + 1
        raise RateLimitExceeded(
            f"You have reached this demo's hourly limit for this action "
            f"({limit}/hour). Please try again in about {wait_min} minutes. "
            f"This cap only exists to keep the public demo's costs bounded."
        )
    events.append(now)
    st.session_state[key] = events


def enforce(action: str) -> None:
    """Raise :class:`RateLimitExceeded` if ``action`` is over budget.

    A no-op unless DEMO_MODE is set, so nothing is throttled locally.
    """
    if action not in _LIMITS or not _enabled():
        return
    _check_session(action)   # session first: a per-user cap should not consume
    _check_global(action)    # global budget before it even applies
