"""
Shared HTTP layer: retries, polite rate limiting, and on-disk JSON caching.

Every outbound request in the project goes through here so that rate limits and
caching are enforced in one place rather than sprinkled across call sites.

The SEC in particular publishes a fair-access policy: requests must carry a
descriptive User-Agent with real contact details, and sustained traffic above
~10 requests/second gets the client IP blocked. `sec_get` enforces both.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

from . import config

log = logging.getLogger(__name__)

_HTTP_CACHE_DIR = config.CACHE_DIR / "http"
_HTTP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class RateLimiter:
    """Thread-safe minimum-interval limiter.

    Streamlit runs callbacks on worker threads, so the lock is not optional.
    """

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            wait = self._min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()


_sec_limiter = RateLimiter(config.SEC_MAX_REQUESTS_PER_SECOND)


# ---------------------------------------------------------------------------
# Core request helper
# ---------------------------------------------------------------------------

def request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    timeout: int | None = None,
    retries: int = 3,
    backoff: float = 1.5,
    stream: bool = False,
    allow_redirects: bool = True,
) -> requests.Response | None:
    """Perform an HTTP request with bounded exponential backoff.

    Returns ``None`` rather than raising when every attempt fails: document
    discovery is inherently best-effort over dozens of candidate URLs, and one
    dead link must not abort an ingestion run.
    """
    headers = {"User-Agent": config.HTTP_USER_AGENT, **(headers or {})}
    timeout = timeout or config.HTTP_TIMEOUT_SECONDS

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=timeout,
                stream=stream,
                allow_redirects=allow_redirects,
            )
            # 429/5xx are transient: back off and retry.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"HTTP {response.status_code}")
                time.sleep(backoff ** attempt)
                continue
            return response
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(backoff ** attempt)

    log.warning("request failed after %d attempts: %s (%s)", retries, url, last_error)
    return None


# ---------------------------------------------------------------------------
# SEC-specific access
# ---------------------------------------------------------------------------

def sec_get(url: str, *, stream: bool = False) -> requests.Response | None:
    """Rate-limited SEC request carrying the mandated User-Agent."""
    _sec_limiter.acquire()
    return request(
        url,
        headers={
            "User-Agent": config.SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        },
        stream=stream,
    )


# ---------------------------------------------------------------------------
# Cached JSON
# ---------------------------------------------------------------------------

def _cache_path(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return _HTTP_CACHE_DIR / f"{digest}.json"


def get_json_cached(
    url: str,
    *,
    sec: bool = False,
    ttl_seconds: int = 86_400,
) -> Any | None:
    """GET a JSON document, served from disk cache when fresh.

    Reference data such as the SEC ticker->CIK map changes at most daily, so a
    24 hour TTL keeps repeated company loads instant and keeps us well inside
    the SEC's fair-access limits.
    """
    path = _cache_path(url)

    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry; fall through and refetch

    response = sec_get(url) if sec else request(url)
    if response is None or response.status_code != 200:
        # Stale cache beats no data at all when the upstream is down.
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    try:
        payload = response.json()
    except ValueError:
        log.warning("response was not valid JSON: %s", url)
        return None

    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # caching is an optimisation, never a hard requirement

    return payload
