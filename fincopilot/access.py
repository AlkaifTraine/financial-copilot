"""
Who may use this, and whose credits they spend.

A public URL in front of a paid API is an open tab on someone's card. Rate
limits bound the damage but do not decide *who* gets in, and the two questions
have different answers here:

``own_key``      the visitor pastes their own OpenAI key. They pay for their own
                 usage, so metering them against the owner's budget would be
                 wrong — their limits are their own account's.
``access_code``  the visitor has a shared secret the owner handed out. They are
                 spending the OWNER'S credits, so every rate limit and the spend
                 ceiling apply to them in full.

The distinction matters more than it looks: collapsing the two either bills the
owner for guests who brought their own key, or lets code-holders run unmetered
on the owner's account.

**The code has no default and is never committed.** The repository is public.
A literal code in source is readable by anyone who opens GitHub, and it
authorises spending real money on the owner's account — so it is read from the
environment and, when unset, the access-code route simply does not exist and
the app is bring-your-own-key only. Failing closed on a missing secret is the
only safe direction.

Keys the visitor provides live in Streamlit session state for the duration of
their session and nowhere else: never written to disk, never entered in the
analytics database, never included in a log line. :mod:`fincopilot.guardrails`
independently strips key-shaped strings from anything outbound, so a key pasted
into the wrong box does not travel either.
"""

from __future__ import annotations

import hmac
import logging
import re
import secrets
from dataclasses import dataclass

from . import config

log = logging.getLogger(__name__)

MODE_OWN_KEY = "own_key"
MODE_ACCESS_CODE = "access_code"

# Shape check only — whether the key actually works is decided by the provider
# on the first call. Rejecting a well-formed key here because it looks unusual
# would be worse than letting the API return 401.
_KEY_SHAPE = re.compile(r"^sk-[A-Za-z0-9_\-]{20,}$")


@dataclass
class Grant:
    """An authenticated session's right to use the service."""

    mode: str
    api_key: str | None = None      # session-scoped; the owner's key is never held here

    @property
    def uses_owner_credits(self) -> bool:
        """Whether this session spends the owner's money.

        Everything that costs the owner — rate limits, the spend ceiling — keys
        off this, not off whether the visitor is 'logged in'.
        """
        return self.mode == MODE_ACCESS_CODE


def access_code_configured() -> bool:
    """Whether the owner has set a code on this deployment."""
    return bool(config.ACCESS_CODE)


def verify_access_code(submitted: str) -> bool:
    """Check a submitted code against the configured one.

    Compared with :func:`hmac.compare_digest` so the check takes the same time
    whatever the input. A short code is guessable by brute force regardless, so
    the real protection is the rate limits behind it and the owner's ability to
    rotate the code by changing one environment variable and restarting.
    """
    if not config.ACCESS_CODE:
        return False
    return hmac.compare_digest(
        (submitted or "").strip().encode("utf-8"),
        config.ACCESS_CODE.encode("utf-8"),
    )


def looks_like_openai_key(candidate: str) -> bool:
    return bool(_KEY_SHAPE.match((candidate or "").strip()))


def new_session_id() -> str:
    """An opaque id for correlating one visitor's events.

    Random rather than derived from anything about the visitor: the analytics
    only ever need to group a session's own events together, never to identify
    or re-identify a person.
    """
    return secrets.token_hex(8)


def grant_from_code(submitted: str) -> Grant | None:
    if verify_access_code(submitted):
        log.info("access granted via access code")
        return Grant(mode=MODE_ACCESS_CODE)
    return None


def grant_from_key(api_key: str) -> Grant | None:
    key = (api_key or "").strip()
    if looks_like_openai_key(key):
        log.info("access granted via a visitor-supplied key")
        return Grant(mode=MODE_OWN_KEY, api_key=key)
    return None
