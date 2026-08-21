"""
Analyst assumption overrides — the institutional pattern where a human owns the
value drivers and the model computes from them.

An analyst pins any of the key drivers in a small JSON file, ``overrides/{slug}.json``:

    {"terminal_operating_margin": 0.45, "year_one_revenue_growth": 0.30, "wacc": 0.12}

These win over both the language-model proposal and the agentic critique, are bounded
only by hard sanity rails (not the model's anti-hallucination bounds), and are recorded
as analyst-set so the report shows the number is human, not model. The QA gate still
enforces downstream consistency (no double-count, scenarios reconcile, probabilities
sum), so the analyst cannot ship an internally inconsistent valuation.
"""

from __future__ import annotations

import json
import logging

from .. import config

log = logging.getLogger(__name__)

# The drivers an analyst may pin. Values are decimals (0.30 = 30%).
OVERRIDE_KEYS = {
    "year_one_revenue_growth",
    "terminal_operating_margin",
    "terminal_growth_rate",
    "wacc",
}


def load_overrides(slug: str) -> dict:
    """Return the analyst overrides for ``slug`` from ``overrides/{slug}.json``.

    Only recognised, numeric drivers are returned; anything else is ignored. A
    missing or malformed file yields no overrides (a pure model run).
    """
    path = config.OVERRIDES_DIR / f"{slug}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read overrides for %s: %s", slug, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    clean = {
        key: float(value)
        for key, value in data.items()
        if key in OVERRIDE_KEYS and isinstance(value, (int, float))
    }
    if clean:
        log.info("loaded analyst overrides for %s: %s", slug, clean)
    return clean
