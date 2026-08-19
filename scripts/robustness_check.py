"""
Valuation robustness harness.

Runs the valuation engine across a deliberately diverse set of companies — hyper
and mature growth, US and non-US, cheap and expensive, healthy and troubled — and
prints one line each plus a set of invariant checks. The point is that the engine
was validated on two similar mega-caps (NVDA, AAPL) and that hid systematic
problems; every change to the valuation core should be run through this so a fix
for one profile is not a regression for another.

This is NOT a pytest test: it makes live network + model calls (minutes to run,
non-deterministic on the first pass before caches warm). Run it by hand:

    venv/Scripts/python.exe scripts/robustness_check.py

The invariant checks are the hard floor — a fair value must be positive or the
name must be NOT RATED (never a −150% "SELL"), WACC must be sane, and the rating
must agree with the sign of the upside. The fair-value *levels* are judgement,
printed for a human to eyeball, not asserted.
"""

from __future__ import annotations

import math
import pathlib
import sys

# Make the package importable however the script is launched.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# A spread of profiles. Tickers, not names, so resolution is unambiguous.
UNIVERSE = [
    ("NVDA", "hyper-growth, expensive"),
    ("MSFT", "durable compounder, premium"),
    ("AAPL", "mature mega-cap, premium"),
    ("KO", "low-growth staple"),
    ("PFE", "declining pharma, cheap"),
    ("INTC", "troubled, thin margins"),
    ("F", "cyclical, low margin"),
    ("GOOGL", "mega-cap, moderate growth"),
]


def _row(tk: str, note: str) -> dict:
    from fincopilot.fundamentals import load_financials
    from fincopilot.resolve import resolve_company
    from fincopilot.valuation import value_company

    company = resolve_company(tk)
    history = load_financials(company)
    if not (history and history.is_sufficient_for_dcf):
        return {"ticker": tk, "note": note, "ok": True, "skip": "insufficient history"}

    v = value_company(company, history, use_model=True)
    if not v.dcf:
        return {"ticker": tk, "note": note, "ok": True, "skip": "no DCF"}

    forecast = v.dcf.forecast
    cagr = math.prod(1 + f.revenue_growth for f in forecast) ** (1 / len(forecast)) - 1
    checks = _invariants(v)
    return {
        "ticker": tk, "note": note, "currency": history.currency,
        "wacc": v.dcf.wacc, "y1": v.assumptions.value("year_one_growth"),
        "cagr": cagr, "margin": v.assumptions.value("terminal_margin"),
        "fair_value": v.fair_value, "upside": v.upside, "rating": v.rating,
        "revised": any("calibration review revised" in w for w in v.warnings),
        "outlier": any("we are the outlier" in w for w in v.warnings),
        "degenerate": any("non-positive equity value" in w for w in v.warnings),
        "ok": not checks, "failures": checks,
    }


def _invariants(v) -> list[str]:
    """Hard checks. A violation is a bug, not a judgement call."""
    failures: list[str] = []
    fv = v.dcf.fair_value_per_share

    # A negative equity value must never surface as a rating.
    if fv <= 0 and v.rating != "NOT RATED":
        failures.append(f"non-positive fair value {fv:.2f} but rating is {v.rating}")
    # WACC must be inside sane rails.
    if not (0.04 <= v.dcf.wacc <= 0.20):
        failures.append(f"WACC {v.dcf.wacc:.1%} outside [4%, 20%]")
    # Rating must agree with the sign of upside when there is one.
    if v.upside is not None:
        if v.rating == "BUY" and v.upside <= 0:
            failures.append("BUY with non-positive upside")
        if v.rating == "SELL" and v.upside >= 0.15:
            failures.append("SELL with large positive upside")
    # Terminal margin cannot exceed 100% of revenue.
    tm = v.assumptions.value("terminal_margin")
    if tm is not None and tm > 1.0:
        failures.append(f"terminal margin {tm:.0%} exceeds 100%")
    return failures


def main() -> int:
    print(f"{'ticker':7} {'cur':4} {'WACC':>5} {'y1':>5} {'CAGR':>5} {'margin':>6} "
          f"{'fair':>10} {'upside':>7} {'rating':10} flags")
    print("-" * 96)
    any_fail = False
    for tk, note in UNIVERSE:
        try:
            r = _row(tk, note)
        except Exception as exc:  # a crash is itself a robustness failure
            any_fail = True
            print(f"{tk:7} ERROR {type(exc).__name__}: {exc}")
            continue
        if r.get("skip"):
            print(f"{tk:7} skipped — {r['skip']}")
            continue
        flags = " ".join(f for f, on in (
            ("revised", r["revised"]), ("outlier", r["outlier"]), ("degenerate", r["degenerate"]),
        ) if on)
        upside = f"{r['upside'] * 100:+.0f}%" if r["upside"] is not None else "n/a"
        fv = f"{r['fair_value']:,.2f}" if r["fair_value"] is not None else "n/a"
        print(f"{tk:7} {r['currency']:4} {r['wacc'] * 100:4.1f}% {r['y1'] * 100:4.0f}% "
              f"{r['cagr'] * 100:4.0f}% {r['margin'] * 100:5.0f}% {fv:>10} {upside:>7} "
              f"{r['rating']:10} {flags}")
        if not r["ok"]:
            any_fail = True
            for f in r["failures"]:
                print(f"        !! INVARIANT: {f}")

    print("-" * 96)
    print("RESULT:", "FAIL — invariant violated" if any_fail else "pass — all invariants held")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
