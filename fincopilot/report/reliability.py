"""
Post-generation reliability scorecard.

Turns the signals the pipeline already produces into a single, honest read on how far
a reader can trust a given report. The headline is a numeric-traceability check — the
defensible form of a "hallucination %": every figure that appears in the NARRATIVE
prose is matched against (a) the deterministic model's own numbers and (b) the text of
the cited sources. A figure matching neither is "unverified" — not proof of fabrication,
but a number we could not automatically trace, which is exactly what a reader should be
wary of.

The scorecard is deterministic (no LLM), computed once after QA, and stored on the
report so the website can render it after generation.
"""

from __future__ import annotations

import re
from collections import Counter

_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
# One non-overlapping pass: alt 1 = a currency-prefixed figure ("$216 billion", "$78B",
# "$216.85"); alt 2 = a bare magnitude figure ("216 billion") not preceded by a currency
# symbol or digit. finditer never overlaps, so "$216 billion" is matched once, by alt 1.
_MONEY = re.compile(
    r"(?:\$|us\$|usd|₹|€|£)\s?(\d[\d,]*(?:\.\d+)?)\s*"
    r"(trillion|tn|billion|bn|million|mn|thousand|[tbmk])?\b"
    r"|(?<![$\d.])(\d[\d,]*(?:\.\d+)?)\s+(trillion|billion|million|thousand)\b",
    re.I)
_MULT = {"trillion": 1e12, "tn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
         "million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


def _money_value(number: str, magnitude: str | None) -> float:
    value = float(number.replace(",", ""))
    if magnitude:
        value *= _MULT.get(magnitude.lower(), 1.0)
    return value


def _extract(text: str) -> tuple[list[float], list[float]]:
    """Percentages and currency/magnitude values found in ``text``."""
    pcts = [float(m.group(1)) for m in _PCT.finditer(text)]
    money: list[float] = []
    for match in _MONEY.finditer(text):
        if match.group(1) is not None:                 # currency-prefixed
            money.append(_money_value(match.group(1), match.group(2)))
        else:                                          # bare magnitude
            money.append(_money_value(match.group(3), match.group(4)))
    return pcts, money


def _context(text: str, start: int, end: int, width: int = 70) -> str:
    """A short window of ``text`` around a match, for the drill-down list."""
    left, right = max(0, start - width), min(len(text), end + width)
    snippet = text[left:right].strip()
    return ("…" if left > 0 else "") + snippet + ("…" if right < len(text) else "")


def _located_figures(text: str, location: str) -> list[dict]:
    """Every figure in ``text`` with its raw form, value, kind, location and context."""
    figures: list[dict] = []
    for match in _PCT.finditer(text):
        figures.append({"figure": match.group(0).strip(), "value": float(match.group(1)),
                        "kind": "pct", "location": location,
                        "context": _context(text, match.start(), match.end())})
    for match in _MONEY.finditer(text):
        if match.group(1) is not None:
            value = _money_value(match.group(1), match.group(2))
        else:
            value = _money_value(match.group(3), match.group(4))
        figures.append({"figure": match.group(0).strip(), "value": value, "kind": "money",
                        "location": location,
                        "context": _context(text, match.start(), match.end())})
    return figures


def _prose_text(report) -> str:
    parts: list[str] = []
    for section in getattr(report, "sections", None) or []:
        parts += [section.summary, *section.paragraphs, *section.bullets, section.implication]
    thesis = getattr(report, "thesis", None) or {}
    if isinstance(thesis, dict):
        parts += [str(v) for v in thesis.values() if isinstance(v, str)]
    for risk in (getattr(report, "risks", None) or []):
        parts += [risk.get("financial_impact", ""), risk.get("valuation_impact", "")]
    return " ".join(p for p in parts if p)


def _source_text(report) -> str:
    parts: list[str] = []
    for section in getattr(report, "sections", None) or []:
        for ev in section.evidence:
            if getattr(ev, "snippet", ""):
                parts.append(ev.snippet)
    return " ".join(parts)


# Field names that carry a RATIO (stored 0-1, shown as %) vs an absolute money value.
# Used to walk the deterministic tables so every model-produced number is registered.
_PCT_KEYS = {"revenue_growth", "operating_margin", "gross_margin", "net_margin",
             "implied_cagr", "year_one_growth", "bottom_up_cagr", "top_down_cagr",
             "cagr", "probability", "fair_value_change", "revenue_change"}
_MONEY_KEYS = {"revenue", "operating_income", "net_income", "free_cash_flow",
               "present_value", "terminal_revenue", "latest_revenue", "fair_value",
               "fair_value_per_share", "value_per_share", "market_cap",
               "implied_value_per_share", "latest_segment_sum", "latest_total_revenue",
               "bottom_up_terminal_revenue", "base_fair_value", "base_terminal_revenue"}


def _walk(obj, pcts: set, money: list) -> None:
    """Collect numbers from a nested table structure, classified by field name."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                _walk(value, pcts, money)
            elif isinstance(value, bool):
                continue
            elif isinstance(value, (int, float)):
                if key in _PCT_KEYS:
                    pcts.add(round(value * 100, 1))
                elif key in _MONEY_KEYS:
                    money.append(float(value))
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, pcts, money)


def _model_numbers(report) -> tuple[set[float], list[float]]:
    pcts: set[float] = set()
    money: list[float] = []

    for value in (report.fair_value, report.share_price, report.consensus_target):
        if value:
            money.append(value)
    if report.market_implied_growth is not None:
        pcts.add(round(report.market_implied_growth * 100, 1))

    # canonical metrics and assumptions carry an explicit unit
    for metric in (report.canonical_metrics or []):
        value, unit = metric.get("value"), metric.get("unit")
        if value is None:
            continue
        if unit == "%":
            pcts.add(round(value * 100, 1))
        elif unit in ("currency", "shares"):
            money.append(value)
    for assumption in (report.assumptions or []):
        value = assumption.get("value")
        if value is not None and assumption.get("unit") == "%":
            pcts.add(round(value * 100, 1))

    # reverse-DCF rows are all percentages
    for row in (report.priced_in or {}).get("rows", []):
        if row.get("unit") == "%":
            for key in ("base_value", "implied_value"):
                if row.get(key) is not None:
                    pcts.add(round(row[key] * 100, 1))

    # every number the deterministic tables produced (financial, forecast, segment,
    # comps, competition sensitivity, scenarios, blended) is a legitimate model number
    for structure in (report.financial_table, report.forecast_table,
                      report.segment_forecast, report.comps,
                      report.competition_sensitivity, report.scenarios, report.blended):
        _walk(structure, pcts, money)

    return pcts, money


def _pct_traced(value: float, refs) -> bool:
    return any(abs(value - r) <= 0.6 for r in refs)


def _money_traced(value: float, refs) -> bool:
    return any(r > 0 and abs(value / r - 1) <= 0.02 for r in refs)


def compute_reliability(report) -> dict:
    """A deterministic reliability scorecard for a finished report."""
    model_pcts, model_money = _model_numbers(report)
    source_pcts, source_money = _extract(_source_text(report))
    grounded_pcts = list(model_pcts) + source_pcts
    grounded_money = list(model_money) + source_money

    # Gather every narrative figure WITH its location, so the scorecard can list the
    # specific unverified ones for a reviewer to jump to.
    located: list[dict] = []
    for section in (report.sections or []):
        text = " ".join(p for p in [section.summary, *section.paragraphs, section.implication] if p)
        located += _located_figures(text, section.title or section.key)
    thesis = getattr(report, "thesis", None) or {}
    if isinstance(thesis, dict):
        thesis_text = " ".join(str(v) for v in thesis.values() if isinstance(v, str))
        located += _located_figures(thesis_text, "Investment thesis")
    for risk in (getattr(report, "risks", None) or []):
        risk_text = " ".join([risk.get("financial_impact", ""), risk.get("valuation_impact", "")])
        if risk_text.strip():
            located += _located_figures(risk_text, f"Risk: {risk.get('risk', '')}".strip(": "))

    untraced_occurrences = 0
    unverified: list[dict] = []
    seen: set = set()
    for fig in located:
        traced = (_pct_traced(fig["value"], grounded_pcts) if fig["kind"] == "pct"
                  else _money_traced(fig["value"], grounded_money))
        if traced:
            continue
        untraced_occurrences += 1
        key = (fig["figure"], fig["context"][:40])       # dedupe the DISPLAY list only
        if key not in seen:
            seen.add(key)
            unverified.append({"figure": fig["figure"], "location": fig["location"],
                               "context": fig["context"]})

    total = len(located)
    unverified_pct = round(untraced_occurrences / total * 100, 1) if total else 0.0

    # citation coverage: fraction of narrative paragraphs carrying an inline [n] marker
    cited = para_total = 0
    for section in (report.sections or []):
        for paragraph in section.paragraphs:
            para_total += 1
            if re.search(r"\[\d+\]", paragraph):
                cited += 1
    citation_coverage = round(cited / para_total * 100) if para_total else 100

    counts = Counter(f["severity"] for f in (report.qa_findings or []))

    sources = report.sources or []
    fresh = [s for s in sources if s.get("freshness") in ("Current", "Recent")]
    freshness = round(len(fresh) / len(sources) * 100) if sources else 100

    # Composite score (0-100). Untraceable figures and missing citations are the main
    # penalties; QA findings and low valuation confidence pull it down further.
    score = 100.0
    score -= unverified_pct * 0.5
    score -= (100 - citation_coverage) * 0.3
    score -= (counts.get("CRITICAL", 0) * 25 + counts.get("HIGH", 0) * 15
              + counts.get("MEDIUM", 0) * 4 + counts.get("LOW", 0) * 1)
    score -= {"Low": 8, "Medium": 3}.get(report.valuation_confidence, 0)
    score = max(0, min(100, round(score)))
    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"
    label = {"A": "High reliability", "B": "Good reliability",
             "C": "Moderate reliability", "D": "Low reliability"}[grade]

    return {
        "score": score,
        "grade": grade,
        "label": label,
        "unverified_figures_pct": unverified_pct,
        "traceable_figures_pct": round(100 - unverified_pct, 1),
        "figures_checked": total,
        "figures_traced": total - untraced_occurrences,
        "citation_coverage_pct": citation_coverage,
        "qa_status": report.qa_status,
        "qa_findings": dict(counts),
        "source_freshness_pct": freshness,
        "valuation_confidence": report.valuation_confidence or "n/a",
        # The specific unverified figures (deduped), so a reviewer can jump to each.
        "unverified": unverified[:25],
        "unverified_count": untraced_occurrences,
    }
