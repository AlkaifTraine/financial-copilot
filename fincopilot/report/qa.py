"""
Report QA: citation grounding and internal consistency.

Everything upstream is generated section by section, table by table. That makes
each piece defensible on its own but leaves one gap nothing else closes: does the
finished report hang together? Two deterministic passes check it — no model, so
the check itself cannot hallucinate a problem or miss one inconsistently.

Citation QA (#23): a narrative section that makes claims must ground them. A
paragraph-heavy section with no inline [n] markers, or a marker pointing past the
evidence list, is flagged — an ungrounded claim is exactly what an equity report
cannot afford.

Consistency QA (#22): the numbers in different exhibits must agree. Scenario fair
values must order bear <= base <= bull; probabilities must sum to one; a blended
value must lie inside its own reconciled range. These are invariants that a
healthy report satisfies silently and only a real bug violates.

Anything found is surfaced in the report's own "Model Notes and Limitations" —
the honest place for it — rather than hidden. A clean report adds nothing.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_MARKER = re.compile(r"\[(\d+)\]")


def audit_citations(report) -> list[str]:
    """Flag narrative sections whose claims are not grounded in the evidence."""
    issues: list[str] = []
    for section in report.sections:
        text = " ".join(
            [section.summary, *section.paragraphs, *section.bullets, section.implication]
        )
        markers = [int(n) for n in _MARKER.findall(text)]
        evidence_count = len(section.evidence)

        # A marker pointing past the evidence list is a dead reference.
        dangling = sorted({n for n in markers if n > evidence_count or n < 1})
        if dangling:
            issues.append(
                f'The "{section.title}" section cites source '
                f'{"s " if len(dangling) > 1 else ""}'
                f'{", ".join(f"[{n}]" for n in dangling)} but only {evidence_count} '
                f"were provided; the reference resolves to nothing."
            )

        # A section that asserts several paragraphs with no citation at all is
        # ungrounded — the one case where silence is a defect, not brevity.
        if len(section.paragraphs) >= 2 and not markers and evidence_count:
            issues.append(
                f'The "{section.title}" section makes claims without any inline '
                f"citation, though {evidence_count} sources were available."
            )
    return issues


def check_consistency(report) -> list[str]:
    """Cross-check numbers across exhibits for contradictions."""
    issues: list[str] = []

    scenarios = report.scenarios
    if scenarios and scenarios.get("cases"):
        values = {c["key"]: c.get("fair_value_per_share") for c in scenarios["cases"]}
        bear, base, bull = values.get("bear"), values.get("base"), values.get("bull")
        if None not in (bear, base, bull) and not (bear <= base <= bull):
            issues.append(
                f"Scenario fair values are not ordered bear ≤ base ≤ bull "
                f"({bear:,.2f}, {base:,.2f}, {bull:,.2f}); the scenario set is inconsistent."
            )
        prob_sum = sum(c.get("probability", 0.0) for c in scenarios["cases"])
        if abs(prob_sum - 1.0) > 0.01:
            issues.append(f"Scenario probabilities sum to {prob_sum:.2f}, not 1.0.")

    blended = report.blended
    if blended and blended.get("blended_value"):
        value = blended["blended_value"]
        low, high = blended.get("low"), blended.get("high")
        if low is not None and high is not None and not (low - 1e-6 <= value <= high + 1e-6):
            issues.append(
                f"Blended value {value:,.2f} lies outside its own reconciled range "
                f"[{low:,.2f}, {high:,.2f}]."
            )

    # The lead recommendation is derived from upside; verify the sign still agrees
    # with the printed rating, so a refactor upstream cannot silently divorce them.
    if report.upside is not None:
        if report.rating == "BUY" and report.upside <= 0:
            issues.append(
                f"Rating is BUY but the fair value is at or below the price "
                f"({report.upside * 100:+.0f}%)."
            )
        elif report.rating == "SELL" and report.upside >= 0:
            issues.append(
                f"Rating is SELL but the fair value is at or above the price "
                f"({report.upside * 100:+.0f}%)."
            )

    return issues


_UNIT = {"tn": 1e12, "trillion": 1e12, "bn": 1e9, "billion": 1e9, "b": 1e9,
         "m": 1e6, "million": 1e6}
# A scaled money figure, as a reusable sub-pattern (value, unit).
_MONEY_SUB = r"(?:\$|usd|us\$)?\s*([0-9]+(?:\.[0-9]+)?)\s*(tn|trillion|bn|billion|b|million|m)\b"
# Connective words that bind a figure to a metric as its VALUE, so we only test a
# number the prose actually presents AS the metric — not an incidental figure
# (a buyback, a dividend) that merely sits near the metric's name.
_BIND = r"(?:of|was|were|at|reached|totall?ed|hit|stood at|came in at|:|=)"
_FILLER = r"(?:\s+(?:a|an|the|about|roughly|approximately|record|around|nearly|~))*"
# A figure carrying any of these is a DIFFERENT period or a projection, not a
# restatement of the canonical full-year actual — so it is not a contradiction.
# (The quarterly-vs-annual discipline is enforced separately, in the prompts.)
_OTHER_PERIOD = re.compile(
    r"q[1-4]\b|quarter|three months|sequential|per quarter|"          # quarterly
    r"expect|project|forecast|guidanc|estimat|target|assum|imply|implie|"  # forward / modelled
    r"next (?:year|quarter|fiscal)|by 20\d\d|fy\s?20(?:2[7-9]|[3-9]\d)|"   # future periods
    r"could|would|we see|potential|forward|over the (?:forecast|next)",
    re.I,
)

# Metrics most prone to a quarterly/annual or GAAP/non-GAAP mix-up in prose.
_SCANNED = {
    "free_cash_flow": ("free cash flow", "fcf"),
    "net_income": ("net income",),
    "revenue": ("full-year revenue", "full year revenue", "annual revenue", "total revenue"),
}


def _bound_figures(text: str, keyword: str) -> list[tuple[float, str]]:
    """(value, context) for figures the prose presents AS this metric's value.

    Only two grammatical shapes count — "<metric> of/was/reached $X" and
    "$X in/of <metric>" — so an incidental number near the metric's name (a
    dividend beside "net income", a segment figure beside "revenue") is not
    mistaken for a restatement of the metric.
    """
    kw = re.escape(keyword)
    patterns = [
        rf"{kw}\s+{_BIND}{_FILLER}\s*{_MONEY_SUB}",     # metric of/was $X
        rf"{_MONEY_SUB}\s+(?:in|of)\s+(?:{kw})",         # $X in/of metric
    ]
    found: list[tuple[float, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            value = float(match.group(1)) * _UNIT[match.group(2).lower()]
            # Widen the context so a "we expect"/"in Q1" qualifier earlier in the
            # sentence is visible to the different-period check.
            context = text[max(0, match.start() - 80): match.end() + 10]
            found.append((value, context))
    return found


def check_metric_consistency(report) -> list[str]:
    """Flag section prose that states a headline figure conflicting with its canonical value.

    A safety net behind the canonical-figures injection: if a section still states,
    say, a free cash flow materially different from the authoritative full-year
    value — and does not label it as a quarter — that is the contradiction the
    report cannot ship with.
    """
    issues: list[str] = []
    canonical = {m["key"]: m for m in report.canonical_metrics if m.get("unit") == "currency"}
    currency = report.currency

    for key, keywords in _SCANNED.items():
        metric = canonical.get(key)
        if not metric or not metric.get("value"):
            continue
        value = metric["value"]
        for section in report.sections:
            text = " ".join(
                [section.summary, *section.paragraphs, *section.bullets, section.implication]
            )
            for keyword in keywords:
                for figure, context in _bound_figures(text, keyword):
                    if _OTHER_PERIOD.search(context):
                        continue                      # a quarterly or forward figure is a different metric
                    if value > 0 and abs(figure - value) / value > 0.40:
                        issues.append(
                            f'The "{section.title}" section states {metric["label"].lower()} of '
                            f'~{currency} {figure / 1e9:,.1f}bn, but the canonical '
                            f'{metric["label"]} ({metric["period"]}) is {currency} '
                            f'{value / 1e9:,.1f}bn ({metric["definition"]}).'
                        )
                        break                         # one flag per section/metric is enough
    return issues


_DECREASE = re.compile(
    r"reduc|lower|cut|declin|fall|drop|compress|shrink|erod|weaken|pressur|slow|"
    r"deteriorat|hurt|dent|impair|soften|contract",
    re.I,
)
_INCREASE = re.compile(
    r"rais|increas|lift|boost|expand|improv|accelerat|strengthen|widen", re.I,
)
_FROM_TO = re.compile(
    r"from\s+(?:our\s+|the\s+)?(\d+(?:\.\d+)?)\s*%"      # from A%
    r"[^.]{0,50}?\b(?:to|toward|towards)\s+"              # to / toward
    r"(?:the\s+|a\s+|market[- ]implied\s+)?(?:[^.\d]{0,25}?)(\d+(?:\.\d+)?)\s*%",  # B%
    re.I,
)
# Deliberately narrow: only an explicit "raise/increase OUR/THE fair value" counts.
# Generic phrases a risk section uses constantly — "higher valuation", "richer
# valuation multiple" — must NOT trip this, or clean reports get false-flagged.
_LIFT_FV = re.compile(
    r"(?:rais|increas|lift|boost)\w*\s+(?:our|the)\s+(?:fair value|intrinsic value)",
    re.I,
)

_QUARTERLY_WORD = re.compile(
    r"\b(quarterly|qoq|quarter[- ]over[- ]quarter|sequential|per quarter)\b", re.I
)
_ANNUAL_WORD = re.compile(
    r"\b(annual|full[- ]year|fiscal year|fy\s?20\d\d|yoy|year[- ]over[- ]year|"
    r"per year|10[- ]year|cagr)\b",
    re.I,
)


def check_risk_direction(report) -> list[str]:
    """Flag risks whose stated effect points the wrong way.

    A risk is a downside: it must push the affected metric — and fair value — DOWN.
    The failure this catches is real: "a supply-chain event could reduce revenue
    CAGR from our 13.3% base toward market-implied 25.3%" — a decrease described as
    moving the metric ABOVE the base, which is directionally impossible.
    """
    issues: list[str] = []
    for risk in (report.risks or []):
        name = risk.get("risk", "this")
        text = f"{risk.get('financial_impact', '')} {risk.get('valuation_impact', '')}"

        for match in _FROM_TO.finditer(text):
            a, b = float(match.group(1)), float(match.group(2))
            window = text[max(0, match.start() - 60): match.end()]
            decreasing = _DECREASE.search(window)
            increasing = _INCREASE.search(window)
            if decreasing and b > a + 0.05:
                issues.append(
                    f'The "{name}" risk (a downside) describes {decreasing.group(0).lower()}ing '
                    f'a metric from {a:g}% toward {b:g}% — but {b:g}% is ABOVE {a:g}%, so the '
                    f'negative event is written as IMPROVING the metric. Directional contradiction.'
                )
            elif increasing and b < a - 0.05:
                issues.append(
                    f'The "{name}" risk describes {increasing.group(0).lower()}ing a metric from '
                    f'{a:g}% to {b:g}% — a rise stated as a fall. Directional contradiction.'
                )

        if _LIFT_FV.search(text):
            issues.append(
                f'The "{name}" risk describes RAISING fair value, but a risk is a downside — the '
                f'valuation impact points the wrong way.'
            )
    return issues


def check_metric_semantics(report) -> list[str]:
    """Flag a metric whose period/comparison label conflicts with its own reading.

    Catches "Quarterly revenue growth rate" carrying a full-year YoY figure — a
    quarterly label on an annual number is a different metric wearing the wrong name.
    """
    issues: list[str] = []
    forward = report.forward or {}
    for item in forward.get("watch_items", []):
        label = item.get("metric", "")
        if not _QUARTERLY_WORD.search(label):
            continue
        context = " ".join(
            [item.get("current", ""), item.get("expected", ""), item.get("assumption", "")]
        )
        annual = _ANNUAL_WORD.search(context)
        if annual:
            issues.append(
                f'The watch metric "{label}" is labelled quarterly, but its reading/assumption is '
                f'{annual.group(0).lower()} ("{context.strip()[:90]}") — a period/comparison '
                f'mismatch: a quarterly label on an annual metric.'
            )
    return issues


# ---------------------------------------------------------------------------
# Financial type-safety: every number must carry the right dimension. These are
# advisory heuristics over prose — they catch real category errors (a segment %
# read as an end-market %, a quarter annualised as a year, a past period cited as
# a forward trigger, our scenario probability pinned on external consensus) but,
# being regex over language, they inform rather than gate.
# ---------------------------------------------------------------------------

def _section_prose(report) -> str:
    parts: list[str] = []
    for section in getattr(report, "sections", None) or []:
        parts.extend([section.summary, *section.paragraphs, *section.bullets, section.implication])
    return " ".join(p for p in parts if p)


def _risk_prose(report) -> str:
    parts: list[str] = []
    for risk in (report.risks or []):
        parts.extend([risk.get("financial_impact", ""), risk.get("valuation_impact", ""),
                      risk.get("description", "")])
    return " ".join(p for p in parts if p)


def _assumption_prose(report) -> str:
    parts: list[str] = []
    for item in (getattr(report, "assumptions", None) or []):
        if isinstance(item, dict):
            parts.append(item.get("derivation", ""))
            parts.append(item.get("rationale", ""))
            prov = item.get("provenance")
            if isinstance(prov, dict):
                parts.extend(str(v) for v in prov.values())
    return " ".join(p for p in parts if p)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text)


_PCT_OF_REVENUE = re.compile(
    r"~?\s*(\d{1,3}(?:\.\d+)?)\s*%\s+of\s+"
    r"(?:its\s+|the\s+|total\s+|approximately\s+|about\s+)*(?:fy\s?20\d\d\s+)?(?:total\s+)?revenue",
    re.I,
)
_PROPER_PHRASE = re.compile(r"[A-Z][A-Za-z0-9.+/-]*(?:\s+(?:&|and|[A-Z][A-Za-z0-9.+/-]*)){0,3}")
_GENERIC_LEAD = {"the", "its", "total", "approximately", "about", "revenue", "and", "a", "an",
                 "of", "customer", "company", "management", "gaap"}


def check_segment_end_market(report) -> list[str]:
    """The same named line must not appear with two incompatible shares of revenue.

    Catches a reportable SEGMENT share (Compute & Networking ~90% of revenue) being
    conflated with an END-MARKET share (~60%): same name, two far-apart denominators.
    Works sentence by sentence — the named line is the proper-noun phrase nearest to
    the "N% of revenue" clause — so a "$193.5B" figure between them is not a barrier.
    """
    text = _section_prose(report) + " " + _risk_prose(report)
    seen: dict[str, list[float]] = {}
    for sentence in _sentences(text):
        pct = _PCT_OF_REVENUE.search(sentence)
        if not pct:
            continue
        name = None
        for candidate in reversed(_PROPER_PHRASE.findall(sentence[:pct.start()])):
            trimmed = re.sub(r"^(?:The|A|An|Its|Our|This|Their)\s+", "", candidate).strip()
            if trimmed and trimmed.lower() not in _GENERIC_LEAD and len(trimmed) >= 4:
                name = trimmed
                break
        if name is None:
            continue
        seen.setdefault(name.lower(), []).append(float(pct.group(1)))
    issues: list[str] = []
    for name, pcts in seen.items():
        if len(pcts) >= 2 and max(pcts) - min(pcts) > 15:
            issues.append(
                f'"{name.title()}" appears as both {min(pcts):g}% and {max(pcts):g}% of revenue — a '
                f'>15pp gap suggests a reportable-SEGMENT share and an END-MARKET share are being '
                f'conflated. State each denominator (segment vs end-market, and the period) explicitly.'
            )
    return issues


_ANNUALIZE = re.compile(r"annualiz", re.I)
_QUARTER_REF = re.compile(r"\bQ[1-4]\b|\bquarter", re.I)
_RUN_RATE = re.compile(r"run[- ]rate", re.I)


def check_quarterly_annualization(report) -> list[str]:
    """Flag a single quarter annualised and passed off as an annual figure."""
    text = _section_prose(report) + " " + _assumption_prose(report)
    issues: list[str] = []
    for sentence in _sentences(text):
        if _ANNUALIZE.search(sentence) and _QUARTER_REF.search(sentence) and not _RUN_RATE.search(sentence):
            issues.append(
                'A single quarter appears to be annualised without being labelled a run-rate: '
                f'"{sentence.strip()[:120]}". Compare a quarter to the year-ago QUARTER, or label a '
                f'4x figure a "run-rate" — never as annual guidance.'
            )
    return issues


def _latest_reported_fy(report) -> int | None:
    years = [
        int(m.group(1))
        for metric in (report.canonical_metrics or [])
        for m in [re.search(r"FY\s?(20\d\d)", str(metric.get("period", "")))]
        if m
    ]
    return max(years) if years else None


_FORWARD_TRIGGER = re.compile(
    r"remain|stay|declin|fall|drop|rise|below|above|by\s+fy|reach|exceed|hold|sustain", re.I
)


def check_forward_period_staleness(report) -> list[str]:
    """A forward-looking watch trigger must not be keyed to an already-reported year."""
    latest = _latest_reported_fy(report)
    if latest is None:
        return []
    issues: list[str] = []
    for item in (report.forward or {}).get("watch_items", []):
        for field_name in ("expected", "bull_bear"):
            value = item.get(field_name, "") or ""
            for match in re.finditer(r"FY\s?(20\d\d)", value):
                if int(match.group(1)) <= latest and _FORWARD_TRIGGER.search(value):
                    issues.append(
                        f'A watch trigger is keyed to FY{match.group(1)} ("{value.strip()[:80]}"), but '
                        f'FY{latest} is the latest reported year, so FY{match.group(1)} is historical. A '
                        f'forward trigger must reference a FUTURE period or be labelled a past observation.'
                    )
                    break
    return issues


_CONSENSUS = re.compile(r"consensus|analyst(?:s'?)?\s+(?:target|price\s+target|estimate)", re.I)
_OUR_SCENARIO = re.compile(r"\b(?:bull|bear|base)\b.{0,20}(?:case|probability|scenario)|probability", re.I)
_ATTRIBUTION = re.compile(r"assume|impl(?:y|ies)|reflect|bake[sd]?|price[sd]?\s+in", re.I)


def check_consensus_scenario(report) -> list[str]:
    """Flag our internal scenario probability being attributed to external consensus."""
    issues: list[str] = []
    for sentence in _sentences(_section_prose(report)):
        if (_CONSENSUS.search(sentence) and _ATTRIBUTION.search(sentence)
                and _OUR_SCENARIO.search(sentence) and re.search(r"\d{1,3}\s*%", sentence)):
            issues.append(
                'External consensus is being tied to our internal scenario probability: '
                f'"{sentence.strip()[:130]}". The consensus/target is what the market believes; the '
                f'bull/bear probabilities are ours — keep the two distinct.'
            )
    return issues


def _thesis_prose(report) -> str:
    thesis = getattr(report, "thesis", None) or {}
    if not isinstance(thesis, dict):
        return ""
    return " ".join(str(v) for v in thesis.values() if isinstance(v, str))


_MARKET_IMPLIED = re.compile(
    r"market\s+(?:implies|prices?\s+in|expects|assumes|is\s+pricing\s+in|bakes?\s+in|needs?)"
    r"[^.]{0,70}?(\d{1,3}(?:\.\d+)?)\s*%[^.]{0,45}?(margin|growth|cagr|revenue)",
    re.I,
)


def check_market_implied_claims(report) -> list[str]:
    """Every 'the market implies X%' claim must trace to an actual reverse-DCF value (#2).

    The reverse DCF is the only authority on what the price is pricing in. A prose
    figure ("the market assumes >60% margins") that does not match a computed implied
    value is invented and must be flagged.
    """
    rows = (getattr(report, "priced_in", None) or {}).get("rows", [])
    implied = [
        row["implied_value"] * 100
        for row in rows
        if row.get("unit") == "%" and row.get("implied_value") is not None and row.get("reachable", True)
    ]
    text = _section_prose(report) + " " + _thesis_prose(report)
    issues: list[str] = []
    for match in _MARKET_IMPLIED.finditer(text):
        claimed = float(match.group(1))
        subject = match.group(2).lower()
        if not implied:
            issues.append(
                f'The text says the market implies {claimed:g}% {subject}, but the reverse DCF '
                f'produced no reachable single-lever figure — this market-implied number is not '
                f'traceable to the calculation.'
            )
        elif not any(abs(claimed - value) <= 3.0 for value in implied):
            shown = ", ".join(f"{value:.0f}%" for value in implied)
            issues.append(
                f'The text says the market implies {claimed:g}% {subject}, which matches no '
                f'reverse-DCF implied value (computed: {shown}). Every "market implies" figure '
                f'must trace to the reverse DCF, not be invented.'
            )
    return issues


_ANNUALIZE_ARITH = re.compile(
    r"\$?\s*(\d{1,4}(?:\.\d+)?)\s*(?:billion|bn)\b[^.]{0,90}?annualiz[^.]{0,90}?"
    r"(\d{1,3}(?:\.\d+)?)\s*%[^.]{0,45}?(?:growth|increase|above|versus|vs\.?|over|higher)"
    r"[^.]{0,35}?(?:fy\s?20\d\d|prior[- ]year|last[- ]year|20\d\d|the\s+prior)",
    re.I,
)


def check_annualization_arithmetic(report) -> list[str]:
    """Deterministic (CRITICAL): a quarter annualised to a wrong YoY growth number (#3).

    Uses the canonical full-year revenue: a "$X bn ... annualized ... Y% growth over
    FY..." claim is checked by computing the real run-rate growth; if it is off by more
    than 10 percentage points, the arithmetic is wrong and blocks publication.
    """
    revenue_bn = None
    for metric in (report.canonical_metrics or []):
        if metric.get("key") == "revenue" and metric.get("value"):
            revenue_bn = metric["value"] / 1e9
            break
    if not revenue_bn:
        return []
    text = _section_prose(report) + " " + _assumption_prose(report) + " " + _thesis_prose(report)
    issues: list[str] = []
    for match in _ANNUALIZE_ARITH.finditer(text):
        quarter_bn = float(match.group(1))
        claimed_pct = float(match.group(2))
        run_rate = quarter_bn * 4
        # Guard: only treat the figure as a QUARTER if 4x is a plausible run-rate.
        if not (0.5 * revenue_bn <= run_rate <= 3.0 * revenue_bn):
            continue
        correct_pct = (run_rate / revenue_bn - 1) * 100
        if abs(correct_pct - claimed_pct) > 10:
            issues.append(
                f"Annualization arithmetic error: {quarter_bn:g}bn annualized is a {run_rate:g}bn "
                f"run-rate, ~{correct_pct:.0f}% above {revenue_bn:.0f}bn revenue — not the "
                f"{claimed_pct:g}% stated. A quarterly figure was annualized and mislabeled."
            )
    return issues


def check_numeric_consistency(report) -> list[str]:
    """Prose numbers attributed to a model driver or scenario must match the model.

    Extends the canonical-metric check to the forward DRIVERS (terminal margin, year-1
    growth, terminal growth) and the SCENARIO fair values — model outputs a prose
    passage can misquote. Deterministic and deliberately PRECISE: it only fires on a
    tight attribution ("terminal operating margin of X%", "bull case of $X"), so it
    does not flag a number that merely appears near the phrase (e.g. the current margin
    in a bridge), keeping false positives low.
    """
    issues: list[str] = []

    # -- forward drivers, from the assumption ledger ----------------------
    # A prose figure for a driver is valid if it matches EITHER our model assumption OR
    # the reverse-DCF implied value (prose legitimately cites both: "our terminal growth
    # of 2.5%" and "the price implies 7%"). Only a number matching NEITHER is a mismatch.
    priced = {row.get("key"): row for row in (report.priced_in or {}).get("rows", [])}

    def _implied_pct(key):
        row = priced.get(key)
        value = row.get("implied_value") if row else None
        return value * 100 if value is not None else None

    driver_specs = {
        # driver_key: (pattern, tolerance, label, priced_in key for the implied value)
        "terminal_margin": (
            r"terminal[- ]?(?:operating )?margin\s+(?:of|at|near|around|assumption of)\s+"
            r"(?:about |approximately |~)?(\d{1,3}(?:\.\d+)?)\s*%"
            r"|(\d{1,3}(?:\.\d+)?)\s*%\s+terminal[- ]?(?:operating )?margin",
            1.5, "terminal operating margin", "operating_margin"),
        "terminal_growth": (
            r"terminal growth(?:\s+rate)?\s+(?:of|at)\s+(?:about |~)?(\d{1,3}(?:\.\d+)?)\s*%"
            r"|(\d{1,3}(?:\.\d+)?)\s*%\s+terminal growth",
            0.8, "terminal growth", "terminal_growth"),
        "year_one_growth": (
            r"year[- ]?(?:one|1)\s+(?:revenue\s+)?growth\s+(?:of|at)\s+(?:about |~)?(\d{1,3}(?:\.\d+)?)\s*%",
            2.5, "year-1 revenue growth", None),
    }
    values = {a.get("key"): a.get("value") for a in (report.assumptions or [])
              if isinstance(a, dict)}
    text = _section_prose(report) + " " + _thesis_prose(report)
    for key, (pattern, tol, label, implied_key) in driver_specs.items():
        model_value = values.get(key)
        if model_value is None:
            continue
        acceptable = [model_value * 100]
        implied = _implied_pct(implied_key) if implied_key else None
        if implied is not None:
            acceptable.append(implied)
        if key == "year_one_growth" and report.market_implied_growth is not None:
            acceptable.append(report.market_implied_growth * 100)
        for match in re.finditer(pattern, text, re.I):
            claimed = float(next(g for g in match.groups() if g is not None))
            if not any(abs(claimed - value) <= tol for value in acceptable):
                refs = " / ".join(f"{v:.1f}%" for v in acceptable)
                issues.append(
                    f"The report states a {label} of {claimed:g}% but neither the model nor the "
                    f"DCF-implied value ({refs}) supports it — a numeric mismatch."
                )

    # -- scenario fair values ---------------------------------------------
    by_key = {c.get("key"): c.get("fair_value_per_share")
              for c in (report.scenarios or {}).get("cases", [])}
    for case in ("bear", "bull"):
        model_value = by_key.get(case)
        if not model_value:
            continue
        pattern = (rf"{case}[- ]case\s+(?:of|value of|fair value of|at)\s+"
                   rf"(?:USD |US\$|\$|₹|€)?\s*([\d,]+(?:\.\d+)?)")
        for match in re.finditer(pattern, text, re.I):
            claimed = float(match.group(1).replace(",", ""))
            if claimed > 0 and abs(claimed / model_value - 1) > 0.03:
                issues.append(
                    f"The report states a {case} case of {claimed:g} but the scenario model "
                    f"has {model_value:.1f} — a numeric mismatch; prose must quote the model value."
                )
    return issues


def check_valuation_integrity(report) -> list[str]:
    """Deterministic guard against valuation double-counting (item #1).

    The target price must equal the intrinsic DCF — the scenario/comps/consensus are
    cross-checks, never blended in. And the scenario BASE case must equal the DCF,
    since the scenario set is built from it. A divergence means the frameworks were
    combined or drifted apart, which is disqualifying.
    """
    issues: list[str] = []
    dcf = report.dcf_fair_value or report.fair_value
    blended = report.blended or {}
    target = blended.get("blended_value")
    if dcf and dcf > 0 and target and abs(target / dcf - 1) > 0.005:
        issues.append(
            f"Valuation double-counting: the target price ({target:,.2f}) is not the intrinsic "
            f"DCF ({dcf:,.2f}). The DCF, scenario expected value and comps must be reported "
            f"separately, not blended into a single averaged number."
        )
    cases = (report.scenarios or {}).get("cases", [])
    base = next((c for c in cases if c.get("key") == "base"), None)
    if base and dcf and dcf > 0:
        base_fv = base.get("fair_value_per_share")
        if base_fv and abs(base_fv / dcf - 1) > 0.02:
            issues.append(
                f"Scenario base case ({base_fv:,.2f}) does not match the DCF ({dcf:,.2f}); the "
                f"scenario framework must be built on the base-case DCF."
            )
    return issues


def check_segment_reconciliation(report) -> list[str]:
    """Deterministic: reported segment revenues must sum to consolidated revenue."""
    seg = report.segment_forecast or {}
    gap = seg.get("reconciliation_gap")
    if not seg.get("segments") or gap is None:
        return []
    if abs(gap) > 0.15:                      # a genuine sum mismatch, not extraction noise
        seg_sum = seg.get("latest_segment_sum", 0.0) or 0.0
        total = seg.get("latest_total_revenue", 0.0) or 0.0
        return [
            f"Segment revenues sum to {seg_sum / 1e9:,.1f}bn but consolidated revenue is "
            f"{total / 1e9:,.1f}bn — a {gap * 100:+.0f}% reconciliation gap; the segment set is "
            f"incomplete or mis-extracted."
        ]
    return []


# Severity of each check, and the publication gate (v8): CRITICAL and HIGH both BLOCK
# publication; MEDIUM warns; LOW passes. The system must never publish a report that
# still contains a known CRITICAL/HIGH issue, so both are on the gate. CRITICAL is
# reserved for deterministic numeric/structural contradictions; HIGH covers the
# integrity errors (unsupported market-implied claims, annualization, wrong-period
# labels, grounding failures) that a research report must not ship either.
CHECK_SEVERITY = {
    "consistency": "CRITICAL",              # probabilities sum, scenario order, rating sign
    "canonical_metric": "CRITICAL",         # a figure contradicting a canonical value
    "valuation_integrity": "CRITICAL",      # double-counting / base-case != DCF
    "segment_reconciliation": "CRITICAL",   # segments don't sum to consolidated revenue
    "annualization_arithmetic": "CRITICAL", # a quarter annualized to a wrong YoY number
    "numeric_consistency": "CRITICAL",      # a prose driver/scenario number != the model
    "citation": "HIGH",                     # grounding failure on a claim-heavy section
    "risk_direction": "HIGH",               # a downside risk that improves its metric
    "quarterly_annualization": "HIGH",      # a quarter annualized as annual (fuzzy)
    "market_implied": "HIGH",               # a market-implied figure not traced to reverse DCF
    "consensus_scenario": "HIGH",           # our probability pinned on external consensus
    "segment_end_market": "HIGH",           # segment % conflated with end-market %
    "metric_semantics": "HIGH",             # a quarterly label on an annual figure (period error)
    "forward_period": "MEDIUM",             # a forward trigger keyed to a past year
}
_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
BLOCKING_SEVERITIES = ("CRITICAL", "HIGH")


def run_qa(report) -> list[dict]:
    """Run every QA pass, tag each finding with a severity, and set the publication gate.

    v8 gate: a CRITICAL or HIGH finding BLOCKS publication (report.qa_status = "FAILED",
    report.blocked = True). MEDIUM/LOW are delivered as warnings. The report must never
    reach final output while a CRITICAL/HIGH issue remains; the correction loop
    (report/correction.py) tries to fix them first, then this gate withholds if it
    cannot. Idempotent: re-runnable by the correction loop without duplicating warnings.
    """
    results = {
        "consistency": check_consistency(report),
        "canonical_metric": check_metric_consistency(report),
        "valuation_integrity": check_valuation_integrity(report),
        "segment_reconciliation": check_segment_reconciliation(report),
        "annualization_arithmetic": check_annualization_arithmetic(report),
        "numeric_consistency": check_numeric_consistency(report),
        "citation": audit_citations(report),
        "risk_direction": check_risk_direction(report),
        "quarterly_annualization": check_quarterly_annualization(report),
        "market_implied": check_market_implied_claims(report),
        "consensus_scenario": check_consensus_scenario(report),
        "segment_end_market": check_segment_end_market(report),
        "metric_semantics": check_metric_semantics(report),
        "forward_period": check_forward_period_staleness(report),
    }

    findings: list[dict] = []
    for check_name, issues in results.items():
        severity = CHECK_SEVERITY.get(check_name, "MEDIUM")
        for message in issues:
            findings.append({"severity": severity, "check": check_name, "message": message})
    findings.sort(key=lambda f: _SEVERITY_ORDER[f["severity"]])

    # Idempotent warning refresh: drop any QA lines from a prior run before re-adding,
    # so the correction loop can call run_qa repeatedly without stacking duplicates.
    report.warnings = [w for w in report.warnings if not w.lstrip().startswith("[")
                       or "QA — " not in w]
    for finding in findings:
        log.info("QA [%s] %s", finding["severity"], finding["message"])
        report.warnings.append(f"[{finding['severity']}] QA — {finding['message']}")

    blocking = [f["message"] for f in findings if f["severity"] in BLOCKING_SEVERITIES]
    report.qa_findings = findings
    report.blocking_issues = blocking
    report.blocked = bool(blocking)
    report.qa_status = "FAILED" if blocking else "PASSED"
    return findings
