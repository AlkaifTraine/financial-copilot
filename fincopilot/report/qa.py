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
_LIFT_FV = re.compile(
    r"(?:rais|increas|lift|boost|improv|higher|expand)\w*\s+(?:our\s+|the\s+)?"
    r"(?:fair value|target price|intrinsic value|valuation)",
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


def run_qa(report) -> list[str]:
    """Run every QA pass; classify findings; set the publication gate.

    Critical failures — a dead or missing citation on a claim-heavy section, or a
    section figure that contradicts a canonical metric — block publication. Softer
    findings are surfaced in the report's limitations. All findings, blocking or
    not, are also listed under limitations so nothing is hidden.
    """
    citation = audit_citations(report)
    consistency = check_consistency(report)
    metric = check_metric_consistency(report)
    direction = check_risk_direction(report)
    semantics = check_metric_semantics(report)

    # A numeric contradiction (a figure disagreeing with itself or with a canonical
    # value), a citation failure on a claim-heavy section, a directional impossibility
    # (a downside risk that improves its metric), or a period/comparison mismatch (a
    # quarterly label on an annual figure) are each disqualifying in a research
    # report. All block publication.
    blocking = citation + consistency + metric + direction + semantics

    for issue in blocking:
        log.info("QA: %s", issue)
        report.warnings.append(f"Automated QA flagged: {issue}")

    report.blocking_issues = blocking
    report.blocked = bool(blocking)
    return blocking
