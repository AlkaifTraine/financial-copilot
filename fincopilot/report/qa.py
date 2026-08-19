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


def run_qa(report) -> list[str]:
    """Run both passes and surface any findings in the report's limitations."""
    issues = audit_citations(report) + check_consistency(report)
    for issue in issues:
        log.info("QA: %s", issue)
        report.warnings.append(f"Automated QA flagged: {issue}")
    return issues
