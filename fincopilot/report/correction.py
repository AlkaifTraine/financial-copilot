"""
Self-correction loop (v8).

When QA finds a HIGH/CRITICAL issue, regenerate the component responsible, re-run QA,
and repeat until the report is clean or the retry budget is spent. The report is NEVER
published while a HIGH/CRITICAL issue remains: if the loop cannot clear it, the QA gate
leaves ``report.blocked = True`` and the app withholds the report with
"REPORT BLOCKED — unresolved integrity issue."

The loop is decoupled from generation: callers pass a ``regenerators`` map of
``{component: callable(corrections: list[str]) -> None}`` that regenerates that part of
the report with the QA feedback folded in. This keeps the loop unit-testable (inject
stub regenerators) and lets build_report wire the real section/thesis/risk/forward
generators.
"""

from __future__ import annotations

import logging

from .qa import BLOCKING_SEVERITIES, run_qa

log = logging.getLogger(__name__)

MAX_QA_RETRIES = 3      # correction attempts before the report is left blocked

# Which regenerable component owns each check's findings. Deterministic checks map to
# None: a numeric/structural contradiction there is a data/calculation bug that
# regenerating prose cannot fix, so the loop stops and the report stays blocked.
CHECK_TO_COMPONENT: dict[str, str | None] = {
    "annualization_arithmetic": "sections",
    "quarterly_annualization": "sections",
    "canonical_metric": "sections",
    "numeric_consistency": "sections",
    "consensus_scenario": "sections",
    "segment_end_market": "sections",
    "citation": "sections",
    "metric_semantics": "forward",
    "forward_period": "forward",
    "market_implied": "thesis",
    "risk_direction": "risks",
    # deterministic — not fixable by regenerating prose:
    "consistency": None,
    "valuation_integrity": None,
    "segment_reconciliation": None,
    # A mis-specified model cannot be written around. Rewriting the prose to
    # sound less confident about a 93.8% implied margin would hide the defect
    # rather than fix it — the inputs have to change, which is a human's call.
    "valuation_plausibility": None,
}


def correction_instruction(corrections: list[str] | None) -> str:
    """A prompt suffix that hands the QA feedback back to a generator for a re-draft."""
    if not corrections:
        return ""
    joined = "\n- ".join(corrections)
    return (
        "\n\nSTOP — a prior draft of this content FAILED automated integrity QA with the "
        "issues below. You MUST fix every one and must NOT repeat them:\n- " + joined +
        "\n(Where an issue names a wrong period, unit, annualization, management attribution, "
        "or an unsupported number, correct the underlying statement — do not merely reword it.)"
    )


def section_level_findings(report) -> dict:
    """Attribute section-facing QA findings to the specific section that triggers them.

    Runs the section-prose checks against each section in isolation, so the correction
    loop can regenerate ONLY the offending section (not the whole report) with feedback
    scoped to that section. Returns {section_key: [messages]}.
    """
    from types import SimpleNamespace

    from . import qa

    section_checks = (
        qa.check_annualization_arithmetic, qa.check_quarterly_annualization,
        qa.check_metric_consistency, qa.check_consensus_scenario,
        qa.check_segment_end_market, qa.check_market_implied_claims,
        qa.check_numeric_consistency, qa.audit_citations,
    )
    found: dict[str, list[str]] = {}
    for section in getattr(report, "sections", None) or []:
        probe = SimpleNamespace(
            sections=[section],
            canonical_metrics=getattr(report, "canonical_metrics", None) or [],
            priced_in=getattr(report, "priced_in", None) or {},
            assumptions=getattr(report, "assumptions", None) or [],
            scenarios=getattr(report, "scenarios", None) or {},
            risks=[], forward={}, thesis={},
        )
        for check in section_checks:
            try:
                for message in check(probe):
                    found.setdefault(section.key, []).append(message)
            except Exception:                       # a check that can't run on a bare section
                continue
    return found


def blocking_findings(report) -> list[dict]:
    return [f for f in report.qa_findings if f["severity"] in BLOCKING_SEVERITIES]


def run_correction_loop(report, regenerators: dict, *, max_retries: int = MAX_QA_RETRIES,
                        qa=run_qa) -> int:
    """Regenerate components responsible for blocking findings, re-run QA, until clean.

    ``regenerators`` maps a component name to ``callable(corrections: list[str])`` that
    regenerates it with the QA feedback. Returns the number of correction attempts made.
    Assumes ``run_qa`` has already populated ``report.qa_findings`` once.
    """
    attempts = 0
    for _ in range(max_retries):
        blocking = blocking_findings(report)
        if not blocking:
            break

        by_component: dict[str, list[str]] = {}
        for finding in blocking:
            component = CHECK_TO_COMPONENT.get(finding["check"])
            if component is not None:
                by_component.setdefault(component, []).append(finding["message"])

        # Only act on components we can actually regenerate. If nothing is regenerable
        # (e.g. a deterministic contradiction, or a component with no regenerator), stop
        # — the report stays blocked rather than looping pointlessly.
        regenerable = {c: msgs for c, msgs in by_component.items() if c in regenerators}
        if not regenerable:
            break

        attempts += 1
        for component, messages in regenerable.items():
            log.info("correction attempt %d: regenerating %s for %d issue(s)",
                     attempts, component, len(messages))
            try:
                regenerators[component](messages)
            except Exception as exc:                      # never let a regen crash the build
                log.warning("correction regen of %s failed: %s", component, exc)

        qa(report)      # re-runs the gate; updates report.blocked / qa_status / findings

    if blocking_findings(report):
        log.warning("correction loop exhausted; report remains blocked (%d issue(s))",
                    len(blocking_findings(report)))
    return attempts
