"""
Self-correction loop (v8): regenerate the component responsible for a blocking QA
finding, re-run QA, repeat until clean or the retry budget is spent — and never clear
the block while an unfixable (deterministic) contradiction remains.
"""

from __future__ import annotations

from types import SimpleNamespace

from fincopilot.report.correction import blocking_findings, run_correction_loop


def _report(*findings):
    return SimpleNamespace(qa_findings=list(findings), blocked=bool(findings), qa_status="FAILED")


def _finding(severity, check, message="x"):
    return {"severity": severity, "check": check, "message": message}


class TestCorrectionLoop:
    def test_fixable_high_issue_is_corrected_then_passes(self):
        report = _report(_finding("HIGH", "market_implied"))   # -> "thesis" component
        state = {"regens": 0}

        def regen_thesis(corrections):
            state["regens"] += 1

        def qa(rep):
            if state["regens"] > 0:                            # the re-draft fixed it
                rep.qa_findings = []
                rep.blocked = False
                rep.qa_status = "PASSED"
            return rep.qa_findings

        attempts = run_correction_loop(report, {"thesis": regen_thesis}, qa=qa)
        assert attempts == 1
        assert report.qa_findings == [] and report.blocked is False

    def test_deterministic_issue_is_unfixable_and_stays_blocked(self):
        # A consistency contradiction maps to no component; regenerating prose can't fix it.
        report = _report(_finding("CRITICAL", "consistency", "probabilities do not sum"))

        def qa(rep):
            return rep.qa_findings                             # never clears

        attempts = run_correction_loop(report, {"thesis": lambda c: None}, qa=qa)
        assert attempts == 0                                   # nothing regenerable -> stop
        assert blocking_findings(report)                       # still blocked

    def test_retries_exhaust_then_report_stays_blocked(self):
        # A fixable-component finding the regen never actually resolves.
        report = _report(_finding("HIGH", "market_implied"))
        calls = {"n": 0}

        def regen(corrections):
            calls["n"] += 1

        def qa(rep):
            return rep.qa_findings                             # regen doesn't fix it

        attempts = run_correction_loop(report, {"thesis": regen}, qa=qa, max_retries=2)
        assert attempts == 2 and calls["n"] == 2               # tried the budget
        assert blocking_findings(report)                       # still blocked

    def test_medium_findings_do_not_trigger_correction(self):
        report = _report(_finding("MEDIUM", "forward_period"))
        report.blocked = False

        def qa(rep):
            raise AssertionError("qa should not be re-run for a MEDIUM finding")

        attempts = run_correction_loop(report, {"forward": lambda c: None}, qa=qa)
        assert attempts == 0

    def test_feedback_is_passed_to_the_regenerator(self):
        report = _report(_finding("HIGH", "risk_direction", "a downside that improves the metric"))
        received = {}

        def regen_risks(corrections):
            received["msgs"] = corrections
            report.qa_findings = []                            # fixed
            report.blocked = False

        run_correction_loop(report, {"risks": regen_risks}, qa=lambda r: r.qa_findings)
        assert received["msgs"] == ["a downside that improves the metric"]


class TestCorrectionEndToEnd:
    """The exact path build_report wires: run_qa -> run_correction_loop with a real
    ReportModel and a sections regenerator."""

    def test_bad_annualization_section_is_corrected_then_passes(self):
        from fincopilot.report.models import Evidence, ReportModel, Section
        from fincopilot.report.qa import run_qa

        def _ev():
            return [Evidence(doc_title="10-K", section=None, page=1, source_url="u")]

        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.canonical_metrics = [{"key": "revenue", "label": "Revenue", "value": 215.9e9,
                                     "unit": "currency", "period": "FY2026", "definition": "x"}]
        bad = ("The $78 billion Q1 FY2027 revenue guidance annualized implies about 14% growth "
               "over FY2026 [1].")
        report.sections = [Section(key="outlook", title="Outlook", paragraphs=[bad], evidence=_ev())]

        run_qa(report)
        assert report.blocked is True                       # annualization CRITICAL -> blocked

        calls = {"n": 0}

        def regen_sections(corrections):
            calls["n"] += 1
            assert corrections and any("annualiz" in c.lower() for c in corrections)
            report.sections = [Section(key="outlook", title="Outlook",
                                       paragraphs=["Management provided a near-term outlook we "
                                                   "track on a quarterly basis [1]."],
                                       evidence=_ev())]

        run_correction_loop(report, {"sections": regen_sections})
        assert calls["n"] >= 1                              # the loop regenerated sections
        assert report.blocked is False and report.qa_status == "PASSED"   # fixed
        assert "annualized" not in " ".join(report.sections[0].paragraphs)

    def test_unfixable_report_is_never_published(self):
        # A deterministic contradiction (probabilities) cannot be fixed by regenerating
        # prose; the loop must leave the report blocked (never published).
        from fincopilot.report.models import ReportModel
        from fincopilot.report.qa import run_qa

        report = ReportModel(company_name="X", ticker="X", currency="USD")
        report.scenarios = {"cases": [
            {"key": "bear", "fair_value_per_share": 80.0, "probability": 0.25},
            {"key": "base", "fair_value_per_share": 120.0, "probability": 0.30},   # sums to 0.80
            {"key": "bull", "fair_value_per_share": 160.0, "probability": 0.25},
        ]}
        run_qa(report)
        assert report.blocked is True
        run_correction_loop(report, {"sections": lambda c: None})
        assert report.blocked is True and report.qa_status == "FAILED"   # still blocked
