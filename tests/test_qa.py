"""
Unit tests for the report QA passes: citation grounding (#23) and internal
consistency (#22). Both are deterministic, so they are tested against small
hand-built report models with known defects — a clean report must stay silent,
and each specific defect must be caught.
"""

from __future__ import annotations

from fincopilot.report.models import Evidence, ReportModel, Section
from fincopilot.report.qa import audit_citations, check_consistency, run_qa


def _evidence(n: int) -> list[Evidence]:
    return [Evidence(doc_title=f"Doc {i}", section=None, page=i, source_url="u") for i in range(1, n + 1)]


class TestCitationAudit:
    def test_clean_section_is_silent(self):
        report = ReportModel(company_name="X", ticker="X")
        report.sections = [Section(
            key="s", title="Growth",
            paragraphs=["Revenue rose 60% [1].", "Margins held [2]."],
            evidence=_evidence(2),
        )]
        assert audit_citations(report) == []

    def test_dangling_marker_is_flagged(self):
        report = ReportModel(company_name="X", ticker="X")
        report.sections = [Section(
            key="s", title="Growth",
            paragraphs=["Revenue rose 60% [3]."],   # only 2 sources exist
            evidence=_evidence(2),
        )]
        issues = audit_citations(report)
        assert len(issues) == 1
        assert "[3]" in issues[0]

    def test_uncited_multiparagraph_section_is_flagged(self):
        report = ReportModel(company_name="X", ticker="X")
        report.sections = [Section(
            key="s", title="Outlook",
            paragraphs=["A claim with no citation.", "A second claim, also bare."],
            evidence=_evidence(3),
        )]
        issues = audit_citations(report)
        assert len(issues) == 1
        assert "without any inline citation" in issues[0]

    def test_short_section_without_evidence_is_not_flagged(self):
        # A single-paragraph section with no evidence available is brevity, not a defect.
        report = ReportModel(company_name="X", ticker="X")
        report.sections = [Section(key="s", title="Note", paragraphs=["One line."])]
        assert audit_citations(report) == []


class TestConsistency:
    def test_clean_report_is_silent(self):
        report = ReportModel(company_name="X", ticker="X", share_price=100.0, fair_value=120.0)
        report.rating = "BUY"
        report.upside = 0.20
        report.scenarios = {"cases": [
            {"key": "bear", "fair_value_per_share": 80.0, "probability": 0.25},
            {"key": "base", "fair_value_per_share": 120.0, "probability": 0.50},
            {"key": "bull", "fair_value_per_share": 160.0, "probability": 0.25},
        ]}
        assert check_consistency(report) == []

    def test_misordered_scenarios_flagged(self):
        report = ReportModel(company_name="X", ticker="X")
        report.scenarios = {"cases": [
            {"key": "bear", "fair_value_per_share": 130.0, "probability": 0.25},
            {"key": "base", "fair_value_per_share": 120.0, "probability": 0.50},
            {"key": "bull", "fair_value_per_share": 160.0, "probability": 0.25},
        ]}
        issues = check_consistency(report)
        assert any("not ordered" in i for i in issues)

    def test_probabilities_must_sum_to_one(self):
        report = ReportModel(company_name="X", ticker="X")
        report.scenarios = {"cases": [
            {"key": "bear", "fair_value_per_share": 80.0, "probability": 0.25},
            {"key": "base", "fair_value_per_share": 120.0, "probability": 0.30},
            {"key": "bull", "fair_value_per_share": 160.0, "probability": 0.25},
        ]}
        issues = check_consistency(report)
        assert any("sum to" in i for i in issues)

    def test_blended_outside_range_flagged(self):
        report = ReportModel(company_name="X", ticker="X")
        report.blended = {"blended_value": 200.0, "low": 80.0, "high": 160.0}
        issues = check_consistency(report)
        assert any("outside its own reconciled range" in i for i in issues)

    def test_rating_sign_contradiction_flagged(self):
        report = ReportModel(company_name="X", ticker="X")
        report.rating = "BUY"
        report.upside = -0.30           # BUY with the price above fair value
        issues = check_consistency(report)
        assert any("Rating is BUY" in i for i in issues)


class TestRunQa:
    def test_findings_appended_to_warnings(self):
        report = ReportModel(company_name="X", ticker="X")
        report.rating = "SELL"
        report.upside = 0.40            # SELL with big upside — a contradiction
        before = len(report.warnings)
        issues = run_qa(report)
        assert issues
        assert len(report.warnings) == before + len(issues)
        assert all(w.startswith("Automated QA flagged:") for w in report.warnings[before:])


class TestMetricGate:
    """The canonical-metric consistency scan and the publication gate."""

    def _report(self, section_text: str, fcf: float = 96.7e9):
        report = ReportModel(company_name="X", ticker="X", currency="USD")
        report.canonical_metrics = [{
            "key": "free_cash_flow", "label": "Free cash flow", "value": fcf,
            "unit": "currency", "period": "FY2026",
            "definition": "Operating cash flow minus capital expenditure, full fiscal year",
        }]
        report.sections = [Section(
            key="financials", title="Financial Performance",
            paragraphs=[section_text], evidence=_evidence(2),
        )]
        return report

    def test_contradicting_fcf_blocks_publication(self):
        report = self._report("Free cash flow was $34.9 billion [1] this year, a strong result [2].")
        run_qa(report)
        assert report.blocked is True
        assert any("free cash flow" in i.lower() for i in report.blocking_issues)

    def test_matching_fcf_does_not_block(self):
        report = self._report("Free cash flow reached $96.7 billion [1], a record [2].")
        run_qa(report)
        assert report.blocked is False

    def test_quarterly_labelled_figure_is_not_a_contradiction(self):
        report = self._report("Q1 free cash flow was $27.0 billion [1], up sequentially [2].")
        run_qa(report)
        assert report.blocked is False
