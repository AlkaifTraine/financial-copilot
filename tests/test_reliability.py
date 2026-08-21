"""
Post-generation reliability scorecard: the numeric-traceability ("hallucination") check
plus citation coverage and the composite grade.
"""

from __future__ import annotations

from fincopilot.report.models import Evidence, ReportModel, Section
from fincopilot.report.reliability import compute_reliability


def _report(paragraphs, *, canonical=None, assumptions=None, snippets=None,
            confidence="High"):
    report = ReportModel(company_name="X", ticker="X", currency="USD")
    report.canonical_metrics = canonical or []
    report.assumptions = assumptions or []
    report.qa_findings = []
    report.qa_status = "PASSED"
    report.valuation_confidence = confidence
    evidence = [Evidence(doc_title="d", section=None, page=1, source_url="u", snippet=s)
                for s in (snippets or [])]
    report.sections = [Section(key="s", title="S", paragraphs=paragraphs, evidence=evidence)]
    return report


class TestNumericTraceability:
    def test_all_figures_traced_scores_high(self):
        report = _report(
            ["Revenue was $216 billion [1] with a 48% terminal margin [1]."],
            canonical=[{"key": "revenue", "value": 216e9, "unit": "currency"}],
            assumptions=[{"key": "terminal_margin", "value": 0.48, "unit": "%"}])
        rel = compute_reliability(report)
        assert rel["figures_checked"] == 2 and rel["figures_traced"] == 2
        assert rel["unverified_figures_pct"] == 0.0
        assert rel["citation_coverage_pct"] == 100
        assert rel["grade"] == "A"

    def test_untraceable_figure_is_counted(self):
        report = _report(["Margins hit 99% [1]."],
                         assumptions=[{"key": "terminal_margin", "value": 0.48, "unit": "%"}])
        rel = compute_reliability(report)
        assert rel["unverified_figures_pct"] == 100.0     # 99% matches nothing

    def test_source_grounded_figure_counts_as_traced(self):
        # A number that appears in the cited source snippet is traced even if it is not
        # one of the model's own numbers.
        report = _report(["The Data Center segment grew to $115 billion [1]."],
                         snippets=["Data Center revenue was $115 billion in FY2026"])
        rel = compute_reliability(report)
        assert rel["figures_traced"] == 1 and rel["unverified_figures_pct"] == 0.0

    def test_no_figures_is_not_a_failure(self):
        report = _report(["The company designs accelerated-computing platforms [1]."])
        rel = compute_reliability(report)
        assert rel["figures_checked"] == 0 and rel["unverified_figures_pct"] == 0.0


class TestScorecard:
    def test_citation_coverage(self):
        report = _report(["A cited claim [1].", "An uncited claim."])
        rel = compute_reliability(report)
        assert rel["citation_coverage_pct"] == 50

    def test_low_confidence_and_untraceable_lower_the_grade(self):
        report = _report(["Growth of 200% and margins of 150% [1] — figures with no basis."],
                         confidence="Low")
        rel = compute_reliability(report)
        assert rel["unverified_figures_pct"] > 0
        assert rel["score"] < 85 and rel["grade"] in ("B", "C", "D")

    def test_scorecard_has_all_fields(self):
        rel = compute_reliability(_report(["Revenue $216 billion [1]."],
                                          canonical=[{"key": "revenue", "value": 216e9,
                                                      "unit": "currency"}]))
        for key in ("score", "grade", "label", "unverified_figures_pct",
                    "traceable_figures_pct", "figures_checked", "figures_traced",
                    "citation_coverage_pct", "qa_status", "qa_findings",
                    "source_freshness_pct", "valuation_confidence"):
            assert key in rel
