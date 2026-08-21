"""
Unit tests for the report QA passes: citation grounding (#23) and internal
consistency (#22). Both are deterministic, so they are tested against small
hand-built report models with known defects — a clean report must stay silent,
and each specific defect must be caught.
"""

from __future__ import annotations

import pytest

from fincopilot.report.models import Evidence, ReportModel, Section
from fincopilot.report.qa import (
    audit_citations,
    check_consensus_scenario,
    check_consistency,
    check_forward_period_staleness,
    check_metric_semantics,
    check_quarterly_annualization,
    check_risk_direction,
    check_segment_end_market,
    check_segment_reconciliation,
    check_annualization_arithmetic,
    check_market_implied_claims,
    check_valuation_integrity,
    run_qa,
)


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
        findings = run_qa(report)
        assert findings
        assert all(isinstance(f, dict) and "severity" in f for f in findings)
        assert len(report.warnings) == before + len(findings)
        assert all(w.startswith("[") and "QA — " in w for w in report.warnings[before:])


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


class TestRiskDirection:
    """Directional QA: a downside risk cannot improve its own metric. Advisory —
    it flags the finding but must NOT block the report (regex-over-prose heuristic)."""

    def _report(self, financial_impact="", valuation_impact=""):
        report = ReportModel(company_name="X", ticker="X", currency="USD")
        report.risks = [{
            "risk": "Supply chain", "financial_impact": financial_impact,
            "valuation_impact": valuation_impact,
        }]
        return report

    def test_reduce_toward_higher_value_is_flagged(self):
        report = self._report(
            financial_impact="A disruption could reduce revenue CAGR from our 13.3% base case "
                             "toward market-implied 25.3%."
        )
        issues = check_risk_direction(report)
        assert any("directional" in i.lower() for i in issues)

    def test_clean_downside_not_flagged(self):
        report = self._report(
            financial_impact="A disruption could cut revenue CAGR from our 13.3% base toward 8.0%."
        )
        assert check_risk_direction(report) == []

    def test_risk_that_raises_fair_value_is_flagged(self):
        report = self._report(valuation_impact="This would raise our fair value materially.")
        assert any("wrong way" in i.lower() for i in check_risk_direction(report))

    def test_higher_valuation_phrase_is_not_a_false_positive(self):
        # A risk mentioning "a higher valuation multiple" must NOT trip the fair-value check.
        report = self._report(
            valuation_impact="Peers trade at a higher valuation, and any de-rating toward them "
                             "would pressure the stock."
        )
        assert check_risk_direction(report) == []

    def test_directional_finding_is_advisory_not_blocking(self):
        report = self._report(
            financial_impact="A disruption could reduce revenue CAGR from our 13.3% base case "
                             "toward market-implied 25.3%."
        )
        findings = run_qa(report)
        assert report.blocked is False                                        # advisory, not a gate
        assert any(f["severity"] == "HIGH" and "directional" in f["message"].lower()
                   for f in findings)                                          # surfaced as HIGH
        assert any("directional" in w.lower() for w in report.warnings)       # and in the QA notes


class TestMetricSemantics:
    """Semantic QA: a metric's period label must match its reading. Advisory."""

    def _report(self, label, assumption=""):
        report = ReportModel(company_name="X", ticker="X", currency="USD")
        report.forward = {"watch_items": [{
            "metric": label, "assumption": assumption, "current": "", "expected": "",
            "trend": "", "bull_bear": "",
        }]}
        return report

    def test_quarterly_label_on_annual_metric_is_flagged(self):
        report = self._report("Quarterly revenue growth rate", assumption="Our 65% YoY FY2026 growth")
        issues = check_metric_semantics(report)
        assert any("mismatch" in i.lower() for i in issues)

    def test_matching_period_not_flagged(self):
        report = self._report("Revenue growth (YoY)", assumption="Our 65% YoY growth")
        assert check_metric_semantics(report) == []

    def test_semantic_finding_is_advisory_not_blocking(self):
        report = self._report("Quarterly revenue growth rate", assumption="Our 65% YoY FY2026 growth")
        run_qa(report)
        assert report.blocked is False


class TestFinancialTypeSafety:
    """Advisory validators: segment/end-market, quarter-as-annual, stale forward
    triggers, consensus-vs-scenario. All flag, none block."""

    def _with_sections(self, *paragraphs):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.sections = [Section(key="s", title="S", paragraphs=list(paragraphs),
                                   evidence=_evidence(2))]
        return report

    def test_segment_vs_endmarket_contradiction_flagged(self):
        report = self._with_sections(
            "The Compute & Networking segment generated $193.5B, roughly 90% of FY2026 revenue [1].",
            "Customer concentration is high in Compute & Networking, about 60% of revenue [2].",
        )
        issues = check_segment_end_market(report)
        assert any("conflated" in i.lower() for i in issues)

    def test_consistent_segment_share_not_flagged(self):
        report = self._with_sections(
            "Compute & Networking was 90% of FY2026 revenue [1]; it remained about 90% of revenue [2].",
        )
        assert check_segment_end_market(report) == []

    def test_quarter_annualized_as_annual_flagged(self):
        report = self._with_sections(
            "The $78B Q1 FY2027 guidance annualized implies about 14% growth over FY2026 [1].",
        )
        assert any("run-rate" in i.lower() for i in check_quarterly_annualization(report))

    def test_labelled_run_rate_not_flagged(self):
        report = self._with_sections(
            "The $78B Q1 figure is a ~$312B run-rate (4x the quarter), not annual guidance [1].",
        )
        assert check_quarterly_annualization(report) == []

    def test_stale_forward_trigger_flagged(self):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.canonical_metrics = [{"key": "rev", "label": "Revenue", "value": 1.0,
                                     "unit": "currency", "period": "FY2026", "definition": "x"}]
        report.forward = {"watch_items": [{
            "metric": "Operating margin", "assumption": "x", "current": "x", "trend": "x",
            "expected": "Operating margin remains above 54% in FY2024", "bull_bear": "x",
        }]}
        assert any("historical" in i.lower() for i in check_forward_period_staleness(report))

    def test_future_trigger_not_flagged(self):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.canonical_metrics = [{"key": "rev", "label": "Revenue", "value": 1.0,
                                     "unit": "currency", "period": "FY2026", "definition": "x"}]
        report.forward = {"watch_items": [{
            "metric": "Operating margin", "assumption": "x", "current": "x", "trend": "x",
            "expected": "Operating margin stays above 54% through FY2028", "bull_bear": "x",
        }]}
        assert check_forward_period_staleness(report) == []

    def test_consensus_tied_to_scenario_probability_flagged(self):
        report = self._with_sections(
            "The $300 analyst consensus assumes our 25% bull-case probability [1].",
        )
        assert any("distinct" in i.lower() for i in check_consensus_scenario(report))

    def test_all_type_safety_checks_are_advisory(self):
        report = self._with_sections(
            "Compute & Networking was 90% of FY2026 revenue [1]; also Compute & Networking ~60% of revenue [2].",
            "The $78B Q1 FY2027 guidance annualized implies 14% growth over FY2026 [1].",
        )
        run_qa(report)
        assert report.blocked is False   # advisory only, never blocks


class TestQaSeverityGate:
    """CRITICAL findings block (QA FAILED); softer findings deliver (QA PASSED)."""

    def _valued(self, dcf=79.07, target=None, base_fv=None):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.dcf_fair_value = dcf
        report.fair_value = dcf
        if target is not None:
            report.blended = {"blended_value": target}
        if base_fv is not None:
            report.scenarios = {"cases": [
                {"key": "bear", "fair_value_per_share": 35.0, "probability": 0.25},
                {"key": "base", "fair_value_per_share": base_fv, "probability": 0.50},
                {"key": "bull", "fair_value_per_share": 190.0, "probability": 0.25},
            ]}
        return report

    def test_double_counted_target_is_critical(self):
        # Target (blended) != intrinsic DCF -> double-counting -> CRITICAL, blocks.
        report = self._valued(dcf=79.07, target=95.83)
        run_qa(report)
        assert report.qa_status == "FAILED" and report.blocked is True
        assert any("double-counting" in i.lower() for i in report.blocking_issues)

    def test_target_equal_to_dcf_passes(self):
        report = self._valued(dcf=79.07, target=79.07, base_fv=79.07)
        run_qa(report)
        assert report.qa_status == "PASSED" and report.blocked is False

    def test_valuation_integrity_direct(self):
        assert check_valuation_integrity(self._valued(dcf=79.07, target=79.07)) == []
        assert check_valuation_integrity(self._valued(dcf=79.07, target=95.83))

    def test_scenario_base_must_match_dcf(self):
        report = self._valued(dcf=79.07, target=79.07, base_fv=120.0)   # base != DCF
        run_qa(report)
        assert report.blocked is True

    def test_segment_non_reconciliation_is_critical(self):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.segment_forecast = {
            "segments": [{"name": "A"}, {"name": "B"}],
            "reconciliation_gap": -0.30,          # 30% short of consolidated
            "latest_segment_sum": 150e9, "latest_total_revenue": 215e9,
        }
        assert check_segment_reconciliation(report)
        run_qa(report)
        assert report.blocked is True and report.qa_status == "FAILED"

    def test_small_segment_gap_is_not_critical(self):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.segment_forecast = {
            "segments": [{"name": "A"}], "reconciliation_gap": 0.04,
            "latest_segment_sum": 207e9, "latest_total_revenue": 215e9,
        }
        assert check_segment_reconciliation(report) == []

    def test_probabilities_not_summing_blocks(self):
        report = ReportModel(company_name="X", ticker="X")
        report.scenarios = {"cases": [
            {"key": "bear", "fair_value_per_share": 80.0, "probability": 0.25},
            {"key": "base", "fair_value_per_share": 120.0, "probability": 0.30},
            {"key": "bull", "fair_value_per_share": 160.0, "probability": 0.25},
        ]}
        run_qa(report)
        assert report.blocked is True and report.qa_status == "FAILED"


class TestReverseDcfTraceability:
    """#2: a 'market implies X%' claim must trace to a reverse-DCF value."""

    def _report(self, prose, implied):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.sections = [Section(key="t", title="Thesis", paragraphs=[prose], evidence=_evidence(2))]
        report.priced_in = {"rows": [
            {"label": lbl, "unit": "%", "implied_value": v, "reachable": True}
            for lbl, v in implied
        ]}
        return report

    def test_invented_margin_claim_flagged(self):
        report = self._report(
            "At today's price the market assumes >60% operating margins forever [1].",
            [("Revenue CAGR", 0.256), ("Operating margin", 0.882)],
        )
        assert any("trace to the reverse" in i.lower() for i in check_market_implied_claims(report))

    def test_traceable_claim_not_flagged(self):
        report = self._report(
            "The price requires the market to imply 26% revenue growth over the decade [1].",
            [("Revenue CAGR", 0.256), ("Operating margin", 0.882)],
        )
        assert check_market_implied_claims(report) == []


class TestAnnualizationArithmetic:
    """#3 CRITICAL: a quarter annualized to a wrong YoY growth number blocks."""

    def _report(self, prose, revenue=215.9e9):
        report = ReportModel(company_name="NVIDIA", ticker="NVDA", currency="USD")
        report.canonical_metrics = [{"key": "revenue", "label": "Revenue", "value": revenue,
                                     "unit": "currency", "period": "FY2026", "definition": "x"}]
        report.sections = [Section(key="a", title="A", paragraphs=[prose], evidence=_evidence(2))]
        return report

    def test_wrong_annualized_growth_is_critical(self):
        report = self._report("$78 billion Q1 guidance annualized implies ~14% growth over FY2026 [1].")
        assert check_annualization_arithmetic(report)      # 78x4=312 -> ~45%, not 14%
        run_qa(report)
        assert report.blocked is True and report.qa_status == "FAILED"

    def test_correct_run_rate_not_flagged(self):
        report = self._report("$78 billion annualizes to a run-rate ~45% above the prior FY2026 year [1].")
        assert check_annualization_arithmetic(report) == []

    def test_annual_figure_not_treated_as_quarter(self):
        # A full-year figure (4x would be implausible) must not trip the check.
        report = self._report("$216 billion annualized is 30% growth over FY2025 [1].")
        assert check_annualization_arithmetic(report) == []


class TestMarginBridge:
    def test_parses_bridge_and_confidence(self):
        from fincopilot.valuation.assumptions import _model_margin_bridge
        proposal = {"terminal_operating_margin": {
            "value": 0.48, "margin_confidence": "medium",
            "margin_bridge": [
                {"component": "Current operating margin", "value": 60.0},
                {"component": "-Pricing normalization", "value": -8.0},
                {"component": "-Competitive pressure", "value": -4.0},
            ],
        }}
        bridge, confidence = _model_margin_bridge(proposal)
        assert confidence == "Medium"
        assert bridge[0] == {"component": "Current operating margin", "value": 60.0}
        assert sum(s["value"] for s in bridge) == pytest.approx(48.0)

    def test_missing_bridge_is_empty(self):
        from fincopilot.valuation.assumptions import _model_margin_bridge
        assert _model_margin_bridge({"terminal_operating_margin": {"value": 0.48}}) == ([], "")
