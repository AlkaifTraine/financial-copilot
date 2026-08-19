"""
Unit tests for the quantified risk assessment.

These avoid the network: the deterministic fallback is exercised directly, and
the model path is exercised with a stubbed ``complete_json`` so the parsing,
ordering and fallback-on-garbage behaviour are pinned without an API call.
"""

from __future__ import annotations

import fincopilot.report.risks as risks_mod
from fincopilot.report.risks import RiskAssessment, generate_risks
from fincopilot.valuation.models import (
    DCFResult,
    PricedInComparison,
    PricedInRow,
    ScenarioAnalysis,
    ScenarioCase,
    Valuation,
)


def _valuation() -> Valuation:
    """A valuation with the numbers the risk pass anchors to, no DCF object."""
    v = Valuation(ticker="TST", company_name="Test Co", currency="USD", share_price=100.0)
    v.priced_in = PricedInComparison(
        rows=[
            PricedInRow("revenue_cagr", "Revenue CAGR (10-year)", "%", 0.08, 0.30),
            PricedInRow("operating_margin", "Operating margin (mature)", "%", 0.36, None,
                        reachable=False, note="Even a 100% margin does not reach the price."),
        ],
        currency="USD", share_price=100.0, dcf_fair_value=60.0, horizon=10,
    )
    v.scenarios = ScenarioAnalysis(
        cases=[
            ScenarioCase(key="bear", label="Bear", probability=0.25, narrative="",
                         fair_value_per_share=40.0, upside=-0.60),
            ScenarioCase(key="base", label="Base", probability=0.50, narrative="",
                         fair_value_per_share=60.0, upside=-0.40),
            ScenarioCase(key="bull", label="Bull", probability=0.25, narrative="",
                         fair_value_per_share=90.0, upside=-0.10),
        ],
        currency="USD", share_price=100.0,
    )
    return v


class TestFallback:
    def test_fallback_is_marked_not_generated(self):
        assessment = generate_risks(None, None, _valuation(), use_model=False)
        assert isinstance(assessment, RiskAssessment)
        assert assessment.generated is False

    def test_fallback_surfaces_the_growth_gap(self):
        assessment = generate_risks(None, None, _valuation(), use_model=False)
        # The reverse-DCF growth gap must produce a concrete, anchored risk.
        growth = next((r for r in assessment.risks if "rowth" in r.risk), None)
        assert growth is not None
        assert "30.0%" in growth.valuation_impact   # the implied CAGR
        assert "8.0%" in growth.valuation_impact     # our base CAGR
        assert growth.early_warning                  # never blank

    def test_every_fallback_risk_is_complete(self):
        assessment = generate_risks(None, None, _valuation(), use_model=False)
        assert assessment.risks
        for risk in assessment.risks:
            assert risk.risk and risk.probability and risk.early_warning


class TestModelPath:
    def test_parses_and_preserves_order(self, monkeypatch):
        payload = {"risks": [
            {"risk": "Export controls", "description": "China restrictions cut Data Center sales.",
             "probability": "High", "financial_impact": "~20% of revenue at risk",
             "valuation_impact": "-15% to fair value", "early_warning": "China revenue disclosure"},
            {"risk": "Margin normalisation", "description": "Competition compresses pricing.",
             "probability": "Medium", "financial_impact": "Gross margin",
             "valuation_impact": "bear case USD 40", "early_warning": "Gross margin trend"},
        ]}
        monkeypatch.setattr(risks_mod, "complete_json", lambda *a, **k: payload)

        # A DCF object must be present for the model path to run; a bare stub with
        # the attribute set to a truthy value is enough for the guard.
        v = _valuation()
        v.dcf = DCFResult(fair_value_per_share=60.0)
        assessment = generate_risks(_Company(), _History(), v, qualitative_context="ctx")

        assert assessment.generated is True
        assert [r.risk for r in assessment.risks] == ["Export controls", "Margin normalisation"]
        assert assessment.risks[0].probability == "High"

    def test_garbage_payload_falls_back(self, monkeypatch):
        monkeypatch.setattr(risks_mod, "complete_json", lambda *a, **k: "not a dict")
        v = _valuation()
        v.dcf = DCFResult(fair_value_per_share=60.0)
        assessment = generate_risks(_Company(), _History(), v)
        assert assessment.generated is False

    def test_empty_risk_list_falls_back(self, monkeypatch):
        monkeypatch.setattr(risks_mod, "complete_json", lambda *a, **k: {"risks": []})
        v = _valuation()
        v.dcf = DCFResult(fair_value_per_share=60.0)
        assessment = generate_risks(_Company(), _History(), v)
        assert assessment.generated is False


class _Company:
    name = "Test Co"
    ticker = "TST"


class _History:
    years: list = []

    def growth_rates(self, _field):
        return []
