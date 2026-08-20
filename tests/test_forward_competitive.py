"""
Unit tests for the forward view (catalysts + monitoring) and the competitive
moat / management-vs-us passes. The network is never touched: the deterministic
fallbacks are exercised directly and the model paths use a stubbed complete_json.
"""

from __future__ import annotations

import fincopilot.report.competitive as competitive_mod
import fincopilot.report.monitoring as monitoring_mod
from fincopilot.report.competitive import CompetitiveAnalysis, generate_competitive
from fincopilot.report.monitoring import ForwardView, generate_forward
from fincopilot.valuation.models import (
    Assumption,
    AssumptionLedger,
    DCFResult,
    PricedInComparison,
    PricedInRow,
    Valuation,
)


def _valuation(with_dcf: bool = True) -> Valuation:
    v = Valuation(ticker="TST", company_name="Test Co", currency="USD", share_price=100.0)
    if with_dcf:
        v.dcf = DCFResult(fair_value_per_share=60.0)
    ledger = AssumptionLedger()
    ledger.add(Assumption("terminal_margin", "Terminal operating margin", 0.36, "%", "model", ""))
    ledger.add(Assumption("year_one_growth", "Year 1 revenue growth", 0.20, "%", "model", ""))
    ledger.add(Assumption("terminal_growth", "Terminal growth", 0.025, "%", "model", ""))
    v.assumptions = ledger
    v.priced_in = PricedInComparison(
        rows=[
            PricedInRow("revenue_cagr", "Revenue CAGR (10-year)", "%", 0.08, 0.30),
            PricedInRow("operating_margin", "Operating margin (mature)", "%", 0.36, 0.55),
        ],
        currency="USD", share_price=100.0, dcf_fair_value=60.0, horizon=10,
    )
    return v


class _History:
    def __init__(self):
        from fincopilot.fundamentals.models import FiscalYear
        self.years = [FiscalYear(fiscal_year=2025, period_end="2025-01-31", revenue=130e9)]

    @property
    def latest(self):
        return self.years[-1]

    def growth_rates(self, _field):
        return [(2025, 0.60)]


class _Company:
    name = "Test Co"
    ticker = "TST"


class TestForwardFallback:
    def test_fallback_builds_from_numbers(self):
        view = generate_forward(_Company(), _History(), _valuation(), use_model=False)
        assert view.generated is False
        assert view.catalysts or view.watch_items
        # The growth watch item must carry both the implied and our base CAGR.
        watch = next((w for w in view.watch_items if "rowth" in w.metric), None)
        assert watch is not None
        # It maps to our base-case assumption and carries the bull/bear thresholds.
        assert "8.0%" in watch.assumption
        assert "30.0%" in watch.bull_bear and "8.0%" in watch.bull_bear


class TestForwardModelPath:
    def test_parses_catalysts_and_watch(self, monkeypatch):
        payload = {
            "catalysts": [
                {"event": "Q4 earnings", "timing": "Feb 2026", "direction": "Two-sided",
                 "metric": "Data Center revenue"},
            ],
            "watch_items": [
                {"metric": "DC revenue", "current": "$40bn/q", "trend": "up",
                 "expected": "decelerating", "why": "drives the whole thesis"},
            ],
        }
        monkeypatch.setattr(monitoring_mod, "complete_json", lambda *a, **k: payload)
        view = generate_forward(_Company(), _History(), _valuation())
        assert view.generated is True
        assert view.catalysts[0].event == "Q4 earnings"
        assert view.watch_items[0].metric == "DC revenue"

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setattr(monitoring_mod, "complete_json", lambda *a, **k: "nope")
        assert generate_forward(_Company(), _History(), _valuation()).generated is False


class TestCompetitiveFallback:
    def test_fallback_anchors_moat_to_terminal_margin(self):
        analysis = generate_competitive(_Company(), _History(), _valuation(), use_model=False)
        assert analysis.generated is False
        assert "36.0%" in analysis.moat_summary


class TestCompetitiveModelPath:
    def test_parses_moat_strength_direction_and_management(self, monkeypatch):
        payload = {
            "moat_strength": "Wide",
            "moat_direction": "Narrowing",
            "moat_summary": "CUDA lock-in defends the margin, but custom silicon is closing in.",
            "dimensions": [
                {"dimension": "Software ecosystem (CUDA)", "assessment": "Deep lock-in", "strength": "Strong"},
            ],
            "competitive_threats": ["custom ASICs"],
            "management_vs_us": [
                {"topic": "Growth", "management_statement": "We expect continued strong demand",
                 "source": "FY2026 10-K MD&A", "our_interpretation": "Bullish tone",
                 "our_model": "growth fades toward mature rate", "difference": "We are more cautious"},
            ],
        }
        monkeypatch.setattr(competitive_mod, "complete_json", lambda *a, **k: payload)
        analysis = generate_competitive(_Company(), _History(), _valuation())
        assert analysis.generated is True
        # Strength and direction are scored SEPARATELY.
        assert analysis.moat_strength == "Wide"
        assert analysis.moat_direction == "Narrowing"
        assert analysis.dimensions[0].strength == "Strong"
        assert analysis.management_vs_us[0].source == "FY2026 10-K MD&A"

    def test_management_row_without_a_real_statement_is_dropped(self, monkeypatch):
        # A row with no sourced statement must not be fabricated into the table.
        payload = {
            "moat_strength": "Narrow", "moat_direction": "Stable", "moat_summary": "ok",
            "management_vs_us": [{"topic": "Growth", "our_model": "8%", "difference": "x"}],
        }
        monkeypatch.setattr(competitive_mod, "complete_json", lambda *a, **k: payload)
        analysis = generate_competitive(_Company(), _History(), _valuation())
        assert analysis.management_vs_us == []

    def test_garbage_falls_back(self, monkeypatch):
        monkeypatch.setattr(competitive_mod, "complete_json", lambda *a, **k: 123)
        assert generate_competitive(_Company(), _History(), _valuation()).generated is False
