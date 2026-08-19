"""
Unit tests for the bottom-up segment forecast.

The LLM extraction is not exercised here — the pure forecasting arithmetic is.
Given a set of segment histories, the bottom-up total must equal the sum of the
segment forecasts, the reconciliation gap must be measured against reported
total revenue, and a segment with only one extracted year must fall back to the
top-down growth rate rather than inventing one.
"""

from __future__ import annotations

import pytest

from fincopilot.fundamentals.models import FinancialHistory, FiscalYear
from fincopilot.valuation.segments import (
    _SEG_MAX_GROWTH,
    forecast_segments,
)


def _history(total_revenue: float) -> FinancialHistory:
    return FinancialHistory(
        ticker="TST", company_name="Test Co", currency="USD",
        years=[FiscalYear(fiscal_year=2025, period_end="2025-01-31", revenue=total_revenue)],
    )


TOP_DOWN = [0.10] * 10   # a flat 10% top-down path → 10% CAGR


class TestForecastArithmetic:
    def setup_method(self):
        # Values in millions, as the model is now asked to return them.
        self.extracted = [
            {"name": "Data Center", "revenue_by_year": {"2024": 90000.0, "2025": 115000.0}},
            {"name": "Graphics", "revenue_by_year": {"2024": 48000.0, "2025": 50000.0}},
        ]
        # Segments sum to 165bn; report a total slightly above to create a gap.
        self.analysis = forecast_segments(
            self.extracted, history=_history(170e9), top_down_growth=TOP_DOWN, generated=True,
        )

    def test_two_segments_built(self):
        assert self.analysis is not None
        assert [s.name for s in self.analysis.segments] == ["Data Center", "Graphics"]

    def test_latest_sum_in_base_units(self):
        # 115 + 50 = 165 billion, in base units.
        assert self.analysis.latest_segment_sum == pytest.approx(165e9)

    def test_bottom_up_total_equals_sum_of_segments(self):
        total = sum(s.terminal_revenue for s in self.analysis.segments)
        assert self.analysis.bottom_up_terminal_revenue == pytest.approx(total)

    def test_reconciliation_gap_against_reported_total(self):
        # 165 / 170 - 1 = -2.94%
        assert self.analysis.reconciliation_gap == pytest.approx(165e9 / 170e9 - 1)
        assert self.analysis.reconciles is True

    def test_segment_year_one_from_own_history(self):
        dc = self.analysis.segments[0]
        assert dc.year_one_growth == pytest.approx(115.0 / 90.0 - 1)

    def test_top_down_cagr_recorded(self):
        assert self.analysis.top_down_cagr == pytest.approx(0.10, abs=1e-9)


class TestEdgeCases:
    def test_single_year_segment_falls_back_to_top_down(self):
        extracted = [{"name": "Only", "revenue_by_year": {"2025": 100000.0}}]
        analysis = forecast_segments(
            extracted, history=_history(100e9), top_down_growth=TOP_DOWN, generated=True,
        )
        assert analysis.segments[0].year_one_growth == pytest.approx(TOP_DOWN[0])

    def test_growth_is_clamped(self):
        # A segment that tripled must not be forecast to keep tripling.
        extracted = [{"name": "Rocket", "revenue_by_year": {"2024": 10000.0, "2025": 40000.0}}]
        analysis = forecast_segments(
            extracted, history=_history(40e9), top_down_growth=TOP_DOWN, generated=True,
        )
        assert analysis.segments[0].year_one_growth == pytest.approx(_SEG_MAX_GROWTH)

    def test_empty_extraction_returns_none(self):
        assert forecast_segments(
            [], history=_history(100e9), top_down_growth=TOP_DOWN, generated=True,
        ) is None

    def test_moderate_gap_shown_as_indicative(self):
        # A ~20% gap (an "all other"/corporate bucket) still renders, flagged.
        extracted = [{"name": "Most", "revenue_by_year": {"2024": 72000.0, "2025": 80000.0}}]
        analysis = forecast_segments(
            extracted, history=_history(100e9), top_down_growth=TOP_DOWN, generated=True,
        )
        assert analysis is not None
        assert analysis.reconciles is False
        assert "indicative" in analysis.note

    def test_hopeless_gap_is_suppressed(self):
        # Segments nowhere near the total → failed extraction → no exhibit at all.
        extracted = [{"name": "Half", "revenue_by_year": {"2024": 45000.0, "2025": 50000.0}}]
        assert forecast_segments(
            extracted, history=_history(100e9), top_down_growth=TOP_DOWN, generated=True,
        ) is None
