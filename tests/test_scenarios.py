"""
Unit tests for the bear / base / bull scenario engine.

These assert the properties a scenario set must have to be defensible, not that
the code still returns what it returned last time: the three cases must be
correctly ordered, the base case must reproduce the headline DCF exactly, the
expected value must be a genuine probability weighting, and the spread must
widen with the company's own historical volatility.
"""

from __future__ import annotations

import pytest

from fincopilot import config
from fincopilot.fundamentals.models import FinancialHistory, FiscalYear
from fincopilot.valuation.assumptions import ForecastInputs
from fincopilot.valuation.dcf import decay_path, fade, run_dcf
from fincopilot.valuation.scenarios import build_scenarios

HORIZON = config.DCF_FORECAST_YEARS


def _history(growths: list[float], margins: list[float]) -> FinancialHistory:
    """Build a history with the given YoY revenue growths and operating margins.

    ``growths`` are year-over-year rates; ``margins`` are per-year operating
    margins. The two lists describe the same set of years.
    """
    assert len(growths) == len(margins)
    years: list[FiscalYear] = []
    revenue = 1000.0
    for i, (g, m) in enumerate(zip(growths, margins)):
        revenue = revenue * (1 + g) if i else revenue
        years.append(
            FiscalYear(
                fiscal_year=2015 + i,
                period_end=f"{2015 + i}-12-31",
                revenue=revenue,
                operating_income=revenue * m,
            )
        )
    return FinancialHistory(ticker="TEST", company_name="Test Co", years=years)


def _inputs(year_one_growth: float = 0.20, terminal_margin: float = 0.25) -> ForecastInputs:
    """Forecast inputs shaped exactly as ``derive_inputs`` would produce them."""
    terminal_growth = 0.025
    current_margin = 0.25
    return ForecastInputs(
        base_revenue=1000.0,
        base_year=2024,
        growth_path=decay_path(year_one_growth, terminal_growth, HORIZON, config.DCF_GROWTH_DECAY),
        margin_path=fade(current_margin, terminal_margin, HORIZON),
        tax_rate=0.21,
        depreciation_pct=0.03,
        capex_pct=0.04,
        working_capital_pct=0.05,
        terminal_growth=terminal_growth,
    )


def _build(history=None, inputs=None, base_wacc=0.10, base_terminal_margin=0.25,
           share_price=None):
    return build_scenarios(
        inputs=inputs or _inputs(),
        history=history or _history([0.30, 0.25, 0.20, 0.18], [0.24, 0.25, 0.26, 0.25]),
        base_wacc=base_wacc,
        base_terminal_margin=base_terminal_margin,
        net_debt=0.0,
        shares_outstanding=100.0,
        share_price=share_price,
        currency="USD",
    )


class TestShape:
    def test_three_cases_in_order(self):
        analysis = _build()
        assert [c.key for c in analysis.cases] == ["bear", "base", "bull"]

    def test_values_are_monotonic(self):
        # The whole point: a coherent downside is worth less and a coherent
        # upside more, because every driver moves the same way.
        analysis = _build()
        assert (
            analysis.bear.fair_value_per_share
            < analysis.base.fair_value_per_share
            < analysis.bull.fair_value_per_share
        )

    def test_all_values_positive_and_finite(self):
        analysis = _build()
        for case in analysis.cases:
            assert case.fair_value_per_share > 0
            assert case.enterprise_value > 0


class TestBaseCaseReproducesHeadlineDCF:
    def test_base_equals_standalone_dcf(self):
        inputs = _inputs()
        standalone = run_dcf(
            base_revenue=inputs.base_revenue,
            base_year=inputs.base_year,
            growth_path=inputs.growth_path,
            margin_path=inputs.margin_path,
            tax_rate=inputs.tax_rate,
            depreciation_pct=inputs.depreciation_pct,
            capex_pct=inputs.capex_pct,
            working_capital_pct=inputs.working_capital_pct,
            wacc=0.10,
            terminal_growth=inputs.terminal_growth,
            net_debt=0.0,
            shares_outstanding=100.0,
        )
        analysis = _build(inputs=inputs, base_wacc=0.10)
        assert analysis.base.fair_value_per_share == pytest.approx(
            standalone.fair_value_per_share
        )


class TestDrivers:
    def test_bear_raises_wacc_and_bull_lowers_it(self):
        analysis = _build(base_wacc=0.10)

        def wacc(case):
            return next(d.value for d in case.drivers if d.key == "wacc")

        assert wacc(analysis.bear) > wacc(analysis.base) == pytest.approx(0.10)
        assert wacc(analysis.bull) < wacc(analysis.base)

    def test_bear_growth_not_above_base_and_bull_not_below(self):
        analysis = _build()

        def g1(case):
            return next(d.value for d in case.drivers if d.key == "year_one_growth")

        assert g1(analysis.bear) <= g1(analysis.base) <= g1(analysis.bull)

    def test_terminal_growth_stays_within_bounds(self):
        analysis = _build()
        lo, hi = config.TERMINAL_GROWTH_BOUNDS
        for case in analysis.cases:
            tg = next(d.value for d in case.drivers if d.key == "terminal_growth")
            assert lo <= tg <= hi

    def test_driver_delta_display_signs(self):
        analysis = _build()
        bull_wacc = next(d for d in analysis.bull.drivers if d.key == "wacc")
        assert bull_wacc.delta_display.startswith("-")  # bull discounts at a lower rate


class TestExpectedValue:
    def test_expected_value_is_probability_weighted(self):
        analysis = _build()
        total_p = sum(c.probability for c in analysis.cases)
        expected = sum(
            c.probability * c.fair_value_per_share for c in analysis.cases
        ) / total_p
        assert analysis.expected_value == pytest.approx(expected)

    def test_expected_value_lies_within_the_range(self):
        analysis = _build()
        assert (
            analysis.bear.fair_value_per_share
            <= analysis.expected_value
            <= analysis.bull.fair_value_per_share
        )

    def test_upside_computed_against_price(self):
        analysis = _build(share_price=50.0)
        assert analysis.expected_upside == pytest.approx(
            analysis.expected_value / 50.0 - 1
        )
        for case in analysis.cases:
            assert case.upside == pytest.approx(case.fair_value_per_share / 50.0 - 1)

    def test_no_upside_without_a_price(self):
        analysis = _build(share_price=None)
        assert analysis.expected_upside is None
        assert all(c.upside is None for c in analysis.cases)


class TestRangeAndDispersion:
    def test_value_range_matches_min_and_max(self):
        analysis = _build()
        low, high = analysis.value_range
        assert low == pytest.approx(analysis.bear.fair_value_per_share)
        assert high == pytest.approx(analysis.bull.fair_value_per_share)

    def test_dispersion_is_positive(self):
        analysis = _build()
        assert analysis.dispersion > 0


class TestHistoricalDispersionSizesTheSpread:
    def _spread(self, analysis) -> float:
        """Bull-minus-bear year-one growth, the data-derived part of the band."""
        def g1(case):
            return next(d.value for d in case.drivers if d.key == "year_one_growth")
        return g1(analysis.bull) - g1(analysis.bear)

    def test_volatile_history_widens_the_band(self):
        steady = _build(
            history=_history([0.20, 0.20, 0.20, 0.20], [0.25, 0.25, 0.25, 0.25]),
            inputs=_inputs(year_one_growth=0.20),
        )
        volatile = _build(
            history=_history([0.05, 0.45, 0.10, 0.40], [0.25, 0.25, 0.25, 0.25]),
            inputs=_inputs(year_one_growth=0.20),
        )
        assert self._spread(volatile) > self._spread(steady)

    def test_spread_respects_config_floor(self):
        # A perfectly steady history should still produce a distinct band, no
        # narrower than the configured minimum on each side.
        steady = _build(
            history=_history([0.20, 0.20, 0.20, 0.20], [0.25, 0.25, 0.25, 0.25]),
            inputs=_inputs(year_one_growth=0.20),
        )
        assert self._spread(steady) >= config.SCENARIO_MIN_GROWTH_SPREAD


class TestDeterminism:
    def test_identical_inputs_reproduce_identical_values(self):
        a = _build(share_price=50.0)
        b = _build(share_price=50.0)
        assert [c.fair_value_per_share for c in a.cases] == [
            c.fair_value_per_share for c in b.cases
        ]
        assert a.expected_value == b.expected_value
