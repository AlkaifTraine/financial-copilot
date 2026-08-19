"""
Unit tests for the discounted cash flow arithmetic.

The expected values here are computed by hand in the comments, not copied from
a previous run of the code. A test that asserts the implementation still does
what it did last time will happily lock in a bug; these assert that it does
what a DCF is supposed to do.
"""

from __future__ import annotations

import math

import pytest

from fincopilot.valuation.dcf import decay_path, fade, run_dcf


def _simple_model(**overrides):
    """A deliberately round-numbered model that can be checked on paper."""
    params = dict(
        base_revenue=1000.0,
        base_year=2025,
        growth_path=[0.10, 0.10],
        margin_path=[0.20, 0.20],
        tax_rate=0.25,
        depreciation_pct=0.05,
        capex_pct=0.05,
        working_capital_pct=0.0,
        wacc=0.10,
        terminal_growth=0.02,
        net_debt=0.0,
        shares_outstanding=100.0,
    )
    params.update(overrides)
    return run_dcf(**params)


class TestForecastMechanics:
    def test_revenue_compounds(self):
        result = _simple_model()
        # 1000 -> 1100 -> 1210
        assert result.forecast[0].revenue == pytest.approx(1100.0)
        assert result.forecast[1].revenue == pytest.approx(1210.0)

    def test_free_cash_flow_build_up(self):
        result = _simple_model()
        year = result.forecast[0]
        # EBIT   = 1100 * 0.20            = 220
        # NOPAT  = 220 * (1 - 0.25)       = 165
        # D&A    = 1100 * 0.05            =  55
        # capex  = 1100 * 0.05            =  55
        # dNWC   = 0
        # FCF    = 165 + 55 - 55 - 0      = 165
        assert year.ebit == pytest.approx(220.0)
        assert year.nopat == pytest.approx(165.0)
        assert year.free_cash_flow == pytest.approx(165.0)

    def test_working_capital_consumes_cash(self):
        result = _simple_model(working_capital_pct=0.10)
        # Revenue rises by 100, so 10 of cash is absorbed: 165 - 10 = 155.
        assert result.forecast[0].free_cash_flow == pytest.approx(155.0)

    def test_discounting(self):
        result = _simple_model()
        assert result.forecast[0].discount_factor == pytest.approx(1 / 1.10)
        assert result.forecast[1].discount_factor == pytest.approx(1 / 1.21)
        # PV of year 1 = 165 / 1.1 = 150
        assert result.forecast[0].present_value == pytest.approx(150.0)


class TestTerminalValue:
    def test_gordon_growth(self):
        result = _simple_model()
        # Year 2 FCF: revenue 1210 -> EBIT 242 -> NOPAT 181.5, D&A = capex, so
        # FCF = 181.5.  TV = 181.5 * 1.02 / (0.10 - 0.02) = 2314.125
        assert result.forecast[1].free_cash_flow == pytest.approx(181.5)
        assert result.terminal_value == pytest.approx(2314.125)
        # PV(TV) = 2314.125 / 1.21
        assert result.pv_terminal == pytest.approx(2314.125 / 1.21)

    def test_enterprise_value_is_sum_of_parts(self):
        result = _simple_model()
        assert result.enterprise_value == pytest.approx(
            result.pv_forecast + result.pv_terminal
        )

    def test_terminal_growth_at_or_above_wacc_is_rejected(self):
        # The Gordon formula divides by (wacc - g); at g >= wacc it is either a
        # division by zero or a negative value presented as a valuation.
        with pytest.raises(ValueError, match="terminal growth"):
            _simple_model(terminal_growth=0.10)
        with pytest.raises(ValueError, match="terminal growth"):
            _simple_model(terminal_growth=0.12)

    def test_higher_wacc_lowers_value(self):
        assert (
            _simple_model(wacc=0.12).fair_value_per_share
            < _simple_model(wacc=0.10).fair_value_per_share
        )

    def test_higher_terminal_growth_raises_value(self):
        assert (
            _simple_model(terminal_growth=0.03).fair_value_per_share
            > _simple_model(terminal_growth=0.02).fair_value_per_share
        )


class TestEquityBridge:
    def test_net_debt_reduces_equity_value(self):
        without = _simple_model(net_debt=0.0)
        with_debt = _simple_model(net_debt=500.0)
        assert with_debt.equity_value == pytest.approx(without.equity_value - 500.0)

    def test_net_cash_increases_equity_value(self):
        # Net debt is negative when a company holds more cash than debt.
        result = _simple_model(net_debt=-200.0)
        assert result.equity_value == pytest.approx(result.enterprise_value + 200.0)

    def test_per_share_value(self):
        result = _simple_model(shares_outstanding=100.0)
        assert result.fair_value_per_share == pytest.approx(result.equity_value / 100.0)

    def test_terminal_value_share_is_a_fraction(self):
        result = _simple_model()
        assert 0.0 < result.terminal_value_share < 1.0
        assert result.terminal_value_share == pytest.approx(
            result.pv_terminal / result.enterprise_value
        )


class TestInputValidation:
    def test_mismatched_paths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            _simple_model(growth_path=[0.1, 0.1, 0.1], margin_path=[0.2, 0.2])

    def test_empty_forecast_rejected(self):
        with pytest.raises(ValueError, match="at least one year"):
            _simple_model(growth_path=[], margin_path=[])

    def test_non_positive_revenue_rejected(self):
        with pytest.raises(ValueError, match="base revenue"):
            _simple_model(base_revenue=0.0)


class TestFade:
    def test_endpoints_are_exact(self):
        path = fade(0.60, 0.02, 5)
        assert path[0] == pytest.approx(0.60)
        assert path[-1] == pytest.approx(0.02)

    def test_monotonic_decline(self):
        path = fade(0.60, 0.02, 5)
        assert all(earlier > later for earlier, later in zip(path, path[1:]))

    def test_evenly_spaced(self):
        path = fade(0.0, 1.0, 5)
        assert path == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_single_period_lands_on_target(self):
        assert fade(0.60, 0.02, 1) == pytest.approx([0.02])

    def test_zero_periods(self):
        assert fade(0.6, 0.02, 0) == []


class TestRealism:
    """Guards against the failure mode that makes DCFs embarrassing."""

    def test_hypergrowth_fades_rather_than_compounding_forever(self):
        # A company growing 65% is not worth what flat 65% growth implies.
        faded = _simple_model(
            growth_path=fade(0.65, 0.025, 5),
            margin_path=[0.60] * 5,
        )
        flat = _simple_model(
            growth_path=[0.65] * 5,
            margin_path=[0.60] * 5,
        )
        assert faded.fair_value_per_share < flat.fair_value_per_share / 2

    def test_all_values_finite(self):
        result = _simple_model(growth_path=fade(0.65, 0.025, 5), margin_path=[0.6] * 5)
        assert math.isfinite(result.fair_value_per_share)
        assert math.isfinite(result.enterprise_value)
        assert all(math.isfinite(y.free_cash_flow) for y in result.forecast)


class TestDecayPath:
    """The growth path shape, which a linear fade got badly wrong."""

    def test_starts_at_start_and_approaches_end(self):
        path = decay_path(0.65, 0.025, 10)
        assert path[0] == pytest.approx(0.65)
        # At the default decay the path fades close to — but not exactly onto —
        # the terminal rate within ten years, stepping down to the perpetuity in
        # the final year. It must be far below the start and near the end.
        assert path[-1] == pytest.approx(0.025, abs=0.03)
        assert path[-1] < 0.10

    def test_monotonic_decline(self):
        path = decay_path(0.65, 0.025, 10)
        assert all(a > b for a, b in zip(path, path[1:]))

    def test_decays_faster_than_linear_early(self):
        decayed = decay_path(0.65, 0.025, 10)
        linear = fade(0.65, 0.025, 10)
        assert decayed[2] < linear[2]

    def test_cumulative_growth_stays_defensible(self):
        # A linear fade from 65% over ten years compounds to ~17x, forecasting
        # NVIDIA to $3.6tn of revenue. Geometric decay must stay far below that:
        # at the default decay a 65% starter reaches ~7x, long enough to credit a
        # real compounder but nowhere near the linear absurdity.
        def cumulative(path):
            total = 1.0
            for g in path:
                total *= 1 + g
            return total

        assert cumulative(fade(0.65, 0.025, 10)) > 12
        assert cumulative(decay_path(0.65, 0.025, 10)) < 8

    def test_zero_periods(self):
        assert decay_path(0.6, 0.02, 0) == []
