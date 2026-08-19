"""
Unit tests for the reverse DCF — the "what is priced in" solver.

The property that matters is a round trip: when the target price is set to the
DCF's own fair value, every driver the price implies must come back to our base
case, because nothing has been asked to change. If a solve does not reproduce
the base value there, the two columns of the table are not comparable and the
whole exhibit is misleading. These tests pin that, plus the CAGR arithmetic and
the unreachable path, against hand-reasoned expectations rather than a prior run.
"""

from __future__ import annotations

import math

import pytest

from fincopilot.valuation.assumptions import ForecastInputs
from fincopilot.valuation.dcf import decay_path, fade, run_dcf
from fincopilot.valuation.reverse import _cagr, build_priced_in, implied_growth

HORIZON = 10
WACC = 0.10
NET_DEBT = 0.0
SHARES = 100.0
TERMINAL_GROWTH = 0.025
TERMINAL_MARGIN = 0.25


def _inputs() -> ForecastInputs:
    return ForecastInputs(
        base_revenue=1000.0,
        base_year=2025,
        growth_path=decay_path(0.20, TERMINAL_GROWTH, HORIZON, 0.70),
        margin_path=fade(0.30, TERMINAL_MARGIN, HORIZON),
        tax_rate=0.25,
        depreciation_pct=0.05,
        capex_pct=0.05,
        working_capital_pct=0.05,
        terminal_growth=TERMINAL_GROWTH,
    )


def _base_dcf(inputs: ForecastInputs):
    return run_dcf(
        base_revenue=inputs.base_revenue,
        base_year=inputs.base_year,
        growth_path=inputs.growth_path,
        margin_path=inputs.margin_path,
        tax_rate=inputs.tax_rate,
        depreciation_pct=inputs.depreciation_pct,
        capex_pct=inputs.capex_pct,
        working_capital_pct=inputs.working_capital_pct,
        wacc=WACC,
        terminal_growth=inputs.terminal_growth,
        net_debt=NET_DEBT,
        shares_outstanding=SHARES,
    )


def _row(comparison, key):
    return next(r for r in comparison.rows if r.key == key)


class TestCagr:
    def test_constant_path(self):
        # A flat 10% for any number of years compounds to a 10% CAGR.
        assert _cagr([0.10] * 5) == pytest.approx(0.10)

    def test_matches_endpoint_ratio(self):
        path = [0.40, 0.20, 0.10]
        # (1.4 * 1.2 * 1.1) ^ (1/3) - 1
        expected = (1.4 * 1.2 * 1.1) ** (1 / 3) - 1
        assert _cagr(path) == pytest.approx(expected)

    def test_empty_is_zero(self):
        assert _cagr([]) == 0.0


class TestRoundTrip:
    """At a price equal to fair value, every implied driver returns to base."""

    def setup_method(self):
        self.inputs = _inputs()
        self.dcf = _base_dcf(self.inputs)
        self.price = self.dcf.fair_value_per_share
        self.y1 = implied_growth(
            base_revenue=self.inputs.base_revenue,
            base_year=self.inputs.base_year,
            horizon=HORIZON,
            terminal_margin=TERMINAL_MARGIN,
            current_margin=self.inputs.margin_path[0],
            tax_rate=self.inputs.tax_rate,
            depreciation_pct=self.inputs.depreciation_pct,
            capex_pct=self.inputs.capex_pct,
            working_capital_pct=self.inputs.working_capital_pct,
            wacc=WACC,
            terminal_growth=self.inputs.terminal_growth,
            net_debt=NET_DEBT,
            shares_outstanding=SHARES,
            target_price=self.price,
        )
        self.pi = build_priced_in(
            inputs=self.inputs,
            wacc=WACC,
            net_debt=NET_DEBT,
            shares_outstanding=SHARES,
            terminal_margin=TERMINAL_MARGIN,
            base_dcf=self.dcf,
            share_price=self.price,
            currency="USD",
            implied_year_one_growth=self.y1,
        )

    def test_all_drivers_present(self):
        keys = {r.key for r in self.pi.rows}
        assert keys == {"revenue_cagr", "operating_margin", "fcf_margin", "terminal_growth"}

    def test_growth_returns_to_base(self):
        # implied_growth should recover the base year-one growth (~20%), and so
        # the implied CAGR should equal our base CAGR.
        assert self.y1 == pytest.approx(0.20, abs=1e-3)
        row = _row(self.pi, "revenue_cagr")
        assert row.implied_value == pytest.approx(row.base_value, abs=1e-3)

    def test_operating_margin_returns_to_base(self):
        row = _row(self.pi, "operating_margin")
        assert row.base_value == pytest.approx(TERMINAL_MARGIN)
        assert row.implied_value == pytest.approx(TERMINAL_MARGIN, abs=1e-3)

    def test_terminal_growth_returns_to_base(self):
        row = _row(self.pi, "terminal_growth")
        assert row.implied_value == pytest.approx(TERMINAL_GROWTH, abs=1e-3)

    def test_fcf_margin_returns_to_base(self):
        row = _row(self.pi, "fcf_margin")
        # Base FCF margin is the final forecast year's FCF over its revenue.
        final = self.dcf.forecast[-1]
        assert row.base_value == pytest.approx(final.free_cash_flow / final.revenue)
        assert row.implied_value == pytest.approx(row.base_value, abs=1e-3)


class TestPricedInDirection:
    """A price above fair value must imply richer assumptions than our base."""

    def test_higher_price_implies_more(self):
        inputs = _inputs()
        dcf = _base_dcf(inputs)
        price = dcf.fair_value_per_share * 1.08   # market pays a modest premium
        y1 = implied_growth(
            base_revenue=inputs.base_revenue,
            base_year=inputs.base_year,
            horizon=HORIZON,
            terminal_margin=TERMINAL_MARGIN,
            current_margin=inputs.margin_path[0],
            tax_rate=inputs.tax_rate,
            depreciation_pct=inputs.depreciation_pct,
            capex_pct=inputs.capex_pct,
            working_capital_pct=inputs.working_capital_pct,
            wacc=WACC,
            terminal_growth=inputs.terminal_growth,
            net_debt=NET_DEBT,
            shares_outstanding=SHARES,
            target_price=price,
        )
        pi = build_priced_in(
            inputs=inputs, wacc=WACC, net_debt=NET_DEBT, shares_outstanding=SHARES,
            terminal_margin=TERMINAL_MARGIN, base_dcf=dcf, share_price=price,
            currency="USD", implied_year_one_growth=y1,
        )
        # Every lever that can reach the price must do so from above our base —
        # a premium cannot be justified by assuming less than we already do.
        for row in pi.rows:
            if row.reachable:
                assert row.implied_value > row.base_value, row.key
        # At a modest premium the margin lever is reachable; confirm it moved up.
        assert _row(pi, "operating_margin").implied_value > TERMINAL_MARGIN

    def test_extreme_price_is_unreachable_on_weak_levers(self):
        # A price far above fair value cannot be reached by capping levers: an
        # operating margin cannot exceed 100%, capex cannot fall below zero, and
        # terminal growth cannot reach the discount rate. Those rows report None.
        inputs = _inputs()
        dcf = _base_dcf(inputs)
        price = dcf.fair_value_per_share * 5.0
        pi = build_priced_in(
            inputs=inputs, wacc=WACC, net_debt=NET_DEBT, shares_outstanding=SHARES,
            terminal_margin=TERMINAL_MARGIN, base_dcf=dcf, share_price=price,
            currency="USD", implied_year_one_growth=None,
        )
        margin = _row(pi, "operating_margin")
        assert margin.implied_value is None
        assert not margin.reachable
        assert margin.note                       # a caveat is always given when unreachable

    def test_returns_none_without_price(self):
        inputs = _inputs()
        dcf = _base_dcf(inputs)
        assert build_priced_in(
            inputs=inputs, wacc=WACC, net_debt=NET_DEBT, shares_outstanding=SHARES,
            terminal_margin=TERMINAL_MARGIN, base_dcf=dcf, share_price=0.0,
            currency="USD", implied_year_one_growth=None,
        ) is None
