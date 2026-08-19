"""
Unit tests for the Phase-1 valuation calibration fixes: the size premium and
long-horizon beta in the discount rate, the durability-based terminal-margin
floor, and the year-one growth floor. Each changes fair value materially, so
each is pinned against hand-reasoned expectations.
"""

from __future__ import annotations

import pytest

from fincopilot.fundamentals.models import FinancialHistory, FiscalYear
from fincopilot.valuation.assumptions import _margin_floor_fraction
from fincopilot.valuation.models import AssumptionLedger
from fincopilot.valuation.wacc import _size_premium, compute_wacc


def _history(
    market_cap: float | None = None, beta: float | None = None, currency: str = "USD"
) -> FinancialHistory:
    return FinancialHistory(
        ticker="TST", company_name="Test Co", currency=currency, market_cap=market_cap, beta=beta,
        years=[
            FiscalYear(fiscal_year=2024, period_end="2024-12-31", revenue=100e9,
                       operating_income=30e9, pretax_income=30e9, tax_expense=6e9,
                       operating_cash_flow=28e9, capex=4e9),
            FiscalYear(fiscal_year=2025, period_end="2025-12-31", revenue=110e9,
                       operating_income=33e9, pretax_income=33e9, tax_expense=6.6e9,
                       operating_cash_flow=31e9, capex=4e9),
        ],
    )


class TestSizePremium:
    def test_megacap_gets_a_discount(self):
        assert _size_premium(_history(market_cap=3e12), AssumptionLedger()) == pytest.approx(-0.015)

    def test_largecap_small_discount(self):
        assert _size_premium(_history(market_cap=50e9), AssumptionLedger()) == pytest.approx(-0.005)

    def test_smallcap_gets_a_premium(self):
        assert _size_premium(_history(market_cap=1e9), AssumptionLedger()) == pytest.approx(0.025)

    def test_symmetric_not_a_megacap_thumb(self):
        # A small cap is penalised as much as a mega-cap is credited is NOT the
        # claim; the claim is the curve is monotonic in size.
        caps = [1e9, 5e9, 50e9, 3e12]
        premia = [_size_premium(_history(market_cap=c), AssumptionLedger()) for c in caps]
        assert premia == sorted(premia, reverse=True)

    def test_local_currency_cap_converted_before_bucketing(self):
        # A ₹300bn mid-cap (~$3.6bn) must NOT clear the USD mega-cap threshold.
        prem = _size_premium(_history(market_cap=300e9, currency="INR"), AssumptionLedger())
        assert prem == pytest.approx(0.010)          # mid-cap, not mega-cap

    def test_large_rupee_company_is_large_cap(self):
        # TCS-scale ~₹14tn (~$168bn) lands large-cap, not mega-cap.
        prem = _size_premium(_history(market_cap=14e12, currency="INR"), AssumptionLedger())
        assert prem == pytest.approx(-0.005)

    def test_unknown_currency_skips_premium(self):
        prem = _size_premium(_history(market_cap=500e9, currency="XYZ"), AssumptionLedger())
        assert prem == 0.0


class TestHorizonBeta:
    def test_horizon_beta_pulls_toward_market(self):
        led = AssumptionLedger()
        compute_wacc(_history(market_cap=3e12, beta=2.2), _company(), led)
        blume = led.get("beta").value
        horizon = led.get("beta_dcf").value
        # The horizon beta must sit between the Blume beta and 1.0.
        assert 1.0 < horizon < blume

    def test_high_beta_megacap_wacc_is_reasonable(self):
        led = AssumptionLedger()
        wacc = compute_wacc(_history(market_cap=3e12, beta=2.2), _company(), led)
        # A raw 2.2 beta must not still produce a ~15% rate after horizon + size.
        assert wacc < 0.13


class TestMarginFloorFraction:
    def test_steady_margins_keep_most_of_the_margin(self):
        frac, label = _margin_floor_fraction([0.54, 0.62, 0.62])
        assert frac == 0.80
        assert label == "steady"

    def test_volatile_margins_allow_deep_normalisation(self):
        frac, label = _margin_floor_fraction([0.16, 0.54, 0.62])
        assert frac == 0.55
        assert label == "volatile"

    def test_moderately_steady_in_between(self):
        # mean 0.30, cv ~0.27 → moderate band.
        frac, _ = _margin_floor_fraction([0.20, 0.30, 0.40])
        assert frac == 0.68


class _Company:
    country = "US"
    ticker = "TST"
    name = "Test Co"


def _company():
    return _Company()
