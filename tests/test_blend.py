"""
Unit tests for the valuation triangulation (blend).

These assert the properties that make a blend trustworthy: it is a genuine
weighted average, coverage buys the consensus more weight (up to a cap), a lone
outlier is thrown out when there are enough sources to judge one, and the whole
thing degrades gracefully to the DCF when there is no analyst coverage.
"""

from __future__ import annotations

import pytest

from fincopilot import config
from fincopilot.fundamentals.models import FinancialHistory
from fincopilot.valuation.blend import (
    _analyst_weight_multiple,
    blend_estimates,
    build_analyst_estimate,
    build_blend,
)
from fincopilot.valuation.models import ValuationEstimate


def _history(**kwargs) -> FinancialHistory:
    h = FinancialHistory(ticker="TEST", company_name="Test Co", currency="USD")
    for key, value in kwargs.items():
        setattr(h, key, value)
    return h


def _estimate(key, value, weight, source_type="model") -> ValuationEstimate:
    return ValuationEstimate(
        key=key, label=key, source_type=source_type,
        value_per_share=value, weight=weight, currency="USD",
    )


class TestAnalystWeight:
    def test_scales_with_coverage(self):
        few = _analyst_weight_multiple(3)
        many = _analyst_weight_multiple(config.BLEND_ANALYST_REFERENCE_COUNT * 2)
        assert many > few

    def test_capped(self):
        assert _analyst_weight_multiple(10_000) == pytest.approx(
            config.BLEND_ANALYST_MAX_WEIGHT_MULTIPLE
        )

    def test_floor_of_one_unit(self):
        assert _analyst_weight_multiple(None) == 1.0
        assert _analyst_weight_multiple(0) == 1.0
        assert _analyst_weight_multiple(1) == 1.0


class TestAnalystEstimate:
    def test_prefers_median_over_mean(self):
        h = _history(analyst_target_median=200.0, analyst_target_mean=180.0,
                     analyst_opinion_count=40)
        est = build_analyst_estimate(h, "TEST", "USD")
        assert est.value_per_share == pytest.approx(200.0)
        assert est.source_type == "analyst"
        assert est.url and "TEST" in est.url

    def test_falls_back_to_mean(self):
        h = _history(analyst_target_mean=180.0, analyst_opinion_count=5)
        est = build_analyst_estimate(h, "TEST", "USD")
        assert est.value_per_share == pytest.approx(180.0)

    def test_none_without_targets(self):
        assert build_analyst_estimate(_history(), "TEST", "USD") is None


class TestWeightedAverage:
    def test_two_source_blend_is_weighted(self):
        # DCF 50 at weight 1, analyst 200 at weight 3 -> (50 + 600)/4 = 162.5
        estimates = [_estimate("dcf", 50.0, 1.0), _estimate("analyst", 200.0, 3.0, "analyst")]
        result = blend_estimates(estimates, share_price=150.0, currency="USD")
        assert result.blended_value == pytest.approx(162.5)

    def test_range_spans_included_values(self):
        estimates = [_estimate("dcf", 50.0, 1.0), _estimate("analyst", 200.0, 3.0, "analyst")]
        result = blend_estimates(estimates, share_price=None, currency="USD")
        assert (result.low, result.high) == pytest.approx((50.0, 200.0))

    def test_equal_weights_reduce_to_mean(self):
        estimates = [_estimate("a", 100.0, 1.0), _estimate("b", 200.0, 1.0)]
        result = blend_estimates(estimates, share_price=None, currency="USD")
        assert result.blended_value == pytest.approx(150.0)


class TestOutlierRejection:
    def test_two_sources_never_rejected(self):
        # Even a huge gap is kept with only two sources — nothing to judge against.
        estimates = [_estimate("dcf", 10.0, 1.0), _estimate("analyst", 500.0, 1.0, "analyst")]
        result = blend_estimates(estimates, share_price=None, currency="USD")
        assert len(result.included) == 2

    def test_lone_outlier_dropped_with_three_sources(self):
        # Median of {90,100,1000} is 100; 1000 > 2.5x median -> excluded.
        estimates = [
            _estimate("a", 90.0, 1.0),
            _estimate("b", 100.0, 1.0),
            _estimate("c", 1000.0, 1.0),
        ]
        result = blend_estimates(estimates, share_price=None, currency="USD")
        excluded = result.excluded
        assert len(excluded) == 1 and excluded[0].key == "c"
        # Blend is the average of the two survivors, not dragged up by the outlier.
        assert result.blended_value == pytest.approx(95.0)

    def test_rejection_reverts_if_too_few_survive(self):
        # If rejection would leave fewer than two, keep everything instead.
        estimates = [
            _estimate("a", 100.0, 1.0),
            _estimate("b", 1000.0, 1.0),
            _estimate("c", 5000.0, 1.0),
        ]
        result = blend_estimates(estimates, share_price=None, currency="USD")
        assert len(result.included) == 3


class TestRatingAndUpside:
    def test_upside_and_rating_from_blend(self):
        estimates = [_estimate("dcf", 100.0, 1.0), _estimate("analyst", 100.0, 1.0, "analyst")]
        result = blend_estimates(estimates, share_price=50.0, currency="USD")
        assert result.upside == pytest.approx(1.0)          # 100 vs 50
        assert result.rating == "BUY"

    def test_sell_when_below_price(self):
        estimates = [_estimate("dcf", 40.0, 1.0), _estimate("analyst", 40.0, 1.0, "analyst")]
        result = blend_estimates(estimates, share_price=100.0, currency="USD")
        assert result.rating == "SELL"

    def test_not_rated_without_price(self):
        estimates = [_estimate("dcf", 40.0, 1.0)]
        result = blend_estimates(estimates, share_price=None, currency="USD")
        assert result.rating == "NOT RATED"


class TestBuildBlend:
    def test_consensus_is_an_external_reference_not_blended(self):
        # The analyst consensus is shown but NOT folded into our target: our
        # number stays intrinsic (the DCF here), with the consensus recorded as an
        # excluded external reference.
        h = _history(analyst_target_median=200.0, analyst_opinion_count=40,
                     share_price=150.0)
        result = build_blend(dcf_value=50.0, history=h, ticker="TEST",
                             share_price=150.0, currency="USD")
        assert {e.key for e in result.estimates} == {"dcf", "analyst_consensus"}
        assert result.blended_value == pytest.approx(50.0)
        analyst = next(e for e in result.estimates if e.key == "analyst_consensus")
        assert analyst.included is False
        assert analyst.weight == 0.0
        assert "external market reference" in analyst.exclude_reason.lower()

    def test_scenario_value_is_a_cross_check_not_blended(self):
        # The scenario expected value is built FROM the DCF (base = DCF), so it is a
        # cross-check, NOT blended — the target stays the intrinsic DCF (no double count).
        h = _history(share_price=150.0)
        result = build_blend(dcf_value=50.0, history=h, ticker="TEST",
                             share_price=150.0, currency="USD", scenario_value=70.0)
        assert {e.key for e in result.estimates} == {"dcf", "scenario"}
        assert result.blended_value == pytest.approx(50.0)          # target = intrinsic DCF
        scenario = next(e for e in result.estimates if e.key == "scenario")
        assert scenario.included is False and scenario.weight == 0.0
        assert "double-count" in scenario.exclude_reason.lower()

    def test_degrades_to_dcf_without_coverage(self):
        h = _history(share_price=150.0)  # no analyst targets
        result = build_blend(dcf_value=50.0, history=h, ticker="TEST",
                             share_price=150.0, currency="USD")
        assert len(result.estimates) == 1
        assert result.blended_value == pytest.approx(50.0)

    def test_empty_when_nothing_to_blend(self):
        result = build_blend(dcf_value=None, history=_history(), ticker="TEST",
                             share_price=None, currency="USD")
        assert result.blended_value == 0.0
        assert result.estimates == []

    def test_comps_shown_as_cross_check_not_blended(self):
        h = _history(analyst_target_median=200.0, analyst_opinion_count=40)
        result = build_blend(dcf_value=50.0, history=h, ticker="TEST",
                             share_price=None, currency="USD", comps_value=120.0)
        assert {e.key for e in result.estimates} == {"dcf", "analyst_consensus", "comps"}
        # Comps is a market-based cross-check: present, but not blended into the target.
        comps = next(e for e in result.estimates if e.key == "comps")
        assert comps.included is False and comps.weight == 0.0
        assert result.blended_value == pytest.approx(50.0)          # target = intrinsic DCF


class TestDeterminism:
    def test_identical_inputs_reproduce(self):
        h = _history(analyst_target_median=200.0, analyst_opinion_count=40, share_price=150.0)
        a = build_blend(dcf_value=50.0, history=h, ticker="TEST", share_price=150.0, currency="USD")
        b = build_blend(dcf_value=50.0, history=h, ticker="TEST", share_price=150.0, currency="USD")
        assert a.blended_value == b.blended_value
