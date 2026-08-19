"""
Unit tests for the assumption-critique agent's response handling.

The agent's reasoning is the model's; what must be pinned is that its output is
turned into a safe revision — decimals coerced, percentages caught, a "no change"
verdict respected. The percentage-vs-decimal normalisation is here because a real
run had the reviewer answer 60 instead of 0.60, which the clamp masked by pinning
to a bound instead of surfacing the 100x error.
"""

from __future__ import annotations

import pytest

from fincopilot.valuation.agent import _normalise, _parse


class TestNormalise:
    def test_decimal_passes_through(self):
        assert _normalise(0.35) == pytest.approx(0.35)
        assert _normalise(0.025) == pytest.approx(0.025)

    def test_percentage_is_rescaled(self):
        assert _normalise(60) == pytest.approx(0.60)
        assert _normalise(48.0) == pytest.approx(0.48)
        assert _normalise(2.5) == pytest.approx(0.025)   # terminal growth as a percent

    def test_boundary_below_threshold_kept(self):
        # 1.2 (120% growth) is a plausible decimal and must not be rescaled.
        assert _normalise(1.2) == pytest.approx(1.2)

    def test_garbage_is_none(self):
        assert _normalise(None) is None
        assert _normalise("not a number") is None


class TestParse:
    def test_revise_false_returns_none(self):
        payload = {"revise": False, "year_one_revenue_growth": {"value": 0.2}}
        assert _parse(payload) is None

    def test_non_dict_returns_none(self):
        assert _parse("nope") is None
        assert _parse(None) is None

    def test_revision_is_parsed_and_normalised(self):
        payload = {
            "revise": True,
            "year_one_revenue_growth": {"value": 60, "rationale": "momentum"},   # percent slip
            "terminal_operating_margin": {"value": 0.48, "rationale": "moat"},
            "terminal_growth_rate": {"value": 2.5, "rationale": "gdp"},          # percent slip
        }
        revision = _parse(payload)
        assert revision["year_one_revenue_growth"]["value"] == pytest.approx(0.60)
        assert revision["terminal_operating_margin"]["value"] == pytest.approx(0.48)
        assert revision["terminal_growth_rate"]["value"] == pytest.approx(0.025)
        assert revision["year_one_revenue_growth"]["rationale"] == "momentum"

    def test_revise_true_but_no_usable_values_returns_none(self):
        payload = {"revise": True, "year_one_revenue_growth": {"rationale": "no value key"}}
        assert _parse(payload) is None
