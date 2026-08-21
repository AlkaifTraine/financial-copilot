"""
Analyst assumption overrides: a human pins the value drivers and the model computes
from them. Tested at the derive_inputs level (no network, use_model=False) plus the
file loader.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fincopilot.fundamentals.models import FinancialHistory, FiscalYear
from fincopilot.valuation.assumptions import derive_inputs
from fincopilot.valuation.models import SOURCE_ANALYST, AssumptionLedger


def _history() -> FinancialHistory:
    years = []
    revenue = 1000.0
    for i in range(4):
        revenue = revenue * 1.30 if i else revenue
        years.append(FiscalYear(
            fiscal_year=2022 + i, period_end=f"{2022 + i}-12-31",
            revenue=revenue, operating_income=revenue * 0.40,
        ))
    return FinancialHistory(ticker="TEST", company_name="Test Co", years=years)


_COMPANY = SimpleNamespace(ticker="TEST", name="Test Co", slug="test", country="US")


def _derive(overrides):
    ledger = AssumptionLedger()
    inputs = derive_inputs(_history(), _COMPANY, ledger, tax_rate=0.21,
                           use_model=False, overrides=overrides)
    return ledger, inputs


class TestDriverOverrides:
    def test_override_pins_margin_and_growth(self):
        ledger, inputs = _derive({
            "terminal_operating_margin": 0.50, "year_one_revenue_growth": 0.35,
        })
        margin = ledger.get("terminal_margin")
        growth = ledger.get("year_one_growth")
        assert margin.source == SOURCE_ANALYST and margin.value == pytest.approx(0.50)
        assert growth.source == SOURCE_ANALYST and growth.value == pytest.approx(0.35)
        # The forecast paths compute from the analyst's numbers.
        assert inputs.margin_path[-1] == pytest.approx(0.50)
        assert inputs.growth_path[0] == pytest.approx(0.35)

    def test_override_is_clamped_to_hard_sanity_rail(self):
        ledger, _ = _derive({"terminal_operating_margin": 2.0})   # 200% is impossible
        margin = ledger.get("terminal_margin")
        assert margin.value == pytest.approx(0.95)                 # hard ceiling
        assert margin.clamped is True and margin.raw_value == pytest.approx(2.0)

    def test_partial_override_leaves_others_to_the_model(self):
        ledger, _ = _derive({"terminal_growth_rate": 0.03})
        assert ledger.get("terminal_growth").source == SOURCE_ANALYST
        # margin/growth were not pinned, so they are not analyst-set
        assert ledger.get("terminal_margin").source != SOURCE_ANALYST

    def test_no_overrides_is_a_pure_model_run(self):
        ledger, _ = _derive(None)
        assert all(a.source != SOURCE_ANALYST for a in ledger.items)


class TestLoadOverrides:
    def test_reads_recognised_keys_only(self, tmp_path, monkeypatch):
        import json
        from fincopilot import config
        from fincopilot.valuation import overrides as ov
        monkeypatch.setattr(config, "OVERRIDES_DIR", tmp_path)
        (tmp_path / "nvda.json").write_text(json.dumps({
            "terminal_operating_margin": 0.45, "wacc": 0.12, "garbage": "x",
        }))
        loaded = ov.load_overrides("nvda")
        assert loaded == {"terminal_operating_margin": 0.45, "wacc": 0.12}

    def test_missing_file_is_empty(self, tmp_path, monkeypatch):
        from fincopilot import config
        from fincopilot.valuation import overrides as ov
        monkeypatch.setattr(config, "OVERRIDES_DIR", tmp_path)
        assert ov.load_overrides("absent") == {}
