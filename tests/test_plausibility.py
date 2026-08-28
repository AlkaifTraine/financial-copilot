"""
Economic plausibility of a finished valuation.

The regression: a Bikaji report published a DCF fair value of INR 77.37 against
a market price of INR 618.80, and stated that the market must expect a **93.8%
mature operating margin**. The arithmetic was internally consistent — which is
why the existing integrity check passed it — and economically impossible.

These tests use company-relative thresholds throughout, because an absolute
"margin above 40%" rule has to be re-tuned for every sector, while "several
times the best this company has ever reported" means the same thing for a
snacks maker and a software company.
"""

from __future__ import annotations

import pytest

from fincopilot.fundamentals.models import FinancialHistory, FiscalYear
from fincopilot.valuation import plausibility


class _Row:
    def __init__(self, key, implied_value):
        self.key = key
        self.implied_value = implied_value


class _PricedIn:
    def __init__(self, rows):
        self.rows = rows


class _Forecast:
    def __init__(self, operating_margin):
        self.operating_margin = operating_margin


class _Dcf:
    def __init__(self, wacc=0.11, margins=None):
        self.wacc = wacc
        self.forecast = [_Forecast(m) for m in (margins or [0.14, 0.15, 0.16])]


class _Valuation:
    def __init__(self, *, price=618.80, fair_value=440.0,
                 implied_margin=None, implied_growth=None, wacc=0.11, margins=None):
        self.share_price = price
        self.fair_value = fair_value
        self.dcf_fair_value = fair_value
        self.dcf = _Dcf(wacc=wacc, margins=margins)
        rows = []
        if implied_margin is not None:
            rows.append(_Row("operating_margin", implied_margin))
        if implied_growth is not None:
            rows.append(_Row("terminal_growth", implied_growth))
        self.priced_in = _PricedIn(rows) if rows else None


def _history(best_margin=0.142, capex=-1_283.0, da=-900.0, recency=None):
    h = FinancialHistory(ticker="BIKAJI.NS", company_name="Bikaji",
                         currency="INR", source="nse_indas_xbrl")
    h.years = [
        FiscalYear(fiscal_year=2023, period_end="2023-03-31", revenue=19_660.0,
                   operating_income=19_660.0 * 0.103, net_income=1_285.0,
                   operating_cash_flow=1_761.0, capex=capex,
                   depreciation_amortisation=da),
        FiscalYear(fiscal_year=2024, period_end="2024-03-31", revenue=23_293.0,
                   operating_income=23_293.0 * best_margin, net_income=2_634.0,
                   operating_cash_flow=2_446.0, capex=capex,
                   depreciation_amortisation=da),
    ]
    h.recency = recency
    return h


class TestTheBikajiFailure:
    """The exact numbers that shipped."""

    def test_a_938_percent_implied_margin_is_critical(self):
        findings = plausibility.assess(
            _Valuation(price=618.80, fair_value=77.37, implied_margin=0.938),
            _history(),
        )
        criticals = [f for f in findings if f.severity == "CRITICAL"]
        assert criticals, "a 93.8% implied operating margin must block"
        assert any("93.8%" in f.message for f in criticals)

    def test_it_says_the_model_is_wrong_not_the_market(self):
        findings = plausibility.assess(
            _Valuation(price=618.80, fair_value=77.37, implied_margin=0.938),
            _history(),
        )
        text = " ".join(f.message for f in findings)
        assert "mis-specified" in text
        assert "must not be published" in text

    def test_the_87_percent_gap_also_blocks(self):
        findings = plausibility.assess(
            _Valuation(price=618.80, fair_value=77.37), _history()
        )
        assert any(f.severity == "CRITICAL" for f in findings)

    def test_it_names_actionable_causes(self):
        """A bare "implausible" tells nobody where to look."""
        from fincopilot.fundamentals import recency as rec
        stale = rec.assess(_history(), as_of="2026-08-28")
        findings = plausibility.assess(
            _Valuation(price=618.80, fair_value=77.37, implied_margin=0.938,
                       wacc=0.141, margins=[0.142, 0.142, 0.141]),
            _history(recency=stale, capex=-1_283.0, da=-600.0),
        )
        causes = " ".join(c for f in findings for c in f.likely_causes)
        assert "months old" in causes           # stale base year
        assert "discount rate" in causes        # 14.1% WACC
        assert "held flat" in causes            # no operating leverage
        assert "maintenance capex" in causes    # capex >> D&A


class TestCompanyRelativeThresholds:
    def test_a_high_margin_business_is_not_penalised_for_being_one(self):
        """40% implied on a company that already earns 38% is not absurd."""
        findings = plausibility.assess(
            _Valuation(fair_value=600.0, price=618.80, implied_margin=0.40),
            _history(best_margin=0.38),
        )
        assert not [f for f in findings if f.severity == "CRITICAL"]

    def test_the_same_margin_is_absurd_for_a_thin_margin_business(self):
        findings = plausibility.assess(
            _Valuation(fair_value=600.0, price=618.80, implied_margin=0.60),
            _history(best_margin=0.05),
        )
        assert any(f.severity == "CRITICAL" for f in findings)

    def test_a_loss_making_history_falls_back_to_an_absolute_ceiling(self):
        findings = plausibility.assess(
            _Valuation(fair_value=600.0, price=618.80, implied_margin=0.80),
            _history(best_margin=-0.05),
        )
        assert any(f.severity == "CRITICAL" for f in findings)


class TestTerminalGrowth:
    def test_growth_above_indian_nominal_gdp_is_flagged(self):
        findings = plausibility.assess(
            _Valuation(fair_value=600.0, price=618.80, implied_growth=0.133),
            _history(), country="IN",
        )
        assert any("perpetual growth" in f.message for f in findings)

    def test_it_warns_rather_than_blocks(self):
        """A bound, not a belief — worth saying, not worth withholding over."""
        findings = plausibility.assess(
            _Valuation(fair_value=600.0, price=618.80, implied_growth=0.133),
            _history(), country="IN",
        )
        growth = [f for f in findings if "perpetual growth" in f.message]
        assert growth and all(f.severity == "MEDIUM" for f in growth)

    def test_the_bar_is_country_specific(self):
        """6% is unremarkable in India and above trend in the US."""
        v = _Valuation(fair_value=600.0, price=618.80, implied_growth=0.06)
        india = plausibility.assess(v, _history(), country="IN")
        us = plausibility.assess(v, _history(), country="US")
        assert not any("perpetual growth" in f.message for f in india)
        assert any("perpetual growth" in f.message for f in us)


class TestGapBands:
    def test_a_clean_valuation_produces_nothing(self):
        assert plausibility.assess(
            _Valuation(price=618.80, fair_value=600.0), _history()
        ) == []

    def test_a_strong_call_warns_without_blocking(self):
        findings = plausibility.assess(
            _Valuation(price=618.80, fair_value=340.0), _history()   # -45%
        )
        assert findings
        assert not any(f.severity == "CRITICAL" for f in findings)

    def test_missing_inputs_do_not_raise(self):
        class _Bare:
            share_price = None
            fair_value = None
            dcf_fair_value = None
            dcf = None
            priced_in = None
        assert plausibility.assess(_Bare(), None) == []


class TestGateWiring:
    def test_critical_findings_reach_the_qa_gate(self):
        from fincopilot.report import qa
        from fincopilot.report.models import ReportModel

        report = ReportModel(company_name="X", ticker="X")
        report.plausibility = [{
            "check": "valuation_plausibility", "severity": "CRITICAL",
            "message": "implied margin impossible", "likely_causes": ["stale base year"],
        }]
        issues = qa.check_valuation_plausibility(report, "CRITICAL")
        assert len(issues) == 1
        assert "stale base year" in issues[0]
        assert qa.check_valuation_plausibility(report, "MEDIUM") == []

    def test_the_check_is_registered_as_blocking(self):
        from fincopilot.report import qa
        assert qa.CHECK_SEVERITY["valuation_plausibility"] == "CRITICAL"
        assert "CRITICAL" in qa.BLOCKING_SEVERITIES

    def test_it_is_not_fixable_by_regenerating_prose(self):
        from fincopilot.report.correction import CHECK_TO_COMPONENT
        assert CHECK_TO_COMPONENT["valuation_plausibility"] is None
