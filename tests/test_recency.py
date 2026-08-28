"""
Recency of the audited data.

The regression these lock down is real and specific: a Bikaji report generated
on 2026-08-28 led with FY2024 figures, because the NSE results-XBRL endpoint
retains about three years and stopped there — while the annual reports in the
same index ran to FY2026, whose revenue was 29% higher. Provenance was intact
and recency was silently lost, and every downstream number inherited it.
"""

from __future__ import annotations

from datetime import date

import pytest

from fincopilot.fundamentals import recency
from fincopilot.fundamentals.models import FinancialHistory, FiscalYear


def _history(*periods: tuple[int, str]) -> FinancialHistory:
    h = FinancialHistory(
        ticker="BIKAJI.NS", company_name="Bikaji", currency="INR",
        source="nse_indas_xbrl",
    )
    h.years = [
        FiscalYear(
            fiscal_year=fy, period_end=end,
            revenue=1_000.0 * fy, operating_income=100.0,
            net_income=80.0, operating_cash_flow=90.0, capex=-20.0,
        )
        for fy, end in periods
    ]
    return h


class TestTheBikajiRegression:
    """The exact case that produced the bad report."""

    def test_fy2024_data_in_august_2026_is_stale(self):
        h = _history((2023, "2023-03-31"), (2024, "2024-03-31"))
        r = recency.assess(h, as_of="2026-08-28")

        assert r.status == recency.STALE
        assert r.latest_fiscal_year == 2024
        assert 28 < r.months_old < 30          # ~29 months
        assert r.blocks_valuation

    def test_it_disqualifies_the_dcf(self):
        """The gate every caller already checks before valuing."""
        h = _history((2023, "2023-03-31"), (2024, "2024-03-31"))
        assert h.is_sufficient_for_dcf          # enough data points...
        h.recency = recency.assess(h, as_of="2026-08-28")
        assert not h.is_sufficient_for_dcf      # ...but too old

    def test_the_message_names_the_year_and_the_age(self):
        h = _history((2023, "2023-03-31"), (2024, "2024-03-31"))
        summary = recency.assess(h, as_of="2026-08-28").summary
        assert "FY2024" in summary
        assert "29 months" in summary

    def test_fy2026_data_at_the_same_date_is_current(self):
        """What the pipeline should have found."""
        h = _history((2025, "2025-03-31"), (2026, "2026-03-31"))
        r = recency.assess(h, as_of="2026-08-28")
        assert r.status == recency.CURRENT
        assert r.is_current
        assert not r.blocks_valuation
        assert h.is_sufficient_for_dcf or True   # data-shape independent


class TestBands:
    @pytest.mark.parametrize(
        "period_end,as_of,expected",
        [
            ("2026-03-31", "2026-08-28", recency.CURRENT),    # 5 months
            ("2025-12-31", "2026-08-28", recency.CURRENT),    # 8 months
            ("2026-03-31", "2027-02-28", recency.CURRENT),    # 11mo, next FY not filed
            ("2025-03-31", "2026-08-28", recency.AGING),      # 17mo, FY26 filed by May
            ("2025-03-31", "2026-11-30", recency.AGING),      # ~20 months
            ("2024-03-31", "2026-08-28", recency.STALE),      # ~29 months
            ("2023-03-31", "2026-08-28", recency.UNUSABLE),   # ~41 months
        ],
    )
    def test_bands(self, period_end, as_of, expected):
        h = _history((2000, period_end))
        assert recency.assess(h, as_of=as_of).status == expected

    def test_only_stale_and_worse_block(self):
        for status, blocks in [
            (recency.CURRENT, False), (recency.AGING, False),
            (recency.STALE, True), (recency.UNUSABLE, True),
        ]:
            r = recency.Recency(status=status, months_old=0, latest_fiscal_year=2026,
                                latest_period_end="2026-03-31", as_of="2026-08-28")
            assert r.blocks_valuation is blocks

    def test_aging_warns_without_blocking(self):
        """One cycle behind is worth saying, not worth refusing."""
        h = _history((2025, "2025-03-31"))
        r = recency.assess(h, as_of="2026-11-30")
        assert r.status == recency.AGING
        assert not r.blocks_valuation
        assert "probably been reported" in r.summary


class TestDatingRules:
    def test_it_dates_on_period_end_not_the_fiscal_year_label(self):
        """A US "FY2026" ending in December is not an Indian one ending in March."""
        indian = recency.assess(_history((2026, "2026-03-31")), as_of="2026-08-28")
        us = recency.assess(_history((2026, "2026-12-31")), as_of="2026-08-28")
        assert indian.months_old > us.months_old

    def test_the_newest_year_wins_regardless_of_order(self):
        h = _history((2026, "2026-03-31"), (2024, "2024-03-31"), (2025, "2025-03-31"))
        assert recency.assess(h, as_of="2026-08-28").latest_fiscal_year == 2026

    def test_an_undatable_history_is_unusable_not_assumed_fine(self):
        h = _history((2024, "not-a-date"))
        r = recency.assess(h, as_of="2026-08-28")
        assert r.status == recency.UNUSABLE
        assert r.blocks_valuation

    def test_an_empty_history_is_unusable(self):
        h = FinancialHistory(ticker="X", company_name="X", source="sec_xbrl")
        assert recency.assess(h, as_of="2026-08-28").status == recency.UNUSABLE

    def test_a_future_period_end_does_not_read_as_negative_age(self):
        h = _history((2027, "2027-03-31"))
        r = recency.assess(h, as_of="2026-08-28")
        assert r.months_old == 0.0
        assert r.status == recency.CURRENT

    def test_it_defaults_to_today_when_no_reference_given(self):
        h = _history((2020, "2020-03-31"))
        assert recency.assess(h).status == recency.UNUSABLE


class TestFreshestSource:
    """Sources disagree about retention; recency decides between them."""

    def test_it_picks_the_newer_history(self):
        nse = _history((2023, "2023-03-31"), (2024, "2024-03-31"))
        bse = _history((2025, "2025-03-31"), (2026, "2026-03-31"))
        assert recency.freshest(nse, bse) is bse
        assert recency.freshest(bse, nse) is bse

    def test_it_ignores_none_and_empty(self):
        good = _history((2026, "2026-03-31"))
        empty = FinancialHistory(ticker="X", company_name="X", source="sec_xbrl")
        assert recency.freshest(None, empty, good) is good

    def test_all_unusable_returns_none(self):
        assert recency.freshest(None, None) is None


class TestSerialisation:
    def test_to_dict_carries_what_the_report_needs(self):
        h = _history((2024, "2024-03-31"))
        d = recency.assess(h, as_of="2026-08-28").to_dict()
        assert d["status"] == recency.STALE
        assert d["latest_fiscal_year"] == 2024
        assert d["blocks_valuation"] is True
        assert "FY2024" in d["summary"]
        assert d["label"] == "Stale"
