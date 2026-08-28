"""
Ind-AS XBRL extraction for Indian issuers.

These tests run offline against XBRL instances built to reproduce the exact
defects seen in real NSE filings — the ones that would otherwise produce
confident, plausible, wrong numbers:

* an annual filing tags the fourth quarter and the full year side by side, and
  ``xbrli:period`` carries the *quarter's* dates on both contexts;
* pre-2023 instances reference contexts (``OneD``/``FourD``/``OneI``) they
  never define;
* ``ProfitOrLossAttributableToOwnersOfParent`` is tagged ``0.00`` by filers with
  no non-controlling interest.

Each is a real failure mode, not a hypothetical.
"""

from __future__ import annotations

import pytest

from fincopilot.fundamentals import indas
from fincopilot.fundamentals.models import FiscalYear
from fincopilot.ingest import nse


def _instance(*, dangling: bool = False, owners_zero: bool = True) -> bytes:
    """A minimal but realistic annual results instance.

    Quarter (``OneD``) and year (``FourD``) both carry the quarter's dates in
    ``xbrli:period``, exactly as NSE filings do; the true window is only in the
    ``DateOf*ReportingPeriod`` facts.
    """
    contexts = ""
    if not dangling:
        contexts = """
  <xbrli:context id="OneD"><xbrli:period>
    <xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-03-31</xbrli:endDate>
  </xbrli:period></xbrli:context>
  <xbrli:context id="FourD"><xbrli:period>
    <xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-03-31</xbrli:endDate>
  </xbrli:period></xbrli:context>
  <xbrli:context id="OneI"><xbrli:period>
    <xbrli:instant>2024-03-31</xbrli:instant>
  </xbrli:period></xbrli:context>"""

    owners = "0.00" if owners_zero else "2000000000.00"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
            xmlns:f="http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin">
{contexts}
  <f:DateOfStartOfReportingPeriod contextRef="OneD">2024-01-01</f:DateOfStartOfReportingPeriod>
  <f:DateOfEndOfReportingPeriod contextRef="OneD">2024-03-31</f:DateOfEndOfReportingPeriod>
  <f:DateOfStartOfReportingPeriod contextRef="FourD">2023-04-01</f:DateOfStartOfReportingPeriod>
  <f:DateOfEndOfReportingPeriod contextRef="FourD">2024-03-31</f:DateOfEndOfReportingPeriod>
  <f:NatureOfReportStandaloneConsolidated contextRef="OneD">Consolidated</f:NatureOfReportStandaloneConsolidated>
  <f:NatureOfReportStandaloneConsolidated contextRef="FourD">Consolidated</f:NatureOfReportStandaloneConsolidated>
  <f:WhetherResultsAreAuditedOrUnaudited contextRef="FourD">Audited</f:WhetherResultsAreAuditedOrUnaudited>

  <f:RevenueFromOperations contextRef="OneD">2500000000.00</f:RevenueFromOperations>
  <f:RevenueFromOperations contextRef="FourD">10000000000.00</f:RevenueFromOperations>
  <f:OtherIncome contextRef="FourD">100000000.00</f:OtherIncome>
  <f:FinanceCosts contextRef="FourD">50000000.00</f:FinanceCosts>
  <f:ProfitBeforeExceptionalItemsAndTax contextRef="FourD">1250000000.00</f:ProfitBeforeExceptionalItemsAndTax>
  <f:ProfitBeforeTax contextRef="FourD">1250000000.00</f:ProfitBeforeTax>
  <f:TaxExpense contextRef="FourD">250000000.00</f:TaxExpense>
  <f:ProfitLossForPeriod contextRef="FourD">1000000000.00</f:ProfitLossForPeriod>
  <f:ProfitOrLossAttributableToOwnersOfParent contextRef="FourD">{owners}</f:ProfitOrLossAttributableToOwnersOfParent>
  <f:DepreciationDepletionAndAmortisationExpense contextRef="FourD">300000000.00</f:DepreciationDepletionAndAmortisationExpense>
  <f:CashFlowsFromUsedInOperatingActivities contextRef="FourD">1400000000.00</f:CashFlowsFromUsedInOperatingActivities>
  <f:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities contextRef="FourD">400000000.00</f:PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities>
  <f:CostOfMaterialsConsumed contextRef="FourD">6000000000.00</f:CostOfMaterialsConsumed>

  <f:Assets contextRef="OneI">8000000000.00</f:Assets>
  <f:CurrentAssets contextRef="OneI">3000000000.00</f:CurrentAssets>
  <f:CurrentLiabilities contextRef="OneI">1500000000.00</f:CurrentLiabilities>
  <f:Equity contextRef="OneI">5000000000.00</f:Equity>
  <f:CashAndCashEquivalents contextRef="OneI">200000000.00</f:CashAndCashEquivalents>
  <f:BorrowingsCurrent contextRef="OneI">300000000.00</f:BorrowingsCurrent>
  <f:BorrowingsNoncurrent contextRef="OneI">700000000.00</f:BorrowingsNoncurrent>
</xbrli:xbrl>""".encode("utf-8")


class TestPeriodSelection:
    """The quarter must never be mistaken for the year."""

    def test_annual_context_is_the_twelve_month_one(self):
        periods, _facts = indas._parse_instance(_instance())
        annual = indas._pick_annual_context(periods)

        assert annual is not None
        assert annual.context_id == "FourD"
        assert annual.days == 365

    def test_quarter_context_is_not_annual(self):
        periods, _facts = indas._parse_instance(_instance())

        # Both contexts carry the quarter's dates in xbrli:period; only the
        # reporting-period facts distinguish them.
        assert periods["OneD"].is_annual is False
        assert periods["FourD"].is_annual is True

    def test_standalone_context_is_rejected(self):
        payload = _instance().replace(
            b'<f:NatureOfReportStandaloneConsolidated contextRef="FourD">Consolidated',
            b'<f:NatureOfReportStandaloneConsolidated contextRef="FourD">Standalone',
        )
        periods, _facts = indas._parse_instance(payload)
        assert indas._pick_annual_context(periods) is None

    def test_unaudited_context_is_rejected(self):
        payload = _instance().replace(
            b'<f:WhetherResultsAreAuditedOrUnaudited contextRef="FourD">Audited',
            b'<f:WhetherResultsAreAuditedOrUnaudited contextRef="FourD">Un-Audited',
        )
        periods, _facts = indas._parse_instance(payload)
        assert indas._pick_annual_context(periods) is None


class TestMalformedInstances:
    """NSE's pre-2023 instances reference contexts they never define."""

    def test_dangling_contexts_are_reconstructed(self):
        periods, facts = indas._parse_instance(_instance(dangling=True))

        annual = indas._pick_annual_context(periods)
        assert annual is not None and annual.context_id == "FourD"
        assert indas._value(facts, ["RevenueFromOperations"], "FourD") == 10_000_000_000.0

    def test_undated_balance_sheet_context_is_found(self):
        periods, facts = indas._parse_instance(_instance(dangling=True))
        annual = indas._pick_annual_context(periods)

        instant = indas._pick_instant_context(periods, facts, annual)
        assert instant == "OneI"
        assert indas._value(facts, ["Assets"], instant) == 8_000_000_000.0

    def test_no_assets_context_yields_no_balance_sheet(self):
        payload = _instance(dangling=True).replace(
            b'<f:Assets contextRef="OneI">8000000000.00</f:Assets>', b""
        )
        periods, facts = indas._parse_instance(payload)
        annual = indas._pick_annual_context(periods)

        # Better an empty balance sheet than one attached to a guessed date.
        assert indas._pick_instant_context(periods, facts, annual) is None


class TestConceptMapping:
    def test_zero_attributable_profit_falls_back_to_total(self):
        _periods, facts = indas._parse_instance(_instance(owners_zero=True))
        assert indas._net_income(facts, "FourD") == 1_000_000_000.0

    def test_real_attributable_profit_is_preferred(self):
        _periods, facts = indas._parse_instance(_instance(owners_zero=False))
        assert indas._net_income(facts, "FourD") == 2_000_000_000.0

    def test_ebit_unwinds_finance_costs_and_other_income(self):
        _periods, facts = indas._parse_instance(_instance())
        # 1,250m PBT + 50m finance costs - 100m other income
        assert indas._operating_income(facts, "FourD") == 1_200_000_000.0

    def test_ebit_is_none_when_a_component_is_missing(self):
        payload = _instance().replace(
            b'<f:OtherIncome contextRef="FourD">100000000.00</f:OtherIncome>', b""
        )
        _periods, facts = indas._parse_instance(payload)
        assert indas._operating_income(facts, "FourD") is None

    def test_debt_sums_current_and_noncurrent_borrowings(self):
        _periods, facts = indas._parse_instance(_instance())
        assert indas._sum_values(facts, indas._DEBT_CONCEPTS, "OneI") == 1_000_000_000.0

    def test_missing_concepts_sum_to_none_not_zero(self):
        _periods, facts = indas._parse_instance(_instance())
        assert indas._sum_values(facts, ["NotATag", "AlsoNotATag"], "OneI") is None


class TestValidation:
    """Accounting identities are what catch a mis-mapped concept."""

    def test_a_clean_year_passes(self):
        entry = FiscalYear(
            fiscal_year=2024, period_end="2024-03-31",
            revenue=10_000_000_000.0, operating_income=1_200_000_000.0,
            pretax_income=1_250_000_000.0, tax_expense=250_000_000.0,
            net_income=1_000_000_000.0, total_assets=8_000_000_000.0,
            shareholders_equity=5_000_000_000.0, current_liabilities=1_500_000_000.0,
            cash_and_equivalents=200_000_000.0,
        )
        assert indas._validate(entry) == []

    def test_tax_bridge_that_does_not_reconcile_is_rejected(self):
        entry = FiscalYear(
            fiscal_year=2024, period_end="2024-03-31",
            revenue=10_000_000_000.0,
            pretax_income=1_250_000_000.0, tax_expense=250_000_000.0,
            net_income=0.0,          # the real Bikaji FY2024 mis-map
        )
        problems = indas._validate(entry)
        assert any("does not reconcile" in p for p in problems)

    def test_implausible_margin_is_rejected(self):
        entry = FiscalYear(
            fiscal_year=2024, period_end="2024-03-31",
            revenue=10_000_000_000.0,
            operating_income=90_000_000_000.0,   # a scale error
        )
        assert any("operating margin implausible" in p for p in indas._validate(entry))

    def test_missing_revenue_is_rejected(self):
        entry = FiscalYear(fiscal_year=2024, period_end="2024-03-31")
        assert indas._validate(entry) == ["revenue missing or non-positive"]

    def test_negative_cash_is_rejected(self):
        entry = FiscalYear(
            fiscal_year=2024, period_end="2024-03-31",
            revenue=10_000_000_000.0, cash_and_equivalents=-1.0,
        )
        assert "negative cash balance" in indas._validate(entry)


class TestFetchGate:
    def test_non_indian_company_is_not_handled_here(self):
        class _Company:
            country = "US"
            base_ticker = "NVDA"
            ticker = "NVDA"
            name = "NVIDIA"

        assert indas.fetch(_Company()) is None

    def test_no_exchange_filings_returns_none(self, monkeypatch):
        class _Company:
            country = "IN"
            base_ticker = "NOSUCH"
            ticker = "NOSUCH.NS"
            name = "No Such Ltd"

        monkeypatch.setattr(nse, "annual_xbrl_filings", lambda symbol: [])
        assert indas.fetch(_Company()) is None


class TestNSEClient:
    def test_placeholder_xbrl_url_is_dropped(self):
        assert nse._clean_url("-") is None
        assert nse._clean_url("") is None
        assert nse._clean_url(None) is None
        assert nse._clean_url(" https://x/y.xml ") == "https://x/y.xml"

    @pytest.mark.parametrize(
        "raw,expected",
        [("31-Mar-2024", "2024-03-31"), ("01-Apr-2020", "2020-04-01"), ("bad", None)],
    )
    def test_date_parsing(self, raw, expected):
        assert nse._parse_date(raw) == expected

    def test_unaudited_quarterly_and_pdf_only_filings_are_dropped(self, monkeypatch):
        rows = [
            # keep: audited, consolidated, annual, has XBRL
            {"toDate": "31-Mar-2024", "relatingTo": "Annual", "audited": "Audited",
             "consolidated": "Consolidated", "xbrl": "https://x/good.xml"},
            # drop: unaudited
            {"toDate": "31-Dec-2023", "relatingTo": "Third Quarter", "audited": "Un-Audited",
             "consolidated": "Consolidated", "xbrl": "https://x/q3.xml"},
            # drop: no XBRL instance filed
            {"toDate": "31-Mar-2022", "relatingTo": "Annual", "audited": "Audited",
             "consolidated": "Consolidated", "xbrl": "-"},
        ]
        monkeypatch.setattr(nse, "_get", lambda path: rows)

        assert [f.xbrl_url for f in nse.annual_xbrl_filings("TEST")] == [
            "https://x/good.xml"
        ]

    def test_standalone_is_kept_but_ranked_below_consolidated(self, monkeypatch):
        """Nestle India files standalone only; refusing it would be wrong."""
        rows = [
            {"toDate": "31-Mar-2024", "relatingTo": "Annual", "audited": "Audited",
             "consolidated": "Non-Consolidated", "xbrl": "https://x/standalone.xml"},
            {"toDate": "31-Mar-2024", "relatingTo": "Annual", "audited": "Audited",
             "consolidated": "Consolidated", "xbrl": "https://x/consolidated.xml"},
            {"toDate": "31-Mar-2023", "relatingTo": "Annual", "audited": "Audited",
             "consolidated": "Non-Consolidated", "xbrl": "https://x/standalone23.xml"},
        ]
        monkeypatch.setattr(nse, "_get", lambda path: rows)

        filings = nse.annual_xbrl_filings("TEST")
        # Consolidated first for the period where both exist, so the caller
        # takes it; the standalone-only year still survives.
        assert filings[0].xbrl_url == "https://x/consolidated.xml"
        assert "https://x/standalone23.xml" in [f.xbrl_url for f in filings]

    def test_standalone_context_is_accepted_when_explicitly_allowed(self):
        payload = _instance().replace(
            b'<f:NatureOfReportStandaloneConsolidated contextRef="FourD">Consolidated',
            b'<f:NatureOfReportStandaloneConsolidated contextRef="FourD">Standalone',
        )
        periods, _facts = indas._parse_instance(payload)

        assert indas._pick_annual_context(periods) is None
        allowed = indas._pick_annual_context(periods, allow_standalone=True)
        assert allowed is not None and allowed.context_id == "FourD"
