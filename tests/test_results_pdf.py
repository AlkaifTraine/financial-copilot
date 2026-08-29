"""
Extraction of audited figures from a SEBI-format results PDF.

Every test here is offline and synthetic. The real Bikaji filing this was built
against is a 3MB scanned document that does not belong in the repository, so
what is pinned instead is the logic that makes reading it safe — the four ways
this format fails silently and expensively:

* a quarter column mistaken for the year column (4x understatement),
* a missed units line (100,000x),
* the standalone statements read instead of the consolidated ones,
* OCR damage to labels, minus signs and digit grouping.

Verified against the filing itself at build time: FY2026 revenue INR 2,993.9 Cr,
PAT INR 254.4 Cr, CFO INR 304.1 Cr, FCF INR 126 Cr, and FY2025 revenue
INR 2,616.8 Cr with PAT INR 194.2 Cr — each matching an independently written
human research report on the same company.
"""

from __future__ import annotations

from datetime import date

import pytest

from fincopilot.fundamentals import results_pdf as rp


class TestNumberParsing:
    @pytest.mark.parametrize("text,expected", [
        ("2,93,474.32", 293474.32),      # Indian lakh grouping
        ("1,234.56", 1234.56),
        ("70,699.43", 70699.43),
        ("10.30", 10.30),
        ("0.46", 0.46),
    ])
    def test_indian_grouping(self, text, expected):
        assert rp.parse_number(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["(398.55)", "/398.55)", "[398.55]", "(398.55|"])
    def test_parenthesised_negatives_including_ocr_damage(self, text):
        """These scans read "(" as "/" or "l". A dropped minus flips a cash
        outflow into an inflow."""
        assert rp.parse_number(text) == pytest.approx(-398.55)

    @pytest.mark.parametrize("text", ["", "   ", "-", "—", "nil", None, "abc"])
    def test_blanks_are_none_not_zero(self, text):
        assert rp.parse_number(text) is None

    def test_split_digit_groups_are_rejoined(self):
        """OCR drops the separator: "30,409.78" arrives as "30" and "409.78".
        Left alone that reads as 409.78 — a 74x understatement."""
        assert rp._merge_split_groups(["30", "409.78"]) == ["30409.78"]
        assert rp.parse_number("30409.78") == pytest.approx(30409.78)

    def test_unrelated_adjacent_numbers_are_not_merged(self):
        assert rp._merge_split_groups(["172.20", "177.72"]) == ["172.20", "177.72"]
        assert rp._merge_split_groups(["2026", "2025"]) == ["2026", "2025"]


class TestUnits:
    @pytest.mark.parametrize("text,multiplier", [
        ("(All Amounts In INR Lakhs, Unless Otherwise Stated)", 1e5),
        ("All amounts in INR Crores", 1e7),
        ("All Amounts in INR Millions", 1e6),
        ("all amounts are in Rs. Thousands", 1e3),
    ])
    def test_declared_units(self, text, multiplier):
        assert rp.detect_units(text)[0] == multiplier

    def test_a_missing_declaration_refuses_rather_than_assuming_rupees(self):
        """Assuming rupees when the statement is in lakhs is a 100,000x error."""
        assert rp.detect_units("Statement of Consolidated Financial Results")[0] is None

    def test_an_unrecognised_unit_refuses(self):
        assert rp.detect_units("All Amounts In INR Zorkles")[0] is None


class TestLabelMatching:
    def test_ocr_damaged_labels_still_match(self):
        assert rp.match_label("Revenue from oaerations", ["revenue from operations"])
        assert rp.match_label("Total exoenses", ["total expenses"])
        assert rp.match_label(
            "Deoreciation, amortisation and imoairment exoenses",
            ["depreciation, amortisation and impairment expenses"],
        )

    def test_a_row_goes_to_its_best_field_not_every_plausible_one(self):
        """The bug this prevents: "Total exoenses" was read as the tax charge,
        putting 2,70,000 lakhs of expenses into a 9,035 lakh line."""
        rows = [
            ["Total exoenses", "269616.12"],
            ["Total tax expenses", "9035.21"],
        ]
        mapping = {
            "total_expenses": ["total expenses"],
            "tax_expense": ["total tax expenses", "tax expense"],
        }
        assigned = rp.assign_rows(rows, mapping)
        assert assigned["total_expenses"] == 0
        assert assigned["tax_expense"] == 1

    def test_a_field_takes_its_best_row_not_the_first_passable_one(self):
        """"Other expenses" appears before "Total expenses" in the statement."""
        rows = [["Other expenses", "44050.01"], ["Total exoenses", "269616.12"]]
        assigned = rp.assign_rows(rows, {"total_expenses": ["total expenses"]})
        assert assigned["total_expenses"] == 1

    def test_the_ind_as_top_line_wins_over_the_narrower_one(self):
        rows = [
            ["Revenue from oaerations", "293474.32"],
            ["Total revenue from operations", "299386.34"],
        ]
        assigned = rp.assign_rows(
            rows, {"revenue": ["total revenue from operations", "revenue from operations"]}
        )
        assert assigned["revenue"] == 1

    def test_a_section_header_prefix_does_not_hide_the_row(self):
        """Capex arrives as "CASH FLOW FROM INVESTING ACTIVITIES:- Purchase of
        property, plant and equipment…"."""
        assert rp.match_label(
            "CASH FLOW FROM INVESTING ACTIVITIES:- Purchase of property, plant "
            "and equipment, intangi",
            ["purchase of property, plant and equipment"],
        )

    def test_the_opposite_sign_neighbour_is_not_confused_with_capex(self):
        rows = [
            ["Proceeds from sale of property, plant and equipment", "172.20"],
            ["Purchase of property, plant and equipment, intangible", "-17842.05"],
        ]
        assigned = rp.assign_rows(
            rows, {"capex": ["purchase of property, plant and equipment"]}
        )
        assert assigned["capex"] == 1


class TestAnnualColumnResolution:
    """The four-times error lives here."""

    def _pl_header(self):
        return [
            ["Particulars", "Quarter Ended", "", "", "Year Ended", ""],
            ["", "March 31 2026", "December 31 2025", "March 31 2025",
             "March 31 2026", "March 31 2025"],
            ["", "(Audited)", "(Unaudited)", "(Audited)", "(Audited)", "(Audited)"],
        ]

    def test_it_selects_the_year_columns_not_the_quarters(self):
        columns = rp.resolve_annual_columns(self._pl_header())
        assert columns is not None
        assert columns.indices == [4, 5]
        assert columns.period_ends[0] == date(2026, 3, 31)

    def test_the_newest_year_comes_first(self):
        columns = rp.resolve_annual_columns(self._pl_header())
        assert columns.period_ends == [date(2026, 3, 31), date(2025, 3, 31)]

    def test_quarters_without_a_year_header_refuse(self):
        """A Q4 column and a full-year column both end 31 March, so with the
        header gone there is no way to tell them apart — and guessing wrong
        understates by roughly four times."""
        rows = [
            ["Particulars", "Quarter Ended", "", ""],
            ["", "March 31 2026", "December 31 2025", "March 31 2025"],
        ]
        assert rp.resolve_annual_columns(rows) is None

    def test_an_annual_only_table_uses_every_dated_column(self):
        rows = [
            ["Particulars", "Year ended", "Year ended"],
            ["", "March 31, 2026", "March 31, 2025"],
        ]
        columns = rp.resolve_annual_columns(rows)
        assert columns is not None
        assert len(columns.indices) == 2

    def test_no_dates_and_no_hint_refuses(self):
        assert rp.resolve_annual_columns([["Particulars", "a", "b"]]) is None

    def test_a_hint_rescues_a_fragmented_annual_header(self):
        """Balance-sheet headers fragment as "March 31 2026 Ma" | "rch" |
        "31 2025", so the dates cannot be read — but those statements carry
        only annual columns."""
        rows = [
            ["Particulars", "", "", ""],
            ["Total assets", "10", "223954.23", "193407.14"],
            ["Total equity", "11", "170740.83", "148049.46"],
            ["Cash", "12", "3985.00", "2480.00"],
        ]
        hint = [date(2026, 3, 31), date(2025, 3, 31)]
        columns = rp.resolve_annual_columns(rows, period_hint=hint)
        assert columns is not None
        assert columns.period_ends == hint

    def test_the_hint_never_rescues_a_table_with_quarters(self):
        rows = [
            ["Particulars", "Quarter Ended", "", ""],
            ["Revenue", "1", "2", "3"],
            ["Profit", "4", "5", "6"],
            ["Tax", "7", "8", "9"],
        ]
        assert rp.resolve_annual_columns(
            rows, period_hint=[date(2026, 3, 31), date(2025, 3, 31)]
        ) is None


class TestValidation:
    def _sound(self):
        return {
            "revenue": 29_938_634_000.0, "other_income": 514_111_000.0,
            "total_income": 30_452_745_000.0, "total_expenses": 26_961_612_000.0,
            "pretax_income": 3_447_619_000.0, "tax_expense": 903_521_000.0,
            "net_income": 2_544_098_000.0,
            "total_assets": 22_395_423_000.0, "shareholders_equity": 17_074_083_000.0,
        }

    def test_a_sound_statement_passes(self):
        assert rp.validate(self._sound()) == []

    def test_a_misread_tax_line_is_caught(self):
        """The exact failure: total expenses read into the tax field."""
        values = self._sound()
        values["tax_expense"] = 26_961_612_000.0
        assert any("PBT - tax" in p for p in rp.validate(values))

    def test_revenue_and_other_income_must_reconcile_to_total_income(self):
        values = self._sound()
        values["total_income"] = 99_000_000_000.0
        assert any("total income" in p for p in rp.validate(values))

    def test_negative_revenue_is_rejected(self):
        values = self._sound()
        values["revenue"] = -1.0
        assert any("not positive" in p for p in rp.validate(values))

    def test_equity_above_assets_is_rejected(self):
        values = self._sound()
        values["shareholders_equity"] = values["total_assets"] * 2
        assert any("equity exceeds" in p for p in rp.validate(values))

    def test_missing_lines_do_not_fail_the_year(self):
        """Not every statement discloses every line."""
        assert rp.validate({"revenue": 1_000.0}) == []


class TestUnitSanityCheck:
    def test_a_lakh_crore_mixup_is_caught_by_price_to_sales(self):
        """Revenue 100x too small makes the implied multiple absurd."""
        problem = rp._sanity_scale({"revenue": 299_386_340.0}, 155_000_000_000.0)
        assert problem is not None and "price-to-sales" in problem

    def test_correct_units_pass(self):
        assert rp._sanity_scale({"revenue": 29_938_634_000.0}, 155_000_000_000.0) is None

    def test_it_is_skipped_without_a_market_cap(self):
        assert rp._sanity_scale({"revenue": 1.0}, None) is None


class TestConsolidatedSelection:
    def test_a_missing_file_returns_none_rather_than_raising(self):
        assert rp.extract("/no/such/file.pdf") is None


class TestDiscoveryVersioning:
    """A manifest written by older discovery logic must not be reused.

    This is what kept Bikaji pinned to a stale document set after the scoring
    fix: every file in the cached manifest still existed, so nothing looked
    wrong, and the newly-preferred annual filing was simply never fetched.
    """

    def _company(self, tmp_path, monkeypatch):
        from fincopilot import config
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)

        class _C:
            ticker = "TEST.NS"
            slug = "test_ns"
            def to_dict(self):
                return {"ticker": self.ticker, "slug": self.slug}
        return _C()

    def test_a_current_manifest_is_reused(self, tmp_path, monkeypatch):
        from fincopilot.ingest import registry
        company = self._company(tmp_path, monkeypatch)
        registry.write_manifest(company, [], [])
        assert registry.read_manifest(company) is not None

    def test_a_manifest_from_older_discovery_is_discarded(self, tmp_path, monkeypatch):
        import json
        from fincopilot.ingest import registry
        company = self._company(tmp_path, monkeypatch)
        registry.write_manifest(company, [], [])

        path = registry.manifest_path(company)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["discovery_version"] = "v1"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert registry.read_manifest(company) is None

    def test_an_unversioned_manifest_is_discarded(self, tmp_path, monkeypatch):
        """Everything cached before versioning existed."""
        import json
        from fincopilot.ingest import registry
        company = self._company(tmp_path, monkeypatch)
        registry.write_manifest(company, [], [])

        path = registry.manifest_path(company)
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["discovery_version"]
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert registry.read_manifest(company) is None


class TestOcrTolerantUnits:
    """One Bikaji filing declares "All Amounts In JNR Lakhs" — the OCR turned
    INR into JNR. Requiring a correctly spelled currency discarded a statement
    whose unit word was perfectly legible."""

    @pytest.mark.parametrize("header,multiplier", [
        ("(All Amounts In JNR Lakhs Unless Otherwise Stated)", 1e5),
        ("(All Amounts In INR Lakhs, Unless Otherwise Stated)", 1e5),
        ("All amounts in Rs. Crores", 1e7),
        ("All Amounts in ₹ Millions", 1e6),
        ("all amounts are in INR thousands", 1e3),
    ])
    def test_the_unit_is_read_even_when_the_currency_is_mangled(self, header, multiplier):
        assert rp.detect_units(header)[0] == multiplier

    def test_it_still_refuses_when_no_unit_is_present(self):
        assert rp.detect_units("Statement of Consolidated Financial Results")[0] is None
        assert rp.detect_units("All Amounts In INR Zorkles")[0] is None


class TestCombiningSources:
    """XBRL and the results filings cover different spans; taking only one
    leaves the DCF with two fiscal years and a single growth rate."""

    def _history(self, source, years):
        from fincopilot.fundamentals.models import FinancialHistory, FiscalYear
        h = FinancialHistory(ticker="X.NS", company_name="X", currency="INR", source=source)
        h.years = [
            FiscalYear(fiscal_year=fy, period_end=f"{fy}-03-31", revenue=rev)
            for fy, rev in years
        ]
        return h

    def test_non_overlapping_spans_are_merged(self):
        from fincopilot.fundamentals.recency import combine
        older = self._history("nse_indas_xbrl", [(2023, 1.0), (2024, 2.0)])
        newer = self._history("sebi_results_pdf", [(2025, 3.0), (2026, 4.0)])
        merged = combine(older, newer)
        assert [y.fiscal_year for y in merged.years] == [2023, 2024, 2025, 2026]

    def test_a_shared_year_comes_from_the_fresher_source(self):
        from fincopilot.fundamentals.recency import combine
        older = self._history("nse_indas_xbrl", [(2024, 100.0), (2025, 200.0)])
        newer = self._history("sebi_results_pdf", [(2025, 200.5), (2026, 300.0)])
        merged = combine(older, newer)
        fy2025 = next(y for y in merged.years if y.fiscal_year == 2025)
        assert fy2025.revenue == pytest.approx(200.5)

    def test_sources_that_disagree_are_not_spliced(self):
        """A shared year read two entirely different ways is a free correctness
        check. Disagreement means one is wrong, so the merge is abandoned."""
        from fincopilot.fundamentals.recency import combine
        older = self._history("nse_indas_xbrl", [(2024, 100.0), (2025, 200.0)])
        newer = self._history("sebi_results_pdf", [(2025, 500.0), (2026, 300.0)])
        merged = combine(older, newer)
        assert [y.fiscal_year for y in merged.years] == [2025, 2026]
        assert merged.source == "sebi_results_pdf"

    def test_a_single_source_passes_through(self):
        from fincopilot.fundamentals.recency import combine
        only = self._history("sebi_results_pdf", [(2025, 1.0), (2026, 2.0)])
        assert combine(None, only) is only

    def test_no_sources_returns_none(self):
        from fincopilot.fundamentals.recency import combine
        assert combine(None, None) is None
