"""
The audited-figures block handed to grounded chat.

Chat and the valuation must not disagree about what revenue was. These tests
pin the three properties that guarantee it: the figures reach the prompt, they
are pre-formatted so the model never has to rescale them, and the block never
claims to cover a year it does not have.
"""

from __future__ import annotations

from fincopilot.chat import qa
from fincopilot.fundamentals.models import FinancialHistory, FiscalYear


def _history(currency: str = "INR") -> FinancialHistory:
    history = FinancialHistory(
        ticker="BIKAJI.NS",
        company_name="Bikaji Foods International Limited",
        currency=currency,
        source="nse_indas_xbrl",
    )
    history.years = [
        FiscalYear(
            fiscal_year=2023, period_end="2023-03-31",
            revenue=19_660_722_000.0, operating_income=2_029_948_000.0,
            net_income=1_285_413_000.0, operating_cash_flow=1_761_505_000.0,
            capex=-864_779_000.0, total_debt=1_435_027_000.0,
            cash_and_equivalents=168_124_000.0,
        ),
        FiscalYear(
            fiscal_year=2024, period_end="2024-03-31",
            revenue=23_293_366_000.0, operating_income=3_312_572_000.0,
            net_income=2_634_626_000.0, operating_cash_flow=2_446_826_000.0,
            capex=-1_283_016_000.0, total_debt=1_186_999_000.0,
            cash_and_equivalents=86_537_000.0,
        ),
    ]
    return history


class TestMoneyFormatting:
    """Both a readable form and the exact value — the model must never rescale."""

    def test_rupees_are_shown_in_crore_with_the_exact_value(self):
        out = qa._format_money(23_293_366_000.0, "INR")
        assert "2,329.34 crore" in out
        assert "23,293,366,000" in out

    def test_small_rupee_amounts_use_lakh(self):
        assert "lakh" in qa._format_money(2_500_000.0, "INR")

    def test_dollars_use_billions_with_the_exact_value(self):
        out = qa._format_money(130_497_000_000.0, "USD")
        assert "130.50bn" in out
        assert "130,497,000,000" in out

    def test_negative_capex_keeps_its_sign(self):
        assert qa._format_money(-1_283_016_000.0, "INR").startswith("INR -128.30 crore")


class TestAuditedBlock:
    def test_absent_financials_yield_no_block(self):
        assert qa._audited_block(None) == ""

    def test_history_with_no_years_yields_no_block(self):
        empty = FinancialHistory(ticker="X", company_name="X", source="nse_indas_xbrl")
        assert qa._audited_block(empty) == ""

    def test_every_year_appears_with_its_period_end(self):
        block = qa._audited_block(_history())
        assert "FY2023 (year ended 2023-03-31)" in block
        assert "FY2024 (year ended 2024-03-31)" in block

    def test_figures_are_present_and_exact(self):
        block = qa._audited_block(_history())
        assert "23,293,366,000" in block      # FY2024 revenue
        assert "2,634,626,000" in block       # FY2024 net income
        assert "2,446,826,000" in block       # FY2024 operating cash flow

    def test_block_states_its_provenance_and_coverage(self):
        block = qa._audited_block(_history())
        assert "Ind-AS XBRL" in block
        assert "AUTHORITATIVE" in block
        assert "FY2023, FY2024" in block

    def test_uncovered_periods_are_directed_to_the_sources(self):
        """The reports reach FY2026 while the XBRL stops earlier."""
        block = qa._audited_block(_history())
        assert "period NOT listed here" in block
        assert "numbered sources" in block

    def test_missing_line_items_are_omitted_not_zeroed(self):
        history = _history()
        history.years[0].operating_income = None
        block = qa._audited_block(history)
        fy2023 = next(l for l in block.splitlines() if l.startswith("FY2023"))
        assert "operating income" not in fy2023
        assert "revenue" in fy2023

    def test_sec_filer_block_uses_its_own_label_and_currency(self):
        history = _history(currency="USD")
        history.source = "sec_xbrl"
        block = qa._audited_block(history)
        assert "SEC XBRL" in block
        assert "USD" in block and "crore" not in block


class TestPromptWiring:
    def test_prompt_accepts_the_audited_block(self):
        rendered = qa._PROMPT.format(
            history="", audited=qa._audited_block(_history()),
            context="[1] something", question="What was revenue?", latest_fy="FY2024",
        )
        assert "AUDITED FIGURES" in rendered
        assert "23,293,366,000" in rendered

    def test_prompt_renders_without_financials(self):
        rendered = qa._PROMPT.format(
            history="", audited=qa._audited_block(None),
            context="[1] something", question="What was revenue?", latest_fy="FY2024",
        )
        assert "AUDITED FIGURES" not in rendered
        assert "Sources:" in rendered

    def test_system_prompt_exempts_audited_figures_from_n_markers(self):
        # An audited figure has no numbered passage behind it, so demanding an
        # [n] marker for it would produce a dangling citation that gets stripped.
        assert "do NOT attach an [n] marker" in qa._SYSTEM


class TestCitationMarkerNormalisation:
    """A correctly-grounded answer must not lose its sources to formatting.

    The context block labels passages "[SOURCE 3]", and models echo that label
    back instead of the bare "[3]" the prompt asks for — observed on real
    answers, including grouped forms like "[SOURCE 1, SOURCE 2]". Unnormalised,
    none of those match the [n] pattern, every citation is dropped, and the
    reader is shown a sourced claim with no sources. The failure is silent,
    which is what makes it worth a test.
    """

    def test_grouped_source_labels_expand_to_individual_markers(self):
        assert qa._normalise_markers(
            "as filed [SOURCE 1, SOURCE 2, SOURCE 3]."
        ) == "as filed [1][2][3]."

    def test_single_source_label_becomes_a_bare_marker(self):
        assert qa._normalise_markers("revenue rose [SOURCE 2].") == "revenue rose [2]."

    def test_an_and_separated_group_is_handled(self):
        assert qa._normalise_markers(
            "[SOURCE 1, SOURCE 2, and SOURCE 3]"
        ) == "[1][2][3]"

    def test_lowercase_labels_are_handled(self):
        assert qa._normalise_markers("see [source 7].") == "see [7]."

    def test_already_correct_markers_are_untouched(self):
        assert qa._normalise_markers("fine [1][2].") == "fine [1][2]."

    def test_mixed_forms_both_resolve(self):
        assert qa._normalise_markers("[SOURCE 4] and [5]") == "[4] and [5]"

    def test_text_without_markers_is_unchanged(self):
        assert qa._normalise_markers("no markers here.") == "no markers here."

    def test_normalised_markers_resolve_to_real_citations(self):
        """End to end: the label form must produce actual Citation objects."""
        class _Chunk:
            doc_title, section, page = "Annual Report 2025-26", "Note 27", 314
            source_url, body = "https://x/ar.pdf", "Revenue from operations..."

        class _Passage:
            chunk = _Chunk()

        class _Result:
            passages = [_Passage()]

        text, citations = qa._resolve_citations(
            "Revenue was INR 2,993.86 crore [SOURCE 1].", _Result()
        )
        assert "[1]" in text
        assert len(citations) == 1
        assert citations[0].page == 314
