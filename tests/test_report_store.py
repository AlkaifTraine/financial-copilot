"""
The report store: fingerprinting, round-tripping, and what must NOT be cached.

The store exists to stop paying $0.15 and two minutes to regenerate a document
whose inputs have not changed. That makes two properties load-bearing: the
fingerprint must change when — and only when — something that legitimately
alters the report changes, and a report the QA gate *blocked* must never be
served from cache.
"""

from __future__ import annotations

import pytest

from fincopilot.fundamentals.models import FinancialHistory, FiscalYear
from fincopilot.report import store
from fincopilot.report.models import Evidence, KPI, ReportModel, Section


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Never touch the real store from a test."""
    monkeypatch.setattr(store, "_DB_PATH", tmp_path / "reports.db")


class _Company:
    ticker = "BIKAJI.NS"
    slug = "bikaji_ns"


class _Doc:
    def __init__(self, sha):
        self.sha256 = sha
        self.url = f"https://x/{sha}.pdf"


class _Ingest:
    def __init__(self, shas):
        self.accepted = [_Doc(s) for s in shas]


def _history(revenue: float = 23_293_366_000.0) -> FinancialHistory:
    history = FinancialHistory(
        ticker="BIKAJI.NS", company_name="Bikaji", currency="INR",
        source="nse_indas_xbrl",
    )
    history.years = [
        FiscalYear(fiscal_year=2024, period_end="2024-03-31", revenue=revenue,
                   operating_income=3_312_572_000.0, net_income=2_634_626_000.0,
                   operating_cash_flow=2_446_826_000.0, capex=-1_283_016_000.0)
    ]
    return history


def _report(blocked: bool = False) -> ReportModel:
    report = ReportModel(
        company_name="Bikaji Foods International Limited",
        ticker="BIKAJI.NS", currency="INR", rating="HOLD",
        share_price=620.2, fair_value=580.0, upside=-0.065,
    )
    report.kpis = [KPI(label="Revenue", value="INR 2,329cr", tone="positive")]
    report.sections = [
        Section(
            key="business", title="The business",
            summary="Ethnic snacks at national scale.",
            paragraphs=["Revenue grew 18.5% in FY2024."],
            evidence=[Evidence(doc_title="Annual Report 2023-24", section="MD&A",
                               page=42, source_url="https://x/ar.pdf", snippet="...")],
        )
    ]
    report.blocked = blocked
    if blocked:
        report.blocking_issues = ["scenario probabilities do not sum to 100%"]
        report.qa_status = "FAILED"
    return report


class TestFingerprint:
    def test_identical_inputs_give_the_same_key(self):
        args = (_Company(), _history(), _Ingest(["a", "b"]), None)
        assert store.fingerprint(*args) == store.fingerprint(*args)

    def test_document_order_does_not_matter(self):
        a = store.fingerprint(_Company(), _history(), _Ingest(["a", "b"]), None)
        b = store.fingerprint(_Company(), _history(), _Ingest(["b", "a"]), None)
        assert a == b

    def test_a_new_filing_changes_the_key(self):
        before = store.fingerprint(_Company(), _history(), _Ingest(["a"]), None)
        after = store.fingerprint(_Company(), _history(), _Ingest(["a", "new"]), None)
        assert before != after

    def test_a_restatement_changes_the_key_with_no_new_document(self):
        """The whole point of hashing the numbers as well as the documents."""
        docs = _Ingest(["a"])
        before = store.fingerprint(_Company(), _history(), docs, None)
        after = store.fingerprint(_Company(), _history(revenue=23_400_000_000.0), docs, None)
        assert before != after

    def test_an_analyst_override_changes_the_key(self):
        docs = _Ingest(["a"])
        before = store.fingerprint(_Company(), _history(), docs, None)
        after = store.fingerprint(
            _Company(), _history(), docs, {"wacc": 0.11}
        )
        assert before != after

    def test_a_logic_version_bump_changes_the_key(self, monkeypatch):
        docs = _Ingest(["a"])
        before = store.fingerprint(_Company(), _history(), docs, None)
        monkeypatch.setattr(store, "REPORT_LOGIC_VERSION", "v9.0")
        assert store.fingerprint(_Company(), _history(), docs, None) != before


class TestRoundTrip:
    def test_a_stored_report_comes_back_equivalent(self):
        store.put("k1", _report(), cost_usd=0.15)
        loaded = store.get("k1")

        assert loaded is not None
        assert loaded.company_name == "Bikaji Foods International Limited"
        assert loaded.rating == "HOLD"
        assert loaded.fair_value == pytest.approx(580.0)

    def test_nested_sections_and_evidence_survive(self):
        store.put("k2", _report())
        loaded = store.get("k2")

        assert isinstance(loaded.sections[0], Section)
        assert loaded.sections[0].title == "The business"
        assert isinstance(loaded.sections[0].evidence[0], Evidence)
        assert loaded.sections[0].evidence[0].page == 42
        assert isinstance(loaded.kpis[0], KPI)
        assert loaded.kpis[0].label == "Revenue"

    def test_a_missing_key_is_a_miss_not_an_error(self):
        assert store.get("never-stored") is None

    def test_storing_again_replaces_rather_than_duplicates(self):
        store.put("k3", _report())
        updated = _report()
        updated.rating = "SELL"
        store.put("k3", updated)
        assert store.get("k3").rating == "SELL"


class TestBlockedReportsAreNeverServed:
    """A cache must not become the way a QA block gets reversed."""

    def test_a_blocked_report_is_not_returned(self):
        store.put("blocked", _report(blocked=True))
        assert store.get("blocked") is None

    def test_a_clean_report_is_returned(self):
        store.put("clean", _report(blocked=False))
        assert store.get("clean") is not None


class TestOperations:
    def test_invalidate_drops_a_companys_reports(self):
        store.put("k4", _report())
        assert store.invalidate("BIKAJI.NS") == 1
        assert store.get("k4") is None

    def test_stats_counts_reports_and_serves(self):
        store.put("k5", _report(), cost_usd=0.20)
        store.get("k5")
        store.get("k5")

        stats = store.stats()
        assert stats["reports"] == 1
        assert stats["serves"] == 2
        assert stats["spent_usd"] == pytest.approx(0.20)
        # Two serves of a $0.20 report is $0.40 of generation avoided.
        assert stats["saved_usd"] == pytest.approx(0.40)

    def test_an_unreadable_payload_is_a_miss_not_a_crash(self):
        store.put("k6", _report())
        with store._connect() as connection:
            connection.execute(
                "UPDATE reports SET payload = ? WHERE fingerprint = ?",
                ("{not json", "k6"),
            )
        assert store.get("k6") is None

    def test_unknown_fields_in_a_stored_payload_are_ignored(self):
        """A report written by an older build must not crash a newer one."""
        store.put("k7", _report())
        with store._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM reports WHERE fingerprint = 'k7'"
            ).fetchone()
        import json
        data = json.loads(row["payload"])
        data["a_field_that_no_longer_exists"] = 123
        with store._connect() as connection:
            connection.execute(
                "UPDATE reports SET payload = ? WHERE fingerprint = 'k7'",
                (json.dumps(data),),
            )
        assert store.get("k7") is not None
