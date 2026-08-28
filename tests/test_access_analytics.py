"""
Access control and usage analytics.

The load-bearing properties here are security ones, so they are tested as such:
a missing access code must fail closed rather than open, a visitor's API key
must never reach the analytics database, and a guest paying with their own key
must not be metered against the owner's budget.
"""

from __future__ import annotations

import json

import pytest

from fincopilot import access, analytics, config


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

class TestAccessCode:
    def test_the_correct_code_grants_access(self, monkeypatch):
        monkeypatch.setattr(config, "ACCESS_CODE", "TESTCODE123")
        grant = access.grant_from_code("TESTCODE123")
        assert grant is not None
        assert grant.mode == access.MODE_ACCESS_CODE

    def test_a_wrong_code_is_refused(self, monkeypatch):
        monkeypatch.setattr(config, "ACCESS_CODE", "TESTCODE123")
        assert access.grant_from_code("ALK2005") is None
        assert access.grant_from_code("") is None

    def test_surrounding_whitespace_is_tolerated(self, monkeypatch):
        """People paste codes out of chat apps."""
        monkeypatch.setattr(config, "ACCESS_CODE", "TESTCODE123")
        assert access.grant_from_code("  TESTCODE123 ") is not None

    def test_the_code_is_case_sensitive(self, monkeypatch):
        monkeypatch.setattr(config, "ACCESS_CODE", "TESTCODE123")
        assert access.grant_from_code("testcode123") is None

    def test_an_unconfigured_deployment_fails_closed(self, monkeypatch):
        """The critical one. No code set must not mean 'anything works'."""
        monkeypatch.setattr(config, "ACCESS_CODE", "")
        assert not access.access_code_configured()
        assert access.grant_from_code("") is None
        assert access.grant_from_code("TESTCODE123") is None
        assert access.grant_from_code("anything at all") is None

    def test_no_code_is_baked_into_the_source(self):
        """The repository is public; a literal code here would be harvestable
        and would authorise spending on the owner's account."""
        from pathlib import Path

        source = Path(access.__file__).read_text(encoding="utf-8")
        assert "TESTCODE123" not in source
        # It must come from the environment, with no fallback value.
        assert 'get_secret("ACCESS_CODE"' in Path(config.__file__).read_text(
            encoding="utf-8"
        )


class TestOwnKey:
    def test_a_well_formed_key_grants_access(self):
        grant = access.grant_from_key("sk-" + "a" * 32)
        assert grant is not None
        assert grant.mode == access.MODE_OWN_KEY
        assert grant.api_key == "sk-" + "a" * 32

    @pytest.mark.parametrize(
        "bad", ["", "   ", "hello", "pk-abc", "sk-short", "TESTCODE123"]
    )
    def test_malformed_keys_are_refused(self, bad):
        assert access.grant_from_key(bad) is None


class TestWhoPays:
    """Rate limits and the spend ceiling protect the OWNER's credits only."""

    def test_an_access_code_session_spends_owner_credits(self):
        assert access.Grant(mode=access.MODE_ACCESS_CODE).uses_owner_credits

    def test_an_own_key_session_does_not(self):
        grant = access.Grant(mode=access.MODE_OWN_KEY, api_key="sk-" + "a" * 32)
        assert not grant.uses_owner_credits

    def test_an_access_code_grant_never_carries_a_key(self):
        """The owner's key belongs to the process, never to a session object."""
        assert access.grant_from_code and True
        grant = access.Grant(mode=access.MODE_ACCESS_CODE)
        assert grant.api_key is None


class TestSessionIds:
    def test_ids_are_unique_and_opaque(self):
        ids = {access.new_session_id() for _ in range(200)}
        assert len(ids) == 200
        assert all(len(i) == 16 for i in ids)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _temp_analytics_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ANALYTICS_DB_PATH", str(tmp_path / "usage.db"))
    monkeypatch.setattr(config, "ANALYTICS_ENABLED", True)
    monkeypatch.setattr(config, "ANALYTICS_STORE_QUESTION_TEXT", True)


class TestEventRecording:
    def test_an_event_round_trips(self):
        analytics.record(
            analytics.COMPANY_LOAD, session_id="s1", access_mode="access_code",
            ticker="BIKAJI.NS", company="Bikaji", ok=True, duration_ms=1200,
            cost_usd=0.02, detail={"passages": 5483},
        )
        events = analytics.recent()
        assert len(events) == 1
        assert events[0]["ticker"] == "BIKAJI.NS"
        assert json.loads(events[0]["detail"])["passages"] == 5483

    def test_failures_are_queryable_on_their_own(self):
        analytics.record(analytics.COMPANY_LOAD, session_id="s1", ok=True)
        analytics.record(
            analytics.COMPANY_LOAD, session_id="s2", ticker="XYZ", ok=False,
            detail={"reason": "no_documents_indexed"},
        )
        failed = analytics.failures()
        assert len(failed) == 1
        assert failed[0]["ticker"] == "XYZ"

    def test_the_summary_counts_each_kind(self):
        analytics.record(analytics.COMPANY_LOAD, session_id="s1", ticker="A")
        analytics.record(analytics.COMPANY_LOAD, session_id="s2", ticker="A")
        analytics.record(analytics.QUESTION, session_id="s1")
        analytics.record(analytics.REPORT, session_id="s1", cost_usd=0.15)

        stats = analytics.summary()
        assert stats["sessions"] == 2
        assert stats["companies_loaded"] == 2
        assert stats["questions"] == 1
        assert stats["reports"] == 1
        assert stats["cost_usd"] == pytest.approx(0.15)
        assert stats["top_companies"][0] == ("A", 2)

    def test_disabling_analytics_writes_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "ANALYTICS_ENABLED", False)
        analytics.record(analytics.QUESTION, session_id="s1")
        assert analytics.recent() == []

    def test_a_broken_database_never_raises(self, monkeypatch):
        """Diagnostics must not become the failure they were meant to observe."""
        monkeypatch.setattr(config, "ANALYTICS_DB_PATH", "/nonexistent\x00/bad.db")
        analytics.record(analytics.QUESTION, session_id="s1")   # must not raise
        assert analytics.summary()["sessions"] == 0
        assert analytics.recent() == []


class TestQuestionLogging:
    def test_question_text_is_stored_when_enabled(self):
        analytics.record_question(
            "What drove the margin decline?", session_id="s1",
            ticker="X", citations=3,
        )
        detail = json.loads(analytics.recent()[0]["detail"])
        assert detail["question"] == "What drove the margin decline?"
        assert detail["citations"] == 3

    def test_text_is_withheld_when_disabled_but_shape_is_kept(self, monkeypatch):
        monkeypatch.setattr(config, "ANALYTICS_STORE_QUESTION_TEXT", False)
        analytics.record_question("What drove the margin decline?", session_id="s1")
        detail = json.loads(analytics.recent()[0]["detail"])
        assert "question" not in detail
        assert detail["length"] == len("What drove the margin decline?")

    def test_a_key_pasted_into_the_chat_box_is_never_stored(self):
        """The failure that would matter most: a visitor pastes their key into
        the wrong field and it lands in a database on someone else's server."""
        analytics.record_question(
            "here is my key sk-abcdefghijklmnopqrstuvwxyz0123 use it",
            session_id="s1",
        )
        stored = analytics.recent()[0]["detail"]
        assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in stored
        assert "redacted" in stored

    def test_personal_identifiers_are_scrubbed_from_questions(self):
        analytics.record_question("does PAN ABCDE1234F appear?", session_id="s1")
        assert "ABCDE1234F" not in analytics.recent()[0]["detail"]

    def test_an_uncited_answer_is_recorded_as_a_failure(self):
        """The signal worth having: retrieval missed, or the filings genuinely
        do not answer it."""
        analytics.record_question("obscure question", session_id="s1",
                                  ok=False, citations=0)
        assert len(analytics.failures()) == 1


class TestNoCredentialsEverStored:
    def test_the_event_schema_has_no_column_for_a_key(self):
        """Structural guarantee: there is no field a key could occupy."""
        analytics.record(analytics.QUESTION, session_id="s1")
        columns = set(analytics.recent()[0].keys())
        assert not any(
            k in columns for k in ("api_key", "key", "secret", "token", "password")
        )
