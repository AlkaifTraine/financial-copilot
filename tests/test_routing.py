"""
Router construction: deployments, load-balancing groups, and fallback wiring.

These tests build the router's *configuration* without making a network call.
What they pin is the part that is easy to get silently wrong: that a Gemini key
produces real fallback routes, that its absence degrades to OpenAI-only instead
of raising, and — most importantly — that the two providers never end up in the
same load-balancing group, which would quietly serve half a report's prose from
a different model.
"""

from __future__ import annotations

from fincopilot import config, llm


class TestDeploymentRegistration:
    def test_both_keys_register_primary_and_fallback_groups(self, monkeypatch):
        monkeypatch.setattr(
            config, "get_secret",
            lambda name, default=None: {
                "OPENAI_API_KEY": "sk-test", "GEMINI_API_KEY": "gem-test"
            }.get(name, default),
        )
        groups = {d["model_name"] for d in llm._model_list()}
        assert groups == {
            config.FAST_GROUP, config.WRITER_GROUP,
            config.FAST_FALLBACK_GROUP, config.WRITER_FALLBACK_GROUP,
        }

    def test_without_a_gemini_key_it_runs_openai_only(self, monkeypatch):
        monkeypatch.setattr(
            config, "get_secret",
            lambda name, default=None: (
                "sk-test" if name == "OPENAI_API_KEY" else default
            ),
        )
        groups = {d["model_name"] for d in llm._model_list()}
        assert groups == {config.FAST_GROUP, config.WRITER_GROUP}
        assert llm._fallback_map() == []

    def test_no_keys_registers_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "get_secret", lambda name, default=None: default)
        assert llm._model_list() == []

    def test_each_deployment_carries_a_rate_ceiling_and_timeout(self, monkeypatch):
        monkeypatch.setattr(
            config, "get_secret",
            lambda name, default=None: {
                "OPENAI_API_KEY": "sk-test", "GEMINI_API_KEY": "gem-test"
            }.get(name, default),
        )
        for deployment in llm._model_list():
            params = deployment["litellm_params"]
            assert params["rpm"] > 0
            assert params["timeout"] == config.ROUTER_TIMEOUT_SECONDS
            assert params["api_key"]


class TestProviderSeparation:
    """Load balancing is within a provider; fallback is across providers."""

    def test_openai_and_gemini_are_never_in_the_same_group(self, monkeypatch):
        monkeypatch.setattr(
            config, "get_secret",
            lambda name, default=None: {
                "OPENAI_API_KEY": "sk-test", "GEMINI_API_KEY": "gem-test"
            }.get(name, default),
        )
        by_group: dict[str, set[str]] = {}
        for deployment in llm._model_list():
            provider = deployment["litellm_params"]["model"].split("/")[0]
            by_group.setdefault(deployment["model_name"], set()).add(provider)

        for group, providers in by_group.items():
            assert len(providers) == 1, (
                f"group {group} mixes providers {providers}; a report's prose "
                f"would be split across vendors run to run"
            )

    def test_fallback_routes_point_primary_to_the_matching_tier(self, monkeypatch):
        monkeypatch.setattr(
            config, "get_secret",
            lambda name, default=None: {
                "OPENAI_API_KEY": "sk-test", "GEMINI_API_KEY": "gem-test"
            }.get(name, default),
        )
        routes = {k: v for mapping in llm._fallback_map() for k, v in mapping.items()}
        assert routes[config.FAST_GROUP] == [config.FAST_FALLBACK_GROUP]
        assert routes[config.WRITER_GROUP] == [config.WRITER_FALLBACK_GROUP]


class TestModelGroupMapping:
    def test_plain_model_names_map_to_their_groups(self):
        assert llm._group_for(config.FAST_MODEL) == config.FAST_GROUP
        assert llm._group_for(config.WRITER_MODEL) == config.WRITER_GROUP

    def test_default_is_the_cheap_tier(self):
        assert llm._group_for(None) == config.FAST_GROUP

    def test_an_unmapped_name_passes_through(self):
        """So a one-off experiment does not need this module edited."""
        assert llm._group_for("openai/o3-mini") == "openai/o3-mini"


class TestUsageAccounting:
    def test_reset_clears_usage_and_spend(self):
        llm._usage["calls"] = 7
        llm._usage["cost_usd"] = 1.25
        llm.reset_usage()
        usage = llm.get_usage()
        assert usage["calls"] == 0
        assert usage["cost_usd"] == 0.0

    def test_usage_reports_fallback_and_guardrail_counters(self):
        llm.reset_usage()
        assert "fallbacks" in llm.get_usage()
        assert "guardrail_findings" in llm.get_usage()


class TestFailureHandling:
    def test_complete_returns_none_without_a_provider(self, monkeypatch):
        """No key configured must degrade, not raise, at every call site."""
        monkeypatch.setattr(llm, "_router", None)
        monkeypatch.setattr(llm, "_router_failed", False)
        monkeypatch.setattr(config, "get_secret", lambda name, default=None: default)

        assert llm.complete("anything") is None
        assert llm.complete_json("anything") is None
