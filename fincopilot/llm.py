"""
The single place model calls happen: routing, fallback, guardrails, cost.

Calls go through a **LiteLLM Router** rather than a provider SDK. That buys
three things this application needs to be usable by people rather than
demonstrated to them:

* **Fallback across providers.** An OpenAI incident used to mean every call
  returned ``None``, the report silently lost sections, and the QA gate blocked
  it. Now the router crosses to Gemini and the work completes. Fallback is
  *between* groups: only when every deployment in the primary group has failed
  or is cooling down.

* **Load balancing within a provider.** Extra deployments registered under the
  same group name — another key, another region, an Azure mirror — are spread
  across automatically, each respecting its own rpm ceiling, so traffic moves
  before the provider starts returning 429s. Adding capacity is a config
  change, not a code change.

* **One chokepoint for safety and spend.** Every prompt is scanned before it
  leaves the process and every response before it is used, and the real dollar
  cost of each call is metered against a hard ceiling. Because there is exactly
  one function that talks to a model, none of that can be bypassed by a new
  call site.

The public surface — :func:`complete` and :func:`complete_json` — is unchanged
from the direct-SDK version, so no caller needed to know any of this happened.

Model selection stays expressed in plain model names at the call sites
(``model=config.WRITER_MODEL``); they are mapped to router groups here. A name
with no mapping is passed through to LiteLLM untouched, which keeps one-off
experiments possible without editing this module.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from . import config, guardrails

log = logging.getLogger(__name__)

_router = None
_router_failed = False

# Per-process usage, so a report's cost is observable rather than inferred.
_usage: dict[str, float] = {
    "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
    "cost_usd": 0.0, "fallbacks": 0, "guardrail_findings": 0,
}


def reset_usage() -> None:
    _usage.update(
        calls=0, prompt_tokens=0, completion_tokens=0,
        cost_usd=0.0, fallbacks=0, guardrail_findings=0,
    )
    guardrails.reset_spend()


def get_usage() -> dict:
    return dict(_usage)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _deployment(group: str, model: str, api_key: str, rpm: int) -> dict:
    return {
        "model_name": group,
        "litellm_params": {
            "model": model,
            "api_key": api_key,
            "rpm": rpm,
            "timeout": config.ROUTER_TIMEOUT_SECONDS,
        },
    }


def _model_list() -> list[dict]:
    """Deployments to register, given the keys actually present.

    A missing Gemini key is not an error: the router is built OpenAI-only and
    simply has nothing to fall back to. That keeps the app runnable with one
    provider configured, which is how most people will first run it.
    """
    deployments: list[dict] = []

    openai_key = config.get_secret("OPENAI_API_KEY")
    if openai_key:
        deployments += [
            _deployment(config.FAST_GROUP, f"openai/{config.FAST_MODEL}",
                        openai_key, config.OPENAI_RPM),
            _deployment(config.WRITER_GROUP, f"openai/{config.WRITER_MODEL}",
                        openai_key, config.OPENAI_RPM),
        ]

    gemini_key = config.get_secret("GEMINI_API_KEY")
    if gemini_key:
        deployments += [
            _deployment(config.FAST_FALLBACK_GROUP, config.FALLBACK_FAST_MODEL,
                        gemini_key, config.GEMINI_RPM),
            _deployment(config.WRITER_FALLBACK_GROUP, config.FALLBACK_WRITER_MODEL,
                        gemini_key, config.GEMINI_RPM),
        ]
    else:
        log.info(
            "GEMINI_API_KEY is not set; running without a cross-provider "
            "fallback. Set it to make an OpenAI outage survivable."
        )

    return deployments


def _fallback_map() -> list[dict]:
    """Primary group -> ordered fallback groups, for groups that exist."""
    registered = {d["model_name"] for d in _model_list()}
    pairs = [
        (config.FAST_GROUP, config.FAST_FALLBACK_GROUP),
        (config.WRITER_GROUP, config.WRITER_FALLBACK_GROUP),
    ]
    return [
        {primary: [fallback]}
        for primary, fallback in pairs
        if primary in registered and fallback in registered
    ]


def router():
    """The process-wide router, built on first use.

    Built lazily and cached, including the failure case: constructing it needs
    a key, and the test suite and offline tooling import this module without
    one. A failed build is remembered so every subsequent call does not retry
    the same import.
    """
    global _router, _router_failed

    if _router is not None or _router_failed:
        return _router

    deployments = _model_list()
    if not deployments:
        log.error("no LLM provider keys configured; model calls will fail")
        _router_failed = True
        return None

    try:
        import litellm
        from litellm import Router

        # LiteLLM logs the selected deployment dict at INFO on every single
        # call. That is one long line per model call — and a report makes
        # dozens — which buries our own logs. Failures still surface: the
        # router raises, and this module logs fallbacks at WARNING.
        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)

        _router = Router(
            model_list=deployments,
            fallbacks=_fallback_map(),
            routing_strategy=config.ROUTER_STRATEGY,
            num_retries=config.ROUTER_NUM_RETRIES,
            timeout=config.ROUTER_TIMEOUT_SECONDS,
            allowed_fails=config.ROUTER_ALLOWED_FAILS,
            cooldown_time=config.ROUTER_COOLDOWN_SECONDS,
            # Retrying a 400 or a 401 just burns latency: the request is wrong
            # or the key is, and neither improves on a second attempt.
            retry_policy=None,
        )
        log.info(
            "LLM router ready: %d deployments, %d fallback routes, strategy=%s",
            len(deployments), len(_fallback_map()), config.ROUTER_STRATEGY,
        )
    except Exception as exc:
        log.error("could not build the LLM router: %s", exc)
        _router_failed = True
        _router = None

    return _router


# The model each group is *expected* to be served by, used to detect a fallback.
_PRIMARY_MODEL = {
    config.FAST_GROUP: config.FAST_MODEL,
    config.WRITER_GROUP: config.WRITER_MODEL,
}


def _group_for(model: str | None) -> str:
    """Map a plain model name to its router group."""
    name = model or config.FAST_MODEL
    return {
        config.FAST_MODEL: config.FAST_GROUP,
        config.WRITER_MODEL: config.WRITER_GROUP,
    }.get(name, name)


def _record(response, group: str) -> None:
    usage = getattr(response, "usage", None)
    _usage["calls"] += 1
    if usage is not None:
        _usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    # Real cost, from the model that actually served the request — which after
    # a fallback is not the one that was asked for.
    try:
        from litellm import completion_cost

        cost = completion_cost(completion_response=response) or 0.0
    except Exception:
        cost = 0.0

    _usage["cost_usd"] += cost
    guardrails.record_spend(cost)

    # Did the primary serve this, or did we cross to the fallback provider?
    # Compared against the configured primary *model* rather than the group
    # name — a provider returns "gpt-4.1-mini-2025-04-14", which never contains
    # the group name, so comparing to the group counts every call as a
    # fallback. A run served largely by the fallback is a useful signal that
    # the primary provider is degraded even when nothing failed outright.
    served_by = getattr(response, "model", "") or ""
    primary = _PRIMARY_MODEL.get(group)
    if served_by and primary and not served_by.startswith(primary):
        _usage["fallbacks"] += 1
        log.warning(
            "call for %s fell back to %s (primary is %s)", group, served_by, primary
        )


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = config.TEMPERATURE_FACTUAL,
    max_tokens: int | None = None,
    json_mode: bool = False,
    retries: int = 3,          # kept for signature compatibility; see below
) -> str | None:
    """Single-turn completion. Returns ``None`` if the call cannot be served.

    Retries and provider failover are the router's job now, so ``retries`` is
    accepted but not used to loop here — a second loop on top of the router's
    would multiply attempts and defeat the cooldown that protects a struggling
    provider.
    """
    group = _group_for(model)

    # Outbound: never send a secret or a person's identifiers to a provider.
    scanned_prompt = guardrails.scan_outbound(prompt)
    scanned_system = guardrails.scan_outbound(system or "")
    findings = scanned_prompt.findings + scanned_system.findings
    if findings:
        _usage["guardrail_findings"] += len(findings)

    messages = []
    if scanned_system.text:
        messages.append({"role": "system", "content": scanned_system.text})
    messages.append({"role": "user", "content": scanned_prompt.text})

    kwargs: dict[str, Any] = {
        "model": group,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    active = router()
    if active is None:
        return None

    # A hard stop, not a warning: past the ceiling the next call is refused.
    guardrails.enforce_budget()

    try:
        response = active.completion(**kwargs)
    except Exception as exc:
        log.error("LLM call failed after routing and fallback: %s", exc)
        return None

    _record(response, group)

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError):
        log.error("malformed response from %s", group)
        return None

    if not content:
        return None

    checked = guardrails.scan_response(content)
    if checked.findings:
        _usage["guardrail_findings"] += len(checked.findings)
    return checked.text


def complete_json(prompt: str, **kwargs) -> Any | None:
    """Completion parsed as JSON, tolerant of the usual formatting noise.

    Even in JSON mode a model occasionally wraps output in a markdown fence or
    adds a sentence of preamble — and a fallback provider's JSON mode is not
    always as strict as the primary's, which makes this more important now, not
    less. Rather than discarding an otherwise good response, the payload is
    located and extracted.
    """
    kwargs.setdefault("json_mode", True)
    raw = complete(prompt, **kwargs)
    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to the outermost JSON object or array in the response.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue

    log.warning("could not parse JSON from model response: %.200s", raw)
    return None
