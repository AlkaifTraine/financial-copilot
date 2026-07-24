"""
Thin LLM helper.

Calls the OpenAI SDK directly rather than going through LangChain. The project
uses the model in four narrow ways — rewrite a query, score relevance, extract
structured JSON, write a report section — and none of them need chains, agents
or memory abstractions. Calling the API directly means fewer moving parts, no
breakage when a wrapper's interface shifts between releases, and a stack trace
that points at our own code.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from . import config

log = logging.getLogger(__name__)

_client = None


def client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=config.require_secret("OPENAI_API_KEY"))
    return _client


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = config.TEMPERATURE_FACTUAL,
    max_tokens: int | None = None,
    json_mode: bool = False,
    retries: int = 3,
) -> str | None:
    """Single-turn completion. Returns ``None`` if every attempt fails."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model or config.FAST_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            response = client().chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            if attempt == retries - 1:
                log.error("LLM call failed permanently: %s", exc)
                return None
            time.sleep(2**attempt)
    return None


def complete_json(prompt: str, **kwargs) -> Any | None:
    """Completion parsed as JSON, tolerant of the usual formatting noise.

    Even in JSON mode a model occasionally wraps output in a markdown fence or
    adds a sentence of preamble. Rather than discarding an otherwise good
    response — which is what the previous `json.loads` in
    `utils/financial_data_extractor.py` did, silently returning `{}` — the
    payload is located and extracted.
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
