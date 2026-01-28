"""Shared helpers for Task App proxy endpoints (OpenAI, Groq, etc.).

The proxy is tool-agnostic - each task app provides its own tools schema.
"""

from __future__ import annotations

from typing import Any

import synth_ai_py


def prepare_for_openai(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitise an OpenAI chat completions payload for Task App usage.

    The task app is responsible for providing tools in the payload.
    This function only handles model-specific parameter normalization.
    """

    return synth_ai_py.localapi_prepare_for_openai(payload, model)


def prepare_for_groq(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Groq uses the OpenAI schema; reuse most normalisation rules."""

    return synth_ai_py.localapi_prepare_for_groq(payload, model)


def normalize_response_format_for_groq(model: str | None, payload: dict[str, Any]) -> None:
    """Normalize response_format for Groq provider compatibility."""

    normalized = synth_ai_py.localapi_normalize_response_format_for_groq(payload, model)
    if isinstance(normalized, dict):
        payload.clear()
        payload.update(normalized)


def inject_system_hint(payload: dict[str, Any], hint: str) -> dict[str, Any]:
    """Insert or augment a system message with the provided hint (idempotent)."""

    return synth_ai_py.localapi_inject_system_hint(payload, hint)


def extract_message_text(message: Any) -> str:
    """Return best-effort text from an OpenAI-style message structure."""

    return synth_ai_py.localapi_extract_message_text(message)


def parse_tool_call_from_text(text: str) -> tuple[list[str], str]:
    """Derive tool-call actions and reasoning from assistant text."""

    return synth_ai_py.localapi_parse_tool_call_from_text(text)


def synthesize_tool_call_if_missing(
    openai_response: dict[str, Any], fallback_tool_name: str = "interact"
) -> dict[str, Any]:
    """Ensure the first choice carries a tool_call derived from text if absent."""

    return synth_ai_py.localapi_synthesize_tool_call_if_missing(openai_response, fallback_tool_name)
