"""Shared helpers for Container proxy endpoints (OpenAI, Groq, etc.).

The proxy is tool-agnostic - each container provides its own tools schema.
"""

from __future__ import annotations

from typing import Any

try:
    import synth_ai_py
except Exception:  # pragma: no cover
    synth_ai_py = None


def prepare_for_openai(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitise an OpenAI chat completions payload for Container usage.

    The container is responsible for providing tools in the payload.
    This function only handles model-specific parameter normalization.
    """

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_prepare_for_openai"):
        return synth_ai_py.container_prepare_for_openai(payload, model)
    return payload


def prepare_for_groq(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Groq uses the OpenAI schema; reuse most normalisation rules."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_prepare_for_groq"):
        return synth_ai_py.container_prepare_for_groq(payload, model)
    return payload


def normalize_response_format_for_groq(model: str | None, payload: dict[str, Any]) -> None:
    """Normalize response_format for Groq provider compatibility."""

    if synth_ai_py is not None and hasattr(
        synth_ai_py, "container_normalize_response_format_for_groq"
    ):
        normalized = synth_ai_py.container_normalize_response_format_for_groq(payload, model)
        if isinstance(normalized, dict):
            payload.clear()
            payload.update(normalized)


def inject_system_hint(payload: dict[str, Any], hint: str) -> dict[str, Any]:
    """Insert or augment a system message with the provided hint (idempotent)."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_inject_system_hint"):
        return synth_ai_py.container_inject_system_hint(payload, hint)

    if not hint:
        return payload
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str) and hint not in content:
                    msg["content"] = f"{content}\n{hint}" if content else hint
                return payload
        messages.insert(0, {"role": "system", "content": hint})
    return payload


def extract_message_text(message: Any) -> str:
    """Return best-effort text from an OpenAI-style message structure."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_extract_message_text"):
        return synth_ai_py.container_extract_message_text(message)
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    if isinstance(message, str):
        return message
    return ""


def parse_tool_call_from_text(text: str) -> tuple[list[str], str]:
    """Derive tool-call actions and reasoning from assistant text."""

    if synth_ai_py is not None and hasattr(synth_ai_py, "container_parse_tool_call_from_text"):
        return synth_ai_py.container_parse_tool_call_from_text(text)
    return ([], text)


def synthesize_tool_call_if_missing(
    openai_response: dict[str, Any], fallback_tool_name: str = "interact"
) -> dict[str, Any]:
    """Ensure the first choice carries a tool_call derived from text if absent."""

    if synth_ai_py is not None and hasattr(
        synth_ai_py, "container_synthesize_tool_call_if_missing"
    ):
        return synth_ai_py.container_synthesize_tool_call_if_missing(
            openai_response, fallback_tool_name
        )

    choices = openai_response.get("choices") if isinstance(openai_response, dict) else None
    if not isinstance(choices, list) or not choices:
        return openai_response
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return openai_response
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return openai_response
    content = message.get("content")
    message["tool_calls"] = [
        {
            "id": "fallback",
            "type": "function",
            "function": {"name": fallback_tool_name, "arguments": "{}"},
        }
    ]
    if content is not None:
        message.setdefault("content", content)
    return openai_response
