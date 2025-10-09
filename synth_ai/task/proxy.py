"""Shared helpers for Task App proxy endpoints (OpenAI, Groq, etc.)."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from typing import Any

INTERACT_TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "interact",
            "description": "Perform one or more environment actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of environment actions to execute in order.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Optional reasoning for the chosen actions.",
                    },
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
        },
    }
]

_REMOVE_FIELDS = {
    "stop_after_tool_calls",
    "thinking_mode",
    "thinking_budget",
    "reasoning",
}
_REMOVE_SAMPLING_FIELDS = {"temperature", "top_p"}
_GPT5_MIN_COMPLETION_TOKENS = 16000


def _ensure_tools(payload: dict[str, Any]) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        payload["tools"] = copy.deepcopy(INTERACT_TOOL_SCHEMA)


def prepare_for_openai(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitise an OpenAI chat completions payload for Task App usage."""

    sanitized = copy.deepcopy(payload)
    for field in _REMOVE_FIELDS:
        sanitized.pop(field, None)

    if model and "gpt-5" in model:
        max_tokens = sanitized.pop("max_tokens", None)
        if "max_completion_tokens" not in sanitized and isinstance(max_tokens, int):
            sanitized["max_completion_tokens"] = max_tokens
        elif max_tokens is not None:
            sanitized.setdefault("max_completion_tokens", max_tokens)
        for field in _REMOVE_SAMPLING_FIELDS:
            sanitized.pop(field, None)
        mct = sanitized.get("max_completion_tokens")
        if not isinstance(mct, int) or mct < _GPT5_MIN_COMPLETION_TOKENS:
            sanitized["max_completion_tokens"] = _GPT5_MIN_COMPLETION_TOKENS
        sanitized["tool_choice"] = {"type": "function", "function": {"name": "interact"}}
        sanitized["parallel_tool_calls"] = False

    _ensure_tools(sanitized)
    return sanitized


def prepare_for_groq(model: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Groq uses the OpenAI schema; reuse most normalisation rules."""

    sanitized = prepare_for_openai(model, payload)
    # Groq supports `max_tokens`; prefer their native parameter when present
    if (
        model
        and "gpt-5" not in model
        and "max_completion_tokens" in sanitized
        and "max_tokens" not in payload
    ):
        sanitized["max_tokens"] = sanitized.pop("max_completion_tokens")
    return sanitized


def inject_system_hint(payload: dict[str, Any], hint: str) -> dict[str, Any]:
    """Insert or augment a system message with the provided hint (idempotent)."""

    if not hint:
        return payload
    cloned = copy.deepcopy(payload)
    messages = cloned.get("messages")
    if not isinstance(messages, list):
        return cloned
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, str) and hint not in content:
            messages[0] = dict(messages[0])
            messages[0]["content"] = content.rstrip() + ("\n\n" if content else "") + hint
    else:
        messages.insert(0, {"role": "system", "content": hint})
    cloned["messages"] = messages
    return cloned


def extract_message_text(message: Any) -> str:
    """Return best-effort text from an OpenAI-style message structure."""

    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = [extract_message_text(part) for part in message]
        return "\n".join(part for part in parts if part)
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                text = extract_message_text(item)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        if "text" in message and isinstance(message["text"], str):
            return message["text"]
    return str(message)


def _parse_actions_from_json_candidate(candidate: Any) -> tuple[list[str], str]:
    actions: list[str] = []
    reasoning = ""
    if isinstance(candidate, dict):
        potential_actions = candidate.get("actions")
        if isinstance(potential_actions, list):
            actions = [str(a).strip() for a in potential_actions if str(a).strip()]
        elif isinstance(potential_actions, str):
            actions = [a.strip() for a in potential_actions.split(";") if a.strip()]
        if "reasoning" in candidate and isinstance(candidate["reasoning"], str):
            reasoning = candidate["reasoning"].strip()
    return actions, reasoning


def parse_tool_call_from_text(text: str) -> tuple[list[str], str]:
    """Derive tool-call actions and reasoning from assistant text."""

    text = (text or "").strip()
    if not text:
        return [], ""

    # Attempt to interpret the entire payload as JSON
    try:
        data = json.loads(text)
    except Exception:
        data = None
    else:
        actions, reasoning = _parse_actions_from_json_candidate(data)
        if actions:
            return actions, reasoning or text

    # Look for embedded JSON objects containing an "actions" field
    json_like_matches = re.findall(r"\{[^{}]*actions[^{}]*\}", text, flags=re.IGNORECASE)
    for fragment in json_like_matches:
        try:
            data = json.loads(fragment)
        except Exception:
            continue
        actions, reasoning = _parse_actions_from_json_candidate(data)
        if actions:
            return actions, reasoning or text

    # Patterns like "Actions: move_right, jump"
    m = re.search(r"actions?\s*:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        items = [part.strip() for part in m.group(1).split(",") if part.strip()]
        if items:
            reasoning = text[: m.start()].strip()
            return items, reasoning

    # Patterns like "Action 1: move_right"
    actions: list[str] = []
    reasoning_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"action\s*\d*\s*[:\-]\s*(.+)", stripped, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                actions.append(candidate)
        else:
            reasoning_lines.append(stripped)
    if actions:
        return actions, "\n".join(reasoning_lines).strip()

    return [], text


def _build_tool_call(actions: Iterable[str], reasoning: str) -> dict[str, Any]:
    payload = {
        "actions": [str(a).strip() for a in actions if str(a).strip()],
    }
    if reasoning.strip():
        payload["reasoning"] = reasoning.strip()
    return {
        "id": "tool_interact_fallback",
        "type": "function",
        "function": {
            "name": INTERACT_TOOL_SCHEMA[0]["function"]["name"],
            "arguments": json.dumps(payload, ensure_ascii=False),
        },
    }


def synthesize_tool_call_if_missing(openai_response: dict[str, Any]) -> dict[str, Any]:
    """Ensure the first choice carries a tool_call derived from text if absent."""

    if not isinstance(openai_response, dict):
        return openai_response
    choices = openai_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return openai_response
    first = choices[0]
    if not isinstance(first, dict):
        return openai_response
    message = first.get("message")
    if not isinstance(message, dict):
        return openai_response
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return openai_response

    text = extract_message_text(message)
    actions, reasoning = parse_tool_call_from_text(text)
    if not actions:
        return openai_response

    new_message = copy.deepcopy(message)
    new_message["tool_calls"] = [_build_tool_call(actions, reasoning)]
    if "content" not in new_message:
        new_message["content"] = None

    new_first = copy.deepcopy(first)
    new_first["message"] = new_message
    new_choices = [new_first] + choices[1:]

    result = copy.deepcopy(openai_response)
    result["choices"] = new_choices
    return result
