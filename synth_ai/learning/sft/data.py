from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SFTMessageContent = str | dict[str, Any] | list[Any] | None


class SFTDataError(ValueError):
    """Raised when a JSONL record cannot be coerced into an SFTExample."""


@dataclass(slots=True)
class SFTToolDefinition:
    name: str
    description: str | None
    parameters: dict[str, Any] | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTToolCall:
    name: str
    arguments: Any
    call_id: str | None = None
    type: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTMessage:
    role: str
    content: SFTMessageContent
    tool_calls: list[SFTToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SFTExample:
    messages: list[SFTMessage]
    tools: list[SFTToolDefinition] = field(default_factory=list)
    tool_choice: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _parse_tool_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _coerce_tool_definition(raw: Any, *, index: int) -> SFTToolDefinition:
    if not isinstance(raw, dict):
        raise SFTDataError(f"tool {index} is not an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SFTDataError(f"tool {index} missing name")
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise SFTDataError(f"tool {index} description must be a string if present")
    parameters = raw.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise SFTDataError(f"tool {index} parameters must be an object if present")
    return SFTToolDefinition(
        name=name, description=description, parameters=parameters, raw=dict(raw)
    )


def _coerce_tool_call(raw: Any, *, index: int) -> SFTToolCall:
    if not isinstance(raw, dict):
        raise SFTDataError(f"tool_call {index} is not an object")

    call_id = raw.get("id")
    call_type = raw.get("type")

    fn_payload: dict[str, Any] | None = None
    name: str | None = None
    arguments: Any = None

    if isinstance(raw.get("function"), dict):
        fn_payload = raw["function"]
        name = fn_payload.get("name") if isinstance(fn_payload.get("name"), str) else None
        arguments = fn_payload.get("arguments")
    if name is None:
        maybe_name = raw.get("name")
        if isinstance(maybe_name, str):
            name = maybe_name
            arguments = raw.get("arguments")

    if not isinstance(name, str) or not name.strip():
        raise SFTDataError(f"tool_call {index} missing function name")

    parsed_arguments = _parse_tool_arguments(arguments)

    normalized_id = None
    if call_id is not None:
        normalized_id = str(call_id)
    normalized_type = None
    if call_type is not None:
        normalized_type = str(call_type)

    return SFTToolCall(
        name=name,
        arguments=parsed_arguments,
        call_id=normalized_id,
        type=normalized_type,
        raw=dict(raw),
    )


def _coerce_message(raw: Any, *, index: int) -> SFTMessage:
    if not isinstance(raw, dict):
        raise SFTDataError(f"message {index} is not an object")
    role = raw.get("role")
    if not isinstance(role, str) or not role.strip():
        raise SFTDataError(f"message {index} has invalid role")

    content = raw.get("content")
    if content is not None and not isinstance(content, str | list | dict):
        raise SFTDataError(f"message {index} has unsupported content type {type(content).__name__}")

    raw_tool_calls = raw.get("tool_calls")
    tool_calls: list[SFTToolCall] = []
    if raw_tool_calls is not None:
        if not isinstance(raw_tool_calls, list | tuple):
            raise SFTDataError(f"message {index} tool_calls must be a list")
        for call_index, call in enumerate(raw_tool_calls):
            tool_calls.append(_coerce_tool_call(call, index=call_index))

    tool_call_id = raw.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        tool_call_id = str(tool_call_id)

    name = raw.get("name")
    if name is not None and not isinstance(name, str):
        raise SFTDataError(f"message {index} name must be a string if present")

    extra = {
        key: value
        for key, value in raw.items()
        if key not in {"role", "content", "tool_calls", "tool_call_id", "name"}
    }

    return SFTMessage(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
        extra=extra,
    )


def coerce_example(raw: Any, *, min_messages: int = 1) -> SFTExample:
    if not isinstance(raw, dict):
        raise SFTDataError("record is not an object")

    messages_raw = raw.get("messages")
    if not isinstance(messages_raw, Sequence):
        raise SFTDataError("missing messages[] list")
    if len(messages_raw) < min_messages:
        raise SFTDataError(f"missing messages[] with at least {min_messages} turns")

    messages = [_coerce_message(msg, index=i) for i, msg in enumerate(messages_raw)]

    tools: list[SFTToolDefinition] = []
    if "tools" in raw and raw["tools"] is not None:
        tools_raw = raw["tools"]
        if not isinstance(tools_raw, Sequence):
            raise SFTDataError("tools must be provided as a list when present")
        for tool_index, tool in enumerate(tools_raw):
            tools.append(_coerce_tool_definition(tool, index=tool_index))

    tool_choice = raw.get("tool_choice")

    metadata_field = raw.get("metadata")
    metadata: dict[str, Any] = {}
    if metadata_field is not None:
        if not isinstance(metadata_field, dict):
            raise SFTDataError("metadata must be an object if present")
        metadata = dict(metadata_field)

    extra = {
        key: value
        for key, value in raw.items()
        if key not in {"messages", "tools", "tool_choice", "metadata"}
    }

    return SFTExample(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        metadata=metadata,
        extra=extra,
    )


def parse_jsonl_line(line: str, *, min_messages: int = 1) -> SFTExample:
    record = json.loads(line)
    return coerce_example(record, min_messages=min_messages)


def iter_sft_examples(
    source: Iterable[str], *, min_messages: int = 1, skip_empty: bool = True
) -> Iterator[SFTExample]:
    for line in source:
        if skip_empty and not line.strip():
            continue
        yield parse_jsonl_line(line, min_messages=min_messages)


def collect_sft_jsonl_errors(
    path: Path,
    *,
    min_messages: int = 1,
    max_lines: int | None = None,
    max_errors: int | None = None,
) -> list[str]:
    errors: list[str] = []
    lines_checked = 0

    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            if max_lines is not None and lines_checked >= max_lines:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            lines_checked += 1
            try:
                parse_jsonl_line(stripped, min_messages=min_messages)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {lineno}: invalid JSON ({exc.msg})")
            except SFTDataError as exc:
                errors.append(f"Line {lineno}: {exc}")
            if max_errors is not None and len(errors) >= max_errors:
                break
    if lines_checked == 0 and (max_errors is None or len(errors) < max_errors):
        errors.append("File contains no SFT examples")
    return errors


def validate_jsonl_or_raise(
    path: Path,
    *,
    min_messages: int = 1,
    max_lines: int | None = None,
    max_errors: int | None = None,
    error_factory: type[Exception] = ValueError,
) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))

    issues = collect_sft_jsonl_errors(
        path,
        min_messages=min_messages,
        max_lines=max_lines,
        max_errors=max_errors,
    )
    if issues:
        truncated = max_errors is not None and len(issues) >= max_errors
        suffix = "" if not truncated else f" (showing first {max_errors} issues)"
        details = "\n - ".join(issues)
        raise error_factory(f"{path}: Dataset validation failed{suffix}:\n - {details}")


def load_jsonl(path: Path, *, min_messages: int = 1) -> list[SFTExample]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as fh:
        return list(iter_sft_examples(fh, min_messages=min_messages))


__all__ = [
    "SFTDataError",
    "SFTExample",
    "SFTMessage",
    "SFTToolCall",
    "SFTToolDefinition",
    "collect_sft_jsonl_errors",
    "coerce_example",
    "iter_sft_examples",
    "load_jsonl",
    "parse_jsonl_line",
    "validate_jsonl_or_raise",
]
