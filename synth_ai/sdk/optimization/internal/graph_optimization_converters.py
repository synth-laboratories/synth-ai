"""Dataset converters for Graph Optimization."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "synth_ai_py is required for optimization.graph_optimization_converters."
    ) from exc

# Known field prefix patterns for template detection (order matters - check longer patterns first)
KNOWN_PREFIXES: list[tuple[str, str]] = [
    ("### Instruction", "instruction"),
    ("### Input", "input"),
    ("### Response", "response"),
    ("Instruction:", "instruction"),
    ("Question:", "question"),
    ("Context:", "context"),
    ("Input:", "input"),
    ("Output:", "output"),
    ("Query:", "query"),
    ("Document:", "document"),
    ("Text:", "text"),
    ("Passage:", "passage"),
]


class ConversionError(Exception):
    """Raised when conversion fails completely."""

    pass


@dataclass
class ConversionWarning:
    """Non-fatal issue during conversion."""

    message: str
    example_idx: int | None = None


@dataclass
class ConversionResult:
    """Result of converting SFT to Graph Opt.

    Attributes:
        dataset: The Graph Opt dataset dict
        warnings: Non-fatal issues encountered
        stats: Conversion statistics
    """

    dataset: dict[str, Any]
    warnings: list[ConversionWarning] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def parse_sft_example(example: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Extract system, user, assistant from messages array."""
    messages = example.get("messages", [])
    system = None
    user = None
    assistant = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            user = content  # Take last user message
        elif role == "assistant":
            assistant = content  # Take last assistant message

    return system, user, assistant


def detect_system_prompt(examples: list[dict[str, Any]]) -> str | None:
    """Find common system prompt across all examples."""
    system_prompts = [parse_sft_example(ex)[0] for ex in examples]
    system_prompts = [s for s in system_prompts if s]

    if not system_prompts:
        return None

    if len(set(system_prompts)) == 1:
        return system_prompts[0]

    return Counter(system_prompts).most_common(1)[0][0]


def infer_template(user_messages: list[str]) -> tuple[str | None, list[str]]:
    """Detect if user messages follow a template pattern."""
    if not user_messages:
        return None, ["user_message"]

    sample = user_messages[: min(10, len(user_messages))]

    detected_fields: list[tuple[str, str]] = []

    for prefix, field_name in KNOWN_PREFIXES:
        matches = sum(1 for msg in sample if prefix in msg)
        if matches >= len(sample) * 0.8:
            detected_fields.append((prefix, field_name))

    if not detected_fields:
        return None, ["user_message"]

    first_msg = sample[0]
    detected_fields.sort(key=lambda x: first_msg.find(x[0]) if x[0] in first_msg else 9999)

    template_parts = []
    field_names = []
    for prefix, field_name in detected_fields:
        template_parts.append(f"{prefix} {{{field_name}}}")
        field_names.append(field_name)

    template = "\n".join(template_parts)
    return template, field_names


def extract_fields(user_message: str, field_names: list[str]) -> dict[str, str]:
    """Extract field values from user message using detected prefixes."""
    result: dict[str, str] = {}

    for field_name in field_names:
        prefix = None
        for p, fn in KNOWN_PREFIXES:
            if fn == field_name:
                prefix = p
                break

        if not prefix or prefix not in user_message:
            continue

        start = user_message.find(prefix) + len(prefix)
        while start < len(user_message) and user_message[start] in " :\n":
            start += 1

        end = len(user_message)
        for p, _ in KNOWN_PREFIXES:
            if p != prefix:
                pos = user_message.find(p, start)
                if pos != -1 and pos < end:
                    end = pos

        value = user_message[start:end].strip()
        result[field_name] = value

    return result if result else {"user_message": user_message}


def validate_sft_file(path: str | Path) -> tuple[list[dict[str, Any]], list[ConversionWarning]]:
    """Validate and load SFT file, collecting warnings."""
    warnings: list[ConversionWarning] = []
    examples: list[dict[str, Any]] = []
    path = Path(path)

    if not path.exists():
        raise ConversionError(f"File not found: {path}")

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(ConversionWarning(f"Invalid JSON on line {i + 1}", i))
                continue

            if "messages" not in ex:
                warnings.append(ConversionWarning(f"Missing 'messages' on line {i + 1}", i))
                continue

            messages = ex["messages"]
            if not isinstance(messages, list):
                warnings.append(ConversionWarning(f"'messages' is not a list on line {i + 1}", i))
                continue

            roles = {m.get("role") for m in messages}
            if "user" not in roles:
                warnings.append(ConversionWarning(f"No 'user' role on line {i + 1}", i))
                continue
            if "assistant" not in roles:
                warnings.append(ConversionWarning(f"No 'assistant' role on line {i + 1}", i))
                continue

            assistant_content = None
            for msg in messages:
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
            if not assistant_content or not assistant_content.strip():
                warnings.append(ConversionWarning(f"Empty assistant response on line {i + 1}", i))
                continue

            examples.append(ex)

    if not examples:
        raise ConversionError("No valid examples found in file")

    return examples, warnings


def validate_sft_examples(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[ConversionWarning]]:
    """Validate a list of SFT examples in memory."""
    warnings: list[ConversionWarning] = []
    valid_examples: list[dict[str, Any]] = []

    for i, ex in enumerate(examples):
        if "messages" not in ex:
            warnings.append(ConversionWarning(f"Missing 'messages' in example {i}", i))
            continue

        messages = ex["messages"]
        if not isinstance(messages, list):
            warnings.append(ConversionWarning(f"'messages' is not a list in example {i}", i))
            continue

        roles = {m.get("role") for m in messages}
        if "user" not in roles:
            warnings.append(ConversionWarning(f"No 'user' role in example {i}", i))
            continue
        if "assistant" not in roles:
            warnings.append(ConversionWarning(f"No 'assistant' role in example {i}", i))
            continue

        assistant_content = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
        if not assistant_content or not assistant_content.strip():
            warnings.append(ConversionWarning(f"Empty assistant response in example {i}", i))
            continue

        valid_examples.append(ex)

    if not valid_examples:
        raise ConversionError("No valid examples found")

    return valid_examples, warnings


def convert_openai_sft(
    source: str | Path | list[dict[str, Any]],
    dataset_name: str = "converted_sft",
    detect_template: bool = True,
    max_examples: int | None = None,
) -> ConversionResult:
    """Convert OpenAI SFT format to Graph Opt dataset."""
    if synth_ai_py is None or not hasattr(synth_ai_py, "convert_openai_sft"):
        raise ConversionError("Rust core converter required; synth_ai_py unavailable.")

    payload = source
    if isinstance(source, Path):
        payload = str(source)

    try:
        result = synth_ai_py.convert_openai_sft(
            payload, dataset_name, detect_template, max_examples
        )
    except Exception as exc:
        raise ConversionError(str(exc)) from exc
    warnings = [
        ConversionWarning(w.get("message", ""), w.get("example_idx"))
        for w in result.get("warnings", [])
        if isinstance(w, dict)
    ]

    return ConversionResult(
        dataset=cast(dict[str, Any], result.get("dataset", {})),
        warnings=warnings,
        stats=cast(dict[str, Any], result.get("stats", {})),
    )

    # Rust core handles conversion; no Python fallback.


def preview_conversion(
    source: str | Path | list[dict[str, Any]],
    num_examples: int = 3,
) -> dict[str, Any]:
    """Preview what conversion would produce without full processing."""
    if synth_ai_py is None or not hasattr(synth_ai_py, "convert_openai_sft"):
        raise ConversionError("Rust core converter required; synth_ai_py unavailable.")

    payload = source
    if isinstance(source, Path):
        payload = str(source)
    try:
        result = synth_ai_py.convert_openai_sft(payload, "converted_sft", True, num_examples)
    except Exception as exc:
        raise ConversionError(str(exc)) from exc

    return {
        "sample_tasks": result.get("dataset", {}).get("tasks", []),
        "sample_gold_outputs": result.get("dataset", {}).get("gold_outputs", []),
        "metadata": result.get("dataset", {}).get("metadata", {}),
        "stats": result.get("stats", {}),
        "warnings": [w.get("message") for w in result.get("warnings", []) if isinstance(w, dict)],
    }

    # Rust core handles conversion; no Python fallback.


__all__ = [
    "convert_openai_sft",
    "preview_conversion",
    "ConversionResult",
    "ConversionWarning",
    "ConversionError",
]
