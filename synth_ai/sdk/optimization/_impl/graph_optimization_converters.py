"""Dataset converters for Graph Optimization."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    all_warnings: list[ConversionWarning] = []

    if isinstance(source, (str, Path)):
        examples, warnings = validate_sft_file(source)
    else:
        examples, warnings = validate_sft_examples(source)
    all_warnings.extend(warnings)

    if max_examples is not None and len(examples) > max_examples:
        examples = examples[:max_examples]

    system_prompt = detect_system_prompt(examples)
    unique_system_prompts = len(
        {parse_sft_example(ex)[0] for ex in examples if parse_sft_example(ex)[0]}
    )

    parsed = [parse_sft_example(ex) for ex in examples]
    user_messages = [p[1] for p in parsed if p[1]]
    assistant_messages = [p[2] for p in parsed]

    template: str | None = None
    field_names = ["user_message"]
    if detect_template and user_messages:
        template, field_names = infer_template(user_messages)

    tasks: list[dict[str, Any]] = []
    gold_outputs: list[dict[str, Any]] = []

    for i, (parsed_ex, assistant) in enumerate(zip(parsed, assistant_messages, strict=False)):
        _, user, _ = parsed_ex
        if not user or not assistant:
            continue

        task_id = f"sft_{i:04d}"

        if template and field_names != ["user_message"]:
            input_dict = extract_fields(user, field_names)
        else:
            input_dict = {"user_message": user}

        tasks.append({"task_id": task_id, "input": input_dict})
        gold_outputs.append({"task_id": task_id, "output": {"response": assistant}, "score": 1.0})

    metadata: dict[str, Any] = {
        "name": dataset_name,
        "task_description": system_prompt or "Complete the assistant response",
        "source_format": "openai_sft",
    }

    if template:
        metadata["detected_template"] = template
        metadata["input_schema"] = {
            "type": "object",
            "properties": {f: {"type": "string"} for f in field_names},
        }

    stats = {
        "total_examples": len(examples),
        "skipped_examples": len(all_warnings),
        "output_examples": len(tasks),
        "template_detected": template is not None,
        "detected_fields": field_names,
        "unique_system_prompts": unique_system_prompts,
    }

    if unique_system_prompts > 1:
        all_warnings.append(
            ConversionWarning(
                f"Found {unique_system_prompts} different system prompts; using most common"
            )
        )

    dataset = {
        "tasks": tasks,
        "gold_outputs": gold_outputs,
        "metadata": metadata,
    }

    return ConversionResult(dataset=dataset, warnings=all_warnings, stats=stats)


def preview_conversion(
    source: str | Path | list[dict[str, Any]],
    num_examples: int = 3,
) -> dict[str, Any]:
    """Preview what conversion would produce without full processing."""
    result = convert_openai_sft(source, max_examples=num_examples)

    return {
        "sample_tasks": result.dataset["tasks"],
        "sample_gold_outputs": result.dataset["gold_outputs"],
        "metadata": result.dataset["metadata"],
        "stats": result.stats,
        "warnings": [w.message for w in result.warnings],
    }


__all__ = [
    "convert_openai_sft",
    "preview_conversion",
    "ConversionResult",
    "ConversionWarning",
    "ConversionError",
]
