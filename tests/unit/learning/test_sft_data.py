from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_ai.learning.sft import (
    collect_sft_jsonl_errors,
    iter_sft_examples,
    load_jsonl,
    validate_jsonl_or_raise,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CRAFT_SFT = REPO_ROOT / "examples" / "warming_up_to_rl" / "ft_data" / "crafter_sft.jsonl"


@pytest.mark.parametrize("path", [CRAFT_SFT])
def test_collect_sft_jsonl_errors_clean(path: Path) -> None:
    assert path.exists(), f"fixture dataset missing: {path}"
    errors = collect_sft_jsonl_errors(path, min_messages=2, max_errors=5)
    assert errors == []


def test_iter_sft_examples_parses_tool_calls() -> None:
    with CRAFT_SFT.open("r", encoding="utf-8") as fh:
        example = next(iter_sft_examples(fh, min_messages=2))

    assert example.messages[-1].tool_calls, "expected tool calls in assistant turn"
    call = example.messages[-1].tool_calls[0]
    assert call.name == "interact"
    assert isinstance(call.arguments, dict)
    assert call.arguments.get("actions"), "parsed actions payload missing"
    assert example.metadata.get("model"), "model metadata missing"


def test_load_jsonl_supports_tools_and_metadata(tmp_path: Path) -> None:
    sample = {
        "messages": [
            {"role": "system", "content": "keep outputs concise"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": "{\"value\": 2}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": {"result": 4},
            },
        ],
        "tools": [
            {
                "name": "calculator",
                "description": "compute simple expressions",
                "parameters": {"type": "object"},
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "calculator"}},
        "metadata": {"split": "train"},
        "source": "unit-test",
    }
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    example = load_jsonl(path, min_messages=1)[0]

    assert example.tools and example.tools[0].name == "calculator"
    assert example.tool_choice == {"type": "function", "function": {"name": "calculator"}}
    call = example.messages[1].tool_calls[0]
    assert call.call_id == "call-1"
    assert call.type == "function"
    assert call.arguments == {"value": 2}
    assert example.metadata == {"split": "train"}
    assert example.extra["source"] == "unit-test"


def test_collect_sft_jsonl_errors_reports_issues(tmp_path: Path) -> None:
    invalid = {
        "messages": [
            {
                "role": "assistant",
                "content": "oops",
                "tool_calls": {"not": "a list"},
            }
        ]
    }
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    errors = collect_sft_jsonl_errors(path, min_messages=1, max_errors=5)
    assert errors and "tool_calls" in errors[0]


def test_collect_sft_jsonl_errors_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    errors = collect_sft_jsonl_errors(path, min_messages=1, max_errors=5)
    assert errors and "no sft examples" in errors[0].lower()


def test_validate_jsonl_or_raise_includes_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        validate_jsonl_or_raise(path, min_messages=2)

    assert str(path) in str(excinfo.value)
