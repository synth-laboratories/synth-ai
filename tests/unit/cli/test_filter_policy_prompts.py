from __future__ import annotations

import json

from synth_ai.cli.commands.filter.core import _select_messages


def _encode(obj: object) -> str:
    return json.dumps(obj)


def test_select_messages_includes_system_prompt_and_tool_call():
    # Simulate message history returned by the CLI trace serializer
    message_rows = [
        {
            "message_type": "policy_system_prompt",
            "content": _encode({"role": "system", "content": "system instructions"}),
        },
        {
            "message_type": "policy_user_prompt",
            "content": _encode({"role": "user", "content": "user observation"}),
        },
        {
            "message_type": "policy_tool_call",
            "content": _encode(
                [
                    {
                        "tool_name": "interact_many",
                        "arguments": {"actions": ["move_right", "do"]},
                    }
                ]
            ),
        },
    ]

    records = _select_messages(message_rows)
    assert len(records) == 1

    conversation = records[0]["messages"]
    assert conversation[0] == {"role": "system", "content": "system instructions"}
    assert conversation[1] == {"role": "user", "content": "user observation"}
    # Assistant turn should preserve the tool payload structure
    assert conversation[2]["role"] == "assistant"
    assert conversation[2]["content"] == [
        {
            "tool_name": "interact_many",
            "arguments": {"actions": ["move_right", "do"]},
        }
    ]


def test_select_messages_skips_missing_system_prompt_gracefully():
    message_rows = [
        {
            "message_type": "policy_user_prompt",
            "content": _encode({"role": "user", "content": "user without system"}),
        },
        {
            "message_type": "policy_tool_call",
            "content": _encode(
                [{"tool_name": "interact_many", "arguments": {"actions": ["noop"]}}]
            ),
        },
    ]

    records = _select_messages(message_rows)
    assert len(records) == 1
    conversation = records[0]["messages"]
    # No synthetic system message should be injected when none is present
    assert conversation[0] == {"role": "user", "content": "user without system"}
