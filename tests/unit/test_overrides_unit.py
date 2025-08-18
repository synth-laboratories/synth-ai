from types import SimpleNamespace
from typing import Any

import pytest

from synth_ai.lm.overrides import LMOverridesContext
from synth_ai.lm.vendors.openai_standard import OpenAIStandard


class _FakeChatCompletions:
    def __init__(self, recorder):
        self._rec = recorder

    def create(self, **kwargs):
        # Record the outgoing API params for assertions
        self._rec["last_kwargs"] = kwargs
        # Minimal OpenAI-like response shape used by vendor
        message = SimpleNamespace(content="OK", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason=None)
        return SimpleNamespace(choices=[choice], usage=None)


class _FakeSyncClient:
    def __init__(self, recorder):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(recorder))


class _FakeAsyncClient:
    # Not used by sync tests, but OpenAIStandard expects it to exist
    def __init__(self):
        self.base_url = "https://api.groq.com/openai/v1"


def _make_vendor(recorder: dict[str, Any]) -> OpenAIStandard:
    return OpenAIStandard(sync_client=_FakeSyncClient(recorder), async_client=_FakeAsyncClient())


def _text_messages(user_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "System."},
        {"role": "user", "content": user_text},
    ]


@pytest.mark.unit
def test_injection_rules_applied_to_messages_sync():
    rec: dict[str, Any] = {}
    vendor = _make_vendor(rec)
    messages = _text_messages("I used the atm to withdraw cash.")

    overrides = [
        {
            "match": {"contains": "atm", "role": "user"},
            "injection_rules": [{"find": "atm", "replace": "ATM"}],
        }
    ]

    with LMOverridesContext(overrides):
        _ = vendor._hit_api_sync(
            model="openai/gpt-oss-20b",
            messages=messages,
            lm_config={},
        )

    sent_messages = rec["last_kwargs"]["messages"]
    assert any(
        "ATM" in (m.get("content") if isinstance(m.get("content"), str) else "")
        for m in sent_messages
    ), f"Expected 'ATM' substitution in outgoing messages, got: {sent_messages}"


@pytest.mark.unit
def test_param_overrides_model_and_temperature_sync():
    rec: dict[str, Any] = {}
    vendor = _make_vendor(rec)
    messages = _text_messages("Hello")

    overrides = [
        {
            "match": {"contains": "hello", "role": "user"},
            "params": {"model": "openai/gpt-oss-120b", "temperature": 0.1},
        }
    ]

    with LMOverridesContext(overrides):
        _ = vendor._hit_api_sync(
            model="openai/gpt-oss-20b",
            messages=messages,
            lm_config={},
        )

    assert rec["last_kwargs"]["model"] == "openai/gpt-oss-120b"
    assert rec["last_kwargs"].get("temperature") == 0.1


@pytest.mark.unit
def test_tool_overrides_set_add_remove_and_choice_sync():
    rec: dict[str, Any] = {}
    vendor = _make_vendor(rec)
    messages = _text_messages("use atm tool")

    base_tools = [
        {"type": "function", "function": {"name": "foo", "description": "a"}},
        {"type": "function", "function": {"name": "bar", "description": "b"}},
    ]
    add_tools = [
        {"type": "function", "function": {"name": "baz", "description": "c"}},
    ]

    overrides = [
        {
            "match": {"contains": "atm"},
            "tools": {
                "set_tools": base_tools,
                "add_tools": add_tools,
                "remove_tools_by_name": ["bar"],
                "tool_choice": "required",
            },
        }
    ]

    with LMOverridesContext(overrides):
        _ = vendor._hit_api_sync(
            model="openai/gpt-oss-20b",
            messages=messages,
            lm_config={},
        )

    tools_out = rec["last_kwargs"].get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools_out]
    # set_tools + add_tools -> foo, bar, baz then remove bar => foo, baz
    assert names == ["foo", "baz"]
    assert rec["last_kwargs"].get("tool_choice") == "required"
