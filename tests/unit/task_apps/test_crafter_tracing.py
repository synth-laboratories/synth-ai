import json
from types import SimpleNamespace
from datetime import datetime, UTC

import pytest

from synth_ai.tracing_v3.session_tracer import SessionTracer

from examples.task_apps.crafter.task_app.synth_envs_hosted.rollout import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRequest,
    RolloutTracingContext,
)
from synth_ai.task.contracts import RolloutMode


def _dummy_request():
    app_state = SimpleNamespace(sft_output_dir=None)
    app = SimpleNamespace(state=app_state)
    return SimpleNamespace(app=app, state=SimpleNamespace())


@pytest.mark.asyncio
async def test_tracing_context_records_full_messages(monkeypatch):
    tracer = SessionTracer(auto_save=False)

    async def _noop():
        return None

    monkeypatch.setattr(tracer, "initialize", _noop)

    request = RolloutRequest(
        run_id="run-test",
        env=RolloutEnvSpec(env_name="crafter", config={}, seed=123),
        policy=RolloutPolicySpec(
            policy_name="crafter-react",
            config={
                "inference_url": (
                    "https://modal-host.fake/v1/chat/completions?cid=trace_run-test"
                )
            },
        ),
        ops=["agent", "env"],
        record={"return_trace": True, "trace_format": "structured"},
        safety={"max_ops": 100},
        mode=RolloutMode.RL,
    )

    tracing = RolloutTracingContext(tracer, request, _dummy_request())

    await tracing.start_session()
    await tracing.start_decision(0)
    await tracing.record_policy_prompts(
        system_messages=[{"role": "system", "text": "You are helpful."}],
        user_messages=[{"role": "user", "text": "Plant a tree."}],
    )

    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "interact_many",
                "arguments": '{"actions": [{"button": "UP", "frames": 12}]}',
            },
        }
    ]

    inference_request = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Plant a tree."},
        ],
        "temperature": 0.2,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "interact_many",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Thinking... executing actions."}],
        "tool_calls": tool_calls,
        "reasoning": [{"type": "text", "text": "We should gather wood first."}],
    }
    inference_response = {
        "choices": [{"message": assistant_message}],
        "usage": {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18},
    }

    started = datetime.now(UTC)
    completed = started

    await tracing.record_llm_call(
        inference_request=inference_request,
        inference_response=inference_response,
        tool_calls=tool_calls,
        provider="test-provider",
        model_name="test-model",
        started_at=started,
        completed_at=completed,
        latency_ms=123,
    )
    await tracing.record_tool_invocation(tool_calls)
    await tracing.end_decision()

    session_trace = await tracing.finalize(
        total_reward=1.0, achievement_state={}, total_steps=1
    )
    payload = tracing.build_trace_payload(session_trace)

    assert payload is not None
    messages = payload.get("markov_blanket_message_history") or []
    assert len(messages) >= 3

    def _json_content(entry):
        content = entry.get("content") or {}
        payload_json = content.get("json_payload")
        if payload_json:
            return json.loads(payload_json)
        return {"text": content.get("text")}

    system_msgs = [m for m in messages if m["message_type"] == "system"]
    user_msgs = [m for m in messages if m["message_type"] == "user"]
    assistant_msgs = [m for m in messages if m["message_type"] == "assistant"]

    assert system_msgs, "missing system prompt in trace"
    assert user_msgs, "missing user prompt in trace"
    assert assistant_msgs, "missing assistant response in trace"

    assistant_payloads = [_json_content(m) for m in assistant_msgs]
    has_reasoning = any("reasoning" in payload for payload in assistant_payloads)
    has_tool_calls = any("tool_calls" in payload for payload in assistant_payloads)

    assert has_reasoning, "assistant reasoning not captured in trace"
    assert has_tool_calls, "assistant tool calls not captured in trace"

    tool_call_metadata = [
        m for m in assistant_msgs if m.get("metadata", {}).get("is_tool_call")
    ]
    assert tool_call_metadata, "tool call message missing metadata flag"
