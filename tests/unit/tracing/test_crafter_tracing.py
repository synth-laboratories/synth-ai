from __future__ import annotations
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from examples.warming_up_to_rl.task_app.synth_envs_hosted.envs.crafter.policy import CrafterPolicy
from examples.warming_up_to_rl.task_app.synth_envs_hosted.rollout import (
    RolloutEnvSpec,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutTracingContext,
)
from synth_ai.tracing_v3.abstractions import LMCAISEvent, SessionMessageContent
from synth_ai.tracing_v3.session_tracer import SessionTracer

pytestmark = pytest.mark.unit


def _build_rollout_request(policy_name: str) -> RolloutRequest:
    return RolloutRequest(
        run_id="run-test",
        env=RolloutEnvSpec(env_name="CrafterClassic", config={}, seed=7),
        policy=RolloutPolicySpec(
            policy_name=policy_name,
            config={"inference_url": "mock://llm", "model": "mock-model"},
        ),
        ops=[],
        record=RolloutRecordConfig(return_trace=True, trace_format="full"),
    )


def _make_fastapi_request() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(),
        app=SimpleNamespace(state=SimpleNamespace()),
    )


def _text_observation() -> dict[str, object]:
    return {
        "observation": {
            "health": 10,
            "inventory": {"wood": 0},
            "achievements_status": {},
            "player_position": [31, 31],
            "player_direction": [0, 1],
            "num_steps_taken": 0,
            "max_steps_episode": 20,
            "reward_last_step": 0.0,
            "total_reward_episode": 0.0,
            "terminated": False,
            "truncated": False,
        },
        "step_idx": 0,
    }


def _image_observation(data_url: str, base64_data: str) -> dict[str, object]:
    payload = _text_observation()
    obs = payload["observation"]
    assert isinstance(obs, dict)
    obs["observation_image_data_url"] = data_url
    obs["observation_image_base64"] = base64_data
    obs["observation_image_format"] = "png"
    obs["observation_image_width"] = 1
    obs["observation_image_height"] = 1
    return payload


async def _run_policy_and_trace(
    policy: CrafterPolicy,
    observation: dict[str, object],
) -> tuple[RolloutTracingContext, SessionTracer]:
    tracer = SessionTracer(auto_save=False)
    rollout_request = _build_rollout_request(policy.name)
    fastapi_request = _make_fastapi_request()
    context = RolloutTracingContext(tracer, rollout_request, fastapi_request)

    await context.start_session()
    await context.start_decision(turn_number=0)

    messages, _ = policy.prepare_inference_request(observation)
    system_prompts = [msg["content"] for msg in messages if msg["role"] == "system"]
    user_prompts = [msg["content"] for msg in messages if msg["role"] == "user"]
    await context.record_policy_prompts(system_prompts, user_prompts)

    inference_request = {"messages": messages, "temperature": 0.1}
    inference_response = {
        "choices": [{"message": {"content": "move_right"}, "finish_reason": "stop"}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    now = datetime.now(UTC)
    await context.record_llm_call(
        inference_request=inference_request,
        inference_response=inference_response,
        tool_calls=[],
        provider="mock-provider",
        model_name="mock-model",
        started_at=now,
        completed_at=now,
        latency_ms=0,
    )
    await context.end_decision()
    return context, tracer


@pytest.mark.asyncio
async def test_crafter_text_prompt_tracing():
    policy = CrafterPolicy(inference_url="mock://text", model="mock-model")
    await policy.initialize({"use_tools": False})
    _, tracer = await _run_policy_and_trace(policy, _text_observation())

    trace = await tracer.end_session(save=False)
    user_messages = [
        msg for msg in trace.markov_blanket_message_history if msg.message_type == "policy_user_prompt"
    ]
    assert user_messages, "Expected user prompts to be recorded"
    for recorded in user_messages:
        assert isinstance(recorded.content, SessionMessageContent)
        assert recorded.content.json_payload is None
        assert "crafter" in recorded.content.as_text().lower()

    events = [event for event in trace.event_history if isinstance(event, LMCAISEvent)]
    assert events, "No LLM events recorded"
    input_parts = events[0].call_records[0].input_messages[1].parts
    assert input_parts[0].type == "text"


@pytest.mark.asyncio
async def test_crafter_image_prompt_tracing():
    pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/5/hPwAF+wL+z7xGXwAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{pixel_base64}"

    policy = CrafterPolicy(inference_url="mock://image", model="mock-model")
    await policy.initialize({"use_tools": False})
    observation = _image_observation(
        data_url=data_url,
        base64_data=pixel_base64,
    )
    _, tracer = await _run_policy_and_trace(policy, observation)

    trace = await tracer.end_session(save=False)
    user_messages = [
        msg for msg in trace.markov_blanket_message_history if msg.message_type == "policy_user_prompt"
    ]
    assert user_messages, "Expected user prompts to be recorded"
    for recorded in user_messages:
        assert isinstance(recorded.content, SessionMessageContent)
        assert recorded.content.json_payload is not None
        assert "data:image/png;base64" in recorded.content.json_payload

    events = [event for event in trace.event_history if isinstance(event, LMCAISEvent)]
    assert events, "No LLM events recorded"
    multimodal_parts = events[0].call_records[0].input_messages[1].parts
    assert any(part.type == "image" for part in multimodal_parts), "Image content part missing"
