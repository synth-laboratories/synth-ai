from synth_ai.cli.task_app import _markov_message_from_dict


def test_markov_message_normalizes_policy_prompts_to_system():
    payload = {
        "message_type": "policy_system_prompt",
        "content": {"text": "system prompt", "json_payload": None},
        "metadata": {"step_id": "decision_0"},
        "time_record": {"event_time": 123.4, "message_time": None},
    }

    message = _markov_message_from_dict(payload)

    assert message.message_type == "system"
    # The original value is preserved for debugging/backfills
    assert message.metadata["original_message_type"] == "policy_system_prompt"


def test_markov_message_normalizes_policy_tool_call_to_assistant():
    payload = {
        "message_type": "policy_tool_call",
        "content": {"text": None, "json_payload": '[{"tool_name":"interact_many"}]'},
        "metadata": {"step_id": "decision_0"},
        "time_record": {"event_time": 123.5, "message_time": None},
    }

    message = _markov_message_from_dict(payload)

    assert message.message_type == "assistant"
    assert message.metadata["original_message_type"] == "policy_tool_call"
