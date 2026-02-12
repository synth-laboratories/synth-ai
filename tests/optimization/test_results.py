from synth_ai.sdk.optimization.models import (
    GraphJobStatus,
    GraphOptimizationResult,
    PolicyJobStatus,
    PolicyOptimizationResult,
    PromptLearningResult,
)


def test_policy_result_parsing() -> None:
    payload = {
        "status": "succeeded",
        "best_reward": 0.9,
        "best_candidate": "do the thing",
    }
    result = PolicyOptimizationResult.from_response("job_123", payload, algorithm="gepa")
    assert result.status == PolicyJobStatus.SUCCEEDED
    assert result.best_reward == 0.9
    assert result.best_candidate == "do the thing"
    assert result.algorithm == "gepa"


def test_graph_result_parsing() -> None:
    payload = {
        "status": "completed",
        "best_score": 0.77,
        "best_yaml": "graph: foo",
        "generations_completed": 3,
    }
    result = GraphOptimizationResult.from_response("graph_1", payload, algorithm="graph_evolve")
    assert result.status == GraphJobStatus.COMPLETED
    assert result.best_reward == 0.77
    assert result.best_yaml == "graph: foo"
    assert result.generations_completed == 3
    assert result.algorithm == "graph_evolve"


def test_prompt_learning_result_parsing() -> None:
    payload = {
        "status": "completed",
        "best_train_score": 0.42,
        "best_candidate": "be precise",
    }
    result = PromptLearningResult.from_response("pl_1", payload)
    assert result.status == PolicyJobStatus.SUCCEEDED
    assert result.best_reward == 0.42
    assert result.best_candidate == "be precise"


def test_prompt_learning_result_parses_levers_and_sensors_from_metadata() -> None:
    payload = {
        "status": "succeeded",
        "metadata": {
            "best_prompt": {"messages": [{"role": "system", "content": "old key fallback"}]},
            "lever_summary": {"prompt_lever_id": "mipro.prompt.abc"},
            "sensor_frames": [{"frame_id": "frame_1"}],
            "lever_versions": {"mipro.prompt.abc": "3"},
        },
    }
    result = PromptLearningResult.from_response("pl_2", payload)
    assert result.best_candidate == payload["metadata"]["best_prompt"]
    assert result.lever_summary == {"prompt_lever_id": "mipro.prompt.abc"}
    assert result.sensor_frames == [{"frame_id": "frame_1"}]
    assert result.lever_versions == {"mipro.prompt.abc": 3}
    assert result.best_lever_version == 3


def test_policy_result_best_prompt_alias() -> None:
    payload = {"status": "completed", "best_prompt": "legacy"}
    result = PolicyOptimizationResult.from_response("job_legacy", payload)
    assert result.best_candidate == "legacy"
    assert result.best_prompt == "legacy"
