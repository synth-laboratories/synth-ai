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
        "best_prompt": "do the thing",
    }
    result = PolicyOptimizationResult.from_response("job_123", payload, algorithm="gepa")
    assert result.status == PolicyJobStatus.SUCCEEDED
    assert result.best_score == 0.9
    assert result.best_prompt == "do the thing"
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
    assert result.best_score == 0.77
    assert result.best_yaml == "graph: foo"
    assert result.generations_completed == 3
    assert result.algorithm == "graph_evolve"


def test_prompt_learning_result_parsing() -> None:
    payload = {
        "status": "completed",
        "best_train_score": 0.42,
        "best_prompt": "be precise",
    }
    result = PromptLearningResult.from_response("pl_1", payload)
    assert result.status == PolicyJobStatus.SUCCEEDED
    assert result.best_score == 0.42
    assert result.best_prompt == "be precise"
