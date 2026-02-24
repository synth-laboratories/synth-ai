from synth_ai.sdk.eval.job import EvalResult, EvalStatus


def test_eval_status_strict_success_mapping() -> None:
    assert EvalStatus.from_string("succeeded") == EvalStatus.SUCCEEDED


def test_eval_status_rejects_noncanonical_aliases() -> None:
    aliases = ["completed", "success", "done", "in_progress", "queued", "canceled", "error"]
    for alias in aliases:
        try:
            EvalStatus.from_string(alias)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for noncanonical alias: {alias}")


def test_eval_result_treats_succeeded_as_success() -> None:
    result = EvalResult.from_response(
        "eval_test",
        {
            "status": "succeeded",
            "results": {
                "completed": 1,
                "failed": 0,
                "total": 1,
                "mean_reward": 0.0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            },
        },
    )
    assert result.status == EvalStatus.SUCCEEDED
    assert result.succeeded
    assert result.mean_reward == 0.0
    assert result.total_tokens == 0
    assert result.total_cost_usd == 0.0
    assert result.num_completed == 1
    assert result.num_total == 1
