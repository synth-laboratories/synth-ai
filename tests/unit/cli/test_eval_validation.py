from synth_ai.core.eval.validation import validate_eval_options


def test_validate_eval_options_normalizes_seed_and_metadata() -> None:
    options = {
        "app_id": "demo",
        "model": "model",
        "seeds": "1, 2 ,3",
        "metadata": ["difficulty=easy", "env=demo"],
    }

    normalized = validate_eval_options(options)
    assert normalized["seeds"] == [1, 2, 3]
    assert normalized["metadata"] == {"difficulty": "easy", "env": "demo"}


def test_validate_eval_options_coerces_numbers() -> None:
    options = {
        "max_turns": "15",
        "max_llm_calls": "5",
        "concurrency": "2",
        "return_trace": "true",
    }

    normalized = validate_eval_options(options)
    assert normalized["max_turns"] == 15
    assert normalized["max_llm_calls"] == 5
    assert normalized["concurrency"] == 2
    assert normalized["return_trace"] is True
