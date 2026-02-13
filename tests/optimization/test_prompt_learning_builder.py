from __future__ import annotations

from synth_ai.sdk.optimization.internal.builders import build_prompt_learning_payload_from_mapping


def _gepa_config_dict() -> dict:
    return {
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_url": "http://config.example.com",
            "policy": {
                "inference_mode": "synth_hosted",
                "provider": "openai",
                "model": "gpt-4o-mini",
            },
            "gepa": {
                "evaluation": {
                    "train_seeds": list(range(70)),
                    "val_seeds": list(range(70, 80)),
                },
                "archive": {
                    "pareto_set_size": 10,
                },
            },
        }
    }


def test_build_prompt_learning_payload_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "env_key")
    monkeypatch.setenv("SYNTH_LOCALAPI_AUTH_PERSIST", "0")

    overrides = {
        "task_url": "http://override.example.com",
        "prompt_learning.policy.model": "gpt-4o",
    }

    result = build_prompt_learning_payload_from_mapping(
        raw_config=_gepa_config_dict(),
        task_url=None,
        overrides=overrides,
    )

    assert result.task_url == "http://override.example.com"
    assert (
        result.payload["config_body"]["prompt_learning"]["policy"]["model"] == "gpt-4o"
    )


def test_build_prompt_learning_payload_config_precedence(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "env_key")
    monkeypatch.setenv("SYNTH_LOCALAPI_AUTH_PERSIST", "0")
    monkeypatch.setenv("TASK_APP_URL", "http://env.example.com")

    result = build_prompt_learning_payload_from_mapping(
        raw_config=_gepa_config_dict(),
        task_url=None,
        overrides={},
    )

    assert result.task_url == "http://config.example.com"


def test_build_prompt_learning_verifier_backend_base_defaults_to_job_backend(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "env_key")
    monkeypatch.setenv("SYNTH_LOCALAPI_AUTH_PERSIST", "0")

    config = _gepa_config_dict()
    config["prompt_learning"]["verifier"] = {
        "enabled": True,
        "reward_source": "fused",
        "verifier_graph_id": "zero_shot_verifier_rubric_rlm",
        "weight_event": 0.2,
        "weight_outcome": 0.2,
        # backend_base intentionally omitted (should default to overrides["backend"])
    }

    result = build_prompt_learning_payload_from_mapping(
        raw_config=config,
        task_url=None,
        overrides={"backend": "http://127.0.0.1:8000"},
    )

    verifier = result.payload["config_body"]["prompt_learning"]["verifier"]
    assert verifier["backend_base"] == "http://127.0.0.1:8000"
