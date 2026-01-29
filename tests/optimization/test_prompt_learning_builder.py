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
                    "train_seeds": [0],
                    "val_seeds": [1],
                }
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
