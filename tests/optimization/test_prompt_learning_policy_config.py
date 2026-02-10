from __future__ import annotations

from synth_ai.sdk.optimization.internal.configs.prompt_learning import PromptLearningConfig


def test_policy_extras_forward_to_policy_config() -> None:
    config = PromptLearningConfig.from_mapping(
        {
            "prompt_learning": {
                "algorithm": "gepa",
                "task_app_url": "http://example.com",
                "policy": {
                    "provider": "openai",
                    "inference_mode": "synth_hosted",
                    "model": "gpt-4o-mini",
                    "timeout": 600,
                    "config": {"agent": "opencode"},
                },
                "gepa": {
                    "evaluation": {
                        "train_seeds": [0],
                        "val_seeds": [1],
                    }
                },
            }
        }
    )

    policy = config.policy
    assert policy is not None
    assert policy.config["agent"] == "opencode"
    assert policy.config["model"] == "gpt-4o-mini"
    assert policy.config["timeout"] == 600
