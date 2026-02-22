from synth_ai.sdk.optimization.policy.job import (
    PolicyOptimizationJob,
    PolicyOptimizationJobConfig,
    _extract_container_url,
    _infer_container_url,
)


def test_extract_container_url_supports_container_url_base_alias() -> None:
    payload = {"policy_optimization": {"container_url_base": "https://container.example.com"}}
    assert _extract_container_url(payload) == "https://container.example.com"


def test_extract_container_url_supports_top_level_aliases() -> None:
    payload = {"task_url": "https://task.example.com"}
    assert _extract_container_url(payload) == "https://task.example.com"


def test_from_dict_auto_skip_health_check_detects_tunnel_alias() -> None:
    job = PolicyOptimizationJob.from_dict(
        config_dict={
            "policy_optimization": {
                "algorithm": "gepa",
                "container_url_base": "https://abc.trycloudflare.com",
            }
        },
        backend_url="https://api.example.com",
        api_key="sk_test",
        container_api_key="env_test",
        skip_health_check=False,
    )
    assert job._skip_health_check is True


def test_infer_container_url_accepts_dotted_override_keys() -> None:
    cfg = PolicyOptimizationJobConfig(
        config_dict={"policy_optimization": {"algorithm": "gepa"}},
        backend_url="https://api.example.com",
        api_key="sk_test",
        container_api_key="env_test",
        overrides={"prompt_learning.container_url": "https://override.example.com"},
    )
    assert _infer_container_url(cfg) == "https://override.example.com"


def test_to_prompt_learning_config_does_not_mutate_input_config() -> None:
    source = {
        "policy_optimization": {
            "container_url": "https://container.example.com",
            "policy": {"model": "gpt-4.1-nano"},
        },
        "extra": {"flag": True},
    }
    cfg = PolicyOptimizationJobConfig(
        config_dict=source,
        backend_url="https://api.example.com",
        api_key="sk_test",
        container_api_key="env_test",
    )

    converted = cfg.to_prompt_learning_config()
    assert "prompt_learning" in converted
    assert "policy_optimization" not in converted
    assert "policy_optimization" in source
    assert "prompt_learning" not in source

    converted["prompt_learning"]["policy"]["model"] = "changed"
    assert source["policy_optimization"]["policy"]["model"] == "gpt-4.1-nano"
