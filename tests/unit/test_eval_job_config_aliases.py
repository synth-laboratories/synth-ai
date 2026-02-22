from synth_ai.sdk.eval.job import EvalJobConfig


def test_eval_job_config_respects_container_url_without_duplicate_aliasing() -> None:
    cfg = EvalJobConfig(
        container_url="http://127.0.0.1:8103",
        api_key="sk_test",
        backend_url="https://api.example.com",
        container_api_key="env_test",
        env_name="banking77",
        seeds=[0],
        policy_config={"model": "gpt-4.1-nano", "provider": "openai"},
    )
    assert cfg.container_url == "http://127.0.0.1:8103"


def test_eval_job_config_container_key_alias_sets_container_api_key() -> None:
    cfg = EvalJobConfig(
        container_url="http://127.0.0.1:8103",
        api_key="sk_test",
        backend_url="https://api.example.com",
        container_key="legacy_container_key",
        env_name="banking77",
        seeds=[0],
        policy_config={"model": "gpt-4.1-nano", "provider": "openai"},
    )
    assert cfg.container_api_key == "legacy_container_key"
