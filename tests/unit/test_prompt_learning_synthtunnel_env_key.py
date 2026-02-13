"""PromptLearningJobConfig behavior for SynthTunnel container_url."""

import pytest
from synth_ai.sdk.optimization.internal.prompt_learning import PromptLearningJobConfig


@pytest.mark.unit
def test_synthtunnel_job_config_provisions_backend_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def _fake_ensure_container_auth(*, backend_base: str | None = None, synth_api_key: str | None = None, **kwargs):  # type: ignore[no-untyped-def]
        called["backend_base"] = backend_base
        called["synth_api_key"] = synth_api_key
        return "env_test_key"

    monkeypatch.setattr(
        "synth_ai.sdk.optimization.internal.prompt_learning.ensure_container_auth",
        _fake_ensure_container_auth,
    )

    cfg = PromptLearningJobConfig(
        config_dict={"prompt_learning": {"container_url": "http://localhost:8000/s/rt_test"}},
        backend_url="http://localhost:8000",
        api_key="sk_test",
        container_worker_token="worker_token_test",
    )

    assert called["backend_base"] == "http://localhost:8000"
    assert called["synth_api_key"] == "sk_test"
    assert cfg.container_api_key is None
