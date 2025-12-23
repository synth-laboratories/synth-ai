from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from synth_ai.sdk.task.in_process_runner import (
    InProcessJobResult,
    merge_dot_overrides,
    resolve_backend_api_base,
    run_in_process_job,
)


def test_resolve_backend_api_base_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear then set multiple keys to ensure priority ordering
    for key in (
        "TARGET_BACKEND_BASE_URL",
        "BACKEND_OVERRIDE",
        "SYNTH_BACKEND_URL",
        "BACKEND_BASE_URL",
        "NEXT_PUBLIC_API_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("BACKEND_BASE_URL", "https://base.example/api")
    monkeypatch.setenv("SYNTH_BACKEND_URL", "https://synth.example")
    monkeypatch.setenv("TARGET_BACKEND_BASE_URL", "https://target.example/")

    resolved = resolve_backend_api_base()
    assert resolved == "https://target.example/api"  # highest priority wins

    # Explicit override arg takes precedence
    resolved_override = resolve_backend_api_base("https://override.example")
    assert resolved_override == "https://override.example/api"


def test_merge_dot_overrides() -> None:
    base = {"prompt_learning": {"gepa": {"rollout": {"budget": 5}}}}
    extra = {"prompt_learning.gepa.rollout.budget": 10, "prompt_learning.policy.model": "foo"}

    merged = merge_dot_overrides(base, extra)

    assert merged["prompt_learning"]["gepa"]["rollout"]["budget"] == 10
    assert merged["prompt_learning"]["policy"]["model"] == "foo"


@pytest.mark.asyncio
async def test_run_in_process_job_prompt_learning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "backend-key")
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "task-key")

    config_path = tmp_path / "config.toml"
    config_path.write_text("[prompt_learning]\nalgorithm='gepa'\n")

    captured: Dict[str, Any] = {}

    class DummyJob:
        def submit(self) -> str:
            return "pl_dummy_job"

        def poll_until_complete(self, **_: Any) -> dict[str, Any]:
            return {"status": "succeeded", "job_id": "pl_dummy_job"}

    def fake_from_config(**kwargs: Any) -> DummyJob:
        captured.update(kwargs)
        return DummyJob()

    monkeypatch.setattr(
        "synth_ai.sdk.task.in_process_runner.PromptLearningJob",
        SimpleNamespace(from_config=fake_from_config),
    )

    class DummyTaskApp:
        def __init__(self, **_: Any) -> None:
            self.url = "https://tunnel.example"
            self.port = 9000

        async def __aenter__(self) -> "DummyTaskApp":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr("synth_ai.sdk.task.in_process_runner.InProcessTaskApp", DummyTaskApp)

    class DummyHealth:
        ok = True
        detail = None
        health_status = 200
        task_info_status = 200

    monkeypatch.setattr(
        "synth_ai.sdk.task.in_process_runner.check_local_api_health",
        lambda url, key: DummyHealth(),
    )

    result = await run_in_process_job(
        job_type="prompt_learning",
        config_path=config_path,
        app="dummy-app",
        poll=True,
        poll_interval=0.01,
        timeout=0.1,
    )

    assert isinstance(result, InProcessJobResult)
    assert result.job_id == "pl_dummy_job"
    assert result.status["status"] == "succeeded"
    # Overrides should include task_url and api key propagation
    assert captured["overrides"]["task_url"] == "https://tunnel.example"
    assert captured["overrides"]["task_app_api_key"] == "task-key"
    pl_overrides = captured["overrides"]["prompt_learning"]
    assert pl_overrides["task_app_url"] == "https://tunnel.example"


@pytest.mark.asyncio
async def test_run_in_process_job_rl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "backend-key")
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "task-key")

    config_path = tmp_path / "config.toml"
    config_path.write_text("[rl]\nname='demo'\n")

    captured: Dict[str, Any] = {}

    class DummyJob:
        def submit(self) -> str:
            return "rl_dummy_job"

        def poll_until_complete(self, **_: Any) -> dict[str, Any]:
            return {"status": "succeeded", "job_id": "rl_dummy_job"}

    def fake_from_config(**kwargs: Any) -> DummyJob:
        captured.update(kwargs)
        return DummyJob()

    monkeypatch.setattr(
        "synth_ai.sdk.task.in_process_runner.RLJob",
        SimpleNamespace(from_config=fake_from_config),
    )

    class DummyTaskApp:
        def __init__(self, **_: Any) -> None:
            self.url = "https://rl-tunnel.example"
            self.port = 9100

        async def __aenter__(self) -> "DummyTaskApp":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr("synth_ai.sdk.task.in_process_runner.InProcessTaskApp", DummyTaskApp)

    class DummyHealth:
        ok = True
        detail = None
        health_status = 200
        task_info_status = 200

    monkeypatch.setattr(
        "synth_ai.sdk.task.in_process_runner.check_local_api_health",
        lambda url, key: DummyHealth(),
    )

    result = await run_in_process_job(
        job_type="rl",
        config_path=config_path,
        app="dummy-app",
        poll=False,
    )

    assert result.job_id == "rl_dummy_job"
    assert result.status["status"] == "submitted"
    assert captured["task_app_url"] == "https://rl-tunnel.example"
    assert captured["overrides"]["task_url"] == "https://rl-tunnel.example"
