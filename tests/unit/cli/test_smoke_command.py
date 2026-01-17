import importlib
from pathlib import Path
from typing import Any

import pytest
import synth_ai.cli.smoke as smoke_module
from click.testing import CliRunner


@pytest.fixture()
def smoke_core_module(monkeypatch: pytest.MonkeyPatch):
    module = importlib.reload(smoke_module)

    monkeypatch.setattr(module, "_ensure_local_libsql", lambda: None)
    monkeypatch.setattr(module, "_refresh_tracing_config", lambda: None)
    monkeypatch.setattr(module, "resolve_trace_db_settings", lambda: ("libsql://local", None))
    return module


def test_smoke_command_invokes_run_smoke_async(
    smoke_core_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = smoke_core_module
    captured: dict[str, Any] = {}

    async def fake_run_smoke_async(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    called_train = False

    async def fake_run_train_step(**kwargs: Any) -> int:
        nonlocal called_train
        called_train = True
        return 0

    monkeypatch.setattr(module, "_run_smoke_async", fake_run_smoke_async)
    monkeypatch.setattr(module, "_run_train_step", fake_run_train_step)

    runner = CliRunner()
    env = {"SYNTH_TRACES_DIR": str(tmp_path)}
    result = runner.invoke(
        module.smoke,
        [
            "--url",
            "http://task.local",
            "--api-key",
            "api-key-123",
            "--env-name",
            "crafter",
            "--policy-name",
            "custom",
            "--model",
            "test-model",
            "--policy",
            "mock",
            "--inference-url",
            "https://api.example.com/v1/chat/completions",
            "--max-steps",
            "5",
            "--return-trace",
            "--no-mock",
            "--mock-backend",
            "synthetic",
            "--mock-port",
            "3210",
            "--rollouts",
            "2",
            "--group-size",
            "3",
            "--batch-size",
            "4",
        ],
        env=env,
    )

    assert result.exit_code == 0
    assert captured["task_app_url"] == "http://task.local"
    assert captured["api_key"] == "api-key-123"
    assert captured["env_name_opt"] == "crafter"
    assert captured["policy_name"] == "custom"
    assert captured["model"] == "test-model"
    assert captured["inference_policy"] == "mock"
    assert captured["inference_url_opt"] == "https://api.example.com/v1/chat/completions"
    assert captured["max_steps"] == 5
    assert captured["return_trace"] is True
    assert captured["use_mock"] is False
    assert captured["mock_backend"] == "synthetic"
    assert captured["mock_port"] == 3210
    assert captured["rollouts"] == 2
    assert captured["group_size"] == 3
    assert captured["batch_size"] == 4
    assert captured["config_path"] is None
    assert called_train is False


def test_smoke_command_parallel_uses_train_step(
    smoke_core_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = smoke_core_module
    called_smoke = False

    async def fake_run_smoke_async(**kwargs: Any) -> int:
        nonlocal called_smoke
        called_smoke = True
        return 0

    captured: dict[str, Any] = {}

    async def fake_run_train_step(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 5

    monkeypatch.setattr(module, "_run_smoke_async", fake_run_smoke_async)
    monkeypatch.setattr(module, "_run_train_step", fake_run_train_step)

    runner = CliRunner()
    env = {"SYNTH_TRACES_DIR": str(tmp_path)}
    result = runner.invoke(
        module.smoke,
        [
            "--url",
            "http://task.local",
            "--api-key",
            "api-key-123",
            "--parallel",
            "2",
        ],
        env=env,
    )

    assert result.exit_code == 5
    assert called_smoke is False
    assert captured["parallel"] == 2
    assert captured["task_app_url"] == "http://task.local"
    assert captured["api_key"] == "api-key-123"
    assert captured["env_name_opt"] is None
    assert captured["policy_name"] == "react"
    assert captured["model"] == "gpt-5-nano"
    assert captured["inference_policy"] is None
    assert captured["inference_url_opt"] is None
    assert captured["max_steps"] == 3
    assert captured["return_trace"] is False
    assert captured["use_mock"] is True
    assert captured["mock_backend"] == "synthetic"
    assert captured["mock_port"] == 0
    assert captured["config_path"] is None
