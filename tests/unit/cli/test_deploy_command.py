from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import click
from click.testing import CliRunner

from synth_ai.cli.deploy import deploy_cmd
from synth_ai.task_app_cfgs import LocalTaskAppConfig, ModalTaskAppConfig


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _write_stub(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_deploy_local_runtime_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_deploy(cfg: LocalTaskAppConfig) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr("synth_ai.cli.deploy.deploy_uvicorn_app", fake_deploy)

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")

    result = runner.invoke(
        deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "local",
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--no-trace",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["cfg"]
    assert isinstance(cfg, LocalTaskAppConfig)
    assert cfg.task_app_path == task_app
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9001
    assert cfg.trace is False


def test_deploy_modal_runtime_invokes_modal(monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_modal(cfg: ModalTaskAppConfig) -> None:
        captured["cfg"] = cfg

    modal_cli_path = tmp_path / "modal"
    monkeypatch.setattr("synth_ai.cli.deploy.deploy_modal_app", fake_modal)
    monkeypatch.setattr("synth_ai.cli.deploy.get_default_modal_bin_path", lambda: modal_cli_path)

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")
    modal_app = _write_stub(tmp_path / "modal_app.py", "from modal import App\nApp('demo')\n")

    result = runner.invoke(
        deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "modal",
            "--modal-app",
            str(modal_app),
            "--name",
            "demo-app",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["cfg"]
    assert isinstance(cfg, ModalTaskAppConfig)
    assert cfg.task_app_path == task_app
    assert cfg.modal_app_path == modal_app
    assert cfg.task_app_name == "demo-app"
    assert cfg.cmd_arg == "deploy"
    assert cfg.dry_run is False
    assert cfg.modal_bin_path == modal_cli_path


def test_deploy_modal_requires_modal_app_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("synth_ai.cli.deploy.get_default_modal_bin_path", lambda: tmp_path / "modal")
    monkeypatch.setattr("synth_ai.cli.deploy.deploy_modal_app", lambda cfg: None)

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")

    with pytest.raises(click.ClickException) as exc:
        deploy_cmd.callback(task_app_path=task_app, runtime="modal")

    assert "Modal app path required" in str(exc.value)


def test_deploy_modal_disallows_dry_run_with_serve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("synth_ai.cli.deploy.get_default_modal_bin_path", lambda: tmp_path / "modal")
    monkeypatch.setattr("synth_ai.cli.deploy.deploy_modal_app", lambda cfg: None)

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")
    modal_app = _write_stub(tmp_path / "modal_app.py", "from modal import App\nApp('demo')\n")
    modal_cli = tmp_path / "modal"

    with pytest.raises(click.ClickException) as exc:
        deploy_cmd.callback(
            task_app_path=task_app,
            runtime="modal",
            modal_app_path=modal_app,
            cmd_arg="serve",
            dry_run=True,
            modal_bin_path=modal_cli,
        )

    assert "--modal-mode=serve cannot be combined with --dry-run" in str(exc.value)
