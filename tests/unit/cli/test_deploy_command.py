from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import click
from click.testing import CliRunner

from synth_ai.cli import deploy as deploy_module


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_local_runtime_invokes_uvicorn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    task_app = tmp_path / "local_app.py"
    task_app.write_text("app = object()\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_uvicorn(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(deploy_module, "deploy_uvicorn_app", fake_uvicorn)

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "local",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["config"]
    assert cfg.task_app_path == task_app
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.trace is True


def test_modal_runtime_invokes_deploy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    task_app = tmp_path / "task_app.py"
    task_app.write_text("app = object()\n", encoding="utf-8")
    modal_app = tmp_path / "modal_app.py"
    modal_app.write_text("app = object()\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_deploy(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(deploy_module, "deploy_modal_app", fake_deploy)
    monkeypatch.setattr(deploy_module, "get_default_modal_bin_path", lambda: modal_app)

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "modal",
            "--modal-app",
            str(modal_app),
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["config"]
    assert cfg.task_app_path == task_app
    assert cfg.modal_app_path == modal_app
    assert cfg.modal_bin_path == modal_app
    assert cfg.cmd_arg == "deploy"


def test_modal_requires_modal_app(tmp_path: Path) -> None:
    task_app = tmp_path / "task_app.py"
    task_app.write_text("app = object()\n", encoding="utf-8")

    with pytest.raises(click.ClickException, match="Modal app path required"):
        deploy_module.deploy_cmd.callback(  # type: ignore[attr-defined]
            task_app_path=task_app,
            runtime="modal",
            trace=True,
            host="127.0.0.1",
            port=8000,
            task_app_name=None,
            cmd_arg="deploy",
            modal_app_path=None,
            modal_bin_path=None,
            dry_run=False,
        )


def test_modal_serve_and_dry_run_conflict(tmp_path: Path, runner: CliRunner) -> None:
    task_app = tmp_path / "task_app.py"
    task_app.write_text("app = object()\n", encoding="utf-8")
    modal_app = tmp_path / "modal_app.py"
    modal_app.write_text("app = object()\n", encoding="utf-8")

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "modal",
            "--modal-app",
            str(modal_app),
            "--modal-mode",
            "serve",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "--modal-mode=serve cannot be combined with --dry-run"

def test_modal_mode_serve_sets_cmd_arg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    task_app = tmp_path / "task_app.py"
    task_app.write_text("app = object()\n", encoding="utf-8")
    modal_app = tmp_path / "modal_app.py"
    modal_app.write_text("app = object()\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_deploy(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(deploy_module, "deploy_modal_app", fake_deploy)
    monkeypatch.setattr(deploy_module, "get_default_modal_bin_path", lambda: modal_app)

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "modal",
            "--modal-app",
            str(modal_app),
            "--modal-mode",
            "serve",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].cmd_arg == "serve"


def test_modal_cli_default_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    task_app = tmp_path / "task_app.py"
    task_app.write_text("app = object()\n", encoding="utf-8")
    modal_app = tmp_path / "modal_app.py"
    modal_app.write_text("app = object()\n", encoding="utf-8")

    monkeypatch.setattr(deploy_module, "get_default_modal_bin_path", lambda: None)

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "modal",
            "--modal-app",
            str(modal_app),
        ],
    )

    assert result.exit_code != 0
    assert "Modal CLI not found" in result.output


def test_modal_cli_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    task_app = tmp_path / "task_app.py"
    task_app.write_text("app = object()\n", encoding="utf-8")
    modal_app = tmp_path / "modal_app.py"
    modal_app.write_text("app = object()\n", encoding="utf-8")
    modal_cli = tmp_path / "modal"
    modal_cli.write_text("#!/bin/sh\n", encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_deploy(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(deploy_module, "deploy_modal_app", fake_deploy)

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "--task-app",
            str(task_app),
            "--runtime",
            "modal",
            "--modal-app",
            str(modal_app),
            "--modal-cli",
            str(modal_cli),
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["config"]
    assert cfg.modal_bin_path == modal_cli
