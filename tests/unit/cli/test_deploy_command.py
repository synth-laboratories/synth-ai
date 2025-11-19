from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from synth_ai.cfgs import LocalDeployCfg, ModalDeployCfg


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _write_stub(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def _reload_deploy(monkeypatch: pytest.MonkeyPatch):
    import synth_ai.cli.deploy as deploy_module

    deploy_module = importlib.reload(deploy_module)
    # After reload, patch the imported functions in the deploy module's namespace
    monkeypatch.setattr(deploy_module, "validate_task_app", lambda path: None)
    return deploy_module


def test_deploy_local_runtime_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_deploy(cfg: LocalDeployCfg) -> None:
        captured["cfg"] = cfg

    monkeypatch.setenv("SYNTH_API_KEY", "synth-key")
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-key")

    deploy_module = _reload_deploy(monkeypatch)

    # Patch after reload - the reload imports deploy_app_uvicorn into the module namespace
    monkeypatch.setattr("synth_ai.cli.deploy.deploy_app_uvicorn", fake_deploy)

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "local",
            str(task_app),
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--no-trace",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["cfg"]
    assert isinstance(cfg, LocalDeployCfg)
    assert cfg.task_app_path == task_app
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9001
    assert cfg.trace is False


def test_deploy_modal_runtime_invokes_modal(monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_modal(cfg: ModalDeployCfg) -> None:
        captured["cfg"] = cfg

    modal_cli_path = tmp_path / "modal"
    modal_cli_path.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("SYNTH_API_KEY", "synth-key")
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-key")
    deploy_module = _reload_deploy(monkeypatch)
    monkeypatch.setattr("synth_ai.cli.deploy.deploy_app_modal", fake_modal)
    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")
    modal_app = _write_stub(tmp_path / "modal_app.py", "from modal import App\nApp('demo')\n")

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "modal",
            str(task_app),
            "--modal-app",
            str(modal_app),
            "--modal-cli",
            str(modal_cli_path),
            "--name",
            "demo-app",
        ],
    )

    if result.exit_code != 0:
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.output}")
        if result.exception:
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

    assert result.exit_code == 0, result.output
    assert "cfg" in captured, f"fake_modal not called. Output: {result.output}"
    cfg = captured["cfg"]
    assert isinstance(cfg, ModalDeployCfg)
    assert cfg.task_app_path == task_app
    assert cfg.modal_app_path == modal_app
    assert cfg.modal_app_name == "demo-app"
    assert cfg.cmd_arg == "deploy"
    assert cfg.dry_run is False
    assert cfg.modal_bin_path == modal_cli_path


def test_deploy_modal_requires_modal_app_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.setenv("SYNTH_API_KEY", "synth-key")
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-key")
    deploy_module = _reload_deploy(monkeypatch)
    modal_cli = tmp_path / "modal"
    modal_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")

    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "modal",
            str(task_app),
            "--modal-cli",
            str(modal_cli),
        ],
    )

    assert result.exit_code == 0
    assert "modal_app is required" in result.output


def test_deploy_modal_disallows_dry_run_with_serve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = CliRunner()
    captured: dict[str, any] = {}

    def fake_modal(cfg: ModalDeployCfg) -> None:
        print(f"fake_modal called: cmd_arg={cfg.cmd_arg}, dry_run={cfg.dry_run}")
        captured["cfg"] = cfg

    monkeypatch.setenv("SYNTH_API_KEY", "synth-key")
    monkeypatch.setenv("ENVIRONMENT_API_KEY", "test-key")
    deploy_module = _reload_deploy(monkeypatch)
    monkeypatch.setattr("synth_ai.cli.deploy.deploy_app_modal", fake_modal)
    modal_cli = tmp_path / "modal"
    modal_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")
    modal_app = _write_stub(tmp_path / "modal_app.py", "from modal import App\nApp('demo')\n")
    result = runner.invoke(
        deploy_module.deploy_cmd,
        [
            "modal",
            str(task_app),
            "--modal-app",
            str(modal_app),
            "--modal-cli",
            str(modal_cli),
            "--modal-mode",
            "serve",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "--modal-mode serve` cannot be used with `--dry-run" in result.output
