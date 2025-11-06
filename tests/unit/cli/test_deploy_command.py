from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

# Import deploy_cmd after potential monkeypatching
from synth_ai.cli.deploy import deploy_cmd
from synth_ai.cfgs import LocalDeployCfg, ModalDeployCfg


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _write_stub(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_deploy_local_runtime_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_deploy(cfg: LocalDeployCfg) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr("synth_ai.uvicorn.deploy_app_uvicorn", fake_deploy)
    # Reload the deploy module to pick up the patched function
    import synth_ai.cli.deploy
    importlib.reload(synth_ai.cli.deploy)
    from synth_ai.cli.deploy import deploy_cmd as reloaded_deploy_cmd

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")

    result = runner.invoke(
        reloaded_deploy_cmd,
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
    monkeypatch.setattr("synth_ai.utils.modal.deploy_modal_app", fake_modal)
    monkeypatch.setattr("synth_ai.utils.modal.get_default_modal_bin_path", lambda: modal_cli_path)
    # Reload the deploy module to pick up the patched function
    import synth_ai.cli.deploy
    importlib.reload(synth_ai.cli.deploy)
    from synth_ai.cli.deploy import deploy_cmd as reloaded_deploy_cmd

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")
    modal_app = _write_stub(tmp_path / "modal_app.py", "from modal import App\nApp('demo')\n")

    result = runner.invoke(
        reloaded_deploy_cmd,
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
    assert isinstance(cfg, ModalDeployCfg)
    assert cfg.task_app_path == task_app
    assert cfg.modal_app_path == modal_app
    assert cfg.task_app_name == "demo-app"
    assert cfg.cmd_arg == "deploy"
    assert cfg.dry_run is False
    assert cfg.modal_bin_path == modal_cli_path


def test_deploy_modal_requires_modal_app_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("synth_ai.utils.modal.get_default_modal_bin_path", lambda: tmp_path / "modal")
    monkeypatch.setattr("synth_ai.utils.modal.deploy_modal_app", lambda cfg: None)

    task_app = _write_stub(tmp_path / "task_app.py", "app = object()\n")

    with pytest.raises(click.ClickException) as exc:
        deploy_cmd.callback(task_app_path=task_app, runtime="modal", env_file=())

    assert "Modal app path required" in str(exc.value)


def test_deploy_modal_disallows_dry_run_with_serve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("synth_ai.utils.modal.get_default_modal_bin_path", lambda: tmp_path / "modal")
    monkeypatch.setattr("synth_ai.utils.modal.deploy_modal_app", lambda cfg: None)

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
            env_file=(),
        )

    assert "--modal-mode=serve cannot be combined with --dry-run" in str(exc.value)
