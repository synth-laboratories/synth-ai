from __future__ import annotations

from typing import Any, Literal

import pytest
from click.testing import CliRunner
import click

from synth_ai.cli.commands.deploy import core as deploy_core
from synth_ai.cli.commands.deploy.errors import (
    EnvFileDiscoveryError,
    EnvKeyPreflightError,
    EnvironmentKeyLoadError,
    MissingEnvironmentApiKeyError,
    ModalCliResolutionError,
    ModalExecutionError,
    TaskAppNotFoundError,
)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_deploy_defaults_to_modal_runtime(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_modal_runtime(
        app_id: str | None,
        *,
        command: str,
        modal_name: str | None,
        dry_run: bool,
        modal_cli: str,
        env_file: tuple[str, ...],
        use_demo_dir: bool = True,
    ) -> None:
        captured.update(
            {
                "app_id": app_id,
                "command": command,
                "modal_name": modal_name,
                "dry_run": dry_run,
                "modal_cli": modal_cli,
                "env_file": env_file,
                "use_demo_dir": use_demo_dir,
            }
        )

    monkeypatch.setattr(deploy_core, "run_modal_runtime", fake_modal_runtime)
    result = runner.invoke(deploy_core.command, ["math-single-step"])
    assert result.exit_code == 0, result.output
    assert captured["command"] == "deploy"
    assert captured["app_id"] == "math-single-step"
    assert captured["use_demo_dir"] is True


def test_deploy_uvicorn_runtime(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_uvicorn_runtime(
        app_id: str | None,
        host: str,
        port: int | None,
        env_file: tuple[str, ...],
        reload_flag: bool,
        force: bool,
        trace_dir: str | None,
        trace_db: str | None,
    ) -> None:
        captured.update(
            {
                "app_id": app_id,
                "host": host,
                "port": port,
                "env_file": env_file,
                "reload_flag": reload_flag,
                "force": force,
                "trace_dir": trace_dir,
                "trace_db": trace_db,
            }
        )

    monkeypatch.setattr(deploy_core, "run_uvicorn_runtime", fake_uvicorn_runtime)
    result = runner.invoke(
        deploy_core.command,
        [
            "local-app",
            "--runtime",
            "uvicorn",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--reload",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["app_id"] == "local-app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9001
    assert captured["reload_flag"] is True


def test_deploy_rejects_uvicorn_only_options_in_modal_mode(runner: CliRunner) -> None:
    result = runner.invoke(
        deploy_core.command,
        [
            "math-single-step",
            "--host",
            "127.0.0.1",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be used with --runtime=modal" in result.output


def test_deploy_rejects_dry_run_for_modal_serve(runner: CliRunner) -> None:
    result = runner.invoke(
        deploy_core.command,
        [
            "--modal-mode",
            "serve",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "--dry-run is not supported with --modal-mode=serve" in result.output


def test_modal_serve_command_invokes_runtime(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def fake_modal_runtime(
        app_id: str | None,
        *,
        command: str,
        modal_name: str | None,
        dry_run: bool,
        modal_cli: str,
        env_file: tuple[str, ...],
        use_demo_dir: bool,
    ) -> None:
        captured.update(
            {
                "app_id": app_id,
                "command": command,
                "modal_name": modal_name,
                "dry_run": dry_run,
                "modal_cli": modal_cli,
                "env_file": env_file,
                "use_demo_dir": use_demo_dir,
            }
        )

    monkeypatch.setattr(deploy_core, "run_modal_runtime", fake_modal_runtime)
    result = runner.invoke(
        deploy_core.modal_serve_command,
        [
            "math-single-step",
            "--modal-cli",
            "echo",
            "--name",
            "demo",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["command"] == "serve"
    assert captured["app_id"] == "math-single-step"
    assert captured["modal_name"] == "demo"
    assert captured["use_demo_dir"] is False


def _invoke_with_error(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    *,
    runtime: Literal["uvicorn", "modal"],
    invoke_args: list[str],
) -> click.testing.Result:
    if runtime == "uvicorn":
        monkeypatch.setattr(
            deploy_core,
            "run_uvicorn_runtime",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )
    else:
        monkeypatch.setattr(
            deploy_core,
            "run_modal_runtime",
            lambda *args, **kwargs: (_ for _ in ()).throw(error),
        )
    runner = CliRunner()
    return runner.invoke(deploy_core.command, invoke_args)


@pytest.mark.parametrize(
    ("error", "expected", "runtime", "args"),
    [
        (
            MissingEnvironmentApiKeyError(hint="Supply via --env-file"),
            "Supply via --env-file",
            "uvicorn",
            ["--runtime", "uvicorn"],
        ),
        (
            EnvironmentKeyLoadError(path=".env"),
            "Failed to persist or reload ENVIRONMENT_API_KEY from .env",
            "uvicorn",
            ["--runtime", "uvicorn"],
        ),
        (
            EnvFileDiscoveryError(attempted=(".env",), hint="run setup"),
            "Unable to locate a usable env file (.env)",
            "uvicorn",
            ["--runtime", "uvicorn"],
        ),
        (
            TaskAppNotFoundError(app_id="foo", available=("bar", "baz")),
            "Could not find task app 'foo'. Available choices: bar, baz.",
            "uvicorn",
            ["--runtime", "uvicorn"],
        ),
        (
            ModalCliResolutionError(detail="Modal CLI not found"),
            "Modal CLI not found",
            "modal",
            [],
        ),
        (
            ModalExecutionError(command="deploy", exit_code=2),
            "Modal deploy exited with status 2",
            "modal",
            [],
        ),
        (
            EnvKeyPreflightError(detail="Upload failed"),
            "Upload failed",
            "modal",
            [],
        ),
    ],
)
def test_deploy_formats_known_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected: str,
    runtime: Literal["uvicorn", "modal"],
    args: list[str],
) -> None:
    result = _invoke_with_error(monkeypatch, error, runtime=runtime, invoke_args=args)
    assert result.exit_code != 0
    assert expected in result.output


def test_deploy_translates_env_click_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    error = click.ClickException("No .env file discovered automatically. Pass --env-file")
    result = _invoke_with_error(
        monkeypatch,
        error,
        runtime="uvicorn",
        invoke_args=["--runtime", "uvicorn"],
    )
    assert result.exit_code != 0
    assert "Unable to locate a usable env file" in result.output


def test_deploy_translates_modal_cli_click_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    error = click.ClickException("Modal CLI not found (looked for 'modal')")
    result = _invoke_with_error(monkeypatch, error, runtime="modal", invoke_args=[])
    assert result.exit_code != 0
    assert "Modal CLI not found" in result.output


def test_modal_serve_formats_task_app_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        deploy_core,
        "run_modal_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            TaskAppNotFoundError(app_id="foo", available=("bar",))
        ),
    )
    runner = CliRunner()
    result = runner.invoke(deploy_core.modal_serve_command, ["foo"])
    assert result.exit_code != 0
    assert "Could not find task app 'foo'. Available choices: bar." in result.output
