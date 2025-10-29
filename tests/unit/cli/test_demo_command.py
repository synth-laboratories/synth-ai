from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import click
import pytest
from click.testing import CliRunner

MODULE_PATH = Path(__file__).resolve().parents[3] / "synth_ai" / "cli" / "commands" / "demo" / "core.py"


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def demo_core_module(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ModuleType]:
    fake_cli = ModuleType("synth_ai.demos.core.cli")
    fake_cli.init = mock.Mock(return_value=None)
    fake_cli.deploy = mock.Mock(return_value=None)
    fake_cli.run = mock.Mock(return_value=None)
    fake_cli.setup = mock.Mock(return_value=None)
    monkeypatch.setitem(sys.modules, "synth_ai.demos.core.cli", fake_cli)

    spec = importlib.util.spec_from_file_location("synth_ai.cli.commands.demo.core", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_demo_cli", fake_cli)
    return module, fake_cli


def _invoke(runner: CliRunner, command, args: list[str] | None = None, **kwargs):
    return runner.invoke(command, args or [], **kwargs)


def test_demo_default_invokes_init(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    result = _invoke(runner, demo_core.command)
    assert result.exit_code == 0
    fake_cli.init.assert_called_once_with(force=False)


def test_demo_force_flag_passes_through(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    result = _invoke(runner, demo_core.command, ["--force"])
    assert result.exit_code == 0
    fake_cli.init.assert_called_once_with(force=True)


def test_demo_list_without_scripts_prints_message(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    with mock.patch.object(demo_core, "_find_demo_scripts", return_value=[]):
        result = _invoke(runner, demo_core.command, ["--list"])
    assert result.exit_code == 0
    assert "No run_demo.sh scripts found" in result.output
    fake_cli.init.assert_not_called()


def test_demo_list_runs_selected_script(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    with runner.isolated_filesystem():
        script_path = Path("examples/demo/run_demo.sh")
        script_path.parent.mkdir(parents=True)
        script_path.write_text("#!/bin/bash\necho demo\n", encoding="utf-8")
        expected = str(script_path)

        with mock.patch("synth_ai.cli.commands.demo.core.subprocess.run") as run_mock:
            result = _invoke(runner, demo_core.command, ["--list"], input="1\n")

    assert result.exit_code == 0
    run_mock.assert_called_once()
    args, kwargs = run_mock.call_args
    assert args[0][0] == "bash"
    assert Path(args[0][1]).name == Path(expected).name
    assert kwargs == {"check": True}
    fake_cli.init.assert_not_called()


def test_demo_deploy_subcommand_calls_cli(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    result = _invoke(runner, demo_core.command, ["deploy", "--name", "custom"])
    assert result.exit_code == 0
    fake_cli.deploy.assert_called_once_with(local=False, app=None, name="custom", script=None)


def test_demo_configure_subcommand_calls_run(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    result = _invoke(runner, demo_core.command, ["configure"])
    assert result.exit_code == 0
    fake_cli.run.assert_called_once_with()


def test_demo_setup_subcommand_calls_setup(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    result = _invoke(runner, demo_core.command, ["setup"])
    assert result.exit_code == 0
    fake_cli.setup.assert_called_once_with()


def test_demo_run_subcommand_passes_options(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    result = _invoke(
        runner,
        demo_core.command,
        [
            "run",
            "--batch-size",
            "8",
            "--group-size",
            "32",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--timeout",
            "900",
        ],
    )
    assert result.exit_code == 0
    fake_cli.run.assert_called_once_with(
        batch_size=8,
        group_size=32,
        model="Qwen/Qwen3-0.6B",
        timeout=900,
    )


def test_setup_alias_uses_demo_setup(runner: CliRunner, demo_core_module) -> None:
    demo_core, fake_cli = demo_core_module
    result = _invoke(runner, demo_core.setup_alias)
    assert result.exit_code == 0
    fake_cli.setup.assert_called_once_with()


def test_run_demo_command_bubbles_non_zero(demo_core_module) -> None:
    demo_core, _ = demo_core_module

    def _fake() -> int:
        return 5

    with pytest.raises(click.exceptions.Exit) as excinfo:
        demo_core._run_demo_command(_fake)  # type: ignore[attr-defined]
    assert excinfo.value.exit_code == 5
