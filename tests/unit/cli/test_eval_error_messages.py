from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner
from synth_ai.cli.task_apps import eval_command


def test_eval_config_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            eval_command,
            [
                "demo-app",
                "--model",
                "demo-model",
                "--config",
                str(missing),
                "--trace-db",
                "none",
            ],
        )
    assert result.exit_code != 0
    assert "Eval config not found" in result.output


def test_eval_metadata_filter_format_error(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            eval_command,
            [
                "demo-app",
                "--model",
                "demo-model",
                "--metadata",
                "invalid-entry",
                "--trace-db",
                "none",
            ],
        )
    assert result.exit_code != 0
    assert "Metadata filter" in result.output
