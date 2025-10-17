#!/usr/bin/env python3
"""
CLI-level regression tests that exercise trace fixtures via Synth AI commands.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click
from click.testing import CliRunner

from synth_ai.cli.status import register as register_status
from synth_ai.cli.traces import register as register_traces


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "traces"


def _fixture_db(scenario: str) -> Path:
    db_path = FIXTURE_ROOT / scenario / "synth_ai.db"
    if not db_path.exists():
        raise RuntimeError(f"Fixture database missing for scenario '{scenario}': {db_path}")
    return db_path


def _build_cli(include_traces: bool = False) -> click.Group:
    cli = click.Group()
    register_status(cli)
    if include_traces:
        register_traces(cli)
    return cli


def test_status_cli_reports_fixture_counts():
    """`synth-ai status` should read fixture DBs and emit aggregate counts."""
    runner = CliRunner()
    cli = _build_cli()
    db_path = _fixture_db("env_rollout")
    result = runner.invoke(
        cli,
        [
            "status",
            "--url",
            f"sqlite+aiosqlite:///{db_path}",
            "--service-url",
            "http://127.0.0.1:9",
        ],
    )
    assert result.exit_code == 0, result.output
    # Expect totals block containing Sessions count (value taken from manifest).
    assert "Sessions: 2" in result.output
    assert "Events:" in result.output


def test_traces_cli_lists_fixture_database(tmp_path):
    """`synth-ai traces` should enumerate trace DB roots built from fixtures."""
    runner = CliRunner()
    cli = _build_cli(include_traces=True)

    root = tmp_path / "synth_ai.db" / "dbs"
    fixture_dir = root / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_fixture_db("chat_small"), fixture_dir / "data")

    result = runner.invoke(
        cli,
        ["traces"],
        env={"SYNTH_TRACES_ROOT": str(root)},
    )
    assert result.exit_code == 0, result.output
    # Output should reference the synthetic DB directory name
    assert "fixture" in result.output
    assert "Trace Databases" in result.output
