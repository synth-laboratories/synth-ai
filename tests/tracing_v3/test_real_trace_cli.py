#!/usr/bin/env python3
"""
CLI-level regression tests that exercise trace fixtures via Synth AI commands.
"""

import shutil
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from synth_ai.cli.status import status
from synth_ai.cli.traces import traces
from synth_ai.core.tracing_v3.constants import TRACE_DB_BASENAME, canonical_trace_db_name

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "traces"


def _fixture_db(scenario: str) -> Path:
    scenario_dir = FIXTURE_ROOT / scenario
    candidates = sorted(
        scenario_dir.glob(f"{TRACE_DB_BASENAME}*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            scenario_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    if not candidates:
        raise RuntimeError(f"Fixture database missing for scenario '{scenario}' in {scenario_dir}")
    return candidates[0]


def _build_cli(include_traces: bool = False) -> click.Group:
    cli = click.Group()
    cli.add_command(status)
    if include_traces:
        cli.add_command(traces)
    return cli


@pytest.mark.fast
@pytest.mark.skip(
    reason="Status CLI now requires backend HTTP endpoints; local fixture flow removed."
)
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


@pytest.mark.fast
def test_traces_cli_lists_fixture_database(tmp_path):
    """`synth-ai traces` should enumerate trace DB roots built from fixtures."""
    runner = CliRunner()
    cli = _build_cli(include_traces=True)

    root = tmp_path / canonical_trace_db_name().removesuffix(".db") / "dbs"  # type: ignore[attr-defined]
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
