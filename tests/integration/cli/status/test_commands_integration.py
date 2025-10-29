from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from synth_ai.cli.commands.status.subcommands.jobs import jobs_group
from synth_ai.cli.commands.status.subcommands.summary import summary_command


DEV_BASE_URL = os.getenv("SYNTH_STATUS_DEV_BASE_URL", "https://synth-backend-dev-docker.onrender.com/api")
DEV_API_KEY = os.getenv("SYNTH_STATUS_DEV_API_KEY")


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.integration
def test_jobs_list_real_backend(runner: CliRunner):
    if not DEV_API_KEY:
        pytest.skip("SYNTH_STATUS_DEV_API_KEY not set")

    result = runner.invoke(
        jobs_group,
        ["list", "--base-url", DEV_BASE_URL, "--api-key", DEV_API_KEY, "--limit", "1", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert "job_" in result.output


@pytest.mark.integration
def test_summary_real_backend(runner: CliRunner):
    if not DEV_API_KEY:
        pytest.skip("SYNTH_STATUS_DEV_API_KEY not set")

    result = runner.invoke(
        summary_command,
        ["--base-url", DEV_BASE_URL, "--api-key", DEV_API_KEY, "--limit", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "Training Jobs" in result.output
