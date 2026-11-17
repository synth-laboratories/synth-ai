"""Hosted source binding controls; no provider or installed service is called."""

import sys
from pathlib import Path

import pytest

from scripts.acceptance.hosted_journey import main, validate_target
from scripts.acceptance.j1 import CLIDriver
from scripts.acceptance.surfaces import CLISurface
from synth_ai.environments import Environment


def test_hosted_plan_performs_no_execution(tmp_path, capsys):
    assert (
        main(
            [
                "--environment",
                "staging",
                "--origin",
                "https://staging.example.test",
                "--driver",
                "sdk",
                "--journey",
                "steps",
                "--steps",
                "off_whitelist_sku_refusal",
                "--receipt-dir",
                str(tmp_path / "never-created"),
            ]
        )
        == 0
    )
    assert '"network_calls": 0' in capsys.readouterr().out
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize(
    "origin", ["http://localhost:1", "https://user:secret@host", "https://host/path"]
)
def test_hosted_origin_refuses_before_work(origin):
    with pytest.raises(ValueError):
        validate_target(Environment.STAGING, origin, None)


def test_cli_adapters_keep_local_default_and_explicit_hosted_profile(tmp_path):
    surface = CLISurface(
        python=Path(sys.executable),
        environment={},
        work_directory=tmp_path,
        profile=Environment.STAGING,
    )
    assert surface.argv(["whoami"])[3:5] == ["--env", "staging"]
    local = CLISurface(python=Path(sys.executable), environment={}, work_directory=tmp_path)
    assert "local" in local.argv(["whoami"])
    driver = CLIDriver(
        python=Path(sys.executable),
        environment={},
        working_directory=tmp_path,
        profile=Environment.PROD,
    )
    assert driver.profile == Environment.PROD
