"""SYN-5737: the hosted (staging/prod) journey entrypoint synth-dev's hosted_journey invokes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.acceptance import hosted_journey, journey_main
from scripts.acceptance.journeys import Step, StepReceipt, Verdict
from scripts.acceptance.surfaces import CLISurface, SurfaceName

ORIGIN = "https://core-api-staging-staging-acef.up.railway.app"


def argv(tmp_path: Path, origin: str = ORIGIN) -> list[str]:
    return [
        "--journey", "steps", "--environment", "staging", "--origin", origin,
        "--driver", "sdk", "--steps", str(Step.CROSS_ORG_404),
        "--receipt-dir", str(tmp_path / "sdk"), "--execute",
    ]  # fmt: skip


def test_cli_surface_uses_the_named_profile(tmp_path):
    surface = CLISurface(
        python=Path(sys.executable), environment={}, work_directory=tmp_path, profile="staging"
    )
    assert surface.argv(["runs", "get"])[3:5] == ["--env", "staging"]


@pytest.mark.parametrize("origin", ["http://127.0.0.1:1", ORIGIN + "/v1", "https://other.example"])
def test_origin_must_equal_the_named_profile(tmp_path, monkeypatch, origin):
    monkeypatch.setenv("SYNTH_STAGING_API_BASE_URL", ORIGIN)
    monkeypatch.setenv("SYNTH_API_KEY", "fixture-a")
    with pytest.raises(SystemExit):
        hosted_journey.main(argv(tmp_path, origin))
    assert not (tmp_path / "sdk").exists()


def test_driver_receipt_is_written_with_profile_bound_surfaces(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SYNTH_STAGING_API_BASE_URL", ORIGIN)
    monkeypatch.setenv("SYNTH_API_KEY", "fixture-a")
    seen = {}

    def fake_run_all(options, make, primary, origin, secrets, steps, receipts):
        seen.update(origin=origin, profile=make.profile)
        receipts.append(StepReceipt(steps[0], options.driver, Verdict.PASS, ""))

    monkeypatch.setattr(journey_main, "run_all", fake_run_all)
    assert hosted_journey.main(argv(tmp_path)) == 0
    assert seen == {"origin": ORIGIN, "profile": "staging"}
    document = json.loads((tmp_path / "sdk/driver-receipt.json").read_text())
    assert document["driver"] == SurfaceName.SDK.value
    assert [row["verdict"] for row in document["steps"]] == ["Pass"]
    assert "fixture-a" not in capsys.readouterr().out
