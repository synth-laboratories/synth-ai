"""scripts/qualify_release.py binds one exact artifact and writes a receipt on failure too."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qualify_release", ROOT / "scripts/qualify_release.py"
)
qualify = importlib.util.module_from_spec(spec)
sys.modules["qualify_release"] = qualify  # dataclasses resolve their module
spec.loader.exec_module(qualify)


def test_substituted_wheel_fails_at_identity_and_skips_the_rest(tmp_path):
    wheel = tmp_path / "synth_ai-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"not the qualified bytes")
    out = tmp_path / "receipt"
    code = qualify.main(
        [
            "--core-checkout",
            str(tmp_path),
            "--core-revision",
            "0" * 40,
            "--out",
            str(out),
            "--wheel",
            str(wheel),
            "--expect-sha256",
            "406b5a3cfa2c5333c09212046352a79d2dcc0777d9957b5ab5bd7ef791b1c305",
        ]
    )
    receipt = json.loads((out / "qualification.json").read_text())
    assert code == 1 and receipt["verdict"] == "fail"
    assert [(s["name"], s["status"]) for s in receipt["steps"]] == [
        ("artifact_identity", "fail"),
        ("generation_and_parity", "skipped"),
        ("cohort", "skipped"),
        ("release_tests", "skipped"),
        ("package", "skipped"),
        ("identity_after_install", "skipped"),
    ]
    assert receipt["published"] is False and receipt["tagged"] is False


def test_an_exact_wheel_requires_its_digest(tmp_path):
    with pytest.raises(SystemExit):
        qualify.main(
            [
                "--core-checkout",
                ".",
                "--core-revision",
                "x",
                "--out",
                str(tmp_path / "o"),
                "--wheel",
                str(tmp_path / "w.whl"),
            ]
        )


def test_release_test_cap_stays_the_workspace_cap():
    assert qualify.TEST_CAP_SECONDS == 120
