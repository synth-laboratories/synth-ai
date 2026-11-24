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
assert spec is not None and spec.loader is not None
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
        ("quality_checks", "skipped"),
        ("cohort", "skipped"),
        *((name, "skipped") for name, _ in qualify.RELEASE_TEST_STEPS),
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


def test_each_release_test_step_names_exactly_one_existing_file():
    names = [name for name, _ in qualify.RELEASE_TEST_STEPS]
    assert len(names) == len(set(names))
    for name, files in qualify.RELEASE_TEST_STEPS:
        assert len(files) == 1, f"{name} runs {len(files)} files under one cap"
        assert (ROOT / files[0]).is_file(), files[0]
    assert [f for _, files in qualify.RELEASE_TEST_STEPS for f in files] == list(
        qualify.RELEASE_TESTS
    )


class ScriptedRun(qualify.Qualification):
    """Only the parts tests() reads: a scripted clock and a pytest that passes."""

    def __init__(self, seconds):
        self.ticks = iter([0.0, seconds])

    def clock(self):
        return next(self.ticks)

    def run(self, step, argv, **_):
        return "1 passed\n"


def test_a_release_test_step_over_the_cap_fails():
    run = ScriptedRun(qualify.TEST_CAP_SECONDS + 1.0)
    step = qualify.Step("release_tests:test_x")
    with pytest.raises(qualify.StepFailed, match="over the 120 s cap"):
        run.tests(("tests/test_x.py",), step)
    assert step.detail["over_cap"] is True and step.detail["files"] == ["tests/test_x.py"]
