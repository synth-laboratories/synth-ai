"""Qualify one exact SDK artifact locally and write a receipt (pass or fail).

    uv run --extra mcp python scripts/qualify_release.py --core-checkout <core repo> \
        --core-revision <frozen sha> --out <empty receipt dir> \
        [--wheel <candidate.whl> --expect-sha256 <hex>] [--sdk-checkout <source at the wheel's commit>]

Steps, in order; the first failure stops the run and every later step is recorded
as skipped:

1. artifact identity: with --wheel, the file's SHA-256 must equal --expect-sha256
   (a substituted wheel fails here). Without --wheel, build once into an empty
   owned directory under --out and hash what was built.
2. generated-code freshness and parity: check_surface_parity, gen_contract,
   gen_sync and gen_file_limits --check from the checkout root.
3. cohort: check_cohort.py against the frozen Core revision; both `aligned` and
   `provenance_verified` must be true.
4. targeted tests: the release set (contract, public surface, stdio MCP, offline
   journeys). Its wall time is recorded against the 120 s workspace cap; the
   recorded 0.41.0 candidate took 178 s, so `over_cap` stays visible in the
   receipt rather than shortening the set.
5. package: twine check, then check_package.py --wheel on the exact file (source
   budget measured from the wheel's own contents, isolated SDK/CLI/MCP install).
6. identity after install: the wheel's SHA-256 is unchanged.

A later publisher must compare its upload bytes with `artifact.sha256`; the
current tag workflow rebuilds. This command never tags, uploads or publishes.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "synth-ai.release-qualification.v1"
TEST_CAP_SECONDS = 120
RELEASE_TESTS = (
    "tests/test_contract.py",
    "tests/test_public_surface.py",
    "tests/test_mcp_stdio.py",
    "tests/acceptance/test_journeys_access.py",
    "tests/acceptance/test_journeys_isolation.py",
    "tests/acceptance/test_journeys_isolation_owner.py",
    "tests/acceptance/test_journeys_money.py",
    "tests/acceptance/test_journeys_refusals.py",
)
# One pytest step per test file, each under its own TEST_CAP_SECONDS: a step
# that ran several files would let their sum pass one cap unnoticed.
RELEASE_TEST_STEPS = tuple((f"release_tests:{Path(path).stem}", (path,)) for path in RELEASE_TESTS)
GENERATION_CHECKS = (
    ("surface_parity", ["scripts/check_surface_parity.py"]),
    ("gen_contract", ["scripts/gen_contract.py", "--check"]),
    ("gen_sync", ["scripts/gen_sync.py", "--check"]),
    ("gen_file_limits", ["scripts/gen_file_limits.py", "--check"]),
)


class StepFailed(Exception):
    pass


@dataclass
class Step:
    name: str
    status: str = "skipped"
    seconds: float = 0.0
    commands: list[list[str]] = field(default_factory=list)
    detail: dict[str, object] = field(default_factory=dict)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class Qualification:
    # The release-test wall clock; tests substitute a scripted clock.
    clock: Callable[[], float] = staticmethod(time.monotonic)

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out: Path = args.out
        self.steps: list[Step] = []
        self.wheel: Path | None = args.wheel
        self.wheel_sha256 = ""
        # The source the artifact was built from; checks run there (python -m puts it first on sys.path).
        self.checkout: Path = args.sdk_checkout.resolve()

    def run(self, step: Step, argv: list[str], *, cwd: Path | None = None) -> str:
        step.commands.append(argv)
        done = subprocess.run(
            argv, cwd=cwd or self.checkout, capture_output=True, text=True, check=False
        )
        if done.returncode != 0:
            step.detail["exit"] = done.returncode
            step.detail["output_tail"] = (done.stdout + done.stderr)[-1500:]
            raise StepFailed(f"{step.name}: exit {done.returncode}")
        return done.stdout

    def identity(self, step: Step) -> None:
        if self.wheel is None:
            dist = self.out / "dist"
            dist.mkdir()
            self.run(step, ["uv", "build", "--out-dir", str(dist)])
            [self.wheel] = sorted(dist.glob("synth_ai-*.whl"))
        self.wheel_sha256 = sha256(self.wheel)
        step.detail.update(
            wheel=self.wheel.name, bytes=self.wheel.stat().st_size, sha256=self.wheel_sha256
        )
        expected = self.args.expect_sha256
        if expected is not None and expected != self.wheel_sha256:
            step.detail["expected_sha256"] = expected
            raise StepFailed("artifact_identity: wheel sha256 differs from the qualified digest")

    def generation(self, step: Step) -> None:
        for name, argv in GENERATION_CHECKS:
            self.run(step, [sys.executable, *argv])
            step.detail[name] = "pass"

    def cohort(self, step: Step) -> None:
        report = json.loads(
            self.run(
                step,
                [
                    sys.executable,
                    "scripts/check_cohort.py",
                    "--core-checkout",
                    str(self.args.core_checkout),
                    "--core-revision",
                    self.args.core_revision,
                ],
            )
        )
        step.detail["report"] = report
        if report.get("aligned") is not True or report.get("provenance_verified") is not True:
            raise StepFailed("cohort: aligned and provenance_verified must both be true")

    def tests(self, files: tuple[str, ...], step: Step) -> None:
        started = self.clock()
        output = self.run(step, [sys.executable, "-m", "pytest", "-q", *files])
        elapsed = round(self.clock() - started, 1)
        summary = output.strip().splitlines()[-1] if output.strip() else ""
        step.detail.update(
            files=list(files),
            summary=summary,
            wall_seconds=elapsed,
            cap_seconds=TEST_CAP_SECONDS,
            over_cap=elapsed > TEST_CAP_SECONDS,
        )
        if elapsed > TEST_CAP_SECONDS:
            raise StepFailed(f"{step.name}: {elapsed} s over the {TEST_CAP_SECONDS} s cap")

    def package(self, step: Step) -> None:
        assert self.wheel is not None
        self.run(step, ["uvx", "twine", "check", str(self.wheel)])
        # This maintained check_package (exact-wheel mode), whatever the candidate's checkout carries.
        output = self.run(
            step,
            [sys.executable, str(ROOT / "scripts/check_package.py"), "--wheel", str(self.wheel)],
            cwd=ROOT,
        )
        step.detail["check_package"] = output.strip().splitlines()[-1] if output.strip() else ""

    def identity_after(self, step: Step) -> None:
        assert self.wheel is not None
        after = sha256(self.wheel)
        step.detail["sha256"] = after
        if after != self.wheel_sha256:
            raise StepFailed("identity_after_install: wheel bytes changed during qualification")

    def execute(self) -> dict[str, object]:
        plan: list[tuple[str, Callable[[Step], None]]] = [
            ("artifact_identity", self.identity),
            ("generation_and_parity", self.generation),
            ("cohort", self.cohort),
            *((name, functools.partial(self.tests, files)) for name, files in RELEASE_TEST_STEPS),
            ("package", self.package),
            ("identity_after_install", self.identity_after),
        ]
        failure = ""
        for name, action in plan:
            step = Step(name)
            self.steps.append(step)
            if failure:
                step.detail["reason"] = "prior step failed"
                continue
            started = time.monotonic()
            try:
                action(step)
                step.status = "pass"
            except (StepFailed, OSError, ValueError, subprocess.SubprocessError) as error:
                step.status = "fail"
                failure = str(error)
            step.seconds = round(time.monotonic() - started, 1)
        return {
            "schema": SCHEMA,
            "verdict": "fail" if failure else "pass",
            "failure": failure or None,
            "sdk_commit": git("rev-parse", "HEAD", cwd=self.checkout),
            "sdk_tree_clean": git(
                "status", "--porcelain", "--untracked-files=no", cwd=self.checkout
            )
            == "",
            "qualifier_commit": git("rev-parse", "HEAD"),
            "core_checkout": str(self.args.core_checkout),
            "core_revision": self.args.core_revision,
            "python": platform.python_version(),
            "artifact": {
                "path": str(self.wheel) if self.wheel else None,
                "name": self.wheel.name if self.wheel else None,
                "sha256": self.wheel_sha256 or None,
                "bytes": self.wheel.stat().st_size if self.wheel and self.wheel.exists() else None,
            },
            "steps": [asdict(step) for step in self.steps],
            "published": False,
            "tagged": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-checkout", type=Path, required=True)
    parser.add_argument("--core-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--expect-sha256")
    parser.add_argument("--sdk-checkout", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.wheel is not None and args.expect_sha256 is None:
        parser.error("--wheel requires --expect-sha256: qualification binds one exact artifact")
    if args.wheel is not None:
        args.wheel = args.wheel.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = Qualification(args).execute()
    (args.out / "qualification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in ("verdict", "failure", "artifact")}, sort_keys=True))
    return 0 if receipt["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
