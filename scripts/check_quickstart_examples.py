"""Qualify the documented examples locally against an exact SDK wheel and contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    "tests/quickstart_examples.py",
    "tests/test_error_guide.py",
    "tests/test_errors.py",
    "tests/test_message_refusals.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if wheel_digest != args.wheel_sha256:
        raise SystemExit("wheel SHA-256 does not match the selected artifact")
    contract = ROOT / "contracts/synth-v1.openapi.yaml"
    contract_digest = hashlib.sha256(contract.read_bytes()).hexdigest()
    source = tomllib.loads((ROOT / "contracts/SOURCE.toml").read_text())
    if contract_digest != source["sha256"]:
        raise SystemExit("contract SHA-256 does not match SOURCE.toml")
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    with ZipFile(wheel) as archive:
        metadata = archive.read(f"synth_ai-{version}.dist-info/METADATA").decode()
        if f"\nVersion: {version}\n" not in metadata:
            raise SystemExit("wheel version differs from the quickstart SDK version")
        for name in ("operations.py", "models.py"):
            generated = f"synth_ai/_generated/{name}"
            if archive.read(generated) != (ROOT / generated).read_bytes():
                raise SystemExit(f"wheel generated contract differs for {name}")

    env = os.environ.copy()
    env.update(PYTHONPATH=str(wheel), PYTHONNOUSERSITE="1", PYTEST_ADDOPTS="")
    probe = (
        "import pathlib, sys, synth_ai; "
        "origin = str(pathlib.Path(synth_ai.__file__).resolve()); "
        "assert origin.startswith(sys.argv[1] + '/'), origin"
    )
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("output receipt already exists")
    with TemporaryDirectory(prefix="quickstart-", dir=output.parent) as scratch:
        report = Path(scratch) / "pytest.xml"
        subprocess.run(
            [sys.executable, "-c", probe, str(wheel)],
            cwd=scratch,
            env=env,
            check=True,
            timeout=10,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(Path(scratch) / "pytest"),
                "--junitxml",
                str(report),
                *(str(ROOT / suite) for suite in SUITES),
            ],
            cwd=scratch,
            env=env,
            check=True,
            timeout=120,
        )
        cases = ElementTree.parse(report).findall(".//testcase")  # noqa: S314 -- local pytest junitxml report
        if not cases or any(case.find("skipped") is not None for case in cases):
            raise SystemExit("quickstart check requires a nonempty suite with no skips")
    sources = ("docs/quickstart.md", "docs/errors.md", "docs/cli-migration-0.31-0.32.md")
    receipt = {
        "schema": "synth.quickstart-examples.v1",
        "sdk_version": version,
        "wheel_sha256": wheel_digest,
        "contract_sha256": contract_digest,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources
        },
        "suites": SUITES,
        "test_count": len(cases),
        "result": "passed",
        "target": "local contract fixture",
    }
    with output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
