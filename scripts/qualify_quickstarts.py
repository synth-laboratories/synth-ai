"""Execute the quickstart against the exact local release wheel and frozen Core contract."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--release-receipt", required=True, type=Path)
    parser.add_argument("--core-contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    release = json.loads(args.release_receipt.read_text())
    wheel = args.wheel.resolve()
    contract = ROOT / "contracts/synth-v1.openapi.yaml"
    checks = []
    receipt = {
        "schema": "synth.local-quickstart-qualification.v1",
        "wheel": wheel.name,
        "wheel_sha256": sha(wheel),
        "contract_sha256": sha(contract),
        "checks": checks,
        "passed": False,
    }
    if args.out.exists():
        parser.error("choose a new receipt path")
    try:
        if (
            wheel.name != release["wheel"]
            or receipt["wheel_sha256"] != release["wheel_sha256"]
            or sha(args.core_contract) != release["contract_sha256"]
            or sha(contract) != release["contract_sha256"]
        ):
            raise ValueError("wheel_or_contract_identity_mismatch")
        with tempfile.TemporaryDirectory(prefix=".qualify-quickstart-", dir=ROOT) as scratch:
            folder = Path(scratch)
            env = {
                k: os.environ[k]
                for k in ("PATH", "HOME", "LANG", "LC_ALL", "UV_CACHE_DIR")
                if k in os.environ
            }

            def run(name, command, cwd):
                subprocess.run(command, check=True, cwd=cwd, env=env, timeout=240)
                checks.append(name)

            for script, opts in [
                ("gen_contract", ["--check"]),
                ("gen_sync", ["--check"]),
                ("gen_file_limits", ["--check"]),
                ("check_surface_parity", []),
            ]:
                run(script, [sys.executable, f"scripts/{script}.py", *opts], ROOT)
            # Snapshot and re-hash the file actually passed to the installer.
            candidate = folder / wheel.name
            shutil.copyfile(wheel, candidate)
            if sha(candidate) != release["wheel_sha256"]:
                raise ValueError("wheel_changed_before_install")
            interp = folder / "venv/bin/python"
            run(
                "clean_environment",
                ["uv", "venv", "--python", sys.executable, interp.parent.parent],
                folder,
            )
            run(
                "exact_wheel_install",
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    interp,
                    str(candidate) + "[mcp]",
                    "pytest",
                    "pytest-asyncio",
                    "pyyaml",
                ],
                folder,
            )
            # Run from a fixture-only tree. Importing the source checkout would mask a stale wheel.
            for name in (
                "docs/quickstart.md",
                "docs/errors.md",
                "docs/cli-migration-0.31-0.32.md",
                "tests/test_error_guide.py",
                "contracts/synth-v1.openapi.yaml",
                "tests/conftest.py",
                "tests/test_journey.py",
                "tests/test_messages.py",
                "tests/test_quickstart_execution.py",
            ):
                target = folder / name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            run(
                "published_snippets_and_surface_journeys",
                [
                    interp,
                    "-I",
                    "-m",
                    "pytest",
                    "-q",
                    "--asyncio-mode=auto",
                    "--basetemp",
                    str(folder / "pytest-tmp"),
                    "tests/test_error_guide.py",
                    "tests/test_quickstart_execution.py",
                    "tests/test_journey.py",
                ],
                folder,
            )
            run(
                "installed_mcp_catalogue",
                [interp, "-I", str(ROOT / "scripts/check_installed.py"), "--mcp"],
                folder,
            )
        receipt["passed"] = True
    finally:
        with args.out.open("x") as output:
            json.dump(receipt, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
