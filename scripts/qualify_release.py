"""Qualify one local release candidate and retain exact inputs and step results."""

import argparse
import hashlib
import json
import platform
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ["uv", "run", "--locked", "--extra", "mcp"]
TREES = ["synth_ai", "scripts", "tests", "examples", "typing_tests"]


def script(name: str, *arguments: str) -> list[str]:
    return [*RUN, "python", "scripts/" + name + ".py", *arguments]


def qualify(core: Path, revision: str, out: Path) -> bool:
    out.mkdir(parents=True, exist_ok=False)
    dist = out / "dist"
    dist.mkdir()
    pin = tomllib.loads((ROOT / "contracts/SOURCE.toml").read_text())
    commands = {
        "environment": ["uv", "sync", "--locked", "--extra", "mcp"],
        "parity": script("check_surface_parity"),
        "contract": script("gen_contract", "--check"),
        "sync": script("gen_sync", "--check"),
        "file_limits": script("gen_file_limits", "--check"),
        "cohort": script("check_cohort", "--core-checkout", str(core), "--core-revision", revision),
        "lint": [*RUN, "ruff", "check", *TREES],
        "format": [*RUN, "ruff", "format", "--check", *TREES],
        "types": [*RUN, "ty", "check"],
        "type_contracts": script("check_type_contracts"),
        "standards": script("lint_standards"),
        "tests": [*RUN, "pytest", "-q"],
        "build": ["uv", "build", "--out-dir", str(dist)],
        "twine": [*RUN, "twine", "check"],
        "installed_smoke_and_package_budget": script("check_package"),
    }
    steps = {name: {"status": "skipped", "command": cmd} for name, cmd in commands.items()}
    artifacts = []
    report = {
        "sdk_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "core_revision": revision,
        "contract_path": pin["path"],
        "sdk_raw_sha256": hashlib.sha256(
            (ROOT / "contracts/synth-v1.openapi.yaml").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "artifacts": artifacts,
        "steps": steps,
        "passed": False,
    }
    wheel = None
    wheel_digest = ""
    try:
        for name, cmd in commands.items():
            step = steps[name]
            step["status"] = "failed"
            if name == "twine":
                cmd.extend(str(path) for path in sorted(dist.iterdir()))
            if name == "installed_smoke_and_package_budget":
                if wheel is None:
                    raise ValueError("missing built wheel")
                cmd.extend(["--wheel", str(wheel), "--sha256", wheel_digest])
            with (out / f"{name}.log").open("w") as log:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            step["exit_code"] = result.returncode
            if name == "cohort":
                cohort = json.loads((out / "cohort.log").read_text())
                report["cohort"] = cohort
                if (
                    cohort.get("aligned") is not True
                    or cohort.get("provenance_verified") is not True
                ):
                    return False
            if result.returncode:
                return False
            if name == "build":
                (wheel,) = dist.glob("*.whl")
                for path in sorted(dist.iterdir()):
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    artifacts.append(
                        {"name": path.name, "bytes": path.stat().st_size, "sha256": digest}
                    )
                    if path == wheel:
                        wheel_digest = digest
            step["status"] = "passed"
        report["passed"] = True
        return True
    except (OSError, ValueError) as error:
        report["cause"] = str(error)
        return False
    finally:
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-checkout", type=Path, required=True)
    parser.add_argument("--core-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        0 if qualify(args.core_checkout.resolve(), args.core_revision, args.out.resolve()) else 1
    )
