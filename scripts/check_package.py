"""Validate the built wheel, source budget and isolated installation."""

import argparse
import hashlib
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


ROOT = Path(__file__).resolve().parents[1]
version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--wheel", type=Path, default=ROOT / "dist" / f"synth_ai-{version}-py3-none-any.whl"
)
parser.add_argument("--sha256")
parser.add_argument("--check", choices=["all", "budget", "install"], default="all")
options = parser.parse_args()
wheel = options.wheel.resolve()
digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
require(options.sha256 in (None, digest), "wheel hash mismatch before installation")
source_bytes = sum(
    p.stat().st_size
    for p in (ROOT / "synth_ai").rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
)
if options.check != "install":
    require(wheel.stat().st_size < 150 * 1024, "package acceptance invariant failed")
    # 0.32 added the task-verb CLI, run results and stream recovery (about 274 KiB).
    # Contract 1.25/1.26 (rollout views, per-role launch choices, allocation) brought
    # it to about 414 KiB: two operations' generated models, MCP tools and CLI views.
    # 0.41 public source docstrings and their generated sync copies add 44 KiB
    # (469,361 total source bytes). Keep a 480 KiB source cap and the unchanged
    # 150 KiB wheel cap; neither permits an accidental vendored dependency tree.
    source_limit = 480 * 1024
    require(
        source_bytes < source_limit,
        f"SDK source size {source_bytes} exceeds budget {source_limit} bytes",
    )
with zipfile.ZipFile(wheel) as archive:
    require("synth_ai/py.typed" in archive.namelist(), "wheel must ship PEP 561 types")
    require(
        all(
            name.startswith(("synth_ai/", f"synth_ai-{version}.dist-info/"))
            for name in archive.namelist()
        ),
        "package acceptance invariant failed",
    )
    require(
        not any("/public/" in name or "/sdk/" in name for name in archive.namelist()),
        "package acceptance invariant failed",
    )
if options.check == "budget":
    print({"wheel_bytes": wheel.stat().st_size, "package_source_bytes": source_bytes})
    raise SystemExit(0)

# Under the checkout so workspace file boundaries also apply to local validation.
with tempfile.TemporaryDirectory(prefix=".wheel-smoke-", dir=ROOT) as directory:
    environment = Path(directory) / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
    python = environment / "bin/python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-cache", str(wheel)], check=True
    )
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            'import sys; from synth_ai import SynthClient, AsyncSynthClient; from synth_ai.errors import SwarmsAPIError, ConflictError; from synth_ai.models import Money, RunEvent; result = RunEvent.model_fields["observed_at"].annotation == str | None and Money(amount="0.10",currency="USD").amount == "0.10"; result or sys.exit("installed model acceptance failed")',
        ],
        check=True,
        cwd=directory,
    )
    subprocess.run(
        [str(environment / "bin/synth"), "--help"],
        check=True,
        cwd=directory,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [str(python), "-I", str(ROOT / "scripts/check_installed.py")],
        check=True,
        cwd=directory,
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-cache", str(wheel) + "[mcp]"],
        check=True,
    )
    subprocess.run(
        [str(python), "-I", str(ROOT / "scripts/check_installed.py"), "--mcp"],
        check=True,
        cwd=directory,
    )

require(
    hashlib.sha256(wheel.read_bytes()).hexdigest() == digest, "wheel changed during installation"
)

print(
    {
        "wheel_bytes": wheel.stat().st_size,
        "package_source_bytes": source_bytes,
        "isolated_install": "passed",
    }
)
