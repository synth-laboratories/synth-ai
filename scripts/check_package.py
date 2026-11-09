"""Validate the built wheel, source budget and isolated installation."""

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
wheel = ROOT / "dist" / f"synth_ai-{version}-py3-none-any.whl"
source_bytes = sum(
    p.stat().st_size
    for p in (ROOT / "synth_ai").rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
)
require(wheel.stat().st_size < 150 * 1024, "package acceptance invariant failed")
require(source_bytes < 250 * 1024, "package acceptance invariant failed")
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

print(
    {
        "wheel_bytes": wheel.stat().st_size,
        "package_source_bytes": source_bytes,
        "isolated_install": "passed",
    }
)
