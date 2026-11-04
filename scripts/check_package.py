"""Validate the built wheel, source budget and isolated installation."""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
wheel = next((ROOT / "dist").glob("*.whl"))
source_bytes = sum(
    p.stat().st_size
    for p in (ROOT / "synth_ai").rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
)
assert wheel.stat().st_size < 150 * 1024
assert source_bytes < 250 * 1024
with zipfile.ZipFile(wheel) as archive:
    assert all(
        name.startswith(("synth_ai/", "synth_ai-0.26.0.dist-info/")) for name in archive.namelist()
    )
    assert not any("/public/" in name or "/sdk/" in name for name in archive.namelist())
# Under the checkout so workspace file boundaries also apply to local validation.
with tempfile.TemporaryDirectory(prefix=".wheel-smoke-", dir=ROOT) as directory:
    environment = Path(directory) / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
    python = environment / "bin/python"
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True)
    subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            'from synth_ai import SynthClient, AsyncSynthClient; from synth_ai.errors import SwarmsAPIError, ConflictError; from synth_ai.models import Money; assert Money(amount="0.10",currency="USD").amount == "0.10"',
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
print(
    {
        "wheel_bytes": wheel.stat().st_size,
        "package_source_bytes": source_bytes,
        "isolated_install": "passed",
    }
)
