"""Built wheels contain only the public SDK and required contract snapshots."""

import tomllib
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.release


def test_wheel_has_only_sdk_and_distribution_metadata():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    wheels = tuple((root / "dist").glob(f"synth_ai-{version}-*.whl"))
    if not wheels:
        pytest.skip("no local wheel built")
    for path in wheels:
        with zipfile.ZipFile(path) as wheel:
            names = wheel.namelist()
        assert all(
            name.startswith("synth_ai/") or name.split("/", 1)[0].endswith(".dist-info")
            for name in names
        )
        assert all(not name.endswith(".pyc") for name in names)
        assert "synth_ai/py.typed" in names
        for contract in ("index-v1.json", "swarms-v1.json", "core-v1.json", "mcp-v1.json"):
            assert f"synth_ai/contracts/{contract}" in names
