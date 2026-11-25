"""Release copy separates unreleased trunk notes from the published PyPI version."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_changelog_starts_with_unreleased_then_declared_release():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text()
    headings = re.findall(r"^## (.+)$", changelog, re.MULTILINE)
    assert len(headings) >= 2
    assert headings[0] == "Unreleased"
    assert headings[1].startswith(f"{version} — ")
    assert "source candidate" not in headings[1]
    assert "does not attest publication" in changelog


def test_readme_names_the_declared_source_version():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert f"declares SDK {version}" in (ROOT / "README.md").read_text()
