#!/usr/bin/env python3
"""Check that README version badges match pyproject.toml version.

This ensures version consistency across documentation.

Usage:
    uv run python scripts/check_readme_version.py
"""

import re
import sys
import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).parent.parent.resolve()
    pyproject_path = repo_root / "pyproject.toml"
    readme_path = repo_root / "README.md"

    # Extract version from pyproject.toml
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    version = data["project"]["version"]
    print(f"pyproject.toml version: {version}")

    # Read README
    readme_content = readme_path.read_text()

    # Extract badge version (PyPI-X.Y.Z-orange pattern)
    badge_match = re.search(r"PyPI-([0-9]+\.[0-9]+\.[0-9]+[^-]*)-orange", readme_content)
    badge_version = badge_match.group(1) if badge_match else ""
    print(f"README badge version: {badge_version or '(not found)'}")

    # Extract install version (synth-ai==X.Y.Z pattern)
    install_match = re.search(r"synth-ai==([0-9]+\.[0-9]+\.[0-9]+[^\s\"']*)", readme_content)
    install_version = install_match.group(1) if install_match else ""
    print(f"README install version: {install_version or '(not found)'}")

    # Check matches
    errors = []
    if badge_version and badge_version != version:
        errors.append(
            f"README badge version ({badge_version}) does not match pyproject.toml ({version})"
        )
    if install_version and install_version != version:
        errors.append(
            f"README install version ({install_version}) does not match pyproject.toml ({version})"
        )

    if errors:
        for error in errors:
            print(f"❌ {error}")
        return 1

    print(f"✅ README version matches pyproject.toml ({version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
