#!/usr/bin/env python3
"""Check that pyproject.toml version is a .dev version.

Used by CI to guard against publishing dev versions to prod PyPI
or non-dev versions to TestPyPI.

Usage:
    uv run python scripts/check_dev_version.py           # Check IS dev version
    uv run python scripts/check_dev_version.py --not-dev # Check is NOT dev version

Outputs version to GITHUB_OUTPUT if available.
"""

import os
import sys
import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).parent.parent.resolve()
    pyproject_path = repo_root / "pyproject.toml"

    # Parse args
    require_not_dev = "--not-dev" in sys.argv

    # Extract version
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    version = data["project"]["version"]
    print(f"version={version}")

    is_dev = ".dev" in version

    # Write to GITHUB_OUTPUT if available
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a") as f:
            f.write(f"version={version}\n")

    # Check version type
    if require_not_dev:
        if is_dev:
            print(f"❌ Version {version} is a dev version; refusing to publish to prod PyPI")
            return 1
        print(f"✅ Version {version} is a release version")
    else:
        if not is_dev:
            print(f"❌ Version {version} is not a dev version; skipping dev publish")
            return 1
        print(f"✅ Version {version} is a dev version")

    return 0


if __name__ == "__main__":
    sys.exit(main())
