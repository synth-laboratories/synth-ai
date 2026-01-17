#!/usr/bin/env python3
"""Check for unused dependencies in pyproject.toml.

Reads [project].dependencies from pyproject.toml and verifies each one
is actually imported somewhere in synth_ai/. Prevents dependency bloat.

Usage:
    uv run python scripts/check_unused_deps.py
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

# PyPI name -> Python import name (only when they differ)
PACKAGE_TO_IMPORT = {
    "google-genai": "google",
    "pynacl": "nacl",
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "python-dotenv": "dotenv",
}


def get_import_name(package: str) -> str:
    """Convert PyPI package name to Python import name."""
    if package in PACKAGE_TO_IMPORT:
        return PACKAGE_TO_IMPORT[package]
    return package.replace("-", "_")


def extract_package_name(dep: str) -> str:
    """Extract package name from 'requests>=2.0.0' -> 'requests'."""
    match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
    return match.group(1).lower() if match else dep.lower()


def check_import_exists(import_name: str, search_path: Path) -> bool:
    """Check if package is imported anywhere in synth_ai/."""
    for pattern in [f"import {import_name}", f"from {import_name}"]:
        result = subprocess.run(  # noqa: S603
            ["grep", "-r", "-l", pattern, str(search_path)],  # noqa: S607
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            files = [f for f in result.stdout.strip().split("\n") if "__pycache__" not in f]
            if files:
                return True
    return False


def main() -> int:
    repo_root = Path(__file__).parent.parent.resolve()
    pyproject_path = repo_root / "pyproject.toml"
    synth_ai_path = repo_root / "synth_ai"

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    deps = pyproject.get("project", {}).get("dependencies", [])
    unused = []

    for dep in deps:
        pkg = extract_package_name(dep)
        import_name = get_import_name(pkg)
        if not check_import_exists(import_name, synth_ai_path):
            unused.append(pkg)

    if unused:
        print("Unused Dependencies Check FAILED")
        print("=" * 60)
        print("Dependencies not imported anywhere in synth_ai/:")
        for pkg in unused:
            print(f"  - {pkg}")
        print()
        print("Fix: remove from [project].dependencies or move to")
        print("     [project.optional-dependencies]")
        return 1

    print(f"Unused Dependencies Check PASSED ({len(deps)} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
