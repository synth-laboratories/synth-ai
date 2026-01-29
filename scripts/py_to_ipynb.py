#!/usr/bin/env python3
"""Convert demo .py files with percent markers to .ipynb notebooks.

Only processes .py files under the repo's demos/ directory that include
Jupytext-style cell markers (lines starting with "# %%").

Usage:
    uv run python scripts/py_to_ipynb.py demos/gepa_banking77/run_banking77.py
    uv run python scripts/py_to_ipynb.py demos/  # all demo notebooks
    uv run python scripts/py_to_ipynb.py --force demos/  # overwrite existing
"""

import argparse
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _demos_dir() -> Path:
    return _repo_root() / "demos"


def _is_demo_py(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    try:
        path.resolve().relative_to(_demos_dir().resolve())
    except ValueError:
        return False
    return True


def _has_ipynb_markers(py_path: Path) -> bool:
    try:
        for line in py_path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("# %%"):
                return True
    except Exception:
        return False
    return False


def convert_file(py_path: Path, force: bool = False) -> bool:
    """Convert a single .py file to .ipynb."""
    if not _is_demo_py(py_path):
        return False
    if not _has_ipynb_markers(py_path):
        return False  # Not a notebook-style .py file

    ipynb_path = py_path.with_suffix(".ipynb")

    if ipynb_path.exists() and not force:
        print(f"Skipping {py_path} (.ipynb exists, use --force)")
        return False

    # Delete existing file first (jupytext overwrite is unreliable)
    if ipynb_path.exists():
        ipynb_path.unlink()

    result = subprocess.run(
        ["jupytext", "--to", "notebook", str(py_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {py_path}: {result.stderr}")
        return False

    print(f"{py_path} -> {ipynb_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    converted = 0
    for path in map(Path, args.paths):
        if path.is_dir():
            files = path.rglob("*.py")
        else:
            files = [path]
        for f in files:
            if convert_file(f, args.force):
                converted += 1

    print(f"Converted {converted} file(s)")


if __name__ == "__main__":
    main()
