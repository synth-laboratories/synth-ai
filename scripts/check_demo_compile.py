#!/usr/bin/env python3
"""Compile demo scripts and verify synth_ai imports from the repo checkout.

Usage:
    python3 scripts/check_demo_compile.py
    python3 scripts/check_demo_compile.py demos/foo.py demos/bar.py
    python3 scripts/check_demo_compile.py demos/subdir
"""

import argparse
import sys
import tokenize
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        return path.is_relative_to(base)
    except AttributeError:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False


def _collect_files(paths: list[str], repo_root: Path) -> tuple[list[Path], list[str]]:
    missing: list[str] = []
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.is_file():
            if path.suffix == ".py":
                files.append(path)
            else:
                missing.append(str(path))
        else:
            missing.append(str(path))

    seen: set[Path] = set()
    unique_files: list[Path] = []
    for file_path in files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)

    return unique_files, missing


def _compile_file(path: Path, repo_root: Path) -> list[str]:
    rel_path = path
    if _is_relative_to(path, repo_root):
        rel_path = path.relative_to(repo_root)

    try:
        with tokenize.open(path) as handle:
            source = handle.read()
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        location = str(rel_path)
        if exc.lineno and exc.offset:
            location = f"{rel_path}:{exc.lineno}:{exc.offset}"
        lines = [f"SyntaxError in {location}: {exc.msg}"]
        if exc.text:
            lines.append(f"  {exc.text.rstrip()}")
        return lines
    except Exception as exc:
        return [f"Failed to compile {rel_path}: {exc}"]

    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile demo scripts and verify synth_ai imports from the repo checkout."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to compile (defaults to demos/).",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    sys.path.insert(0, str(repo_root))

    try:
        import synth_ai
    except Exception as exc:
        print(f"Failed to import synth_ai: {exc}", file=sys.stderr)
        return 1

    synth_path_raw = getattr(synth_ai, "__file__", None)
    if not synth_path_raw:
        print("synth_ai.__file__ is missing.", file=sys.stderr)
        return 1

    synth_path = Path(synth_path_raw).resolve()
    if not _is_relative_to(synth_path, repo_root):
        print("synth_ai did not import from the repo checkout.", file=sys.stderr)
        print(f"  resolved: {synth_path}", file=sys.stderr)
        print(f"  repo: {repo_root}", file=sys.stderr)
        return 1

    print(f"synth_ai from {synth_path}")

    targets = args.paths if args.paths else ["demos"]
    files, missing = _collect_files(targets, repo_root)
    if missing:
        print("Invalid paths (missing or not .py):", file=sys.stderr)
        for entry in missing:
            print(f"  {entry}", file=sys.stderr)
        return 1

    if not files:
        print("No Python files found to compile.", file=sys.stderr)
        return 1

    errors: list[str] = []
    for file_path in files:
        errors.extend(_compile_file(file_path, repo_root))

    if errors:
        print("Demo compile check FAILED", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    print(f"OK compiled {len(files)} demo files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
