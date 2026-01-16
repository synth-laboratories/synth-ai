#!/usr/bin/env python3
"""Verify demo scripts have valid imports from synth_ai.

This script actually imports demo modules to catch:
- ImportError: when importing non-existent functions/classes
- ModuleNotFoundError: when importing non-existent modules

Usage:
    python3 scripts/check_demo_imports.py
    python3 scripts/check_demo_imports.py demos/foo.py demos/bar.py
    python3 scripts/check_demo_imports.py demos/subdir
"""

import argparse
import ast
import sys
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


class SynthImportVisitor(ast.NodeVisitor):
    """Extract synth_ai imports from AST."""

    def __init__(self):
        self.imports: list[tuple[int, str, list[str]]] = []  # (lineno, module, names)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.startswith("synth_ai"):
                self.imports.append((node.lineno, alias.name, []))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.startswith("synth_ai"):
            names = [alias.name for alias in node.names]
            self.imports.append((node.lineno, node.module, names))
        self.generic_visit(node)


def _check_imports(path: Path, repo_root: Path) -> list[str]:
    """Check if synth_ai imports in a file are valid."""
    rel_path = path
    if _is_relative_to(path, repo_root):
        rel_path = path.relative_to(repo_root)

    errors: list[str] = []

    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # Syntax errors are handled by check_demo_compile.py
        return []
    except Exception as exc:
        return [f"Failed to parse {rel_path}: {exc}"]

    visitor = SynthImportVisitor()
    visitor.visit(tree)

    for lineno, module, names in visitor.imports:
        # Try to import the module
        try:
            imported = __import__(module, fromlist=names if names else [module.split(".")[-1]])
        except ImportError as exc:
            errors.append(f"{rel_path}:{lineno}: ImportError: {exc}")
            continue
        except Exception as exc:
            errors.append(f"{rel_path}:{lineno}: Error importing {module}: {exc}")
            continue

        # Check if specific names exist in the module
        for name in names:
            if not hasattr(imported, name):
                errors.append(
                    f"{rel_path}:{lineno}: ImportError: cannot import name '{name}' from '{module}'"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify demo scripts have valid imports from synth_ai."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to check (defaults to demos/).",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    sys.path.insert(0, str(repo_root))

    # Verify synth_ai imports from repo
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
        print("No Python files found to check.", file=sys.stderr)
        return 1

    errors: list[str] = []
    for file_path in files:
        errors.extend(_check_imports(file_path, repo_root))

    if errors:
        print("Demo import check FAILED", file=sys.stderr)
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    print(f"OK checked imports in {len(files)} demo files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
