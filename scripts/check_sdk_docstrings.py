#!/usr/bin/env python3
"""Verify public SDK modules have docstrings (Modal-style docs gate).

See: specifications/sdk/docstrings.md
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "specifications" / "sdk" / "public_api_manifest.json"
DOCS_OUTPUT = ROOT / "docs" / "reference" / "sdk"


def _public_names(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return not node.name.startswith("_")


def _missing_docstrings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    missing: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _public_names(node):
            if not ast.get_docstring(node):
                missing.append(f"{path.name}::{node.name} (class)")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public_names(item):
                    if not ast.get_docstring(item):
                        missing.append(f"{path.name}::{node.name}.{item.name}")
                elif isinstance(item, ast.AsyncFunctionDef) and _public_names(item):
                    if not ast.get_docstring(item):
                        missing.append(f"{path.name}::{node.name}.{item.name}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _public_names(node):
            if not ast.get_docstring(node):
                missing.append(f"{path.name}::{node.name}")

    if not ast.get_docstring(tree):
        missing.insert(0, f"{path.name} (module)")

    return missing


def _scan_generated_todos() -> list[str]:
    hits: list[str] = []
    if not DOCS_OUTPUT.is_dir():
        return hits
    for mdx in DOCS_OUTPUT.rglob("*.mdx"):
        for line_no, line in enumerate(mdx.read_text(encoding="utf-8").splitlines(), start=1):
            if "TODO:" in line or "TODO " in line:
                hits.append(f"{mdx.relative_to(ROOT)}:{line_no}: {line.strip()[:120]}")
    return hits


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    required = manifest.get("docstring_coverage_required") or []
    failures: list[str] = []

    for rel in required:
        path = ROOT / rel
        if not path.is_file():
            failures.append(f"missing manifest file: {rel}")
            continue
        failures.extend(_missing_docstrings(path))

    todo_hits = _scan_generated_todos()
    failures.extend(f"generated docs TODO: {hit}" for hit in todo_hits)

    if failures:
        print("SDK docstring gate failed:")
        for item in failures:
            print(f"  - {item}")
        print(f"\n{len(failures)} issue(s). See specifications/sdk/docstrings.md")
        return 1

    print(f"SDK docstring gate passed ({len(required)} manifest modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
