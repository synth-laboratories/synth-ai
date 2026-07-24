#!/usr/bin/env python3
"""Enforce monotonic boundaries for the Research SDK core migration.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "synth_ai/core"
# Compatibility trees that must never come back. These were shrink-ratcheted
# while they drained; they are now absence assertions.
RETIRED_TREES = (
    ROOT / "synth_ai/research",
    ROOT / "synth_ai/managed_research",
    ROOT / "synth_ai/core/research/_legacy",
)
RETIRED_MODULES = ROOT / "synth_ai/core/research/compat_advanced.py", ROOT / "synth_ai/core/research/control.py"
LEDGER_PATH = ROOT / "specifications/sdk/research_capability_ledger.json"
FORBIDDEN_CORE_IMPORTS = (
    "synth_ai.cli",
    "synth_ai.managed_research",
    "synth_ai.mcp",
    "synth_ai.research",
)
IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "legacy",
    "old",
    "private_tests",
    "test",
    "tests",
}


def _python_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.py")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root)
            if not any(part in IGNORED_PARTS for part in relative.parts):
                yield path


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    return [f"{module}.{alias.name}" if module else alias.name for alias in node.names]


def _core_import_failures() -> list[str]:
    failures: list[str] = []
    for path in _python_files(CORE_ROOT):
        relative = path.relative_to(ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for name in _import_names(node):
                if name.startswith(FORBIDDEN_CORE_IMPORTS):
                    failures.append(
                        f"{relative}:{node.lineno}: core imports forbidden boundary {name!r}"
                    )
    return failures


def _retired_tree_failures() -> list[str]:
    """The compatibility packages must stay deleted, not merely stop growing."""
    failures: list[str] = []
    for tree in RETIRED_TREES:
        if tree.exists():
            failures.append(f"retired compatibility tree is back: {tree.relative_to(ROOT)}")
    for module in RETIRED_MODULES:
        if module.exists():
            failures.append(f"retired compatibility module is back: {module.relative_to(ROOT)}")
    return failures

def _consumer_import_count(root: Path) -> tuple[int, list[str]]:
    count = 0
    samples: list[str] = []
    for path in _python_files(root):
        try:
            source = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        if "synth_ai.core" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = _import_names(node)
            if any(name.startswith("synth_ai.core") for name in names):
                count += 1
                if len(samples) < 12:
                    samples.append(
                        f"{path.relative_to(root)}:{node.lineno}: {', '.join(names)}"
                    )
    return count, samples


def _external_failures(
    ledger: dict[str, object],
    backend_root: Path | None,
    evals_root: Path | None,
) -> list[str]:
    baseline = ledger["baseline"]
    assert isinstance(baseline, dict)
    limit = int(baseline["deep_consumer_imports"])
    failures: list[str] = []
    count = 0
    samples: list[str] = []
    for root in (backend_root, evals_root):
        if root is None:
            continue
        root_count, root_samples = _consumer_import_count(root)
        count += root_count
        samples.extend(root_samples)
    if (backend_root is not None or evals_root is not None) and count > limit:
        failures.append(
            f"deep backend/evals imports increased: {count} > {limit}; "
            + " | ".join(samples[:12])
        )
    return failures


def _ledger_failures(ledger: dict[str, object]) -> list[str]:
    summary = ledger.get("summary")
    if not isinstance(summary, dict):
        return ["capability ledger summary is missing"]
    if int(summary.get("unclassified", -1)) != 0:
        return [f"capability ledger has {summary.get('unclassified')} unclassified rows"]
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["capability ledger contains no rows"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path)
    parser.add_argument("--evals-root", type=Path)
    arguments = parser.parse_args()
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    failures = [
        *_ledger_failures(ledger),
        *_core_import_failures(),
        *_retired_tree_failures(),
        *_external_failures(ledger, arguments.backend_root, arguments.evals_root),
    ]
    if failures:
        print("Research migration boundary check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Research migration boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
