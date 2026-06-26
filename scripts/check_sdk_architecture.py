#!/usr/bin/env python3
"""Architecture fences for the public Research hero surface."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = ROOT / "synth_ai" / "research"
MONOLITH_CLIENT = ROOT / "synth_ai" / "managed_research" / "sdk" / "client.py"
MONOLITH_LINE_CEILING = 6150

FORBIDDEN_NAME_FRAGMENTS = (
    "manderqueue",
    "ManderQueue",
    "Manderqueue",
)

FORBIDDEN_HERO_METHODS = (
    "def control(",
)

CAMEL_CASE_DEF = re.compile(r"^def ([a-z]+[A-Z][A-Za-z0-9_]*)\(")

REQUIRED_HERO_FILES = (
    RESEARCH_ROOT / "client.py",
    RESEARCH_ROOT / "factories.py",
    RESEARCH_ROOT / "limits.py",
    RESEARCH_ROOT / "secrets.py",
    RESEARCH_ROOT / "async_client.py",
)


def _failures_for_source(path: Path) -> list[str]:
    failures: list[str] = []
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(ROOT)

    for index, line in enumerate(text.splitlines(), start=1):
        for fragment in FORBIDDEN_NAME_FRAGMENTS:
            if fragment in line:
                failures.append(f"{rel}:{index}: forbidden fragment {fragment!r}")
        match = CAMEL_CASE_DEF.search(line.strip())
        if match and not line.strip().startswith("#"):
            failures.append(
                f"{rel}:{index}: public hero methods must be snake_case (found {match.group(1)!r})"
            )

    if path.name == "client.py" and path.parent.name == "research":
        for fragment in FORBIDDEN_HERO_METHODS:
            if fragment in text:
                failures.append(f"{rel}: public hero must not expose {fragment!r}")

    if path.name == "__init__.py":
        for export_name in _parse_all(text):
            if export_name.startswith("Smr"):
                failures.append(f"{rel}: __all__ must not export {export_name!r}")

    return failures


def _parse_all(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        names: list[str] = []
                        for element in node.value.elts:
                            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                                names.append(element.value)
                        return names
    return []


def _top_level_export_violations() -> list[str]:
    init_path = ROOT / "synth_ai" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    failures: list[str] = []
    for export_name in _parse_all(text):
        if export_name.startswith("Smr"):
            failures.append(
                f"synth_ai/__init__.py: remove {export_name!r} from __all__ "
                "(use Research* aliases)"
            )
    return failures


def _monolith_line_violations() -> list[str]:
    if not MONOLITH_CLIENT.is_file():
        return []
    line_count = len(MONOLITH_CLIENT.read_text(encoding="utf-8").splitlines())
    if line_count > MONOLITH_LINE_CEILING:
        return [
            f"{MONOLITH_CLIENT.relative_to(ROOT)}: {line_count} lines exceeds "
            f"ratchet ceiling {MONOLITH_LINE_CEILING} (C19 shrink required before growth)"
        ]
    return []


def main() -> int:
    failures: list[str] = []

    for required in REQUIRED_HERO_FILES:
        if not required.is_file():
            failures.append(f"missing required hero module: {required.relative_to(ROOT)}")

    if RESEARCH_ROOT.is_dir():
        for path in sorted(RESEARCH_ROOT.rglob("*.py")):
            failures.extend(_failures_for_source(path))

    failures.extend(_top_level_export_violations())
    failures.extend(_monolith_line_violations())

    if failures:
        print("SDK architecture check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("SDK architecture check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
