#!/usr/bin/env python3
"""Generate warning-only exact re-export shims for Managed Research imports."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_ROOT = ROOT / "synth_ai/core/research/_legacy"
MCP_ROOT = ROOT / "synth_ai/mcp/research"
COMPATIBILITY_ROOT = ROOT / "synth_ai/managed_research"
TARGET_PREFIX = "synth_ai.core.research._legacy"
MCP_TARGET_PREFIX = "synth_ai.mcp.research"
WARNING = (
    "synth_ai.managed_research is deprecated since synth-ai 0.16.0 and will be "
    "removed in 0.18.0 no earlier than 2026-09-01; use SynthClient().research."
)


def _module_name(path: Path, source_root: Path, target_prefix: str) -> str:
    relative = path.relative_to(source_root)
    if relative.name == "__init__.py":
        suffix = ".".join(relative.parent.parts)
    else:
        suffix = ".".join((*relative.parent.parts, relative.stem))
    return target_prefix if not suffix else f"{target_prefix}.{suffix}"


def _shim_source(path: Path, source_root: Path, target_prefix: str) -> str:
    target = _module_name(path, source_root, target_prefix)
    lines = [
        '"""Deprecated exact re-exports from the core Research implementation."""',
        "",
        "from __future__ import annotations",
        "",
    ]
    if source_root == IMPLEMENTATION_ROOT and path.relative_to(source_root) == Path("__init__.py"):
        lines.extend(
            [
                "import warnings",
                "",
                f"warnings.warn({WARNING!r}, DeprecationWarning, stacklevel=2)",
                "",
            ]
        )
    lines.append(f"from {target} import *  # noqa: F403")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if not IMPLEMENTATION_ROOT.is_dir():
        raise SystemExit(f"missing implementation root: {IMPLEMENTATION_ROOT}")
    mismatches: list[str] = []
    expected_paths: set[Path] = set()
    sources = (
        (IMPLEMENTATION_ROOT, TARGET_PREFIX, Path()),
        (MCP_ROOT, MCP_TARGET_PREFIX, Path("mcp")),
    )
    for source_root, target_prefix, compatibility_prefix in sources:
        if not source_root.is_dir():
            raise SystemExit(f"missing compatibility source: {source_root}")
        for implementation_path in sorted(source_root.rglob("*.py")):
            relative = implementation_path.relative_to(source_root)
            compatibility_path = COMPATIBILITY_ROOT / compatibility_prefix / relative
            expected_paths.add(compatibility_path)
            expected = _shim_source(implementation_path, source_root, target_prefix)
            if arguments.check:
                actual = (
                    compatibility_path.read_text(encoding="utf-8")
                    if compatibility_path.is_file()
                    else None
                )
                if actual != expected:
                    mismatches.append((compatibility_prefix / relative).as_posix())
                continue
            compatibility_path.parent.mkdir(parents=True, exist_ok=True)
            compatibility_path.write_text(expected, encoding="utf-8")
    factory_standup_path = COMPATIBILITY_ROOT / "factory_standup.py"
    factory_standup_source = (
        '"""Deprecated Factory stand-up CLI compatibility export."""\n\n'
        "from __future__ import annotations\n\n"
        "from synth_ai.cli.research_factory_standup import *  # noqa: F403\n"
    )
    expected_paths.add(factory_standup_path)
    if arguments.check:
        if not factory_standup_path.is_file() or factory_standup_path.read_text(
            encoding="utf-8"
        ) != factory_standup_source:
            mismatches.append("factory_standup.py")
    else:
        factory_standup_path.write_text(factory_standup_source, encoding="utf-8")
    unexpected = sorted(
        path.relative_to(COMPATIBILITY_ROOT).as_posix()
        for path in COMPATIBILITY_ROOT.rglob("*.py")
        if path not in expected_paths
    ) if COMPATIBILITY_ROOT.is_dir() else []
    if not arguments.check:
        for relative in unexpected:
            (COMPATIBILITY_ROOT / relative).unlink()
        unexpected = []
    if arguments.check and (mismatches or unexpected):
        print(
            "Managed Research compatibility check failed: "
            f"mismatches={mismatches} unexpected={unexpected}"
        )
        return 1
    print(
        "Managed Research compatibility shims "
        f"{'verified' if arguments.check else 'generated'}: files={len(expected_paths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
