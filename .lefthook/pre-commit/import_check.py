#!/usr/bin/env python3
import importlib
import sys
from pathlib import Path

# Ensure cwd is in path for imports (script runs from .lefthook/pre-commit/)
if "" not in sys.path and "." not in sys.path:
    sys.path.insert(0, "")


def module_name_from_path(path: Path) -> str:
    rel_path = path.with_suffix("")
    parts = list(rel_path.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def main() -> int:
    for raw in sys.argv[1:]:
        path = Path(raw)
        if not path.exists():
            continue
        mod_name = module_name_from_path(path)
        if not mod_name:
            continue
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            print(f"Import failed: {mod_name} ({path}): {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
