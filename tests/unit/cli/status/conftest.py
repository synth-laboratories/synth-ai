from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def status_modules() -> Iterator[dict[str, object]]:
    """Load the status command modules without importing the full CLI stack.

    The CLI package eagerly imports FastAPI and other heavy dependencies; the tests
    only need the status command helpers, so we construct a minimal package tree
    manually and load the modules directly.
    """

    project_root = Path(__file__).resolve().parents[4]
    status_root = project_root / "synth_ai" / "cli" / "commands" / "status"

    package_names = {
        "synth_ai": project_root / "synth_ai",
        "synth_ai.cli": project_root / "synth_ai" / "cli",
        "synth_ai.cli.commands": project_root / "synth_ai" / "cli" / "commands",
        "synth_ai.cli.commands.status": status_root,
    }

    original_modules: dict[str, object] = {}
    try:
        # Stub the package hierarchy so relative imports inside the modules succeed.
        for name, path in package_names.items():
            original_modules[name] = sys.modules.get(name)
            module = types.ModuleType(name)
            module.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = module

        loaded: dict[str, object] = {}
        for mod in ("config", "errors", "client", "utils"):
            full_name = f"synth_ai.cli.commands.status.{mod}"
            spec = importlib.util.spec_from_file_location(full_name, status_root / f"{mod}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)  # type: ignore[misc]
            loaded[mod] = module

        yield loaded
    finally:
        # Remove the dynamically loaded modules.
        for mod in ("config", "errors", "client", "utils"):
            sys.modules.pop(f"synth_ai.cli.commands.status.{mod}", None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
