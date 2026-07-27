"""Static import graph over the shipped package, following lazy imports too.

Importing the package and asking `sys.modules` what loaded answers a narrower question than
"what can a customer reach": it misses every import written inside a function body.  That
distinction is not academic here — `ResearchClient.account` resolves
`from synth_ai.core.research.advanced import open_advanced_session` on first access, so a
runtime walk reports `advanced.py` and the eight modules behind it as orphans while a
customer touches them on the hero path.

This walk reads source instead.  It follows imports at any nesting depth, under
`if TYPE_CHECKING`, and through string literals that name a shipped module — the lazy-export
table in `synth_ai/__init__.py` maps every public name to its module that way, and
`importlib.resources.files` locates package data by the same trick.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "synth_ai"
PACKAGE = "synth_ai"

# `pyproject.toml [project.scripts]`.  A customer invokes these, so what they pull in is
# shipped code, not orphans.
CONSOLE_ENTRY_POINTS = (
    "synth_ai.cli",
    "synth_ai.mcp.research.server",
    "synth_ai.cli.research_factory_standup",
)

# `python -m` roots.  Nothing imports them, and nothing should.
DUNDER_MAIN_ROOTS = (
    "synth_ai.__main__",
    "synth_ai.cli.__main__",
    "synth_ai.mcp.research.__main__",
)

# Everything a customer can reach: the top-level namespace plus the declared entry points.
PUBLIC_ROOTS = (PACKAGE, *CONSOLE_ENTRY_POINTS, *DUNDER_MAIN_ROOTS)


def _module_name(path: Path) -> str:
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


@cache
def _modules() -> dict[str, Path]:
    """Every shipped module, keyed by dotted name.  Packages map to their `__init__.py`."""
    return {
        _module_name(path): path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _resolve(name: str) -> str | None:
    """Map a dotted target onto a shipped module, walking up to its package."""
    modules = _modules()
    if not name.startswith(PACKAGE):
        return None
    while name:
        if name in modules:
            return name
        name, _, _ = name.rpartition(".")
    return None


def _targets(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                prefix = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                base = f"{prefix}.{base}" if base else prefix
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    resolved = {result for name in found if (result := _resolve(name))}

    # A string literal spelling a module name exactly is a deferred import: that is how the
    # `_EXPORTS` tables in `__init__.py` and `public.py` name their targets, and how
    # `importlib.resources.files` locates package data.
    modules = _modules()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value in modules:
            resolved.add(node.value)

    return resolved


def _ancestors(module: str) -> set[str]:
    """Importing a submodule executes every `__init__.py` above it."""
    modules = _modules()
    found = set()
    name = module
    while "." in name:
        name = name.rpartition(".")[0]
        if name in modules:
            found.add(name)
    return found


@cache
def import_graph() -> dict[str, frozenset[str]]:
    graph = {}
    for name, path in _modules().items():
        targets = _targets(path, name)
        graph[name] = frozenset(targets | {a for t in targets for a in _ancestors(t)})
    return graph


def reachable_from(roots: tuple[str, ...]) -> frozenset[str]:
    """Every shipped module reachable from `roots`, lazy imports included."""
    graph = import_graph()
    seen: set[str] = set()
    queue = [root for root in roots if root in graph]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        queue.extend(graph[module] - seen)
    return frozenset(seen)


def shipped_modules() -> frozenset[str]:
    return frozenset(_modules())


def module_path(module: str) -> Path:
    return _modules()[module]


def relative_path(module: str) -> str:
    return _modules()[module].relative_to(REPO_ROOT).as_posix()
