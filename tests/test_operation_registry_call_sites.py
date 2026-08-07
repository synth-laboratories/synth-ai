"""Hold every operation id the SDK *dispatches on* against the registry that resolves it.

``synth_ai/sdk/research/operations.py`` is checked against the backend-authored
OpenAPI, so an id that is in the registry is real. Nothing checked the other
direction: an SDK method may pass an id that is in no registry at all, and
``research_operation`` raises ``ValueError`` for it. That failure happens at
call time, before the transport is touched, so the method is dead on arrival --
and dead in a way no registry/spec parity check can see, because the broken id
exists only in the calling code. ``get_intern_sync_deploy_packet`` and
``list_intern_async_runtime_mcp_actions`` both shipped that way.

This walks the SDK's own source and closes that direction. It is deliberately
static: importing the modules would only reach the ids on code paths a test
happens to execute, and the whole point is to reach the ones nothing executes.

Two things make the walk trustworthy rather than merely suggestive:

* Dispatch through helpers is followed. A function that forwards its first value
  parameter into a resolver is itself a resolver, transitively, so
  ``_request("...", path)`` and ``_lifecycle("...", action)`` are seen without
  either being named here.
* A call site that builds its id at runtime cannot silently opt out. It must
  appear in :data:`DYNAMIC_OPERATION_IDS` with the closed set of ids it can
  produce, and those ids are checked like any other.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from synth_ai.core.http.request import OperationMetadata
from synth_ai.sdk.research.operations import (
    DATASET_REVISION_PUBLICATION_OPERATIONS,
    RESEARCH_OPERATIONS,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "synth_ai"

#: Resolver function name -> the registry it looks the id up in. These are the
#: only two functions that turn a string into an ``OperationMetadata``; every
#: dispatch in the SDK bottoms out in one of them.
RESOLVERS: Mapping[str, Mapping[str, OperationMetadata]] = {
    "research_operation": RESEARCH_OPERATIONS,
    "dataset_revision_publication_operation": DATASET_REVISION_PUBLICATION_OPERATIONS,
}

#: (module, enclosing function) -> the closed set of ids that call site can build.
#: These construct the id from a local or an f-string, so the walk cannot read it
#: off the AST. Enumerating them keeps runtime-built ids under the same guarantee
#: as literal ones; a new dynamic site fails the walk until it is listed here.
DYNAMIC_OPERATION_IDS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("sdk/research/factories.py", "_transition"): (
        "start_factory",
        "pause_factory",
        "resume_factory",
        "archive_factory",
    ),
    ("sdk/research/swarms.py", "preflight"): (
        "preflight_one_off_run",
        "preflight_project_run",
    ),
    ("sdk/research/swarms.py", "create"): (
        "trigger_one_off_run",
        "trigger_project_run",
    ),
}

#: Floor, not a target. It exists so that a refactor which blinds the walk --
#: renaming a helper, moving the id to a keyword argument -- fails loudly instead
#: of quietly checking nothing. Raise it when it starts lagging far behind.
MINIMUM_DISTINCT_OPERATION_IDS = 150


@dataclass(frozen=True, slots=True)
class CallSite:
    """One place the SDK hands a string to a resolver."""

    module: str
    lineno: int
    function: str
    resolver: str
    operation_id: str | None

    def __str__(self) -> str:
        target = self.operation_id if self.operation_id is not None else "<built at runtime>"
        return f"{self.module}:{self.lineno} {self.resolver}({target!r})"


def _first_value_parameter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    parameters = [argument.arg for argument in (*node.args.posonlyargs, *node.args.args)]
    if parameters and parameters[0] in {"self", "cls"}:
        parameters = parameters[1:]
    return parameters[0] if parameters else None


def _called_name(call: ast.Call) -> str | None:
    """The bare name of a call target, whether ``f(...)`` or ``self.f(...)``."""

    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _dispatchers(tree: ast.Module) -> dict[str, str]:
    """Every name in this module that resolves its first argument, and to which registry.

    Seeded with the two resolvers and grown to a fixpoint: a function that passes
    its own first value parameter through as a dispatcher's first argument is a
    dispatcher for the same registry.
    """

    dispatchers = {name: name for name in RESOLVERS}
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    while True:
        discovered = False
        for node in functions:
            if node.name in dispatchers:
                continue
            parameter = _first_value_parameter(node)
            if parameter is None:
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                resolver = dispatchers.get(_called_name(call) or "")
                if resolver is None or not call.args:
                    continue
                first = call.args[0]
                if isinstance(first, ast.Name) and first.id == parameter:
                    dispatchers[node.name] = resolver
                    discovered = True
                    break
        if not discovered:
            return dispatchers


class _CallSiteCollector(ast.NodeVisitor):
    """Collect dispatcher call sites, keeping the enclosing function for context."""

    def __init__(self, module: str, dispatchers: Mapping[str, str]) -> None:
        self._module = module
        self._dispatchers = dispatchers
        self._enclosing: list[tuple[str, str | None]] = []
        self.call_sites: list[CallSite] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enclosing.append((node.name, _first_value_parameter(node)))
        self.generic_visit(node)
        self._enclosing.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        resolver = self._dispatchers.get(_called_name(node) or "")
        if resolver is not None and not self._is_forwarding(node):
            first = node.args[0] if node.args else None
            literal = first.value if isinstance(first, ast.Constant) else None
            self.call_sites.append(
                CallSite(
                    module=self._module,
                    lineno=node.lineno,
                    function=self._enclosing[-1][0] if self._enclosing else "<module>",
                    resolver=resolver,
                    operation_id=literal if isinstance(literal, str) else None,
                )
            )
        self.generic_visit(node)

    def _is_forwarding(self, node: ast.Call) -> bool:
        """True for a dispatcher's own body passing its parameter along.

        That call is the definition of the indirection, not a use of it, and the
        id it carries is checked at the outer call sites instead.
        """

        if not self._enclosing:
            return False
        name, parameter = self._enclosing[-1]
        if name not in self._dispatchers or parameter is None:
            return False
        first = node.args[0] if node.args else None
        return isinstance(first, ast.Name) and first.id == parameter


def _collect_call_sites() -> tuple[CallSite, ...]:
    call_sites: list[CallSite] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = _CallSiteCollector(module, _dispatchers(tree))
        collector.visit(tree)
        call_sites.extend(collector.call_sites)
    return tuple(call_sites)


CALL_SITES = _collect_call_sites()
LITERAL_CALL_SITES = tuple(site for site in CALL_SITES if site.operation_id is not None)
DYNAMIC_CALL_SITES = tuple(site for site in CALL_SITES if site.operation_id is None)


def test_every_literal_operation_id_resolves() -> None:
    unresolved = [
        site
        for site in LITERAL_CALL_SITES
        if site.operation_id not in RESOLVERS[site.resolver]
    ]
    assert not unresolved, "operation ids referenced by the SDK but registered nowhere:\n" + "\n".join(
        str(site) for site in unresolved
    )


def test_every_runtime_built_operation_id_is_enumerated() -> None:
    undeclared = [
        site for site in DYNAMIC_CALL_SITES if (site.module, site.function) not in DYNAMIC_OPERATION_IDS
    ]
    assert not undeclared, (
        "operation ids built at runtime must be enumerated in DYNAMIC_OPERATION_IDS "
        "so they are checked too:\n" + "\n".join(str(site) for site in undeclared)
    )


def test_enumerated_runtime_operation_ids_resolve() -> None:
    unresolved = [
        f"{module}::{function} -> {operation_id!r}"
        for (module, function), operation_ids in DYNAMIC_OPERATION_IDS.items()
        for operation_id in operation_ids
        if operation_id not in RESEARCH_OPERATIONS
    ]
    assert not unresolved, "enumerated runtime operation ids registered nowhere:\n" + "\n".join(
        unresolved
    )


def test_enumerated_runtime_call_sites_still_exist() -> None:
    observed = {(site.module, site.function) for site in DYNAMIC_CALL_SITES}
    stale = sorted(key for key in DYNAMIC_OPERATION_IDS if key not in observed)
    assert not stale, (
        "DYNAMIC_OPERATION_IDS names call sites that no longer build an id at runtime; "
        f"drop them so the table keeps meaning something: {stale}"
    )


def test_walk_reaches_every_dispatching_module() -> None:
    """A module that imports a resolver must yield at least one call site.

    Without this, a rename that hides every call site in a module would leave the
    checks above passing on an empty set.
    """

    reached = {site.module for site in CALL_SITES}
    blind: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = path.relative_to(PACKAGE_ROOT).as_posix()
        if module in reached or module.endswith("operations.py"):
            continue
        source = path.read_text(encoding="utf-8")
        if any(f"import {resolver}" in source for resolver in RESOLVERS):
            blind.append(module)
    assert not blind, f"modules import a resolver but no call site was found: {blind}"


def test_walk_covers_the_dispatch_surface() -> None:
    distinct = {site.operation_id for site in LITERAL_CALL_SITES}
    assert len(distinct) >= MINIMUM_DISTINCT_OPERATION_IDS, (
        f"the walk found only {len(distinct)} distinct operation ids, below the "
        f"{MINIMUM_DISTINCT_OPERATION_IDS} floor -- it has probably gone blind"
    )
