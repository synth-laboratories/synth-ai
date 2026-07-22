#!/usr/bin/env python3
"""Generate the cross-repository Research SDK migration capability ledger.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
COMPATIBILITY_REMOVAL_VERSION = "0.18.0"
RUNTIME_UNRESOLVED = "unresolved"
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
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


def _git_source(root: Path, repository: str) -> dict[str, str | bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = bool(status.stdout.strip())
    return {
        "repository": repository,
        "commit": revision.stdout.strip(),
        "working_tree_dirty": dirty,
        "snapshot": "head_plus_worktree" if dirty else "head",
    }


def _python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if (
            path.is_file()
            and not path.is_symlink()
            and not any(part in IGNORED_PARTS for part in path.relative_to(root).parts)
        ):
            yield path


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _constant_text(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _expression_reference(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _expression_reference(node.value)
        if owner is not None:
            return f"{owner}.{node.attr}"
    return None


def _definition_fields(
    *,
    definition_status: str = "source_defined",
    handler: str | None,
    operation_id: str | None,
    operation_ids: list[str] | None = None,
    canonical_target: str | None,
    canonical_targets: list[str] | None = None,
    legacy_alias_target: str | None = None,
    target_disposition: str,
    target_removal_version: str | None = None,
    invoked_capabilities: list[str] | None = None,
    invoked_capability_status: str = "not_applicable",
) -> dict[str, Any]:
    """Describe source evidence without pretending that it proves runtime reachability."""
    return {
        "definition_status": definition_status,
        "runtime_availability": RUNTIME_UNRESOLVED,
        "canonical_target": canonical_target,
        "canonical_targets": canonical_targets or (
            [canonical_target] if canonical_target is not None else []
        ),
        "legacy_alias_target": legacy_alias_target,
        "operation_id": operation_id,
        "operation_ids": operation_ids or ([operation_id] if operation_id is not None else []),
        "handler": handler,
        "target_disposition": target_disposition,
        "target_removal_version": target_removal_version,
        "invoked_capabilities": invoked_capabilities or [],
        "invoked_capability_status": invoked_capability_status,
    }


def _canonical_noun(*values: str) -> str:
    text = " ".join(values).lower().replace("-", "_")
    ordered = (
        (("factory",), "factories"),
        (("effort",), "efforts"),
        (("project",), "projects"),
        (("run", "swarm"), "swarms"),
        (("work_product", "artifact"), "artifacts"),
        (("evidence", "grade", "candidate"), "evidence"),
        (("message", "actor", "task", "collaboration", "tag"), "collaboration"),
        (("billing", "economics", "cost", "budget", "credit"), "economics"),
        (("limit", "quota", "usage"), "limits"),
        (("secret", "credential", "key"), "secrets"),
        (("image",), "images"),
        (("deployment", "claim"), "deployments"),
        (("environment", "workspace"), "environments"),
        (("repository", "dataset", "resource"), "resources"),
    )
    for fragments, noun in ordered:
        if any(fragment in text for fragment in fragments):
            return noun
    return "research"


def _public_operation_ids(backend_root: Path) -> dict[tuple[str, str], str]:
    path = backend_root / "app/api/v1/managed_research/openapi_contract.py"
    tree = _parse(path)
    operations: dict[tuple[str, str], str] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "PUBLIC_OPERATION_IDS" or not isinstance(node.value, ast.Dict):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values, strict=True):
            if not isinstance(key_node, ast.Tuple) or len(key_node.elts) != 2:
                continue
            method = _constant_text(key_node.elts[0])
            route = _constant_text(key_node.elts[1])
            operation_id = _constant_text(value_node)
            if method and route and operation_id:
                operations[(method.upper(), route)] = operation_id
    return operations


def _sdk_route_literals(sdk_root: Path) -> set[str]:
    routes: set[str] = set()
    for base in (
        sdk_root / "synth_ai/research",
        sdk_root / "synth_ai/core/research/_legacy/sdk",
    ):
        if not base.is_dir():
            continue
        for path in _python_files(base):
            for node in ast.walk(_parse(path)):
                text = _constant_text(node)
                if text and text.startswith("/") and ("smr" in text or "project" in text or "run" in text):
                    routes.add(text)
    return routes


def _route_is_referenced(route: str, sdk_routes: set[str]) -> bool:
    normalized = route.rstrip("/")
    return any(
        candidate.rstrip("/") == normalized
        or candidate.rstrip("/").endswith(normalized)
        or normalized.endswith(candidate.rstrip("/"))
        for candidate in sdk_routes
    )


def _backend_rows(
    backend_root: Path,
    public_operations: dict[tuple[str, str], str],
    sdk_routes: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    route_root = backend_root / "app/api/v1/managed_research"
    for path in _python_files(route_root):
        relative = path.relative_to(backend_root).as_posix()
        for node in ast.walk(_parse(path)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
                    continue
                method = decorator.func.attr.lower()
                if method not in HTTP_METHODS or not decorator.args:
                    continue
                route = _constant_text(decorator.args[0])
                if route is None:
                    continue
                candidates = ((method.upper(), route), (method.upper(), f"/smr{route}"))
                operation_id = next(
                    (public_operations[key] for key in candidates if key in public_operations),
                    None,
                )
                if operation_id is not None:
                    disposition = "public"
                    reason = "backend-authored stable public operation ID"
                    target_disposition = "stable_public_operation"
                    canonical_target = f"operation:{operation_id}"
                elif _route_is_referenced(route, sdk_routes):
                    disposition = "advanced"
                    reason = (
                        "supported operator route referenced only through the explicit "
                        "unstable Research bridge"
                    )
                    target_disposition = "unresolved"
                    canonical_target = None
                else:
                    disposition = "backend_only"
                    reason = "not in the bounded public contract and not referenced by the SDK"
                    target_disposition = "backend_internal"
                    canonical_target = f"backend:{relative}:{node.name}"
                canonical_noun = _canonical_noun(relative, route, node.name)
                row = {
                    "id": f"backend:{method.upper()}:{relative}:{node.name}",
                    "surface": "backend_route",
                    "source_path": relative,
                    "line": node.lineno,
                    "symbol": node.name,
                    "http_method": method.upper(),
                    "route": route,
                    "operation_id": operation_id,
                    "disposition": disposition,
                    "canonical_noun": canonical_noun,
                    "owner": f"backend/{canonical_noun}",
                    "reason": reason,
                }
                row.update(
                    _definition_fields(
                        handler=f"backend:{relative}:{node.name}",
                        operation_id=operation_id,
                        canonical_target=canonical_target,
                        target_disposition=target_disposition,
                    )
                )
                rows.append(row)
    return rows


def _external_public_backend_rows(
    backend_root: Path,
    public_operations: dict[tuple[str, str], str],
    primary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find public handlers outside the Managed Research router by unique route suffix."""
    matched_operation_ids = {
        str(row["operation_id"])
        for row in primary_rows
        if row["operation_id"] is not None
    }
    route_root = backend_root / "app/api/v1/managed_research"
    candidates_by_operation: dict[str, list[tuple[Path, ast.AST, str]]] = {}
    for path in _python_files(backend_root / "app/api/v1"):
        if path.is_relative_to(route_root):
            continue
        for node in ast.walk(_parse(path)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if (
                    not isinstance(decorator, ast.Call)
                    or not isinstance(decorator.func, ast.Attribute)
                    or not decorator.args
                ):
                    continue
                method = decorator.func.attr.upper()
                declared_route = _constant_text(decorator.args[0])
                if not declared_route.rstrip("/"):
                    continue
                for (public_method, public_route), operation_id in public_operations.items():
                    if operation_id in matched_operation_ids or public_method != method:
                        continue
                    if public_route.rstrip("/").endswith(declared_route.rstrip("/")):
                        candidates_by_operation.setdefault(operation_id, []).append(
                            (path, node, declared_route)
                        )

    rows: list[dict[str, Any]] = []
    for (method, route), operation_id in public_operations.items():
        candidates = candidates_by_operation.get(operation_id, [])
        if operation_id in matched_operation_ids or len(candidates) != 1:
            continue
        path, node, declared_route = candidates[0]
        relative = path.relative_to(backend_root).as_posix()
        canonical_noun = _canonical_noun(relative, route, node.name)
        row = {
            "id": f"backend:{method}:{relative}:{node.name}",
            "surface": "backend_route",
            "source_path": relative,
            "line": node.lineno,
            "symbol": node.name,
            "http_method": method,
            "route": route,
            "declared_route": declared_route,
            "disposition": "public",
            "canonical_noun": canonical_noun,
            "owner": f"backend/{canonical_noun}",
            "reason": "backend-authored public operation outside the primary Research router",
        }
        row.update(
            _definition_fields(
                handler=f"backend:{relative}:{node.name}",
                operation_id=operation_id,
                canonical_target=f"operation:{operation_id}",
                target_disposition="stable_public_operation",
            )
        )
        rows.append(row)
    return rows


def _public_methods(path: Path) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    tree = _parse(path)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            yield node.name, node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"):
                    yield f"{node.name}.{child.name}", child


def _operation_id_for_node(
    node: ast.AST,
    public_operations: dict[tuple[str, str], str],
) -> str | None:
    direct: set[str] = set()
    route_candidates: set[str] = set()
    operations_by_route: dict[str, set[str]] = {}
    for (_, route), operation_id in public_operations.items():
        operations_by_route.setdefault(route.rstrip("/"), set()).add(operation_id)
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            function = _expression_reference(child.func)
            if function is not None and function.endswith("research_operation") and child.args:
                operation_id = _constant_text(child.args[0])
                if operation_id in public_operations.values():
                    direct.add(operation_id)
        text = _constant_text(child)
        if text is not None:
            route_candidates.update(operations_by_route.get(text.rstrip("/"), set()))
    if len(direct) == 1:
        return next(iter(direct))
    if not direct and len(route_candidates) == 1:
        return next(iter(route_candidates))
    return None


def _sdk_rows(
    sdk_root: Path,
    public_operations: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    surfaces = (
        (sdk_root / "synth_ai/research", "public_research", "public"),
        (sdk_root / "synth_ai/core/research", "core_research", "core_implementation"),
        (
            sdk_root / "synth_ai/core/research/_legacy/sdk",
            "internal_compatibility_sdk",
            "compatibility_migrate",
        ),
    )
    for base, surface, disposition in surfaces:
        for path in _python_files(base):
            if surface == "core_research" and "_legacy" in path.relative_to(base).parts:
                continue
            relative = path.relative_to(sdk_root).as_posix()
            for symbol, node in _public_methods(path):
                operation_id = _operation_id_for_node(node, public_operations)
                handler = f"python:{relative}:{symbol}"
                if surface == "public_research":
                    if symbol.endswith(".advanced"):
                        row_disposition = "advanced"
                        target_disposition = "unresolved"
                        canonical_target = None
                        target_removal_version = None
                    elif symbol.endswith(".runs"):
                        row_disposition = "compatibility_alias"
                        target_disposition = "remove_compatibility_alias"
                        canonical_target = handler.removesuffix(".runs") + ".swarms"
                        target_removal_version = COMPATIBILITY_REMOVAL_VERSION
                    else:
                        row_disposition = disposition
                        target_disposition = "stable_public_python"
                        canonical_target = handler
                        target_removal_version = None
                elif surface == "core_research":
                    if "/advanced" in relative:
                        row_disposition = "advanced"
                        target_disposition = "unresolved"
                        canonical_target = None
                    else:
                        row_disposition = disposition
                        target_disposition = "core_implementation"
                        canonical_target = handler
                    target_removal_version = None
                elif operation_id is not None:
                    row_disposition = disposition
                    target_disposition = "replace_with_stable_operation"
                    canonical_target = f"operation:{operation_id}"
                    target_removal_version = COMPATIBILITY_REMOVAL_VERSION
                else:
                    row_disposition = disposition
                    target_disposition = "unresolved"
                    canonical_target = None
                    target_removal_version = None
                row = {
                    "id": f"{surface}:{relative}:{symbol}",
                    "surface": surface,
                    "source_path": relative,
                    "line": node.lineno,
                    "symbol": symbol,
                    "disposition": row_disposition,
                    "canonical_noun": _canonical_noun(relative, symbol),
                    "reason": (
                        "documented Research facade"
                        if surface == "public_research"
                        else (
                            "core Research source definition"
                            if surface == "core_research"
                            else (
                                "legacy implementation must move to core or become "
                                "an exact alias"
                            )
                        )
                    ),
                }
                row.update(
                    _definition_fields(
                        handler=handler,
                        operation_id=operation_id,
                        canonical_target=canonical_target,
                        target_disposition=target_disposition,
                        target_removal_version=target_removal_version,
                    )
                )
                rows.append(row)
    return rows


def _public_compatibility_alias_rows(sdk_root: Path) -> list[dict[str, Any]]:
    path = sdk_root / "synth_ai/research/__init__.py"
    relative = path.relative_to(sdk_root).as_posix()
    rows: list[dict[str, Any]] = []
    for node in _parse(path).body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "_COMPATIBILITY_EXPORTS" or not isinstance(node.value, ast.Dict):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values, strict=True):
            alias = _constant_text(key_node)
            if (
                alias is None
                or not isinstance(value_node, ast.Tuple)
                or len(value_node.elts) != 2
            ):
                continue
            module = _constant_text(value_node.elts[0])
            symbol = _constant_text(value_node.elts[1])
            if module is None or symbol is None:
                continue
            target = f"python:{module}:{symbol}"
            row = {
                "id": f"public_compatibility_alias:{alias}",
                "surface": "public_compatibility_alias",
                "source_path": relative,
                "line": key_node.lineno,
                "symbol": alias,
                "disposition": "compatibility_alias",
                "canonical_noun": _canonical_noun(alias, module, symbol),
                "owner": "synth-ai/research",
                "reason": "lazy public compatibility export scheduled for removal",
            }
            row.update(
                _definition_fields(
                    definition_status="source_alias",
                    handler=None,
                    operation_id=None,
                    canonical_target=target,
                    legacy_alias_target=target,
                    target_disposition="remove_compatibility_alias",
                    target_removal_version=COMPATIBILITY_REMOVAL_VERSION,
                )
            )
            rows.append(row)
    return rows


def _legacy_module_alias_rows(sdk_root: Path) -> list[dict[str, Any]]:
    base = sdk_root / "synth_ai/managed_research"
    rows: list[dict[str, Any]] = []
    for path in _python_files(base):
        relative = path.relative_to(sdk_root).as_posix()
        source_module = relative.removesuffix(".py").replace("/", ".")
        if source_module.endswith(".__init__"):
            source_module = source_module[: -len(".__init__")]
        targets: list[tuple[int, str]] = []
        for node in _parse(path).body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("synth_ai"):
                targets.append((node.lineno, node.module or ""))
        if not targets:
            target = None
            line = 1
        elif len({module for _, module in targets}) == 1:
            line, target = targets[0]
        else:
            target = None
            line = targets[0][0]
        canonical_target = f"python:{target}" if target is not None else None
        row = {
            "id": f"legacy_alias_module:{source_module}",
            "surface": "legacy_alias_module",
            "source_path": relative,
            "line": line,
            "symbol": source_module,
            "disposition": "compatibility_alias",
            "canonical_noun": _canonical_noun(source_module, target or ""),
            "owner": "synth-ai/research-compatibility",
            "reason": (
                "exact deprecated module alias scheduled for removal"
                if target is not None
                else "compatibility module has multiple or no statically resolvable alias targets"
            ),
        }
        row.update(
            _definition_fields(
                definition_status="source_alias",
                handler=None,
                operation_id=None,
                canonical_target=canonical_target,
                legacy_alias_target=canonical_target,
                target_disposition=(
                    "remove_compatibility_alias" if target is not None else "unresolved"
                ),
                target_removal_version=(
                    COMPATIBILITY_REMOVAL_VERSION if target is not None else None
                ),
            )
        )
        rows.append(row)
    return rows


def _stable_mcp_tool_names(server_path: Path) -> set[str]:
    """Read stable discovery without importing the SDK runtime graph."""
    for node in _parse(server_path).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_STABLE_TOOL_NAMES"
            for target in node.targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and len(value.args) == 1
        ):
            value = value.args[0]
        if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
            names = {_constant_text(item) for item in value.elts}
            return {name for name in names if name is not None}
    raise RuntimeError(f"stable MCP tool allowlist is missing from {server_path}")


def _mcp_operation_ids(
    advertised_name: str,
    public_operations: dict[tuple[str, str], str],
) -> list[str]:
    suffix = advertised_name.removeprefix("research_")
    candidate_ids = {suffix}
    substitutions = (
        ("get_", "retrieve_"),
        ("patch_", "update_"),
        ("watch_", "stream_"),
    )
    for source, target in substitutions:
        if suffix.startswith(source):
            candidate_ids.add(target + suffix.removeprefix(source))
    candidate_ids.update(
        {
            {
                "attach_source_repo": "set_project_workspace_source_repository",
                "branch_run_from_checkpoint": "branch_run",
                "create_environment": "create_research_environment",
                "create_project_repository": "create_project_external_repository",
                "create_runnable_project": "create_project",
                "delete_project_repository": "delete_project_external_repository",
                "get_environment": "retrieve_research_environment",
                "get_workspace_inputs": "retrieve_project_workspace_inputs",
                "list_environments": "list_research_environments",
                "list_project_repositories": "list_project_external_repositories",
                "preflight_environment": "preflight_research_environment",
                "start_one_off_run": "trigger_one_off_run",
                "trigger_run": "trigger_project_run",
                "update_project_repository": "update_project_external_repository",
                "upload_project_dataset": "create_project_dataset",
                "upload_workspace_files": "upload_project_workspace_files",
            }.get(suffix, "")
        }
    )
    candidate_ids.update(
        {
            operation_id
            for operation_id in {
                "get_launch_preflight": (
                    "preflight_one_off_run",
                    "preflight_project_run",
                ),
                "get_limits": ("retrieve_research_limits",),
                "get_run_transcript": ("list_run_transcript",),
                "list_active_runs": ("list_project_active_runs",),
            }.get(suffix, ())
        }
    )
    available = set(public_operations.values())
    return sorted((candidate_ids - {""}) & available)


def _mcp_rows(
    sdk_root: Path,
    public_operations: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = sdk_root / "synth_ai/mcp/research"
    stable_tool_names = _stable_mcp_tool_names(base / "server.py")
    for path in _python_files(base):
        relative = path.relative_to(sdk_root).as_posix()
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name != "ToolDefinition":
                continue
            keyword_nodes = {keyword.arg: keyword.value for keyword in node.keywords}
            name = _constant_text(keyword_nodes.get("name"))
            if not name:
                continue
            advertised_name = f"research_{name[4:]}" if name.startswith("smr_") else name
            is_stable = advertised_name in stable_tool_names
            handler = _expression_reference(keyword_nodes.get("handler"))
            operation_ids = _mcp_operation_ids(advertised_name, public_operations)
            operation_id = operation_ids[0] if len(operation_ids) == 1 else None
            row = {
                "id": f"mcp:{relative}:{name}",
                "surface": "mcp_tool",
                "source_path": relative,
                "line": node.lineno,
                "symbol": name,
                "disposition": "public_adapter" if is_stable else "advanced_adapter",
                "canonical_noun": _canonical_noun(relative, name),
                "owner": "synth-ai/mcp",
                "reason": (
                    "stable noun-first MCP adapter"
                    if is_stable
                    else (
                        "explicitly unstable MCP adapter pending core-operation "
                        "cutover or removal"
                    )
                ),
            }
            row.update(
                _definition_fields(
                    handler=handler,
                    operation_id=operation_id,
                    operation_ids=operation_ids,
                    canonical_target=(
                        f"operation:{operation_id}" if operation_id is not None else None
                    ),
                    canonical_targets=[
                        f"operation:{candidate}" for candidate in operation_ids
                    ],
                    legacy_alias_target=(
                        f"mcp:{advertised_name}" if advertised_name != name else None
                    ),
                    target_disposition=(
                        "stable_public_adapter" if is_stable else "unresolved"
                    ),
                )
            )
            rows.append(row)
    return rows


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    return [f"{module}.{alias.name}" if module else alias.name for alias in node.names]


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local_name = alias.asname or alias.name
                bindings[local_name] = f"{module}.{alias.name}" if module else alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                bindings[local_name] = alias.name if alias.asname else local_name
    return bindings


def _invoked_capabilities(tree: ast.Module) -> list[str]:
    bindings = _import_bindings(tree)
    capabilities: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        reference = _expression_reference(node.func)
        if reference is None:
            continue
        parts = reference.split(".")
        root_binding = bindings.get(parts[0])
        if root_binding is not None and root_binding.startswith("synth_ai"):
            capabilities.add(".".join((root_binding, *parts[1:])))
            continue
        if "research" in parts:
            capabilities.add(".".join(parts[parts.index("research") :]))
    return sorted(capabilities)


def _consumer_rows(root: Path, repository: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _python_files(root):
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            # Shared workspaces may remove generated paths after discovery.
            continue
        if "synth_ai" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        invoked_capabilities = _invoked_capabilities(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = sorted(name for name in _import_names(node) if name.startswith("synth_ai"))
            if not names:
                continue
            deep = any(
                name.startswith("synth_ai.managed_research") or name.startswith("synth_ai.core")
                for name in names
            )
            advanced = any(name.startswith("synth_ai.research.advanced") for name in names)
            if deep:
                disposition = "migrate_public"
                reason = "deep/legacy SDK import must move to documented synth_ai.research surfaces"
            elif advanced:
                disposition = "advanced"
                reason = (
                    "repository-owned operator workflow explicitly accepts the unstable "
                    "Research bridge and may not advertise it as customer-stable"
                )
            else:
                disposition = "supported_public"
                reason = "already uses a supported public import boundary"
            if disposition == "supported_public":
                target_disposition = "supported_public_consumer"
                canonical_target = ",".join(names)
            elif disposition == "migrate_public":
                target_disposition = "migrate_to_public_surface"
                canonical_target = "python:synth_ai.research"
            else:
                target_disposition = "unresolved"
                canonical_target = None
            row = {
                "id": f"consumer:{repository}:{relative}:{node.lineno}",
                "surface": f"{repository}_consumer",
                "source_path": relative,
                "line": node.lineno,
                "symbol": ",".join(names),
                "disposition": disposition,
                "canonical_noun": _canonical_noun(relative, *names),
                "owner": repository,
                "reason": reason,
            }
            row.update(
                _definition_fields(
                    definition_status="source_reference",
                    handler=None,
                    operation_id=None,
                    canonical_target=canonical_target,
                    target_disposition=target_disposition,
                    invoked_capabilities=invoked_capabilities,
                    invoked_capability_status=(
                        "source_detected" if invoked_capabilities else "unresolved"
                    ),
                )
            )
            rows.append(row)
    return rows


def _inventory(sdk_root: Path, relative_root: str) -> tuple[list[str], int]:
    base = sdk_root / relative_root
    files = [path.relative_to(sdk_root).as_posix() for path in _python_files(base)]
    lines = sum(len((sdk_root / path).read_text(encoding="utf-8").splitlines()) for path in files)
    return files, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--evals-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "specifications/sdk/research_capability_ledger.json",
    )
    parser.add_argument(
        "--reset-baseline",
        action="store_true",
        help="replace the frozen Phase 0 ratchet (never use during normal refreshes)",
    )
    arguments = parser.parse_args()
    backend_root = arguments.backend_root.resolve()
    evals_root = arguments.evals_root.resolve()

    public_operations = _public_operation_ids(backend_root)
    sdk_routes = _sdk_route_literals(ROOT)
    backend_rows = _backend_rows(backend_root, public_operations, sdk_routes)
    rows = [
        *backend_rows,
        *_external_public_backend_rows(backend_root, public_operations, backend_rows),
        *_sdk_rows(ROOT, public_operations),
        *_public_compatibility_alias_rows(ROOT),
        *_legacy_module_alias_rows(ROOT),
        *_mcp_rows(ROOT, public_operations),
        *_consumer_rows(backend_root, "backend"),
        *_consumer_rows(evals_root, "evals"),
    ]
    rows.sort(key=lambda row: row["id"])
    legacy_files, legacy_lines = _inventory(ROOT, "synth_ai/managed_research")
    internal_files, internal_lines = _inventory(
        ROOT,
        "synth_ai/core/research/_legacy",
    )
    disposition_counts = Counter(row["disposition"] for row in rows)
    surface_counts = Counter(row["surface"] for row in rows)
    definition_status_counts = Counter(row["definition_status"] for row in rows)
    runtime_availability_counts = Counter(row["runtime_availability"] for row in rows)
    target_disposition_counts = Counter(row["target_disposition"] for row in rows)
    deep_consumer_rows = [row for row in rows if row["disposition"] == "migrate_public"]
    source_classification_unresolved = [
        row for row in rows if not row.get("disposition") or not row.get("canonical_noun")
    ]
    if source_classification_unresolved:
        raise RuntimeError(
            "source classification is missing for capability rows: "
            f"{len(source_classification_unresolved)}"
        )
    runtime_unresolved = [
        row for row in rows if row["runtime_availability"] == RUNTIME_UNRESOLVED
    ]
    target_unresolved = [
        row for row in rows if row["target_disposition"] == "unresolved"
    ]
    target_unresolved_by_surface = Counter(row["surface"] for row in target_unresolved)
    eval_invocation_unresolved = [
        row
        for row in rows
        if row["surface"] == "evals_consumer"
        and row["invoked_capability_status"] == "unresolved"
    ]
    canonical_target_unresolved = [
        row
        for row in rows
        if not row["canonical_targets"]
        and row["target_disposition"]
        not in {"backend_internal", "remove_compatibility_alias"}
    ]
    backend_handler_operation_ids = {
        str(row["operation_id"])
        for row in rows
        if row["surface"] == "backend_route" and row["operation_id"] is not None
    }
    public_operation_ids = set(public_operations.values())
    stable_mcp_without_operation_mapping = sorted(
        str(row["symbol"])
        for row in rows
        if row["target_disposition"] == "stable_public_adapter"
        and not row["operation_ids"]
    )

    frozen_baseline: dict[str, Any] | None = None
    frozen_legacy_files: list[str] | None = None
    frozen_internal_files: list[str] | None = None
    if arguments.output.is_file() and not arguments.reset_baseline:
        existing = json.loads(arguments.output.read_text(encoding="utf-8"))
        existing_baseline = existing.get("baseline")
        existing_legacy_files = existing.get("legacy_files")
        existing_internal_files = existing.get("internal_compatibility_files")
        if isinstance(existing_baseline, dict) and isinstance(existing_legacy_files, list):
            frozen_baseline = existing_baseline
            frozen_legacy_files = [str(value) for value in existing_legacy_files]
            if isinstance(existing_internal_files, list):
                frozen_internal_files = [str(value) for value in existing_internal_files]

    current = {
        "legacy_implementation_files": len(legacy_files),
        "legacy_implementation_lines": legacy_lines,
        "internal_compatibility_files": len(internal_files),
        "internal_compatibility_lines": internal_lines,
        "deep_consumer_imports": len(deep_consumer_rows),
        "stable_public_operation_ids": len(public_operations),
    }
    baseline = frozen_baseline or dict(current)
    baseline.setdefault("internal_compatibility_files", len(internal_files))
    baseline.setdefault("internal_compatibility_lines", internal_lines)

    payload = {
        "schema": "synth.research-capability-ledger.v2",
        "sources": {
            "synth_ai": _git_source(ROOT, "synth-ai"),
            "backend": _git_source(backend_root, "backend"),
            "evals": _git_source(evals_root, "evals"),
        },
        "compatibility": {
            "deprecated_since_version": "0.16.0",
            "last_compatible_release_line": "0.17.x",
            "removal_version": COMPATIBILITY_REMOVAL_VERSION,
            "removal_not_before": "2026-09-01",
        },
        "evidence_semantics": {
            "definition_status": (
                "AST-discovered source definition, compatibility alias, or consumer "
                "reference only"
            ),
            "runtime_availability": (
                "requires an external invocation receipt; this source generator does not "
                "assert import, registration, authorization, or service reachability"
            ),
            "target_disposition": (
                "statically resolved migration target when known; unresolved is intentional"
            ),
            "invoked_capabilities": (
                "source-file call expressions, not proof that an eval executed successfully"
            ),
        },
        "baseline": baseline,
        "current": current,
        "legacy_files": frozen_legacy_files or legacy_files,
        "internal_compatibility_files": frozen_internal_files or internal_files,
        "summary": {
            "rows": len(rows),
            "unclassified": len(source_classification_unresolved),
            "source_classification_unresolved": len(source_classification_unresolved),
            "runtime_availability_unresolved": len(runtime_unresolved),
            "target_disposition_unresolved": len(target_unresolved),
            "target_disposition_unresolved_by_surface": dict(
                sorted(target_unresolved_by_surface.items())
            ),
            "canonical_target_unresolved": len(canonical_target_unresolved),
            "eval_invocation_unresolved": len(eval_invocation_unresolved),
            "public_operation_ids_with_backend_handler": len(
                backend_handler_operation_ids
            ),
            "public_operation_ids_without_backend_handler": sorted(
                public_operation_ids - backend_handler_operation_ids
            ),
            "stable_mcp_adapters_without_operation_mapping": (
                stable_mcp_without_operation_mapping
            ),
            "by_disposition": dict(sorted(disposition_counts.items())),
            "by_surface": dict(sorted(surface_counts.items())),
            "by_definition_status": dict(sorted(definition_status_counts.items())),
            "by_runtime_availability": dict(sorted(runtime_availability_counts.items())),
            "by_target_disposition": dict(sorted(target_disposition_counts.items())),
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {arguments.output} rows={len(rows)} "
        f"source_classification_unresolved={len(source_classification_unresolved)} "
        f"runtime_availability_unresolved={len(runtime_unresolved)} "
        f"target_disposition_unresolved={len(target_unresolved)} "
        f"eval_invocation_unresolved={len(eval_invocation_unresolved)} "
        f"legacy_files={len(legacy_files)} legacy_lines={legacy_lines} "
        f"deep_imports={len(deep_consumer_rows)} public_operations={len(public_operations)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
