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


def _git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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
                elif _route_is_referenced(route, sdk_routes):
                    disposition = "advanced_public_candidate"
                    reason = "currently referenced by the SDK; requires an approved operation ID"
                else:
                    disposition = "backend_only"
                    reason = "not in the bounded public contract and not referenced by the SDK"
                rows.append(
                    {
                        "id": f"backend:{method.upper()}:{relative}:{node.name}",
                        "surface": "backend_route",
                        "source_path": relative,
                        "line": node.lineno,
                        "symbol": node.name,
                        "http_method": method.upper(),
                        "route": route,
                        "operation_id": operation_id,
                        "disposition": disposition,
                        "canonical_noun": _canonical_noun(relative, route, node.name),
                        "reason": reason,
                    }
                )
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


def _sdk_rows(sdk_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    surfaces = (
        (sdk_root / "synth_ai/research", "public_research", "public"),
        (
            sdk_root / "synth_ai/core/research/_legacy/sdk",
            "internal_compatibility_sdk",
            "compatibility_migrate",
        ),
    )
    for base, surface, disposition in surfaces:
        for path in _python_files(base):
            relative = path.relative_to(sdk_root).as_posix()
            for symbol, node in _public_methods(path):
                rows.append(
                    {
                        "id": f"{surface}:{relative}:{symbol}",
                        "surface": surface,
                        "source_path": relative,
                        "line": node.lineno,
                        "symbol": symbol,
                        "disposition": disposition,
                        "canonical_noun": _canonical_noun(relative, symbol),
                        "reason": (
                            "documented Research facade"
                            if surface == "public_research"
                            else "legacy implementation must move to core or become an exact alias"
                        ),
                    }
                )
    return rows


def _mcp_rows(sdk_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = sdk_root / "synth_ai/mcp/research"
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
            keywords = {keyword.arg: _constant_text(keyword.value) for keyword in node.keywords}
            name = keywords.get("name")
            if not name:
                continue
            rows.append(
                {
                    "id": f"mcp:{relative}:{name}",
                    "surface": "mcp_tool",
                    "source_path": relative,
                    "line": node.lineno,
                    "symbol": name,
                    "disposition": "adapter_migrate",
                    "canonical_noun": _canonical_noun(relative, name),
                    "reason": "move to the thin MCP adapter over the core operation registry",
                }
            )
    return rows


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    return [f"{module}.{alias.name}" if module else alias.name for alias in node.names]


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
                disposition = "advanced_explicit"
                reason = (
                    "explicit unstable Research dependency requires a named owner and "
                    "graduation, retention, or removal decision"
                )
            else:
                disposition = "supported_public"
                reason = "already uses a supported public import boundary"
            rows.append(
                {
                    "id": f"consumer:{repository}:{relative}:{node.lineno}",
                    "surface": f"{repository}_consumer",
                    "source_path": relative,
                    "line": node.lineno,
                    "symbol": ",".join(names),
                    "disposition": disposition,
                    "canonical_noun": _canonical_noun(relative, *names),
                    "reason": reason,
                }
            )
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
    rows = [
        *_backend_rows(backend_root, public_operations, sdk_routes),
        *_sdk_rows(ROOT),
        *_mcp_rows(ROOT),
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
    deep_consumer_rows = [row for row in rows if row["disposition"] == "migrate_public"]
    unclassified = [row for row in rows if not row.get("disposition") or not row.get("canonical_noun")]
    if unclassified:
        raise RuntimeError(f"unclassified capability rows: {len(unclassified)}")

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
        "schema": "synth.research-capability-ledger.v1",
        "sources": {
            "synth_ai": {"repository": "synth-ai", "commit": _git_sha(ROOT)},
            "backend": {"repository": "backend", "commit": _git_sha(backend_root)},
            "evals": {"repository": "evals", "commit": _git_sha(evals_root)},
        },
        "compatibility": {
            "deprecated_since_version": "0.16.0",
            "last_compatible_release_line": "0.17.x",
            "removal_version": "0.18.0",
            "removal_not_before": "2026-09-01",
        },
        "baseline": baseline,
        "current": current,
        "legacy_files": frozen_legacy_files or legacy_files,
        "internal_compatibility_files": frozen_internal_files or internal_files,
        "summary": {
            "rows": len(rows),
            "unclassified": len(unclassified),
            "by_disposition": dict(sorted(disposition_counts.items())),
            "by_surface": dict(sorted(surface_counts.items())),
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {arguments.output} rows={len(rows)} unclassified=0 "
        f"legacy_files={len(legacy_files)} legacy_lines={legacy_lines} "
        f"deep_imports={len(deep_consumer_rows)} public_operations={len(public_operations)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
