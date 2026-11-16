#!/usr/bin/env python3
"""T0 source audit: contract operations must reach each public customer surface.

See SYN-5691. This conservative audit reports unresolved dynamic dispatch as a
missing proof. A generated catalog alone is not proof of executable SDK coverage.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import yaml

__all__ = ["Finding", "audit", "main"]
ROOT = Path(__file__).resolve().parents[1]
HTTP_METHODS = frozenset({"get", "post", "patch", "put", "delete", "head", "options"})
TRANSPORT_METHODS = frozenset({"request", "invoke", "stream", "event_stream", "download"})


class Code(StrEnum):
    MISSING = "operation_missing"
    PRIVATE = "private_sdk_access"
    TRANSPORT = "raw_transport_access"
    NAME = "mcp_name_mismatch"
    CHANGED = "audit_input_changed"


@dataclass(frozen=True, slots=True, order=True)
class Finding:
    surface: str
    code: Code
    operation: str
    cause: str
    path: str
    line: int


def public_bindings(source: str) -> frozenset[tuple[str, str]]:
    """Bind literal wire operations only from public class methods."""
    bindings: set[tuple[str, str]] = set()
    tree = ast.parse(source)
    for owner in tree.body:
        if not isinstance(owner, ast.ClassDef) or owner.name.startswith("_"):
            continue
        for method in owner.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if method.name.startswith("_"):
                continue
            for call in ast.walk(method):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                if call.func.attr not in TRANSPORT_METHODS or not call.args:
                    continue
                operation = call.args[0]
                if isinstance(operation, ast.Constant) and isinstance(operation.value, str):
                    bindings.add((operation.value, owner.name + "." + method.name))
    return frozenset(bindings)


def surface_calls(
    source: str, *, surface: str, path: str
) -> tuple[frozenset[str], tuple[Finding, ...]]:
    """Collect calls rooted in explicitly constructed or annotated SDK objects.

    Matching a bare method name such as ``get`` would let dict.get accidentally
    satisfy getRun. Receiver types must instead originate in the SDK composition.
    """
    tree = ast.parse(source)
    receivers: dict[str, str] = {"client": "AsyncSynthClient"}
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and isinstance(node.annotation, ast.Name):
            receivers[node.arg] = node.annotation.id
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "client":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        receivers[target.id] = "client." + node.value.attr
    methods: set[str] = set()
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and not node.attr.startswith("__")
        ):
            findings.append(
                Finding(surface, Code.PRIVATE, "*", ast.unparse(node), path, node.lineno)
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = ast.unparse(node.func.value)
            if receiver in receivers:
                methods.add(receivers[receiver] + "." + node.func.attr)
            elif receiver.startswith("client."):
                methods.add(receiver + "." + node.func.attr)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"requests", "urllib", "aiohttp"}:
                    findings.append(
                        Finding(surface, Code.TRANSPORT, "*", alias.name, path, node.lineno)
                    )
        # httpx exception classes are permitted; constructing a raw client is not.
        if isinstance(node, ast.Call) and ast.unparse(node.func) in {
            "httpx.Client",
            "httpx.AsyncClient",
            "requests.get",
            "requests.post",
        }:
            findings.append(
                Finding(surface, Code.TRANSPORT, "*", ast.unparse(node.func), path, node.lineno)
            )
    return frozenset(methods), tuple(findings)


def composition(root: Path) -> dict[str, str]:
    """Resolve public client paths from explicit constructor assignments."""
    children: dict[str, tuple[tuple[str, str], ...]] = {}
    for path in sorted((root / "synth_ai").glob("*.py")):
        if path.stem.startswith("_"):
            continue
        for owner in ast.parse(path.read_text()).body:
            if not isinstance(owner, ast.ClassDef):
                continue
            fields: list[tuple[str, str]] = []
            for node in ast.walk(owner):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                    continue
                if not isinstance(node.value.func, ast.Name):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self" and not target.attr.startswith("_"):
                            fields.append((target.attr, node.value.func.id))
            children[owner.name] = tuple(fields)
    paths = {"AsyncSynthClient": "client"}
    pending = ["AsyncSynthClient"]
    while pending:
        owner = pending.pop()
        for field, child in children.get(owner, ()):
            if child not in paths:
                paths[child] = paths[owner] + "." + field
                pending.append(child)
    return paths


def surface_sources(root: Path, relative: str) -> tuple[Path, ...]:
    """Follow the surface's local imports; unused dispatch files prove nothing."""
    entry = root / relative
    pending = [entry]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        tree = ast.parse(path.read_text())
        invoked = frozenset(
            ast.unparse(node.func).split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or node.module is None:
                continue
            # CLI domain clients are SDK dependencies, not CLI implementation.
            if relative == "synth_ai/cli.py" and not node.module.startswith("_cli"):
                continue
            if not any((alias.asname or alias.name) in invoked for alias in node.names):
                continue
            candidate = path.parent / (node.module.replace(".", "/") + ".py")
            if candidate.is_file():
                pending.append(candidate)
    return tuple(sorted(visited))


def audit(root: Path) -> tuple[Finding, ...]:
    document = yaml.safe_load((root / "contracts/synth-v1.openapi.yaml").read_text())
    operations = tuple(
        (operation["operationId"], operation.get("x-synth-mcp-name"))
        for path, item in document["paths"].items()
        if path.startswith("/v1/")
        for method, operation in item.items()
        if method in HTTP_METHODS
    )
    bindings = frozenset(
        binding
        for path in sorted((root / "synth_ai").glob("*.py"))
        if not path.stem.startswith("_") and path.stem not in {"cli", "auth"}
        for binding in public_bindings(path.read_text())
    )
    paths = composition(root)
    findings: list[Finding] = []
    calls: dict[str, frozenset[str]] = {}
    named_mcp_calls: dict[str, frozenset[str]] = {}
    for surface, relative in (("cli", "synth_ai/cli.py"), ("mcp", "synth_ai/mcp/server.py")):
        collected: set[str] = set()
        for path in surface_sources(root, relative):
            methods, violations = surface_calls(
                path.read_text(), surface=surface, path=str(path.relative_to(root))
            )
            collected.update(methods)
            findings.extend(violations)
            if surface == "mcp":
                for case in ast.walk(ast.parse(path.read_text())):
                    if not isinstance(case, ast.match_case) or not isinstance(
                        case.pattern, ast.MatchValue
                    ):
                        continue
                    value = case.pattern.value
                    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                        continue
                    branch_methods, _ = surface_calls(
                        ast.unparse(ast.Module(body=case.body, type_ignores=[])),
                        surface=surface,
                        path=str(path.relative_to(root)),
                    )
                    named_mcp_calls[value.value] = branch_methods
        calls[surface] = frozenset(collected)
    tools = json.loads((root / "contracts/mcp-tools.json").read_text())["tools"]
    names = {tool["operationId"]: tool["name"] for tool in tools}
    for operation, expected_name in operations:
        methods = frozenset(name for identifier, name in bindings if identifier == operation)
        reachable = frozenset(
            paths[owner] + "." + method
            for name in methods
            for owner, method in (name.split(".", 1),)
            if owner in paths
        )
        if not reachable:
            findings.append(
                Finding(
                    "sdk",
                    Code.MISSING,
                    operation,
                    "no public method binds this operation",
                    "synth_ai",
                    0,
                )
            )
        for surface in ("cli", "mcp"):
            observed = calls[surface]
            if surface == "mcp" and named_mcp_calls:
                observed = named_mcp_calls.get(operation, frozenset())
            if not (reachable | methods).intersection(observed):
                findings.append(
                    Finding(
                        surface,
                        Code.MISSING,
                        operation,
                        "no statically proven public SDK call",
                        surface,
                        0,
                    )
                )
        if expected_name is None or names.get(operation) != expected_name:
            findings.append(
                Finding(
                    "mcp",
                    Code.NAME,
                    operation,
                    "contract/catalog name is missing or differs",
                    "contracts/mcp-tools.json",
                    0,
                )
            )
    return tuple(sorted(set(findings)))


def input_hashes(root: Path) -> dict[str, str]:
    paths = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "synth_ai",
            "contracts/synth-v1.openapi.yaml",
            "contracts/mcp-tools.json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.split("\0")
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in paths
        if relative and (relative.endswith(".py") or relative.startswith("contracts/"))
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    options = parser.parse_args(arguments)
    before = input_hashes(options.root)
    findings = audit(options.root)
    if input_hashes(options.root) != before:
        findings = (
            *findings,
            Finding("input", Code.CHANGED, "*", "source bytes changed during audit", "*", 0),
        )
    print(
        json.dumps(
            {
                "stage": "T0",
                "gate": "surface_parity",
                "detector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_sha256": before,
                "result": "Fail" if findings else "Pass",
                "source_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=options.root,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                ).stdout.strip(),
                "findings": [asdict(item) for item in findings],
            },
            indent=2,
        )
    )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
