#!/usr/bin/env python3
"""Verify the vendored backend Research contract and core operation registry."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "openapi/research-v1.json"
OPERATIONS_PATH = ROOT / "synth_ai/core/research/operations.py"
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _contract_operations(payload: dict[str, Any]) -> dict[str, tuple[str, str]]:
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Research OpenAPI is missing paths")
    operations: dict[str, tuple[str, str]] = {}
    for path, path_item in paths.items():
        if not isinstance(path, str) or not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            operation_id = operation.get("operationId")
            if not isinstance(operation_id, str) or not operation_id:
                raise ValueError(f"{method.upper()} {path} has no operationId")
            if operation_id in operations:
                raise ValueError(f"duplicate Research operationId {operation_id!r}")
            operations[operation_id] = (method.upper(), path)
    return operations


def _source_operations() -> dict[str, tuple[str, str]]:
    tree = ast.parse(
        OPERATIONS_PATH.read_text(encoding="utf-8"),
        filename=str(OPERATIONS_PATH),
    )
    operations: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_operation" or len(node.args) < 3:
            continue
        operation_node, method_node, path_node = node.args[:3]
        if not (
            isinstance(operation_node, ast.Constant)
            and isinstance(operation_node.value, str)
            and isinstance(method_node, ast.Attribute)
            and isinstance(path_node, ast.Constant)
            and isinstance(path_node.value, str)
        ):
            raise ValueError(f"operation registry entry at line {node.lineno} is not static")
        operations[operation_node.value] = (method_node.attr, path_node.value)
    return operations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-contract", type=Path)
    arguments = parser.parse_args()
    contract_bytes = CONTRACT_PATH.read_bytes()
    payload = json.loads(contract_bytes)
    expected = _contract_operations(payload)
    actual = _source_operations()
    failures: list[str] = []
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        drifted = sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        )
        failures.append(
            f"operation registry drift: missing={missing} extra={extra} drifted={drifted}"
        )
    if arguments.backend_contract is not None:
        backend_bytes = arguments.backend_contract.read_bytes()
        if backend_bytes != contract_bytes:
            failures.append("vendored Research OpenAPI differs from backend-authored artifact")
    if failures:
        print("Research OpenAPI contract check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    digest = hashlib.sha256(contract_bytes).hexdigest()
    print(
        f"Research OpenAPI contract check passed: operations={len(expected)} "
        f"sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
