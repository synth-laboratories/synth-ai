"""Keep the vendored Index contract aligned with the public SDK registry."""

import json
from pathlib import Path

from synth_ai.sdk.index.client import OPERATIONS, PUBLIC_OPERATIONS

ROOT = Path(__file__).resolve().parents[1]
OPERATOR_ONLY = {"index.contributions.research.lookup"}


def test_index_openapi_operation_registry_is_exact() -> None:
    schema = json.loads((ROOT / "openapi/index-v1.json").read_text())
    wire = {
        operation["operationId"]: (method.upper(), path)
        for path, methods in schema["paths"].items()
        for method, operation in methods.items()
        if isinstance(operation, dict)
        and str(operation.get("operationId", "")).startswith("index.")
    }
    sdk = {**OPERATIONS, **PUBLIC_OPERATIONS}

    assert set(wire) == set(sdk) | OPERATOR_ONLY
    assert {
        name: wire.get(name)
        for name, binding in sdk.items()
        if wire.get(name) != (binding[0].upper(), binding[1])
    } == {}

