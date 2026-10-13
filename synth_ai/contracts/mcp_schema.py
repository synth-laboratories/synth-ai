"""Generate the MCP reference from the same catalog used by protocol discovery."""

from __future__ import annotations

import json

from synth_ai.contracts import operations, swarms
from synth_ai.mcp.catalog import TOOLS


def export_catalog() -> str:
    """Build a deterministic provisional schema artifact, without starting a server."""
    operation_ids = {
        operation.id for operation in (*operations.ALL_OPERATIONS, *swarms.ALL_OPERATIONS)
    }
    names = [tool.name for tool in TOOLS]
    if len(names) != len(set(names)) or any(tool.operation not in operation_ids for tool in TOOLS):
        raise ValueError("MCP catalog has duplicate names or unknown SDK operations")
    return (
        json.dumps(
            {
                "schema_version": "synth.mcp.v1",
                "status": "sdk_first_pending_backend_and_protocol_qualification",
                "tools": [
                    {
                        "name": tool.name,
                        "operation": tool.operation,
                        "description": tool.description,
                        "required_scope": tool.scope,
                        "mutates": tool.mutates,
                        "may_charge": tool.paid,
                        "keyed": tool.keyed,
                        "idempotent": tool.idempotent,
                        "input_schema": tool.input_schema(),
                        "output_schema": tool.output_schema(),
                    }
                    for tool in TOOLS
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    print(export_catalog(), end="")


if __name__ == "__main__":
    main()
