"""Deterministic shared-reference contract export, independent of transport."""

from __future__ import annotations

import json

from synth_ai.types.refs import Link, Producer, Ref

from .core import ALL_OPERATIONS
from .operations import PREFIX


def export_contract() -> str:
    return (
        json.dumps(
            {
                "schema_version": "core.v1",
                "status": "sdk_first_pending_backend",
                "models": {
                    name: model.model_json_schema()
                    for name, model in (("Ref", Ref), ("Producer", Producer), ("Link", Link))
                },
                "operations": [
                    {
                        "id": op.id,
                        "method": op.method,
                        "path": PREFIX + op.path,
                        "keyed": op.keyed,
                        "mutation": op.mutation,
                        "required_scope": op.required_scope,
                        "may_charge": op.may_charge,
                        "idempotent": op.idempotent,
                    }
                    for op in ALL_OPERATIONS
                ],
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    print(export_contract(), end="")
