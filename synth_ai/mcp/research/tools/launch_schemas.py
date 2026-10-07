"""Shared backend launch input projection for MCP.

See backend notes/specifications/tanha/current/systems/smr/launch_resource_inventory.md
and packages/smr/run_provenance.py. A launch names exact deployment receipts.
"""

from typing import Any


def launch_provenance_properties() -> dict[str, Any]:
    return {
        "deployment_pins": {
            "type": "array",
            "minItems": 1,
            "description": "Exact resolved deployment receipts.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "repository": {"type": "string", "minLength": 1, "maxLength": 512},
                    "commit_sha": {"type": "string", "minLength": 1, "maxLength": 64},
                    "deployment_id": {"type": ["string", "null"], "maxLength": 512},
                    "artifact_digest": {"type": ["string", "null"], "maxLength": 512},
                    "environment": {"type": "string", "minLength": 1, "maxLength": 128},
                    "resolved_at": {"type": "string", "format": "date-time"},
                },
                "required": ["repository", "commit_sha", "environment", "resolved_at"],
            },
        },
        "provenance_mode": {
            "type": "string",
            "enum": ["live", "dry_run"],
            "description": "Launch provenance posture; both modes require non-empty pins.",
        },
    }


def launch_resource_bindings_schema() -> dict[str, Any]:
    inventories = ("model_file_ids", "external_repository_ids", "credential_ref_ids")
    return {
        "type": "object",
        "description": "Explicit launch resource inventories; empty lists select none.",
        "properties": {
            **{name: {"type": "array", "items": {"type": "string"}} for name in inventories},
            "external_repositories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 255},
                        "url": {"type": "string", "minLength": 1, "maxLength": 4000},
                        "default_branch": {"type": ["string", "null"]},
                        "role": {"type": "string"},
                        "split_role": {
                            "type": ["string", "null"],
                            "enum": ["visible_train", "heldout", None],
                        },
                        "metadata": {"type": "object"},
                    },
                    "required": ["name", "url"],
                },
            },
        },
        "required": list(inventories),
    }
