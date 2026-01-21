"""Generate OpenAPI spec from JSON Schema event definitions.

This script generates an OpenAPI 3.1.0 spec from the event JSON Schemas
defined in synth_ai.sdk.shared.orchestration.events. The generated spec
can be used for:
- Rust code generation (openapi-generator)
- TypeScript type generation
- API documentation
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

from synth_ai.sdk.shared.orchestration.events import BASE_EVENT_SCHEMAS, get_registry


def _collect_schemas() -> Dict[str, Dict[str, Any]]:
    """Collect all event schemas from base schemas and registry."""
    # Start with base schemas
    schemas = dict(BASE_EVENT_SCHEMAS)

    # Add any algorithm-specific schemas from registry
    registry = get_registry()
    registered_schemas = registry.export_all_schemas()

    # Merge, preferring registered schemas over base
    schemas.update(registered_schemas)

    return schemas


def _schema_name(event_type: str) -> str:
    """Convert event type to a valid OpenAPI schema name."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", event_type).strip("_")
    return f"{cleaned}_event"


def _strip_schema_key(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove $schema key from schema (not valid in OpenAPI components)."""
    cleaned = copy.deepcopy(dict(schema))
    cleaned.pop("$schema", None)
    return cleaned


def main() -> None:
    """Generate OpenAPI spec and write to file."""
    schemas = _collect_schemas()
    components = {
        _schema_name(event_type): _strip_schema_key(schema)
        for event_type, schema in schemas.items()
    }

    openapi_spec = {
        "openapi": "3.1.0",
        "info": {"title": "Synth AI Job Events", "version": "1.0.0"},
        "components": {"schemas": components},
    }

    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "synth_ai" / "sdk" / "shared" / "orchestration" / "events" / "openapi.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(openapi_spec, indent=2))

    print(f"Generated OpenAPI spec with {len(components)} schemas at {output_path}")


if __name__ == "__main__":
    main()
