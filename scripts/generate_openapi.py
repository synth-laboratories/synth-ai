"""Generate OpenAPI spec from JSON Schema event definitions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping

from synth_ai.sdk.api.orchestration.events import registry
from synth_ai.sdk.api.orchestration.events import (
    graphgen,
    policy_eval,
    prompt_learning,
    verifier_eval,
)


def _collect_schemas() -> Dict[str, Dict[str, Any]]:
    return registry.merge_event_schemas(
        prompt_learning.EVENT_SCHEMAS,
        graphgen.EVENT_SCHEMAS,
        policy_eval.POLICY_EVAL_EVENT_SCHEMAS,
        verifier_eval.VERIFIER_EVAL_EVENT_SCHEMAS,
    )


def _schema_name(event_type: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", event_type).strip("_")
    return f"{cleaned}_event"


def _strip_schema_key(schema: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned = copy.deepcopy(schema)
    cleaned.pop("$schema", None)
    return cleaned


def main() -> None:
    schemas = _collect_schemas()
    components = { _schema_name(event_type): _strip_schema_key(schema) for event_type, schema in schemas.items() }

    openapi_spec = {
        "openapi": "3.1.0",
        "info": {"title": "Synth AI Job Events", "version": "1.0.0"},
        "components": {"schemas": components},
    }

    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "synth_ai/sdk/api/orchestration/events/openapi.yaml"
    output_path.write_text(json.dumps(openapi_spec, indent=2))


if __name__ == "__main__":
    main()
