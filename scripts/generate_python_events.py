"""Generate Pydantic event models from JSON Schema definitions."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping

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


def _class_name(event_type: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", event_type).strip("_")
    parts = [p for p in cleaned.split("_") if p]
    return "".join(part.capitalize() for part in parts) + "Event"


def _data_class_name(event_class: str) -> str:
    return f"{event_class}Data"


def _schema_type_to_py(schema: Mapping[str, Any]) -> str:
    if "enum" in schema:
        values = ", ".join(repr(v) for v in schema["enum"])
        return f"Literal[{values}]"

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        optional = "null" in schema_type
        types = [t for t in schema_type if t != "null"]
        inner = _schema_type_to_py({**schema, "type": types[0]}) if types else "Any"
        return f"Optional[{inner}]" if optional else inner

    if schema_type == "string":
        return "str"
    if schema_type == "integer":
        return "int"
    if schema_type == "number":
        return "float"
    if schema_type == "boolean":
        return "bool"
    if schema_type == "array":
        items = schema.get("items")
        inner = _schema_type_to_py(items) if isinstance(items, dict) else "Any"
        return f"List[{inner}]"
    if schema_type == "object":
        return "Dict[str, Any]"

    return "Any"


def _extract_data_schema(schema: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for subschema in schema.get("allOf", []):
        properties = subschema.get("properties", {})
        if "data" in properties:
            return properties["data"]
    return None


def _requires_data(schema: Mapping[str, Any]) -> bool:
    for subschema in schema.get("allOf", []):
        required = subschema.get("required", [])
        if "data" in required:
            return True
    return False


def _emit_model_fields(
    properties: Mapping[str, Any],
    required: Iterable[str],
) -> List[str]:
    required_set = set(required)
    lines: List[str] = []
    for name in sorted(properties.keys()):
        prop_schema = properties[name]
        annotation = _schema_type_to_py(prop_schema)
        if name not in required_set and not annotation.startswith("Optional["):
            annotation = f"Optional[{annotation}]"
        default = "" if name in required_set else " = None"
        lines.append(f"    {name}: {annotation}{default}")
    return lines


def generate_pydantic_models(schemas: Mapping[str, Mapping[str, Any]]) -> str:
    lines: List[str] = []
    lines.append('"""Generated event models. Do not edit by hand."""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from typing import Any, Dict, List, Literal, Optional")
    lines.append("")
    lines.append("from pydantic import BaseModel, ConfigDict")
    lines.append("")
    lines.append('EventLevel = Literal["info", "warn", "error"]')
    lines.append("")
    lines.append("class BaseJobEvent(BaseModel):")
    lines.append("    model_config = ConfigDict(extra=\"allow\")")
    lines.append("    job_id: str")
    lines.append("    seq: int")
    lines.append("    ts: str")
    lines.append("    type: str")
    lines.append("    level: EventLevel")
    lines.append("    message: str")
    lines.append("    data: Dict[str, Any] | None = None")
    lines.append("    run_id: str | None = None")
    lines.append("")

    for event_type, schema in sorted(schemas.items()):
        class_name = _class_name(event_type)
        data_schema = _extract_data_schema(schema)
        require_data = _requires_data(schema)

        data_class_name = _data_class_name(class_name)
        if data_schema:
            data_properties = data_schema.get("properties", {})
            data_required = data_schema.get("required", [])
            lines.append(f"class {data_class_name}(BaseModel):")
            lines.append("    model_config = ConfigDict(extra=\"allow\")")
            lines.extend(_emit_model_fields(data_properties, data_required))
            if not data_properties:
                lines.append("    pass")
            lines.append("")

        lines.append(f"class {class_name}(BaseJobEvent):")
        lines.append(f"    type: Literal[{event_type!r}]")
        if data_schema:
            data_type = data_class_name
            if not require_data:
                data_type = f"Optional[{data_type}]"
            default = "" if require_data else " = None"
            lines.append(f"    data: {data_type}{default}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    schemas = _collect_schemas()
    output_code = generate_pydantic_models(schemas)
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "backend/app/routes/prompt_learning/events/schemas.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_code)


if __name__ == "__main__":
    main()
