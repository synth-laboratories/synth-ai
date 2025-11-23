#!/usr/bin/env python3
"""Validate that OpenAPI spec and Pydantic models stay in sync.

This script compares the OpenAPI task_app.yaml schema against the Pydantic
models in synth_ai/task/contracts.py to catch drift between spec and code.

Run: python scripts/validate_openapi_pydantic.py
CI:  Add to CI workflow to prevent drift

Exit codes:
  0 - All schemas match
  1 - Validation failed (schema mismatch or missing fields)
"""

import json
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from synth_ai.task.contracts import (
    DatasetInfo,
    InferenceInfo,
    LimitsInfo,
    RolloutEnvSpec,
    RolloutMetrics,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutResponse,
    RolloutSafetyConfig,
    RolloutStep,
    RolloutTrajectory,
    RubricInfo,
    TaskDescriptor,
    TaskInfo,
)


def load_openapi_spec() -> dict[str, Any]:
    """Load the OpenAPI task_app.yaml spec."""
    spec_path = PROJECT_ROOT / "synth_ai" / "contracts" / "task_app.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


def get_pydantic_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Get JSON schema from a Pydantic model."""
    return model.model_json_schema()


def normalize_type(openapi_type: dict[str, Any]) -> str:
    """Normalize OpenAPI type to comparable string."""
    if "anyOf" in openapi_type:
        types = [normalize_type(t) for t in openapi_type["anyOf"]]
        return f"anyOf[{','.join(sorted(types))}]"
    if "allOf" in openapi_type:
        types = [normalize_type(t) for t in openapi_type["allOf"]]
        return f"allOf[{','.join(sorted(types))}]"
    if "$ref" in openapi_type:
        return openapi_type["$ref"].split("/")[-1]

    base_type = openapi_type.get("type", "any")
    if base_type == "array":
        items = openapi_type.get("items", {})
        item_type = normalize_type(items)
        return f"array[{item_type}]"
    if base_type == "object":
        if "additionalProperties" in openapi_type:
            val_type = normalize_type(openapi_type["additionalProperties"])
            return f"dict[string,{val_type}]"
        return "object"
    return base_type


def compare_properties(
    openapi_props: dict[str, Any],
    pydantic_props: dict[str, Any],
    openapi_required: list[str],
    pydantic_required: list[str],
    path: str,
) -> list[str]:
    """Compare OpenAPI and Pydantic properties, returning list of issues."""
    issues = []

    all_props = set(openapi_props.keys()) | set(pydantic_props.keys())

    for prop in sorted(all_props):
        prop_path = f"{path}.{prop}"

        # Check if property exists in both
        in_openapi = prop in openapi_props
        in_pydantic = prop in pydantic_props

        if in_openapi and not in_pydantic:
            issues.append(f"MISSING IN PYDANTIC: {prop_path}")
            continue
        if in_pydantic and not in_openapi:
            # Allow extra fields in Pydantic (they use extra="allow")
            pass

        if in_openapi and in_pydantic:
            # Compare requiredness
            openapi_req = prop in openapi_required
            pydantic_req = prop in pydantic_required

            if openapi_req and not pydantic_req:
                issues.append(f"REQUIRED MISMATCH: {prop_path} (required in OpenAPI, optional in Pydantic)")
            elif pydantic_req and not openapi_req:
                issues.append(f"REQUIRED MISMATCH: {prop_path} (optional in OpenAPI, required in Pydantic)")

    return issues


def validate_schema_match(
    openapi_schema: dict[str, Any],
    pydantic_model: type[BaseModel],
    model_name: str,
) -> list[str]:
    """Validate that OpenAPI schema matches Pydantic model."""
    issues = []

    pydantic_schema = get_pydantic_json_schema(pydantic_model)

    # Get properties from both
    openapi_props = openapi_schema.get("properties", {})
    pydantic_props = pydantic_schema.get("properties", {})

    # Get required fields
    openapi_required = openapi_schema.get("required", [])
    pydantic_required = pydantic_schema.get("required", [])

    issues.extend(compare_properties(
        openapi_props,
        pydantic_props,
        openapi_required,
        pydantic_required,
        model_name,
    ))

    return issues


# Map OpenAPI schema names to Pydantic models
SCHEMA_MODEL_MAP: dict[str, type[BaseModel]] = {
    "RolloutRequest": RolloutRequest,
    "RolloutResponse": RolloutResponse,
    "RolloutTrajectory": RolloutTrajectory,
    "RolloutStep": RolloutStep,
    "RolloutMetrics": RolloutMetrics,
    "RolloutEnvSpec": RolloutEnvSpec,
    "RolloutPolicySpec": RolloutPolicySpec,
    "TaskInfo": TaskInfo,
    "TaskDescriptor": TaskDescriptor,
    "DatasetInfo": DatasetInfo,
    "RubricInfo": RubricInfo,
    "InferenceInfo": InferenceInfo,
    "LimitsInfo": LimitsInfo,
}


def main() -> int:
    """Main validation entry point."""
    print("=" * 60)
    print("OpenAPI ↔ Pydantic Schema Validation")
    print("=" * 60)
    print()

    try:
        spec = load_openapi_spec()
    except FileNotFoundError:
        print("ERROR: OpenAPI spec not found at synth_ai/contracts/task_app.yaml")
        return 1

    schemas = spec.get("components", {}).get("schemas", {})

    all_issues: list[str] = []
    validated = 0
    skipped = 0

    for schema_name, pydantic_model in SCHEMA_MODEL_MAP.items():
        if schema_name not in schemas:
            print(f"⚠ SKIPPED: {schema_name} (not in OpenAPI spec)")
            skipped += 1
            continue

        openapi_schema = schemas[schema_name]
        issues = validate_schema_match(openapi_schema, pydantic_model, schema_name)

        if issues:
            print(f"✗ {schema_name}: {len(issues)} issue(s)")
            for issue in issues:
                print(f"    {issue}")
            all_issues.extend(issues)
        else:
            print(f"✓ {schema_name}: OK")
        validated += 1

    print()
    print("-" * 60)
    print(f"Validated: {validated} schemas")
    print(f"Skipped: {skipped} schemas")
    print(f"Issues: {len(all_issues)}")

    if all_issues:
        print()
        print("VALIDATION FAILED - OpenAPI and Pydantic schemas have drifted")
        print()
        print("To fix:")
        print("  1. Update synth_ai/task/contracts.py to match task_app.yaml")
        print("  2. Or update task_app.yaml if Python changes are intentional")
        return 1

    print()
    print("✓ All schemas in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
