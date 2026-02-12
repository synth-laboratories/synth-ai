from __future__ import annotations

import json
from pathlib import Path


def _load_contract() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    contract_path = repo_root / "synth_ai_core" / "assets" / "lever_sensor_v1_schema.json"
    return json.loads(contract_path.read_text(encoding="utf-8"))


def _enum_at(schema: dict, *path: str) -> list[str]:
    node: object = schema
    for key in path:
        if not isinstance(node, dict):
            raise AssertionError(f"Expected dict while resolving path segment '{key}'")
        node = node[key]
    if not isinstance(node, list):
        raise AssertionError(f"Expected list at path {'/'.join(path)}")
    return [str(item) for item in node]


def test_lever_sensor_contract_has_frozen_scope_order() -> None:
    schema = _load_contract()
    assert _enum_at(schema, "properties", "scope_key_order", "items", "enum") == [
        "org",
        "project",
        "horizon",
        "job",
        "stage",
        "seed",
        "rollout",
        "graph_node",
        "tool_call",
        "user",
    ]


def test_lever_sensor_contract_has_required_kinds_and_frame_fields() -> None:
    schema = _load_contract()
    assert _enum_at(schema, "properties", "required_lever_kinds", "items", "enum") == [
        "prompt",
        "context",
        "code",
        "constraint",
        "note",
        "spec",
        "graph_yaml",
        "variable",
        "experiment",
    ]
    assert _enum_at(schema, "properties", "required_sensor_kinds", "items", "enum") == [
        "reward",
        "timing",
        "rollout",
        "resource",
        "safety",
        "quality",
        "trace",
        "context_apply",
        "experiment",
    ]
    assert _enum_at(
        schema,
        "properties",
        "sensor_frame_minimum",
        "properties",
        "required_fields",
        "items",
        "enum",
    ) == ["scope", "sensors", "lever_versions", "trace_ids", "created_at"]
    assert _enum_at(
        schema,
        "properties",
        "sensor_frame_minimum",
        "properties",
        "required_sensor_fields",
        "items",
        "enum",
    ) == ["sensor_id", "kind", "scope", "value", "timestamp"]
