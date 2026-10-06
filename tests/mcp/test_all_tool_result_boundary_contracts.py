"""SYN-4007: every advertised tool crosses the same JSON result boundary.

Schema-derived arguments and substituted handlers isolate serialization for
all tools without invoking local I/O, Index or providers. The 14 original
backend-response behavioral cases separately exercise their real handlers.
"""

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from synth_ai.mcp.research.server import ResearchMcpServer


@dataclass
class Response:
    observed_at: datetime
    evidence: tuple[dict, ...]


def example(schema, definitions, depth=0):
    assert depth < 30, "schema fixture recursion exceeded bound"
    if "$ref" in schema:
        return example(definitions[schema["$ref"].rsplit("/", 1)[-1]], definitions, depth + 1)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    for union in ("oneOf", "anyOf"):
        if union in schema:
            selected = next(
                (item for item in schema[union] if item.get("type") == "null"), schema[union][0]
            )
            return example(selected, definitions, depth + 1)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = "null" if "null" in kind else kind[0]
    if kind == "null":
        return None
    if kind == "boolean":
        return False
    if kind in {"number", "integer"}:
        return max(schema.get("minimum", 1), schema.get("exclusiveMinimum", 0) + 1)
    if kind == "array":
        return [
            example(schema.get("items", {}), definitions, depth + 1)
            for _ in range(schema.get("minItems", 0))
        ]
    if kind == "object" or "properties" in schema:
        return {
            key: example(schema.get("properties", {}).get(key, {}), definitions, depth + 1)
            for key in schema.get("required", [])
        }
    if kind == "string":
        if schema.get("format") == "date-time":
            return "2026-10-06T00:00:00Z"
        prefix = re.fullmatch(
            r"\^([a-zA-Z0-9_.:-]+)\[0-9a-f\]\{(\d+)\}\$", schema.get("pattern", "")
        )
        if prefix:
            return prefix.group(1) + "a" * int(prefix.group(2))
        candidates = [
            "audit",
            "00000000-0000-4000-8000-000000000001",
            "a" * 64,
            "a" * 40,
            "a" * 32,
            "sha256:" + "a" * 64,
            "a" * 40,
            "a" * 32,
            "adm_" + "0" * 26,
            "https://offline.invalid",
            "1",
        ]
        for value in candidates:
            if len(value) < schema.get("minLength", 0):
                value += "a" * (schema["minLength"] - len(value))
            if len(value) > schema.get("maxLength", 1000000):
                continue
            if "pattern" not in schema or re.search(schema["pattern"], value):
                return value
        raise AssertionError(
            "schema fixture needs a reviewed string pattern example: " + str(schema.get("pattern"))
        )
    return None


def test_every_advertised_tool_returns_json_through_public_call__SYN4007():
    server = ResearchMcpServer(
        api_key="offline-dummy", backend_base="http://offline.invalid", include_advanced_tools=True
    )
    advertised = server.tool_definitions()
    assert len(advertised) >= 330
    payload = Response(datetime(2026, 10, 6, tzinfo=timezone.utc), ({"retained": True},))
    count = 0
    for definition in advertised:
        arguments = example(definition.input_schema, definition.input_schema.get("$defs", {}))
        server._tools[definition.name] = replace(definition, handler=lambda args: payload)
        result = server.call_tool(definition.name, arguments)
        assert result["observed_at"] == "2026-10-06T00:00:00+00:00"
        assert result["evidence"] == [{"retained": True}]
        json.dumps(result, allow_nan=False)
        count += 1
    assert count == len(advertised)
