"""Schema-derived fake responses exercise confirmed call_tool serialization defects.

See backend scientific delivery contract; audit mechanical-sdk-backend.md MX-10.
No local I/O tools, Index calls or provider calls are made.
"""

import json
import re
from pathlib import Path

import httpx
import pytest
from synth_ai.mcp.research.server import ResearchMcpServer

SPEC = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "contract/fixtures/backend_full_openapi.generated.json"
    ).read_text()
)
schemas = SPEC["components"]["schemas"]
UID = "00000000-0000-4000-8000-000000000001"


def example(s, depth=0):
    if not isinstance(s, dict) or depth > 6:
        return None
    if "$ref" in s:
        return example(schemas.get(s["$ref"].rsplit("/", 1)[-1], {}), depth + 1)
    for k in ("anyOf", "oneOf"):
        if k in s:
            for x in s[k]:
                if x.get("type") != "null":
                    return example(x, depth + 1)
            return None
    if "allOf" in s:
        out = {}
        for x in s["allOf"]:
            v = example(x, depth + 1)
            if isinstance(v, dict):
                out.update(v)
        return out
    if "const" in s:
        return s["const"]
    if "enum" in s:
        return s["enum"][0]
    if "default" in s and s["default"] is not None:
        return s["default"]
    t = s.get("type")
    if t == "string":
        f = s.get("format")
        if f == "date-time":
            return "2026-10-06T00:00:00Z"
        if f == "date":
            return "2026-10-06"
        if f == "uuid":
            return UID
        return "audit"
    if t == "integer":
        return max(1, s.get("minimum", 1))
    if t == "number":
        return 1.0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object" or "properties" in s:
        return {k: example(v, depth + 1) for k, v in s.get("properties", {}).items()}
    return None


def assert_serializable_public_tool(monkeypatch, case):
    routes = []
    for path, item in SPEC["paths"].items():
        pattern = re.compile("^" + re.sub(r"\\\{[^}]*\\\}", "[^/]+", re.escape(path)) + "$")
        for method, operation in item.items():
            if method in {"get", "post", "put", "patch", "delete"}:
                routes.append((method.upper(), path, pattern, operation))
    routes.sort(key=lambda route: -sum("{" not in segment for segment in route[1].split("/")))

    def send(client, request, *args, **kwargs):
        hit = next(
            (
                route
                for route in routes
                if route[0] == request.method and route[2].match(request.url.path)
            ),
            None,
        )
        assert hit is not None, f"MX-10: fixture route absent for {case['tool']}"
        operation = hit[3]
        response_schema = next(
            (
                operation["responses"][code]["content"]["application/json"]["schema"]
                for code in ("200", "201", "202")
                if "application/json"
                in operation.get("responses", {}).get(code, {}).get("content", {})
            ),
            {},
        )
        body = example(response_schema)
        return httpx.Response(200, json=body if body is not None else {}, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    server = ResearchMcpServer(
        api_key="offline-dummy", backend_base="http://offline.invalid", include_advanced_tools=True
    )
    result = server.call_tool(case["tool"], case["args"])
    try:
        json.dumps(result)
    except TypeError as failure:
        pytest.fail(
            f"MX-10/RR-12: {case['tool']} public result is not JSON serializable: {failure}"
        )
