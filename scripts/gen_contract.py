#!/usr/bin/env python3
"""Generate public models, operations and MCP tools from the pinned OpenAPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import keyword
import pprint
import subprocess
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "contracts/synth-v1.openapi.yaml"


def generate() -> dict[Path, str]:
    pin = tomllib.loads((ROOT / "contracts/SOURCE.toml").read_text())
    if pin["sha256"] != hashlib.sha256(SPEC.read_bytes()).hexdigest():
        raise ValueError("contract pin hash mismatch")
    doc = yaml.safe_load(SPEC.read_text())
    schemas = dict(doc["components"]["schemas"])

    def resolve(value):
        if "$ref" in value:
            resolved = doc
            for part in value["$ref"][2:].split("/"):
                resolved = resolved[part]
            return resolved
        return value

    def flatten(s):
        if "allOf" not in s:
            return s
        out = {"type": "object", "properties": {}, "required": []}
        for part in s["allOf"]:
            part = flatten(resolve(part))
            out["properties"].update(part.get("properties", {}))
            out["required"] += part.get("required", [])
        return out

    ops = {}
    for path, item in doc["paths"].items():
        if not path.startswith("/v1/"):
            continue
        for method, op in item.items():
            if method not in {"get", "post", "patch", "delete"}:
                continue
            if op.get("x-synth-status") != "implemented":
                continue
            oid = op["operationId"]
            parameters = [resolve(p) for p in op.get("parameters", [])]
            req = None
            if "requestBody" in op:
                body = resolve(op["requestBody"])["content"]["application/json"]["schema"]
                req = body.get("$ref", "").split("/")[-1]
                if not req:
                    req = oid[0].upper() + oid[1:] + "Request"
                    schemas[req] = body
            result = None
            binary = False
            for status, r in op["responses"].items():
                if status.startswith("2"):
                    r = resolve(r)
                    content = r.get("content", {})
                    if "application/json" in content:
                        result = (
                            content["application/json"]["schema"].get("$ref", "").split("/")[-1]
                        )
                    elif content:
                        binary = True
                    break
            ops[oid] = {
                "method": method.upper(),
                "path": path,
                "scope": op["x-synth-scope"],
                "path_params": [p["name"] for p in parameters if p["in"] == "path"],
                "query_params": [p["name"] for p in parameters if p["in"] == "query"],
                "idempotent": op["x-synth-idempotency-key"],
                "retry": op["x-synth-retry"],
                "request_model": req,
                "response_model": result,
                "binary": binary,
                "auth": op.get("security") != [],
            }
    schemas = {n: flatten(s) for n, s in schemas.items()}

    def annotation(s):
        if "$ref" in s:
            return s["$ref"].split("/")[-1]
        variants = s.get("anyOf") or s.get("oneOf")
        if variants:
            return " | ".join(dict.fromkeys(annotation(v) for v in variants))
        if "const" in s:
            return "Literal[" + repr(s["const"]) + "]"
        if "enum" in s:
            return "Literal[" + ", ".join(repr(v) for v in s["enum"]) + "]"
        typ = s.get("type")
        if isinstance(typ, list):
            return " | ".join(annotation({**s, "type": t}) for t in typ)
        if typ == "array":
            return "list[" + annotation(s.get("items", {})) + "]"
        if typ == "object":
            additional = s.get("additionalProperties")
            return (
                "dict[str, "
                + (annotation(additional) if isinstance(additional, dict) else "Any")
                + "]"
            )
        base = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "null": "None",
        }.get(typ, "Any")
        if typ == "null":
            return "None"  # a nullable string's pattern or length applies only to the string
        args = []
        for k, v in s.items():
            dest = {
                "pattern": "pattern",
                "minLength": "min_length",
                "maxLength": "max_length",
                "minimum": "ge",
                "maximum": "le",
            }.get(k)
            if dest:
                args.append(dest + "=" + repr(v))
        metadata = ["Field(" + ", ".join(args) + ")"] if args else []
        if "x-synth-max-bytes" in s:
            metadata.append("AfterValidator(utf8_max_bytes(" + str(s["x-synth-max-bytes"]) + "))")
        return "Annotated[" + base + ", " + ", ".join(metadata) + "]" if metadata else base

    def dependencies(s):
        if isinstance(s, dict):
            found = {s["$ref"].split("/")[-1]} if "$ref" in s else set()
            for v in s.values():
                found |= dependencies(v)
            return found
        if isinstance(s, list):
            found = set()
            for v in s:
                found |= dependencies(v)
            return found
        return set()

    ordered = []
    remaining = set(schemas)
    while remaining:
        ready = sorted(n for n in remaining if not (dependencies(schemas[n]) & remaining))
        if not ready:
            raise ValueError("recursive schemas require explicit generation support")
        ordered.extend(ready)
        remaining -= set(ready)
    lines = [
        "# Generated by scripts/gen_contract.py; do not edit.",
        "from __future__ import annotations",
        "from typing import Annotated, Any, Literal, TypeAlias",
        "from pydantic import AfterValidator, BaseModel, ConfigDict, Field",
        "from synth_ai._validation import utf8_max_bytes",
        "",
    ]
    modelnames = []
    for n in ordered:
        s = schemas[n]
        if s.get("type") == "object" and "properties" in s:
            modelnames.append(n)
            lines += [
                "class " + n + "(BaseModel):",
                "    model_config = ConfigDict(extra="
                + repr("forbid" if s.get("additionalProperties") is False else "allow")
                + ", populate_by_name=True)",
            ]
            for field, value in s["properties"].items():
                name = (
                    field + "_"
                    if keyword.iskeyword(field)
                    or field in ("model_config", "schema", "json", "copy")
                    else field
                )
                typ = annotation(value)
                required = field in s.get("required", [])
                if not required and "None" not in typ.split(" | "):
                    typ += " | None"
                default = "" if required else " = None"
                if name != field:
                    default = (
                        " = Field("
                        + ("" if required else "default=None, ")
                        + "alias="
                        + repr(field)
                        + ")"
                    )
                lines.append("    " + name + ": " + typ + default)
            lines += [""]
        else:
            lines += [n + ": TypeAlias = " + annotation(s), ""]
    lines += (
        ["MODELS: dict[str, type[BaseModel]] = {"]
        + ["    " + repr(n) + ": " + n + "," for n in modelnames]
        + ["}", ""]
    )
    operations = (
        """# Generated by scripts/gen_contract.py; do not edit.
from typing import TypedDict

class Operation(TypedDict):
    method: str
    path: str
    scope: str
    path_params: list[str]
    query_params: list[str]
    idempotent: bool
    retry: str
    request_model: str | None
    response_model: str | None
    binary: bool
    auth: bool

OPERATIONS: dict[str, Operation] = """
        + pprint.pformat(ops, sort_dicts=True, width=100)
        + "\n"
    )
    catalog = json.loads((ROOT / "contracts/mcp-tools.json").read_text())

    def expand(value):
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            if "$ref" in value:
                value = {**resolve(value), **{k: v for k, v in value.items() if k != "$ref"}}
            return {k: expand(v) for k, v in value.items()}
        return value

    expected_tools = []
    for path, item in doc["paths"].items():
        for operation in item.values():
            if (
                not isinstance(operation, dict)
                or "x-synth-mcp-name" not in operation
                or operation.get("x-synth-status") != "implemented"
            ):
                continue
            params = [resolve(p) for p in operation.get("parameters", [])]
            properties = {
                p["name"]: expand(p["schema"]) for p in params if p["in"] in ("path", "query")
            }
            required = [
                p["name"] for p in params if p["in"] in ("path", "query") and p.get("required")
            ]
            if "requestBody" in operation:
                body = operation["requestBody"]
                properties["body"] = expand(body["content"]["application/json"]["schema"])
                if body.get("required"):
                    required.append("body")
            if any(p["name"] == "Idempotency-Key" for p in params):
                properties["idempotency_key"] = {"type": "string", "minLength": 1, "maxLength": 128}
                required.append("idempotency_key")
            expected_tools.append(
                {
                    "name": operation["x-synth-mcp-name"],
                    "description": operation["summary"],
                    "scope": operation["x-synth-scope"],
                    "operationId": operation["operationId"],
                    "inputSchema": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            )
    # The contract is the authority. The catalog is a generated consumer too.
    catalog["tools"] = expected_tools
    tools = (
        "# Generated by scripts/gen_contract.py; do not edit.\nfrom typing import Any\nTOOLS: list[dict[str, Any]] = "
        + pprint.pformat(catalog["tools"], sort_dicts=True, width=100)
        + "\n"
    )
    docs = (
        "# MCP tools\n\nGenerated from the pinned public OpenAPI and hosted catalog.\n\n| Tool | Operation | Scope |\n| --- | --- | --- |\n"
        + "".join(
            "| " + t["name"] + " | " + t["operationId"] + " | " + t["scope"] + " |\n"
            for t in catalog["tools"]
        )
    )
    registry = doc["x-synth-scope-registry"]
    policy_source = "# Generated by scripts/gen_contract.py; do not edit.\n"
    policy_source += (
        "SDK_OAUTH_SCOPES = " + repr(tuple(s["name"] for s in registry if s["sdk"])) + "\n"
    )
    policy_source += (
        "MESSAGE_MAX_BYTES = "
        + str(schemas["LaunchRequest"]["properties"]["message"]["x-synth-max-bytes"])
        + "\n"
    )
    outputs = {
        ROOT / "synth_ai/_generated/policy.py": policy_source,
        ROOT
        / "synth_ai/_version.py": "# Generated from pyproject.toml; do not edit.\n__version__ = "
        + repr(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"])
        + "\n",
        ROOT / "contracts/mcp-tools.json": json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        ROOT / "synth_ai/_generated/models.py": "\n".join(lines),
        ROOT / "synth_ai/_generated/operations.py": operations,
        ROOT / "synth_ai/mcp/tools.py": tools,
        ROOT / "docs/mcp-tools.md": docs,
    }
    for path, source in list(outputs.items()):
        if path.suffix == ".py":
            fixed = subprocess.run(
                ["ruff", "check", "--select", "I", "--fix", "--stdin-filename", str(path), "-"],
                input=source,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
            outputs[path] = subprocess.run(
                ["ruff", "format", "--stdin-filename", str(path), "-"],
                input=fixed,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, source in generate().items():
        if args.check:
            if not path.exists() or path.read_text() != source:
                raise SystemExit("generated drift: " + str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
    print("contract sha256 " + hashlib.sha256(SPEC.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
