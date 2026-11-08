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
                "idempotency": op["x-synth-idempotency"],
                "request_model": req,
                "response_model": result,
                "binary": binary,
                "auth": op.get("security") != [],
            }
    schemas = {n: flatten(s) for n, s in schemas.items()}

    # Lift every structured inline object into a named model. JSON maps remain
    # maps only when the contract deliberately leaves their keys open.
    def lift(schema, name):
        schema = dict(schema)
        if schema.get("type") == "object" and len(schema.get("oneOf", [])) == 1:
            branch = schema.pop("oneOf")[0]
            schema["properties"] = {
                key: {**value, **branch.get("properties", {}).get(key, {})}
                for key, value in schema.get("properties", {}).items()
            }
        if "properties" in schema:
            schema["properties"] = {
                key: lift_child(value, name + "".join(part.title() for part in key.split("_")))
                for key, value in schema["properties"].items()
            }
        if "items" in schema:
            schema["items"] = lift_child(schema["items"], name + "Item")
        for variant in ("oneOf", "anyOf"):
            if variant in schema:
                schema[variant] = [
                    lift_child(v, name + str(i)) for i, v in enumerate(schema[variant])
                ]
        return schema

    inline_names = {
        "LaunchRequestInputs": "RunInputs",
        "RunInputsFilesItem": "InputFile",
        "UploadPut": "UploadGrant",
        "RunTaskPageItemsItem": "RunTask",
        "OwnerResourceRequestSelection": "PoolSelection",
    }

    def lift_child(schema, name):
        name = inline_names.get(name, name)
        types = schema.get("type", [])
        if "properties" in schema and (types == "object" or "object" in types):
            nullable = isinstance(types, list) and "null" in types
            schemas[name] = lift({**schema, "type": "object"}, name)
            ref = {"$ref": "#/components/schemas/" + name}
            return {"anyOf": [ref, {"type": "null"}]} if nullable else ref
        return lift(schema, name)

    for name in list(schemas):
        schemas[name] = lift(schemas[name], name)
    enums = {"RunState": schemas["RunSnapshot"]["properties"]["state"]["examples"]}

    def annotation(s, name=""):
        if "$ref" in s:
            return s["$ref"].split("/")[-1]
        variants = s.get("anyOf") or s.get("oneOf")
        if variants:
            result = " | ".join(dict.fromkeys(annotation(v) for v in variants))
            event_types = {
                v["x-synth-event-kind"]: annotation(v)
                for v in variants
                if "x-synth-event-kind" in v
            }
            if event_types:
                choices = ", ".join(
                    repr(kind) + ": " + model for kind, model in event_types.items()
                )
                return (
                    "Annotated[" + result + ", BeforeValidator(event_payload({" + choices + "}))]"
                )
            return result
        if "const" in s:
            literal = "Literal[" + repr(s["const"]) + "]"
            return (
                "Annotated[" + literal + ", BeforeValidator(strict_int)]"
                if type(s["const"]) is int
                else literal
            )
        if "enum" in s:
            if name and all(isinstance(value, str) for value in s["enum"]):
                enums[name] = s["enum"]
                return "Annotated[" + name + ", BeforeValidator(enum_value(" + name + "))]"
            return "Literal[" + ", ".join(repr(v) for v in s["enum"]) + "]"
        typ = s.get("type")
        if isinstance(typ, list):
            return " | ".join(annotation({**s, "type": t}) for t in typ)
        if typ == "array":
            base = "list[" + annotation(s.get("items", {})) + "]"
            meta = []
            bounds = [
                name + "=" + str(s[key])
                for key, name in (("minItems", "min_length"), ("maxItems", "max_length"))
                if key in s
            ]
            if bounds:
                meta.append("Field(" + ", ".join(bounds) + ")")
            if s.get("uniqueItems"):
                meta.append("AfterValidator(unique_items)")
            return "Annotated[" + base + ", " + ", ".join(meta) + "]" if meta else base
        if typ == "object":
            additional = s.get("additionalProperties")
            return (
                "dict[str, "
                + (annotation(additional) if isinstance(additional, dict) else "JsonValue")
                + "]"
            )
        base = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "null": "None",
        }.get(typ, "JsonValue")
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
        if s.get("x-synth-budget"):
            metadata.append("AfterValidator(budget_amount)")
        if s.get("x-synth-no-nul"):
            metadata.append("AfterValidator(no_nul)")
        if s.get("x-synth-message-text"):
            metadata.append("AfterValidator(message_text)")
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
        "from typing import Annotated, Literal, TypeAlias",
        "from enum import StrEnum",
        "from pydantic import AfterValidator, BeforeValidator, BaseModel, ConfigDict, Field, JsonValue",
        "from synth_ai._wire import WireModel",
        "from synth_ai._protocol import CompletionMilestone",
        "from synth_ai._validation import budget_amount, enum_value, event_payload, message_text, no_nul, strict_int, unique_items, utf8_max_bytes",
        "",
    ]
    preamble_length = len(lines)
    modelnames = []
    for n in ordered:
        s = schemas[n]
        if s.get("type") == "object" and "properties" in s:
            modelnames.append(n)
            lines += [
                "class " + n + "(WireModel):",
                "    model_config = ConfigDict(extra="
                + repr("forbid" if s.get("additionalProperties") is False else "allow")
                + ", populate_by_name=True, strict=True, frozen=True, allow_inf_nan=False)",
            ]
            for field, value in s["properties"].items():
                name = (
                    field + "_"
                    if keyword.iskeyword(field)
                    or field in ("model_config", "schema", "json", "copy")
                    else field
                )
                typ = annotation(value, n + "".join(part.title() for part in field.split("_")))
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
            if n == "RunSnapshot":
                lines += [
                    "    @property",
                    "    def known_state(self) -> RunState | None:",
                    "        try:",
                    "            return RunState(self.state)",
                    "        except ValueError:",
                    "            return None",
                    "",
                    "    @property",
                    "    def finished(self) -> bool:",
                    "        return self.execution_closed and self.accounting_final and self.cleanup_complete",
                    "",
                    "    def reached(self, milestone: CompletionMilestone) -> bool:",
                    "        milestone = CompletionMilestone(milestone)",
                    "        if milestone is CompletionMilestone.FINISHED:",
                    "            return self.finished",
                    "        if milestone is CompletionMilestone.EXECUTION_CLOSED:",
                    "            return self.execution_closed",
                    "        return self.result_available",
                ]
            lines += [""]
        else:
            lines += [n + ": TypeAlias = " + annotation(s), ""]
    enum_lines = []
    for name, values in sorted(enums.items()):
        enum_lines += ["class " + name + "(StrEnum):"]
        for value in values:
            member = "".join(c.upper() if c.isalnum() else "_" for c in value)
            if member[0].isdigit():
                member = "VALUE_" + member
            enum_lines.append("    " + member + " = " + repr(value))
        enum_lines.append("")
    lines[preamble_length:preamble_length] = enum_lines
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
    idempotency: str
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
    for _path, item in doc["paths"].items():
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
        "# Generated by scripts/gen_contract.py; do not edit.\n"
        "from typing import TypedDict\nfrom pydantic import JsonValue\n"
        "class ToolSchema(TypedDict):\n    type: str\n    properties: dict[str, JsonValue]\n    required: list[str]\n    additionalProperties: bool\n"
        "class ToolDefinition(TypedDict):\n    name: str\n    description: str\n    scope: str\n    operationId: str\n    inputSchema: ToolSchema\n"
        "TOOLS: list[ToolDefinition] = "
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
                [
                    "ruff",
                    "check",
                    "--select",
                    "I,RUF036",
                    "--fix",
                    "--stdin-filename",
                    str(path),
                    "-",
                ],
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
