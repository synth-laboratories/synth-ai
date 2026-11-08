#!/usr/bin/env python3
"""Generate public models, operations and MCP tools from the pinned OpenAPI."""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import keyword
import pprint
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import JsonValue, TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "contracts/synth-v1.openapi.yaml"


class JSONObject(dict[str, JsonValue]):
    """Validated open JSON, with typed accessors for schema vocabulary."""

    @classmethod
    def parse(cls, value: builtins.object) -> JSONObject:
        return cls.require(TypeAdapter(JsonValue).validate_python(value, strict=True))

    @classmethod
    def require(cls, value: JsonValue) -> JSONObject:
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return cls({key: cls.convert(item) for key, item in value.items()})

    @classmethod
    def convert(cls, value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return cls.require(value)
        if isinstance(value, list):
            return [cls.convert(item) for item in value]
        return value

    def object(self, key: str, *, optional: bool = False) -> JSONObject:
        return self.require(self.get(key, {}) if optional else self[key])

    def objects(self, key: str | None = None, *, optional: bool = False) -> dict[str, JSONObject]:
        source = self if key is None else self.object(key, optional=optional)
        return {name: self.require(value) for name, value in source.items()}

    def array(self, key: str, *, optional: bool = False) -> list[JsonValue]:
        value = self.get(key, []) if optional else self[key]
        if not isinstance(value, list):
            raise ValueError(f"{key}: expected an array")
        return value

    def object_array(self, key: str, *, optional: bool = False) -> list[JSONObject]:
        return [self.require(value) for value in self.array(key, optional=optional)]

    def text(self, key: str, default: str | None = None) -> str:
        value = self.get(key, default)
        if not isinstance(value, str):
            raise ValueError(f"{key}: expected a string")
        return value

    def flag(self, key: str, default: bool = False) -> bool:
        value = self.get(key, default)
        if type(value) is not bool:
            raise ValueError(f"{key}: expected a boolean")
        return value

    def strings(self, key: str, *, optional: bool = False) -> list[str]:
        result: list[str] = []
        for value in self.array(key, optional=optional):
            if not isinstance(value, str):
                raise ValueError(f"{key}: expected strings")
            result.append(value)
        return result


def schema_references(value: JsonValue) -> set[str]:
    if isinstance(value, dict):
        found: set[str] = (
            {JSONObject.require(value).text("$ref").split("/")[-1]} if "$ref" in value else set()
        )
        for child in value.values():
            found |= schema_references(child)
        return found
    if isinstance(value, list):
        found = set()
        for child in value:
            found |= schema_references(child)
        return found
    return set()


def sdk_model_roots(operations: Mapping[str, JSONObject]) -> set[str]:
    """Public exports, typed runtime imports, and dynamic transport lookups are live."""
    roots: set[str] = set()
    for operation in operations.values():
        for key in ("request_model", "response_model"):
            if operation.get(key) is not None:
                roots.add(operation.text(key))
    for path in (ROOT / "synth_ai").rglob("*.py"):
        if "_generated" in path.relative_to(ROOT).parts or path.name == "_sync.py":
            continue  # Generated files cannot keep themselves alive.
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "_generated.models",
                "synth_ai._generated.models",
            }:
                for alias in node.names:
                    if alias.name == "*":
                        raise ValueError(f"explicit model imports required: {path}")
                    if alias.name != "MODELS":
                        roots.add(alias.name)
    return roots


def select_sdk_schemas(schemas: Mapping[str, JSONObject], roots: set[str]) -> dict[str, JSONObject]:
    """Select the schema closure after flattening/lifting, before emitting Python."""
    owners = {name: name for name in schemas}
    for name, schema in schemas.items():
        for field, value in schema.objects("properties", optional=True).items():
            if "enum" in value and all(isinstance(item, str) for item in value.array("enum")):
                owners[name + "".join(part.title() for part in field.split("_"))] = name
    if "RunSnapshot" in schemas:
        owners["RunState"] = "RunSnapshot"  # Known-state enum emitted from its examples.
    if missing := roots - owners.keys():
        raise ValueError("SDK imports unknown generated types: " + ", ".join(sorted(missing)))
    pending = [owners[name] for name in roots]
    keep: set[str] = set()
    while pending:
        name = pending.pop()
        if name in keep:
            continue
        if name not in schemas:
            raise ValueError("SDK schema references unknown type: " + name)
        keep.add(name)
        pending.extend(schema_references(schemas[name]) - keep)
    return {name: schema for name, schema in schemas.items() if name in keep}


def generate() -> dict[Path, str]:
    pin = tomllib.loads((ROOT / "contracts/SOURCE.toml").read_text())
    if pin["sha256"] != hashlib.sha256(SPEC.read_bytes()).hexdigest():
        raise ValueError("contract pin hash mismatch")
    doc = JSONObject.parse(yaml.safe_load(SPEC.read_text()))
    schemas = doc.object("components").objects("schemas")

    def resolve(value: JSONObject) -> JSONObject:
        if "$ref" in value:
            resolved = doc
            for part in value.text("$ref")[2:].split("/"):
                resolved = resolved.object(part)
            return resolved
        return value

    def flatten(s: JSONObject) -> JSONObject:
        if "allOf" not in s:
            return s
        properties: dict[str, JsonValue] = {}
        required: list[JsonValue] = []
        for part in s.object_array("allOf"):
            part = flatten(resolve(part))
            properties.update(part.object("properties", optional=True))
            required.extend(part.array("required", optional=True))
        return JSONObject({"type": "object", "properties": properties, "required": required})

    ops: dict[str, JSONObject] = {}
    for path, item in doc.objects("paths").items():
        if not path.startswith("/v1/"):
            continue
        for method, op in item.items():
            if method not in {"get", "post", "patch", "delete"}:
                continue
            op = JSONObject.require(op)
            if op.get("x-synth-status") != "implemented":
                continue
            oid = op.text("operationId")
            parameters = [resolve(p) for p in op.object_array("parameters", optional=True)]
            req = None
            if "requestBody" in op:
                body = (
                    resolve(op.object("requestBody"))
                    .object("content")
                    .object("application/json")
                    .object("schema")
                )
                req = body.text("$ref", "").split("/")[-1]
                if not req:
                    req = oid[0].upper() + oid[1:] + "Request"
                    schemas[req] = body
            result = None
            binary = False
            for status, r in op.objects("responses").items():
                if status.startswith("2"):
                    r = resolve(r)
                    content = r.object("content", optional=True)
                    if "application/json" in content:
                        result = (
                            content.object("application/json")
                            .object("schema")
                            .text("$ref", "")
                            .split("/")[-1]
                        )
                    elif content:
                        binary = True
                    break
            ops[oid] = JSONObject(
                {
                    "method": method.upper(),
                    "path": path,
                    "scope": op.text("x-synth-scope"),
                    "path_params": [p.text("name") for p in parameters if p["in"] == "path"],
                    "query_params": [p.text("name") for p in parameters if p["in"] == "query"],
                    "idempotent": op["x-synth-idempotency-key"],
                    "retry": op["x-synth-retry"],
                    "idempotency": op["x-synth-idempotency"],
                    "request_model": req,
                    "response_model": result,
                    "binary": binary,
                    "auth": op.get("security") != [],
                }
            )
    schemas = {n: flatten(s) for n, s in schemas.items()}

    # Lift every structured inline object into a named model. JSON maps remain
    # maps only when the contract deliberately leaves their keys open.
    def lift(schema: JSONObject, name: str) -> JSONObject:
        schema = JSONObject(schema)
        if schema.get("type") == "object" and len(schema.array("oneOf", optional=True)) == 1:
            branch = schema.object_array("oneOf")[0]
            del schema["oneOf"]
            schema["properties"] = {
                key: JSONObject(
                    {
                        **value,
                        **branch.object("properties", optional=True).object(key, optional=True),
                    }
                )
                for key, value in schema.objects("properties", optional=True).items()
            }
        if "properties" in schema:
            schema["properties"] = {
                key: lift_child(value, name + "".join(part.title() for part in key.split("_")))
                for key, value in schema.objects("properties").items()
            }
        if "items" in schema:
            schema["items"] = lift_child(schema.object("items"), name + "Item")
        for variant in ("oneOf", "anyOf"):
            if variant in schema:
                schema[variant] = [
                    lift_child(v, name + str(i)) for i, v in enumerate(schema.object_array(variant))
                ]
        return schema

    inline_names = {
        "LaunchRequestInputs": "RunInputs",
        "RunInputsFilesItem": "InputFile",
        "UploadPut": "UploadGrant",
        "RunTaskPageItemsItem": "RunTask",
        "OwnerResourceRequestSelection": "PoolSelection",
    }

    def lift_child(schema: JSONObject, name: str) -> JSONObject:
        name = inline_names.get(name, name)
        types = schema.get("type", [])
        if "properties" in schema and (
            types == "object" or (isinstance(types, list) and "object" in types)
        ):
            nullable = isinstance(types, list) and "null" in types
            schemas[name] = lift(JSONObject({**schema, "type": "object"}), name)
            ref = JSONObject({"$ref": "#/components/schemas/" + name})
            return JSONObject({"anyOf": [ref, {"type": "null"}]}) if nullable else ref
        return lift(schema, name)

    for name in list(schemas):
        schemas[name] = lift(schemas[name], name)
    schemas = select_sdk_schemas(schemas, sdk_model_roots(ops))
    enums = {
        "RunState": schemas["RunSnapshot"].object("properties").object("state").strings("examples")
    }

    def annotation(s: JSONObject, name: str = "") -> str:
        if "$ref" in s:
            return s.text("$ref").split("/")[-1]
        variants = s.object_array("anyOf", optional=True) or s.object_array("oneOf", optional=True)
        if variants:
            result = " | ".join(dict.fromkeys(annotation(v) for v in variants))
            event_types = {
                v.text("x-synth-event-kind"): annotation(v)
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
            if name and all(isinstance(value, str) for value in s.array("enum")):
                enums[name] = s.strings("enum")
                return "Annotated[" + name + ", BeforeValidator(enum_value(" + name + "))]"
            return "Literal[" + ", ".join(repr(v) for v in s.array("enum")) + "]"
        typ = s.get("type")
        if isinstance(typ, list):
            return " | ".join(annotation(JSONObject({**s, "type": t})) for t in typ)
        if typ == "array":
            base = "list[" + annotation(s.object("items", optional=True)) + "]"
            meta = []
            bounds = [
                name + "=" + str(s[key])
                for key, name in (("minItems", "min_length"), ("maxItems", "max_length"))
                if key in s
            ]
            if bounds:
                meta.append("Field(" + ", ".join(bounds) + ")")
            if s.flag("uniqueItems"):
                meta.append("AfterValidator(unique_items)")
            return "Annotated[" + base + ", " + ", ".join(meta) + "]" if meta else base
        if typ == "object":
            additional = s.get("additionalProperties")
            return (
                "dict[str, "
                + (
                    annotation(JSONObject.require(additional))
                    if isinstance(additional, dict)
                    else "JsonValue"
                )
                + "]"
            )
        base = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "null": "None",
        }.get(typ if isinstance(typ, str) else None, "JsonValue")
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
            if dest is not None:
                args.append(dest + "=" + repr(v))
        metadata = ["Field(" + ", ".join(args) + ")"] if args else []
        if s.flag("x-synth-budget"):
            metadata.append("AfterValidator(budget_amount)")
        if s.flag("x-synth-no-nul"):
            metadata.append("AfterValidator(no_nul)")
        if s.flag("x-synth-message-text"):
            metadata.append("AfterValidator(message_text)")
        if "x-synth-max-bytes" in s:
            metadata.append("AfterValidator(utf8_max_bytes(" + str(s["x-synth-max-bytes"]) + "))")
        return "Annotated[" + base + ", " + ", ".join(metadata) + "]" if metadata else base

    ordered = []
    remaining = set(schemas)
    while remaining:
        ready = sorted(n for n in remaining if not (schema_references(schemas[n]) & remaining))
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
            for field, value in s.objects("properties").items():
                name = (
                    field + "_"
                    if keyword.iskeyword(field)
                    or field in ("model_config", "schema", "json", "copy")
                    else field
                )
                typ = annotation(value, n + "".join(part.title() for part in field.split("_")))
                required = field in s.strings("required", optional=True)
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
    catalog = JSONObject.parse(json.loads((ROOT / "contracts/mcp-tools.json").read_text()))

    def expand(value: JsonValue) -> JsonValue:
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, dict):
            if "$ref" in value:
                value = {
                    **resolve(JSONObject.require(value)),
                    **{k: v for k, v in value.items() if k != "$ref"},
                }
            return {k: expand(v) for k, v in value.items()}
        return value

    expected_tools: list[JSONObject] = []
    for _path, item in doc.objects("paths").items():
        for operation in item.values():
            if (
                not isinstance(operation, JSONObject)
                or "x-synth-mcp-name" not in operation
                or operation.get("x-synth-status") != "implemented"
            ):
                continue
            params = [resolve(p) for p in operation.object_array("parameters", optional=True)]
            properties: dict[str, JsonValue] = {
                p.text("name"): expand(p["schema"]) for p in params if p["in"] in ("path", "query")
            }
            required: list[JsonValue] = [
                p.text("name")
                for p in params
                if p["in"] in ("path", "query") and p.flag("required")
            ]
            if "requestBody" in operation:
                body = operation.object("requestBody")
                properties["body"] = expand(
                    body.object("content").object("application/json")["schema"]
                )
                if body.flag("required"):
                    required.append("body")
            if any(p["name"] == "Idempotency-Key" for p in params):
                properties["idempotency_key"] = {"type": "string", "minLength": 1, "maxLength": 128}
                required.append("idempotency_key")
            expected_tools.append(
                JSONObject(
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
            )
    # The contract is the authority. The catalog is a generated consumer too.
    tool_values: list[JsonValue] = [tool for tool in expected_tools]
    catalog["tools"] = tool_values
    tools = (
        "# Generated by scripts/gen_contract.py; do not edit.\n"
        "from typing import TypedDict\nfrom pydantic import JsonValue\n"
        "class ToolSchema(TypedDict):\n    type: str\n    properties: dict[str, JsonValue]\n    required: list[str]\n    additionalProperties: bool\n"
        "class ToolDefinition(TypedDict):\n    name: str\n    description: str\n    scope: str\n    operationId: str\n    inputSchema: ToolSchema\n"
        "TOOLS: list[ToolDefinition] = "
        + pprint.pformat(expected_tools, sort_dicts=True, width=100)
        + "\n"
    )
    docs = (
        "# MCP tools\n\nGenerated from the pinned public OpenAPI and hosted catalog.\n\n| Tool | Operation | Scope |\n| --- | --- | --- |\n"
        + "".join(
            "| " + t.text("name") + " | " + t.text("operationId") + " | " + t.text("scope") + " |\n"
            for t in expected_tools
        )
    )
    registry = doc.object_array("x-synth-scope-registry")
    policy_source = "# Generated by scripts/gen_contract.py; do not edit.\n"
    policy_source += (
        "SDK_OAUTH_SCOPES = "
        + repr(tuple(s.text("name") for s in registry if s.flag("sdk")))
        + "\n"
    )
    policy_source += (
        "MESSAGE_MAX_BYTES = "
        + str(schemas["LaunchRequest"].object("properties").object("message")["x-synth-max-bytes"])
        + "\n"
    )
    outputs: dict[Path, str] = {
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


def main() -> None:
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
