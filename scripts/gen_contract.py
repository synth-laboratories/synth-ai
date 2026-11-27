#!/usr/bin/env python3
"""Generate public models, operations and MCP tools from the pinned OpenAPI."""

from __future__ import annotations

import argparse
import ast
import hashlib
import keyword
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    import builtins
    from collections.abc import Mapping

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "contracts/synth-v1.openapi.yaml"
# Server-managed catalog enums decode known members into StrEnum and accept unknown
# future strings without ResponseContractError (SYN-6038).
OPEN_CATALOG_ENUMS = frozenset(
    {
        "CheckoutRequestPackId",
        "OptimizerPinTopology",
        "PromoTrialGrantPlanId",
        "ResolvedOptimizerTopology",
        "SandboxProfileGpuKind",
        "SubscriptionChangeRequestPlanId",
        "TopologyCatalogEntryTopology",
    }
)


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


def field_name(name: str) -> str:
    if not name.isidentifier():
        raise ValueError(f"unsupported schema field name: {name}")
    return (
        name + "_"
        if keyword.iskeyword(name) or name in {"model_config", "schema", "json", "copy"}
        else name
    )


def named_string_enums(schemas: Mapping[str, JSONObject]) -> dict[str, list[str]]:
    """Preserve canonical enum names instead of reducing them to literals."""
    return {
        name: schema.strings("enum")
        for name, schema in schemas.items()
        if schema.get("type") == "string" and "enum" in schema
    }


def constraint_expression(schema: JSONObject, value: str) -> str:
    """Compile supported refinements against already parsed, typed model fields."""
    supported = {
        "$ref",
        "allOf",
        "if",
        "then",
        "else",
        "properties",
        "required",
        "const",
        "enum",
        "not",
        "pattern",
        "type",
        "minItems",
        "contains",
        "description",
    }
    if unknown := schema.keys() - supported:
        raise ValueError(f"unsupported schema constraint: {sorted(unknown)}")
    terms: list[str] = []
    if "$ref" in schema:
        reference = schema.text("$ref").split("/")[-1]
        if not reference.isidentifier():
            raise ValueError("invalid constraint reference")
        terms.append(f"isinstance({value}, {reference})")
    for part in schema.object_array("allOf", optional=True):
        terms.append(constraint_expression(part, value))
    if "not" in schema:
        terms.append(f"not {constraint_expression(schema.object('not'), value)}")
    if "if" in schema:
        condition = constraint_expression(schema.object("if"), value)
        accepted = constraint_expression(schema.object("then", optional=True), value)
        rejected = constraint_expression(schema.object("else", optional=True), value)
        terms.append(f"({accepted} if {condition} else {rejected})")
    elif "then" in schema or "else" in schema:
        raise ValueError("conditional branch without if")
    for name in schema.strings("required", optional=True):
        terms.append(f"{name!r} in {value}.model_fields_set")
    for name, child in schema.objects("properties", optional=True).items():
        expression = constraint_expression(child, f"{value}.{field_name(name)}")
        terms.append(f"({name!r} not in {value}.model_fields_set or {expression})")
    if "const" in schema:
        const = schema["const"]
        operator = "is" if const is None or isinstance(const, bool) else "=="
        terms.append(f"{value} {operator} {const!r}")
    if "enum" in schema:
        terms.append(f"{value} in {tuple(schema.strings('enum'))!r}")
    if "pattern" in schema:
        terms.append(f"re.search({schema.text('pattern')!r}, {value}) is not None")
    if "type" in schema:
        match schema.text("type"):
            case "null":
                terms.append(f"{value} is None")
            case "number":
                terms.append(f"isinstance({value}, (int, float)) and not isinstance({value}, bool)")
            case other:
                raise ValueError(f"unsupported constraint type: {other}")
    if "minItems" in schema:
        minimum = schema["minItems"]
        if type(minimum) is not int or minimum < 0:
            raise ValueError("invalid minItems constraint")
        terms.append(f"len({value}) >= {minimum}")
    if "contains" in schema:
        expression = constraint_expression(schema.object("contains"), "item")
        terms.append(f"any({expression} for item in {value})")
    return "(" + " and ".join(terms) + ")" if terms else "True"


def model_constraints(schema: JSONObject) -> list[str]:
    constraints = [
        constraint_expression(part, "self") for part in schema.object_array("allOf", optional=True)
    ]
    for name, child in schema.objects("properties", optional=True).items():
        value = "self." + field_name(name)
        if "allOf" in child:
            constraints.append(constraint_expression(child, value))
        else:
            for variant in child.object_array("oneOf", optional=True):
                if "allOf" in variant:
                    variants = child.object_array("oneOf")
                    if len(variants) != 2 or not any(
                        part.get("type") == "null" for part in variants
                    ):
                        raise ValueError("unsupported refined union")
                    constraints.append(
                        f"({value} is None or {constraint_expression(variant, value)})"
                    )
    return constraints


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
            continue
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
        owners["RunState"] = "RunSnapshot"
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


class _AnnotationFn(Protocol):
    def __call__(self, s: JSONObject, name: str = "") -> str: ...


def _enum_annotation(name: str) -> str:
    if name in OPEN_CATALOG_ENUMS:
        return "Annotated[" + name + " | str, BeforeValidator(open_enum_value(" + name + "))]"
    return "Annotated[" + name + ", BeforeValidator(enum_value(" + name + "))]"


def _annotation_formatter(enums: dict[str, list[str]]) -> _AnnotationFn:
    def annotation(s: JSONObject, name: str = "") -> str:
        if "$ref" in s:
            reference = s.text("$ref").split("/")[-1]
            if reference in enums:
                return _enum_annotation(reference)
            return reference
        refinements = s.object_array("allOf", optional=True)
        if refinements and "$ref" in refinements[0]:
            return annotation(refinements[0])
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
            literal_names = {"OwnerResourceRequestPlugin"}
            if (
                name
                and name not in literal_names
                and all(isinstance(value, str) for value in s.array("enum"))
            ):
                enums[name] = s.strings("enum")
                return _enum_annotation(name)
            return "Literal[" + ", ".join(repr(v) for v in s.array("enum")) + "]"
        typ = s.get("type")
        if isinstance(typ, list):
            return " | ".join(annotation(JSONObject({**s, "type": t})) for t in typ)
        if typ == "array":
            base = "list[" + annotation(s.object("items", optional=True)) + "]"
            meta = []
            bounds = [
                n + "=" + str(s[k])
                for k, n in (("minItems", "min_length"), ("maxItems", "max_length"))
                if k in s
            ]
            if bounds:
                meta.append("Field(" + ", ".join(bounds) + ")")
            if s.flag("uniqueItems"):
                meta.append("AfterValidator(unique_items)")
            return "Annotated[" + base + ", " + ", ".join(meta) + "]" if meta else base
        if typ == "object":
            additional = s.get("additionalProperties")
            value_type = (
                annotation(JSONObject.require(additional))
                if isinstance(additional, dict)
                else "JsonValue"
            )
            return "dict[str, " + value_type + "]"
        base = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "null": "None",
        }.get(typ if isinstance(typ, str) else None, "JsonValue")
        if typ == "null":
            return "None"
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

    return annotation


def _render_models(
    schemas: dict[str, JSONObject],
    ops: Mapping[str, JSONObject],
    enums: dict[str, list[str]],
) -> str:
    annotation = _annotation_formatter(enums)
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
        "from typing import Annotated, Literal, Self, TypeAlias",
        "import re",
        "from enum import StrEnum",
        "from pydantic import AfterValidator, BeforeValidator, BaseModel, ConfigDict, Field, JsonValue, model_validator",
        "from synth_ai._wire import WireModel",
        "from synth_ai._protocol import CompletionMilestone",
        "from synth_ai._validation import budget_amount, enum_value, event_payload, message_text, no_nul, open_enum_value, strict_int, unique_items, utf8_max_bytes",
        "",
    ]
    preamble_length = len(lines)
    response_names = {operation.get("response_model") for operation in ops.values()}
    modelnames = []
    for n in ordered:
        s = schemas[n]
        if n in enums:
            continue
        if s.get("type") == "object" and "properties" in s:
            modelnames.append(n)
            lines += [
                "class " + n + "(WireModel):",
                "    model_config = ConfigDict(extra="
                + repr("forbid" if s.get("additionalProperties") is False else "allow")
                + ", populate_by_name=True, serialize_by_alias=True, strict=True, frozen=True, allow_inf_nan=False)",
            ]
            for field, value in s.objects("properties").items():
                name = field_name(field)
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
            constraints = model_constraints(s)
            if constraints:
                lines += [
                    "",
                    '    @model_validator(mode="after")',
                    "    def validate_schema_constraints(self) -> Self:",
                ]
                for index, expression in enumerate(constraints):
                    lines += [
                        f"        if not {expression}:",
                        f"            raise ValueError({f'{n}: schema constraint {index + 1} failed'!r})",
                    ]
                lines += ["        return self"]
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
                    "        if milestone is CompletionMilestone.PAUSED:",
                    '            return self.state == "paused"',
                    "        if milestone is CompletionMilestone.FINISHED:",
                    "            return self.finished",
                    "        if milestone is CompletionMilestone.EXECUTION_CLOSED:",
                    "            return self.execution_closed",
                    "        return self.result_available",
                ]
            lines += [""]
        elif n in response_names:
            modelnames.append(n)
            lines += ["class " + n + "(RootModel[" + annotation(s) + "]):", "    pass", ""]
        else:
            lines += [n + ": TypeAlias = " + annotation(s), ""]
    aliased_requests = sorted(
        {op.text("request_model") for op in ops.values() if op["request_model"] is not None}
        - set(modelnames)
    )
    if aliased_requests or any("(RootModel[" in line for line in lines):
        target = "from pydantic import AfterValidator, BeforeValidator, BaseModel, ConfigDict, Field, JsonValue, model_validator"
        lines[lines.index(target)] += ", RootModel"
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
        + ["    " + repr(n) + ": RootModel[" + n + "]," for n in aliased_requests]
        + ["}", ""]
    )
    return "\n".join(lines)


def generate() -> dict[Path, str | bytes]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts._gen_contract_emit import finalize_contract_outputs, prepare_contract

    doc, schemas, ops = prepare_contract(ROOT, SPEC, JSONObject)
    schemas = select_sdk_schemas(schemas, sdk_model_roots(ops))
    enums = {
        **named_string_enums(schemas),
        "RunState": schemas["RunSnapshot"].object("properties").object("state").strings("examples"),
    }
    models_py = _render_models(schemas, ops, enums)
    return finalize_contract_outputs(ROOT, doc, schemas, ops, models_py, JSONObject)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, source in generate().items():
        content = source.encode() if isinstance(source, str) else source
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit("generated drift: " + str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print("contract sha256 " + hashlib.sha256(SPEC.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
