"""Response-only views of values also used in strict requests.

Do not put a model-level before/wrap validator on strict wire models: pydantic
then sees Python lists/dates instead of JSON inputs and refuses valid tuples.
Clean unknown model keys once, retaining JSON validation and the original class.
Mappings keep their data keys. No known scalar or semantic validator is skipped.
"""

from __future__ import annotations

import json
from copy import deepcopy
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel


def _alternatives(annotations: tuple[Any, ...]) -> tuple[Any, ...]:
    result = []
    for annotation in annotations:
        origin = get_origin(annotation)
        if origin is Annotated:
            result.extend(_alternatives((get_args(annotation)[0],)))
        elif origin in (Union, UnionType):
            result.extend(_alternatives(get_args(annotation)))
        else:
            result.append(annotation)
    return tuple(result)


def _clean(value: Any, annotations: tuple[Any, ...]) -> bool:
    annotations = _alternatives(annotations)
    if isinstance(value, dict):
        mappings = [arg for arg in annotations if get_origin(arg) is dict]
        if mappings:
            # Preserve data keys; only nested declared model values are traversed.
            values = tuple(get_args(arg)[1] for arg in mappings if len(get_args(arg)) == 2)
            changed = False
            for item in value.values():
                changed = _clean(item, values) or changed
            return changed
        models = tuple(
            arg for arg in annotations if isinstance(arg, type) and issubclass(arg, BaseModel)
        )
        matches = tuple(
            model
            for model in models
            if any(
                name in value
                and get_origin(field.annotation) is Literal
                and value[name] in get_args(field.annotation)
                for name, field in model.model_fields.items()
            )
        )
        models = matches or models
        if not models:
            return False
        fields = {}
        for model in models:
            for name, field in model.model_fields.items():
                for key in (name, field.alias, field.validation_alias):
                    if isinstance(key, str):
                        fields.setdefault(key, []).append(field.annotation)
        changed = False
        strip_unknown = all(model.model_config.get("extra") == "forbid" for model in models)
        for key in tuple(value):
            if key not in fields:
                if strip_unknown:
                    del value[key]
                    changed = True
            else:
                changed = _clean(value[key], tuple(fields[key])) or changed
        return changed
    if isinstance(value, (list, tuple)):
        elements = tuple(
            get_args(arg)[0]
            for arg in annotations
            if get_origin(arg) in (list, tuple) and get_args(arg)
        )
        if not elements:
            return False
        changed = False
        for item in value:
            changed = _clean(item, elements) or changed
        return changed
    return False


def response_json(
    value: str | bytes | bytearray, model: type[BaseModel]
) -> str | bytes | bytearray:
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value  # Let pydantic render the typed JSON validation error.
    if not _clean(parsed, (model,)):
        return value
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def response_value(value: Any, model: type[BaseModel]) -> Any:
    if not isinstance(value, dict):
        return value
    # Never mutate a caller's model input or a retained wire fixture.
    result = deepcopy(value)
    _clean(result, (model,))
    return result


def response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Export tolerant response views while preserving strict shared request models."""
    from .base import IDENTIFIER_PATTERN, ResponseBase

    schema = model.model_json_schema()
    if not issubclass(model, ResponseBase):
        return schema
    identifier_patterns = {
        IDENTIFIER_PATTERN,
        "^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$",
    }

    def relax(value):
        if isinstance(value, list):
            for child in value:
                relax(child)
        elif isinstance(value, dict):
            if value.get("type") == "object" and "properties" in value:
                value["additionalProperties"] = True
            if value.get("type") == "string" and value.get("pattern") in identifier_patterns:
                value.pop("pattern")
                value.update(minLength=1, maxLength=512)
            if (
                value.get("pattern")
                == "^(?:[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}|intern-exec:[0-9a-f]{64})$"
            ):
                value.update(
                    pattern=r"^(?:[^:]{1,512}|intern-exec:[0-9a-f]{64})$",
                    minLength=1,
                    maxLength=512,
                )
            for child in value.values():
                relax(child)

    relax(schema)
    return schema
