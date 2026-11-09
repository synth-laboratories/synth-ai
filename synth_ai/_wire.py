"""Validated JSON extension boundary shared by generated wire records."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, model_validator

JSON_VALUE = TypeAdapter(JsonValue, config=ConfigDict(allow_inf_nan=False))


class WireModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_extensions(self) -> Self:
        # OpenAPI permits extension keys on some responses. They must still be JSON.
        for value in (self.model_extra or {}).values():
            JSON_VALUE.validate_python(value, strict=True)
        return self
