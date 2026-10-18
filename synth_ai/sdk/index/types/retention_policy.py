"""Organization retention policy. See backend notes/specifications/sdk/index_retention.md."""

from __future__ import annotations

from typing import Self

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    StrictInt,
    model_serializer,
    model_validator,
)

from synth_ai.types.base import Model, ResponseModel

from .retention import RetentionMode


class RetentionPolicy(ResponseModel):
    """The organization's default and permitted Index retention modes."""

    default_mode: RetentionMode
    allowed_modes: tuple[RetentionMode, ...] = Field(min_length=1, max_length=3)
    enforced: bool
    revision: StrictInt = Field(ge=0)


class RetentionPolicyUpdate(Model):
    """Org-admin intent; absent optional fields are omitted from the wire."""

    default_mode: RetentionMode
    allowed_modes: tuple[RetentionMode, ...] = Field(min_length=1, max_length=3)
    enforced: bool | None = None
    expected_revision: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_modes(self) -> Self:
        if len(set(self.allowed_modes)) != len(self.allowed_modes):
            raise ValueError("duplicate retention modes")
        if self.default_mode not in self.allowed_modes:
            raise ValueError("retention default must be an allowed mode")
        return self

    @model_serializer(mode="wrap")
    def omit_absent(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        return {key: value for key, value in data.items() if value is not None}
