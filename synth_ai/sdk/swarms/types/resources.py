"""Frozen resource grants; credential references never contain secret values."""

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_serializer, model_validator

from synth_ai.types.base import Model, ResponseBase
from synth_ai.types.refs import Ref
from synth_ai.types.unknown import UnknownValue
from synth_ai.types.usage import ResourceUsage

__all__ = ["PoolSelection", "ResourceGrant", "ResourceGrantSet", "ResourceRequest", "ResourceUsage"]

ResourcePluginId = Literal["openrouter.inference", "index.search", "rhodes.pool"]
ResourceRole = Literal["planner", "worker", "reviewer", "run_assessor"]
ResourceDimension = Literal[
    "spend_micros",
    "tokens",
    "tokens_in",
    "tokens_out",
    "calls",
    "index_searches",
    "wall_ms",
    "rollouts",
]
ResourceAmount = Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]
PoolId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class PoolSelection(Model):
    """``rhodes.pool`` grant selection; see rhodes-pool-program-plugin.v1.md."""

    pool_ids: tuple[PoolId, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def distinct(self) -> Self:
        if len(set(self.pool_ids)) != len(self.pool_ids):
            raise ValueError("Pool selections must be distinct")
        return self


class ResourceRequest(Model):
    plugin: ResourcePluginId
    version: StrictInt = Field(default=1, ge=1, le=1)
    roles: tuple[ResourceRole, ...] = Field(min_length=1, max_length=4)
    limits: dict[ResourceDimension, ResourceAmount] = Field(min_length=1, max_length=8)
    role_limits: dict[ResourceRole, dict[ResourceDimension, ResourceAmount]] = Field(
        default_factory=dict, max_length=4
    )
    models: tuple[str, ...] = Field(default=(), max_length=16)
    collections: tuple[str, ...] = Field(default=(), max_length=16)
    credential_ref: Ref | None = None
    selection: PoolSelection | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if len(self.roles) != len(set(self.roles)) or set(self.role_limits) - set(self.roles):
            raise ValueError("Resource roles and per-role limits must be distinct and admitted")
        for limits in self.role_limits.values():
            if set(limits) - set(self.limits) or any(
                value > self.limits[key] for key, value in limits.items()
            ):
                raise ValueError("Role limits may only narrow run limits")
        if self.plugin == "index.search" and (self.models or self.credential_ref is not None):
            raise ValueError("Index grants carry collection scope and no provider credentials")
        if self.plugin == "openrouter.inference" and (not self.models or self.collections):
            raise ValueError("Inference grants require models and no Index collections")
        if (self.plugin == "rhodes.pool") != (self.selection is not None):
            raise ValueError("Only Rhodes pool grants carry, and require, a pool selection")
        if self.plugin == "rhodes.pool" and (
            self.models or self.collections or self.credential_ref is not None
        ):
            raise ValueError("Rhodes pool grants carry pool ids and no credentials")
        if any(not value or len(value) > 256 for value in (*self.models, *self.collections)):
            raise ValueError("Resource selections must be nonempty bounded identifiers")
        return self

    @model_serializer(mode="wrap")
    def omit_absent(self, handler):
        value = handler(self)
        if self.credential_ref is None:
            value.pop("credential_ref", None)
        if self.selection is None:
            value.pop("selection", None)
        for key in ("role_limits", "models", "collections"):
            if not value[key]:
                value.pop(key, None)
        return value


class ResourceGrant(ResponseBase):
    """A frozen grant as observed. Tolerant: unknown fields and future plugins decode.

    Requests (ResourceRequest) stay strict; this observation never constructs one.
    """

    plugin: ResourcePluginId | UnknownValue
    version: StrictInt = Field(default=1, ge=1)
    roles: tuple[ResourceRole | UnknownValue, ...] = Field(default=(), max_length=16)
    limits: dict[str, ResourceAmount] = Field(default_factory=dict)
    role_limits: dict[str, dict[str, ResourceAmount]] = Field(default_factory=dict)
    models: tuple[str, ...] = Field(default=())
    collections: tuple[str, ...] = Field(default=())
    credential_ref: Ref | None = None
    selection: PoolSelection | None = None
    funding_lane: Literal["platform", "user_connected"] | UnknownValue
    credential_refs: tuple[Ref, ...] = Field(default=())
    registry_digest: str = Field(pattern="^[0-9a-f]{64}$")
    grant_id: str = Field(min_length=1, max_length=512)


class ResourceGrantSet(ResponseBase):
    schema_version: str = "orchestra.resource-grants.v1"
    grants: tuple[ResourceGrant, ...] = Field(default=())
