"""Model/effort profile overrides on a Swarms launch (SYN-4848). See docs/swarms.md.

A launch may fix the model and/or reasoning effort of the program's declared
roles. Per field, precedence is template default (the admitted authored
profile) -> launch-wide choice -> named-role choice. Only roles and
model/effort pairs the selected program declares are accepted; the server
refuses anything else with typed ``profile_override_*`` codes before any paid
work and never substitutes. Omitting ``profile_overrides`` keeps the launch
request bytes unchanged.

The server resolves the choice into immutable derived profiles (only the model
and effort differ from the admitted base, so tools, limits and spend never
widen) and echoes the resolution, with provenance and digests, on preflight and
on the run snapshot.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, model_serializer, model_validator

from synth_ai.types.base import Model, ResponseBase, ResponseIdentifier

Effort = Literal["none", "low", "medium", "high", "xhigh"]
OverrideRole = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$"),
]
CanonicalModel = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$"),
]
ProfileDigest = Annotated[str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")]
OverrideSource = Literal["template_default", "launch", "role"]


class ModelEffortChoice(Model):
    """A fixed model, a fixed effort, or both."""

    canonical_model: CanonicalModel | None = None
    effort: Effort | None = None

    @model_validator(mode="after")
    def selects_something(self) -> Self:
        if self.canonical_model is None and self.effort is None:
            raise ValueError("A model/effort choice selects a model, an effort or both")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_fields(self, handler) -> dict[str, Any]:
        result = handler(self)
        for name in ("canonical_model", "effort"):
            if getattr(self, name) is None:
                result.pop(name, None)
        return result


class ProfileOverrides(Model):
    """Launch-wide and named-role choices; at least one is required.

    ``launch`` reaches declared coordinator roles only; the evaluated system
    changes only through a named role choice.
    """

    launch: ModelEffortChoice | None = None
    roles: dict[OverrideRole, ModelEffortChoice] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def selects_something(self) -> Self:
        if self.launch is None and not self.roles:
            raise ValueError("Profile overrides need a launch-wide or a named-role choice")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_parts(self, handler) -> dict[str, Any]:
        result = handler(self)
        if self.launch is None:
            result.pop("launch", None)
        if not self.roles:
            result.pop("roles", None)
        return result


class ResolvedProfileActor(ResponseBase):
    """One actor's effective model/effort, its provenance and immutable pins."""

    actor_id: ResponseIdentifier
    actor_type: ResponseIdentifier
    base_profile_digest: ProfileDigest
    profile_digest: ProfileDigest
    canonical_model: ResponseIdentifier
    effort: Effort
    model_source: OverrideSource
    effort_source: OverrideSource
    evaluated: bool


class ProfileOverrideResolution(ResponseBase):
    """The server's single pre-execution resolution of a launch's overrides."""

    schema_version: Literal["synth.profile-override-resolution.v1"]
    declaration_digest: ProfileDigest
    request_digest: ProfileDigest
    actors: tuple[ResolvedProfileActor, ...] = Field(max_length=256)
    evaluated_system_changed: bool
    digest: ProfileDigest


class ProfileOverrideReadback(ResponseBase):
    """The frozen request and, once resolved, its resolution.

    Preflight resolves against the project's admitted program; a run snapshot
    carries the request frozen at create and the resolution recorded when the
    run was prepared (absent before preparation).
    """

    requested: ProfileOverrides
    resolution: ProfileOverrideResolution | None = None

    def satisfies(self, requested: ProfileOverrides) -> bool:
        return self.requested == requested


__all__ = [
    "Effort",
    "ModelEffortChoice",
    "OverrideSource",
    "ProfileOverrideReadback",
    "ProfileOverrideResolution",
    "ProfileOverrides",
    "ResolvedProfileActor",
]
