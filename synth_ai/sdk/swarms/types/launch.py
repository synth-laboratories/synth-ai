"""Four explicit launch option groups. See backend SDK swarms_overrides.md."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_serializer, model_validator

from synth_ai.sdk.core.artifact_types import Labels
from synth_ai.types.base import (
    Digest,
    Identifier,
    Model,
    ResponseBase,
    ResponseIdentifier,
    ResponseModel,
)
from synth_ai.types.money import Money
from synth_ai.types.refs import Ref

from .objectives import Guidance
from .resources import ResourceRequest


class Limits(Model):
    # Explicit null requests operator-funded local execution; backend authority
    # decides whether the selected target and project permit it.
    maximum_charge: Money | None
    wall_time_seconds: StrictInt = Field(ge=1, le=86400)
    maximum_tasks: StrictInt = Field(ge=1, le=1000)
    maximum_parallel_tasks: StrictInt = Field(ge=1, le=1000)

    @model_serializer(mode="wrap")
    def serialize_limits(self, handler):
        body = handler(self)
        if self.maximum_charge is None:
            body["maximum_charge"] = None
        return body

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.maximum_charge is not None and self.maximum_charge.coefficient < 0:
            raise ValueError("maximum charge must be nonnegative")
        if self.maximum_parallel_tasks > self.maximum_tasks:
            raise ValueError("parallel tasks cannot exceed total tasks")
        return self


class FileRef(Model):
    file_id: Identifier
    revision_id: Identifier
    project_id: Identifier
    digest_sha256: Digest


class Inputs(Model):
    files: tuple[FileRef, ...] = Field(default=(), max_length=32)
    artifacts: tuple[Ref, ...] = Field(default=(), max_length=32)
    derived_from: Ref | None = None

    @model_serializer(mode="wrap")
    def omit_empty_artifacts(self, handler):
        result = handler(self)
        if not self.artifacts:
            result.pop("artifacts", None)
        if self.derived_from is None:
            result.pop("derived_from", None)
        return result

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        identities = [(file.file_id, file.revision_id) for file in self.files]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate file revision selection")
        selected = set()
        for ref in self.artifacts:
            if ref.authority != "artifacts" or ref.kind != "artifact":
                raise ValueError("Inputs must select exact artifact revisions")
            Ref.artifact(ref.record_id, ref.revision, ref.digest_sha256)
            key = (ref.record_id, ref.revision)
            if key in selected:
                raise ValueError("Duplicate artifact input")
            selected.add(key)
        if self.derived_from is not None and self.derived_from not in self.artifacts:
            raise ValueError("Derived input must be one of the exact admitted artifact inputs")
        if len(self.files) + len(self.artifacts) > 32:
            raise ValueError("Inputs exceed the total selection bound")
        return self


class RepoSelector(Model):
    """Launch on a project repository at an exact base commit (SYN-4885).

    ``base`` is the git-server commit Ref (core.v1: authority ``git``, kind
    ``commit``, record_id = ``repo_id``, revision = full SHA-1, digest =
    git-server commit manifest digest). The backend verifies the repository
    belongs to the launch project and the commit exists with that digest, then
    freezes it into the run configuration and seeds the workspace from it.
    """

    repo_id: Identifier
    base: Ref

    @model_validator(mode="after")
    def validate_base(self) -> Self:
        if self.base.authority != "git" or self.base.kind != "commit":
            raise ValueError("repo.base must be an exact git commit Ref")
        if self.base.record_id != self.repo_id:
            raise ValueError("repo.base must belong to repo.repo_id")
        Ref.commit(self.base.record_id, self.base.revision, self.base.digest_sha256)
        return self


class Providers(Model):
    default: Literal["openai"] = "openai"


class Models(Model):
    default: Literal["gpt-6-luna"] = "gpt-6-luna"


ResolvedModelId = Annotated[str, Field(strict=True, min_length=1, max_length=256)]


class ResolvedModels(ResponseBase):
    """Read-only backend selection; not a caller override allowlist."""

    default: ResolvedModelId


class ResolvedProviders(ResponseBase):
    """Read-only backend routing; changing it requires an admitted override."""

    default: ResponseIdentifier


class LaunchRequest(Model):
    message: str | None = Field(default=None, min_length=1, max_length=8192)
    legacy_objective: str | None = Field(
        default=None, alias="objective", min_length=1, max_length=8192
    )
    project_id: Identifier
    limits: Limits
    inputs: Inputs | None = None
    providers: Providers | None = None
    models: Models | None = None
    labels: Labels | None = None
    repo: RepoSelector | None = None
    guidance: tuple[Guidance, ...] = Field(default=(), max_length=32)
    resources: tuple[ResourceRequest, ...] = Field(default=(), max_length=3)

    @property
    def objective(self) -> str:
        return self.message if self.message is not None else self.legacy_objective or ""

    @model_serializer(mode="wrap")
    def omit_absent_labels(self, handler):
        result = handler(self)
        for name in ("labels", "inputs", "providers", "models", "repo"):
            if getattr(self, name) is None:
                result.pop(name, None)
        if not self.guidance:
            result.pop("guidance", None)
        if not self.resources:
            result.pop("resources", None)
        result.pop("message", None)
        result.pop("legacy_objective", None)
        result.pop("objective", None)
        return {("message" if self.message is not None else "objective"): self.objective, **result}

    @model_validator(mode="after")
    def validate_intent(self) -> Self:
        if (self.message is None) == (self.legacy_objective is None):
            raise ValueError("Select message or the deprecated objective alias")
        if len({item.plugin for item in self.resources}) != len(self.resources):
            raise ValueError("Resource plugins may be granted once")
        if len({item.id for item in self.guidance}) != len(self.guidance):
            raise ValueError("Guidance ids must be unique")
        if (
            not self.objective.strip()
            or "\x00" in self.objective
            or len(self.objective.encode("utf-8")) > 8192
        ):
            raise ValueError("objective must be nonblank and fit 8192 UTF-8 bytes")
        if self.inputs is not None and any(
            file.project_id != self.project_id for file in self.inputs.files
        ):
            raise ValueError("selected input files must belong to the selected project")
        return self


class Project(ResponseModel):
    project_id: ResponseIdentifier
    name: str = Field(min_length=1, max_length=256)
    launch_enabled: bool
