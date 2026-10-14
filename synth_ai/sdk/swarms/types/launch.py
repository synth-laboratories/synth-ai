"""Four explicit launch option groups. See backend SDK swarms_overrides.md."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from synth_ai.types.base import Digest, Identifier, Model, ResponseModel
from synth_ai.types.money import Money


class Limits(Model):
    maximum_charge: Money
    wall_time_seconds: StrictInt = Field(ge=1, le=86400)
    maximum_tasks: StrictInt = Field(ge=1, le=1000)
    maximum_parallel_tasks: StrictInt = Field(ge=1, le=1000)

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.maximum_charge.coefficient < 0:
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

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        identities = [(file.file_id, file.revision_id) for file in self.files]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate file revision selection")
        return self


class Providers(Model):
    default: Literal["openai"] = "openai"


class Models(Model):
    default: Literal["gpt-6-luna"] = "gpt-6-luna"


ResolvedModelId = Annotated[str, Field(strict=True, min_length=1, max_length=256)]


class ResolvedModels(Model):
    """Read-only backend selection; not a caller override allowlist."""

    default: ResolvedModelId


class ResolvedProviders(Model):
    """Read-only backend routing; changing it requires an admitted override."""

    default: Identifier


class LaunchRequest(Model):
    objective: str = Field(min_length=1, max_length=8192)
    project_id: Identifier
    limits: Limits
    inputs: Inputs | None = None
    providers: Providers | None = None
    models: Models | None = None

    @model_validator(mode="after")
    def validate_intent(self) -> Self:
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
    project_id: Identifier
    name: str = Field(min_length=1, max_length=256)
    launch_enabled: bool
