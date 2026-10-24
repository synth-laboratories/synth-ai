"""Default customer Swarms project provisioning request (SYN-4509).

A customer creates a launchable project from a closed template without operator
authoring. The server authors the configuration; the request carries only the
customer's choices.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Self

from pydantic import Field, model_validator

from synth_ai.types.base import Model


class ProjectTemplate(StrEnum):
    """Closed set of server-authored project templates."""

    REPORT_NATIVE = "report_native"
    CODE_NATIVE = "code_native"


CodeDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
CodePath = Annotated[str, Field(min_length=1, max_length=1024)]


class CodeExecutorSource(Model):
    """Source identity checked by the registered verifier authority."""

    reference: str = Field(min_length=1, max_length=512)
    sha256: CodeDigest


class CodeValidation(Model):
    """Accepted code baseline and registered verifier; never an execution verdict.

    See notes/specifications/tanha/current/systems/platform/orchestra_port_authority.md.
    Real measurements and the host grade remain independently owned.
    """

    executor_source: CodeExecutorSource
    image_release_id: str = Field(pattern=r"^imgrel_[0-9a-f]{64}$")
    native_image_id: CodeDigest
    baseline_digest: CodeDigest
    allowed_files: tuple[CodePath, ...] = Field(min_length=1, max_length=100)
    baseline_file_sha256: dict[CodePath, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]] = Field(
        min_length=1, max_length=100
    )
    required_artifacts: tuple[CodePath, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def exact_submission_paths(self) -> Self:
        paths = (*self.allowed_files, *self.required_artifacts)
        if any(
            PurePosixPath(p).is_absolute()
            or ".." in PurePosixPath(p).parts
            or str(PurePosixPath(p)) != p
            or "\x00" in p
            for p in paths
        ):
            raise ValueError("code_validation_paths_invalid")
        if (
            len(set(self.allowed_files)) != len(self.allowed_files)
            or set(self.baseline_file_sha256) != set(self.allowed_files)
            or len(set(self.required_artifacts)) != len(self.required_artifacts)
        ):
            raise ValueError("code_validation_baseline_files_mismatch")
        for path in self.allowed_files:
            suffix = "submission/" + path
            if sum(p == suffix or p.endswith("/" + suffix) for p in self.required_artifacts) != 1:
                raise ValueError("code_validation_submission_path_unbound")
        return self


class ProjectCreateRequest(Model):
    name: str = Field(min_length=1, max_length=256)
    template: ProjectTemplate
    objective: str = Field(min_length=1, max_length=8192)
    code_validation: CodeValidation | None = None

    @model_validator(mode="after")
    def validate_intent(self) -> Self:
        if (self.template is ProjectTemplate.CODE_NATIVE) != (self.code_validation is not None):
            raise ValueError("code_native_requires_code_validation_only")
        if (
            not self.name.strip()
            or not self.objective.strip()
            or "\x00" in self.name
            or "\x00" in self.objective
            or len(self.objective.encode("utf-8")) > 8192
        ):
            raise ValueError("name and objective must be nonblank and bounded")
        return self
