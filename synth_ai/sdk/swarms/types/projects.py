"""Default customer Swarms project provisioning request (SYN-4509).

A customer creates a launchable project from a closed template without operator
authoring. The server authors the configuration; the request carries only the
customer's choices.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from synth_ai.types.base import Model


class ProjectTemplate(StrEnum):
    """Closed set of server-authored project templates."""

    REPORT_NATIVE = "report_native"


class ProjectCreateRequest(Model):
    name: str = Field(min_length=1, max_length=256)
    template: ProjectTemplate
    objective: str = Field(min_length=1, max_length=8192)

    @model_validator(mode="after")
    def validate_intent(self) -> Self:
        if (
            not self.name.strip()
            or not self.objective.strip()
            or "\x00" in self.name
            or "\x00" in self.objective
            or len(self.objective.encode("utf-8")) > 8192
        ):
            raise ValueError("name and objective must be nonblank and bounded")
        return self
