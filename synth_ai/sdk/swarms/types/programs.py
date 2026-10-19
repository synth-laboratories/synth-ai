"""Program inspection models. See backend modular Orchestra program contract."""

from typing import Any

from pydantic import Field

from synth_ai.types.base import Identifier, ResponseModel


class ProgramRunView(ResponseModel):
    run_id: Identifier
    project_id: Identifier
    program: dict[str, Any]
    usage: dict[str, int] = Field(default_factory=dict)
    resources: dict[str, Any] | None = None
    resource_extension_ready: bool = False
    sequence: int = Field(ge=0)
    state_digest: str | None
    open: bool
    status: str
    terminal: dict[str, Any] | None
    projection: dict[str, Any]
    close_holds: tuple[str, ...] = ()
    last_refusal: dict[str, Any] | None = None
