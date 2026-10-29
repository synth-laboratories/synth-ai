"""Program inspection models. See backend modular Orchestra program contract."""

from typing import Annotated, Any, Literal

from pydantic import Field, StrictInt, StringConstraints, model_serializer

from synth_ai.types.base import Model, ResponseBase, ResponseIdentifier, ResponseModel


class ProgramRunView(ResponseModel):
    run_id: ResponseIdentifier
    project_id: ResponseIdentifier
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


# SYN-4849: public program families. A launch may select one exact family
# version Orchestra declares over the project's registered program; there is
# no "latest". ``swarms.general`` owns no research domain; ``swarms.research``
# requires a research binding whose Forge project the same launch grants with
# a ``forge.research`` resource. Orchestra resolves the exact program,
# definition and actor profiles before dispatch and refuses with typed
# ``program_family_*`` codes; the backend refuses the same codes first.
ProgramFamilyId = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$"),
]
ProgramFamilyVersion = Annotated[StrictInt, Field(ge=1, le=2**31 - 1)]
ForgeProjectId = Annotated[
    str,
    StringConstraints(
        strict=True,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    ),
]
RecordProtocol = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]*$"),
]
ResearchBindingRule = Literal["required", "forbidden"]


class ProgramFamilyRef(Model):
    """An exact public family version, e.g. ``swarms.general`` version 1."""

    id: ProgramFamilyId
    version: ProgramFamilyVersion


class ResearchBinding(Model):
    """The research family's bound Forge project and record protocol."""

    forge_project_id: ForgeProjectId
    record_protocol: RecordProtocol


class ProgramChoice(Model):
    family: ProgramFamilyRef
    research: ResearchBinding | None = None

    @model_serializer(mode="wrap")
    def omit_absent_research(self, handler):
        result = handler(self)
        if self.research is None:
            result.pop("research", None)
        return result


class ProgramFamily(ResponseBase):
    """One family version a launch may select."""

    id: ResponseIdentifier
    version: ProgramFamilyVersion
    summary: Annotated[str, Field(min_length=1, max_length=1024)]
    research_binding: ResearchBindingRule
    required_actor_types: tuple[ResponseIdentifier, ...]


class ProgramFamilyList(ResponseModel):
    families: tuple[ProgramFamily, ...]


__all__ = [
    "ProgramChoice",
    "ProgramFamily",
    "ProgramFamilyList",
    "ProgramFamilyRef",
    "ProgramRunView",
    "ResearchBinding",
    "ResearchBindingRule",
]
