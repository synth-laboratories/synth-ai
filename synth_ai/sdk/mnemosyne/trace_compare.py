"""Typed ``mnemosyne_compare_project_traces`` request/response (SYN-5334/5338).

A deterministic structural diff of 2..8 distinct retained project archives,
each pinned to its immutable ``archive_sha256``. The tool is capability-gated:
it is only called when the actor MCP session advertises it. Scores and rewards
are reported as found; unknown stays ``None``. The result is bounded by the
backend (480 KiB); truncated listings say so via ``truncated``.
"""

from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from pydantic import Field, StrictBool, StrictInt, ValidationError, model_validator

from synth_ai.types.base import Digest, Model, ResponseBase

from .actor_tools import (
    COMPARE_PROJECT_TRACES_TOOL,
    PROJECT_TRACE_COMPARE_MAX,
    PROJECT_TRACE_COMPARE_MIN,
    ActorToolCaller,
    ActorTraceRefusal,
    AsyncActorToolCaller,
    refuse,
    require_tool,
    structured_result,
)
from .trace_chunks import ArchiveSize, AssociationId

COMPARE_SCHEMA = "synth.mnemosyne.actor-project-trace-compare.v1"
Count = Annotated[StrictInt, Field(ge=0)]
T = TypeVar("T")


class PinnedProjectTrace(Model):
    association_id: AssociationId
    archive_sha256: Digest


class ProjectTraceComparison(Model):
    """Request; order is the caller's labelling (``index`` in the result)."""

    traces: tuple[PinnedProjectTrace, ...] = Field(
        min_length=PROJECT_TRACE_COMPARE_MIN, max_length=PROJECT_TRACE_COMPARE_MAX
    )

    @model_validator(mode="after")
    def distinct_associations(self):
        ids = [item.association_id for item in self.traces]
        if len(set(ids)) != len(ids):
            raise ValueError("distinct project trace associations required")
        return self


class TraceSummary(ResponseBase):
    path: str
    format: str
    step_count: Count
    steps_complete: StrictBool
    reward_steps: Count
    reward_total: float | None
    final_score: float | None
    terminal_status: str


class ManifestEntry(ResponseBase):
    path: str
    size_bytes: Count
    sha256: Digest


class Manifest(ResponseBase):
    truncated: StrictBool
    entries: tuple[ManifestEntry, ...]


class ComparedArchive(ResponseBase):
    index: Annotated[StrictInt, Field(ge=0, lt=PROJECT_TRACE_COMPARE_MAX)]
    association_id: AssociationId
    rollout_id: str
    execution_epoch: str
    archive_sha256: Digest
    archive_size_bytes: ArchiveSize
    entry_count: Count
    expanded_bytes: Count
    manifest_sha256: Digest
    manifest: Manifest
    traces: tuple[TraceSummary, ...]
    traces_not_summarized: Count


class CommonEntry(ResponseBase):
    path: str
    sha256: Digest


class EntryVersion(ResponseBase):
    size_bytes: Count
    sha256: Digest


class ChangedEntry(ResponseBase):
    path: str
    versions: tuple[EntryVersion, ...]


class OnlyInEntry(ResponseBase):
    path: str
    present_in: tuple[Count, ...]


class StepComparison(ResponseBase):
    path: str
    step_counts: tuple[Count | None, ...]
    first_divergence_index: Count | None
    comparable: StrictBool


class Listing(ResponseBase, Generic[T]):
    total: Count
    truncated: StrictBool
    items: tuple[T, ...]


class EntryDiff(ResponseBase):
    common: Listing[CommonEntry]
    changed: Listing[ChangedEntry]
    only_in: Listing[OnlyInEntry]


class ProjectTraceComparisonResult(ResponseBase):
    schema_version: Literal["synth.mnemosyne.actor-project-trace-compare.v1"]
    archive_count: Annotated[
        StrictInt, Field(ge=PROJECT_TRACE_COMPARE_MIN, le=PROJECT_TRACE_COMPARE_MAX)
    ]
    archives: tuple[ComparedArchive, ...]
    entries: EntryDiff
    trace_steps: Listing[StepComparison]


def _verified(request: ProjectTraceComparison, result: object) -> ProjectTraceComparisonResult:
    stage = "compare"
    try:
        parsed = ProjectTraceComparisonResult.model_validate_json(structured_result(result, stage))
    except ValidationError as error:
        first = error.errors(include_input=False)[0]
        where = ".".join(str(p) for p in first["loc"])
        raise refuse(ActorTraceRefusal.RESULT_INVALID, stage, f"{where}: {first['msg']}") from None
    pins = [(t.association_id, t.archive_sha256) for t in request.traces]
    seen = [(a.association_id, a.archive_sha256) for a in parsed.archives]
    if (
        parsed.archive_count != len(pins)
        or seen != pins
        or [a.index for a in parsed.archives] != list(range(len(pins)))
    ):
        raise refuse(
            ActorTraceRefusal.COMPARE_MISMATCH, stage, "result archives differ from the pinned set"
        )
    return parsed


def compare_project_traces(
    caller: ActorToolCaller, request: ProjectTraceComparison
) -> ProjectTraceComparisonResult:
    require_tool(caller, COMPARE_PROJECT_TRACES_TOOL)
    result = caller.call_tool(COMPARE_PROJECT_TRACES_TOOL, request.model_dump(mode="json"))
    return _verified(request, result)


async def acompare_project_traces(
    caller: AsyncActorToolCaller, request: ProjectTraceComparison
) -> ProjectTraceComparisonResult:
    require_tool(caller, COMPARE_PROJECT_TRACES_TOOL)
    result = await caller.call_tool(COMPARE_PROJECT_TRACES_TOOL, request.model_dump(mode="json"))
    return _verified(request, result)
