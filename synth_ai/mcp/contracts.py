"""Closed MCP request shapes. See SDK_0230_MCP.md; no credentials in tool inputs."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    StrictInt,
    TypeAdapter,
    model_serializer,
    model_validator,
)

from synth_ai.sdk.index.types.contents import ContentOptions, ContentRef
from synth_ai.sdk.index.types.contributions import ContributionFilter
from synth_ai.sdk.index.types.history import SearchHistoryFilter
from synth_ai.sdk.index.types.search import (
    KeyedSearchOptions,
    PublicSearchOptions,
    Query,
    SearchResponse,
)
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.projects import ProjectCreateRequest
from synth_ai.types.base import Cursor, Digest, Identifier, Model
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import RateLimitSnapshot

JsonObject = dict[str, JsonValue]
JSON_OBJECT = TypeAdapter(JsonObject)


class EmptyArguments(Model):
    pass


class PageArguments(Model):
    limit: StrictInt = Field(default=20, ge=1, le=100)
    cursor: Cursor | None = None


class SearchHistoryArguments(PageArguments):
    filter: SearchHistoryFilter = Field(default_factory=SearchHistoryFilter)


class PublicContributionsArguments(PageArguments):
    filter: ContributionFilter = Field(default_factory=ContributionFilter)

    @model_validator(mode="after")
    def validate_public_filter(self) -> Self:
        if self.filter.status not in {None, "published"}:
            raise ValueError("public Contribution tools cannot select draft metadata")
        return self


class ContributionIdentity(Model):
    contribution_id: Identifier


class SearchArguments(Model):
    query: Query
    options: KeyedSearchOptions
    idempotency_key: Identifier


class PublicSearchArguments(Model):
    query: Query
    options: PublicSearchOptions = Field(default_factory=PublicSearchOptions)


class SearchIdentity(Model):
    search_id: Identifier


class SearchControl(SearchIdentity):
    operation_id: Identifier


class SearchPage(SearchIdentity, PageArguments):
    pass


class SearchMutation(SearchIdentity):
    idempotency_key: Identifier


class SearchHistoryMutation(SearchMutation):
    expected_archive_digest_sha256: Digest


class SearchCreate(SearchArguments):
    pass


class ContentArguments(Model):
    reference: ContentRef
    options: ContentOptions


class ProjectIdentity(Model):
    project_id: Identifier


class RunIdentity(Model):
    run_id: Identifier


class RunControl(RunIdentity):
    operation_id: Identifier


class RunPage(RunIdentity, PageArguments):
    pass


class RunList(PageArguments):
    project_id: Identifier | None = None


class RunCreate(LaunchRequest):
    idempotency_key: Identifier


class ProjectCreate(ProjectCreateRequest):
    idempotency_key: Identifier


class RunMutation(RunIdentity):
    idempotency_key: Identifier


class OutputIdentity(RunIdentity):
    output_id: Identifier


class OutputText(OutputIdentity):
    maximum_bytes: StrictInt = Field(default=16384, ge=1, le=65536)


class ProjectPage(ProjectIdentity, PageArguments):
    pass


class FileIdentity(ProjectIdentity):
    file_id: Identifier


class Recovery(Model):
    identity: MutationIdentity


class SearchCreated(Model):
    search_id: Identifier
    # FAST zdr_strict: the answer is delivered here and nowhere else.
    inline_result: SearchResponse | None = None

    @model_serializer(mode="wrap")
    def omit_absent_inline_result(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        # Wire rule: an absent field is omitted, so a backend that never sends it
        # (its monitored body is {"search_id"}) is not contradicted by ``null``.
        data: dict[str, object] = handler(self)
        if data.get("inline_result") is None:
            data.pop("inline_result", None)
        return data


class RunCreated(Model):
    run_id: Identifier


class ToolFailure(Model):
    code: Annotated[str, Field(max_length=128)]
    message: Annotated[str, Field(max_length=512)]
    request_id: Identifier | None = None
    identity: MutationIdentity | None = None
    retry_after_seconds: float | None = None
    required_scopes: tuple[str, ...] = ()
    idempotency_key: Identifier | None = None
    rate_limits: tuple[RateLimitSnapshot, ...] = ()
    outcome: Literal["not_accepted", "unknown"] | None = None
    # retention_mode_not_allowed: the modes the organization permits.
    allowed_modes: tuple[str, ...] = ()
