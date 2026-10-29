"""Closed MCP request shapes. See SDK_0230_MCP.md; no credentials in tool inputs."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
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
from synth_ai.sdk.index.types.retention_policy import RetentionPolicyUpdate
from synth_ai.sdk.index.types.search import (
    KeyedSearchOptions,
    PublicSearchOptions,
    Query,
    SearchResponse,
    StrictSearchEventRequest,
)
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.projects import ProjectCreateRequest
from synth_ai.sdk.swarms.types.runs import LaunchPreflight, ResolvedConfiguration
from synth_ai.types.base import (
    Cursor,
    Digest,
    Identifier,
    Model,
    ResponseBase,
    ResponseIdentifier,
    ResponseModel,
)
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import RateLimitSnapshot
from synth_ai.types.refusal import Refusal

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


class RetentionPolicyMutation(RetentionPolicyUpdate):
    """One policy replacement bound to the revision the caller read."""

    expected_revision: StrictInt = Field(ge=0)
    idempotency_key: Identifier


class SearchCreate(SearchArguments):
    pass


class StrictSearchDelivery(StrictSearchEventRequest):
    search_id: Identifier
    idempotency_key: Identifier


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


class ApprovalDecide(RunMutation):
    approval_id: Identifier
    decision: Literal["approve", "refuse"]
    expected_revision: StrictInt = Field(ge=1)
    reason: str | None = Field(default=None, min_length=1, max_length=4096)


class WaitAnswer(RunMutation):
    approval_id: Identifier
    decision: str = Field(min_length=1, max_length=4096)
    expected_revision: StrictInt = Field(ge=1)
    answer_scope: Literal["once", "session", "policy"] | None = None
    answer_schema: str | None = Field(default=None, min_length=1, max_length=4096)
    answerer: str | None = Field(default=None, min_length=1, max_length=4096)


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


class SearchCreated(ResponseBase):
    search_id: ResponseIdentifier
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


class LaunchPreflightResult(ResponseModel):
    """swarms_preflight result: the preflight body plus response-header capacity (SYN-4680)."""

    ready: bool
    blockers: tuple[str, ...] = Field(max_length=32)
    configuration: ResolvedConfiguration | None
    expires_at: AwareDatetime | None
    reservation_created: Literal[False] = False
    rate_limits: tuple[RateLimitSnapshot, ...] = ()

    @classmethod
    def of(cls, preflight: LaunchPreflight) -> Self:
        fields = {name: getattr(preflight, name) for name in LaunchPreflight.model_fields}
        return cls(**fields, rate_limits=preflight.rate_limits)


class RunCreated(ResponseBase):
    run_id: ResponseIdentifier
    # x-ratelimit-* capacity reported on the admitting create response (SYN-4680).
    rate_limits: tuple[RateLimitSnapshot, ...] = ()


class ToolFailure(ResponseBase):
    refusal: Refusal | None = None
    code: Annotated[str, Field(max_length=128)]
    category: Annotated[str, Field(max_length=128)] | None = None
    message: Annotated[str, Field(max_length=512)]
    status: Annotated[StrictInt, Field(ge=100, le=599)] | None = None
    retryable: bool | None = None
    operation: ResponseIdentifier | None = None
    request_id: ResponseIdentifier | None = None
    identity: MutationIdentity | None = None
    retry_after_seconds: float | None = None
    required_scopes: tuple[str, ...] = ()
    idempotency_key: ResponseIdentifier | None = None
    rate_limits: tuple[RateLimitSnapshot, ...] = ()
    outcome: Literal["not_accepted", "unknown"] | None = None
    # retention_mode_not_allowed: the modes the organization permits.
    allowed_modes: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def omit_absent_refusal(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        # Keep the retained legacy shape on the supported Pydantic 2.11 floor;
        # Field(exclude_if=...) is only implemented by newer Pydantic releases.
        data: dict[str, object] = handler(self)
        if self.refusal is None:
            data.pop("refusal", None)
        return data
