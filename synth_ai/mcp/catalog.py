"""One typed operation catalog for stdio, hosted MCP and generated reference.

See SDK_0230_MCP.md. This is a provisional adapter catalog; native backend
qualification is required before any tool is advertised in a release.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

from synth_ai import AsyncSynthClient
from synth_ai.contracts import forge as forge_operations
from synth_ai.contracts import operations as index_operations
from synth_ai.contracts import swarms as swarm_operations
from synth_ai.contracts.operations import Operation
from synth_ai.contracts.registry import mutation_scope
from synth_ai.options import MutationOptions
from synth_ai.sdk.forge.index_types import (
    ForgeIndexDraftNoticesResponse,
    ForgeIndexDraftResponse,
    ForgeIndexDraftUploadResponse,
)
from synth_ai.sdk.forge.types import (
    ProjectListResponse,
    ProjectMutationResponse,
    ProjectResponse,
    ScientificArtifactEvidenceResponse,
    ScientificReceiptResponse,
    ScientificRecordResponse,
)
from synth_ai.sdk.index.search import validate_search_retention
from synth_ai.sdk.index.types.contents import ContentResponse
from synth_ai.sdk.index.types.contributions import Contribution, SubmissionReceipt
from synth_ai.sdk.index.types.history import SearchHistoryDeletionReceipt, SearchHistoryPage
from synth_ai.sdk.index.types.retention_policy import RetentionPolicy
from synth_ai.sdk.index.types.search import SearchEvent, SearchResponse, SearchSnapshot
from synth_ai.sdk.swarms.types.files import File
from synth_ai.sdk.swarms.types.launch import LaunchRequest, Project
from synth_ai.sdk.swarms.types.outputs import Output, TextContent
from synth_ai.sdk.swarms.types.runs import (
    LaunchPreflight,
    ResolvedConfiguration,
    RunSnapshot,
    SwarmEventPage,
    SwarmResult,
    TaskOutcome,
)
from synth_ai.types.account import AccountEntitlements
from synth_ai.types.auth import AuthContext
from synth_ai.types.base import EventCursor, Page, PageCursor
from synth_ai.types.receipts import ControlReceipt, MutationResolution
from synth_ai.types.response_values import response_schema
from synth_ai.types.usage import UsageOperation, UsageSummary

from . import contribution_owner as owner_tools
from . import forge as forge_tools
from .contracts import (
    JSON_OBJECT,
    ContentArguments,
    ContributionIdentity,
    EmptyArguments,
    FileIdentity,
    JsonObject,
    OutputIdentity,
    OutputText,
    PageArguments,
    ProjectCreate,
    ProjectIdentity,
    ProjectPage,
    PublicContributionsArguments,
    PublicSearchArguments,
    Recovery,
    RetentionPolicyMutation,
    RunControl,
    RunCreate,
    RunCreated,
    RunIdentity,
    RunList,
    RunMutation,
    RunPage,
    SearchArguments,
    SearchControl,
    SearchCreate,
    SearchCreated,
    SearchHistoryArguments,
    SearchHistoryMutation,
    SearchIdentity,
    SearchMutation,
    SearchPage,
    StrictSearchDelivery,
)

RequestT = TypeVar("RequestT", bound=BaseModel)
ResultT = TypeVar("ResultT", bound=BaseModel)


class ToolSpec(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def operation(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def scope(self) -> str | None: ...
    @property
    def additional_scopes(self) -> tuple[str, ...]: ...
    @property
    def mutates(self) -> bool: ...
    @property
    def paid(self) -> bool: ...
    @property
    def keyed(self) -> bool: ...
    @property
    def idempotent(self) -> bool: ...
    def input_schema(self) -> JsonObject: ...
    def output_schema(self) -> JsonObject: ...
    def scope_for(self, arguments: JsonObject) -> str | None: ...
    def validate_arguments(self, arguments: JsonObject) -> JsonObject: ...
    def validate_result(self, result: JsonObject) -> JsonObject: ...
    async def invoke(self, client: AsyncSynthClient, arguments: JsonObject) -> JsonObject: ...


@dataclass(frozen=True, slots=True)
class ToolDefinition(Generic[RequestT, ResultT]):
    name: str
    definition: Operation
    description: str
    request: type[RequestT]
    result: type[ResultT]
    execute: Callable[[AsyncSynthClient, RequestT], Awaitable[ResultT]]
    additional_scopes: tuple[str, ...] = ()

    @property
    def operation(self) -> str:
        return self.definition.id

    @property
    def scope(self) -> str | None:
        return self.definition.required_scope

    @property
    def mutates(self) -> bool:
        return self.definition.mutation

    @property
    def paid(self) -> bool:
        return self.definition.may_charge

    @property
    def keyed(self) -> bool:
        return self.definition.keyed

    @property
    def idempotent(self) -> bool:
        return self.definition.idempotent

    def scope_for(self, arguments: JsonObject) -> str | None:
        if self.definition is index_operations.RESOLVE:
            parsed = Recovery.model_validate_json(JSON_OBJECT.dump_json(arguments))
            return mutation_scope(parsed.identity)
        return self.scope

    def input_schema(self) -> JsonObject:
        return JSON_OBJECT.validate_python(self.request.model_json_schema())

    def output_schema(self) -> JsonObject:
        return JSON_OBJECT.validate_python(response_schema(self.result))

    def validate_arguments(self, arguments: JsonObject) -> JsonObject:
        parsed = self.request.model_validate_json(JSON_OBJECT.dump_json(arguments))
        return JSON_OBJECT.validate_json(parsed.model_dump_json())

    def validate_result(self, result: JsonObject) -> JsonObject:
        parsed = self.result.model_validate_json(JSON_OBJECT.dump_json(result))
        if isinstance(parsed, (SearchResponse, SearchSnapshot)):
            validate_search_retention(parsed)
        elif isinstance(parsed, Page):
            for item in parsed.items:
                if isinstance(item, SearchSnapshot):
                    validate_search_retention(item)
        return JSON_OBJECT.validate_json(parsed.model_dump_json())

    async def invoke(self, client: AsyncSynthClient, arguments: JsonObject) -> JsonObject:
        parsed = self.request.model_validate_json(JSON_OBJECT.dump_json(arguments))
        result = await self.execute(client, parsed)
        return self.validate_result(JSON_OBJECT.validate_json(result.model_dump_json()))


async def auth_context(client: AsyncSynthClient, args: EmptyArguments) -> AuthContext:
    return await client.auth_context()


async def account_entitlements(
    client: AsyncSynthClient, args: EmptyArguments
) -> AccountEntitlements:
    return await client.account.entitlements()


async def retention_get(client: AsyncSynthClient, args: EmptyArguments) -> RetentionPolicy:
    return await client.index.retention.get()


async def retention_update(
    client: AsyncSynthClient, args: RetentionPolicyMutation
) -> RetentionPolicy:
    return await client.index.retention.update(
        default_mode=args.default_mode,
        allowed_modes=args.allowed_modes,
        enforced=args.enforced,
        expected_revision=args.expected_revision,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def public_search(client: AsyncSynthClient, args: PublicSearchArguments) -> SearchResponse:
    return await client.index.search(args.query, options=args.options)


async def search(client: AsyncSynthClient, args: SearchArguments) -> SearchResponse:
    return await client.index.search(
        args.query,
        options=args.options,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def create_search(client: AsyncSynthClient, args: SearchCreate) -> SearchCreated:
    run = await client.index.searches.create(
        args.query,
        options=args.options,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )
    return SearchCreated(search_id=run.search_id, inline_result=run.inline_result)


async def search_status(client: AsyncSynthClient, args: SearchIdentity) -> SearchSnapshot:
    return await (await client.index.searches.get(args.search_id)).status()


async def search_result(client: AsyncSynthClient, args: SearchIdentity) -> SearchResponse:
    return await (await client.index.searches.get(args.search_id)).result()


async def search_list(client: AsyncSynthClient, args: PageArguments) -> Page[SearchSnapshot]:
    return await client.index.searches.list(
        limit=args.limit, cursor=PageCursor(args.cursor) if args.cursor else None
    )


async def public_contributions(
    client: AsyncSynthClient, args: PublicContributionsArguments
) -> Page[Contribution]:
    return await client.index.contributions.list(
        limit=args.limit,
        cursor=PageCursor(args.cursor) if args.cursor else None,
        filter=args.filter,
    )


async def public_contribution(client: AsyncSynthClient, args: ContributionIdentity) -> Contribution:
    return await client.index.contributions.get(args.contribution_id)


async def search_history(
    client: AsyncSynthClient, args: SearchHistoryArguments
) -> SearchHistoryPage:
    return await client.index.searches.history(
        limit=args.limit,
        cursor=PageCursor(args.cursor) if args.cursor else None,
        filter=args.filter,
    )


async def delete_search_history(
    client: AsyncSynthClient, args: SearchHistoryMutation
) -> SearchHistoryDeletionReceipt:
    return await (await client.index.searches.get(args.search_id)).delete_history(
        expected_archive_digest_sha256=args.expected_archive_digest_sha256,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def search_events(
    client: AsyncSynthClient, args: SearchPage
) -> Page[SearchEvent, EventCursor]:
    return await (await client.index.searches.get(args.search_id)).event_page(
        limit=args.limit, cursor=EventCursor(args.cursor) if args.cursor else None
    )


async def search_usage(client: AsyncSynthClient, args: SearchIdentity) -> UsageSummary:
    return await (await client.index.searches.get(args.search_id)).usage()


async def search_usage_items(client: AsyncSynthClient, args: SearchPage) -> Page[UsageOperation]:
    return await (await client.index.searches.get(args.search_id)).usage_operations(
        limit=args.limit, cursor=EventCursor(args.cursor) if args.cursor else None
    )


async def search_cancel(client: AsyncSynthClient, args: SearchMutation) -> ControlReceipt:
    return await (await client.index.searches.get(args.search_id)).cancel(
        mutation=MutationOptions(idempotency_key=args.idempotency_key)
    )


async def content(client: AsyncSynthClient, args: ContentArguments) -> ContentResponse:
    return await client.index.contents.get(args.reference, options=args.options)


async def recovery(client: AsyncSynthClient, args: Recovery) -> MutationResolution:
    return await client.resolve_mutation(args.identity)


async def project(client: AsyncSynthClient, args: ProjectIdentity) -> Project:
    return await client.swarms.projects.get(args.project_id)


async def create_project(client: AsyncSynthClient, args: ProjectCreate) -> Project:
    return await client.swarms.projects.create(
        args.name,
        objective=args.objective,
        template=args.template,
        code_validation=args.code_validation,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )


async def preflight(client: AsyncSynthClient, args: LaunchRequest) -> LaunchPreflight:
    return await client.swarms.preflight(
        args.legacy_objective,
        message=args.message,
        project_id=args.project_id,
        limits=args.limits,
        inputs=args.inputs,
        providers=args.providers,
        models=args.models,
        labels=args.labels,
        repo=args.repo,
        guidance=args.guidance,
        resources=args.resources,
    )


async def create_run(client: AsyncSynthClient, args: RunCreate) -> RunCreated:
    run = await client.swarms.create(
        args.legacy_objective,
        message=args.message,
        project_id=args.project_id,
        limits=args.limits,
        inputs=args.inputs,
        providers=args.providers,
        models=args.models,
        labels=args.labels,
        repo=args.repo,
        guidance=args.guidance,
        resources=args.resources,
        mutation=MutationOptions(idempotency_key=args.idempotency_key),
    )
    return RunCreated(run_id=run.run_id)


async def runs(client: AsyncSynthClient, args: RunList) -> Page[RunSnapshot]:
    return await client.swarms.list(
        limit=args.limit,
        cursor=PageCursor(args.cursor) if args.cursor else None,
        project_id=args.project_id,
    )


async def run_status(client: AsyncSynthClient, args: RunIdentity) -> RunSnapshot:
    return await (await client.swarms.get(args.run_id)).status()


async def run_configuration(client: AsyncSynthClient, args: RunIdentity) -> ResolvedConfiguration:
    return await (await client.swarms.get(args.run_id)).configuration()


async def run_result(client: AsyncSynthClient, args: RunIdentity) -> SwarmResult:
    return await (await client.swarms.get(args.run_id)).result()


async def run_events(client: AsyncSynthClient, args: RunPage) -> SwarmEventPage:
    return await (await client.swarms.get(args.run_id)).event_page(
        limit=args.limit, cursor=EventCursor(args.cursor) if args.cursor else None
    )


async def run_tasks(client: AsyncSynthClient, args: RunPage) -> Page[TaskOutcome]:
    return await (await client.swarms.get(args.run_id)).tasks(
        limit=args.limit, cursor=PageCursor(args.cursor) if args.cursor else None
    )


async def run_usage(client: AsyncSynthClient, args: RunIdentity) -> UsageSummary:
    return await (await client.swarms.get(args.run_id)).usage()


async def run_usage_items(client: AsyncSynthClient, args: RunPage) -> Page[UsageOperation]:
    return await (await client.swarms.get(args.run_id)).usage_operations(
        limit=args.limit, cursor=PageCursor(args.cursor) if args.cursor else None
    )


async def run_cancel(client: AsyncSynthClient, args: RunMutation) -> ControlReceipt:
    return await (await client.swarms.get(args.run_id)).cancel(
        mutation=MutationOptions(idempotency_key=args.idempotency_key)
    )


async def run_pause(client: AsyncSynthClient, args: RunMutation) -> ControlReceipt:
    return await (await client.swarms.get(args.run_id)).pause(
        mutation=MutationOptions(idempotency_key=args.idempotency_key)
    )


async def run_resume(client: AsyncSynthClient, args: RunMutation) -> ControlReceipt:
    return await (await client.swarms.get(args.run_id)).resume(
        mutation=MutationOptions(idempotency_key=args.idempotency_key)
    )


async def outputs(client: AsyncSynthClient, args: RunPage) -> Page[Output]:
    return await (await client.swarms.get(args.run_id)).outputs.list(
        limit=args.limit, cursor=PageCursor(args.cursor) if args.cursor else None
    )


async def output(client: AsyncSynthClient, args: OutputIdentity) -> Output:
    return await (await client.swarms.get(args.run_id)).outputs.get(args.output_id)


async def output_text(client: AsyncSynthClient, args: OutputText) -> TextContent:
    return await (await client.swarms.get(args.run_id)).outputs.read_text(
        args.output_id, maximum_bytes=args.maximum_bytes
    )


async def files(client: AsyncSynthClient, args: ProjectPage) -> Page[File]:
    return await client.swarms.files.list(
        args.project_id, limit=args.limit, cursor=PageCursor(args.cursor) if args.cursor else None
    )


async def file(client: AsyncSynthClient, args: FileIdentity) -> File:
    return await client.swarms.files.get(args.project_id, args.file_id)


async def search_control(client: AsyncSynthClient, args: SearchControl) -> ControlReceipt:
    return await (await client.index.searches.get(args.search_id)).control(args.operation_id)


async def run_control(client: AsyncSynthClient, args: RunControl) -> ControlReceipt:
    return await (await client.swarms.get(args.run_id)).control(args.operation_id)


async def deliver_search_events(
    client: AsyncSynthClient, args: StrictSearchDelivery
) -> Page[SearchEvent, EventCursor]:
    from synth_ai.sdk.index.search_run import AsyncSearchRun
    from synth_ai.sdk.index.types.search import StrictSearchProgress

    from .progress import publish_progress

    run = AsyncSearchRun(client.index._transport, args.search_id, expected_retention="zdr_strict")
    async with run.deliver_events(
        original_request=args.request,
        creation_key=args.creation_key,
        mutation=MutationOptions(idempotency_key=args.idempotency_key, timeout_seconds=690),
    ) as events:
        terminal = None
        async for event in events:
            if isinstance(event, StrictSearchProgress):
                await publish_progress(event)
            else:
                terminal = event
        if terminal is None:
            from synth_ai.errors import OutcomeUncertainError

            raise OutcomeUncertainError("Strict event terminal was not delivered")
        return Page[SearchEvent, EventCursor](
            metadata=terminal.result.metadata, items=(terminal,), complete=True, next_cursor=None
        )


TOOLS: tuple[ToolSpec, ...] = (
    ToolDefinition(
        "index_contribution_submit",
        index_operations.SUBMIT,
        "Validate and submit one exact finalized private owner revision; retain its retry key. This does not attest rights, qualify research or publish.",
        owner_tools.ContributionSubmit,
        SubmissionReceipt,
        owner_tools.submit,
    ),
    ToolDefinition(
        "index_contribution_owner_get",
        index_operations.OWNED_CONTRIBUTION,
        "Inspect current private owner state through the owner endpoint; public browsing remains public-only.",
        ContributionIdentity,
        Contribution,
        owner_tools.get,
    ),
    ToolDefinition(
        "forge_index_draft_upload",
        forge_operations.INDEX_DRAFT_UPLOAD,
        "Package frozen native Artifact outputs into a private Index draft; retain the key for recovery. Returns IDs and state without transfer credentials.",
        forge_tools.ForgeIndexDraftUpload,
        ForgeIndexDraftUploadResponse,
        forge_tools.index_draft_upload,
        additional_scopes=("index:intake",),
    ),
    ToolDefinition(
        "forge_index_draft_upload_get",
        forge_operations.INDEX_DRAFT_UPLOAD_GET,
        "Recover private upload custody and package IDs under current Forge and Index access.",
        forge_tools.ForgeIndexDraftUploadRead,
        ForgeIndexDraftUploadResponse,
        forge_tools.index_draft_upload_get,
        additional_scopes=("index:intake",),
    ),
    ToolDefinition(
        "forge_artifact_evidence_get",
        forge_operations.ARTIFACT_EVIDENCE_GET,
        "Inspect retained Artifact read evidence for an exact revision; byte access does not prove capture or execution.",
        forge_tools.ForgeRecordRead,
        ScientificArtifactEvidenceResponse,
        forge_tools.artifact_evidence_get,
    ),
    ToolDefinition(
        "forge_index_draft_create",
        forge_operations.INDEX_DRAFT_CREATE,
        "Allocate one private Index draft for an exact frozen Forge export; retain the key for recovery.",
        forge_tools.ForgeIndexDraftCreate,
        ForgeIndexDraftResponse,
        forge_tools.index_draft_create,
        additional_scopes=("index:intake",),
    ),
    ToolDefinition(
        "forge_index_draft_get",
        forge_operations.INDEX_DRAFT_GET,
        "Recover original private draft IDs under current Forge and Index access.",
        forge_tools.ForgeReceiptRead,
        ForgeIndexDraftResponse,
        forge_tools.index_draft_get,
        additional_scopes=("index:intake",),
    ),
    ToolDefinition(
        "forge_index_draft_notices",
        forge_operations.INDEX_DRAFT_NOTICES,
        "Read exact private source correction notices under current Forge and Index access; notices do not change released bytes or publish.",
        forge_tools.ForgeIndexDraftNoticeRead,
        ForgeIndexDraftNoticesResponse,
        forge_tools.index_draft_notices,
        additional_scopes=("index:intake",),
    ),
    ToolDefinition(
        "forge_record_get",
        forge_operations.RECORD_GET,
        "Read an exact scientific record revision with current workspace access.",
        forge_tools.ForgeRecordRead,
        ScientificRecordResponse,
        forge_tools.record_get,
    ),
    ToolDefinition(
        "forge_record_write",
        forge_operations.RECORD_WRITE,
        "Record typed scientific intent under current admission with a stable retry key.",
        forge_tools.ForgeRecordWrite,
        ScientificReceiptResponse,
        forge_tools.record_write,
    ),
    ToolDefinition(
        "forge_receipt_get",
        forge_operations.RECEIPT_GET,
        "Recover the original scientific receipt after an uncertain write outcome.",
        forge_tools.ForgeReceiptRead,
        ScientificReceiptResponse,
        forge_tools.receipt_get,
    ),
    ToolDefinition(
        "forge_project_list",
        forge_operations.PROJECT_LIST,
        "List accessible research workspaces in bounded pages.",
        forge_tools.ForgeProjectPage,
        ProjectListResponse,
        forge_tools.list_projects,
    ),
    ToolDefinition(
        "forge_project_get",
        forge_operations.PROJECT_GET,
        "Inspect a research workspace with current access and retained revision.",
        forge_tools.ForgeProjectIdentity,
        ProjectResponse,
        forge_tools.get,
    ),
    ToolDefinition(
        "forge_project_create",
        forge_operations.PROJECT_APPLY,
        "Create an independent research workspace with a stable retry key.",
        forge_tools.ForgeProjectCreate,
        ProjectMutationResponse,
        forge_tools.create,
    ),
    ToolDefinition(
        "forge_project_update",
        forge_operations.PROJECT_APPLY,
        "Revise a research workspace title and objective at its expected revision.",
        forge_tools.ForgeProjectUpdate,
        ProjectMutationResponse,
        forge_tools.update,
    ),
    ToolDefinition(
        "forge_project_archive",
        forge_operations.PROJECT_APPLY,
        "Archive or unarchive a research workspace while retaining its evidence.",
        forge_tools.ForgeProjectArchive,
        ProjectMutationResponse,
        forge_tools.archive,
    ),
    ToolDefinition(
        "forge_project_members",
        forge_operations.PROJECT_APPLY,
        "Replace explicit workspace membership at its expected revision.",
        forge_tools.ForgeProjectMembers,
        ProjectMutationResponse,
        forge_tools.members,
    ),
    ToolDefinition(
        "index_search_deliver_events",
        index_operations.DELIVER_SEARCH_EVENTS,
        "Deliver live strict DEEP progress and the terminal result using original intent and creation key. Requires a progress token; content cannot be replayed.",
        StrictSearchDelivery,
        Page[SearchEvent, EventCursor],
        deliver_search_events,
    ),
    ToolDefinition(
        "index_search_control",
        index_operations.CONTROL,
        "Read an existing cancellation receipt without issuing another cancellation.",
        SearchControl,
        ControlReceipt,
        search_control,
    ),
    ToolDefinition(
        "swarms_control",
        swarm_operations.CONTROL,
        "Read an existing control receipt with independent cleanup and accounting state.",
        RunControl,
        ControlReceipt,
        run_control,
    ),
    ToolDefinition(
        "auth_context",
        index_operations.AUTH_CONTEXT,
        "Read current identity, organization and credential grants without exposing credentials.",
        EmptyArguments,
        AuthContext,
        auth_context,
    ),
    ToolDefinition(
        "account_entitlements",
        index_operations.ACCOUNT_ENTITLEMENTS,
        "Read this organization's product access, conditions and observed limits.",
        EmptyArguments,
        AccountEntitlements,
        account_entitlements,
    ),
    ToolDefinition(
        "index_public_search",
        index_operations.PUBLIC_SEARCH,
        "Search public Index in one bounded call. No durable handle or private collections.",
        PublicSearchArguments,
        SearchResponse,
        public_search,
    ),
    ToolDefinition(
        "index_search",
        index_operations.SEARCH,
        "Search authorized Index with an explicit customer charge ceiling. May incur cost.",
        SearchArguments,
        SearchResponse,
        search,
    ),
    ToolDefinition(
        "index_search_create",
        index_operations.CREATE_SEARCH,
        "Create a durable search. Requires stable idempotency key and charge ceiling; may incur cost.",
        SearchCreate,
        SearchCreated,
        create_search,
    ),
    ToolDefinition(
        "index_search_status",
        index_operations.GET_SEARCH,
        "Read current authorized search status.",
        SearchIdentity,
        SearchSnapshot,
        search_status,
    ),
    ToolDefinition(
        "index_search_result",
        index_operations.SEARCH_RESULT,
        "Read a saved search result with citations, usage, cost, charge and timing.",
        SearchIdentity,
        SearchResponse,
        search_result,
    ),
    ToolDefinition(
        "index_search_list",
        index_operations.LIST_SEARCHES,
        "Read one page of your durable searches.",
        PageArguments,
        Page[SearchSnapshot],
        search_list,
    ),
    ToolDefinition(
        "index_public_contributions_list",
        index_operations.CONTRIBUTIONS,
        "Read one page of public Contributions and approved revision annotations. No private drafts.",
        PublicContributionsArguments,
        Page[Contribution],
        public_contributions,
    ),
    ToolDefinition(
        "index_public_contribution",
        index_operations.CONTRIBUTION,
        "Read one public Contribution and its approved revision-bound review ratings.",
        ContributionIdentity,
        Contribution,
        public_contribution,
    ),
    ToolDefinition(
        "index_search_history",
        index_operations.SEARCH_HISTORY,
        "Read one private 30-day retained-history page. ZDR searches have no retained history.",
        SearchHistoryArguments,
        SearchHistoryPage,
        search_history,
    ),
    ToolDefinition(
        "index_retention_get",
        index_operations.GET_RETENTION,
        "Read this organization's default and allowed Index retention modes.",
        EmptyArguments,
        RetentionPolicy,
        retention_get,
    ),
    ToolDefinition(
        "index_retention_update",
        index_operations.PUT_RETENTION,
        "Replace org retention policy at the observed revision with a stable key. Org admins only.",
        RetentionPolicyMutation,
        RetentionPolicy,
        retention_update,
    ),
    ToolDefinition(
        "index_search_history_delete",
        index_operations.DELETE_SEARCH_HISTORY,
        "Delete the exact archived payload with a stable key. Does not cancel or rerun the search.",
        SearchHistoryMutation,
        SearchHistoryDeletionReceipt,
        delete_search_history,
    ),
    ToolDefinition(
        "index_search_events",
        index_operations.SEARCH_EVENTS,
        "Read one bounded event page; use next_cursor to follow.",
        SearchPage,
        Page[SearchEvent, EventCursor],
        search_events,
    ),
    ToolDefinition(
        "index_search_usage",
        index_operations.SEARCH_USAGE,
        "Read observed usage, provider cost and customer charge with independent finality.",
        SearchIdentity,
        UsageSummary,
        search_usage,
    ),
    ToolDefinition(
        "index_search_usage_items",
        index_operations.SEARCH_USAGE_ITEMS,
        "Read itemized search usage and cost observations.",
        SearchPage,
        Page[UsageOperation],
        search_usage_items,
    ),
    ToolDefinition(
        "index_search_cancel",
        index_operations.CANCEL_SEARCH,
        "Request cancellation using an stable idempotency key. Does not imply settlement or cleanup.",
        SearchMutation,
        ControlReceipt,
        search_cancel,
    ),
    ToolDefinition(
        "index_content",
        index_operations.CONTENTS,
        "Read currently authorized content at an exact revision and digest. Citations grant no access.",
        ContentArguments,
        ContentResponse,
        content,
    ),
    ToolDefinition(
        "mutation_resolve",
        index_operations.RESOLVE,
        "Resolve a retained mutation identity before repeating uncertain work. Backend checks original operation authority.",
        Recovery,
        MutationResolution,
        recovery,
    ),
    ToolDefinition(
        "swarms_project",
        swarm_operations.PROJECT,
        "Inspect an existing authorized project.",
        ProjectIdentity,
        Project,
        project,
    ),
    ToolDefinition(
        "swarms_project_create",
        swarm_operations.CREATE_PROJECT,
        "Create a launchable project from a server-authored template with a stable idempotency key. Requires an organization admin.",
        ProjectCreate,
        Project,
        create_project,
    ),
    ToolDefinition(
        "swarms_preflight",
        swarm_operations.PREFLIGHT,
        "Preview resolved launch configuration and blockers without starting execution or reserving budget.",
        LaunchRequest,
        LaunchPreflight,
        preflight,
    ),
    ToolDefinition(
        "swarms_create",
        swarm_operations.CREATE,
        "Start a bounded run with an stable idempotency key. May incur cost. Omitted overrides preserve project defaults.",
        RunCreate,
        RunCreated,
        create_run,
    ),
    ToolDefinition(
        "swarms_list",
        swarm_operations.LIST,
        "Read one authorized run page, optionally within a project.",
        RunList,
        Page[RunSnapshot],
        runs,
    ),
    ToolDefinition(
        "swarms_status",
        swarm_operations.GET,
        "Read current run, accounting and cleanup status.",
        RunIdentity,
        RunSnapshot,
        run_status,
    ),
    ToolDefinition(
        "swarms_configuration",
        swarm_operations.CONFIGURATION,
        "Read the exact resolved run configuration, distinct from caller overrides.",
        RunIdentity,
        ResolvedConfiguration,
        run_configuration,
    ),
    ToolDefinition(
        "swarms_result",
        swarm_operations.RESULT,
        "Read the canonical result including partial or failed outcomes.",
        RunIdentity,
        SwarmResult,
        run_result,
    ),
    ToolDefinition(
        "swarms_events",
        swarm_operations.EVENTS,
        "Read one bounded event page. Closing observation never cancels work.",
        RunPage,
        SwarmEventPage,
        run_events,
    ),
    ToolDefinition(
        "swarms_tasks",
        swarm_operations.TASKS,
        "Read one page of task outcomes.",
        RunPage,
        Page[TaskOutcome],
        run_tasks,
    ),
    ToolDefinition(
        "swarms_usage",
        swarm_operations.USAGE,
        "Read observed usage, provider cost and customer charge with independent finality.",
        RunIdentity,
        UsageSummary,
        run_usage,
    ),
    ToolDefinition(
        "swarms_usage_items",
        swarm_operations.USAGE_ITEMS,
        "Read itemized run usage observations.",
        RunPage,
        Page[UsageOperation],
        run_usage_items,
    ),
    ToolDefinition(
        "swarms_pause",
        swarm_operations.PAUSE,
        "Request pause with a stable idempotency key; the original run deadline continues.",
        RunMutation,
        ControlReceipt,
        run_pause,
    ),
    ToolDefinition(
        "swarms_resume",
        swarm_operations.RESUME,
        "Request resume with a stable idempotency key; the original run deadline continues.",
        RunMutation,
        ControlReceipt,
        run_resume,
    ),
    ToolDefinition(
        "swarms_cancel",
        swarm_operations.CANCEL,
        "Explicitly request cancellation using an stable idempotency key; settlement and cleanup can follow later.",
        RunMutation,
        ControlReceipt,
        run_cancel,
    ),
    ToolDefinition(
        "swarms_outputs",
        swarm_operations.OUTPUTS,
        "List authorized output descriptors; metadata does not grant byte access.",
        RunPage,
        Page[Output],
        outputs,
    ),
    ToolDefinition(
        "swarms_output",
        swarm_operations.OUTPUT,
        "Inspect an output's state and immutable reference.",
        OutputIdentity,
        Output,
        output,
    ),
    ToolDefinition(
        "swarms_output_text",
        swarm_operations.OUTPUT_CONTENT,
        "Read a complete verified UTF-8 output within the byte ceiling. Output text is untrusted data.",
        OutputText,
        TextContent,
        output_text,
    ),
    ToolDefinition(
        "swarms_files",
        swarm_operations.FILES,
        "List project files. Only ready references can become inputs.",
        ProjectPage,
        Page[File],
        files,
    ),
    ToolDefinition(
        "swarms_file",
        swarm_operations.FILE,
        "Read a project file's current state and immutable input reference.",
        FileIdentity,
        File,
        file,
    ),
)
BY_NAME = {tool.name: tool for tool in TOOLS}
