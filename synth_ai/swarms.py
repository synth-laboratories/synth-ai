"""Run ergonomics over the generated public operation inventory."""

from __future__ import annotations

from contextlib import aclosing
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ._event_stream import (
    follow_run_messages,
    iter_available_messages,
    reraise_result_unavailable,
    wait_for_correlated_reply,
)
from ._generated.models import (
    EventPage,
    FamilyList,
    File,
    FilePage,
    LaunchChoices,
    LaunchRequest,
    McpOutputContent,
    MessageKind,
    OptimizerRequest,
    OutputPage,
    OwnerResourceRequest,
    RoleAssignment,
    RoutingCatalog,
    RoutingConstraints,
    RunAllocation,
    RunApprovalDecisionReceipt,
    RunApprovalDecisionRequest,
    RunApprovalPage,
    RunConfiguration,
    RunControlReceipt,
    RunEvent,
    RunGuidance,
    RunInputs,
    RunLabels,
    RunLabelsRequest,
    RunLimitsProgress,
    RunLimitUpdate,
    RunMessage,
    RunMessagePage,
    RunOutput,
    RunPage,
    RunPreflightResponse,
    RunProfileCatalog,
    RunProgram,
    RunReport,
    RunResult,
    RunReviewSurface,
    RunSandboxes,
    RunSnapshot,
    RunTask,
    RunTaskPage,
    RunWait,
    SendRunMessage,
    Upload,
)
from ._generated.policy import (
    DEFAULT_RUN_PROFILE,
    MCP_OUTPUT_CONTENT_MAX_BYTES,
    PROFILE_REQUIRES_RESEARCH_TASK,
)
from ._inline_content import inline_content
from ._limits import MESSAGE_REPLY_TIMEOUT_SECONDS_DEFAULT
from ._observer import RunObserver, validate_wait
from ._pagination import paginate
from ._protocol import CompletionMilestone
from ._requests import (
    DomicileRequirement,
    launch_choices_query,
    launch_request,
    limit_update_request,
    prepare_upload_body,
    upload_text_body,
)
from ._stream import StreamCheckpoint
from ._transport import Transport
from ._uploads import ArtifactUploads
from ._validation import verify_sha256
from .errors import ConflictError

if TYPE_CHECKING:
    import builtins
    from collections.abc import AsyncGenerator
    from decimal import Decimal

__all__ = ["AsyncSwarms"]


class AsyncSwarms:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._uploads = ArtifactUploads(transport)
        self._profile_requirements = dict(PROFILE_REQUIRES_RESEARCH_TASK)
        self._default_profile = DEFAULT_RUN_PROFILE

    async def send_message(
        self,
        run_id: str,
        text: str,
        *,
        kind: MessageKind = MessageKind.GUIDANCE,
        idempotency_key: str,
    ) -> RunMessage:
        """Send to the admitted thread; retain idempotency_key across uncertain retries."""
        return await self._transport.request(
            "sendRunMessage",
            RunMessage,
            path={"run_id": run_id},
            body=SendRunMessage(kind=kind, text=text),
            idempotency_key=idempotency_key,
        )

    async def messages(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> RunMessagePage:
        """Read customer-visible history, preserving typed unavailability."""
        return await self._transport.request(
            "listRunMessages",
            RunMessagePage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def iter_messages(
        self, run_id: str, *, cursor: str | None = None, limit: int = 100
    ) -> AsyncGenerator[RunMessage, None]:
        """Traverse message pages without prefetch; see docs/protocols.md."""
        async for item in iter_available_messages(
            lambda position: self.messages(run_id, cursor=position, limit=limit), cursor=cursor
        ):
            yield item

    async def message(self, run_id: str, message_id: str) -> RunMessage:
        """Read one message; a retained body is typed BodyUnavailable."""
        return await self._transport.request(
            "getRunMessage", RunMessage, path={"run_id": run_id, "message_id": message_id}
        )

    async def follow_messages(self, run_id: str) -> AsyncGenerator[RunMessage, None]:
        """Read history, then live message updates from one feed; see docs/protocols.md."""
        async for item in follow_run_messages(
            self._transport,
            run_id,
            lambda: self.iter_messages(run_id),
            lambda message_id: self.message(run_id, message_id),
        ):
            yield item

    async def wait_for_reply(
        self,
        run_id: str,
        message_id: str,
        *,
        timeout: float = MESSAGE_REPLY_TIMEOUT_SECONDS_DEFAULT,
    ) -> RunMessage:
        """Wait for a correlated posted reply without mutating the message."""
        return await wait_for_correlated_reply(
            self.follow_messages(run_id), run_id, message_id, timeout
        )

    def prepare_launch(
        self,
        message: str,
        *,
        budget: str | Decimal,
        domicile_requirement: DomicileRequirement | None = None,
        wall_time_seconds: int | None = None,
        maximum_tokens: int | None = None,
        pause_retention_seconds: int | None = None,
        parallel_workers: int | None = None,
        inputs: RunInputs | None = None,
        labels: dict[str, str] | None = None,
        resources: builtins.list[OwnerResourceRequest] | None = None,
        profile: str | None = None,
        research_task: str | None = None,
        routing: RoutingConstraints | None = None,
        guidance: builtins.list[RunGuidance] | None = None,
        role_assignments: dict[str, RoleAssignment] | None = None,
        launch_menu_revision: str | None = None,
        optimizer: OptimizerRequest | None = None,
        continue_from: str | None = None,
    ) -> LaunchRequest:
        """Validate a launch before uploads; see docs/protocols.md."""
        return launch_request(
            message,
            budget=budget,
            domicile_requirement=domicile_requirement,
            wall_time_seconds=wall_time_seconds,
            maximum_tokens=maximum_tokens,
            pause_retention_seconds=pause_retention_seconds,
            parallel_workers=parallel_workers,
            inputs=inputs,
            labels=labels,
            resources=resources,
            profile=profile,
            research_task=research_task,
            routing=routing,
            guidance=guidance,
            profile_requirements=self._profile_requirements,
            default_profile=self._default_profile,
            role_assignments=role_assignments,
            launch_menu_revision=launch_menu_revision,
            optimizer=optimizer,
            continue_from=continue_from,
        )

    async def launch(
        self,
        message: str,
        *,
        budget: str | Decimal,
        domicile_requirement: DomicileRequirement | None = None,
        wall_time_seconds: int | None = None,
        maximum_tokens: int | None = None,
        pause_retention_seconds: int | None = None,
        parallel_workers: int | None = None,
        inputs: RunInputs | None = None,
        labels: dict[str, str] | None = None,
        resources: builtins.list[OwnerResourceRequest] | None = None,
        profile: str | None = None,
        research_task: str | None = None,
        routing: RoutingConstraints | None = None,
        guidance: builtins.list[RunGuidance] | None = None,
        role_assignments: dict[str, RoleAssignment] | None = None,
        launch_menu_revision: str | None = None,
        optimizer: OptimizerRequest | None = None,
        continue_from: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunSnapshot:
        """Launch a run with a maximum USD charge; use wait/report to confirm completion."""
        request = self.prepare_launch(
            message,
            budget=budget,
            domicile_requirement=domicile_requirement,
            wall_time_seconds=wall_time_seconds,
            maximum_tokens=maximum_tokens,
            pause_retention_seconds=pause_retention_seconds,
            parallel_workers=parallel_workers,
            inputs=inputs,
            labels=labels,
            resources=resources,
            profile=profile,
            research_task=research_task,
            routing=routing,
            guidance=guidance,
            role_assignments=role_assignments,
            launch_menu_revision=launch_menu_revision,
            optimizer=optimizer,
            continue_from=continue_from,
        )
        return await self.launch_request(request, idempotency_key=idempotency_key)

    async def launch_request(
        self, request: LaunchRequest, *, idempotency_key: str | None = None
    ) -> RunSnapshot:
        """Submit a typed launch request and return its initial run snapshot."""
        return await self._transport.request(
            "createRun", RunSnapshot, body=request, idempotency_key=idempotency_key
        )

    async def preflight_request(self, request: LaunchRequest) -> RunPreflightResponse:
        """Dry-run launch admission preflight at zero cost without creating a run."""
        return await self._transport.request("preflightRun", RunPreflightResponse, body=request)

    async def run(
        self,
        message: str,
        *,
        budget: str | Decimal,
        domicile_requirement: DomicileRequirement | None = None,
        profile: str | None = None,
        research_task: str | None = None,
        timeout: float = 600,
        poll_interval: float = 1,
        until: CompletionMilestone = CompletionMilestone.FINISHED,
        raise_on_failure: bool = False,
        wall_time_seconds: int | None = None,
        maximum_tokens: int | None = None,
        pause_retention_seconds: int | None = None,
        parallel_workers: int | None = None,
        inputs: RunInputs | None = None,
        labels: dict[str, str] | None = None,
        resources: builtins.list[OwnerResourceRequest] | None = None,
        routing: RoutingConstraints | None = None,
        guidance: builtins.list[RunGuidance] | None = None,
        role_assignments: dict[str, RoleAssignment] | None = None,
        launch_menu_revision: str | None = None,
        optimizer: OptimizerRequest | None = None,
        continue_from: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunSnapshot:
        """Launch a run, then wait for the selected completion milestone."""
        validate_wait(timeout, poll_interval, until)
        created = await self.launch(
            message,
            budget=budget,
            domicile_requirement=domicile_requirement,
            profile=profile,
            research_task=research_task,
            wall_time_seconds=wall_time_seconds,
            maximum_tokens=maximum_tokens,
            pause_retention_seconds=pause_retention_seconds,
            parallel_workers=parallel_workers,
            inputs=inputs,
            labels=labels,
            resources=resources,
            routing=routing,
            guidance=guidance,
            role_assignments=role_assignments,
            launch_menu_revision=launch_menu_revision,
            optimizer=optimizer,
            continue_from=continue_from,
            idempotency_key=idempotency_key,
        )
        return await self.wait(
            created.run_id,
            timeout=timeout,
            poll_interval=poll_interval,
            until=until,
            raise_on_failure=raise_on_failure,
        )

    async def upload_file(
        self,
        filename: str,
        *,
        media_type: str | None = None,
        idempotency_key: str | None = None,
    ) -> File:
        """Upload a local file and return its verified file record."""
        return await self._uploads.upload_file(
            filename, media_type=media_type, idempotency_key=idempotency_key
        )

    async def upload_directory(
        self, directory: str | int, *, idempotency_key: str
    ) -> builtins.list[File]:
        """Upload a bounded POSIX directory snapshot; partial failures retain completed_files."""
        return await self._uploads.upload_directory(directory, idempotency_key=idempotency_key)

    async def resume_upload(self, upload_id: str, filename: str, *, idempotency_key: str) -> File:
        """Resume an existing upload using the same local file and idempotency key."""
        return await self._uploads.resume(upload_id, filename, idempotency_key=idempotency_key)

    async def upload_text(
        self,
        content: str,
        *,
        name: str = "task.md",
        media_type: str = "text/markdown",
        idempotency_key: str | None = None,
    ) -> File:
        """Upload UTF-8 content inline (up to 32 KiB)."""
        body = upload_text_body(name=name, media_type=media_type, content=content)
        return await self._transport.request(
            "uploadFile", File, body=body, idempotency_key=idempotency_key
        )

    async def prepare_upload(
        self,
        *,
        name: str,
        media_type: str,
        size_bytes: int,
        digest_sha256: str,
        idempotency_key: str | None = None,
    ) -> Upload:
        """Prepare an immutable file upload."""
        body = prepare_upload_body(
            name=name,
            media_type=media_type,
            size_bytes=size_bytes,
            digest_sha256=digest_sha256,
        )
        return await self._transport.request(
            "createUpload", Upload, body=body, idempotency_key=idempotency_key
        )

    async def finalize_upload(
        self, upload_id: str, *, idempotency_key: str | None = None
    ) -> Upload:
        """Scan and finalize an upload."""
        return await self._transport.request(
            "finalizeUpload", Upload, path={"upload_id": upload_id}, idempotency_key=idempotency_key
        )

    async def upload(self, upload_id: str) -> Upload:
        """Read a pending upload record and renewed size/digest-bound PUT grant (write scope)."""
        return await self._transport.request("getUpload", Upload, path={"upload_id": upload_id})

    async def files(self, *, cursor: str | None = None, limit: int = 20) -> FilePage:
        """List the organization's finalized files."""
        return await self._transport.request(
            "listFiles", FilePage, query={"cursor": cursor, "limit": limit}
        )

    async def file(self, file_id: str) -> File:
        """Read a finalized file manifest."""
        return await self._transport.request("getFile", File, path={"file_id": file_id})

    async def update_labels(
        self, run_id: str, labels: dict[str, str], *, expected_revision: int
    ) -> RunLabels:
        """Replace run labels with a revision check."""
        body = RunLabelsRequest(labels=labels, expected_revision=expected_revision)
        return await self._transport.request(
            "updateRunLabels", RunLabels, path={"run_id": run_id}, body=body
        )

    async def get(self, run_id: str) -> RunSnapshot:
        """One run snapshot."""
        return await self._transport.request("getRun", RunSnapshot, path={"run_id": run_id})

    async def wait_update(self, run_id: str, *, timeout_seconds: int = 30) -> RunWait:
        """Read one bounded server wait update; see docs/protocols.md."""
        return await self._transport.request(
            "waitRun", RunWait, path={"run_id": run_id}, query={"timeout_seconds": timeout_seconds}
        )

    async def wait(
        self,
        run_id: str,
        *,
        timeout: float = 600,
        poll_interval: float = 1,
        until: CompletionMilestone = CompletionMilestone.FINISHED,
        raise_on_failure: bool = False,
    ) -> RunSnapshot:
        """Wait for an authoritative snapshot satisfying `until`; timeouts never cancel the run."""
        return await RunObserver(self._transport).wait(
            run_id,
            timeout=timeout,
            poll_interval=poll_interval,
            until=until,
            raise_on_failure=raise_on_failure,
        )

    async def list(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
        state: str | None = None,
        created_after: str | None = None,
    ) -> RunPage:
        """List runs (last recorded state; read a single run for live pause state)."""
        return await self._transport.request(
            "listRuns",
            RunPage,
            query={
                "limit": limit,
                "cursor": cursor,
                "state": state,
                "created_after": created_after,
            },
        )

    async def pause(self, run_id: str, *, idempotency_key: str | None = None) -> RunControlReceipt:
        """Request pause with an empty body; poll get() to observe the resulting state."""
        return await self._transport.request(
            "pauseRun", RunControlReceipt, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def resume(self, run_id: str, *, idempotency_key: str | None = None) -> RunControlReceipt:
        """Request resume with an empty body; poll get() to observe the resulting state."""
        return await self._transport.request(
            "resumeRun", RunControlReceipt, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def program(self, run_id: str) -> RunProgram:
        """Read the admitted program."""
        return await self._transport.request("getRunProgram", RunProgram, path={"run_id": run_id})

    async def limits(self, run_id: str) -> RunLimitsProgress:
        """Read authoritative run limits and usage."""
        return await self._transport.request(
            "getRunLimits", RunLimitsProgress, path={"run_id": run_id}
        )

    async def update_limits(
        self,
        run_id: str,
        *,
        expected_revision: int,
        budget: str | Decimal | None = None,
        wall_time_seconds: int | None = None,
        maximum_tokens: int | None = None,
        pause_retention_seconds: int | None = None,
        resume: bool = False,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunLimitsProgress:
        """Change run limits against an expected revision (at least one limit required)."""
        request = limit_update_request(
            expected_revision=expected_revision,
            budget=budget,
            wall_time_seconds=wall_time_seconds,
            maximum_tokens=maximum_tokens,
            pause_retention_seconds=pause_retention_seconds,
            resume=resume,
            reason=reason,
        )
        return await self._transport.request(
            "updateRunLimits",
            RunLimitsProgress,
            path={"run_id": run_id},
            body=request.model_dump(exclude_none=True),
            idempotency_key=idempotency_key,
        )

    async def update_limits_request(
        self, run_id: str, request: RunLimitUpdate, *, idempotency_key: str | None = None
    ) -> RunLimitsProgress:
        """Apply an exact typed limit request; see docs/protocols.md."""
        return await self._transport.request(
            "updateRunLimits",
            RunLimitsProgress,
            path={"run_id": run_id},
            body=request,
            idempotency_key=idempotency_key,
        )

    async def configuration(self, run_id: str) -> RunConfiguration:
        """Read the frozen run configuration."""
        return await self._transport.request(
            "getRunConfiguration", RunConfiguration, path={"run_id": run_id}
        )

    async def sandboxes(self, run_id: str) -> RunSandboxes:
        """Read actor sandbox lifecycles, resource limits, and teardown state."""
        return await self._transport.request(
            "getRunSandboxes", RunSandboxes, path={"run_id": run_id}
        )

    async def report(self, run_id: str) -> RunReport:
        """Read settled charge/token breakdowns by model, actor, and sandbox."""
        return await self._transport.request("getRunReport", RunReport, path={"run_id": run_id})

    async def allocation(self, run_id: str) -> RunAllocation:
        """Per-actor choices, placements (replacements included), usage and charge."""
        return await self._transport.request(
            "getRunAllocation", RunAllocation, path={"run_id": run_id}
        )

    async def tasks(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> RunTaskPage:
        """Read one cursor page of current run tasks and states."""
        return await self._transport.request(
            "listRunTasks",
            RunTaskPage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def approvals(
        self,
        run_id: str,
        *,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> RunApprovalPage:
        """Read one cursor page of native approval waits and decisions for a run."""
        return await self._transport.request(
            "listRunApprovals",
            RunApprovalPage,
            path={"run_id": run_id},
            query={"status": status, "cursor": cursor, "limit": limit},
        )

    async def decide_approval(
        self,
        run_id: str,
        approval_id: str,
        decision: RunApprovalDecisionRequest,
        *,
        idempotency_key: str | None = None,
    ) -> RunApprovalDecisionReceipt:
        """Record an idempotent decision on a pending run approval."""
        return await self._transport.request(
            "decideRunApproval",
            RunApprovalDecisionReceipt,
            path={"run_id": run_id, "approval_id": approval_id},
            body=decision,
            idempotency_key=idempotency_key,
        )

    async def cancel(self, run_id: str, *, idempotency_key: str | None = None) -> RunSnapshot:
        """Cancel a run with an empty body (`422 control_body_must_be_empty`)."""
        return await self._transport.request(
            "cancelRun", RunSnapshot, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def events(self, run_id: str, *, cursor: str | None = None, limit: int = 20) -> EventPage:
        """Read one JSON page of retained native events; use follow() for SSE."""
        return await self._transport.request(
            "listRunEvents",
            EventPage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def iter_events(
        self, run_id: str, *, cursor: str | None = None, limit: int = 100
    ) -> AsyncGenerator[RunEvent, None]:
        """Yield retained run events across cursor pages; this is not a live subscription."""
        async for item in paginate(
            lambda next_cursor: self.events(run_id, cursor=next_cursor, limit=limit), cursor=cursor
        ):
            yield item

    async def follow(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        checkpoint: StreamCheckpoint | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Stream run events, resuming from an event ID or checkpoint; see docs/protocols.md."""
        async with aclosing(
            self._transport.stream_events(
                run_id, last_event_id=last_event_id, checkpoint=checkpoint, timeout=timeout
            )
        ) as events:
            async for event in events:
                yield event

    async def outputs(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> OutputPage:
        """List published run outputs paginated by cursor."""
        return await self._transport.request(
            "listRunOutputs",
            OutputPage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def output(self, run_id: str, output_id: str) -> RunOutput:
        """One output record."""
        return await self._transport.request(
            "getRunOutput", RunOutput, path={"run_id": run_id, "output_id": output_id}
        )

    async def output_content(
        self, run_id: str, output_id: str, *, revision_id: str, digest_sha256: str
    ) -> bytes:
        """Read the pinned output revision and verify its SHA-256 digest (or DataIntegrityError)."""
        content = cast(
            "bytes",
            await self._transport.invoke(
                "getRunOutputContent",
                path={"run_id": run_id, "output_id": output_id},
                query={"revision_id": revision_id, "digest_sha256": digest_sha256},
            ),
        )
        verify_sha256(content, digest_sha256, label="output")
        return content

    async def output_inline(
        self, run_id: str, output_id: str, *, revision_id: str, digest_sha256: str
    ) -> McpOutputContent:
        """Read a pinned output revision inline (raises OutputContentTooLargeError if oversized)."""
        content, media_type = await self._transport.download(
            "getRunOutputContent",
            path={"run_id": run_id, "output_id": output_id},
            query={"revision_id": revision_id, "digest_sha256": digest_sha256},
            max_bytes=MCP_OUTPUT_CONTENT_MAX_BYTES,
        )
        return inline_content(content, media_type)

    async def download_output(
        self,
        run_id: str,
        output_id: str,
        filename: str,
        *,
        revision_id: str,
        digest_sha256: str,
        max_bytes: int = 64 << 20,
        expected_bytes: int | None = None,
    ) -> Path:
        """Download a pinned output revision to a local path with integrity and bound checks."""
        destination = Path(filename)
        await self._transport.download_output(
            run_id,
            output_id,
            destination,
            revision_id=revision_id,
            digest_sha256=digest_sha256,
            max_bytes=max_bytes,
            expected_bytes=expected_bytes,
        )
        return destination

    async def result(self, run: RunSnapshot | str) -> RunResult:
        """Read Core's answer and charge without waiting (raises ResultUnavailableError if open)."""
        run_id = run if isinstance(run, str) else run.run_id
        try:
            return await self._transport.request("getRunResult", RunResult, path={"run_id": run_id})
        except ConflictError as refusal:
            await reraise_result_unavailable(run_id, refusal, self.get)

    async def review(self, run_id: str) -> RunReviewSurface:
        """Read the consolidated run review surface (pacing, experiments, log, artifacts)."""
        return await self._transport.request(
            "getRunReview", RunReviewSurface, path={"run_id": run_id}
        )

    async def families(self) -> FamilyList:
        """The launchable program families (GA set, swarms.general@1)."""
        return await self._transport.request("listFamilies", FamilyList)

    async def routing_catalog(self) -> RoutingCatalog:
        """Return the routing catalogue exposed by the families endpoint."""
        catalog = await self.families()
        if catalog.routing is None:
            raise ValueError("Server does not expose a routing catalog")
        return catalog.routing

    async def launch_choices(
        self,
        *,
        profile: str | None = None,
        domicile_requirement: DomicileRequirement | None = None,
        parallel_workers: int | None = None,
        wall_time_seconds: int | None = None,
        budget: str | Decimal | None = None,
    ) -> LaunchChoices:
        """Return the per-role menu (and revision) admitted for these launch constraints."""
        query = launch_choices_query(
            profile=profile,
            domicile_requirement=domicile_requirement,
            parallel_workers=parallel_workers,
            wall_time_seconds=wall_time_seconds,
            budget=budget,
        )
        return await self._transport.request("getLaunchChoices", LaunchChoices, query=query)

    async def profiles(self, *, research_task: str | None = None) -> RunProfileCatalog:
        """Published run profiles and caller-specific availability."""
        catalog = await self._transport.request(
            "listProfiles", RunProfileCatalog, query={"research_task": research_task}
        )
        self._default_profile = catalog.default_profile
        self._profile_requirements.update(
            {item.id: item.requires_research_task for item in catalog.profiles}
        )
        return catalog

    async def iter_runs(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
        state: str | None = None,
        created_after: str | None = None,
    ) -> AsyncGenerator[RunSnapshot, None]:
        """Yield run snapshots across cursor pages, preserving the supplied filters."""
        async for item in paginate(
            lambda next_cursor: self.list(
                limit=limit, cursor=next_cursor, state=state, created_after=created_after
            ),
            cursor=cursor,
        ):
            yield item

    async def iter_tasks(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[RunTask, None]:
        """Yield a run's tasks across cursor pages."""
        async for item in paginate(
            lambda next_cursor: self.tasks(run_id, cursor=next_cursor, limit=limit), cursor=cursor
        ):
            yield item

    async def iter_outputs(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[RunOutput, None]:
        """Yield a run's output metadata across cursor pages."""
        async for item in paginate(
            lambda next_cursor: self.outputs(run_id, cursor=next_cursor, limit=limit), cursor=cursor
        ):
            yield item

    async def iter_files(
        self, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[File, None]:
        """Yield uploaded file metadata across cursor pages."""
        async for item in paginate(
            lambda next_cursor: self.files(cursor=next_cursor, limit=limit), cursor=cursor
        ):
            yield item
