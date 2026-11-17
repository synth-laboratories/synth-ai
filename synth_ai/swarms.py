"""Run ergonomics over the generated public operation inventory."""

from __future__ import annotations

import builtins
import hashlib
from collections.abc import AsyncGenerator
from contextlib import aclosing
from decimal import Decimal
from pathlib import Path
from typing import cast

from ._generated.models import (
    EventPage,
    FamilyList,
    File,
    FilePage,
    LaunchChoices,
    LaunchRequest,
    McpOutputContent,
    Money,
    OptimizerRequest,
    OutputPage,
    OwnerResourceRequest,
    RoleAssignment,
    RoutingCatalog,
    RoutingConstraints,
    RunAllocation,
    RunConfiguration,
    RunControlReceipt,
    RunEvent,
    RunGuidance,
    RunInputs,
    RunLabels,
    RunLabelsRequest,
    RunLimitChanges,
    RunLimitsProgress,
    RunLimitUpdate,
    RunOutput,
    RunPage,
    RunProfileCatalog,
    RunProgram,
    RunReport,
    RunResult,
    RunSandboxes,
    RunSnapshot,
    RunTask,
    RunTaskPage,
    RunWait,
    Upload,
)
from ._generated.policy import (
    DEFAULT_RUN_PROFILE,
    MCP_OUTPUT_CONTENT_MAX_BYTES,
    PROFILE_REQUIRES_RESEARCH_TASK,
)
from ._inline_content import inline_content
from ._observer import RunObserver, validate_wait
from ._pagination import paginate
from ._protocol import CompletionMilestone
from ._requests import DomicileRequirement, launch_request
from ._stream import StreamCheckpoint
from ._transport import Transport
from ._uploads import ArtifactUploads
from .errors import (
    ConflictError,
    DataIntegrityError,
    ResultUnavailableError,
    check_upload_format,
)


class AsyncSwarms:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._profile_requirements = dict(PROFILE_REQUIRES_RESEARCH_TASK)
        self._default_profile = DEFAULT_RUN_PROFILE

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
        idempotency_key: str | None = None,
    ) -> RunSnapshot:
        """Launch a run with a maximum charge in USD and return its acknowledgement.

        The budget must be a decimal string or Decimal. Retain an explicit
        idempotency_key for uncertain retries. An acknowledgement does not prove
        execution, accounting or cleanup has finished; use wait and report."""
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
        )
        return await self.launch_request(request, idempotency_key=idempotency_key)

    async def launch_request(
        self, request: LaunchRequest, *, idempotency_key: str | None = None
    ) -> RunSnapshot:
        """Submit a typed launch request and return its initial run snapshot.

        Reuse the same idempotency key and request after an uncertain response."""
        return await self._transport.request(
            "createRun", RunSnapshot, body=request, idempotency_key=idempotency_key
        )

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
        idempotency_key: str | None = None,
    ) -> RunSnapshot:
        """Launch a run, then wait for the selected completion milestone.

        Budget is a maximum charge in USD; timeout and poll_interval are seconds.
        A local wait timeout does not cancel the server run. Set raise_on_failure
        to raise on a failed finished run."""
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
        """Upload a local file and return its verified file record.

        Use an explicit idempotency key to retain identity across uncertain retries."""
        return await ArtifactUploads(self._transport).upload_file(
            filename, media_type=media_type, idempotency_key=idempotency_key
        )

    async def upload_directory(
        self, directory: str | int, *, idempotency_key: str
    ) -> builtins.list[File]:
        """Upload a bounded POSIX directory snapshot, preserving relative file names.

        An integer is a borrowed directory descriptor; the SDK duplicates it.
        Includes dotfiles; refuses links and special files before any API request.
        Reuse the key with unchanged files to recover a partial upload. Failure
        retains committed file records in DirectoryUploadError.completed_files.
        """
        return await ArtifactUploads(self._transport).upload_directory(
            directory, idempotency_key=idempotency_key
        )

    async def resume_upload(self, upload_id: str, filename: str, *, idempotency_key: str) -> File:
        """Resume an existing upload using the same local file and idempotency key."""
        return await ArtifactUploads(self._transport).resume(
            upload_id, filename, idempotency_key=idempotency_key
        )

    async def upload_text(
        self,
        content: str,
        *,
        name: str = "task.md",
        media_type: str = "text/markdown",
        idempotency_key: str | None = None,
    ) -> File:
        """Upload UTF-8 content inline (up to 32 KiB)."""
        return await self._transport.request(
            "uploadFile",
            File,
            body={"name": name, "media_type": check_upload_format(media_type), "content": content},
            idempotency_key=idempotency_key,
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
        return await self._transport.request(
            "createUpload",
            Upload,
            body={
                "name": name,
                "media_type": check_upload_format(media_type),
                "size_bytes": size_bytes,
                "digest_sha256": digest_sha256,
            },
            idempotency_key=idempotency_key,
        )

    async def finalize_upload(
        self, upload_id: str, *, idempotency_key: str | None = None
    ) -> Upload:
        """Scan and finalize an upload."""
        return await self._transport.request(
            "finalizeUpload", Upload, path={"upload_id": upload_id}, idempotency_key=idempotency_key
        )

    async def upload(self, upload_id: str) -> Upload:
        """Returns the retained upload. A caller with swarms:write also receives a renewed size/digest-bound PUT grant while it is pending. Read-only callers receive no write grant."""
        return await self._transport.request("getUpload", Upload, path={"upload_id": upload_id})

    async def files(self, *, cursor: str | None = None, limit: int = 20) -> FilePage:
        """List the organization's finalized files."""
        return await self._transport.request(
            "listFiles",
            FilePage,
            query={"cursor": cursor, "limit": limit},
        )

    async def file(self, file_id: str) -> File:
        """Read a finalized file manifest."""
        return await self._transport.request("getFile", File, path={"file_id": file_id})

    async def update_labels(
        self, run_id: str, labels: dict[str, str], *, expected_revision: int
    ) -> RunLabels:
        """Replace run labels with a revision check."""
        return await self._transport.request(
            "updateRunLabels",
            RunLabels,
            path={"run_id": run_id},
            body=RunLabelsRequest(labels=labels, expected_revision=expected_revision),
        )

    async def get(self, run_id: str) -> RunSnapshot:
        """One run snapshot."""
        return await self._transport.request("getRun", RunSnapshot, path={"run_id": run_id})

    async def wait_update(self, run_id: str, *, timeout_seconds: int = 30) -> RunWait:
        """Read one bounded server wait update; see docs/protocols.md."""
        return await self._transport.request(
            "waitRun",
            RunWait,
            path={"run_id": run_id},
            query={"timeout_seconds": timeout_seconds},
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
        """Wait for an authoritative snapshot satisfying the chosen milestone.

        Timeout and poll_interval are seconds. FINISHED requires execution closed,
        accounting final and cleanup complete; it does not imply success. Paused
        runs and local timeouts raise typed errors without cancelling the run.
        Set raise_on_failure to raise on a failed finished run."""
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
        """Each run's state is its last recorded state. Read a single run for live pause state. A project_id query parameter is refused with 422 project_id_removed."""
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
        """The body must be empty. Acceptance acknowledges the durable command; read the run to observe its current state. When the program refuses the control, the 409 Problem's code is limit_still_reached (the limit that paused the run is still reached; raise it first) or run_expired, otherwise program_control_refused; upstream_status and cause carry the program's status and refusal code."""
        return await self._transport.request(
            "pauseRun", RunControlReceipt, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def resume(self, run_id: str, *, idempotency_key: str | None = None) -> RunControlReceipt:
        """The body must be empty. Acceptance acknowledges the durable command; read the run to observe its current state. When the program refuses the control, the 409 Problem's code is limit_still_reached (the limit that paused the run is still reached; raise it first) or run_expired, otherwise program_control_refused; upstream_status and cause carry the program's status and refusal code."""
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
        """Change run limits against an expected revision.

        Budget is a positive USD decimal string or Decimal; time limits are
        seconds and maximum_tokens counts tokens. At least one limit is required.
        The optional resume flag requests resume with the limit change."""
        if budget is not None and (
            not isinstance(budget, (str, Decimal))
            or not Decimal(budget).is_finite()
            or Decimal(budget) <= 0
        ):
            raise ValueError("budget must be a positive decimal string or Decimal")
        changes = RunLimitChanges(
            maximum_charge=Money(amount=format(Decimal(budget), "f"), currency="USD")
            if budget is not None
            else None,
            maximum_tokens=maximum_tokens,
            wall_time_seconds=wall_time_seconds,
            pause_retention_seconds=pause_retention_seconds,
        )
        if not changes.model_dump(exclude_none=True):
            raise ValueError("at least one limit is required")
        request = RunLimitUpdate(
            expected_revision=expected_revision, limits=changes, resume=resume, reason=reason
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
        """Read actor role/provider, lifecycle, image, resource limits and sandbox times.

        Teardown is complete only after execution closes and all sandboxes are verified
        gone. Unobserved means a record could not be established, never absent.
        See docs/protocols.md."""
        return await self._transport.request(
            "getRunSandboxes", RunSandboxes, path={"run_id": run_id}
        )

    async def report(self, run_id: str) -> RunReport:
        """Read settled charge/token breakdowns by provider/model, actor and sandbox.

        Includes rollouts and timings. Parts sum to the settled total; pending means
        empty breakdowns. Missing retained attribution is not_recorded. Metered
        rollout compute is settlement.rollout within maximum_charge; not_metered
        means uncharged, not zero cost. See docs/protocols.md."""
        return await self._transport.request("getRunReport", RunReport, path={"run_id": run_id})

    async def allocation(self, run_id: str) -> RunAllocation:
        """Per-actor choices, placements (replacements included), usage and charge."""
        return await self._transport.request(
            "getRunAllocation", RunAllocation, path={"run_id": run_id}
        )

    async def tasks(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> RunTaskPage:
        """Each page reads current tasks and states. A cursor stays valid while the run advances, but pages are not a snapshot, so a task added during paging can be missing until a fresh listing."""
        return await self._transport.request(
            "listRunTasks",
            RunTaskPage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def cancel(self, run_id: str, *, idempotency_key: str | None = None) -> RunSnapshot:
        """The body must be empty (`422 control_body_must_be_empty`)."""
        return await self._transport.request(
            "cancelRun", RunSnapshot, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def events(self, run_id: str, *, cursor: str | None = None, limit: int = 20) -> EventPage:
        """Read one JSON page of retained native events.

        next_cursor paginates; resume_cursor permits later reads. Use follow() for
        SSE; see docs/protocols.md for stream recovery and milestone semantics."""
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

        async def fetch(next_cursor: str | None) -> EventPage:
            return await self.events(run_id, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def follow(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        checkpoint: StreamCheckpoint | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Stream run events, resuming from an event ID or mutable checkpoint.

        Timeout is seconds. Events do not establish run finality; use wait or get
        for authoritative lifecycle state."""
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
        """Published outputs for a native run, paginated by cursor. A run whose `selection` has a stored incumbent snapshot also lists it as `selected-snapshot.tar.gz`, one gzipped tar holding the original archive bytes, with `reference.digest_sha256` equal to `selection.incumbent.artifact_digest`."""
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
        """Read the pinned output revision and verify its SHA-256 digest.

        Raises DataIntegrityError if the received bytes differ from the reference."""
        content = cast(
            bytes,
            await self._transport.invoke(
                "getRunOutputContent",
                path={"run_id": run_id, "output_id": output_id},
                query={"revision_id": revision_id, "digest_sha256": digest_sha256},
            ),
        )
        if hashlib.sha256(content).hexdigest() != digest_sha256:
            raise DataIntegrityError("output digest does not match the pinned reference")
        return content

    async def output_inline(
        self, run_id: str, output_id: str, *, revision_id: str, digest_sha256: str
    ) -> McpOutputContent:
        """Read a pinned output revision as the contract's McpOutputContent.

        Text when the media type is textual and the bytes are valid UTF-8, otherwise
        base64. Over MCP_OUTPUT_CONTENT_MAX_BYTES raises OutputContentTooLargeError
        (output_content_too_large_for_mcp); use download_output for larger outputs."""
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
        """Download a pinned output revision to a local path with integrity checks.

        The default size limit is 64 MiB. max_bytes and expected_bytes are bytes.
        Returns the destination path after the transfer succeeds."""
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
        """Read Core's answer, source output, objective answer and charge without waiting.

        Core chooses the latest available text output, then the latest objective-answer
        JSON. Absent answer fields remain None. An unfinished run raises
        ResultUnavailableError with a fresh snapshot. See docs/protocols.md."""
        run_id = run if isinstance(run, str) else run.run_id
        try:
            return await self._transport.request("getRunResult", RunResult, path={"run_id": run_id})
        except ConflictError as refusal:
            if refusal.code != "result_unavailable":
                raise
            snapshot = await self.get(run_id)
            raise ResultUnavailableError(
                "Run " + run_id + " has not finished; wait() for it first", snapshot=snapshot
            ) from refusal

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
        """The per-role menu a launch with these fields admits, and its revision.

        Pass the returned menu_revision as launch_menu_revision with role_assignments.
        With a budget, choices it cannot afford are listed as exclusions.
        """
        return await self._transport.request(
            "getLaunchChoices",
            LaunchChoices,
            query={
                "profile": profile,
                "domicile_requirement": domicile_requirement,
                "parallel_workers": parallel_workers,
                "wall_time_seconds": wall_time_seconds,
                "maximum_charge": format(Decimal(budget), "f") if budget is not None else None,
            },
        )

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

        async def fetch(next_cursor: str | None) -> RunPage:
            return await self.list(
                limit=limit, cursor=next_cursor, state=state, created_after=created_after
            )

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def iter_tasks(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[RunTask, None]:
        """Yield a run's tasks across cursor pages."""

        async def fetch(next_cursor: str | None) -> RunTaskPage:
            return await self.tasks(run_id, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def iter_outputs(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[RunOutput, None]:
        """Yield a run's output metadata across cursor pages."""

        async def fetch(next_cursor: str | None) -> OutputPage:
            return await self.outputs(run_id, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def iter_files(
        self, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[File, None]:
        """Yield uploaded file metadata across cursor pages."""

        async def fetch(next_cursor: str | None) -> FilePage:
            return await self.files(cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item
