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
    LaunchRequest,
    OutputPage,
    OwnerResourceRequest,
    RoutingCatalog,
    RoutingConstraints,
    RunConfiguration,
    RunControlReceipt,
    RunEvent,
    RunGuidance,
    RunInputs,
    RunLabels,
    RunLabelsRequest,
    RunOutput,
    RunPage,
    RunProfileCatalog,
    RunProgram,
    RunReport,
    RunSandboxes,
    RunSnapshot,
    RunTask,
    RunTaskPage,
    Upload,
)
from ._generated.policy import DEFAULT_RUN_PROFILE, PROFILE_REQUIRES_RESEARCH_TASK
from ._observer import RunObserver, validate_wait
from ._pagination import paginate
from ._protocol import CompletionMilestone
from ._requests import launch_request
from ._stream import StreamCheckpoint
from ._transport import Transport
from ._uploads import ArtifactUploads
from .errors import (
    DataIntegrityError,
)


class AsyncSwarms:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._profile_requirements = dict(PROFILE_REQUIRES_RESEARCH_TASK)
        self._default_profile = DEFAULT_RUN_PROFILE

    async def launch(
        self,
        message: str,
        *,
        budget: str | Decimal,
        wall_time_seconds: int | None = None,
        inputs: RunInputs | None = None,
        labels: dict[str, str] | None = None,
        resources: builtins.list[OwnerResourceRequest] | None = None,
        profile: str | None = None,
        research_task: str | None = None,
        routing: RoutingConstraints | None = None,
        guidance: builtins.list[RunGuidance] | None = None,
        idempotency_key: str | None = None,
    ) -> RunSnapshot:
        request = launch_request(
            message,
            budget=budget,
            wall_time_seconds=wall_time_seconds,
            inputs=inputs,
            labels=labels,
            resources=resources,
            profile=profile,
            research_task=research_task,
            routing=routing,
            guidance=guidance,
            profile_requirements=self._profile_requirements,
            default_profile=self._default_profile,
        )
        return await self.launch_request(request, idempotency_key=idempotency_key)

    async def launch_request(
        self, request: LaunchRequest, *, idempotency_key: str | None = None
    ) -> RunSnapshot:
        return await self._transport.request(
            "createRun", RunSnapshot, body=request, idempotency_key=idempotency_key
        )

    async def run(
        self,
        message: str,
        *,
        budget: str | Decimal,
        profile: str | None = None,
        research_task: str | None = None,
        timeout: float = 600,
        poll_interval: float = 1,
        until: CompletionMilestone = CompletionMilestone.FINISHED,
        raise_on_failure: bool = False,
        wall_time_seconds: int | None = None,
        inputs: RunInputs | None = None,
        labels: dict[str, str] | None = None,
        resources: builtins.list[OwnerResourceRequest] | None = None,
        routing: RoutingConstraints | None = None,
        guidance: builtins.list[RunGuidance] | None = None,
        idempotency_key: str | None = None,
    ) -> RunSnapshot:
        validate_wait(timeout, poll_interval, until)
        created = await self.launch(
            message,
            budget=budget,
            profile=profile,
            research_task=research_task,
            wall_time_seconds=wall_time_seconds,
            inputs=inputs,
            labels=labels,
            resources=resources,
            routing=routing,
            guidance=guidance,
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
        return await ArtifactUploads(self._transport).upload_file(
            filename, media_type=media_type, idempotency_key=idempotency_key
        )

    async def resume_upload(self, upload_id: str, filename: str, *, idempotency_key: str) -> File:
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
        return await self._transport.request(
            "uploadFile",
            File,
            body={"name": name, "media_type": media_type, "content": content},
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
        return await self._transport.request(
            "createUpload",
            Upload,
            body={
                "name": name,
                "media_type": media_type,
                "size_bytes": size_bytes,
                "digest_sha256": digest_sha256,
            },
            idempotency_key=idempotency_key,
        )

    async def finalize_upload(
        self, upload_id: str, *, idempotency_key: str | None = None
    ) -> Upload:
        return await self._transport.request(
            "finalizeUpload", Upload, path={"upload_id": upload_id}, idempotency_key=idempotency_key
        )

    async def upload(self, upload_id: str) -> Upload:
        return await self._transport.request("getUpload", Upload, path={"upload_id": upload_id})

    async def files(self, *, cursor: str | None = None, limit: int = 20) -> FilePage:
        return await self._transport.request(
            "listFiles",
            FilePage,
            query={"cursor": cursor, "limit": limit},
        )

    async def file(self, file_id: str) -> File:
        return await self._transport.request("getFile", File, path={"file_id": file_id})

    async def update_labels(
        self, run_id: str, labels: dict[str, str], *, expected_revision: int
    ) -> RunLabels:
        return await self._transport.request(
            "updateRunLabels",
            RunLabels,
            path={"run_id": run_id},
            body=RunLabelsRequest(labels=labels, expected_revision=expected_revision),
        )

    async def get(self, run_id: str) -> RunSnapshot:
        return await self._transport.request("getRun", RunSnapshot, path={"run_id": run_id})

    async def wait(
        self,
        run_id: str,
        *,
        timeout: float = 600,
        poll_interval: float = 1,
        until: CompletionMilestone = CompletionMilestone.FINISHED,
        raise_on_failure: bool = False,
    ) -> RunSnapshot:
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
        return await self._transport.request(
            "pauseRun", RunControlReceipt, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def resume(self, run_id: str, *, idempotency_key: str | None = None) -> RunControlReceipt:
        return await self._transport.request(
            "resumeRun", RunControlReceipt, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def program(self, run_id: str) -> RunProgram:
        return await self._transport.request("getRunProgram", RunProgram, path={"run_id": run_id})

    async def configuration(self, run_id: str) -> RunConfiguration:
        return await self._transport.request(
            "getRunConfiguration", RunConfiguration, path={"run_id": run_id}
        )

    async def sandboxes(self, run_id: str) -> RunSandboxes:
        return await self._transport.request(
            "getRunSandboxes", RunSandboxes, path={"run_id": run_id}
        )

    async def report(self, run_id: str) -> RunReport:
        return await self._transport.request("getRunReport", RunReport, path={"run_id": run_id})

    async def tasks(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> RunTaskPage:
        return await self._transport.request(
            "listRunTasks",
            RunTaskPage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def cancel(self, run_id: str, *, idempotency_key: str | None = None) -> RunSnapshot:
        return await self._transport.request(
            "cancelRun", RunSnapshot, path={"run_id": run_id}, idempotency_key=idempotency_key
        )

    async def events(self, run_id: str, *, cursor: str | None = None, limit: int = 20) -> EventPage:
        return await self._transport.request(
            "listRunEvents",
            EventPage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def iter_events(
        self, run_id: str, *, cursor: str | None = None, limit: int = 100
    ) -> AsyncGenerator[RunEvent, None]:
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
        return await self._transport.request(
            "listRunOutputs",
            OutputPage,
            path={"run_id": run_id},
            query={"cursor": cursor, "limit": limit},
        )

    async def output(self, run_id: str, output_id: str) -> RunOutput:
        return await self._transport.request(
            "getRunOutput", RunOutput, path={"run_id": run_id, "output_id": output_id}
        )

    async def output_content(
        self, run_id: str, output_id: str, *, revision_id: str, digest_sha256: str
    ) -> bytes:
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

    async def families(self) -> FamilyList:
        return await self._transport.request("listFamilies", FamilyList)

    async def routing_catalog(self) -> RoutingCatalog:
        catalog = await self.families()
        if catalog.routing is None:
            raise ValueError("Server does not expose a routing catalog")
        return catalog.routing

    async def profiles(self, *, research_task: str | None = None) -> RunProfileCatalog:
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
        async def fetch(next_cursor: str | None) -> RunPage:
            return await self.list(
                limit=limit, cursor=next_cursor, state=state, created_after=created_after
            )

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def iter_tasks(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[RunTask, None]:
        async def fetch(next_cursor: str | None) -> RunTaskPage:
            return await self.tasks(run_id, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def iter_outputs(
        self, run_id: str, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[RunOutput, None]:
        async def fetch(next_cursor: str | None) -> OutputPage:
            return await self.outputs(run_id, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item

    async def iter_files(
        self, *, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[File, None]:
        async def fetch(next_cursor: str | None) -> FilePage:
            return await self.files(cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item
