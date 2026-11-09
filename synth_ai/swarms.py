"""Run ergonomics over the generated public operation inventory."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import math
from collections.abc import AsyncGenerator
from decimal import Decimal
from pathlib import Path
from typing import cast
from uuid import uuid4

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
    RunTaskPage,
    Upload,
)
from ._limits import UPLOAD_MAX_BYTES
from ._protocol import CompletionMilestone
from ._requests import launch_request
from ._stream import StreamCheckpoint
from ._transport import Transport
from .errors import (
    DataIntegrityError,
    ResultUnavailableError,
    RunFailedError,
    RunHistoryUnavailableError,
    SynthError,
    UploadTransferError,
    WaitTimeoutError,
)


class AsyncSwarms:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def launch(
        self,
        message: str | None = None,
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
        message: str | None = None,
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
        _validate_wait(timeout, poll_interval, until)
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
        source = Path(filename)
        formats = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".json": "application/json",
        }
        media = media_type or formats.get(source.suffix.lower())
        if media not in {"text/plain", "text/markdown", "application/json"}:
            raise ValueError("Supported formats: text/plain, text/markdown, application/json")
        with source.open("rb") as stream:
            content = stream.read(UPLOAD_MAX_BYTES + 1)
        if len(content) > UPLOAD_MAX_BYTES:
            raise ValueError("File exceeds the 32 MiB limit")
        key = idempotency_key or str(uuid4())
        upload = await self.prepare_upload(
            name=source.name,
            media_type=media,
            size_bytes=len(content),
            digest_sha256=hashlib.sha256(content).hexdigest(),
            idempotency_key=key,
        )
        return await self._finish_upload(upload, content, key)

    async def _finish_upload(self, upload: Upload, content: bytes, key: str) -> File:
        if upload.state == "committed":
            return upload.file
        if upload.put is None:
            raise UploadTransferError(upload.upload_id, key)
        try:
            await self._transport.put_upload(upload.put, content, idempotency_key=key)
            finalized = await self.finalize_upload(upload.upload_id, idempotency_key=key)
            if finalized.state != "committed":
                raise UploadTransferError(upload.upload_id, key)
            return finalized.file
        except SynthError as cause:
            raise UploadTransferError(upload.upload_id, key) from cause

    async def resume_upload(self, upload_id: str, filename: str, *, idempotency_key: str) -> File:
        upload = await self.upload(upload_id)
        if upload.state == "committed":
            return upload.file
        with Path(filename).open("rb") as source:
            content = source.read(UPLOAD_MAX_BYTES + 1)
        if (
            len(content) != upload.file.size_bytes
            or hashlib.sha256(content).hexdigest() != upload.file.digest_sha256
        ):
            raise DataIntegrityError("resume content differs from the retained upload")
        return await self._finish_upload(upload, content, idempotency_key)

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
        _validate_wait(timeout, poll_interval, until)
        snapshot: RunSnapshot | None = None
        try:
            async with asyncio.timeout(timeout):
                while True:
                    snapshot = await self.get(run_id)
                    finished = snapshot.finished
                    reached = snapshot.reached(until)
                    if reached:
                        if raise_on_failure and snapshot.state == "failed":
                            raise RunFailedError("Run " + run_id + " failed", snapshot=snapshot)
                        return snapshot
                    if until == "result_available" and finished:
                        raise ResultUnavailableError(
                            "Run has finished without an available result", snapshot=snapshot
                        )
                    if snapshot.metadata.get("imported_history") is not None:
                        raise RunHistoryUnavailableError(
                            "This historical run cannot prove the requested milestone",
                            snapshot=snapshot,
                        )
                    await asyncio.sleep(poll_interval)
        except TimeoutError as error:
            raise WaitTimeoutError(
                "Timed out waiting for " + run_id, snapshot=snapshot, run_id=run_id
            ) from error

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
        while True:
            page = await self.events(run_id, cursor=cursor, limit=limit)
            for item in page.items:
                yield item
            if not page.has_more:
                return
            if page.next_cursor is None or page.next_cursor == "" or page.next_cursor == cursor:
                raise ValueError("nonadvancing event cursor")
            cursor = page.next_cursor

    async def follow(
        self,
        run_id: str,
        *,
        last_event_id: str | None = None,
        checkpoint: StreamCheckpoint | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        async for event in self._transport.stream_events(
            run_id, last_event_id=last_event_id, checkpoint=checkpoint, timeout=timeout
        ):
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
        return await self._transport.request(
            "listProfiles", RunProfileCatalog, query={"research_task": research_task}
        )


def _validate_wait(timeout: float, poll_interval: float, until: str) -> None:
    if (
        not math.isfinite(timeout)
        or not math.isfinite(poll_interval)
        or timeout <= 0
        or poll_interval <= 0
    ):
        raise ValueError("timeout and poll_interval must be finite and positive")
    if until not in {"finished", "execution_closed", "result_available"}:
        raise ValueError("unknown completion milestone")
