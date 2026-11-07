"""Run ergonomics over the generated public operation inventory."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from ._generated.models import (
    EventPage,
    FamilyList,
    File,
    FilePage,
    Limits,
    Money,
    OutputPage,
    RunConfiguration,
    RunControlReceipt,
    RunEvent,
    RunLabels,
    RunOutput,
    RunPage,
    RunProgram,
    RunReport,
    RunSandboxes,
    RunSnapshot,
    RunTaskPage,
    Upload,
)
from ._transport import Transport
from .errors import RunFailedError, WaitTimeoutError


class AsyncSwarms:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def launch(
        self,
        message: str | None = None,
        *,
        budget: str = "1.00",
        wall_time_seconds: int = 3600,
        maximum_tasks: int = 100,
        maximum_parallel_tasks: int = 4,
        inputs: dict[str, Any] | None = None,
        labels: dict[str, str] | None = None,
        resources: list[dict[str, Any]] | None = None,
        program: dict[str, str] | None = None,
        providers: dict[str, str] | None = None,
        models: dict[str, str] | None = None,
        method: Any = None,
        profile_overrides: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> RunSnapshot:
        try:
            amount = Decimal(budget)
            if not amount.is_finite() or amount <= 0:
                raise ValueError("budget must be positive")
        except InvalidOperation as error:
            raise ValueError("budget must be a decimal string") from error
        limits = Limits(
            maximum_charge=Money(amount=format(amount, "f"), currency="USD"),
            wall_time_seconds=wall_time_seconds,
            maximum_tasks=maximum_tasks,
            maximum_parallel_tasks=maximum_parallel_tasks,
        )
        body: dict[str, Any] = {"limits": limits.model_dump(mode="json")}
        for name, value in {
            "message": message,
            "inputs": inputs,
            "labels": labels,
            "resources": resources,
            "program": program,
            "providers": providers,
            "models": models,
            "method": method,
            "profile_overrides": profile_overrides,
        }.items():
            if value is not None:
                body[name] = value
        return await self._transport.request(
            "createRun", RunSnapshot, body=body, idempotency_key=idempotency_key
        )

    async def run(
        self,
        message: str | None = None,
        *,
        timeout: float = 600,
        poll_interval: float = 1,
        **launch: Any,
    ) -> RunSnapshot:
        created = await self.launch(message, **launch)
        return await self.wait(created.run_id, timeout=timeout, poll_interval=poll_interval)

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
            content = stream.read((32 << 20) + 1)
        if len(content) > 32 << 20:
            raise ValueError("File exceeds the 32 MiB limit")
        key = idempotency_key or str(uuid4())
        upload = await self.prepare_upload(
            name=source.name,
            media_type=media,
            size_bytes=len(content),
            digest_sha256=hashlib.sha256(content).hexdigest(),
            idempotency_key=key,
        )
        if upload.put is None:
            raise ValueError("Upload response has no PUT grant")
        await self._transport.put_upload(upload.put, content, idempotency_key=key)
        finalized = await self.finalize_upload(upload.upload_id, idempotency_key=key)
        return finalized.file

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
            body={"labels": labels, "expected_revision": expected_revision},
        )

    async def get(self, run_id: str) -> RunSnapshot:
        return await self._transport.request("getRun", RunSnapshot, path={"run_id": run_id})

    async def wait(
        self, run_id: str, *, timeout: float = 600, poll_interval: float = 1
    ) -> RunSnapshot:
        if timeout <= 0 or poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")
        end = time.monotonic() + timeout
        while True:
            run = await self.get(run_id)
            if run.execution_closed:
                if run.state == "failed":
                    raise RunFailedError("Run " + run.run_id + " failed")
                return run
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise WaitTimeoutError("Timed out waiting for " + run_id)
            await asyncio.sleep(min(poll_interval, remaining))

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
    ) -> AsyncIterator[RunEvent]:
        while True:
            page = await self.events(run_id, cursor=cursor, limit=limit)
            for item in page.items:
                yield item
            if not page.has_more:
                return
            if not page.next_cursor or page.next_cursor == cursor:
                raise ValueError("nonadvancing event cursor")
            cursor = page.next_cursor

    async def follow(
        self, run_id: str, *, last_event_id: str | None = None
    ) -> AsyncIterator[RunEvent]:
        async for event in self._transport.stream_events(run_id, last_event_id=last_event_id):
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
        return cast(
            bytes,
            await self._transport.invoke(
                "getRunOutputContent",
                path={"run_id": run_id, "output_id": output_id},
                query={"revision_id": revision_id, "digest_sha256": digest_sha256},
            ),
        )

    async def families(self) -> FamilyList:
        return await self._transport.request("listFamilies", FamilyList)
