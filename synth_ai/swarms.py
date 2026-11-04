"""Run ergonomics over the generated public operation inventory."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from ._generated.models import (
    EventPage,
    FamilyList,
    Limits,
    Money,
    OutputPage,
    Project,
    ProjectPage,
    RunEvent,
    RunOutput,
    RunPage,
    RunSnapshot,
)
from ._transport import Transport
from .errors import RunFailedError, WaitTimeoutError


class AsyncSwarms:
    def __init__(self, transport: Transport, project_id: str | None = None) -> None:
        self._transport = transport
        self.project_id = project_id

    async def launch(
        self,
        message: str | None = None,
        *,
        project_id: str | None = None,
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
        project_id = project_id or self.project_id
        if not project_id:
            raise ValueError("project_id or SYNTH_PROJECT_ID is required")
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
        body: dict[str, Any] = {"project_id": project_id, "limits": limits.model_dump(mode="json")}
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
        project_id: str | None = None,
        state: str | None = None,
        created_after: str | None = None,
    ) -> RunPage:
        return await self._transport.request(
            "listRuns",
            RunPage,
            query={
                "limit": limit,
                "cursor": cursor,
                "project_id": project_id,
                "state": state,
                "created_after": created_after,
            },
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

    async def create_project(
        self,
        name: str,
        *,
        template: str | None = None,
        objective: str | None = None,
        code_validation: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Project:
        body: dict[str, Any] = {"name": name}
        for k, v in {
            "template": template,
            "objective": objective,
            "code_validation": code_validation,
        }.items():
            if v is not None:
                body[k] = v
        return await self._transport.request(
            "createProject", Project, body=body, idempotency_key=idempotency_key
        )

    async def project(self, project_id: str) -> Project:
        return await self._transport.request("getProject", Project, path={"project_id": project_id})

    async def projects(self, *, cursor: str | None = None, limit: int = 20) -> ProjectPage:
        return await self._transport.request(
            "listProjects", ProjectPage, query={"cursor": cursor, "limit": limit}
        )

    async def families(self) -> FamilyList:
        return await self._transport.request("listFamilies", FamilyList)
