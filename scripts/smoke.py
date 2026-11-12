"""Bounded Swarms journey smoke; explicit --execute, $8 aggregate run caps for new receipts.

Requires SYNTH_API_KEY or saved OAuth and SYNTH_API_BASE_URL. This checks a
completed run with verified output and a second run's cancellation/closure;
it does not claim full API or hosted MCP acceptance. Old v3 receipts stay intact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synth_ai import AsyncSynthClient
from synth_ai._generated.models import Problem, RunSnapshot
from synth_ai._generated.operations import OPERATIONS
from synth_ai.errors import SwarmsAPIError, SynthError


class RunStateView(Protocol):
    @property
    def state(self) -> str: ...


class ResponseObservation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    operation: str
    status: int
    observed_at: str
    headers_present: dict[str, bool]
    problem_code: str | None = None


class ObservedRun(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)
    run_id: str
    state: str | None = None
    execution_closed: bool = False
    accounting_final: bool = False
    cleanup_complete: bool = False
    observed_at: str | None = None

    def observe(self, snapshot: RunSnapshot) -> None:
        self.state = snapshot.state
        self.execution_closed = snapshot.execution_closed
        self.accounting_final = snapshot.accounting_final
        self.cleanup_complete = snapshot.cleanup_complete
        self.observed_at = datetime.now(UTC).isoformat()

    @property
    def finished(self) -> bool:
        return self.execution_closed and self.accounting_final and self.cleanup_complete


class SmokeReceipt(BaseModel):
    """Mutable smoke observations, with an explicit scope and retained identities."""

    model_config = ConfigDict(
        strict=True, extra="forbid", populate_by_name=True, validate_assignment=True
    )
    schema_: Literal["synth.public-sdk.swarms-smoke.v4", "synth.public-sdk.swarms-smoke.v5"] = (
        Field(default="synth.public-sdk.swarms-smoke.v5", alias="schema")
    )
    base_url: str
    aggregate_maximum_usd: Literal["0.20", "8.00"] = "8.00"
    runs: list[ObservedRun] = Field(default_factory=list, max_length=2)
    failures: dict[str, str] = Field(default_factory=dict)
    responses: list[ResponseObservation] = Field(default_factory=list)
    launch_keys: dict[str, str] = Field(default_factory=dict)
    cancel_keys: dict[str, str] = Field(default_factory=dict)
    output_verified: bool = False
    cancellation_requested: bool = False

    @model_validator(mode="after")
    def validate_budget(self) -> SmokeReceipt:
        expected = "0.20" if self.schema_.endswith(".v4") else "8.00"
        if self.aggregate_maximum_usd != expected:
            raise ValueError("receipt budget does not match schema")
        return self

    @property
    def run_limits(self) -> tuple[str, int]:
        # Existing v4 receipts retain their original mutation intent on resume.
        return ("0.10", 120) if self.schema_.endswith(".v4") else ("4.00", 600)

    @property
    def smoke_complete(self) -> bool:
        return (
            not self.failures
            and len(self.runs) == 2
            and self.runs[0].state == "completed"
            and all(run.finished for run in self.runs)
            and self.output_verified
            and self.cancellation_requested
        )

    def to_json(self) -> str:
        value = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        value["smoke_complete"] = self.smoke_complete
        return json.dumps(value, indent=2) + "\n"

    @classmethod
    def from_json(cls, source: str) -> SmokeReceipt:
        # The computed summary is never authority when resuming observations.
        value = json.loads(source)
        if isinstance(value, dict):
            value.pop("smoke_complete", None)
            if value.get("schema") == "synth.public-sdk.swarms-smoke.v4":
                value.setdefault("aggregate_maximum_usd", "0.20")
        return cls.model_validate(value)


def require_successful_run(snapshot: RunStateView) -> None:
    if snapshot.state != "completed":
        raise RuntimeError("first run did not complete successfully: " + snapshot.state)


async def run(output: Path, resume: bool = False) -> None:
    receipt = SmokeReceipt(base_url=os.environ["SYNTH_API_BASE_URL"])
    if resume:
        receipt = SmokeReceipt.from_json(output.read_text())
        if receipt.base_url != os.environ["SYNTH_API_BASE_URL"]:
            raise ValueError("cannot resume against a different API")
    elif output.exists():
        raise ValueError("receipt already exists; use --resume to preserve the run budget")
    receipt.failures.clear()
    receipt.output_verified = receipt.cancellation_requested = False
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_text(receipt.to_json())

    def key() -> str:
        return "public-sdk-smoke:" + str(uuid4())

    async def capture(response: httpx.Response) -> None:
        request = response.request
        operation = next(
            (
                name
                for name, spec in OPERATIONS.items()
                if spec["method"] == request.method
                and re.fullmatch(re.sub(r"\{[^}]+\}", "[^/]+", spec["path"]), request.url.path)
            ),
            None,
        )
        if operation is None:
            return
        entry = ResponseObservation(
            operation=operation,
            status=response.status_code,
            observed_at=datetime.now(UTC).isoformat(),
            headers_present={
                name: name in response.headers
                for name in (
                    "x-request-id",
                    "traceparent",
                    "ratelimit-limit",
                    "ratelimit-remaining",
                    "ratelimit-reset",
                    "retry-after",
                    "etag",
                    "strict-transport-security",
                    "idempotent-replayed",
                )
            },
        )
        if response.status_code >= 400 and "application/problem+json" in response.headers.get(
            "content-type", ""
        ):
            await response.aread()
            entry.problem_code = Problem.model_validate_json(response.content).code
        receipt.responses.append(entry)
        save()

    async with (
        httpx.AsyncClient(event_hooks={"response": [capture]}) as http,
        AsyncSynthClient(http_client=http) as client,
    ):

        async def launch(message: str, index: int) -> str:
            if index not in (0, 1):
                raise ValueError("two-run budget exceeded")
            if len(receipt.runs) > index:
                return receipt.runs[index].run_id
            receipt.launch_keys.setdefault(str(index), key())
            save()
            budget, wall_seconds = receipt.run_limits
            result = await client.swarms.launch(
                message,
                budget=budget,
                wall_time_seconds=wall_seconds,
                profile="swarms-v2",
                idempotency_key=receipt.launch_keys[str(index)],
            )
            receipt.runs.append(ObservedRun(run_id=result.run_id))
            save()
            return result.run_id

        async def cancel(run_id: str) -> None:
            receipt.cancel_keys.setdefault(run_id, key())
            save()
            await client.swarms.cancel(run_id, idempotency_key=receipt.cancel_keys[run_id])

        async def closed(run_id: str) -> RunSnapshot:
            deadline = time.monotonic() + receipt.run_limits[1] + 120
            while time.monotonic() < deadline:
                snapshot = await client.swarms.get(run_id)
                for item in receipt.runs:
                    if item.run_id == run_id:
                        item.observe(snapshot)
                save()
                if snapshot.finished:
                    return snapshot
                await asyncio.sleep(2)
            raise RuntimeError("run closure deadline exceeded: " + run_id)

        first = await launch(
            "Create a file hello.txt containing exactly hello and publish it as the task output.", 0
        )
        try:
            require_successful_run(await closed(first))
            async with asyncio.timeout(15):
                async for _ in client.swarms.follow(first):
                    pass
            outputs = await client.swarms.outputs(first)
            if not outputs.items:
                raise RuntimeError("first run produced no output; content gate is unverified")
            selected = await client.swarms.output(first, outputs.items[0].output_id)
            if selected.reference is None:
                raise RuntimeError("output reference missing")
            await client.swarms.output_content(
                first,
                selected.output_id,
                revision_id=selected.reference["revision_id"],
                digest_sha256=selected.reference["digest_sha256"],
            )
            receipt.output_verified = True
            save()
        except (RuntimeError, SynthError, TimeoutError) as error:
            receipt.failures["first_run"] = type(error).__name__
            save()
        finally:
            if not (await client.swarms.get(first)).execution_closed:
                await cancel(first)
                try:
                    await closed(first)
                except RuntimeError:
                    receipt.failures["first_run_closure"] = "deadline_exceeded"
                    save()
        second = await launch("Write a short explanation of addition.", 1)
        try:
            await cancel(second)
            receipt.cancellation_requested = True
            save()
        finally:
            try:
                await closed(second)
            except RuntimeError:
                receipt.failures["second_run_closure"] = "deadline_exceeded"
                save()
        save()
        if not receipt.smoke_complete:
            raise RuntimeError(
                "smoke incomplete; resume the retained receipt after resolving failures"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the receipt's original run limits, identities and mutation keys",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args.output, args.resume))
    except SwarmsAPIError as error:
        raise SystemExit(
            f"smoke refused: status={error.status} code={error.code} outcome={error.outcome}"
        ) from None
